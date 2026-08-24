#!/usr/bin/env python3
"""Upload strict scoreboard-rwkv campaign and task DTOs.

This module owns transport only: strict input validation, authenticated HTTP,
idempotent resume, and finalize.  Evaluator-specific data conversion lives in
``scripts/convert_scoreboard_payloads.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


try:
    from scripts.convert_scoreboard_payloads import (
        CAMPAIGN_SCHEMA,
        ScoreboardError,
        _artifact_root,
        _json_safe,
        _load_json_object,
        _reject_duplicate_keys,
        _reject_json_constant,
        _write_json_atomic,
        build_lm_eval_publication,
        canonical_json,
        content_digest,
    )
except ModuleNotFoundError as error:
    if error.name not in {"scripts", "scripts.convert_scoreboard_payloads"}:
        raise
    from convert_scoreboard_payloads import (  # type: ignore[no-redef]
        CAMPAIGN_SCHEMA,
        ScoreboardError,
        _artifact_root,
        _json_safe,
        _load_json_object,
        _reject_duplicate_keys,
        _reject_json_constant,
        _write_json_atomic,
        build_lm_eval_publication,
        canonical_json,
        content_digest,
    )


SCOREBOARD_SCHEMA = "scoreboard-v1"
CAMPAIGN_SCHEMA = SCOREBOARD_SCHEMA
TASK_SCHEMA = SCOREBOARD_SCHEMA
LM_EVAL_CAMPAIGN_SCHEMA = SCOREBOARD_SCHEMA
LM_EVAL_TASK_SCHEMA = SCOREBOARD_SCHEMA
LIGHTEVAL_VERSION = "0.13.0"
SUPPORTED_CAMPAIGN_SCHEMAS = {SCOREBOARD_SCHEMA}
SUPPORTED_TASK_SCHEMAS = {SCOREBOARD_SCHEMA}
SUPPORTED_SOURCES = {"lighteval", "evalscope", "lm-eval-harness"}
SCOREBOARD_BASE_URL_ENV = "SCOREBOARD_BASE_URL"
SCOREBOARD_TOKEN_ENV_ENV = "SCOREBOARD_PUBLICATION_TOKEN_ENV"  # noqa: S105
SCOREBOARD_TOKEN_ENV = "SCOREBOARD_PUBLICATION_TOKEN"  # noqa: S105
SCOREBOARD_TIMEOUT_ENV = "SCOREBOARD_UPLOAD_TIMEOUT"
SCOREBOARD_FINALIZE_ENV = "SCOREBOARD_UPLOAD_FINALIZE"
SCOREBOARD_MODEL_SHA256_ENV = "SCOREBOARD_MODEL_SHA256"
SCOREBOARD_MODEL_REVISION_ENV = "SCOREBOARD_MODEL_REVISION"
SCOREBOARD_RETRIES_ENV = "SCOREBOARD_UPLOAD_RETRIES"
SCOREBOARD_RETRY_DELAY_ENV = "SCOREBOARD_UPLOAD_RETRY_DELAY"
CONTROL_REQUEST_TIMEOUT = 30.0
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _environment_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _environment_timeout(default: float = 3600.0) -> float:
    raw = _environment_text(SCOREBOARD_TIMEOUT_ENV)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ScoreboardError(
            f"{SCOREBOARD_TIMEOUT_ENV} must be a positive number"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ScoreboardError(f"{SCOREBOARD_TIMEOUT_ENV} must be a positive number")
    return value


def _environment_finalize(default: bool = True) -> bool:
    raw = _environment_text(SCOREBOARD_FINALIZE_ENV)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ScoreboardError(
        f"{SCOREBOARD_FINALIZE_ENV} must be one of true/false, yes/no, on/off, or 1/0"
    )


def _environment_retries(default: int = 2) -> int:
    raw = _environment_text(SCOREBOARD_RETRIES_ENV)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ScoreboardError(
            f"{SCOREBOARD_RETRIES_ENV} must be a non-negative integer"
        ) from error
    if value < 0:
        raise ScoreboardError(
            f"{SCOREBOARD_RETRIES_ENV} must be a non-negative integer"
        )
    return value


def _environment_retry_delay(default: float = 1.0) -> float:
    raw = _environment_text(SCOREBOARD_RETRY_DELAY_ENV)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ScoreboardError(
            f"{SCOREBOARD_RETRY_DELAY_ENV} must be a positive number"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ScoreboardError(f"{SCOREBOARD_RETRY_DELAY_ENV} must be a positive number")
    return value


def resolve_publication_settings(
    publication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve transport settings without reading the token into artifacts."""

    resolved = dict(publication or {})
    if not resolved.get("base_url"):
        base_url = _environment_text(SCOREBOARD_BASE_URL_ENV)
        if base_url is not None:
            resolved["base_url"] = base_url
    if not resolved.get("token_env"):
        resolved["token_env"] = (
            _environment_text(SCOREBOARD_TOKEN_ENV_ENV) or SCOREBOARD_TOKEN_ENV
        )
    timeout = resolved.get("timeout")
    if timeout is None:
        resolved["timeout"] = _environment_timeout()
    elif (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ScoreboardError("publication.timeout must be a positive number")
    else:
        resolved["timeout"] = float(timeout)
    finalize = resolved.get("finalize")
    if finalize is None:
        resolved["finalize"] = _environment_finalize()
    elif not isinstance(finalize, bool):
        raise ScoreboardError("publication.finalize must be a boolean")
    if not resolved.get("model_sha256"):
        model_sha256 = _environment_text(SCOREBOARD_MODEL_SHA256_ENV)
        if model_sha256 is not None:
            resolved["model_sha256"] = model_sha256
    if not resolved.get("model_revision"):
        model_revision = _environment_text(SCOREBOARD_MODEL_REVISION_ENV)
        if model_revision is not None:
            resolved["model_revision"] = model_revision
    return resolved


def publish_lm_eval_evaluation(
    results: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    *,
    output_dir: str | Path,
    publication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist converted DTOs, then optionally upload them."""

    publication = resolve_publication_settings(publication)
    result_config = results.get("config")
    result_config = result_config if isinstance(result_config, dict) else {}
    result_model_args = result_config.get("model_args")
    model_name = results.get("model_name")
    if (not isinstance(model_name, str) or not model_name) and isinstance(
        result_model_args, dict
    ):
        model_name = result_model_args.get("model") or result_model_args.get(
            "pretrained"
        )
    if not isinstance(model_name, str) or not model_name:
        model_name = result_config.get("model")
    if not isinstance(model_name, str) or not model_name:
        model_name = "model"
    root = _artifact_root(Path(output_dir), model_name)
    root.mkdir(parents=True, exist_ok=True)
    raw_results = _json_safe(results)
    raw_results["samples"] = _json_safe(samples)
    raw_results_path = root / "raw_results.json"
    _write_json_atomic(raw_results_path, raw_results)
    status_path = root / "status.json"
    try:
        campaign, task_payloads = build_lm_eval_publication(
            results, samples, publication=publication
        )
    except Exception as error:  # noqa: BLE001
        status = {
            "schema_version": "lm-eval-publication-status-v1",
            "evaluation": "complete",
            "publication": "failed",
            "uploaded": False,
            "message": "evaluation complete, publication incomplete",
            "error": f"{type(error).__name__}: {error}",
            "raw_results_path": str(raw_results_path),
            "status_path": str(status_path),
        }
        _write_json_atomic(status_path, status)
        return status

    task_paths: list[Path] = []
    task_root = root / "tasks"
    for payload in task_payloads:
        task_name = payload["task"]["task_name"]
        path = task_root / (re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name) + ".json")
        _write_json_atomic(path, payload)
        task_paths.append(path)
    campaign_path = root / "campaign.json"
    _write_json_atomic(campaign_path, campaign)
    status: dict[str, Any] = {
        "schema_version": "lm-eval-publication-status-v1",
        "evaluation": "complete",
        "publication": "disabled"
        if not publication.get("enabled", False)
        else "pending",
        "uploaded": False,
        "campaign_path": str(campaign_path),
        "task_paths": [str(path) for path in task_paths],
        "raw_results_path": str(raw_results_path),
        "status_path": str(status_path),
    }
    _write_json_atomic(status_path, status)
    if not publication.get("enabled", False):
        return status

    incomplete_tasks = [
        payload["task"]["task_name"]
        for payload in task_payloads
        if any(
            not isinstance(sample.get("model_response"), dict)
            or sample["model_response"].get("evidence_complete") is not True
            for sample in payload.get("samples", [])
            if isinstance(sample, dict)
        )
    ]
    if incomplete_tasks:
        status.update(
            {
                "publication": "failed",
                "message": "evaluation complete, publication incomplete",
                "error": (
                    "Scoreboard publication requires complete raw response and token "
                    "evidence; missing for tasks: " + ", ".join(incomplete_tasks)
                ),
            }
        )
        _write_json_atomic(status_path, status)
        return status

    try:
        base_url = publication.get("base_url")
        token_env = publication["token_env"]
        token = os.environ.get(token_env)
        if not base_url:
            raise ScoreboardError(
                f"publication.base_url or {SCOREBOARD_BASE_URL_ENV} is required "
                "when publication is enabled"
            )
        if not token:
            raise ScoreboardError(f"publication token is missing from {token_env}")
        campaign_payload, task_by_identity, expected_identities = (
            load_publication_inputs(campaign_path, task_paths)
        )
        client = ScoreboardClient(
            base_url=base_url,
            token=token,
            timeout=publication["timeout"],
        )
        receipt = publish(
            client=client,
            campaign=campaign_payload,
            task_by_identity=task_by_identity,
            expected_identities=expected_identities,
            finalize=publication["finalize"],
        )
        status.update(
            {
                "publication": "complete",
                "uploaded": True,
                "campaign_id": receipt.get("campaign_id"),
                "receipt": receipt,
            }
        )
    except Exception as error:  # noqa: BLE001
        status.update(
            {
                "publication": "failed",
                "uploaded": False,
                "message": "evaluation complete, publication incomplete",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    _write_json_atomic(status_path, status)
    return status


def _api_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ScoreboardError(
            "--base-url must be an absolute http(s) URL, for example "
            "https://eval.rwkv.rs or https://eval.rwkv.rs/test"
        )
    if parsed.query or parsed.fragment:
        raise ScoreboardError("--base-url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/api"):
        path += "/api"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _response_json(body: bytes, *, url: str) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ScoreboardError(
            f"scoreboard returned invalid JSON from {url}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ScoreboardError(f"scoreboard returned a non-object JSON value from {url}")
    return value


def _remote_error(method: str, url: str, status: int, body: bytes) -> ScoreboardError:
    try:
        detail = _response_json(body, url=url)
        rendered = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
    except ScoreboardError:
        rendered = body.decode("utf-8", errors="replace")
    if len(rendered) > 2000:
        rendered = rendered[:2000] + "..."
    return ScoreboardError(
        f"scoreboard API {method} {url} returned HTTP {status}: {rendered}"
    )


class ScoreboardClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 3600.0,
        retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        if not token:
            raise ScoreboardError("a publication token is required")
        if timeout <= 0:
            raise ScoreboardError("--timeout must be positive")
        self.api_root = _api_root(base_url)
        self.token = token
        self.timeout = timeout
        self.retries = _environment_retries() if retries is None else retries
        self.retry_delay = (
            _environment_retry_delay() if retry_delay is None else retry_delay
        )
        if (
            isinstance(self.retries, bool)
            or not isinstance(self.retries, int)
            or self.retries < 0
        ):
            raise ScoreboardError("retries must be a non-negative integer")
        if (
            isinstance(self.retry_delay, bool)
            or not isinstance(self.retry_delay, (int, float))
            or not math.isfinite(self.retry_delay)
            or self.retry_delay <= 0
        ):
            raise ScoreboardError("retry_delay must be a positive number")

    def _wait_before_retry(self, attempt: int) -> None:
        time.sleep(min(self.retry_delay * (2**attempt), 30.0))

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.api_root}{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        body: bytes | None = None
        if payload is not None:
            body = gzip.compress(canonical_json(payload))
            headers.update(
                {
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                }
            )
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        request = Request(url, data=body, headers=headers, method=method)  # noqa: S310
        request_timeout = (
            self.timeout if timeout is None else min(self.timeout, timeout)
        )
        for attempt in range(self.retries + 1):
            try:
                with urlopen(  # noqa: S310
                    request,
                    timeout=request_timeout,
                ) as response:
                    response_body = response.read()
                    status = getattr(response, "status", 200)
            except HTTPError as error:
                response_body = error.read()
                if error.code in RETRYABLE_HTTP_STATUSES and attempt < self.retries:
                    self._wait_before_retry(attempt)
                    continue
                raise _remote_error(method, url, error.code, response_body) from error
            except (URLError, OSError) as error:
                if attempt < self.retries:
                    self._wait_before_retry(attempt)
                    continue
                reason = error.reason if isinstance(error, URLError) else error
                raise ScoreboardError(
                    f"cannot reach scoreboard API {method} {url}: {reason}"
                ) from error
            if 200 <= status < 300:
                return _response_json(response_body, url=url)
            if status in RETRYABLE_HTTP_STATUSES and attempt < self.retries:
                self._wait_before_retry(attempt)
                continue
            raise _remote_error(method, url, status, response_body)
        raise ScoreboardError(
            f"scoreboard API request exhausted retries: {method} {url}"
        )

    def preflight(
        self,
        *,
        expected_campaign_schema: str | None = None,
        expected_source: str | None = None,
    ) -> dict[str, Any]:
        response = self._request(
            "GET",
            "/v1/evaluation-publication-preflight",
            timeout=CONTROL_REQUEST_TIMEOUT,
        )
        if response.get("status") != "ready":
            raise ScoreboardError(f"scoreboard preflight is not ready: {response}")
        schema_version = response.get("schema_version")
        expected = expected_campaign_schema or SCOREBOARD_SCHEMA
        if schema_version != expected or schema_version != SCOREBOARD_SCHEMA:
            raise ScoreboardError(
                f"scoreboard schema mismatch: expected {expected!r}, got {schema_version!r}"
            )
        sources = response.get("sources")
        if not isinstance(sources, list) or not set(sources).intersection(
            SUPPORTED_SOURCES
        ):
            raise ScoreboardError(
                f"scoreboard preflight has no supported sources: {response}"
            )
        if expected_source is not None and expected_source not in sources:
            raise ScoreboardError(
                f"scoreboard does not support campaign source {expected_source!r}"
            )
        return response

    def create_campaign(self, campaign: dict[str, Any]) -> dict[str, Any]:
        run_key = campaign.get("run_key")
        if not isinstance(run_key, str):
            raise ScoreboardError("campaign.run_key must be a string")
        return self._request(
            "POST",
            "/v1/evaluation-campaigns",
            payload=campaign,
            idempotency_key=f"campaign:{run_key}",
            timeout=CONTROL_REQUEST_TIMEOUT,
        )

    def campaign_status(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}",
            timeout=CONTROL_REQUEST_TIMEOUT,
        )

    def publish_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        campaign_id = payload.get("campaign_id")
        task = payload.get("task")
        if not isinstance(campaign_id, str) or not isinstance(task, dict):
            raise ScoreboardError(
                "task payload must contain campaign_id and task object"
            )
        identity = task.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ScoreboardError(
                "task payload task.identity must be a non-empty string"
            )
        digest = content_digest(payload)
        return self._request(
            "PUT",
            f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/tasks/"
            f"{quote(identity, safe='')}",
            payload=payload,
            idempotency_key=f"publish:{digest}",
        )

    def finalize_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/finalize",
            idempotency_key=f"finalize:{campaign_id}",
            timeout=CONTROL_REQUEST_TIMEOUT,
        )


_EXPECTED_TASK_FIELDS = (
    "identity",
    "weight_sha256",
    "weight_display_name",
    "wkv_mode",
    "selector",
    "task_name",
    "task_version",
    "module_family",
    "module",
    "dataset",
    "subset",
    "evaluation_splits",
    "languages",
    "upstream_tags",
)
_TASK_PUBLICATION_FIELDS = (
    "result_files",
    "task_config",
    "environment",
    "sampling_config",
    "primary_metric",
    "metrics",
    "diagnostics",
    "samples",
)


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ScoreboardError(f"{context} must be a 64-character lowercase SHA-256")
    return value


def _require_string(value: Any, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ScoreboardError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise ScoreboardError(f"{context} must be trimmed")
    return value


def _require_string_list(
    value: Any, *, context: str, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ScoreboardError(f"{context} must be a non-empty string array")
    if any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        raise ScoreboardError(f"{context} must contain trimmed non-empty strings")
    if len(value) != len(set(value)):
        raise ScoreboardError(f"{context} must contain unique strings")
    return list(value)


def _validate_expected_task(task: dict[str, Any], *, context: str) -> None:
    missing = [field for field in _EXPECTED_TASK_FIELDS if field not in task]
    if missing:
        raise ScoreboardError(f"{context} is missing fields: {', '.join(missing)}")
    _require_string(task["identity"], context=f"{context}.identity")
    _require_sha256(task["weight_sha256"], context=f"{context}.weight_sha256")
    _require_string(
        task["weight_display_name"], context=f"{context}.weight_display_name"
    )
    mode = task["wkv_mode"]
    if mode not in {"fp16", "fp32io16"}:
        raise ScoreboardError(f"{context}.wkv_mode must be fp16 or fp32io16")
    for field in (
        "selector",
        "task_name",
        "task_version",
        "module_family",
        "module",
        "dataset",
    ):
        _require_string(task[field], context=f"{context}.{field}")
    _require_string(task["subset"], context=f"{context}.subset", allow_empty=True)
    for field in ("evaluation_splits", "languages", "upstream_tags"):
        _require_string_list(task[field], context=f"{context}.{field}")
    expected_identity = f"{task['weight_sha256']}:{mode}:{task['task_name']}"
    if task["identity"] != expected_identity:
        raise ScoreboardError(
            f"{context}.identity must match weight_sha256:wkv_mode:task_name"
        )


def _validate_lm_eval_expected_task(task: dict[str, Any], *, context: str) -> None:
    for field in (
        "identity",
        "weight_sha256",
        "weight_display_name",
        "wkv_mode",
        "task_name",
        "selector",
        "task_version",
        "module_family",
        "module",
    ):
        _require_string(task.get(field), context=f"{context}.{field}")
    _require_sha256(task["weight_sha256"], context=f"{context}.weight_sha256")
    if task["wkv_mode"] not in {"fp16", "fp32io16"}:
        raise ScoreboardError(f"{context}.wkv_mode must be fp16 or fp32io16")
    if task["identity"] != (
        f"{task['weight_sha256']}:{task['wkv_mode']}:{task['task_name']}"
    ):
        raise ScoreboardError(
            f"{context}.identity must match weight_sha256:wkv_mode:task_name"
        )
    for field in ("dataset", "subset"):
        value = task.get(field)
        if value is not None:
            _require_string(value, context=f"{context}.{field}", allow_empty=True)
    for field in ("evaluation_splits", "languages", "upstream_tags"):
        _require_string_list(
            task.get(field, []), context=f"{context}.{field}", allow_empty=True
        )


def _validate_scoreboard_task(task: dict[str, Any], *, context: str) -> None:
    required = (
        "identity",
        "weight_sha256",
        "weight_display_name",
        "wkv_mode",
        "benchmark",
        "task_name",
        "task_version",
        "evaluation_splits",
        "languages",
        "tags",
    )
    missing = [field for field in required if field not in task]
    if missing:
        raise ScoreboardError(f"{context} is missing fields: {', '.join(missing)}")
    _require_string(task["identity"], context=f"{context}.identity")
    _require_sha256(task["weight_sha256"], context=f"{context}.weight_sha256")
    _require_string(
        task["weight_display_name"], context=f"{context}.weight_display_name"
    )
    if task["wkv_mode"] not in {"fp16", "fp32io16"}:
        raise ScoreboardError(f"{context}.wkv_mode must be fp16 or fp32io16")
    for field in ("benchmark", "task_name", "task_version"):
        _require_string(task[field], context=f"{context}.{field}")
    if (
        task["identity"]
        != f"{task['weight_sha256']}:{task['wkv_mode']}:{task['task_name']}"
    ):
        raise ScoreboardError(f"{context}.identity does not match task dimensions")
    for field in ("evaluation_splits", "languages", "tags"):
        _require_string_list(
            task[field], context=f"{context}.{field}", allow_empty=True
        )
    for field in ("dataset", "subset"):
        if task.get(field) is not None:
            _require_string(task[field], context=f"{context}.{field}", allow_empty=True)


def _validate_campaign(campaign: dict[str, Any]) -> list[str]:
    schema_version = campaign.get("schema_version")
    if schema_version != SCOREBOARD_SCHEMA:
        raise ScoreboardError(f"campaign.schema_version must be {SCOREBOARD_SCHEMA!r}")
    run_key = campaign.get("run_key")
    if not isinstance(run_key, str) or re.fullmatch(r"[0-9a-f]{64}", run_key) is None:
        raise ScoreboardError(
            "campaign.run_key must be a 64-character hexadecimal string"
        )
    if campaign.get("source") not in SUPPORTED_SOURCES:
        raise ScoreboardError("campaign.source must be one of scoreboard sources")
    for field in ("config_sha256", "registry_sha256", "contract_sha256"):
        _require_sha256(campaign.get(field), context=f"campaign.{field}")
    configured = _require_string_list(
        campaign.get("configured_benchmarks"), context="campaign.configured_benchmarks"
    )
    resolved = _require_string_list(
        campaign.get("resolved_benchmarks"), context="campaign.resolved_benchmarks"
    )
    skipped = _require_string_list(
        campaign.get("skipped_benchmarks"),
        context="campaign.skipped_benchmarks",
        allow_empty=True,
    )
    if set(resolved).intersection(skipped) or set(resolved).union(skipped) != set(
        configured
    ):
        raise ScoreboardError(
            "campaign benchmark status must partition configured benchmarks"
        )
    expected = campaign.get("expected_tasks")
    if not isinstance(expected, list) or not expected:
        raise ScoreboardError("campaign.expected_tasks must be a non-empty array")
    identities: list[str] = []
    for index, task in enumerate(expected):
        if not isinstance(task, dict) or not isinstance(task.get("identity"), str):
            raise ScoreboardError(
                f"campaign.expected_tasks[{index}] lacks task identity"
            )
        identity = task["identity"]
        if not identity:
            raise ScoreboardError(
                f"campaign.expected_tasks[{index}] has empty identity"
            )
        _validate_scoreboard_task(task, context=f"campaign.expected_tasks[{index}]")
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise ScoreboardError("campaign.expected_tasks identities must be unique")
    benchmarks = {task["benchmark"] for task in expected}
    if benchmarks != set(resolved):
        raise ScoreboardError(
            "campaign expected task benchmarks must match resolved_benchmarks"
        )
    expected_without_key = dict(campaign)
    expected_without_key.pop("run_key", None)
    try:
        from scripts.convert_scoreboard_payloads import campaign_run_key
    except ModuleNotFoundError:
        from convert_scoreboard_payloads import campaign_run_key  # type: ignore
    if run_key != campaign_run_key(expected_without_key):
        raise ScoreboardError(
            "campaign.run_key does not match normalized campaign payload"
        )
    return identities


def _validate_tasks(
    payloads: list[dict[str, Any]],
    expected_identities: list[str],
    campaign: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    expected_tasks = {
        task["identity"]: task
        for task in campaign["expected_tasks"]
        if isinstance(task, dict) and isinstance(task.get("identity"), str)
    }
    for index, payload in enumerate(payloads):
        if payload.get("schema_version") != SCOREBOARD_SCHEMA:
            raise ScoreboardError(
                f"task file {index} schema_version must be {SCOREBOARD_SCHEMA!r}"
            )
        task = payload.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("identity"), str):
            raise ScoreboardError(f"task file {index} lacks task.identity")
        identity = task["identity"]
        if not identity:
            raise ScoreboardError(f"task file {index} has empty task.identity")
        if identity in by_identity:
            raise ScoreboardError(f"duplicate task payload for identity {identity}")
        if payload.get("task") != expected_tasks.get(identity):
            raise ScoreboardError(
                f"task file {index} task metadata does not match campaign expected_tasks"
            )
        campaign_id = payload.get("campaign_id")
        if campaign_id is not None and (
            not isinstance(campaign_id, str) or not campaign_id
        ):
            raise ScoreboardError(
                f"task file {index}.campaign_id must be null or a string"
            )
        missing = [field for field in _TASK_PUBLICATION_FIELDS if field not in payload]
        if missing:
            raise ScoreboardError(
                f"task file {index} is missing fields: {', '.join(missing)}"
            )
        if not isinstance(payload["result_files"], list) or not payload["result_files"]:
            raise ScoreboardError(f"task file {index}.result_files must be non-empty")
        for field in ("task_config", "environment", "sampling_config", "diagnostics"):
            if not isinstance(payload[field], dict):
                raise ScoreboardError(f"task file {index}.{field} must be an object")
        if not isinstance(payload["metrics"], dict) or not payload["metrics"]:
            raise ScoreboardError(f"task file {index}.metrics must be non-empty")
        if not isinstance(payload["samples"], list):
            raise ScoreboardError(f"task file {index}.samples must be an array")
        by_identity[identity] = payload
    expected = set(expected_identities)
    actual = set(by_identity)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing=" + ",".join(sorted(missing)))
        if extra:
            parts.append("unexpected=" + ",".join(sorted(extra)))
        raise ScoreboardError(
            "task payload set does not match campaign: " + "; ".join(parts)
        )
    return by_identity


def load_publication_inputs(
    campaign_path: Path, task_paths: list[Path]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    """Load strict DTOs only; raw evaluator formats are rejected."""

    campaign = _load_json_object(campaign_path)
    payloads = [_load_json_object(path) for path in task_paths]
    expected_identities = _validate_campaign(campaign)
    tasks = _validate_tasks(payloads, expected_identities, campaign)
    return campaign, tasks, expected_identities


def publish(
    *,
    client: ScoreboardClient,
    campaign: dict[str, Any],
    task_by_identity: dict[str, dict[str, Any]],
    expected_identities: list[str],
    finalize: bool = True,
) -> dict[str, Any]:
    schema_version = campaign.get("schema_version")
    if not isinstance(schema_version, str):
        raise ScoreboardError(
            "campaign.schema_version must be a string before publication"
        )
    preflight = client.preflight(
        expected_campaign_schema=schema_version,
        expected_source=campaign.get("source"),
    )
    campaign_receipt = client.create_campaign(campaign)
    campaign_id = campaign_receipt.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ScoreboardError(
            f"campaign creation returned no campaign_id: {campaign_receipt}"
        )
    status = client.campaign_status(campaign_id)
    if status.get("campaign_id") != campaign_id:
        raise ScoreboardError("campaign status returned a different campaign_id")
    acknowledged = status.get("task_hashes", {})
    if not isinstance(acknowledged, dict):
        raise ScoreboardError("campaign status has invalid task_hashes")
    if status.get("status") == "complete":
        mismatched = []
        for identity in expected_identities:
            payload = deepcopy(task_by_identity[identity])
            payload["campaign_id"] = campaign_id
            if acknowledged.get(identity) != content_digest(payload):
                mismatched.append(identity)
        if mismatched:
            raise ScoreboardError(
                "campaign is already complete and has different task content: "
                + ", ".join(mismatched)
            )
        return {
            "campaign_id": campaign_id,
            "campaign": campaign_receipt,
            "preflight": preflight,
            "tasks": [
                {"identity": identity, "action": "unchanged"}
                for identity in expected_identities
            ],
            "finalize": {"status": "complete", "task_count": len(expected_identities)},
        }

    task_receipts: list[dict[str, Any]] = []
    for identity in expected_identities:
        payload = deepcopy(task_by_identity[identity])
        payload["campaign_id"] = campaign_id
        digest = content_digest(payload)
        if acknowledged.get(identity) == digest:
            task_receipts.append(
                {
                    "identity": identity,
                    "action": "unchanged",
                    "content_sha256": digest,
                }
            )
            continue
        receipt = client.publish_task(payload)
        task_receipts.append({"identity": identity, **receipt})

    final_receipt: dict[str, Any] | None = None
    if finalize:
        final_receipt = client.finalize_campaign(campaign_id)
    return {
        "campaign_id": campaign_id,
        "campaign": campaign_receipt,
        "preflight": preflight,
        "tasks": task_receipts,
        "finalize": final_receipt,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload already-converted scoreboard campaign/task DTOs."
    )
    parser.add_argument(
        "--base-url",
        help=f"scoreboard origin or deployment prefix (env: {SCOREBOARD_BASE_URL_ENV})",
    )
    parser.add_argument(
        "--token",
        help=(
            "publication token override; prefer the environment selected by "
            f"{SCOREBOARD_TOKEN_ENV_ENV} (default: {SCOREBOARD_TOKEN_ENV})"
        ),
    )
    parser.add_argument(
        "--token-env",
        help=(
            "name of the environment variable containing the publication token "
            f"(env selector: {SCOREBOARD_TOKEN_ENV_ENV})"
        ),
    )
    parser.add_argument("--campaign", type=Path, help="strict campaign DTO JSON")
    parser.add_argument(
        "--task",
        type=Path,
        action="append",
        default=[],
        help="strict task DTO JSON; repeat once per expected task",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help=f"HTTP timeout in seconds (env: {SCOREBOARD_TIMEOUT_ENV}; default: 3600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate strict DTOs without contacting the scoreboard",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="only verify the remote publication contract and token",
    )
    finalize = parser.add_mutually_exclusive_group()
    finalize.add_argument(
        "--finalize",
        dest="finalize",
        action="store_true",
        help=f"finalize the campaign (env: {SCOREBOARD_FINALIZE_ENV}; default: true)",
    )
    finalize.add_argument(
        "--no-finalize",
        dest="finalize",
        action="store_false",
        help="upload tasks but leave the campaign incomplete",
    )
    parser.set_defaults(finalize=None)
    return parser


def _require_credentials(args: argparse.Namespace) -> tuple[str, str, float, bool]:
    explicit = {
        key: value
        for key, value in {
            "base_url": args.base_url,
            "token_env": args.token_env,
            "timeout": args.timeout,
            "finalize": args.finalize,
        }.items()
        if value is not None
    }
    settings = resolve_publication_settings(explicit)
    base_url = settings.get("base_url")
    if not base_url:
        raise ScoreboardError(f"--base-url or {SCOREBOARD_BASE_URL_ENV} is required")
    token_env = settings["token_env"]
    token = args.token or os.environ.get(token_env)
    if not token:
        raise ScoreboardError(
            f"--token or {token_env} is required; never put the token in a file"
        )
    return base_url, token, settings["timeout"], settings["finalize"]


def _write_json(value: dict[str, Any], output: TextIO | None = None) -> None:
    if output is None:
        output = sys.stdout
    json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.preflight_only and args.dry_run:
            raise ScoreboardError("--preflight-only and --dry-run cannot be combined")
        if args.preflight_only:
            base_url, token, timeout, _ = _require_credentials(args)
            client = ScoreboardClient(base_url=base_url, token=token, timeout=timeout)
            _write_json(client.preflight())
            return 0
        if args.campaign is None or not args.task:
            raise ScoreboardError("--campaign and at least one --task are required")
        campaign, task_by_identity, expected_identities = load_publication_inputs(
            args.campaign, args.task
        )
        if args.dry_run:
            _write_json(
                {
                    "dry_run": True,
                    "campaign_run_key": campaign["run_key"],
                    "expected_task_count": len(expected_identities),
                    "task_identities": expected_identities,
                }
            )
            return 0
        base_url, token, timeout, finalize = _require_credentials(args)
        client = ScoreboardClient(base_url=base_url, token=token, timeout=timeout)
        result = publish(
            client=client,
            campaign=campaign,
            task_by_identity=task_by_identity,
            expected_identities=expected_identities,
            finalize=finalize,
        )
        _write_json(result)
        return 0
    except (ScoreboardError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
