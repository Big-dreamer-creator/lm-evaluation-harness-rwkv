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
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


try:
    from scripts.convert_scoreboard_payloads import (
        CAMPAIGN_SCHEMA,
        LIGHTEVAL_VERSION,
        LM_EVAL_CAMPAIGN_SCHEMA,
        LM_EVAL_TASK_SCHEMA,
        TASK_SCHEMA,
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
        LIGHTEVAL_VERSION,
        LM_EVAL_CAMPAIGN_SCHEMA,
        LM_EVAL_TASK_SCHEMA,
        TASK_SCHEMA,
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


SUPPORTED_CAMPAIGN_SCHEMAS = {CAMPAIGN_SCHEMA, LM_EVAL_CAMPAIGN_SCHEMA}
SUPPORTED_TASK_SCHEMAS = {TASK_SCHEMA, LM_EVAL_TASK_SCHEMA}
SCOREBOARD_BASE_URL_ENV = "SCOREBOARD_BASE_URL"
SCOREBOARD_TOKEN_ENV_ENV = "SCOREBOARD_PUBLICATION_TOKEN_ENV"
SCOREBOARD_TOKEN_ENV = "SCOREBOARD_PUBLICATION_TOKEN"
SCOREBOARD_TIMEOUT_ENV = "SCOREBOARD_UPLOAD_TIMEOUT"
SCOREBOARD_FINALIZE_ENV = "SCOREBOARD_UPLOAD_FINALIZE"
SCOREBOARD_MODEL_SHA256_ENV = "SCOREBOARD_MODEL_SHA256"
SCOREBOARD_MODEL_REVISION_ENV = "SCOREBOARD_MODEL_REVISION"


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
        if payload.get("diagnostics", {}).get("evidence_complete") is not True
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
    def __init__(self, *, base_url: str, token: str, timeout: float = 3600.0) -> None:
        if not token:
            raise ScoreboardError("a publication token is required")
        if timeout <= 0:
            raise ScoreboardError("--timeout must be positive")
        self.api_root = _api_root(base_url)
        self.token = token
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
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
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                response_body = response.read()
                status = getattr(response, "status", 200)
        except HTTPError as error:
            raise _remote_error(method, url, error.code, error.read()) from error
        except URLError as error:
            raise ScoreboardError(
                f"cannot reach scoreboard API {method} {url}: {error.reason}"
            ) from error
        except OSError as error:
            raise ScoreboardError(
                f"cannot reach scoreboard API {method} {url}: {error}"
            ) from error
        if status < 200 or status >= 300:
            raise _remote_error(method, url, status, response_body)
        return _response_json(response_body, url=url)

    def preflight(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/evaluation-publication-preflight")
        if response.get("status") != "ready":
            raise ScoreboardError(f"scoreboard preflight is not ready: {response}")
        schema_version = response.get("schema_version")
        supported_schemas = response.get("supported_schemas")
        if schema_version not in SUPPORTED_CAMPAIGN_SCHEMAS and not (
            isinstance(supported_schemas, list)
            and LM_EVAL_CAMPAIGN_SCHEMA in supported_schemas
        ):
            raise ScoreboardError(
                "scoreboard schema mismatch: "
                f"expected one of {sorted(SUPPORTED_CAMPAIGN_SCHEMAS)}, "
                f"got {schema_version!r}"
            )
        if (
            schema_version == CAMPAIGN_SCHEMA
            and response.get("lighteval_version") != LIGHTEVAL_VERSION
        ):
            raise ScoreboardError(
                "scoreboard LightEval version mismatch: "
                f"expected {LIGHTEVAL_VERSION}, "
                f"got {response.get('lighteval_version')!r}"
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
        )

    def campaign_status(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}"
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
        )


def _validate_campaign(campaign: dict[str, Any]) -> list[str]:
    schema_version = campaign.get("schema_version")
    if schema_version not in SUPPORTED_CAMPAIGN_SCHEMAS:
        raise ScoreboardError(
            f"campaign.schema_version must be one of "
            f"{sorted(SUPPORTED_CAMPAIGN_SCHEMAS)!r}; run "
            "scripts/convert_scoreboard_payloads.py before uploading raw evaluator data"
        )
    run_key = campaign.get("run_key")
    if not isinstance(run_key, str) or re.fullmatch(r"[0-9a-f]{64}", run_key) is None:
        raise ScoreboardError(
            "campaign.run_key must be a 64-character hexadecimal string"
        )
    if (
        schema_version == CAMPAIGN_SCHEMA
        and campaign.get("lighteval_version") != LIGHTEVAL_VERSION
    ):
        raise ScoreboardError(
            f"campaign.lighteval_version must be {LIGHTEVAL_VERSION!r}"
        )
    if schema_version == LM_EVAL_CAMPAIGN_SCHEMA:
        evaluator = campaign.get("evaluator")
        if not isinstance(evaluator, dict) or evaluator.get("framework") != "lm-eval":
            raise ScoreboardError(
                "lm-eval campaign must declare evaluator.framework='lm-eval'"
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
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise ScoreboardError("campaign.expected_tasks identities must be unique")
    return identities


def _validate_tasks(
    payloads: list[dict[str, Any]], expected_identities: list[str]
) -> dict[str, dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    for index, payload in enumerate(payloads):
        if payload.get("schema_version") not in SUPPORTED_TASK_SCHEMAS:
            raise ScoreboardError(
                f"task file {index} schema_version must be one of "
                f"{sorted(SUPPORTED_TASK_SCHEMAS)!r}"
            )
        task = payload.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("identity"), str):
            raise ScoreboardError(f"task file {index} lacks task.identity")
        identity = task["identity"]
        if not identity:
            raise ScoreboardError(f"task file {index} has empty task.identity")
        if identity in by_identity:
            raise ScoreboardError(f"duplicate task payload for identity {identity}")
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
    tasks = _validate_tasks(payloads, expected_identities)
    return campaign, tasks, expected_identities


def publish(
    *,
    client: ScoreboardClient,
    campaign: dict[str, Any],
    task_by_identity: dict[str, dict[str, Any]],
    expected_identities: list[str],
    finalize: bool = True,
) -> dict[str, Any]:
    preflight = client.preflight()
    campaign_receipt = client.create_campaign(campaign)
    campaign_id = campaign_receipt.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ScoreboardError(
            f"campaign creation returned no campaign_id: {campaign_receipt}"
        )
    status = client.campaign_status(campaign_id)
    if status.get("campaign_id") != campaign_id:
        raise ScoreboardError("campaign status returned a different campaign_id")
    acknowledged = status.get("acknowledged_task_digests", {})
    if not isinstance(acknowledged, dict):
        raise ScoreboardError("campaign status has invalid acknowledged_task_digests")
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
                {"identity": identity, "disposition": "unchanged"}
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
                    "disposition": "unchanged",
                    "content_digest": digest,
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
