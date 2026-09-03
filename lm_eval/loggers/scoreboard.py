"""Publish one completed native lm-eval benchmark to Scoreboard."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from collections.abc import Mapping


SCHEMA_VERSION = "scoreboard-v1"
SOURCE = "lm-eval-harness"
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_SAMPLES_PER_OUTCOME = 20
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
SAMPLING_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_tokens",
    "max_gen_toks": "max_tokens",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}
CONTRACT = {
    "producer": SOURCE,
    "artifact_schema": 1,
    "publication_schema": SCHEMA_VERSION,
    "max_samples_per_outcome": MAX_SAMPLES_PER_OUTCOME,
}


class ScoreboardError(RuntimeError):
    """Raised when Scoreboard publication fails."""


def _json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    return _json(value).decode("utf-8")


@dataclass(frozen=True, slots=True)
class PublicationConfig:
    base_url: str
    token_env: str
    model_sha256: str
    tasks: dict[str, dict[str, Any]]
    timeout: float
    control_timeout: float
    retries: int
    retry_delay: float
    rerun_reason: str | None
    dry_run: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PublicationConfig:
        value = dict(raw)
        model_sha256 = str(value["model_sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None:
            raise ScoreboardError("publication.model_sha256 must be lowercase SHA-256")
        return cls(
            base_url=str(value["base_url"]),
            token_env=str(value.get("token_env", "SCOREBOARD_PUBLICATION_TOKEN")),
            model_sha256=model_sha256,
            tasks=value.get("tasks", value.get("task_metadata", {})),
            timeout=float(value.get("timeout", 3600)),
            control_timeout=float(value.get("control_timeout", 30)),
            retries=int(value.get("retries", 2)),
            retry_delay=float(value.get("retry_delay", 1)),
            rerun_reason=value.get("rerun_reason"),
            dry_run=value.get("dry_run") is True,
        )


def _metrics(results: dict[str, Any], task_name: str) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in results["results"][task_name].items()
        if name != "sample_len"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


def _primary_metric(
    metrics: dict[str, float], task_config: dict[str, Any], policy: dict[str, Any]
) -> str:
    name = policy.get("primary_metric")
    if name is None:
        name = task_config["metric_list"][0]["metric"]
    return next(
        metric
        for metric in metrics
        if metric == name or metric.split(",", 1)[0] == name
    )


def _sample(
    sample_index: int,
    task_name: str,
    row: dict[str, Any],
    metric_name: str,
    include_answer: bool,
) -> tuple[dict[str, Any], str, bool]:
    evidence = [item for group in row.get("response_evidence", []) for item in group]
    first_evidence = evidence[0] if evidence else {}
    raw_response = first_evidence.get("raw_response") or {}
    choices = raw_response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw_completion = _text(
        choice.get("text")
        or message.get("content")
        or message.get("tool_calls")
        or row["resps"]
    )
    extracted_answer = _text(row["filtered_resps"])
    score = row.get(metric_name)
    outcome = (
        "unanswered"
        if not raw_completion.strip()
        else {1: "correct", 0: "incorrect"}.get(score, "undetermined")
    )
    stop_reason = str(
        first_evidence.get("finish_reason")
        or first_evidence.get("stop_reason")
        or choice.get("finish_reason")
        or choice.get("stop_reason")
        or "unknown"
    )
    truncated = first_evidence.get("truncation") is True or stop_reason in {
        "length",
        "max_tokens",
        "model_length",
    }
    document = {
        **row["doc"],
        "task_name": task_name,
        "target": row["target"],
    }
    detail = {
        "sample_index": sample_index,
        "document_index": int(row["doc_id"]),
        "document": document,
        "metrics": {name: row[name] for name in row["metrics"]},
        "model_response": {
            "text": raw_completion,
            "stop_reason": stop_reason,
            "tool_calls": message.get("tool_calls") or [],
        },
    }
    if include_answer:
        latency = sum(item.get("latency_ms") or 0 for item in evidence) or None
        generated_tokens = sum(
            len(item.get("output_token_ids") or []) for item in evidence
        )
        prompts = [item["prompt"] for item in evidence if item.get("prompt")]
        detail["answer"] = {
            "outcome": outcome,
            "problem_id": str(row["doc_id"]),
            "repeat_id": int(row.get("repeat_id", 0)),
            "ground_truth": _text(row["target"]),
            "extracted_answer": extracted_answer,
            "assembled_prompt": _text(prompts),
            "raw_completion": raw_completion,
            "fail_reason": (
                None
                if outcome == "correct"
                else "empty_answer"
                if outcome == "unanswered"
                else "truncated"
                if truncated
                else "answer_mismatch"
                if outcome == "incorrect"
                else "undetermined"
            ),
            "generated_tokens": generated_tokens,
            "latency_ms": latency,
        }
    return detail, outcome, truncated


def _samples(
    path: Path, task_name: str, metric_name: str, include_answer: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details: list[dict[str, Any]] = []
    outcome_counts = dict.fromkeys(
        ("correct", "incorrect", "unanswered", "undetermined"), 0
    )
    uploaded_counts = dict.fromkeys(outcome_counts, 0)
    total = truncated = 0
    with path.open(encoding="utf-8") as sample_file:
        for line in sample_file:
            if not line.strip():
                continue
            detail, outcome, is_truncated = _sample(
                len(details),
                task_name,
                json.loads(line),
                metric_name,
                include_answer,
            )
            total += 1
            truncated += is_truncated
            outcome_counts[outcome] += 1
            if uploaded_counts[outcome] >= MAX_SAMPLES_PER_OUTCOME:
                continue
            uploaded_counts[outcome] += 1
            detail["sample_index"] = len(details)
            details.append(detail)
    return details, {
        "samples_total": total,
        "samples_uploaded": len(details),
        "outcome_counts": outcome_counts,
        "truncation_rate": truncated / total if total else 0.0,
    }


def _sampling_config(
    results: dict[str, Any], task_config: dict[str, Any], model_args: dict[str, Any]
) -> dict[str, Any]:
    run_config = results["config"]
    effective = {
        **model_args,
        **(results.get("sampling_config") or {}),
        **(run_config.get("sampling_config") or {}),
        **(run_config.get("gen_kwargs") or {}),
        **(task_config.get("generation_kwargs") or {}),
    }
    return {
        target: effective[source]
        for source, target in SAMPLING_FIELDS.items()
        if source in effective
    }


def build_task_publication(
    results: dict[str, Any],
    metadata: dict[str, Any],
    samples_path: Path,
    task_name: str,
    config: PublicationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one compact Scoreboard publication from one benchmark artifact."""
    task_config = results["configs"][task_name]
    task_metadata = task_config.get("metadata") or {}
    benchmark = str(task_metadata.get("benchmark_name") or task_name)
    policy = config.tasks.get(task_name, config.tasks.get(benchmark, {}))
    run_config = results["config"]
    model_args = run_config["model_args"]
    model_name = str(
        metadata.get("model")
        or model_args.get("model")
        or model_args.get("pretrained")
        or run_config["model"]
    )
    precision = str(
        metadata.get("precision")
        or task_metadata.get("wkv_mode")
        or task_config.get("wkv_mode")
        or model_args.get("wkv_mode")
        or "fp32io16"
    )
    split = task_config.get("test_split") or task_config.get("validation_split")
    task = {
        "identity": f"{config.model_sha256}:{precision}:{task_name}",
        "weight_sha256": config.model_sha256,
        "weight_display_name": model_name,
        "wkv_mode": precision,
        "benchmark": benchmark,
        "task_name": task_name,
        "task_version": str(
            task_metadata.get("task_version")
            or results["versions"][task_name]
            or task_metadata.get("version")
        ),
        "dataset": task_config.get("dataset_path") or task_metadata.get("dataset"),
        "subset": task_config.get("dataset_name") or task_metadata.get("subset"),
        "evaluation_splits": sorted(
            task_metadata.get("evaluation_splits") or ([split] if split else [])
        ),
        "languages": sorted(task_metadata.get("languages") or []),
        "tags": sorted(
            task_metadata.get("tags") or task_metadata.get("upstream_tags") or []
        ),
    }
    metrics = _metrics(results, task_name)
    primary_metric = _primary_metric(metrics, task_config, policy)
    sample_metric = str(policy.get("outcome_metric") or primary_metric.split(",", 1)[0])
    comparison = policy.get("comparison")
    details, diagnostics = _samples(
        samples_path, task_name, sample_metric, comparison is not None
    )
    counts = results["n-samples"][task_name]
    task_summary = {
        "task": task_name,
        "dataset_path": task["dataset"],
        "dataset_name": task["subset"],
        "output_type": task_config.get("output_type"),
        "num_fewshot": task_config.get("num_fewshot"),
        "original_num_docs": counts["original"],
        "effective_num_docs": counts["effective"],
    }
    publication = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "result_files": [],
        "task_config": task_summary,
        "environment": {
            "framework": SOURCE,
            "version": metadata.get("version") or results["lm_eval_version"],
            "model": model_name,
            "precision": precision,
            "backend": metadata.get("backend") or run_config["model"],
        },
        "sampling_config": _sampling_config(results, task_config, model_args),
        "primary_metric": primary_metric,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "samples": details,
    }
    if comparison is not None:
        publication["comparison"] = {
            **comparison,
            "samples": len(details),
            "truncation_rate": diagnostics["truncation_rate"],
        }
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "run_key": "",
        "source": SOURCE,
        "config_sha256": _sha256(
            {
                "task": task_summary,
                "sampling": publication["sampling_config"],
                "policy": policy,
            }
        ),
        "registry_sha256": _sha256([task]),
        "contract_sha256": _sha256(CONTRACT),
        "configured_benchmarks": [benchmark],
        "resolved_benchmarks": [benchmark],
        "skipped_benchmarks": [],
        "expected_tasks": [task],
        "rerun_reason": config.rerun_reason,
    }
    run_key_payload = dict(campaign)
    run_key_payload.pop("run_key")
    campaign["run_key"] = _sha256(run_key_payload)
    return campaign, publication


def _api_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/api"):
        path += "/api"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _encode(payload: dict[str, Any]) -> tuple[str, bytes]:
    raw = _json(payload)
    compressed = gzip.compress(raw)
    if len(raw) > MAX_UNCOMPRESSED_BYTES or len(compressed) > MAX_COMPRESSED_BYTES:
        raise ScoreboardError(
            "Scoreboard publication exceeds the 64 MiB compressed or 256 MiB raw limit"
        )
    return hashlib.sha256(raw).hexdigest(), compressed


class ScoreboardPublisher:
    """Own campaign creation, task upload, retries, and finalization."""

    def __init__(self, config: PublicationConfig) -> None:
        self.config = config
        self.base_url = _api_root(config.base_url)
        self.token = os.environ[config.token_env]

    def _send(
        self,
        method: str,
        path: str,
        idempotency_key: str,
        *,
        body: bytes | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Idempotency-Key": idempotency_key,
        }
        if body is not None:
            headers.update(
                {"Content-Type": "application/json", "Content-Encoding": "gzip"}
            )
        url = self.base_url + path
        for attempt in range(self.config.retries + 1):
            try:
                request = Request(  # noqa: S310 - publication URL is configured.
                    url, data=body, method=method, headers=headers
                )
                with urlopen(  # noqa: S310 - publication URL is configured.
                    request, timeout=timeout or self.config.timeout
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                if (
                    error.code not in RETRYABLE_HTTP_STATUSES
                    or attempt == self.config.retries
                ):
                    raise ScoreboardError(
                        f"scoreboard {method} {path} returned HTTP {error.code}"
                    ) from error
            except (TimeoutError, URLError) as error:
                if attempt == self.config.retries:
                    raise ScoreboardError(
                        f"cannot reach scoreboard: {error}"
                    ) from error
            time.sleep(min(self.config.retry_delay * 2**attempt, 30))
        raise ScoreboardError("scoreboard retries exhausted")

    def publish(
        self, campaign: dict[str, Any], publication: dict[str, Any]
    ) -> dict[str, Any]:
        preflight = self._send(
            "GET",
            "/v1/evaluation-publication-preflight",
            "preflight:scoreboard-v1",
            timeout=self.config.control_timeout,
        )
        if (
            preflight["status"] != "ready"
            or preflight["schema_version"] != SCHEMA_VERSION
            or SOURCE not in preflight["sources"]
        ):
            raise ScoreboardError(
                f"scoreboard does not support {SCHEMA_VERSION}/{SOURCE}"
            )
        _, campaign_body = _encode(campaign)
        campaign_receipt = self._send(
            "POST",
            "/v1/evaluation-campaigns",
            f"campaign:{campaign['run_key']}",
            body=campaign_body,
            timeout=self.config.control_timeout,
        )
        campaign_id = campaign_receipt["campaign_id"]
        if campaign_receipt["status"] == "complete":
            return {
                "campaign_id": campaign_id,
                "preflight": preflight,
                "campaign": campaign_receipt,
                "task": None,
                "finalize": None,
            }
        task_publication = {**publication, "campaign_id": campaign_id}
        digest, task_body = _encode(task_publication)
        identity = task_publication["task"]["identity"]
        task_receipt = self._send(
            "PUT",
            f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/tasks/"
            f"{quote(identity, safe='')}",
            f"publish:{digest}",
            body=task_body,
        )
        finalize_receipt = self._send(
            "POST",
            f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/finalize",
            f"finalize:{campaign_id}",
            timeout=self.config.control_timeout,
        )
        return {
            "campaign_id": campaign_id,
            "preflight": preflight,
            "campaign": campaign_receipt,
            "task": task_receipt,
            "finalize": finalize_receipt,
        }


class ScoreboardTaskCallback:
    """Read one completed benchmark artifact and publish it immediately."""

    def __init__(self, *, config: PublicationConfig) -> None:
        self.config = config

    def __call__(
        self, *, task_name: str, artifact_paths: Mapping[str, str | Path]
    ) -> dict[str, Any]:
        status = {
            "task_name": task_name,
            "evaluation": "complete",
            "publication": "failed",
            "uploaded": False,
        }
        try:
            results = json.loads(Path(artifact_paths["results"]).read_text("utf-8"))
            metadata = json.loads(Path(artifact_paths["metadata"]).read_text("utf-8"))
            campaign, publication = build_task_publication(
                results,
                metadata,
                Path(artifact_paths["samples"]),
                task_name,
                self.config,
            )
            status["diagnostics"] = publication["diagnostics"]
            if self.config.dry_run:
                status["publication"] = "validated"
                return status
            receipt = ScoreboardPublisher(self.config).publish(campaign, publication)
            status.update(
                publication="complete",
                uploaded=True,
                campaign_id=receipt["campaign_id"],
                receipt=receipt,
            )
        except (KeyError, OSError, ScoreboardError, TypeError, ValueError) as error:
            status["error"] = f"{type(error).__name__}: {error}"
        return status
