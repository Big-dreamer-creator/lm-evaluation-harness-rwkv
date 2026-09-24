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
    "max_completion_tokens": "max_completion_tokens",
    "max_new_tokens": "max_new_tokens",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "penalty_decay": "penalty_decay",
    "repetition_penalty": "repetition_penalty",
    "do_sample": "do_sample",
    "num_samples": "num_samples",
    "seed": "seed",
    "until": "until",
    "stop": "stop",
}
CHAT_TEMPLATE_FIELDS = {
    "prompt_template": "rwkv_prompt_template",
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


def _task_policy(
    config: PublicationConfig, task_name: str, benchmark: str
) -> dict[str, Any]:
    policy = config.tasks.get(task_name, config.tasks.get(benchmark, {}))
    return policy if isinstance(policy, dict) else {}


def _prompt_from_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        first = arguments.get("gen_args_0")
        if isinstance(first, dict) and "arg_0" in first:
            return _text(first["arg_0"])
        if "prompt" in arguments:
            return _text(arguments["prompt"])
    if isinstance(arguments, list) and arguments:
        first = arguments[0]
        if isinstance(first, (list, tuple)) and first:
            return _text(first[0])
        return _text(first)
    return ""


def _loglikelihood_argmax(resps: Any) -> int | None:
    """Index of the highest-logprob choice, if `resps` is a loglikelihood pair list."""
    if not isinstance(resps, list) or not resps:
        return None
    scores = []
    for item in resps:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return None
        try:
            scores.append(float(item[0]))
        except (TypeError, ValueError):
            return None
    return max(range(len(scores)), key=lambda i: scores[i])


def _mc_choice_text(arguments: Any, index: int) -> str | None:
    """Continuation text lm-eval sent for a multiple_choice option, by request index."""
    if isinstance(arguments, dict):
        entry = arguments.get(f"gen_args_{index}")
        if isinstance(entry, dict) and "arg_1" in entry:
            return _text(entry["arg_1"]).strip()
    return None


def _field(
    task_metadata: dict[str, Any],
    task_config: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    value = (
        policy.get("field") or task_metadata.get("field") or task_config.get("field")
    )
    if value is None:
        raise ScoreboardError(
            "publication field is required; set task metadata or publication.tasks"
        )
    return str(value)


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
        configured = task_config["metric_list"][0]["metric"]
        # `!function`-defined metrics (e.g. cruxeval/humaneval_infilling's
        # pass_at_k) serialize to their full source instead of a bare name,
        # so a non-identifier value can never match an actual metric key --
        # there is no configured name to check against, so fall straight
        # through to the alphabetical fallback below.
        name = configured if re.fullmatch(r"[A-Za-z0-9_]+", configured) else None
    if name is not None:
        for metric in metrics:
            if metric == name or metric.split(",", 1)[0] == name:
                return metric
        raise ScoreboardError(f"primary metric {name!r} is not present in results")
    non_stderr = sorted(
        metric for metric in metrics if not metric.split(",", 1)[0].endswith("_stderr")
    )
    if non_stderr:
        return non_stderr[0]
    if metrics:
        return sorted(metrics)[0]
    raise ScoreboardError(f"primary metric {name!r} is not present in results")


def _sample(
    sample_index: int,
    task_name: str,
    row: dict[str, Any],
    metric_name: str,
    include_answer: bool,
) -> tuple[dict[str, Any], str, bool]:
    evidence = row.get("response_evidence") or []
    first_evidence = evidence[0] if evidence else {}
    mc_index = _loglikelihood_argmax(row["filtered_resps"])
    mc_choice_text = (
        _mc_choice_text(row.get("arguments"), mc_index) if mc_index is not None else None
    )
    mc_target_text = None
    if mc_index is not None:
        try:
            mc_target_text = _mc_choice_text(row.get("arguments"), int(row["target"]))
        except (TypeError, ValueError):
            mc_target_text = None
    raw_completion = _text(
        first_evidence.get("text") or mc_choice_text or row["resps"]
    )
    extracted_answer = (
        mc_choice_text if mc_choice_text is not None else _text(row["filtered_resps"])
    )
    score = row.get(metric_name)
    scalar_score = score if isinstance(score, (int, float)) and not isinstance(score, bool) else None
    outcome = (
        "unanswered"
        if not raw_completion.strip()
        else {1: "correct", 0: "incorrect"}.get(scalar_score, "undetermined")
    )
    stop_reason = str(first_evidence.get("finish_reason") or "unknown")
    truncated = first_evidence.get("truncated") is True or stop_reason in {
        "length",
        "max_tokens",
        "model_length",
    }
    detail = {
        "sample_index": sample_index,
        "document_index": int(row["doc_id"]),
        "document": {"task_name": task_name},
        "metrics": {name: row[name] for name in row["metrics"]},
        "model_response": {
            "text": raw_completion,
            "stop_reason": stop_reason,
            "tool_calls": first_evidence.get("tool_calls") or [],
        },
    }
    if include_answer:
        generated_tokens = sum(
            len(item.get("output_token_ids") or []) for item in evidence
        )
        detail["answer"] = {
            "outcome": outcome,
            "problem_id": str(row["doc_id"]),
            "repeat_id": int(row.get("repeat_id", 0)),
            "ground_truth": (
                mc_target_text if mc_target_text is not None else _text(row["target"])
            ),
            "extracted_answer": extracted_answer,
            "assembled_prompt": _prompt_from_arguments(row.get("arguments")),
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
            # no per-request timing instrumentation exists upstream yet
            "latency_ms": None,
        }
    return detail, outcome, truncated


def _samples(
    path: Path,
    task_name: str,
    metric_name: str,
    include_answer: bool,
    uploaded_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one leaf's per-sample details, capped at `MAX_SAMPLES_PER_OUTCOME`.

    `uploaded_counts` tracks how many samples of each outcome have already
    been uploaded for the *benchmark* this leaf belongs to. Callers building
    a multi-leaf benchmark publication must share one `uploaded_counts` dict
    across all `_samples` calls for that benchmark so the cap applies to the
    benchmark as a whole rather than being re-applied per leaf (which would
    let an N-leaf benchmark upload up to `N * MAX_SAMPLES_PER_OUTCOME`
    samples per outcome).
    """
    details: list[dict[str, Any]] = []
    outcome_counts = dict.fromkeys(
        ("correct", "incorrect", "unanswered", "undetermined"), 0
    )
    if uploaded_counts is None:
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
    results: dict[str, Any],
    task_config: dict[str, Any],
    model_args: dict[str, Any],
    task_metadata: dict[str, Any],
) -> dict[str, Any]:
    run_config = results["config"]
    effective = {
        **model_args,
        **(results.get("sampling_config") or {}),
        **(run_config.get("sampling_config") or {}),
        **(run_config.get("gen_kwargs") or {}),
        **(task_config.get("generation_kwargs") or {}),
        **task_metadata,
    }
    sampling = {
        target: effective[source]
        for source, target in SAMPLING_FIELDS.items()
        if source in effective
    }
    chat_template_kwargs: dict[str, Any] = {}
    for source in (
        model_args,
        results.get("sampling_config") or {},
        run_config.get("sampling_config") or {},
        run_config.get("gen_kwargs") or {},
        task_config.get("generation_kwargs") or {},
    ):
        chat_template_kwargs.update(source.get("chat_template_kwargs") or {})
    for source, target in CHAT_TEMPLATE_FIELDS.items():
        if source in effective:
            chat_template_kwargs[target] = effective[source]
    if "cot_mode" in effective:
        chat_template_kwargs["rwkv_generation_prompt"] = effective["cot_mode"]
    if chat_template_kwargs:
        sampling["chat_template_kwargs"] = chat_template_kwargs
    return sampling


def _build_leaf_publication(
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
    policy = _task_policy(config, task_name, benchmark)
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
        "field": _field(task_metadata, task_config, policy),
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
    details, diagnostics = _samples(samples_path, task_name, sample_metric, True)
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
        "sampling_config": _sampling_config(
            results, task_config, model_args, task_metadata
        ),
        "primary_metric": primary_metric,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "samples": details,
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


def _common(values: list[Any]) -> Any:
    """The shared value across `values`, or `None` if any two disagree."""
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _match_metric_key(metrics: dict[str, float], name: str | None) -> str | None:
    if name is None:
        return None
    for metric in metrics:
        if metric == name or metric.split(",", 1)[0] == name:
            return metric
    return None


def _merge_task_metadata(leaf_configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge per-leaf task metadata for a multi-leaf benchmark publication.

    Keeps only metadata keys with an identical value across every leaf
    (fields that genuinely differ per leaf, e.g. `field`, are left for the
    publication policy to disambiguate). `dataset` is recomputed as the
    value shared by every leaf (`None` if leaves disagree); `languages`/
    `tags`/`evaluation_splits` are recomputed as a deduplicated, sorted
    union instead of arbitrarily picking one leaf's value.
    """
    task_metadatas = [(cfg.get("metadata") or {}) for cfg in leaf_configs.values()]
    common_keys = (
        set.intersection(*(set(m) for m in task_metadatas)) if task_metadatas else set()
    )
    merged = {
        key: value
        for key in common_keys
        if (value := _common([m[key] for m in task_metadatas])) is not None
    }

    def _union(getter: Any) -> list[str]:
        values: set[str] = set()
        for cfg in leaf_configs.values():
            meta = cfg.get("metadata") or {}
            values.update(str(v) for v in (getter(cfg, meta) or []))
        return sorted(values)

    # `dataset` is a single string field on Task, unlike languages/tags/
    # evaluation_splits which are lists -- so it must use `_common` (None
    # when leaves disagree) rather than a deduplicated union.
    merged["dataset"] = _common(
        [
            (cfg.get("dataset_path") or (cfg.get("metadata") or {}).get("dataset"))
            or None
            for cfg in leaf_configs.values()
        ]
    )
    merged["languages"] = _union(lambda cfg, meta: meta.get("languages") or [])
    merged["tags"] = _union(
        lambda cfg, meta: meta.get("tags") or meta.get("upstream_tags") or []
    )
    merged["evaluation_splits"] = _union(
        lambda cfg, meta: meta.get("evaluation_splits")
        or (
            [cfg.get("test_split") or cfg.get("validation_split")]
            if (cfg.get("test_split") or cfg.get("validation_split"))
            else []
        )
    )
    return merged


def build_benchmark_publication(
    results: dict[str, Any],
    metadata: dict[str, Any],
    config: PublicationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one compact Scoreboard publication for an aggregated benchmark.

    `results` is the benchmark-level results.json written by
    `save_benchmark_artifacts` -- it never carries leaf sub-entries, only the
    aggregated score(s), plus pointers (`leaf_artifact_paths`) to each leaf's
    own artifacts so sample evidence can still be sourced without
    duplicating it on disk.
    """
    benchmark_name = next(iter(results["results"]))
    leaf_artifact_paths: dict[str, dict[str, str]] = (
        results.get("leaf_artifact_paths") or {}
    )

    if results.get("strategy") == "standalone":
        leaf_name = next(iter(leaf_artifact_paths))
        samples_path = Path(leaf_artifact_paths[leaf_name]["samples"])
        return _build_leaf_publication(
            results, metadata, samples_path, benchmark_name, config
        )

    leaf_docs = {
        leaf_name: json.loads(Path(paths["results"]).read_text("utf-8"))
        for leaf_name, paths in leaf_artifact_paths.items()
    }
    leaf_configs = {
        leaf_name: doc["configs"][leaf_name] for leaf_name, doc in leaf_docs.items()
    }
    task_metadata = _merge_task_metadata(leaf_configs)
    policy = _task_policy(config, benchmark_name, benchmark_name)

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
        or model_args.get("wkv_mode")
        or "fp32io16"
    )
    task_version = str(
        task_metadata.get("task_version")
        or _common(
            [doc["versions"][leaf_name] for leaf_name, doc in leaf_docs.items()]
        )
        or task_metadata.get("version")
        or "aggregate"
    )
    task = {
        "identity": f"{config.model_sha256}:{precision}:{benchmark_name}",
        "weight_sha256": config.model_sha256,
        "weight_display_name": model_name,
        "wkv_mode": precision,
        "field": _field(task_metadata, {}, policy),
        "benchmark": benchmark_name,
        "task_name": benchmark_name,
        "task_version": task_version,
        "dataset": task_metadata["dataset"],
        "subset": _common(
            [cfg.get("dataset_name") or None for cfg in leaf_configs.values()]
        ),
        "evaluation_splits": task_metadata["evaluation_splits"],
        "languages": task_metadata["languages"],
        "tags": task_metadata["tags"],
    }

    metrics = _metrics(results, benchmark_name)
    policy_metric_name = policy.get("primary_metric")
    primary_metric = _match_metric_key(metrics, policy_metric_name)
    if primary_metric is None and policy_metric_name is not None:
        raise ScoreboardError(
            f"primary metric {policy_metric_name!r} is not present in results"
        )
    if primary_metric is None:
        primary_metric = _match_metric_key(metrics, results.get("primary_metric_hint"))
    no_shared_metric = not metrics

    all_details: list[dict[str, Any]] = []
    outcome_counts = dict.fromkeys(
        ("correct", "incorrect", "unanswered", "undetermined"), 0
    )
    samples_total = uploaded_total = 0
    weighted_truncated = 0.0
    shared_uploaded_counts = dict.fromkeys(outcome_counts, 0)
    for leaf_name, paths in leaf_artifact_paths.items():
        leaf_doc = leaf_docs[leaf_name]
        leaf_metrics = _metrics(leaf_doc, leaf_name)
        leaf_policy = _task_policy(config, leaf_name, benchmark_name)
        leaf_primary = _primary_metric(
            leaf_metrics, leaf_configs[leaf_name], leaf_policy
        )
        if no_shared_metric:
            # Benchmarks like cruxeval/infinitebench mix leaves with no
            # metric name shared across all of them, so aggregate_benchmark_
            # fallback intentionally leaves the benchmark-level metrics
            # empty (see its docstring) rather than synthesize a misleading
            # headline score. The Scoreboard schema still requires at least
            # one non-null metric, so surface each leaf's own primary metric
            # namespaced by leaf name instead of a blended score.
            metrics[f"{leaf_name}/{leaf_primary}"] = leaf_metrics[leaf_primary]
        sample_metric = str(
            leaf_policy.get("outcome_metric") or leaf_primary.split(",", 1)[0]
        )
        details, diagnostics = _samples(
            Path(paths["samples"]),
            leaf_name,
            sample_metric,
            True,
            uploaded_counts=shared_uploaded_counts,
        )
        all_details.extend(details)
        for key in outcome_counts:
            outcome_counts[key] += diagnostics["outcome_counts"][key]
        samples_total += diagnostics["samples_total"]
        weighted_truncated += (
            diagnostics["truncation_rate"] * diagnostics["samples_total"]
        )
        uploaded_total += diagnostics["samples_uploaded"]
    if primary_metric is None and metrics:
        primary_metric = sorted(metrics)[0]
    for index, detail in enumerate(all_details):
        detail["sample_index"] = index
    diagnostics = {
        "samples_total": samples_total,
        "samples_uploaded": uploaded_total,
        "outcome_counts": outcome_counts,
        "truncation_rate": (
            weighted_truncated / samples_total if samples_total else 0.0
        ),
    }

    counts = results["n-samples"][benchmark_name]
    task_summary = {
        "task": benchmark_name,
        "dataset_path": task["dataset"],
        "dataset_name": task["subset"],
        "output_type": _common(
            [cfg.get("output_type") for cfg in leaf_configs.values()]
        ),
        "num_fewshot": _common(
            [cfg.get("num_fewshot") for cfg in leaf_configs.values()]
        ),
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
        "sampling_config": _sampling_config(results, {}, model_args, task_metadata),
        "primary_metric": primary_metric,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "samples": all_details,
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
        "configured_benchmarks": [benchmark_name],
        "resolved_benchmarks": [benchmark_name],
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
    if len(raw) > MAX_UNCOMPRESSED_BYTES:
        raise ScoreboardError(
            "Scoreboard publication exceeds the 64 MiB compressed or 256 MiB raw limit"
        )
    compressed = gzip.compress(raw)
    if len(compressed) > MAX_COMPRESSED_BYTES:
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
        self._preflight: dict[str, Any] | None = None

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
        raise AssertionError("unreachable: loop always returns or raises")

    def publish(
        self, campaign: dict[str, Any], publication: dict[str, Any]
    ) -> dict[str, Any]:
        if self._preflight is None:
            self._preflight = self._send(
                "GET",
                "/v1/evaluation-publication-preflight",
                "preflight:scoreboard-v1",
                timeout=self.config.control_timeout,
            )
        preflight = self._preflight
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
        self._publisher: ScoreboardPublisher | None = None

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
            campaign, publication = build_benchmark_publication(
                results, metadata, self.config
            )
            status["diagnostics"] = publication["diagnostics"]
            if self.config.dry_run:
                status["publication"] = "validated"
                return status
            if self._publisher is None:
                self._publisher = ScoreboardPublisher(self.config)
            receipt = self._publisher.publish(campaign, publication)
            status.update(
                publication="complete",
                uploaded=True,
                campaign_id=receipt["campaign_id"],
                receipt=receipt,
            )
        except (KeyError, OSError, ScoreboardError, TypeError, ValueError) as error:
            status["error"] = f"{type(error).__name__}: {error}"
        return status
