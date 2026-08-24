#!/usr/bin/env python3
"""Convert evaluator artifacts into strict scoreboard-rwkv DTO payloads.

This module performs loss-intolerant producer and lm-eval conversion.  It does
not contact the scoreboard; network publication belongs to
``scripts/upload_scoreboard.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO


CAMPAIGN_SCHEMA = "scoreboard-v1"
TASK_SCHEMA = "scoreboard-v1"
LM_EVAL_CAMPAIGN_SCHEMA = CAMPAIGN_SCHEMA
LM_EVAL_TASK_SCHEMA = TASK_SCHEMA
LIGHTEVAL_VERSION = "0.13.0"
PRODUCER_CAMPAIGN_SCHEMA = "rwkv-producer-campaign-v1"
PRODUCER_TASK_SCHEMA = "rwkv-producer-task-v1"

# This is the publication contract enforced by scoreboard-rwkv's current
# LightEval DTO.  The producer keeps its richer RWKV-specific evidence; this
# table only supplies stable benchmark dimensions that the DTO requires.
_BENCHMARK_CONTRACT: dict[str, dict[str, Any]] = {
    "moral_stories": {
        "selector": "moral_stories",
        "task_version": "1.0",
        "module_family": "moral_stories",
        "module": "lm_eval.tasks.moral_stories",
        "dataset": "demelin/moral_stories",
        "subset": "full",
        "evaluation_splits": ["train"],
        "languages": ["english"],
        "upstream_tags": ["moral-reasoning"],
        "primary_metric": "acc",
        "generation_size": 256,
    },
    "haerae": {
        "selector": "haerae",
        "task_version": "1.0",
        "module_family": "haerae",
        "module": "lm_eval.tasks.haerae",
        "dataset": "HAERAE-HUB/HAE_RAE_BENCH",
        "subset": "all",
        "evaluation_splits": ["test"],
        "languages": ["korean"],
        "upstream_tags": ["knowledge"],
        "primary_metric": "acc",
        "generation_size": 256,
    },
    "jsonschema_bench": {
        "selector": "jsonschema_bench",
        "task_version": "1.0",
        "module_family": "jsonschema_bench",
        "module": "lm_eval.tasks.jsonschema_bench",
        "dataset": "epfl-dlab/JSONSchemaBench",
        "subset": "Github_easy|Github_medium|Github_hard",
        "evaluation_splits": ["test"],
        "languages": ["english"],
        "upstream_tags": ["structured-generation"],
        "primary_metric": "schema_compliance",
        "generation_size": 2048,
    },
    "gsm8k_platinum": {
        "selector": "gsm8k_platinum",
        "task_version": "1.0",
        "module_family": "gsm8k_platinum",
        "module": "lm_eval.tasks.gsm8k_platinum",
        "dataset": "madrylab/gsm8k-platinum",
        "subset": "main",
        "evaluation_splits": ["test"],
        "languages": ["english"],
        "upstream_tags": ["math", "chain-of-thought"],
        "primary_metric": "exact_match",
        "generation_size": 512,
    },
    "aexams": {
        "selector": "aexams",
        "task_version": "1.0",
        "module_family": "aexams",
        "module": "lm_eval.tasks.aexams",
        "dataset": "OALL/Arabic_EXAMS",
        "subset": "all",
        "evaluation_splits": ["test"],
        "languages": ["arabic"],
        "upstream_tags": ["multiple-choice", "arabic"],
        "primary_metric": "acc",
        "generation_size": 1,
    },
}

_TARGET_SAMPLING: dict[str, Any] = {
    "temperature": 0.96,
    "top_p": 0.76,
    "top_k": 32,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.1,
    "repetition_penalty": 1.0,
    "penalty_decay": 0.988,
    "max_new_tokens": 8192,
    "ignore_eos": False,
}


class ScoreboardError(RuntimeError):
    """A strict local conversion error."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ScoreboardError(
            f"cannot read strict JSON object {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ScoreboardError(f"JSON document must be an object: {path}")
    return value


def canonical_json(value: Any) -> bytes:
    """Return the canonical JSON bytes used by scoreboard-rwkv digests."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ScoreboardError(
            f"payload cannot be canonicalized as JSON: {error}"
        ) from error


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def campaign_run_key(campaign: dict[str, Any]) -> str:
    payload = deepcopy(campaign)
    payload.pop("run_key", None)
    payload.setdefault("rerun_reason", None)
    for field in (
        "configured_benchmarks",
        "resolved_benchmarks",
        "skipped_benchmarks",
    ):
        values = payload.get(field)
        if isinstance(values, list):
            payload[field] = sorted(values)
    expected = payload.get("expected_tasks")
    if isinstance(expected, list):
        normalized: list[dict[str, Any]] = []
        for task in expected:
            value = deepcopy(task)
            for field in ("evaluation_splits", "languages", "tags"):
                values = value.get(field)
                if isinstance(values, list):
                    value[field] = sorted(values)
            normalized.append(value)
        payload["expected_tasks"] = sorted(
            normalized,
            key=lambda value: (
                str(value.get("identity", "")),
                canonical_json(value),
            ),
        )
    return content_digest(payload)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _producer_provenance(
    campaign: dict[str, Any], task: dict[str, Any] | None = None
) -> dict[str, Any]:
    candidates = [
        task.get("provenance") if isinstance(task, dict) else None,
        campaign.get("provenance"),
    ]
    for value in candidates:
        if isinstance(value, dict):
            return value
    raise ScoreboardError("producer publication lacks campaign provenance")


def _producer_weight_sha256(
    campaign: dict[str, Any], provenance: dict[str, Any]
) -> str:
    value = campaign.get("weight_sha256") or provenance.get("weight_sha256")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ScoreboardError(
            "producer publication lacks a valid weight_sha256; refusing to invent model identity"
        )
    return value


def _producer_model_display_name(
    campaign: dict[str, Any], provenance: dict[str, Any]
) -> str:
    value = campaign.get("model_name") or provenance.get("model_name")
    if not isinstance(value, str) or not value.strip():
        raise ScoreboardError("producer publication lacks model_name")
    weight_path = provenance.get("weight_path")
    if isinstance(weight_path, str) and weight_path:
        filename = Path(weight_path).name
        if filename:
            return filename
    return value.strip()


def _producer_benchmark(expected_task: dict[str, Any], payload: dict[str, Any]) -> str:
    task = payload.get("task")
    if isinstance(task, dict) and isinstance(task.get("task_name"), str):
        benchmark = task["task_name"]
    elif isinstance(expected_task.get("task"), str):
        benchmark = expected_task["task"]
    else:
        benchmark = ""
    if benchmark not in _BENCHMARK_CONTRACT:
        raise ScoreboardError(
            f"producer task {benchmark!r} has no scoreboard metadata mapping"
        )
    return benchmark


def _producer_expected_tasks(
    campaign: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    provenance = _producer_provenance(campaign)
    weight_sha256 = _producer_weight_sha256(campaign, provenance)
    weight_display_name = _producer_model_display_name(campaign, provenance)
    raw_expected = campaign.get("expected_tasks")
    if not isinstance(raw_expected, list) or not raw_expected:
        raise ScoreboardError("producer campaign.expected_tasks must be non-empty")
    expected: list[dict[str, Any]] = []
    producer_to_scoreboard: dict[str, str] = {}
    for index, raw in enumerate(raw_expected):
        if not isinstance(raw, dict):
            raise ScoreboardError(f"producer expected_tasks[{index}] is not an object")
        mode = raw.get("wkv_mode")
        if mode not in {"fp16", "fp32io16"}:
            raise ScoreboardError(
                f"producer expected_tasks[{index}] has invalid wkv_mode"
            )
        benchmark = raw.get("task")
        if not isinstance(benchmark, str) or benchmark not in _BENCHMARK_CONTRACT:
            raise ScoreboardError(
                f"producer expected_tasks[{index}] has unsupported task {benchmark!r}"
            )
        producer_identity = raw.get("identity")
        if not isinstance(producer_identity, str) or not producer_identity:
            raise ScoreboardError(f"producer expected_tasks[{index}] lacks identity")
        contract = _BENCHMARK_CONTRACT[benchmark]
        identity = f"{weight_sha256}:{mode}:{benchmark}"
        task_descriptor = {
            "identity": identity,
            "weight_sha256": weight_sha256,
            "weight_display_name": weight_display_name,
            "wkv_mode": mode,
            "benchmark": benchmark,
            "task_name": benchmark,
            "task_version": contract["task_version"],
            "dataset": contract["dataset"],
            "subset": contract["subset"],
            "evaluation_splits": deepcopy(contract["evaluation_splits"]),
            "languages": deepcopy(contract["languages"]),
            "tags": deepcopy(contract["upstream_tags"]),
        }
        expected.append(task_descriptor)
        if producer_identity in producer_to_scoreboard:
            raise ScoreboardError(
                f"duplicate producer task identity {producer_identity}"
            )
        producer_to_scoreboard[producer_identity] = identity

    # The server requires the same task set in both WKV modes.  Let the
    # regular campaign validator enforce the exact DTO invariant after the
    # conversion, but provide a useful producer-side error first.
    modes_by_task: dict[str, set[str]] = {}
    for item in expected:
        modes_by_task.setdefault(item["task_name"], set()).add(item["wkv_mode"])
    missing = [
        task for task, modes in modes_by_task.items() if modes != {"fp16", "fp32io16"}
    ]
    if missing:
        raise ScoreboardError(
            "producer campaign is not publishable: every task needs both fp16 and fp32io16; "
            + ", ".join(sorted(missing))
        )
    return expected, producer_to_scoreboard


def _numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not isinstance(value, dict):
        return metrics
    for key, item in value.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if _finite_number(item):
            metrics[name] = float(item)
        elif isinstance(item, dict):
            metrics.update(_numeric_metrics(item, name))
    return metrics


def _producer_aggregates(task: dict[str, Any], benchmark: str) -> dict[str, float]:
    results = task.get("results")
    if not isinstance(results, dict):
        raise ScoreboardError(f"producer task {benchmark} lacks results")
    candidates = [results.get("metrics"), results.get("raw_results")]
    aggregate: dict[str, float] = {}
    for candidate in candidates:
        aggregate.update(_numeric_metrics(candidate))
    if not aggregate:
        samples = task.get("samples")
        if isinstance(samples, list):
            names = {
                name
                for sample in samples
                if isinstance(sample, dict)
                for name in sample.get("metrics", [])
                if isinstance(name, str)
            }
            for name in names:
                values = [
                    sample.get(name)
                    for sample in samples
                    if isinstance(sample, dict) and _finite_number(sample.get(name))
                ]
                if values:
                    aggregate[name] = sum(float(item) for item in values) / len(values)
    if not aggregate:
        raise ScoreboardError(
            f"producer task {benchmark} has no finite aggregate metrics; refusing to publish a score"
        )
    # Nested raw_results commonly contains a task-name prefix.  The target
    # DTO uses native metric names, so strip only an unambiguous prefix.
    names = set(aggregate)
    if len(names) > 1:
        stripped: dict[str, float] = {}
        for name, value in aggregate.items():
            metric_name = name.rsplit(".", 1)[-1]
            metric_name = metric_name.split(",", 1)[0]
            stripped[metric_name] = value
        if len(stripped) == len(aggregate):
            aggregate = stripped
    else:
        only = next(iter(aggregate), "")
        normalized = only.rsplit(".", 1)[-1].split(",", 1)[0]
        if normalized and normalized != only:
            aggregate[normalized] = aggregate.pop(only)
    return aggregate


def _raw_choice(evidence: dict[str, Any], ordinal: int) -> dict[str, Any] | None:
    raw = evidence.get("raw_response")
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    candidate = choices[ordinal] if ordinal < len(choices) else choices[0]
    return candidate if isinstance(candidate, dict) else None


def _int_tokens(value: Any, *, context: str, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ScoreboardError(f"{context} must contain integer token IDs")
    if not allow_empty and not value:
        raise ScoreboardError(f"{context} must not be empty")
    return list(value)


def _loglikelihood_response(
    evidence: dict[str, Any], choice: dict[str, Any] | None, *, context: str
) -> tuple[list[float], list[int]]:
    if choice is None or not isinstance(choice.get("logprobs"), dict):
        raise ScoreboardError(
            f"{context} lacks raw log-likelihood evidence; rerun with record_evidence=true"
        )
    raw_values = choice["logprobs"].get("token_logprobs")
    if not isinstance(raw_values, list):
        raise ScoreboardError(f"{context} raw logprobs are not an array")
    values = [float(item) for item in raw_values if _finite_number(item)]
    tokens = _int_tokens(
        evidence.get("output_token_ids"),
        context=f"{context}.output_token_ids",
        allow_empty=False,
    )
    if len(values) < len(tokens):
        raise ScoreboardError(
            f"{context} has fewer logprobs than output tokens; refusing lossy conversion"
        )
    values = values[-len(tokens) :]
    return values, tokens[-len(values) :]


def _model_response(sample: dict[str, Any], *, context: str) -> dict[str, Any]:
    evidence_value = sample.get("response_evidence")
    if not isinstance(evidence_value, list) or not evidence_value:
        raise ScoreboardError(
            f"{context} lacks response_evidence; rerun with record_evidence=true"
        )
    evidence = [item for item in evidence_value if isinstance(item, dict)]
    if len(evidence) != len(evidence_value):
        raise ScoreboardError(f"{context}.response_evidence contains a non-object")

    choices = [_raw_choice(item, index) for index, item in enumerate(evidence)]
    has_loglikelihood = any(
        choice is not None and isinstance(choice.get("logprobs"), dict)
        for choice in choices
    )
    prompts = [
        item.get("prompt") for item in evidence if isinstance(item.get("prompt"), str)
    ]
    response: dict[str, Any] = {
        "input": prompts[0] if prompts else None,
        "input_tokens": evidence[0].get("input_token_ids"),
    }
    if has_loglikelihood:
        if not all(
            choice is not None and isinstance(choice.get("logprobs"), dict)
            for choice in choices
        ):
            raise ScoreboardError(
                f"{context} mixes generation and log-likelihood evidence"
            )
        logprobs: list[float] = []
        output_tokens: list[list[int]] = []
        for index, (item, choice) in enumerate(zip(evidence, choices, strict=True)):
            values, tokens = _loglikelihood_response(
                item, choice, context=f"{context}.evidence[{index}]"
            )
            logprobs.append(values[-1] if len(values) == 1 else sum(values))
            output_tokens.append(tokens)
        response["logprobs"] = logprobs
        response["output_tokens"] = output_tokens
        return response

    texts: list[str] = []
    output_tokens: list[list[int]] = []
    post_processed: list[str] = []
    reasonings: list[str | None] = []
    for index, (item, choice) in enumerate(zip(evidence, choices, strict=True)):
        text = choice.get("text") if isinstance(choice, dict) else None
        if not isinstance(text, str):
            raise ScoreboardError(
                f"{context}.evidence[{index}] lacks raw completion text; refusing to use a filtered answer as model output"
            )
        texts.append(text)
        output_tokens.append(
            _int_tokens(
                item.get("output_token_ids"),
                context=f"{context}.evidence[{index}].output_token_ids",
            )
        )
        value = item.get("post_processed_answer")
        if not isinstance(value, str):
            raise ScoreboardError(
                f"{context}.evidence[{index}] lacks post_processed_answer"
            )
        post_processed.append(value)
        reasoning = item.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ScoreboardError(
                f"{context}.evidence[{index}].reasoning is not a string"
            )
        reasonings.append(reasoning)
    response.update(
        {
            "text": texts,
            "text_post_processed": post_processed,
            "output_tokens": output_tokens,
        }
    )
    if any(value is not None for value in reasonings):
        response["reasonings"] = reasonings
    return response


def _producer_task_config(
    task: dict[str, Any], benchmark: str, sample_count: int
) -> dict[str, Any]:
    value = task.get("task_config")
    if not isinstance(value, dict):
        results = task.get("results")
        value = results.get("task_config") if isinstance(results, dict) else None
    if not isinstance(value, dict):
        raise ScoreboardError(
            f"producer task {benchmark} lacks measured task_config accounting"
        )
    required = (
        "generation_size",
        "original_num_docs",
        "effective_num_docs",
        "skipped_multiselect_docs",
    )
    if any(key not in value for key in required):
        raise ScoreboardError(
            f"producer task {benchmark} task_config lacks measured document accounting"
        )
    if value["effective_num_docs"] != sample_count:
        raise ScoreboardError(
            f"producer task {benchmark} sample count does not match task_config.effective_num_docs"
        )
    return {key: value[key] for key in required} | {
        "producer_schema_version": PRODUCER_TASK_SCHEMA,
        "producer_task_config": deepcopy(value),
    }


def _producer_sampling(
    campaign: dict[str, Any], task: dict[str, Any], *, prompt_template: str
) -> dict[str, Any]:
    provenance = _producer_provenance(campaign, task)
    prompt = provenance.get("prompt")
    generation_prompt = (
        prompt.get("generation_prompt") if isinstance(prompt, dict) else None
    )
    if (
        campaign.get("scoreboard_compatible") is not True
        or provenance.get("scoreboard_compatible") is not True
        or generation_prompt != "open_think"
    ):
        raise ScoreboardError(
            "producer run is not marked scoreboard_compatible with open_think; rerun with --scoreboard-compatible"
        )
    sampling = provenance.get("sampling_config")
    if not isinstance(sampling, dict):
        raise ScoreboardError("producer task lacks sampling_config provenance")
    sampling = deepcopy(sampling)
    stops = {"bot": "✿", "assistant": "\nUser:", "function_calling": "\n### User"}
    expected_stop = [stops[prompt_template]]
    if sampling.get("stop") != expected_stop:
        raise ScoreboardError(
            "producer sampling_config.stop does not match prompt template"
        )
    for key, expected in _TARGET_SAMPLING.items():
        if sampling.get(key) != expected:
            raise ScoreboardError(
                f"producer sampling_config.{key}={sampling.get(key)!r} does not match scoreboard contract {expected!r}"
            )
    return sampling


def _producer_gpu(campaign: dict[str, Any], task: dict[str, Any]) -> str:
    provenance = _producer_provenance(campaign, task)
    values = provenance.get("gpu")
    if not isinstance(values, list) or not values:
        runtime = provenance.get("runtime")
        values = runtime.get("gpu") if isinstance(runtime, dict) else None
    if not isinstance(values, list) or not values:
        raise ScoreboardError("producer provenance lacks GPU execution identity")
    names = [
        item.get("name")
        for item in values
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]
    ]
    if not names:
        raise ScoreboardError("producer provenance lacks GPU names")
    return ", ".join(dict.fromkeys(names))


def _producer_runtime_int(
    campaign: dict[str, Any], task: dict[str, Any], key: str
) -> int:
    provenance = _producer_provenance(campaign, task)
    runtime = provenance.get("runtime")
    value = runtime.get(key) if isinstance(runtime, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScoreboardError(f"producer runtime lacks positive integer {key}")
    return value


def _producer_dependency_versions(
    campaign: dict[str, Any], task: dict[str, Any]
) -> dict[str, str]:
    provenance = _producer_provenance(campaign, task)
    dependencies = provenance.get("dependencies")
    packages = dependencies.get("packages") if isinstance(dependencies, dict) else None
    if not isinstance(packages, dict):
        raise ScoreboardError("producer provenance lacks dependency versions")
    torch = packages.get("torch")
    if not isinstance(torch, str) or not torch:
        raise ScoreboardError("producer provenance lacks torch version")
    lighteval = packages.get("lighteval")
    if not isinstance(lighteval, str) or lighteval != LIGHTEVAL_VERSION:
        raise ScoreboardError(
            "producer provenance does not contain the required LightEval 0.13.0 dependency; "
            "this lm-eval run cannot be presented as a LightEval execution"
        )
    backend_commit = provenance.get("backend_commit")
    if not isinstance(backend_commit, str) or not backend_commit:
        raise ScoreboardError("producer provenance lacks backend commit")
    return {
        "lighteval": lighteval,
        "vllm": f"vllm-rwkv@{backend_commit}",
        "torch": torch,
    }


def _producer_diagnostics(
    details: list[dict[str, Any]], prompt_template: str
) -> dict[str, Any]:
    stops = {"bot": "✿", "assistant": "\nUser:", "function_calling": "\n### User"}
    stop = stops[prompt_template]
    completions = 0
    truncated = 0
    violations = 0
    for detail in details:
        response = detail["model_response"]
        texts = response.get("text")
        output_tokens = response.get("output_tokens")
        if texts in (None, []):
            continue
        if not isinstance(texts, list) or not isinstance(output_tokens, list):
            raise ScoreboardError("converted completion evidence is malformed")
        if len(texts) != len(output_tokens):
            raise ScoreboardError("converted completion/token evidence counts differ")
        for text, tokens in zip(texts, output_tokens, strict=True):
            completions += 1
            truncated += int(len(tokens) >= _TARGET_SAMPLING["max_new_tokens"])
            violations += int(stop in text)
    return {
        "samples": len(details),
        "completions": completions,
        "truncated": truncated,
        "non_truncated": completions - truncated,
        "truncation_rate": truncated / completions if completions else 0.0,
        "turn_boundary_violations": violations,
        "turn_boundary_violation_rate": violations / completions
        if completions
        else 0.0,
    }


def convert_producer_publication(
    campaign: dict[str, Any], payloads: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert the RWKV producer contract into the current Scoreboard DTO.

    The conversion is deliberately loss-intolerant.  It only maps evidence
    already present in the producer payload and rejects fake-think runs,
    missing token/response evidence, or missing model identity instead of
    fabricating values that would look publishable.
    """

    if campaign.get("schema_version") != PRODUCER_CAMPAIGN_SCHEMA:
        raise ScoreboardError("not a rwkv producer campaign")
    expected, identity_map = _producer_expected_tasks(campaign)
    by_producer_identity = {
        payload.get("task", {}).get("identity"): payload
        for payload in payloads
        if isinstance(payload.get("task"), dict)
    }
    if set(by_producer_identity) != set(identity_map):
        missing = sorted(set(identity_map) - set(by_producer_identity))
        extra = sorted(set(by_producer_identity) - set(identity_map))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise ScoreboardError(
            "producer task set does not match campaign: " + "; ".join(details)
        )

    provenance = _producer_provenance(campaign)
    prompt = provenance.get("prompt")
    prompt_template = prompt.get("template") if isinstance(prompt, dict) else None
    if prompt_template not in {"bot", "assistant", "function_calling"}:
        raise ScoreboardError("producer provenance lacks a supported prompt template")
    weight_sha256 = _producer_weight_sha256(campaign, provenance)
    model_display_name = _producer_model_display_name(campaign, provenance)
    converted_campaign = {
        "schema_version": CAMPAIGN_SCHEMA,
        "source": "lm-eval-harness",
        "config_sha256": campaign.get("config_digest"),
        "registry_sha256": campaign.get("registry_digest"),
        "contract_sha256": campaign.get("eval_contract_digest"),
        "configured_benchmarks": deepcopy(campaign.get("configured_selectors")),
        "resolved_benchmarks": deepcopy(campaign.get("resolved_selectors")),
        "skipped_benchmarks": deepcopy(campaign.get("skipped_selectors")),
        "expected_tasks": expected,
        "rerun_reason": None,
    }
    converted_campaign["run_key"] = campaign_run_key(converted_campaign)

    converted_tasks: list[dict[str, Any]] = []
    expected_by_identity = {item["identity"]: item for item in expected}
    for producer_identity, scoreboard_identity in identity_map.items():
        producer_payload = by_producer_identity[producer_identity]
        producer_task = producer_payload.get("task")
        if not isinstance(producer_task, dict):
            raise ScoreboardError(
                f"producer task {producer_identity} lacks task object"
            )
        benchmark = _producer_benchmark(
            next(
                item
                for item in campaign["expected_tasks"]
                if item.get("identity") == producer_identity
            ),
            producer_payload,
        )
        descriptor = expected_by_identity[scoreboard_identity]
        samples = producer_task.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ScoreboardError(
                f"producer task {benchmark} has no complete samples; refusing publication"
            )
        aggregates = _producer_aggregates(producer_task, benchmark)
        primary_metric = _BENCHMARK_CONTRACT[benchmark]["primary_metric"]
        if primary_metric not in aggregates:
            primary_metric = next(
                (name for name in aggregates if not name.endswith("_stderr")),
                next(iter(aggregates)),
            )
        details: list[dict[str, Any]] = []
        for document_index, sample in enumerate(samples):
            if not isinstance(sample, dict) or not isinstance(sample.get("doc"), dict):
                raise ScoreboardError(
                    f"producer task {benchmark} has an invalid sample document"
                )
            doc = deepcopy(sample["doc"])
            doc["task_name"] = benchmark
            specific = doc.get("specific")
            if not isinstance(specific, dict):
                specific = {}
            specific["helicopter_document_index"] = document_index
            doc["specific"] = specific
            metric = {
                name: sample[name]
                for name in sample.get("metrics", [])
                if isinstance(name, str) and _finite_number(sample.get(name))
            }
            details.append(
                {
                    "sample_index": document_index,
                    "document_index": document_index,
                    "doc": doc,
                    "metric": metric,
                    "model_response": _model_response(
                        sample, context=f"{benchmark}.sample[{document_index}]"
                    ),
                }
            )
        task_provenance = producer_task.get("provenance")
        task_for_sampling = producer_task if isinstance(task_provenance, dict) else {}
        converted_tasks.append(
            {
                "schema_version": TASK_SCHEMA,
                # Replaced with the server UUID immediately before PUT.  A
                # non-null placeholder keeps the DTO shape explicit without
                # pretending to know the remote campaign id locally.
                "campaign_id": "assigned-by-uploader",
                "task": descriptor,
                "result_files": [
                    {
                        "role": "metrics",
                        "path": f"results/{model_display_name}/{benchmark}/{descriptor['wkv_mode']}.json",
                    },
                    {
                        "role": "samples",
                        "path": f"details/{model_display_name}/{benchmark}/{descriptor['wkv_mode']}.jsonl",
                    },
                ],
                "task_config": _producer_task_config(
                    producer_task, benchmark, len(details)
                ),
                "environment": {
                    "weight_sha256": weight_sha256,
                    "weight_display_name": model_display_name,
                    "wkv_mode": descriptor["wkv_mode"],
                    "prompt_template": prompt_template,
                    "gemm_policy": (
                        "fp16-accumulation"
                        if descriptor["wkv_mode"] == "fp16"
                        else "fp32-accumulation"
                    ),
                    "gpu": _producer_gpu(campaign, producer_task),
                    "max_num_seqs": _producer_runtime_int(
                        campaign, producer_task, "rwkv_max_num_seqs"
                    ),
                    "max_num_batched_tokens": _producer_runtime_int(
                        campaign, producer_task, "max_num_batched_tokens"
                    ),
                    "dependency_versions": _producer_dependency_versions(
                        campaign, producer_task
                    ),
                    "source": "lm-eval-harness",
                },
                "sampling_config": _producer_sampling(
                    campaign, task_for_sampling, prompt_template=prompt_template
                ),
                "primary_metric": primary_metric,
                "metrics": aggregates,
                "diagnostics": _producer_diagnostics(details, prompt_template),
                "samples": [
                    {
                        "sample_index": detail["sample_index"],
                        "document_index": detail["document_index"],
                        "document": detail["doc"],
                        "metrics": detail["metric"],
                        "model_response": detail["model_response"],
                    }
                    for detail in details
                ],
            }
        )
    return converted_campaign, converted_tasks


def _json_safe(value: Any) -> Any:
    """Convert lm-eval's typed response objects to strict JSON values.

    The evaluator keeps backend evidence next to the normal lm-eval sample
    fields.  ``default=str`` is intentionally only a last-resort encoding for
    third-party task objects; it never replaces the raw evidence field when it
    is already JSON-native.
    """

    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
        )
    except (TypeError, ValueError) as error:
        raise ScoreboardError(
            f"evaluation result is not JSON serializable: {error}"
        ) from error


def _artifact_root(output_dir: Path, model_name: str) -> Path:
    model_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_") or "model"
    root = output_dir.parent if output_dir.suffix == ".json" else output_dir
    if root.name != model_dir:
        root = root / model_dir
    return root / "publication"


def _task_metric_values(results: dict[str, Any], task_name: str) -> dict[str, float]:
    task_results = results.get("results", {})
    if not isinstance(task_results, dict):
        return {}
    value = task_results.get(task_name)
    if not isinstance(value, dict):
        return {}
    numeric = _numeric_metrics(value)
    normalized: dict[str, float] = {}
    for key, item in numeric.items():
        metric_name = key.rsplit(".", 1)[-1].split(",", 1)[0]
        normalized.setdefault(metric_name, item)
    return normalized


def _flatten_response_evidence(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ScoreboardError(f"{context} must be a non-empty evidence array")
    flattened: list[dict[str, Any]] = []
    for request_index, request_evidence in enumerate(value):
        if isinstance(request_evidence, dict):
            request_items = [request_evidence]
        elif isinstance(request_evidence, list):
            request_items = request_evidence
        else:
            raise ScoreboardError(
                f"{context}[{request_index}] must be an object or array"
            )
        if not request_items:
            raise ScoreboardError(f"{context}[{request_index}] must not be empty")
        for response_index, item in enumerate(request_items):
            if not isinstance(item, dict):
                raise ScoreboardError(
                    f"{context}[{request_index}][{response_index}] must be an object"
                )
            flattened.append({"request_index": request_index, **_json_safe(item)})
    if not flattened:
        raise ScoreboardError(f"{context} contains no evidence items")
    return flattened


def _token_ids(value: Any, *, context: str, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ScoreboardError(f"{context} must contain integer token IDs")
    if not allow_empty and not value:
        raise ScoreboardError(f"{context} must not be empty")
    return list(value)


def _lm_eval_response(sample: dict[str, Any], *, context: str) -> dict[str, Any]:
    evidence_items = _flatten_response_evidence(
        sample.get("response_evidence"), context=f"{context}.response_evidence"
    )

    def evidence_has_output(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        raw = item.get("raw_response")
        if not isinstance(raw, dict):
            return False
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        choice = choices[0]
        if not isinstance(choice, dict):
            return False
        if isinstance(choice.get("text"), str):
            return isinstance(item.get("post_processed_answer"), str)
        logprobs = choice.get("logprobs")
        return isinstance(logprobs, dict) and isinstance(
            logprobs.get("token_logprobs"), list
        )

    def token_ids_are_valid(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in value
        )

    response: dict[str, Any] = {
        "raw_resps": _json_safe(sample.get("resps", [])),
        "filtered_resps": _json_safe(sample.get("filtered_resps", [])),
        "arguments": _json_safe(sample.get("arguments", [])),
        "evidence": _json_safe(evidence_items),
    }
    prompts = [item.get("prompt") for item in evidence_items]
    response["input"] = next((item for item in prompts if isinstance(item, str)), None)
    response["input_tokens"] = next(
        (
            _token_ids(item.get("input_token_ids"), context=f"{context}.input_tokens")
            for item in evidence_items
            if isinstance(item.get("input_token_ids"), list)
        ),
        [],
    )
    texts: list[str] = []
    output_tokens: list[list[int]] = []
    answers: list[str] = []
    logprobs: list[float] = []
    for index, item in enumerate(evidence_items):
        raw = item.get("raw_response")
        choices = raw.get("choices") if isinstance(raw, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        text = choice.get("text") if isinstance(choice, dict) else None
        if isinstance(text, str):
            texts.append(text)
        choice_logprobs = choice.get("logprobs") if isinstance(choice, dict) else None
        token_logprobs = (
            choice_logprobs.get("token_logprobs")
            if isinstance(choice_logprobs, dict)
            else None
        )
        if isinstance(token_logprobs, list):
            finite = [item for item in token_logprobs if _finite_number(item)]
            if finite:
                logprobs.append(sum(float(item) for item in finite))
        if isinstance(item.get("output_token_ids"), list):
            output_tokens.append(
                _token_ids(
                    item["output_token_ids"],
                    context=f"{context}.evidence[{index}].output_token_ids",
                )
            )
        answer = item.get("post_processed_answer")
        if isinstance(answer, str):
            answers.append(answer)
    if texts:
        response["text"] = texts
        response["text_post_processed"] = answers
    if output_tokens:
        response["output_tokens"] = output_tokens
    if logprobs:
        response["logprobs"] = logprobs
    response["evidence_complete"] = all(
        isinstance(item.get("input_token_ids"), list)
        and isinstance(item.get("output_token_ids"), list)
        and _token_ids(item["input_token_ids"], context=f"{context}.input_token_ids")
        is not None
        and _token_ids(item["output_token_ids"], context=f"{context}.output_token_ids")
        is not None
        and evidence_has_output(item)
        for item in evidence_items
    )
    return response


def _native_task_metadata(
    publication: dict[str, Any], task_name: str, native_config: dict[str, Any]
) -> dict[str, Any]:
    task_metadata = publication.get("task_metadata", {})
    value = task_metadata.get(task_name, {}) if isinstance(task_metadata, dict) else {}
    if not isinstance(value, dict):
        value = {}
    metadata = native_config.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    selector = metadata.get("benchmark_name")
    selector_value = (
        task_metadata.get(selector, {})
        if isinstance(selector, str) and isinstance(task_metadata, dict)
        else {}
    )
    if not isinstance(selector_value, dict):
        selector_value = {}
    return {**metadata, **selector_value, **value}


def _native_wkv_mode(
    publication: dict[str, Any], task_name: str, native_config: dict[str, Any]
) -> str:
    metadata = _native_task_metadata(publication, task_name, native_config)
    model_args = native_config.get("model_args")
    candidates = [
        metadata.get("wkv_mode"),
        native_config.get("wkv_mode"),
        model_args.get("wkv_mode") if isinstance(model_args, dict) else None,
    ]
    mode = next((value for value in candidates if isinstance(value, str)), None)
    if mode not in {"fp16", "fp32io16"}:
        raise ScoreboardError(
            f"lm-eval task {task_name} lacks a recorded fp16/fp32io16 WKV mode"
        )
    return mode


def _native_gpu(results: dict[str, Any], config: dict[str, Any]) -> str | None:
    for value in (
        results.get("gpu"),
        config.get("gpu"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    environment = results.get("pretty_env_info")
    if isinstance(environment, str):
        matches = re.findall(r"GPU \d+: ([^\n]+)", environment)
        if matches:
            return ", ".join(dict.fromkeys(item.strip() for item in matches))
    return None


def _native_execution(
    results: dict[str, Any], config: dict[str, Any], *, wkv_mode: str, enabled: bool
) -> dict[str, Any]:
    model_args = config.get("model_args")
    model_args = model_args if isinstance(model_args, dict) else {}
    execution = results.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    backend_revision = (
        execution.get("backend_revision")
        or results.get("backend_commit")
        or config.get("backend_commit")
        or model_args.get("backend_commit")
    )
    backend_version = (
        execution.get("backend_version")
        or results.get("backend_version")
        or config.get("backend_version")
        or model_args.get("backend_version")
    )
    gpu = execution.get("gpu") or _native_gpu(results, config)
    max_num_seqs = execution.get("max_num_seqs") or model_args.get("num_concurrent")
    max_num_batched_tokens = (
        execution.get("max_num_batched_tokens")
        or model_args.get("max_num_batched_tokens")
        or model_args.get("max_length")
    )
    torch_version = results.get("torch_version")
    if not isinstance(torch_version, str):
        environment = results.get("pretty_env_info")
        match = (
            re.search(r"^PyTorch version: ([^\n]+)", environment, re.MULTILINE)
            if isinstance(environment, str)
            else None
        )
        torch_version = match.group(1).strip() if match else None
    execution_payload: dict[str, Any] = {
        "wkv_mode": wkv_mode,
        "prompt_template": model_args.get("rwkv_prompt_template", "assistant"),
        "gemm_policy": (
            "fp16-accumulation" if wkv_mode == "fp16" else "fp32-accumulation"
        ),
        "evaluator": "lm-eval",
    }
    if isinstance(gpu, str) and gpu.strip():
        execution_payload["gpu"] = gpu.strip()
    if (
        isinstance(max_num_seqs, int)
        and not isinstance(max_num_seqs, bool)
        and max_num_seqs > 0
    ):
        execution_payload["max_num_seqs"] = max_num_seqs
    if (
        isinstance(max_num_batched_tokens, int)
        and not isinstance(max_num_batched_tokens, bool)
        and max_num_batched_tokens > 0
    ):
        execution_payload["max_num_batched_tokens"] = max_num_batched_tokens
    dependency_versions: dict[str, str] = {}
    evaluator_version = results.get("lm_eval_version")
    if isinstance(evaluator_version, str) and evaluator_version.strip():
        dependency_versions["lm-eval"] = evaluator_version.strip()
    if isinstance(torch_version, str) and torch_version.strip():
        dependency_versions["torch"] = torch_version.strip()
    if isinstance(backend_version, str) and backend_version.strip():
        dependency_versions["vllm"] = (
            f"{backend_version.strip()}@{backend_revision.strip()}"
            if isinstance(backend_revision, str) and backend_revision.strip()
            else backend_version.strip()
        )
    if dependency_versions:
        execution_payload["dependency_versions"] = dependency_versions
    if isinstance(backend_revision, str) and backend_revision.strip():
        execution_payload["backend_revision"] = backend_revision.strip()
    return execution_payload


def _native_task_config(
    results: dict[str, Any],
    native_config: dict[str, Any],
    task_name: str,
    sample_count: int,
) -> dict[str, Any]:
    value = _json_safe(native_config)
    if not isinstance(value, dict):
        value = {}
    counts = results.get("n-samples")
    counts = counts.get(task_name) if isinstance(counts, dict) else None
    original = counts.get("original") if isinstance(counts, dict) else None
    effective = counts.get("effective") if isinstance(counts, dict) else None
    if not isinstance(original, int) or isinstance(original, bool) or original <= 0:
        raise ScoreboardError(f"lm-eval task {task_name} lacks original sample count")
    if not isinstance(effective, int) or isinstance(effective, bool) or effective <= 0:
        raise ScoreboardError(f"lm-eval task {task_name} lacks effective sample count")
    if effective != sample_count:
        raise ScoreboardError(
            f"lm-eval task {task_name} sample count does not match n-samples.effective"
        )
    output_type = value.get("output_type")
    generation_kwargs = value.get("generation_kwargs")
    if not isinstance(generation_kwargs, dict):
        generation_kwargs = {}
    max_gen_toks = generation_kwargs.get("max_gen_toks") or generation_kwargs.get(
        "max_tokens"
    )
    if not isinstance(max_gen_toks, int) or isinstance(max_gen_toks, bool):
        max_gen_toks = 1 if output_type != "generate_until" else None
    if max_gen_toks is None:
        raise ScoreboardError(f"lm-eval task {task_name} lacks generation size")
    value.update(
        {
            "generation_size": max_gen_toks,
            "original_num_docs": original,
            "effective_num_docs": effective,
            "not_evaluated_num_docs": original - effective,
        }
    )
    return value


def _native_sampling(
    results: dict[str, Any], config: dict[str, Any], native_config: dict[str, Any]
) -> dict[str, Any]:
    model_args = config.get("model_args")
    model_args = model_args if isinstance(model_args, dict) else {}
    generation_kwargs = native_config.get("generation_kwargs")
    if not isinstance(generation_kwargs, dict):
        generation_kwargs = config.get("gen_kwargs", {})
    if not isinstance(generation_kwargs, dict):
        generation_kwargs = {}
    sampling = {
        "generation_kwargs": _json_safe(generation_kwargs),
        "rwkv_prompt_template": model_args.get("rwkv_prompt_template"),
        "rwkv_generation_prompt": model_args.get("rwkv_generation_prompt"),
        "rwkv_sampling_mode": model_args.get("rwkv_sampling_mode"),
        "batch_size": config.get("batch_size"),
        "num_concurrent": model_args.get("num_concurrent"),
        "max_length": model_args.get("max_length") or results.get("max_length"),
        "eot_token_id": results.get("eot_token_id"),
    }
    prompt = model_args.get("rwkv_prompt_template")
    generation_prompt = model_args.get("rwkv_generation_prompt")
    if prompt in {"assistant", "bot", "function_calling"}:
        sampling["stop"] = [
            {"assistant": "\nUser:", "bot": "✿", "function_calling": "\n### User"}[
                prompt
            ]
        ]
    if generation_prompt == "open_think":
        sampling.update(
            temperature=0.96,
            top_p=0.76,
            top_k=32,
            presence_penalty=1.0,
            frequency_penalty=0.1,
            penalty_decay=0.988,
        )
    elif generation_prompt == "fake_think":
        sampling.update(temperature=1.0, top_p=0.28, top_k=32)
    return sampling


def _native_diagnostics(details: list[dict[str, Any]]) -> dict[str, Any]:
    completions = 0
    truncated = 0
    violations = 0
    for detail in details:
        response = detail["model_response"]
        texts = response.get("text")
        output_tokens = response.get("output_tokens")
        logprobs = response.get("logprobs")
        if not isinstance(output_tokens, list):
            continue
        completion_count = (
            len(texts)
            if isinstance(texts, list)
            else (len(logprobs) if isinstance(logprobs, list) else len(output_tokens))
        )
        evidence = response.get("evidence")
        evidence_items = evidence if isinstance(evidence, list) else []
        for index in range(min(completion_count, len(output_tokens))):
            tokens = output_tokens[index]
            if not isinstance(tokens, list):
                continue
            completions += 1
            matching = evidence_items[index] if index < len(evidence_items) else {}
            truncated += int(
                isinstance(matching, dict)
                and (
                    matching.get("truncation") is True
                    or matching.get("finish_reason") in {"length", "max_tokens"}
                )
            )
            text = (
                texts[index] if isinstance(texts, list) and index < len(texts) else ""
            )
            violations += int(
                isinstance(text, str)
                and any(stop in text for stop in ("✿", "\nUser:", "\n### User"))
            )
    return {
        "samples": len(details),
        "completions": completions,
        "truncated": truncated,
        "non_truncated": completions - truncated,
        "truncation_rate": truncated / completions if completions else 0.0,
        "turn_boundary_violations": violations,
        "turn_boundary_violation_rate": violations / completions
        if completions
        else 0.0,
    }


def build_lm_eval_publication(
    results: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    *,
    publication: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one publication contract for every native lm-eval task.

    This is deliberately task-name driven rather than benchmark-family driven:
    RACE, DROP, and future task registrations use the same conversion path and
    retain their task-specific metadata in ``task_metadata`` when supplied.
    """

    if not isinstance(samples, dict) or not samples:
        raise ScoreboardError("lm-eval publication requires logged per-sample results")
    publication = publication or {}
    config = results.get("config")
    config = config if isinstance(config, dict) else {}
    model_args = config.get("model_args")
    model_name = results.get("model_name")
    if (not isinstance(model_name, str) or not model_name.strip()) and isinstance(
        model_args, dict
    ):
        model_name = model_args.get("model") or model_args.get("pretrained")
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = config.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ScoreboardError("lm-eval results lack model identity")
    model_name = model_name.strip()
    task_names = [name for name in samples if isinstance(name, str) and name]
    if not task_names:
        raise ScoreboardError("lm-eval publication has no task samples")
    task_metadata = publication.get("task_metadata", {})
    task_metadata = task_metadata if isinstance(task_metadata, dict) else {}
    resolved_configs = results.get("configs", {})
    resolved_configs = resolved_configs if isinstance(resolved_configs, dict) else {}
    evaluator_version = results.get("lm_eval_version")
    if not isinstance(evaluator_version, str) or not evaluator_version.strip():
        raise ScoreboardError("lm-eval results lack evaluator version")
    evaluator = {"name": "lm-eval", "version": evaluator_version}
    enabled = publication.get("enabled") is True
    model_sha256 = publication.get("model_sha256") or config.get("model_sha")
    if publication.get("enabled") and not (
        isinstance(model_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", model_sha256)
    ):
        raise ScoreboardError(
            "publication.model_sha256 is required for an enabled publication; "
            "the uploader will not invent model identity"
        )
    model = {"name": model_name}
    if isinstance(model_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        model["sha256"] = model_sha256
    if publication.get("model_revision"):
        model["revision"] = publication["model_revision"]
    if enabled and "sha256" not in model:
        raise ScoreboardError(
            "publication.model_sha256 is required for enabled publication"
        )
    config_digest = content_digest(
        _json_safe(
            {
                "model": model,
                "config": config,
                "tasks": task_names,
                "metadata": results.get("configs", {}),
            }
        )
    )
    expected_tasks: list[dict[str, Any]] = []
    task_payloads: list[dict[str, Any]] = []
    for task_name in task_names:
        native_config = resolved_configs.get(task_name, {})
        native_config = native_config if isinstance(native_config, dict) else {}
        custom_metadata = _native_task_metadata(publication, task_name, native_config)
        native_metadata = native_config.get("metadata", {})
        native_metadata = native_metadata if isinstance(native_metadata, dict) else {}
        native_split = native_config.get("test_split") or native_config.get(
            "validation_split"
        )
        wkv_mode = _native_wkv_mode(publication, task_name, native_config)
        if "sha256" not in model:
            raise ScoreboardError(
                f"lm-eval task {task_name} cannot construct stable identity without model SHA-256"
            )
        identity = f"{model['sha256']}:{wkv_mode}:{task_name}"
        versions = results.get("versions")
        recorded_version = (
            versions.get(task_name) if isinstance(versions, dict) else None
        )
        task_version = custom_metadata.get(
            "task_version",
            recorded_version or native_metadata.get("version", "unknown"),
        )
        if not isinstance(task_version, str):
            task_version = str(task_version)
        dataset = custom_metadata.get("dataset", native_config.get("dataset_path"))
        if not isinstance(dataset, str) or not dataset.strip():
            dataset = None
        subset = custom_metadata.get("subset", native_config.get("dataset_name"))
        if subset is None:
            subset = ""
        if not isinstance(subset, str):
            subset = str(subset)
        evaluation_splits = custom_metadata.get(
            "evaluation_splits", [native_split] if native_split else ["unknown"]
        )
        if not isinstance(evaluation_splits, list) or not evaluation_splits:
            raise ScoreboardError(f"lm-eval task {task_name} lacks evaluation split")
        evaluation_splits = [str(value) for value in evaluation_splits]
        benchmark = custom_metadata.get("benchmark_name", task_name)
        if not isinstance(benchmark, str) or not benchmark.strip():
            benchmark = task_name
        descriptor = {
            "identity": identity,
            "weight_sha256": model["sha256"],
            "weight_display_name": model_name,
            "wkv_mode": wkv_mode,
            "benchmark": benchmark,
            "task_name": task_name,
            "task_version": task_version,
            "dataset": dataset.strip() if isinstance(dataset, str) else None,
            "subset": subset.strip() or None,
            "evaluation_splits": evaluation_splits,
            "languages": [str(value) for value in custom_metadata.get("languages", [])],
            "tags": [
                str(value)
                for value in custom_metadata.get(
                    "tags", custom_metadata.get("upstream_tags", [])
                )
            ],
        }
        expected_tasks.append(descriptor)
        task_samples = samples.get(task_name)
        if not isinstance(task_samples, list) or not task_samples:
            raise ScoreboardError(f"lm-eval task {task_name} has no samples")
        metrics = _task_metric_values(results, task_name)
        if not metrics:
            raise ScoreboardError(
                f"lm-eval task {task_name} has no finite aggregate metrics"
            )
        primary_metric = custom_metadata.get("primary_metric")
        if not isinstance(primary_metric, str) or primary_metric not in metrics:
            primary_metric = next(
                (name for name in metrics if not name.endswith("_stderr")),
                next(iter(metrics)),
            )
        details: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(task_samples):
            if not isinstance(sample, dict):
                raise ScoreboardError(
                    f"lm-eval task {task_name} sample[{sample_index}] is not an object"
                )
            document_index = sample.get("doc_id")
            if (
                isinstance(document_index, bool)
                or not isinstance(document_index, int)
                or document_index < 0
            ):
                raise ScoreboardError(
                    f"lm-eval task {task_name} sample[{sample_index}] lacks valid doc_id"
                )
            sample_metrics = {
                key: sample[key]
                for key in sample.get("metrics", [])
                if isinstance(key, str) and _finite_number(sample.get(key))
            }
            doc_value = _json_safe(sample.get("doc", {}))
            if not isinstance(doc_value, dict):
                raise ScoreboardError(
                    f"lm-eval task {task_name} sample[{sample_index}] document is not an object"
                )
            doc_value.setdefault("task_name", task_name)
            specific = doc_value.get("specific")
            if not isinstance(specific, dict):
                specific = {}
            specific.setdefault("helicopter_document_index", sample_index)
            specific["lm_eval_document_index"] = sample.get("doc_id", sample_index)
            doc_value["specific"] = specific
            doc_value["target"] = _json_safe(sample.get("target"))
            details.append(
                {
                    "sample_index": sample_index,
                    "document_index": document_index,
                    "doc": doc_value,
                    "metric": _json_safe(sample_metrics),
                    "model_response": _lm_eval_response(
                        sample, context=f"{task_name}.sample[{sample_index}]"
                    ),
                }
            )
        sample_count = len(details)
        task_config = _native_task_config(
            results, native_config, task_name, sample_count
        )
        model_execution = _native_execution(
            results, config, wkv_mode=wkv_mode, enabled=enabled
        )
        model_execution.update(
            weight_sha256=model["sha256"],
            weight_display_name=model_name,
        )
        evidence_complete = all(
            detail["model_response"].get("evidence_complete") is True
            for detail in details
        )
        if enabled and not evidence_complete:
            raise ScoreboardError(
                f"lm-eval task {task_name} lacks complete raw response/token evidence"
            )
        diagnostics = _native_diagnostics(details)
        task_payloads.append(
            {
                "schema_version": LM_EVAL_TASK_SCHEMA,
                "campaign_id": None,
                "task": descriptor,
                "result_files": [
                    {
                        "role": "metrics",
                        "path": "publication/raw_results.json",
                    },
                    {
                        "role": "samples",
                        "path": [
                            "publication/tasks/"
                            + re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name)
                            + ".json"
                        ][0],
                    },
                ],
                "task_config": task_config,
                "environment": {
                    **model_execution,
                    "model_name": model_name,
                    "model_revision": model.get("revision"),
                    "chat_template_sha": results.get("chat_template_sha"),
                    "task_hash": (results.get("task_hashes") or {}).get(task_name)
                    if isinstance(results.get("task_hashes"), dict)
                    else None,
                    "git_hash": results.get("git_hash"),
                },
                "sampling_config": _native_sampling(results, config, native_config),
                "primary_metric": primary_metric,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "samples": [
                    {
                        "sample_index": detail["sample_index"],
                        "document_index": detail["document_index"],
                        "document": detail["doc"],
                        "metrics": detail["metric"],
                        "model_response": detail["model_response"],
                    }
                    for detail in details
                ],
            }
        )
    campaign = {
        "schema_version": CAMPAIGN_SCHEMA,
        "source": "lm-eval-harness",
        "config_sha256": config_digest,
        "registry_sha256": content_digest(expected_tasks),
        "contract_sha256": content_digest(
            {"framework": "lm-eval", "version": evaluator.get("version")}
        ),
        "configured_benchmarks": list(
            dict.fromkeys(
                [
                    str(
                        _native_task_metadata(
                            publication, task_name, resolved_configs.get(task_name, {})
                        ).get("benchmark_name", task_name)
                    )
                    for task_name in task_names
                ]
            )
        ),
        "resolved_benchmarks": list(
            dict.fromkeys(
                [
                    str(
                        _native_task_metadata(
                            publication, task_name, resolved_configs.get(task_name, {})
                        ).get("benchmark_name", task_name)
                    )
                    for task_name in task_names
                ]
            )
        ),
        "skipped_benchmarks": [],
        "expected_tasks": expected_tasks,
        "rerun_reason": None,
    }
    campaign["run_key"] = campaign_run_key(campaign)
    return campaign, task_payloads


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_publication_payloads(
    output_dir: Path,
    campaign: dict[str, Any],
    task_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write converted DTOs without contacting a scoreboard."""

    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_path = output_dir / "campaign.json"
    _write_json_atomic(campaign_path, campaign)
    task_root = output_dir / "tasks"
    task_paths: list[Path] = []
    for payload in task_payloads:
        task = payload.get("task")
        identity = task.get("identity") if isinstance(task, dict) else None
        if not isinstance(identity, str) or not identity:
            raise ScoreboardError("converted task payload lacks task.identity")
        task_path = task_root / (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", identity).strip("_") + ".json"
        )
        _write_json_atomic(task_path, payload)
        task_paths.append(task_path)
    return {
        "converted": True,
        "network_access": False,
        "campaign_path": str(campaign_path),
        "task_paths": [str(path) for path in task_paths],
        "task_count": len(task_paths),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert evaluator artifacts into strict scoreboard DTOs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--producer-campaign",
        type=Path,
        help="rwkv-producer campaign JSON",
    )
    source.add_argument(
        "--lm-eval-raw-results",
        type=Path,
        help="raw_results.json written by the lm-eval publication spool",
    )
    parser.add_argument(
        "--producer-task",
        type=Path,
        action="append",
        default=[],
        help="rwkv-producer task JSON; repeat once per expected task",
    )
    parser.add_argument("--model-sha256")
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(value: dict[str, Any], output: TextIO | None = None) -> None:
    if output is None:
        output = sys.stdout
    json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.producer_campaign is not None:
            if not args.producer_task:
                raise ScoreboardError(
                    "--producer-campaign requires at least one --producer-task"
                )
            producer_campaign = _load_json_object(args.producer_campaign)
            producer_tasks = [_load_json_object(path) for path in args.producer_task]
            campaign, task_payloads = convert_producer_publication(
                producer_campaign, producer_tasks
            )
        else:
            if args.producer_task:
                raise ScoreboardError(
                    "--producer-task can only be used with --producer-campaign"
                )
            raw_results = _load_json_object(args.lm_eval_raw_results)
            samples = raw_results.pop("samples", None)
            if not isinstance(samples, dict):
                raise ScoreboardError(
                    "lm-eval raw_results.json must contain a samples object"
                )
            publication = {
                "enabled": True,
                "model_sha256": args.model_sha256,
                "model_revision": args.model_revision,
            }
            campaign, task_payloads = build_lm_eval_publication(
                raw_results,
                samples,
                publication={
                    key: value
                    for key, value in publication.items()
                    if value is not None
                },
            )
        receipt = write_publication_payloads(
            args.output_dir,
            campaign,
            task_payloads,
        )
        _write_json(receipt)
        return 0
    except (ScoreboardError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
