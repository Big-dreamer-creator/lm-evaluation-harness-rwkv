#!/usr/bin/env python3
"""Publish a scoreboard-rwkv campaign and its task payloads.

The scoreboard API deliberately accepts the versioned publication contracts
defined by scoreboard-rwkv.  This client transports those payloads; it does
not turn an lm-eval ``results_*.json`` summary into a different contract or
invent missing per-sample execution evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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


CAMPAIGN_SCHEMA = "lighteval-campaign-v3"
TASK_SCHEMA = "lighteval-task-v2"
LM_EVAL_CAMPAIGN_SCHEMA = "lm-eval-campaign-v1"
LM_EVAL_TASK_SCHEMA = "lm-eval-task-v1"
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

SUPPORTED_CAMPAIGN_SCHEMAS = {CAMPAIGN_SCHEMA, LM_EVAL_CAMPAIGN_SCHEMA}
SUPPORTED_TASK_SCHEMAS = {TASK_SCHEMA, LM_EVAL_TASK_SCHEMA}
SCOREBOARD_BASE_URL_ENV = "SCOREBOARD_BASE_URL"
SCOREBOARD_TOKEN_ENV_ENV = "SCOREBOARD_PUBLICATION_TOKEN_ENV"
SCOREBOARD_TOKEN_ENV = "SCOREBOARD_PUBLICATION_TOKEN"
SCOREBOARD_TIMEOUT_ENV = "SCOREBOARD_UPLOAD_TIMEOUT"
SCOREBOARD_FINALIZE_ENV = "SCOREBOARD_UPLOAD_FINALIZE"
SCOREBOARD_MODEL_SHA256_ENV = "SCOREBOARD_MODEL_SHA256"
SCOREBOARD_MODEL_REVISION_ENV = "SCOREBOARD_MODEL_REVISION"


class ScoreboardError(RuntimeError):
    """An actionable local or remote publication error."""


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
    """Resolve portable publication settings without reading the token itself."""

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


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _producer_provenance(campaign: dict[str, Any], task: dict[str, Any] | None = None) -> dict[str, Any]:
    candidates = [
        task.get("provenance") if isinstance(task, dict) else None,
        campaign.get("provenance"),
    ]
    for value in candidates:
        if isinstance(value, dict):
            return value
    raise ScoreboardError("producer publication lacks campaign provenance")


def _producer_weight_sha256(campaign: dict[str, Any], provenance: dict[str, Any]) -> str:
    value = campaign.get("weight_sha256") or provenance.get("weight_sha256")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ScoreboardError(
            "producer publication lacks a valid weight_sha256; refusing to invent model identity"
        )
    return value


def _producer_model_display_name(campaign: dict[str, Any], provenance: dict[str, Any]) -> str:
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
            raise ScoreboardError(f"producer expected_tasks[{index}] has invalid wkv_mode")
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
            "task_name": benchmark,
            **{
                key: deepcopy(contract[key])
                for key in (
                    "selector",
                    "task_version",
                    "module_family",
                    "module",
                    "dataset",
                    "subset",
                    "evaluation_splits",
                    "languages",
                    "upstream_tags",
                )
            },
        }
        expected.append(task_descriptor)
        if producer_identity in producer_to_scoreboard:
            raise ScoreboardError(f"duplicate producer task identity {producer_identity}")
        producer_to_scoreboard[producer_identity] = identity

    # The server requires the same task set in both WKV modes.  Let the
    # regular campaign validator enforce the exact DTO invariant after the
    # conversion, but provide a useful producer-side error first.
    modes_by_task: dict[str, set[str]] = {}
    for item in expected:
        modes_by_task.setdefault(item["task_name"], set()).add(item["wkv_mode"])
    missing = [task for task, modes in modes_by_task.items() if modes != {"fp16", "fp32io16"}]
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
        evidence.get("output_token_ids"), context=f"{context}.output_token_ids", allow_empty=False
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
    prompts = [item.get("prompt") for item in evidence if isinstance(item.get("prompt"), str)]
    response: dict[str, Any] = {
        "input": prompts[0] if prompts else None,
        "input_tokens": evidence[0].get("input_token_ids"),
    }
    if has_loglikelihood:
        if not all(
            choice is not None and isinstance(choice.get("logprobs"), dict)
            for choice in choices
        ):
            raise ScoreboardError(f"{context} mixes generation and log-likelihood evidence")
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
            raise ScoreboardError(f"{context}.evidence[{index}].reasoning is not a string")
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


def _producer_task_config(task: dict[str, Any], benchmark: str, sample_count: int) -> dict[str, Any]:
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
    return {
        key: value[key]
        for key in required
    } | {
        "producer_schema_version": PRODUCER_TASK_SCHEMA,
        "producer_task_config": deepcopy(value),
    }


def _producer_sampling(
    campaign: dict[str, Any], task: dict[str, Any], *, prompt_template: str
) -> dict[str, Any]:
    provenance = _producer_provenance(campaign, task)
    prompt = provenance.get("prompt")
    generation_prompt = prompt.get("generation_prompt") if isinstance(prompt, dict) else None
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
        raise ScoreboardError("producer sampling_config.stop does not match prompt template")
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


def _producer_diagnostics(details: list[dict[str, Any]], prompt_template: str) -> dict[str, Any]:
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
        "turn_boundary_violation_rate": violations / completions if completions else 0.0,
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
        raise ScoreboardError("producer task set does not match campaign: " + "; ".join(details))

    provenance = _producer_provenance(campaign)
    prompt = provenance.get("prompt")
    prompt_template = prompt.get("template") if isinstance(prompt, dict) else None
    if prompt_template not in {"bot", "assistant", "function_calling"}:
        raise ScoreboardError("producer provenance lacks a supported prompt template")
    weight_sha256 = _producer_weight_sha256(campaign, provenance)
    model_display_name = _producer_model_display_name(campaign, provenance)
    converted_campaign = {
        "schema_version": CAMPAIGN_SCHEMA,
        "run_key": campaign.get("run_key"),
        "config_digest": campaign.get("config_digest"),
        "registry_digest": campaign.get("registry_digest"),
        "eval_contract_digest": campaign.get("eval_contract_digest"),
        "lighteval_version": LIGHTEVAL_VERSION,
        "configured_selectors": deepcopy(campaign.get("configured_selectors")),
        "resolved_selectors": deepcopy(campaign.get("resolved_selectors")),
        "skipped_selectors": deepcopy(campaign.get("skipped_selectors")),
        "expected_tasks": expected,
    }

    converted_tasks: list[dict[str, Any]] = []
    expected_by_identity = {item["identity"]: item for item in expected}
    for producer_identity, scoreboard_identity in identity_map.items():
        producer_payload = by_producer_identity[producer_identity]
        producer_task = producer_payload.get("task")
        if not isinstance(producer_task, dict):
            raise ScoreboardError(f"producer task {producer_identity} lacks task object")
        benchmark = _producer_benchmark(
            next(item for item in campaign["expected_tasks"] if item.get("identity") == producer_identity),
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
                raise ScoreboardError(f"producer task {benchmark} has an invalid sample document")
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
                "artifact": {
                    "lighteval_version": LIGHTEVAL_VERSION,
                    "results_path": f"results/{model_display_name}/{benchmark}/{descriptor['wkv_mode']}.json",
                    "details_paths": [
                        f"details/{model_display_name}/{benchmark}/{descriptor['wkv_mode']}.jsonl"
                    ],
                },
                "task_config": _producer_task_config(producer_task, benchmark, len(details)),
                "model": {
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
                },
                "sampling_config": _producer_sampling(
                    campaign, task_for_sampling, prompt_template=prompt_template
                ),
                "primary_metric": primary_metric,
                "aggregates": aggregates,
                "diagnostics": _producer_diagnostics(details, prompt_template),
                "details": details,
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
        raise ScoreboardError(f"evaluation result is not JSON serializable: {error}") from error


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


def _lm_eval_response(sample: dict[str, Any], *, context: str) -> dict[str, Any]:
    evidence = sample.get("response_evidence")
    evidence_items = evidence if isinstance(evidence, list) else []

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
        return isinstance(choice.get("text"), str) or isinstance(
            choice.get("logprobs"), dict
        )

    response: dict[str, Any] = {
        "raw_resps": _json_safe(sample.get("resps", [])),
        "filtered_resps": _json_safe(sample.get("filtered_resps", [])),
        "arguments": _json_safe(sample.get("arguments", [])),
        "evidence_complete": False,
    }
    if isinstance(evidence, list):
        response["evidence"] = _json_safe(evidence)
        prompts = [item.get("prompt") for item in evidence if isinstance(item, dict)]
        response["input"] = next((item for item in prompts if isinstance(item, str)), None)
        response["input_tokens"] = next(
            (
                item.get("input_token_ids")
                for item in evidence
                if isinstance(item, dict)
                and isinstance(item.get("input_token_ids"), list)
            ),
            [],
        )
        texts: list[str] = []
        output_tokens: list[list[int]] = []
        answers: list[str] = []
        logprobs: list[float] = []
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                raise ScoreboardError(f"{context}.response_evidence[{index}] is not an object")
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
            token_ids = item.get("output_token_ids")
            if isinstance(token_ids, list) and all(
                isinstance(token, int) and not isinstance(token, bool) for token in token_ids
            ):
                output_tokens.append(token_ids)
            answer = item.get("post_processed_answer")
            if isinstance(answer, str):
                answers.append(answer)
        response["text"] = texts
        response["output_tokens"] = output_tokens
        response["text_post_processed"] = answers
        if logprobs:
            response["logprobs"] = logprobs
        response["evidence_complete"] = bool(evidence_items) and all(
            isinstance(item, dict)
            and isinstance(item.get("input_token_ids"), list)
            and isinstance(item.get("output_token_ids"), list)
            and evidence_has_output(item)
            for item in evidence_items
        )
    return response


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
    evaluator = {
        "framework": "lm-eval",
        "version": results.get("lm_eval_version"),
        "git_hash": results.get("git_hash"),
    }
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
    run_key = content_digest(
        {
            "config_digest": config_digest,
            "model": model,
            "tasks": task_names,
            "results": _json_safe(results.get("results", {})),
        }
    )
    expected_tasks: list[dict[str, Any]] = []
    task_payloads: list[dict[str, Any]] = []
    for task_name in task_names:
        custom_metadata = task_metadata.get(task_name, {})
        if not isinstance(custom_metadata, dict):
            custom_metadata = {}
        native_config = resolved_configs.get(task_name, {})
        native_config = native_config if isinstance(native_config, dict) else {}
        native_metadata = native_config.get("metadata", {})
        native_metadata = native_metadata if isinstance(native_metadata, dict) else {}
        native_split = native_config.get("test_split") or native_config.get(
            "validation_split"
        )
        identity = f"{model_name}:{task_name}"
        descriptor = {
            "identity": identity,
            "task_name": task_name,
            "selector": task_name,
            "task_version": custom_metadata.get(
                "task_version", native_metadata.get("version", "native")
            ),
            "module_family": custom_metadata.get("module_family", task_name),
            "module": custom_metadata.get(
                "module", native_config.get("task", "lm_eval.tasks")
            ),
            "dataset": custom_metadata.get(
                "dataset", native_config.get("dataset_path")
            ),
            "subset": custom_metadata.get(
                "subset", native_config.get("dataset_name")
            ),
            "evaluation_splits": custom_metadata.get(
                "evaluation_splits", [native_split] if native_split else []
            ),
            "languages": custom_metadata.get("languages", []),
            "upstream_tags": custom_metadata.get("upstream_tags", []),
            "model": model,
            "evaluator": evaluator,
        }
        descriptor.update(
            {
                key: _json_safe(value)
                for key, value in custom_metadata.items()
                if key not in descriptor and key != "identity"
            }
        )
        expected_tasks.append(descriptor)
        task_samples = samples.get(task_name)
        if not isinstance(task_samples, list) or not task_samples:
            raise ScoreboardError(f"lm-eval task {task_name} has no samples")
        metrics = _task_metric_values(results, task_name)
        if not metrics:
            raise ScoreboardError(f"lm-eval task {task_name} has no finite aggregate metrics")
        primary_metric = custom_metadata.get("primary_metric")
        if not isinstance(primary_metric, str) or primary_metric not in metrics:
            primary_metric = next(
                (name for name in metrics if not name.endswith("_stderr")),
                next(iter(metrics)),
            )
        details: list[dict[str, Any]] = []
        truncated = 0
        for sample_index, sample in enumerate(task_samples):
            if not isinstance(sample, dict):
                raise ScoreboardError(f"lm-eval task {task_name} sample[{sample_index}] is not an object")
            evidence = sample.get("response_evidence")
            if isinstance(evidence, list):
                truncated += sum(
                    int(
                        isinstance(item, dict)
                        and (
                            item.get("truncation") is True
                            or item.get("finish_reason") in {"length", "max_tokens"}
                        )
                    )
                    for item in evidence
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
            details.append(
                {
                    "sample_index": sample_index,
                    "document_index": sample_index,
                    "doc": doc_value,
                    "target": _json_safe(sample.get("target")),
                    "metric": _json_safe(sample_metrics),
                    "model_response": _lm_eval_response(
                        sample, context=f"{task_name}.sample[{sample_index}]"
                    ),
                }
            )
        task_payloads.append(
            {
                "schema_version": LM_EVAL_TASK_SCHEMA,
                "campaign_id": None,
                "task": descriptor,
                "model": model,
                "evaluator": evaluator,
                "primary_metric": primary_metric,
                "aggregates": metrics,
                "diagnostics": {
                    "samples": len(details),
                    "truncated": truncated,
                    "truncation_rate": truncated / len(details) if details else 0.0,
                    "evidence_complete": all(
                        detail["model_response"].get("evidence_complete") is True
                        for detail in details
                    ),
                },
                "details": details,
            }
        )
    campaign = {
        "schema_version": LM_EVAL_CAMPAIGN_SCHEMA,
        "run_key": run_key,
        "config_digest": config_digest,
        "registry_digest": content_digest({"tasks": task_names}),
        "eval_contract_digest": content_digest(
            {"framework": "lm-eval", "version": evaluator.get("version")}
        ),
        "evaluator": evaluator,
        "model": model,
        "model_name": model_name,
        "configured_selectors": task_names,
        "resolved_selectors": task_names,
        "skipped_selectors": [],
        "expected_tasks": expected_tasks,
        "publication_contract": "lm-eval-native-v1",
    }
    return campaign, task_payloads


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def publish_lm_eval_evaluation(
    results: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    *,
    output_dir: str | Path,
    publication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist native evidence and optionally publish it in one recoverable flow."""

    publication = resolve_publication_settings(publication)
    result_config = results.get("config")
    result_config = result_config if isinstance(result_config, dict) else {}
    result_model_args = result_config.get("model_args")
    model_name = results.get("model_name")
    if (not isinstance(model_name, str) or not model_name) and isinstance(
        result_model_args, dict
    ):
        model_name = result_model_args.get("model") or result_model_args.get("pretrained")
    if not isinstance(model_name, str) or not model_name:
        model_name = result_config.get("model")
    if not isinstance(model_name, str) or not model_name:
        model_name = "model"
    root = _artifact_root(Path(output_dir), model_name)
    root.mkdir(parents=True, exist_ok=True)
    raw_results = _json_safe(results)
    raw_results["samples"] = _json_safe(samples)
    _write_json_atomic(root / "raw_results.json", raw_results)
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
            "raw_results_path": str(root / "raw_results.json"),
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
        "publication": "disabled" if not publication.get("enabled", False) else "pending",
        "uploaded": False,
        "campaign_path": str(campaign_path),
        "task_paths": [str(path) for path in task_paths],
        "raw_results_path": str(root / "raw_results.json"),
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
                    "Scoreboard publication requires complete raw response and token evidence; "
                    "missing for tasks: " + ", ".join(incomplete_tasks)
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
                f"publication.base_url or {SCOREBOARD_BASE_URL_ENV} is required when publication is enabled"
            )
        if not token:
            raise ScoreboardError(f"publication token is missing from {token_env}")
        campaign_payload, task_by_identity, expected_identities = load_publication_inputs(
            campaign_path, task_paths
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
                f"expected one of {sorted(SUPPORTED_CAMPAIGN_SCHEMAS)}, got {schema_version!r}"
            )
        if (
            schema_version == CAMPAIGN_SCHEMA
            and response.get("lighteval_version") != LIGHTEVAL_VERSION
        ):
            raise ScoreboardError(
                "scoreboard LightEval version mismatch: "
                f"expected {LIGHTEVAL_VERSION}, got {response.get('lighteval_version')!r}"
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
            f"campaign.schema_version must be one of {sorted(SUPPORTED_CAMPAIGN_SCHEMAS)!r}; "
            "raw lm-eval results are not scoreboard publication payloads"
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
                f"task file {index} schema_version must be one of {sorted(SUPPORTED_TASK_SCHEMAS)!r}"
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
    campaign = _load_json_object(campaign_path)
    payloads = [_load_json_object(path) for path in task_paths]
    if campaign.get("schema_version") == PRODUCER_CAMPAIGN_SCHEMA:
        campaign, payloads = convert_producer_publication(campaign, payloads)
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
        description="Upload scoreboard-rwkv campaign/task publication JSON payloads."
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
    parser.add_argument("--campaign", type=Path, help="lighteval-campaign-v3 JSON file")
    parser.add_argument(
        "--task",
        type=Path,
        action="append",
        default=[],
        help="lighteval-task-v2 JSON file; repeat once per expected task",
    )
    parser.add_argument(
        "--producer-campaign",
        type=Path,
        help=(
            "rwkv-producer campaign JSON; when set, convert it with the supplied "
            "--producer-task files before upload"
        ),
    )
    parser.add_argument(
        "--producer-task",
        type=Path,
        action="append",
        default=[],
        help="rwkv-producer task JSON; repeat once per producer expected task",
    )
    parser.add_argument(
        "--converted-output-dir",
        type=Path,
        help="write converted lighteval DTOs here and stop before network upload",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help=f"HTTP timeout in seconds (env: {SCOREBOARD_TIMEOUT_ENV}; default: 3600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate local payloads without contacting the scoreboard",
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
        if args.producer_campaign is not None:
            if args.campaign is not None or args.task:
                raise ScoreboardError(
                    "--producer-campaign/--producer-task cannot be combined with --campaign/--task"
                )
            if not args.producer_task:
                raise ScoreboardError(
                    "--producer-campaign requires at least one --producer-task"
                )
            producer_campaign = _load_json_object(args.producer_campaign)
            producer_tasks = [_load_json_object(path) for path in args.producer_task]
            campaign_payload, task_payloads = convert_producer_publication(
                producer_campaign, producer_tasks
            )
            if args.converted_output_dir is not None:
                args.converted_output_dir.mkdir(parents=True, exist_ok=True)
                campaign_path = args.converted_output_dir / "campaign.json"
                campaign_path.write_text(
                    json.dumps(campaign_payload, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                for payload in task_payloads:
                    identity = payload["task"]["identity"]
                    task_path = args.converted_output_dir / (
                        re.sub(r"[^A-Za-z0-9_.-]+", "_", identity) + ".json"
                    )
                    task_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
                _write_json(
                    {
                        "dry_run": True,
                        "converted": True,
                        "campaign_path": str(campaign_path),
                        "task_count": len(task_payloads),
                    }
                )
                return 0
            campaign = campaign_payload
            expected_identities = _validate_campaign(campaign)
            task_by_identity = _validate_tasks(task_payloads, expected_identities)
        else:
            if args.campaign is None or not args.task:
                raise ScoreboardError(
                    "--campaign/--task or --producer-campaign/--producer-task are required"
                )
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
