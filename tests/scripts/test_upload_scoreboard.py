from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).parents[2]
CONVERTER_SCRIPT = ROOT / "scripts/convert_scoreboard_payloads.py"
CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "scripts.convert_scoreboard_payloads", CONVERTER_SCRIPT
)
CONVERTER = importlib.util.module_from_spec(CONVERTER_SPEC)
assert CONVERTER_SPEC.loader is not None
sys.modules[CONVERTER_SPEC.name] = CONVERTER
CONVERTER_SPEC.loader.exec_module(CONVERTER)

SCRIPT = ROOT / "scripts/upload_scoreboard.py"
SPEC = importlib.util.spec_from_file_location("upload_scoreboard", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def campaign() -> dict:
    weight_sha256 = "a" * 64
    value = {
        "schema_version": MODULE.CAMPAIGN_SCHEMA,
        "source": "lm-eval-harness",
        "config_sha256": "b" * 64,
        "registry_sha256": "c" * 64,
        "contract_sha256": "d" * 64,
        "configured_benchmarks": ["task"],
        "resolved_benchmarks": ["task"],
        "skipped_benchmarks": [],
        "expected_tasks": [
            expected_task(weight_sha256, "fp16"),
            expected_task(weight_sha256, "fp32io16"),
        ],
    }
    value["run_key"] = CONVERTER.campaign_run_key(value)
    return value


def expected_task(weight_sha256: str, wkv_mode: str) -> dict:
    return {
        "identity": f"{weight_sha256}:{wkv_mode}:task",
        "weight_sha256": weight_sha256,
        "weight_display_name": "model.pth",
        "wkv_mode": wkv_mode,
        "benchmark": "task",
        "task_name": "task",
        "task_version": "1.0",
        "dataset": "dataset/task",
        "subset": "full",
        "evaluation_splits": ["test"],
        "languages": ["english"],
        "tags": ["test"],
    }


def task(identity: str) -> dict:
    weight_sha256, wkv_mode, _ = identity.split(":", 2)
    return {
        "schema_version": MODULE.TASK_SCHEMA,
        "campaign_id": "assigned-by-uploader",
        "task": expected_task(weight_sha256, wkv_mode),
        "result_files": [{"role": "metrics", "path": "results/model/task.json"}],
        "task_config": {
            "generation_size": 1,
            "original_num_docs": 1,
            "effective_num_docs": 1,
            "skipped_multiselect_docs": 0,
        },
        "environment": {"source": "lm-eval-harness"},
        "sampling_config": {},
        "primary_metric": "acc",
        "metrics": {"acc": 0.5},
        "diagnostics": {},
        "samples": [],
    }


def write_inputs(tmp_path: Path) -> tuple[Path, list[Path]]:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign()), encoding="utf-8")
    task_paths = []
    weight_sha256 = "a" * 64
    for index, mode in enumerate(("fp16", "fp32io16")):
        identity = f"{weight_sha256}:{mode}:task"
        path = tmp_path / f"task-{index}.json"
        path.write_text(json.dumps(task(identity)), encoding="utf-8")
        task_paths.append(path)
    return campaign_path, task_paths


def test_api_root_preserves_deployment_prefix() -> None:
    assert MODULE._api_root("https://eval.rwkv.rs") == "https://eval.rwkv.rs/api"
    assert (
        MODULE._api_root("https://eval.rwkv.rs/test/")
        == "https://eval.rwkv.rs/test/api"
    )
    assert MODULE._api_root("http://127.0.0.1:7872/api") == "http://127.0.0.1:7872/api"

    with pytest.raises(MODULE.ScoreboardError, match=r"absolute http\(s\) URL"):
        MODULE._api_root("eval.rwkv.rs")


def test_load_inputs_rejects_raw_lm_eval_results(tmp_path: Path) -> None:
    raw_result = tmp_path / "results.json"
    raw_result.write_text(
        json.dumps({"results": {"task": {"acc": 0.5}}}), encoding="utf-8"
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task("a" * 64 + ":fp16:task")), encoding="utf-8")

    with pytest.raises(MODULE.ScoreboardError, match="scoreboard-v1"):
        MODULE.load_publication_inputs(raw_result, [task_path])


def test_load_inputs_requires_exact_task_set(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign()), encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task("a" * 64 + ":fp16:task")), encoding="utf-8")

    with pytest.raises(MODULE.ScoreboardError, match="does not match campaign"):
        MODULE.load_publication_inputs(campaign_path, [task_path])


def _producer_inputs(*, scoreboard_compatible: bool = True) -> tuple[dict, list[dict]]:
    weight = "e" * 64
    provenance = {
        "model_name": "rwkv7-g1i-1.5b-20260805-ctx16384",
        "weight_path": "/mnt/e/code/Weights/rwkv7-g1i-1.5b-20260805-ctx16384.pth",
        "weight_sha256": weight,
        "backend_commit": "f" * 40,
        "prompt": {
            "template": "assistant",
            "generation_prompt": "open_think",
        },
        "sampling_config": {
            **CONVERTER._TARGET_SAMPLING,
            "stop": ["\nUser:"],
        },
        "generation_config": {"max_gen_toks": 8192, "do_sample": True},
        "scoreboard_compatible": scoreboard_compatible,
        "gpu": [{"name": "Test GPU"}],
        "runtime": {
            "rwkv_max_num_seqs": 24,
            "max_num_batched_tokens": 16384,
        },
        "dependencies": {
            "packages": {"torch": "2.11.0", "lighteval": "0.13.0"},
        },
    }
    campaign = {
        "schema_version": CONVERTER.PRODUCER_CAMPAIGN_SCHEMA,
        "publication_contract": "rwkv-producer-v1",
        "scoreboard_upload_ready": False,
        "scoreboard_compatible": scoreboard_compatible,
        "run_key": "a" * 64,
        "config_digest": "b" * 64,
        "registry_digest": "c" * 64,
        "eval_contract_digest": "d" * 64,
        "lighteval_version": CONVERTER.LIGHTEVAL_VERSION,
        "configured_selectors": ["moral_stories"],
        "resolved_selectors": ["moral_stories"],
        "skipped_selectors": [],
        "model_name": provenance["model_name"],
        "weight_sha256": weight,
        "provenance": provenance,
        "expected_tasks": [
            {
                "identity": f"{provenance['model_name']}:fp16:moral_stories",
                "task": "moral_stories",
                "wkv_mode": "fp16",
            },
            {
                "identity": f"{provenance['model_name']}:fp32io16:moral_stories",
                "task": "moral_stories",
                "wkv_mode": "fp32io16",
            },
        ],
    }
    payloads = []
    for mode in ("fp16", "fp32io16"):
        identity = f"{provenance['model_name']}:{mode}:moral_stories"
        payloads.append(
            {
                "schema_version": CONVERTER.PRODUCER_TASK_SCHEMA,
                "campaign_id": None,
                "task": {
                    "identity": identity,
                    "task_name": "moral_stories",
                    "model_name": provenance["model_name"],
                    "wkv_mode": mode,
                    "results": {
                        "metrics": {"acc": 1.0},
                        "sample_count": 1,
                        "task_config": {
                            "generation_size": 8192,
                            "original_num_docs": 1,
                            "effective_num_docs": 1,
                            "skipped_multiselect_docs": 0,
                        },
                    },
                    "task_config": {
                        "generation_size": 8192,
                        "original_num_docs": 1,
                        "effective_num_docs": 1,
                        "skipped_multiselect_docs": 0,
                    },
                    "samples": [
                        {
                            "doc_id": 0,
                            "doc": {"query": "Is this right?", "label": 1},
                            "metrics": ["acc"],
                            "acc": 1,
                            "response_evidence": [
                                {
                                    "prompt": "Is this right?",
                                    "input_token_ids": [10],
                                    "output_token_ids": [11],
                                    "raw_response": {
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": "Yes",
                                                "logprobs": {
                                                    "token_logprobs": [None, -0.1]
                                                },
                                            }
                                        ]
                                    },
                                    "post_processed_answer": "True",
                                    "reasoning": None,
                                }
                            ],
                        }
                    ],
                    "provenance": {**provenance, "wkv_mode": mode},
                },
            }
        )
    return campaign, payloads


def test_producer_payload_is_converted_losslessly_to_scoreboard_dto() -> None:
    producer_campaign, producer_tasks = _producer_inputs()

    campaign_payload, task_payloads = CONVERTER.convert_producer_publication(
        producer_campaign, producer_tasks
    )

    assert campaign_payload["schema_version"] == MODULE.CAMPAIGN_SCHEMA
    assert all(
        payload["schema_version"] == MODULE.TASK_SCHEMA for payload in task_payloads
    )
    assert {payload["task"]["identity"] for payload in task_payloads} == {
        f"{'e' * 64}:fp16:moral_stories",
        f"{'e' * 64}:fp32io16:moral_stories",
    }
    detail = task_payloads[0]["samples"][0]
    assert detail["model_response"]["logprobs"] == [-0.1]
    assert detail["model_response"]["output_tokens"] == [[11]]
    assert detail["document"]["specific"]["helicopter_document_index"] == 0


def test_producer_conversion_rejects_non_publishable_sampling_contract() -> None:
    campaign_payload, task_payloads = _producer_inputs(scoreboard_compatible=False)
    with pytest.raises(MODULE.ScoreboardError, match="scoreboard-compatible"):
        CONVERTER.convert_producer_publication(campaign_payload, task_payloads)


def test_converter_and_uploader_have_separate_cli_contracts(
    tmp_path: Path, capsys
) -> None:
    producer_campaign, producer_tasks = _producer_inputs()
    campaign_path = tmp_path / "producer-campaign.json"
    campaign_path.write_text(json.dumps(producer_campaign), encoding="utf-8")
    task_paths = []
    for index, payload in enumerate(producer_tasks):
        path = tmp_path / f"producer-task-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        task_paths.append(path)
    converted = tmp_path / "converted"

    assert (
        CONVERTER.main(
            [
                "--producer-campaign",
                str(campaign_path),
                "--producer-task",
                str(task_paths[0]),
                "--producer-task",
                str(task_paths[1]),
                "--output-dir",
                str(converted),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["network_access"] is False
    assert receipt["task_count"] == 2
    converted_task_paths = sorted((converted / "tasks").glob("*.json"))
    assert len(converted_task_paths) == 2

    assert (
        MODULE.main(
            [
                "--campaign",
                str(converted / "campaign.json"),
                "--task",
                str(converted_task_paths[0]),
                "--task",
                str(converted_task_paths[1]),
                "--dry-run",
            ]
        )
        == 0
    )
    upload_dry_run = json.loads(capsys.readouterr().out)
    assert upload_dry_run["expected_task_count"] == 2
    assert not hasattr(MODULE, "convert_producer_publication")
    assert not hasattr(CONVERTER, "ScoreboardClient")


def test_main_dry_run_does_not_require_credentials(tmp_path: Path, capsys) -> None:
    campaign_path, task_paths = write_inputs(tmp_path)

    assert (
        MODULE.main(
            [
                "--campaign",
                str(campaign_path),
                "--task",
                str(task_paths[0]),
                "--task",
                str(task_paths[1]),
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["expected_task_count"] == 2


def test_publication_settings_resolve_environment_with_explicit_precedence(
    monkeypatch,
) -> None:
    monkeypatch.setenv(MODULE.SCOREBOARD_BASE_URL_ENV, "http://127.0.0.1:7860")
    monkeypatch.setenv(MODULE.SCOREBOARD_TOKEN_ENV_ENV, "LOCAL_SCOREBOARD_TOKEN")
    monkeypatch.setenv("LOCAL_SCOREBOARD_TOKEN", "local-publication-token")
    monkeypatch.setenv(MODULE.SCOREBOARD_TIMEOUT_ENV, "45.5")
    monkeypatch.setenv(MODULE.SCOREBOARD_FINALIZE_ENV, "false")
    monkeypatch.setenv(MODULE.SCOREBOARD_MODEL_SHA256_ENV, "e" * 64)
    monkeypatch.setenv(MODULE.SCOREBOARD_MODEL_REVISION_ENV, "local-revision")

    resolved = MODULE.resolve_publication_settings()
    args = MODULE._build_parser().parse_args(["--preflight-only"])

    assert resolved == {
        "base_url": "http://127.0.0.1:7860",
        "token_env": "LOCAL_SCOREBOARD_TOKEN",
        "timeout": 45.5,
        "finalize": False,
        "model_sha256": "e" * 64,
        "model_revision": "local-revision",
    }
    assert MODULE._require_credentials(args) == (
        "http://127.0.0.1:7860",
        "local-publication-token",
        45.5,
        False,
    )
    assert (
        MODULE.resolve_publication_settings({"timeout": 12, "finalize": True})[
            "timeout"
        ]
        == 12.0
    )
    assert (
        MODULE.resolve_publication_settings({"timeout": 12, "finalize": True})[
            "finalize"
        ]
        is True
    )


def test_publication_settings_reject_invalid_environment(monkeypatch) -> None:
    monkeypatch.setenv(MODULE.SCOREBOARD_TIMEOUT_ENV, "never")
    with pytest.raises(MODULE.ScoreboardError, match=MODULE.SCOREBOARD_TIMEOUT_ENV):
        MODULE.resolve_publication_settings()

    monkeypatch.setenv(MODULE.SCOREBOARD_TIMEOUT_ENV, "12")
    monkeypatch.setenv(MODULE.SCOREBOARD_FINALIZE_ENV, "sometimes")
    with pytest.raises(MODULE.ScoreboardError, match=MODULE.SCOREBOARD_FINALIZE_ENV):
        MODULE.resolve_publication_settings()


def test_preflight_must_advertise_requested_schema(monkeypatch) -> None:
    request_options = {}
    client = MODULE.ScoreboardClient(
        base_url="https://eval.rwkv.rs/test",
        token="secret",  # noqa: S106
        timeout=12,
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda method, path, **kwargs: (
            request_options.update(kwargs)
            or {
                "status": "ready",
                "schema_version": MODULE.CAMPAIGN_SCHEMA,
                "sources": ["lm-eval-harness"],
            }
        ),
    )

    assert (
        client.preflight(expected_campaign_schema=MODULE.SCOREBOARD_SCHEMA)[
            "schema_version"
        ]
        == MODULE.SCOREBOARD_SCHEMA
    )
    assert request_options["timeout"] == MODULE.CONTROL_REQUEST_TIMEOUT


def test_native_conversion_rejects_incomplete_evidence() -> None:
    results = {
        "config": {
            "model": "rwkv7-http",
            "model_args": {
                "model": "rwkv7-g1i-1.5b-20260805-ctx16384",
                "rwkv_prompt_template": "assistant",
                "rwkv_generation_prompt": "fake_think",
                "rwkv_sampling_mode": "profile",
                "num_concurrent": 5,
                "max_length": 16384,
            },
        },
        "configs": {
            "race": {
                "task": "race",
                "dataset_path": "race",
                "test_split": "test",
                "output_type": "generate_until",
                "generation_kwargs": {"max_gen_toks": 32},
                "metadata": {"wkv_mode": "fp32io16"},
            }
        },
        "results": {"race": {"acc,none": 0.5}},
        "n-samples": {"race": {"original": 1, "effective": 1}},
        "lm_eval_version": "0.4.13.dev0",
        "backend_commit": "f" * 40,
        "backend_version": "0.10.0",
        "torch_version": "2.11.0+cu130",
        "gpu": "NVIDIA RTX 4060",
    }
    samples = {
        "race": [
            {
                "doc_id": 0,
                "doc": {"question": "question"},
                "metrics": ["acc"],
                "acc": 0.0,
                "response_evidence": [
                    [
                        {
                            "input_token_ids": [1],
                            "raw_response": {"choices": [{"text": "answer"}]},
                            "post_processed_answer": "answer",
                        }
                    ]
                ],
            }
        ]
    }

    with pytest.raises(CONVERTER.ScoreboardError, match="complete raw response/token"):
        CONVERTER.build_lm_eval_publication(
            results,
            samples,
            publication={"enabled": True, "model_sha256": "e" * 64},
        )


def test_native_conversion_preserves_real_nested_lm_eval_shape() -> None:
    task_name = "rwkv7_g1i_1_5b_20260805_ctx16384_race"
    weight_sha256 = "e" * 64
    results = {
        "model_name": "rwkv7-g1i-1.5b-20260805-ctx16384",
        "config": {
            "model": "rwkv7-http",
            "model_args": {
                "model": "rwkv7-g1i-1.5b-20260805-ctx16384",
                "rwkv_prompt_template": "assistant",
                "rwkv_generation_prompt": "fake_think",
                "rwkv_sampling_mode": "profile",
                "num_concurrent": 5,
                "max_length": 16384,
            },
            "batch_size": "1",
            "gen_kwargs": {},
        },
        "configs": {
            task_name: {
                "task": task_name,
                "dataset_path": "EleutherAI/race",
                "dataset_name": "high",
                "test_split": "test",
                "output_type": "multiple_choice",
                "metric_list": [{"metric": "acc", "aggregation": "mean"}],
                "metadata": {
                    "version": 1.0,
                    "benchmark_name": "race",
                    "wkv_mode": "fp32io16",
                    "prompt_template": "assistant",
                },
            }
        },
        "results": {task_name: {"acc,none": 0.5, "acc_stderr,none": 0.1}},
        "n-samples": {task_name: {"original": 1, "effective": 1}},
        "lm_eval_version": "0.4.13.dev0",
        "backend_commit": "f" * 40,
        "backend_version": "0.10.0",
        "torch_version": "2.11.0+cu130",
        "gpu": "NVIDIA RTX 4060",
        "chat_template_sha": "a" * 64,
        "task_hashes": {task_name: "b" * 64},
    }
    choices = []
    evidence = []
    for index in range(4):
        choices.append([[str(-index - 1.0), "False"]])
        evidence.append(
            [
                {
                    "prompt": "prompt",
                    "input_token_ids": [10, index],
                    "output_token_ids": [20 + index],
                    "raw_response": {
                        "choices": [
                            {
                                "index": 0,
                                "text": "",
                                "logprobs": {"token_logprobs": [None, -0.1]},
                            }
                        ]
                    },
                    "post_processed_answer": "0",
                    "finish_reason": None,
                    "truncation": False,
                }
            ]
        )
    samples = {
        task_name: [
            {
                "doc_id": 0,
                "doc": {"question": "question", "options": ["a", "b", "c", "d"]},
                "target": "0",
                "arguments": {
                    f"gen_args_{index}": {"arg_0": "prompt", "arg_1": choice}
                    for index, choice in enumerate(("a", "b", "c", "d"))
                },
                "resps": choices,
                "filtered_resps": [[str(-index - 1.0), "False"] for index in range(4)],
                "filter": "none",
                "metrics": ["acc"],
                "acc": 1.0,
                "response_evidence": evidence,
            }
        ]
    }

    campaign_payload, task_payloads = CONVERTER.build_lm_eval_publication(
        results,
        samples,
        publication={
            "enabled": True,
            "model_sha256": weight_sha256,
            "model_revision": "20260805",
            "task_metadata": {
                "race": {
                    "selector": "race",
                    "module_family": "race",
                    "module": "lm_eval.tasks.race",
                    "languages": ["english"],
                    "upstream_tags": ["reading-comprehension"],
                }
            },
        },
    )

    descriptor = campaign_payload["expected_tasks"][0]
    assert descriptor["identity"] == f"{weight_sha256}:fp32io16:{task_name}"
    assert descriptor["benchmark"] == "race"
    payload = task_payloads[0]
    assert payload["task"] == descriptor
    assert payload["result_files"][0]["role"] == "metrics"
    assert payload["task_config"]["original_num_docs"] == 1
    assert payload["task_config"]["effective_num_docs"] == 1
    assert payload["environment"]["weight_sha256"] == weight_sha256
    assert payload["environment"]["wkv_mode"] == "fp32io16"
    assert payload["environment"]["gpu"] == "NVIDIA RTX 4060"
    assert payload["environment"]["dependency_versions"] == {
        "lm-eval": "0.4.13.dev0",
        "vllm": "0.10.0@" + "f" * 40,
        "torch": "2.11.0+cu130",
    }
    assert payload["sampling_config"]["temperature"] == 1.0
    assert payload["diagnostics"] == {
        "samples": 1,
        "completions": 4,
        "truncated": 0,
        "non_truncated": 4,
        "truncation_rate": 0.0,
        "turn_boundary_violations": 0,
        "turn_boundary_violation_rate": 0.0,
    }
    detail = payload["samples"][0]
    assert detail["document_index"] == 0
    assert detail["document"]["specific"]["lm_eval_document_index"] == 0
    assert [item["request_index"] for item in detail["model_response"]["evidence"]] == [
        0,
        1,
        2,
        3,
    ]
    assert detail["model_response"]["output_tokens"] == [[20], [21], [22], [23]]
    assert detail["model_response"]["evidence_complete"] is True


class FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self.body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> FakeResponse:  # noqa: PYI034
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_request_retries_transient_network_error(monkeypatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MODULE.URLError("temporary TLS failure")
        return FakeResponse({"status": "ok"})

    monkeypatch.setattr(MODULE, "urlopen", fake_urlopen)
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)
    client = MODULE.ScoreboardClient(
        base_url="https://eval.rwkv.rs/test",
        token="secret",  # noqa: S106
        timeout=12,
        retries=2,
        retry_delay=0.25,
    )

    assert client._request("GET", "/health") == {"status": "ok"}
    assert attempts == 2
    assert sleeps == [0.25]


def test_publish_uses_scoreboard_contract_transport_and_is_resumable(monkeypatch):
    requests: list[dict] = []
    campaign_id = "00000000-0000-0000-0000-000000000001"
    acknowledged: dict[str, str] = {}

    def fake_urlopen(request, timeout):
        body = request.data
        payload = None
        if body is not None:
            assert request.get_header("Content-type") == "application/json"
            assert request.get_header("Content-encoding") == "gzip"
            payload = json.loads(gzip.decompress(body))
        path = urlsplit(request.full_url).path
        requests.append(
            {
                "method": request.method,
                "url": request.full_url,
                "path": path,
                "payload": payload,
                "idempotency": request.get_header("Idempotency-key"),
                "timeout": timeout,
            }
        )
        if path.endswith("publication-preflight"):
            return FakeResponse(
                {
                    "status": "ready",
                    "schema_version": MODULE.CAMPAIGN_SCHEMA,
                    "sources": ["lm-eval-harness"],
                }
            )
        if path.endswith("evaluation-campaigns"):
            return FakeResponse(
                {
                    "campaign_id": campaign_id,
                    "action": "created",
                    "status": "incomplete",
                    "expected_task_count": 2,
                    "task_hashes": {},
                }
            )
        if path.endswith(campaign_id):
            return FakeResponse(
                {
                    "campaign_id": campaign_id,
                    "status": "incomplete",
                    "expected_task_count": 2,
                    "task_hashes": acknowledged,
                    "missing_tasks": [],
                }
            )
        if "/tasks/" in path:
            assert payload is not None
            identity = payload["task"]["identity"]
            digest = MODULE.content_digest(payload)
            acknowledged[identity] = digest
            return FakeResponse(
                {
                    "evaluation_id": identity,
                    "task_identity": identity,
                    "content_sha256": digest,
                    "action": "created",
                }
            )
        if path.endswith("/finalize"):
            return FakeResponse(
                {"campaign_id": campaign_id, "status": "complete", "task_count": 2}
            )
        raise AssertionError(f"unexpected request: {request.method} {path}")

    monkeypatch.setattr(MODULE, "urlopen", fake_urlopen)
    publication_token = "secret"  # noqa: S105
    client = MODULE.ScoreboardClient(
        base_url="https://eval.rwkv.rs/test", token=publication_token, timeout=12
    )
    campaign_payload = campaign()
    tasks = {
        identity: task(identity)
        for identity in ("a" * 64 + ":fp16:task", "a" * 64 + ":fp32io16:task")
    }

    first = MODULE.publish(
        client=client,
        campaign=campaign_payload,
        task_by_identity=tasks,
        expected_identities=list(tasks),
    )

    assert first["finalize"]["status"] == "complete"
    assert requests[0]["path"] == "/test/api/v1/evaluation-publication-preflight"
    campaign_request = next(
        item
        for item in requests
        if item["method"] == "POST" and item["path"].endswith("evaluation-campaigns")
    )
    assert campaign_request["idempotency"] == "campaign:" + campaign_payload["run_key"]
    task_requests = [item for item in requests if "/tasks/" in item["path"]]
    assert [item["payload"]["campaign_id"] for item in task_requests] == [
        campaign_id,
        campaign_id,
    ]
    assert all(item["idempotency"].startswith("publish:") for item in task_requests)

    requests.clear()
    second = MODULE.publish(
        client=client,
        campaign=campaign_payload,
        task_by_identity=tasks,
        expected_identities=list(tasks),
    )
    assert second["tasks"][0]["action"] == "unchanged"
    assert not any("/tasks/" in item["path"] for item in requests)


def test_native_lm_eval_publication_handles_race_and_drop_and_retains_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(MODULE.SCOREBOARD_MODEL_SHA256_ENV, "e" * 64)
    monkeypatch.setenv(MODULE.SCOREBOARD_MODEL_REVISION_ENV, "local-e2e-revision")
    results = {
        "config": {
            "model": "rwkv7-http",
            "model_args": {
                "model": "rwkv7-g1i-1.5b-20260805-ctx16384",
                "rwkv_prompt_template": "assistant",
                "rwkv_generation_prompt": "open_think",
                "rwkv_sampling_mode": "profile",
                "num_concurrent": 5,
                "max_length": 16384,
            },
        },
        "configs": {
            "race": {
                "task": "race",
                "dataset_path": "race",
                "test_split": "test",
                "output_type": "generate_until",
                "generation_kwargs": {"max_gen_toks": 32},
                "metadata": {"wkv_mode": "fp32io16"},
            },
            "drop": {
                "task": "drop",
                "dataset_path": "drop",
                "test_split": "validation",
                "output_type": "generate_until",
                "generation_kwargs": {"max_gen_toks": 32},
                "metadata": {"wkv_mode": "fp32io16"},
            },
        },
        "results": {
            "race": {"acc,none": 0.5},
            "drop": {"f1,none": 0.25},
        },
        "lm_eval_version": "0.4.13.dev0",
        "n-samples": {
            "race": {"original": 1, "effective": 1},
            "drop": {"original": 1, "effective": 1},
        },
        "backend_commit": "f" * 40,
        "backend_version": "0.10.0",
        "torch_version": "2.11.0+cu130",
        "gpu": "NVIDIA RTX 4060",
    }
    samples = {
        task_name: [
            {
                "doc_id": 0,
                "doc": {"question": task_name},
                "target": "answer",
                "arguments": ["prompt"],
                "resps": [["answer"]],
                "filtered_resps": ["answer"],
                "metrics": ["acc" if task_name == "race" else "f1"],
                "acc": 1.0 if task_name == "race" else None,
                "f1": 1.0 if task_name == "drop" else None,
                "response_evidence": [
                    [
                        {
                            "prompt": "prompt",
                            "input_token_ids": [1],
                            "output_token_ids": [2],
                            "raw_response": {"choices": [{"text": "answer"}]},
                            "post_processed_answer": "answer",
                        }
                    ]
                ],
            }
        ]
        for task_name in ("race", "drop")
    }

    status = MODULE.publish_lm_eval_evaluation(
        results,
        samples,
        output_dir=tmp_path,
        publication={
            "enabled": True,
            "token_env": "MISSING_TOKEN",
        },
    )

    assert status["evaluation"] == "complete"
    assert status["publication"] == "failed"
    assert status["uploaded"] is False
    assert "publication incomplete" in status["message"]
    campaign = json.loads(Path(status["campaign_path"]).read_text(encoding="utf-8"))
    assert campaign["schema_version"] == MODULE.LM_EVAL_CAMPAIGN_SCHEMA
    assert [item["benchmark"] for item in campaign["expected_tasks"]] == [
        "race",
        "drop",
    ]
    assert all(
        item["identity"] == f"{'e' * 64}:fp32io16:{item['task_name']}"
        for item in campaign["expected_tasks"]
    )
    raw_results = json.loads(
        Path(status["raw_results_path"]).read_text(encoding="utf-8")
    )
    assert raw_results["samples"]["race"][0]["response_evidence"][0][0][
        "output_token_ids"
    ] == [2]
