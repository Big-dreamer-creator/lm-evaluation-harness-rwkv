from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/upload_scoreboard.py"
SPEC = importlib.util.spec_from_file_location("upload_scoreboard", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def campaign() -> dict:
    return {
        "schema_version": MODULE.CAMPAIGN_SCHEMA,
        "run_key": "a" * 64,
        "config_digest": "b" * 64,
        "registry_digest": "c" * 64,
        "eval_contract_digest": "d" * 64,
        "lighteval_version": MODULE.LIGHTEVAL_VERSION,
        "configured_selectors": ["task"],
        "resolved_selectors": ["task"],
        "skipped_selectors": [],
        "expected_tasks": [
            {"identity": "weight:fp16:task"},
            {"identity": "weight:fp32io16:task"},
        ],
    }


def task(identity: str) -> dict:
    return {
        "schema_version": MODULE.TASK_SCHEMA,
        "campaign_id": "assigned-by-uploader",
        "task": {"identity": identity},
    }


def write_inputs(tmp_path: Path) -> tuple[Path, list[Path]]:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign()), encoding="utf-8")
    task_paths = []
    for index, identity in enumerate(("weight:fp16:task", "weight:fp32io16:task")):
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
    task_path.write_text(json.dumps(task("weight:fp16:task")), encoding="utf-8")

    with pytest.raises(MODULE.ScoreboardError, match="lighteval-campaign-v3"):
        MODULE.load_publication_inputs(raw_result, [task_path])


def test_load_inputs_requires_exact_task_set(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign()), encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task("weight:fp16:task")), encoding="utf-8")

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
            **MODULE._TARGET_SAMPLING,
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
        "schema_version": MODULE.PRODUCER_CAMPAIGN_SCHEMA,
        "publication_contract": "rwkv-producer-v1",
        "scoreboard_upload_ready": False,
        "scoreboard_compatible": scoreboard_compatible,
        "run_key": "a" * 64,
        "config_digest": "b" * 64,
        "registry_digest": "c" * 64,
        "eval_contract_digest": "d" * 64,
        "lighteval_version": MODULE.LIGHTEVAL_VERSION,
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
                "schema_version": MODULE.PRODUCER_TASK_SCHEMA,
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

    campaign_payload, task_payloads = MODULE.convert_producer_publication(
        producer_campaign, producer_tasks
    )

    assert campaign_payload["schema_version"] == MODULE.CAMPAIGN_SCHEMA
    assert all(
        payload["schema_version"] == MODULE.TASK_SCHEMA for payload in task_payloads
    )
    assert {
        payload["task"]["identity"] for payload in task_payloads
    } == {f"{'e' * 64}:fp16:moral_stories", f"{'e' * 64}:fp32io16:moral_stories"}
    detail = task_payloads[0]["details"][0]
    assert detail["model_response"]["logprobs"] == [-0.1]
    assert detail["model_response"]["output_tokens"] == [[11]]
    assert detail["doc"]["specific"]["helicopter_document_index"] == 0


def test_producer_conversion_rejects_non_publishable_sampling_contract() -> None:
    campaign_payload, task_payloads = _producer_inputs(scoreboard_compatible=False)
    with pytest.raises(MODULE.ScoreboardError, match="scoreboard-compatible"):
        MODULE.convert_producer_publication(campaign_payload, task_payloads)


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
                    "publisher_principal": "test",
                    "schema_version": MODULE.CAMPAIGN_SCHEMA,
                    "lighteval_version": MODULE.LIGHTEVAL_VERSION,
                }
            )
        if path.endswith("evaluation-campaigns"):
            return FakeResponse(
                {
                    "campaign_id": campaign_id,
                    "disposition": "created",
                    "status": "incomplete",
                    "expected_task_count": 2,
                    "acknowledged_task_digests": {},
                }
            )
        if path.endswith(campaign_id):
            return FakeResponse(
                {
                    "campaign_id": campaign_id,
                    "status": "incomplete",
                    "expected_task_count": 2,
                    "acknowledged_task_digests": acknowledged,
                    "missing_task_identities": [],
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
                    "content_digest": digest,
                    "disposition": "created",
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
        for identity in ("weight:fp16:task", "weight:fp32io16:task")
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
    assert campaign_request["idempotency"] == "campaign:" + "a" * 64
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
    assert second["tasks"][0]["disposition"] == "unchanged"
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
            "model_args": {"model": "rwkv7-g1i-1.5b-20260805-ctx16384"},
        },
        "configs": {
            "race": {"task": "race", "dataset_path": "race", "test_split": "test"},
            "drop": {"task": "drop", "dataset_path": "drop", "test_split": "validation"},
        },
        "results": {
            "race": {"acc,none": 0.5},
            "drop": {"f1,none": 0.25},
        },
        "lm_eval_version": "0.4.13.dev0",
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
                    {
                        "prompt": "prompt",
                        "input_token_ids": [1],
                        "output_token_ids": [2],
                        "raw_response": {"choices": [{"text": "answer"}]},
                        "post_processed_answer": "answer",
                    }
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
    assert campaign["model"] == {
        "name": "rwkv7-g1i-1.5b-20260805-ctx16384",
        "revision": "local-e2e-revision",
        "sha256": "e" * 64,
    }
    assert [item["task_name"] for item in campaign["expected_tasks"]] == ["race", "drop"]
    raw_results = json.loads(Path(status["raw_results_path"]).read_text(encoding="utf-8"))
    assert raw_results["samples"]["race"][0]["response_evidence"][0]["output_token_ids"] == [2]
