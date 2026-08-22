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
