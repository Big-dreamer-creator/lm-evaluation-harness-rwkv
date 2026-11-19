import json
import os
import random
from pathlib import Path

import pytest

import lm_eval.evaluator as evaluator
from lm_eval.api.benchmark import BenchmarkSpec
from lm_eval.api.metrics import aggregate_benchmark_fallback
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


_TEST_DATA = [
    {"question": "What is 2+2?", "choices": ["1", "2", "3", "4"], "answer": 3},
    {"question": "What is 3+3?", "choices": ["4", "5", "6", "7"], "answer": 2},
    {"question": "What is 4+4?", "choices": ["6", "7", "8", "9"], "answer": 2},
    {"question": "What is 5+5?", "choices": ["8", "9", "10", "11"], "answer": 2},
]

_LEAF_TEMPLATE = """\
task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data_path}
output_type: multiple_choice
doc_to_text: "{{{{question}}}}"
doc_to_target: answer
doc_to_choice: choices
test_split: test
num_fewshot: {num_fewshot}
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""

_GROUP_YAML = """\
group: test_group
task:
  - group_task_fs0
  - group_task_fs1
"""


@pytest.fixture
def benchmark_fixture_task_manager(tmp_path):
    """A `group_fallback` benchmark (no `aggregate_metric_list`) with two leaves
    that share the `acc` metric -- built fresh per-test so the leaves can
    actually run (`build_all_requests`), unlike the load-only
    tests/test_configs/group.yaml fixture used elsewhere.
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    data_path = fixtures_dir / "test_data.json"
    data_path.write_text(json.dumps(_TEST_DATA), encoding="utf-8")

    (fixtures_dir / "task_fs0.yaml").write_text(
        _LEAF_TEMPLATE.format(
            task_name="group_task_fs0", data_path=str(data_path), num_fewshot=0
        ),
        encoding="utf-8",
    )
    (fixtures_dir / "task_fs1.yaml").write_text(
        _LEAF_TEMPLATE.format(
            task_name="group_task_fs1", data_path=str(data_path), num_fewshot=1
        ),
        encoding="utf-8",
    )
    (fixtures_dir / "group.yaml").write_text(_GROUP_YAML, encoding="utf-8")

    return TaskManager(include_path=str(fixtures_dir), include_defaults=False)


@pytest.fixture
def recording_callback():
    calls = []

    def callback(task_name, artifact_paths):
        calls.append({"task_name": task_name, "artifact_paths": artifact_paths})
        return {"publication": "created", "uploaded": True}

    callback.calls = calls
    return callback


def _benchmark_map():
    return {
        "test_group": BenchmarkSpec(
            name="test_group", field="reasoning", strategy="group_fallback"
        )
    }


def _run_test_group(
    task_manager, recording_callback, tmp_path, seed=0, checkpoint_store=None
):
    random.seed(seed)
    task_dict = task_manager.load(["test_group"])
    lm = DummyLM()
    evaluation_tracker = EvaluationTracker(output_path=str(tmp_path / "output"))
    task_callback_context = {
        "config": {"model": "dummy", "model_args": {}},
        "lm_eval_version": "test",
    }

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        checkpoint_store=checkpoint_store,
        task_callback=recording_callback,
        task_callback_context=task_callback_context,
        benchmark_map=_benchmark_map(),
    )
    return recording_callback.calls, evaluation_tracker


def test_leaf_tasks_never_published_only_benchmark_is(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    calls, _ = _run_test_group(
        benchmark_fixture_task_manager, recording_callback, tmp_path
    )

    assert len(calls) == 1
    assert calls[0]["task_name"] == "test_group"
    assert calls[0]["task_name"] not in ("group_task_fs0", "group_task_fs1")
    results_doc = json.loads(
        Path(calls[0]["artifact_paths"]["results"]).read_text("utf-8")
    )
    assert results_doc["benchmark"] == {
        "name": "test_group",
        "field": "reasoning",
    }
    metadata_doc = json.loads(
        Path(calls[0]["artifact_paths"]["metadata"]).read_text("utf-8")
    )
    assert metadata_doc["benchmark"] == results_doc["benchmark"]


def test_simple_evaluate_writes_run_metadata_into_the_published_artifact(
    benchmark_fixture_task_manager, recording_callback, tmp_path, monkeypatch
):
    """The published results.json must carry the run metadata.

    Scoreboard publication reads ``config.metadata.weight_revision`` from the
    artifact to derive the served weight's ``ctx_len``; the API-facing
    ``model_name`` alone is only a short serving alias. ``simple_evaluate``
    builds the artifact's run config, so it has to copy its ``metadata``
    argument there -- both into the live callback context and into the final
    results document.
    """
    # Prototype: RWKVLoglikelihoodResponse; this fixture must retain its actual
    # dummy request context instead of asking publication to reconstruct it.
    original = DummyLM.loglikelihood

    def evidenced_loglikelihood(self, requests, **kwargs):
        from lm_eval.models.rwkv_choice import RWKVLoglikelihoodResponse
        return [
            RWKVLoglikelihoodResponse(ll, greedy, prompt=request.args[0], text=request.args[1],
                raw_response={"choices": [{"text": request.args[1]}]})
            for request, (ll, greedy) in zip(requests, original(self, requests, **kwargs), strict=True)
        ]

    monkeypatch.setattr(DummyLM, "loglikelihood", evidenced_loglikelihood)
    metadata = {
        "model_name": "rwkv7-g1j-1.5b",
        "weight_revision": "rwkv7-g1j-1.5b-20260831-ctx16384",
    }

    results = evaluator.simple_evaluate(
        model="dummy",
        model_args=(
            "model=rwkv7-g1j-1.5b,temperature=0.96,top_p=0.76,top_k=32,"
            "presence_penalty=1.0,frequency_penalty=0.1,penalty_decay=0.988,seed=42"
        ),
        tasks=["test_group"],
        limit=1,
        bootstrap_iters=0,
        task_manager=benchmark_fixture_task_manager,
        evaluation_tracker=EvaluationTracker(output_path=str(tmp_path / "output")),
        metadata=metadata,
        task_callback=recording_callback,
        benchmark_map=_benchmark_map(),
    )

    assert results is not None
    assert results["config"]["metadata"] == metadata
    (call,) = recording_callback.calls
    results_doc = json.loads(Path(call["artifact_paths"]["results"]).read_text("utf-8"))
    assert results_doc["config"]["metadata"] == metadata

    # The artifact is what the scoreboard callback reads. A short API alias
    # plus metadata.weight_revision must be enough to build the upload payload.
    from lm_eval.loggers.scoreboard import PublicationConfig, build_score

    _, score = build_score(
        results_doc,
        PublicationConfig.from_mapping(
            {
                "base_url": "https://scoreboard.example",
                "token_env": "SCOREBOARD_PUBLICATION_TOKEN",
                "cot_mode": "CoT",
                "tasks": {"test_group": {"field": "reasoning"}},
                "timeout": 60,
                "control_timeout": 30,
                "retries": 0,
                "retry_delay": 0,
                "dry_run": True,
            }
        ),
    )
    assert score["model"]["ctx_len"] == 16384


def test_published_aggregate_matches_direct_fallback_computation(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    calls, _ = _run_test_group(
        benchmark_fixture_task_manager, recording_callback, tmp_path
    )

    artifact_paths = calls[0]["artifact_paths"]
    results_path = Path(artifact_paths["results"])
    results_doc = json.loads(results_path.read_text("utf-8"))
    published_aggregate = results_doc["results"]["test_group"]

    leaf_artifact_paths = results_doc["leaf_artifact_paths"]
    leaf_metrics = {}
    for leaf_name in ("group_task_fs0", "group_task_fs1"):
        leaf_results_path = Path(leaf_artifact_paths[leaf_name]["results"])
        leaf_doc = json.loads(leaf_results_path.read_text("utf-8"))
        leaf_metrics[leaf_name] = leaf_doc["results"][leaf_name]

    expected_aggregate, expected_primary = aggregate_benchmark_fallback(
        "test_group", ["group_task_fs0", "group_task_fs1"], leaf_metrics
    )

    assert published_aggregate == expected_aggregate
    assert results_doc["primary_metric_hint"] == expected_primary
    assert expected_primary == "acc"


def test_avg_at_k_aggregation_does_not_overwrite_score_with_stderr():
    metrics = {
        "leaf": {
            "acc,none": 0.25,
            "acc_stderr,none": 0.04,
            "acc_avg@2,none": 0.5,
            "acc_avg@2_stderr,none": 0.02,
        }
    }

    averaged = evaluator._apply_avg_at_k(metrics)

    assert averaged["leaf"]["acc,none"] == 0.5
    assert averaged["leaf"]["acc_stderr,none"] == 0.02


def test_resume_with_all_leaves_precompleted_still_publishes_once(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    from lm_eval.api.checkpoint import CheckpointStore

    checkpoint_store = CheckpointStore(str(tmp_path / "checkpoint.sqlite"), "run-1")

    calls, _ = _run_test_group(
        benchmark_fixture_task_manager,
        recording_callback,
        tmp_path,
        checkpoint_store=checkpoint_store,
    )

    assert len(calls) == 1
    assert calls[0]["task_name"] == "test_group"
    benchmark_row = checkpoint_store.get_benchmark("test_group")
    assert benchmark_row["publish_status"] == "succeeded"


def test_already_published_benchmark_skips_callback_but_updates_checkpoint(
    benchmark_fixture_task_manager, recording_callback, tmp_path
):
    from lm_eval.api.checkpoint import CheckpointStore, compute_task_fingerprint

    checkpoint_store = CheckpointStore(str(tmp_path / "checkpoint.sqlite"), "run-1")
    fingerprint = compute_task_fingerprint(
        {
            "leaves": sorted(["group_task_fs0", "group_task_fs1"]),
            "strategy": "group_fallback",
        }
    )
    checkpoint_store.start_benchmark("test_group", fingerprint)
    checkpoint_store.mark_benchmark_completed("test_group", None, None)
    checkpoint_store.set_publish_status("test_group", "succeeded")

    calls, _ = _run_test_group(
        benchmark_fixture_task_manager,
        recording_callback,
        tmp_path,
        checkpoint_store=checkpoint_store,
    )

    assert len(calls) == 0
    benchmark_row = checkpoint_store.get_benchmark("test_group")
    assert benchmark_row["publish_status"] == "succeeded"


def test_streaming_windows_are_bounded_and_samples_are_persisted(
    benchmark_fixture_task_manager, recording_callback, tmp_path, monkeypatch
):
    """RWKV campaigns keep one request window, not every prompt, in memory."""
    from lm_eval.api.task import Task

    build_windows: list[tuple[str, int]] = []
    original_build = Task.build_requests_for_documents
    original_save = EvaluationTracker.save_results_samples
    saved_windows = []

    def recording_save(self, task_name, samples, **kwargs):
        saved_windows.append((task_name, len(samples), kwargs.get("append", False)))
        return original_save(self, task_name, samples, **kwargs)

    monkeypatch.setattr(EvaluationTracker, "save_results_samples", recording_save)

    def recording_build(self, documents, **kwargs):
        documents = list(documents)
        build_windows.append((self.config.task, len(documents)))
        return original_build(self, documents, **kwargs)

    monkeypatch.setattr(Task, "build_requests_for_documents", recording_build)

    task_dict = benchmark_fixture_task_manager.load(["test_group"])
    lm = DummyLM()
    lm.STREAM_EVALUATION_CHUNKS = True
    lm._concurrent = 1
    tracker = EvaluationTracker(output_path=str(tmp_path / "streamed"))

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        log_samples=True,
        evaluation_tracker=tracker,
        task_callback=recording_callback,
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_benchmark_map(),
    )

    # Four choices per doc and one concurrent request -> one doc per window.
    assert build_windows
    assert {size for _, size in build_windows} == {1}
    assert len(saved_windows) == 8
    assert all(size == 1 for _, size, _ in saved_windows)
    assert [
        append for name, _, append in saved_windows if name == "group_task_fs0"
    ] == [False, True, True, True]
    assert all(not task._instances for task in task_dict["tasks"].values())
    assert "samples" not in results
    assert tracker.saved_sample_tasks >= {"group_task_fs0", "group_task_fs1"}
    for task_name in ("group_task_fs0", "group_task_fs1"):
        [sample_path] = (tmp_path / "streamed").glob(
            f"*/samples_{task_name}_{tracker.date_id}.jsonl"
        )
        assert len(sample_path.read_text("utf-8").splitlines()) == 4

    assert results["n-samples"]["group_task_fs0"] == {"original": 4, "effective": 4}
    assert results["n-samples"]["group_task_fs1"] == {"original": 4, "effective": 4}
    assert not list((tmp_path / "streamed").rglob("*.partial"))


def test_streaming_windows_respect_the_prompt_byte_budget(
    benchmark_fixture_task_manager, recording_callback, tmp_path, monkeypatch
):
    """Long prompts must not fill the whole endpoint concurrency pool."""
    from lm_eval.api.task import Task

    build_windows: list[tuple[str, int]] = []
    original_build = Task.build_requests_for_documents

    def recording_build(self, documents, **kwargs):
        documents = list(documents)
        build_windows.append((self.config.task, len(documents)))
        return original_build(self, documents, **kwargs)

    monkeypatch.setattr(Task, "build_requests_for_documents", recording_build)

    task_dict = benchmark_fixture_task_manager.load(["test_group"])
    lm = DummyLM()
    lm.STREAM_EVALUATION_CHUNKS = True
    # High endpoint concurrency would otherwise allow two docs, but the byte
    # ceiling permits only one before the window must be dispatched.
    lm._concurrent = 8
    lm.STREAM_WINDOW_MAX_PROMPT_BYTES = 1
    tracker = EvaluationTracker(output_path=str(tmp_path / "byte-budget"))

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        log_samples=True,
        evaluation_tracker=tracker,
        task_callback=recording_callback,
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_benchmark_map(),
    )

    assert build_windows
    assert {size for _, size in build_windows} == {1}


def test_streaming_results_match_native_evaluation(
    benchmark_fixture_task_manager, tmp_path
):
    runs = []
    for streaming in (False, True):
        random.seed(42)
        lm = DummyLM()
        lm.STREAM_EVALUATION_CHUNKS = streaming
        lm._concurrent = 1
        lm.loglikelihood = lambda requests: [
            (-float(len(str(request.args[1]))), False) for request in requests
        ]
        task_dict = benchmark_fixture_task_manager.load(["test_group"])
        for task in task_dict["tasks"].values():
            task.set_fewshot_seed(42)
        tracker = EvaluationTracker(output_path=str(tmp_path / str(streaming)))
        results = evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
            bootstrap_iters=0,
            evaluation_tracker=tracker,
            task_callback=lambda **kwargs: None,
            task_callback_context={"config": {"model": "dummy", "model_args": {}}},
            benchmark_map=_benchmark_map(),
        )
        samples = {}
        for task_name in ("group_task_fs0", "group_task_fs1"):
            [sample_path] = (tmp_path / str(streaming)).glob(
                f"*/samples_{task_name}_{tracker.date_id}.jsonl"
            )
            samples[task_name] = [
                json.loads(line) for line in sample_path.read_text("utf-8").splitlines()
            ]
        runs.append((results["results"], results["n-samples"], samples))
    assert runs[0] == runs[1]


def test_streaming_stops_on_sample_write_failure(
    benchmark_fixture_task_manager, recording_callback, tmp_path, monkeypatch
):
    def failed_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(EvaluationTracker, "save_results_samples", failed_save)
    lm = DummyLM()
    lm.STREAM_EVALUATION_CHUNKS = True
    lm._concurrent = 1
    with pytest.raises(OSError, match="disk full"):
        evaluator.evaluate(
            lm=lm,
            task_dict=benchmark_fixture_task_manager.load(["test_group"]),
            bootstrap_iters=0,
            evaluation_tracker=EvaluationTracker(output_path=str(tmp_path)),
            task_callback=recording_callback,
            benchmark_map=_benchmark_map(),
        )
    assert not recording_callback.calls


def test_streaming_reduces_long_prompt_peak_memory(
    benchmark_fixture_task_manager, tmp_path
):
    import gc
    import tracemalloc

    from datasets import Dataset

    peaks = {}
    for streaming in (False, True):
        task_dict = benchmark_fixture_task_manager.load(["group_task_fs0"])
        task = task_dict["tasks"]["group_task_fs0"]
        task.dataset["test"] = Dataset.from_list([
            {**_TEST_DATA[0], "question": f"{doc_id}: " + "context " * 4096}
            for doc_id in range(256)
        ])
        lm = DummyLM()
        lm.STREAM_EVALUATION_CHUNKS = streaming
        lm._concurrent = 16
        tracker = EvaluationTracker(output_path=str(tmp_path / str(streaming)))
        gc.collect()
        tracemalloc.start()
        try:
            evaluator.evaluate(
                lm=lm,
                task_dict=task_dict,
                bootstrap_iters=0,
                evaluation_tracker=tracker,
                task_callback=lambda **kwargs: None,
                task_callback_context={"config": {"model": "dummy", "model_args": {}}},
            )
            peaks[streaming] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    print(f"Long-prompt Python peak bytes: {peaks}")
    assert peaks[True] < peaks[False] / 2


def _sample_row(doc_id: int) -> dict:
    return {
        "doc_id": doc_id,
        "doc": {"question": f"question {doc_id}"},
        "target": 0,
        "arguments": [("prompt", "continuation")],
        "resps": [["response"]],
        "filtered_resps": [["response"]],
        "filter": "none",
        "metrics": ["acc"],
        "doc_hash": f"doc-{doc_id}",
        "prompt_hash": f"prompt-{doc_id}",
        "target_hash": "target",
        "acc": 1.0,
    }


def _artifact_doc(task_name: str) -> dict:
    return {
        "config": {"model": "dummy", "model_args": {"model": "dummy"}},
        "results": {task_name: {"acc,none": 1.0}},
        "configs": {},
        "versions": {},
        "lm_eval_version": "test",
    }


def test_deferred_sample_windows_publish_one_file_at_task_end(tmp_path):
    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.date_id = "run-1"

    first_path = tracker.save_results_samples(
        "leaf", [_sample_row(0)], defer_finalize=True
    )
    assert first_path is not None
    first_path = Path(first_path)
    assert first_path.name == "samples_leaf_run-1.jsonl"
    assert not first_path.exists()
    [partial] = list(
        first_path.parent.glob(".samples_leaf_run-1.jsonl.*.partial")
    )
    assert [json.loads(line)["doc_id"] for line in partial.read_text().splitlines()] == [0]

    second_path = tracker.save_results_samples(
        "leaf", [_sample_row(1)], append=True, defer_finalize=True
    )
    assert second_path == str(first_path)
    assert not first_path.exists()

    artifacts = tracker.save_task_artifacts(
        "leaf", _artifact_doc("leaf"), [], samples_path=str(first_path)
    )

    assert artifacts is not None
    assert Path(artifacts["samples"]) == first_path
    assert first_path.exists()
    assert [
        json.loads(line)["doc_id"] for line in first_path.read_text().splitlines()
    ] == [0, 1]
    assert not list(first_path.parent.glob("*.partial"))


def test_appending_samples_without_a_deferred_file_is_rejected(tmp_path):
    tracker = EvaluationTracker(output_path=str(tmp_path))

    with pytest.raises(ValueError, match="without starting a deferred sample file"):
        tracker.save_results_samples("leaf", [_sample_row(0)], append=True)


def test_restarted_run_reuses_the_same_sample_name_without_duplicates(tmp_path):
    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.date_id = "run-1"
    samples_path = tracker.save_results_samples(
        "leaf", [_sample_row(0)], defer_finalize=True
    )
    tracker.save_task_artifacts(
        "leaf", _artifact_doc("leaf"), [], samples_path=samples_path
    )
    samples_path = Path(samples_path)

    stale_partial = (
        samples_path.parent / ".samples_leaf_run-1.jsonl.999999.partial"
    )
    stale_partial.write_text("stale\n", encoding="utf-8")

    restarted = EvaluationTracker(output_path=str(tmp_path))
    restarted.date_id = "run-1"
    resumed_path = restarted.save_results_samples(
        "leaf", [_sample_row(1)], defer_finalize=True
    )
    assert resumed_path == str(samples_path)
    restarted.save_task_artifacts(
        "leaf", _artifact_doc("leaf"), [], samples_path=resumed_path
    )

    assert list(samples_path.parent.glob("samples_leaf_*.jsonl")) == [samples_path]
    assert [
        json.loads(line)["doc_id"] for line in samples_path.read_text().splitlines()
    ] == [1]
    assert not list(samples_path.parent.glob("*.partial"))


def test_benchmark_aggregate_reuses_combined_file_until_a_leaf_changes(tmp_path):
    tracker = EvaluationTracker(output_path=str(tmp_path / "output"))
    tracker.date_id = "run-1"
    leaf_artifact_paths = {}
    for leaf in ("leaf_a", "leaf_b"):
        samples_path = tracker.save_results_samples(
            leaf, [_sample_row(0)], defer_finalize=True
        )
        leaf_artifact_paths[leaf] = tracker.save_task_artifacts(
            leaf, _artifact_doc(leaf), [], samples_path=samples_path
        )
    results_doc = {
        **_artifact_doc("bench"),
        "leaf_artifact_paths": leaf_artifact_paths,
    }

    first = tracker.save_benchmark_artifacts("bench", results_doc)
    combined = Path(first["samples"])
    assert len(combined.read_text().splitlines()) == 2

    old_ns = 1_000_000_000 * 10**9
    os.utime(combined, ns=(old_ns, old_ns))
    second = tracker.save_benchmark_artifacts("bench", results_doc)
    assert Path(second["samples"]) == combined
    assert combined.stat().st_mtime_ns == old_ns

    stale_partial = combined.parent / f".{combined.name}.999999.partial"
    stale_partial.write_text("stale\n", encoding="utf-8")
    leaf_samples = Path(leaf_artifact_paths["leaf_a"]["samples"])
    with leaf_samples.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_sample_row(1)) + "\n")
    third = tracker.save_benchmark_artifacts("bench", results_doc)
    assert Path(third["samples"]) == combined
    assert combined.stat().st_mtime_ns != old_ns
    assert len(combined.read_text().splitlines()) == 3
    assert not list(combined.parent.glob("*.partial"))
