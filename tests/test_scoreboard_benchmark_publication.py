import json
import random
from pathlib import Path

import pytest

import lm_eval.evaluator as evaluator
from lm_eval.api.benchmark import BenchmarkSpec
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.loggers.scoreboard import (
    PublicationConfig,
    _build_leaf_publication,
    build_benchmark_publication,
)
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
num_fewshot: 0
metric_list:
  - metric: {metric}
    aggregation: mean
    higher_is_better: true
metadata:
  field: test-field
"""

_MODEL_SHA256 = "a" * 64


def _write_data_and_leaves(fixtures_dir: Path, leaves: list[tuple[str, str]]) -> Path:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    data_path = fixtures_dir / "test_data.json"
    data_path.write_text(json.dumps(_TEST_DATA), encoding="utf-8")
    for task_name, metric in leaves:
        (fixtures_dir / f"{task_name}.yaml").write_text(
            _LEAF_TEMPLATE.format(
                task_name=task_name, data_path=str(data_path), metric=metric
            ),
            encoding="utf-8",
        )
    return data_path


def _publish_config(tasks: dict[str, dict] | None = None) -> PublicationConfig:
    return PublicationConfig(
        base_url="https://scoreboard.example",
        token_env="SCOREBOARD_PUBLICATION_TOKEN",
        model_sha256=_MODEL_SHA256,
        tasks=tasks or {},
        timeout=60,
        control_timeout=30,
        retries=0,
        retry_delay=0,
        rerun_reason=None,
        dry_run=True,
    )


def _run_benchmark(
    fixtures_dir: Path,
    group_yaml: str,
    benchmark_name: str,
    strategy: str,
    tmp_path: Path,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Runs a real `evaluate()` pass and returns (results_doc, metadata_doc)
    for the aggregated benchmark, as written by `save_benchmark_artifacts`
    and read back by `ScoreboardTaskCallback` in production.
    """
    (fixtures_dir / "group.yaml").write_text(group_yaml, encoding="utf-8")
    task_manager = TaskManager(include_path=str(fixtures_dir), include_defaults=False)
    task_dict = task_manager.load([benchmark_name])

    random.seed(seed)
    lm = DummyLM()
    evaluation_tracker = EvaluationTracker(output_path=str(tmp_path / "output"))

    captured: dict = {}

    def callback(task_name, artifact_paths):
        captured["task_name"] = task_name
        captured["artifact_paths"] = artifact_paths

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        task_callback=callback,
        task_callback_context={
            "config": {"model": "dummy", "model_args": {}},
            "lm_eval_version": "test",
        },
        benchmark_map={
            benchmark_name: BenchmarkSpec(name=benchmark_name, strategy=strategy)
        },
    )

    results_doc = json.loads(
        Path(captured["artifact_paths"]["results"]).read_text("utf-8")
    )
    metadata_doc = json.loads(
        Path(captured["artifact_paths"]["metadata"]).read_text("utf-8")
    )
    return results_doc, metadata_doc


def test_shared_metric_group_fallback_produces_one_publication_with_real_primary(
    tmp_path,
):
    fixtures_dir = tmp_path / "fixtures_shared"
    _write_data_and_leaves(
        fixtures_dir, [("shared_leaf_a", "acc"), ("shared_leaf_b", "acc")]
    )
    group_yaml = "group: bench_shared\ntask:\n  - shared_leaf_a\n  - shared_leaf_b\n"
    results_doc, metadata_doc = _run_benchmark(
        fixtures_dir, group_yaml, "bench_shared", "group_fallback", tmp_path
    )

    config = _publish_config({"bench_shared": {"field": "test-field"}})
    campaign, publication = build_benchmark_publication(
        results_doc, metadata_doc, config
    )

    assert publication["primary_metric"] == "acc,none"
    assert "acc,none" in publication["metrics"]
    assert publication["task"]["task_name"] == "bench_shared"
    assert publication["task_config"]["original_num_docs"] == 8
    assert len(publication["samples"]) > 0
    assert campaign["configured_benchmarks"] == ["bench_shared"]


def test_no_shared_metric_group_fallback_does_not_fabricate_primary(tmp_path):
    fixtures_dir = tmp_path / "fixtures_disjoint"
    _write_data_and_leaves(
        fixtures_dir, [("disjoint_leaf_a", "acc"), ("disjoint_leaf_b", "acc_norm")]
    )
    group_yaml = "group: bench_disjoint\ntask:\n  - disjoint_leaf_a\n  - disjoint_leaf_b\n"
    results_doc, metadata_doc = _run_benchmark(
        fixtures_dir, group_yaml, "bench_disjoint", "group_fallback", tmp_path
    )

    assert results_doc["primary_metric_hint"] is None

    config = _publish_config({"bench_disjoint": {"field": "test-field"}})
    campaign, publication = build_benchmark_publication(
        results_doc, metadata_doc, config
    )

    assert publication["primary_metric"] is None
    assert len(publication["samples"]) > 0
    assert publication["task"]["task_name"] == "bench_disjoint"


def test_group_aggregate_primary_metric_comes_from_hint(tmp_path):
    fixtures_dir = tmp_path / "fixtures_aggregate"
    _write_data_and_leaves(
        fixtures_dir, [("agg_leaf_a", "acc"), ("agg_leaf_b", "acc")]
    )
    group_yaml = (
        "group: bench_aggregate\n"
        "task:\n  - agg_leaf_a\n  - agg_leaf_b\n"
        "aggregate_metric_list:\n"
        "  - aggregation: mean\n    metric: acc\n    weight_by_size: true\n"
    )
    results_doc, metadata_doc = _run_benchmark(
        fixtures_dir, group_yaml, "bench_aggregate", "group_aggregate", tmp_path
    )

    assert results_doc["primary_metric_hint"] == "acc"

    config = _publish_config({"bench_aggregate": {"field": "test-field"}})
    campaign, publication = build_benchmark_publication(
        results_doc, metadata_doc, config
    )

    assert publication["primary_metric"] == "acc,none"


def test_standalone_matches_leaf_publication_directly(tmp_path):
    fixtures_dir = tmp_path / "fixtures_standalone"
    _write_data_and_leaves(fixtures_dir, [("solo_leaf", "acc")])
    group_yaml = "group: unused\ntask:\n  - solo_leaf\n"
    (fixtures_dir / "group.yaml").write_text(group_yaml, encoding="utf-8")
    task_manager = TaskManager(include_path=str(fixtures_dir), include_defaults=False)
    task_dict = task_manager.load(["solo_leaf"])

    random.seed(0)
    lm = DummyLM()
    evaluation_tracker = EvaluationTracker(output_path=str(tmp_path / "output"))

    captured: dict = {}

    def callback(task_name, artifact_paths):
        captured["task_name"] = task_name
        captured["artifact_paths"] = artifact_paths

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        task_callback=callback,
        task_callback_context={
            "config": {"model": "dummy", "model_args": {}},
            "lm_eval_version": "test",
        },
        benchmark_map={
            "solo_leaf": BenchmarkSpec(
                name="solo_leaf", strategy="standalone", leaf_task_names=("solo_leaf",)
            )
        },
    )

    assert captured["task_name"] == "solo_leaf"
    results_doc = json.loads(
        Path(captured["artifact_paths"]["results"]).read_text("utf-8")
    )
    metadata_doc = json.loads(
        Path(captured["artifact_paths"]["metadata"]).read_text("utf-8")
    )

    config = _publish_config({"solo_leaf": {"field": "test-field"}})
    via_benchmark = build_benchmark_publication(results_doc, metadata_doc, config)

    leaf_samples_path = Path(
        results_doc["leaf_artifact_paths"]["solo_leaf"]["samples"]
    )
    via_leaf_direct = _build_leaf_publication(
        results_doc, metadata_doc, leaf_samples_path, "solo_leaf", config
    )

    assert via_benchmark == via_leaf_direct
