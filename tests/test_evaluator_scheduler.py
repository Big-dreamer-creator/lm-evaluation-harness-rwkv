"""Integration tests for the worker-thread-per-benchmark concurrent scheduler
in ``evaluator.py``'s checkpoint/task_callback branch (feature 4 of the
multi-replica-pool + scheduler plan).
"""

import json
from pathlib import Path

import lm_eval.evaluator as evaluator
from lm_eval.api.benchmark import BenchmarkSpec
from lm_eval.api.checkpoint import CheckpointStore, compute_config_fingerprint
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


_DATA_BY_SIZE = {
    2: [
        {"question": "1+1?", "choices": ["1", "2", "3"], "answer": 1},
        {"question": "2+2?", "choices": ["3", "4", "5"], "answer": 1},
    ],
    4: [
        {"question": "1+1?", "choices": ["1", "2", "3"], "answer": 1},
        {"question": "2+2?", "choices": ["3", "4", "5"], "answer": 1},
        {"question": "3+3?", "choices": ["5", "6", "7"], "answer": 1},
        {"question": "4+4?", "choices": ["7", "8", "9"], "answer": 1},
    ],
}

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
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""


def _write_task(fixtures_dir: Path, task_name: str, num_docs: int) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    data_path = fixtures_dir / f"{task_name}_data.json"
    data_path.write_text(json.dumps(_DATA_BY_SIZE[num_docs]), encoding="utf-8")
    (fixtures_dir / f"{task_name}.yaml").write_text(
        _LEAF_TEMPLATE.format(task_name=task_name, data_path=str(data_path)),
        encoding="utf-8",
    )


def _load_tasks(tmp_path: Path, sized_tasks: dict[str, int]) -> dict:
    fixtures_dir = tmp_path / "fixtures"
    for task_name, num_docs in sized_tasks.items():
        _write_task(fixtures_dir, task_name, num_docs)
    task_manager = TaskManager(include_path=str(fixtures_dir), include_defaults=False)
    return task_manager.load(list(sized_tasks))


def _standalone_benchmark_map(task_names: list) -> dict:
    return {
        task_name: BenchmarkSpec(
            name=task_name, strategy="standalone", leaf_task_names=(task_name,)
        )
        for task_name in task_names
    }


def test_concurrent_scheduling_publishes_every_task_exactly_once(tmp_path):
    task_dict = _load_tasks(
        tmp_path, {"solo_small_a": 2, "solo_small_b": 2, "solo_big": 4}
    )
    lm = DummyLM()

    calls = []

    def callback(task_name, artifact_paths):
        calls.append(task_name)

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        task_callback=callback,
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_standalone_benchmark_map(
            ["solo_small_a", "solo_small_b", "solo_big"]
        ),
        max_concurrent_benchmarks=2,
        max_concurrent_benchmarks_explicit=True,
    )

    assert sorted(calls) == ["solo_big", "solo_small_a", "solo_small_b"]


def test_max_concurrent_benchmarks_one_preserves_original_order(tmp_path):
    task_dict = _load_tasks(
        tmp_path, {"seq_first": 4, "seq_second": 2, "seq_third": 2}
    )
    lm = DummyLM()

    calls = []

    def callback(task_name, artifact_paths):
        calls.append(task_name)

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        task_callback=callback,
        task_callback_context={"config": {"model": "dummy", "model_args": {}}},
        benchmark_map=_standalone_benchmark_map(
            ["seq_first", "seq_second", "seq_third"]
        ),
        max_concurrent_benchmarks=1,
        max_concurrent_benchmarks_explicit=True,
    )

    # Original task_dict insertion order must be preserved exactly when the
    # scheduler is explicitly pinned back to 1 -- this is the "find back old
    # behavior" escape hatch.
    assert calls == ["seq_first", "seq_second", "seq_third"]


def test_concurrent_scheduling_marks_all_benchmarks_completed_in_checkpoint(tmp_path):
    task_dict = _load_tasks(tmp_path, {"ckpt_a": 2, "ckpt_b": 4})
    lm = DummyLM()

    db_path = str(tmp_path / "run.sqlite3")
    fingerprint = compute_config_fingerprint({"config": "scheduler-test"})
    checkpoint_store = CheckpointStore(db_path, fingerprint)

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        checkpoint_store=checkpoint_store,
        benchmark_map=_standalone_benchmark_map(["ckpt_a", "ckpt_b"]),
        max_concurrent_benchmarks=2,
        max_concurrent_benchmarks_explicit=True,
    )

    for task_name in ("ckpt_a", "ckpt_b"):
        benchmark = checkpoint_store.get_benchmark(task_name)
        assert benchmark["status"] == "completed"
    checkpoint_store.close()
