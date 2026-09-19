import pytest

from lm_eval.config.evaluate_config import _classify_benchmark
from lm_eval.tasks import TaskManager


@pytest.fixture(scope="module")
def task_manager():
    return TaskManager()


def test_group_with_aggregate_metric_list_is_group_aggregate(task_manager):
    spec = _classify_benchmark("wmdp", task_manager.match_tasks(["wmdp"]), task_manager)

    assert spec.strategy == "group_aggregate"
    assert spec.leaf_task_names is None


def test_group_without_aggregate_metric_list_is_group_fallback(task_manager):
    spec = _classify_benchmark(
        "infinitebench", task_manager.match_tasks(["infinitebench"]), task_manager
    )

    assert spec.strategy == "group_fallback"
    assert spec.leaf_task_names is None


def test_tag_is_tag_fallback_with_full_leaf_set(task_manager):
    spec = _classify_benchmark(
        "model_written_evals",
        task_manager.match_tasks(["model_written_evals"]),
        task_manager,
    )

    assert spec.strategy == "tag_fallback"
    assert spec.leaf_task_names is not None
    assert len(spec.leaf_task_names) == 187


def test_standalone_task_is_standalone(task_manager):
    spec = _classify_benchmark("race", task_manager.match_tasks(["race"]), task_manager)

    assert spec.strategy == "standalone"
    assert spec.leaf_task_names == ("race",)


def test_glob_matching_a_group_raises(task_manager):
    with pytest.raises(ValueError, match="not a leaf task"):
        _classify_benchmark("wmdp*", ["wmdp", "wmdp_bio"], task_manager)


def test_glob_matching_only_leaf_tasks_is_tag_fallback(task_manager):
    spec = _classify_benchmark(
        "wmdp_*", ["wmdp_bio", "wmdp_chem", "wmdp_cyber"], task_manager
    )

    assert spec.strategy == "tag_fallback"
    assert spec.leaf_task_names == ("wmdp_bio", "wmdp_chem", "wmdp_cyber")
