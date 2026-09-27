from types import SimpleNamespace

import pytest

from lm_eval.api.benchmark_scheduler import (
    _benchmark_pending_rollouts,
    _pending_rollouts_for_task,
    admission_order,
    resolve_max_concurrent_benchmarks,
)
from lm_eval.api.checkpoint import CheckpointStore, ItemKey, compute_config_fingerprint


def _task(*repeats):
    return SimpleNamespace(instances=[SimpleNamespace(repeats=r) for r in repeats])


def _store(tmp_path):
    db_path = str(tmp_path / "run.sqlite3")
    fingerprint = compute_config_fingerprint({"config": "x"})
    return CheckpointStore(db_path, fingerprint)


def _register_and_complete(store, task_name, n_completed, n_total):
    keys = [
        ItemKey(
            item_id=f"{task_name}-{i}",
            request_fingerprint="fp",
            idempotency_key=f"idem-{task_name}-{i}",
            task_name=task_name,
            doc_id=0,
            instance_idx=0,
            request_type="loglikelihood",
            repeat_ord=i,
        )
        for i in range(n_total)
    ]
    store.register_pending_items(keys)
    store.mark_items_completed(
        [(keys[i].item_id, "{}") for i in range(n_completed)]
    )


def test_pending_rollouts_for_task_no_checkpoint():
    task = _task(1, 1, 2)
    assert _pending_rollouts_for_task(task, "t", None) == 4


def test_pending_rollouts_for_task_subtracts_completed(tmp_path):
    store = _store(tmp_path)
    task = _task(1, 1, 1, 1)
    _register_and_complete(store, "t", n_completed=3, n_total=4)
    assert _pending_rollouts_for_task(task, "t", store) == 1
    store.close()


def test_pending_rollouts_for_task_never_negative(tmp_path):
    store = _store(tmp_path)
    task = _task(1)
    _register_and_complete(store, "t", n_completed=1, n_total=1)
    assert _pending_rollouts_for_task(task, "t", store) == 0
    store.close()


def test_benchmark_pending_rollouts_sums_members():
    members = [("t1", _task(1, 1)), ("t2", _task(3))]
    assert _benchmark_pending_rollouts(members, None) == 5


def test_admission_order_ascending():
    pending = {"big": 10, "small": 1, "medium": 5}
    assert admission_order(pending) == ["small", "medium", "big"]


def test_admission_order_stable_on_ties():
    pending = {"a": 1, "b": 1, "c": 1}
    assert admission_order(pending) == ["a", "b", "c"]


def test_resolve_max_concurrent_world_size_one_unchanged():
    assert resolve_max_concurrent_benchmarks(3, explicit=False, world_size=1) == 3
    assert resolve_max_concurrent_benchmarks(1, explicit=True, world_size=1) == 1


def test_resolve_max_concurrent_requested_one_or_less_unchanged():
    assert resolve_max_concurrent_benchmarks(1, explicit=False, world_size=4) == 1
    assert resolve_max_concurrent_benchmarks(0, explicit=False, world_size=4) == 0


def test_resolve_max_concurrent_default_downgrades_and_warns():
    warnings = []
    result = resolve_max_concurrent_benchmarks(
        3, explicit=False, world_size=4, warn=lambda *a, **k: warnings.append((a, k))
    )
    assert result == 1
    assert len(warnings) == 1


def test_resolve_max_concurrent_explicit_raises():
    with pytest.raises(NotImplementedError):
        resolve_max_concurrent_benchmarks(3, explicit=True, world_size=4)
