import threading
import time

import pytest

from lm_eval.models.api_pool import (
    CapacityScheduler,
    ContextLengthExhausted,
    PoolError,
    PoolManifest,
    Replica,
    ReplicaPool,
    _normalize_replicas,
    context_error_prompt_tokens,
    shrink_max_tokens,
)


def test_normalize_replicas_single_base_url():
    replicas = _normalize_replicas("http://a", None, num_concurrent=4)
    assert replicas == (Replica(base_url="http://a", max_concurrency=4),)


def test_normalize_replicas_min_concurrency_one():
    replicas = _normalize_replicas("http://a", None, num_concurrent=0)
    assert replicas[0].max_concurrency == 1


def test_normalize_replicas_pool_manifest_replicas():
    manifest_replicas = (
        Replica(base_url="http://a", max_concurrency=8),
        Replica(base_url="http://b", max_concurrency=4),
    )
    replicas = _normalize_replicas(None, manifest_replicas, num_concurrent=1)
    assert replicas == manifest_replicas


def test_normalize_replicas_conflict():
    with pytest.raises(PoolError):
        _normalize_replicas(
            "http://a", (Replica(base_url="http://b", max_concurrency=1),), 1
        )


def test_normalize_replicas_missing_both():
    with pytest.raises(PoolError):
        _normalize_replicas(None, None, 1)


def test_pool_manifest_read(tmp_path):
    manifest = tmp_path / "pool.toml"
    manifest.write_text(
        """
schema_version = 1

[[replicas]]
base_url = "http://host-a:8000/v1/completions"
max_concurrency = 8

[[replicas]]
base_url = "http://host-b:8000/v1/completions"
max_concurrency = 4
"""
    )
    replicas = PoolManifest.read(str(manifest))
    assert replicas == (
        Replica(base_url="http://host-a:8000/v1/completions", max_concurrency=8),
        Replica(base_url="http://host-b:8000/v1/completions", max_concurrency=4),
    )


def test_pool_manifest_read_missing_file(tmp_path):
    with pytest.raises(PoolError):
        PoolManifest.read(str(tmp_path / "does-not-exist.toml"))


def test_pool_manifest_read_bad_schema_version(tmp_path):
    manifest = tmp_path / "pool.toml"
    manifest.write_text('schema_version = 2\n[[replicas]]\nbase_url = "http://a"\n')
    with pytest.raises(PoolError):
        PoolManifest.read(str(manifest))


def test_pool_manifest_read_empty_replicas(tmp_path):
    manifest = tmp_path / "pool.toml"
    manifest.write_text("schema_version = 1\nreplicas = []\n")
    with pytest.raises(PoolError):
        PoolManifest.read(str(manifest))


def test_pool_manifest_read_duplicate_base_url(tmp_path):
    manifest = tmp_path / "pool.toml"
    manifest.write_text(
        """
schema_version = 1
[[replicas]]
base_url = "http://a"
max_concurrency = 1
[[replicas]]
base_url = "http://a"
max_concurrency = 1
"""
    )
    with pytest.raises(PoolError):
        PoolManifest.read(str(manifest))


def test_pool_manifest_read_non_positive_concurrency(tmp_path):
    manifest = tmp_path / "pool.toml"
    manifest.write_text(
        'schema_version = 1\n[[replicas]]\nbase_url = "http://a"\nmax_concurrency = 0\n'
    )
    with pytest.raises(PoolError):
        PoolManifest.read(str(manifest))


def test_capacity_scheduler_prefers_lower_normalized_load():
    scheduler = CapacityScheduler(
        (
            Replica(base_url="http://a", max_concurrency=1),
            Replica(base_url="http://b", max_concurrency=4),
        )
    )
    first = scheduler.acquire()
    # Replica 0 now has normalized load 1/1=1.0, replica 1 has 0/4=0.0.
    second = scheduler.acquire()
    assert (first, second) == (0, 1)
    scheduler.release(first)
    scheduler.release(second)


def test_capacity_scheduler_respects_capacity_under_concurrency():
    replicas = (
        Replica(base_url="http://a", max_concurrency=2),
        Replica(base_url="http://b", max_concurrency=2),
    )
    scheduler = CapacityScheduler(replicas)
    max_in_flight = [0, 0]
    current_in_flight = [0, 0]
    lock = threading.Lock()

    def worker():
        for _ in range(20):
            index = scheduler.acquire()
            with lock:
                current_in_flight[index] += 1
                max_in_flight[index] = max(max_in_flight[index], current_in_flight[index])
            time.sleep(0.001)
            with lock:
                current_in_flight[index] -= 1
            scheduler.release(index)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_in_flight[0] <= 2
    assert max_in_flight[1] <= 2


def test_capacity_scheduler_acquire_all_excluded_raises():
    scheduler = CapacityScheduler((Replica(base_url="http://a", max_concurrency=1),))
    with pytest.raises(PoolError):
        scheduler.acquire(excluded=frozenset({0}))


def test_replica_pool_first_base_url_and_capacity():
    pool = ReplicaPool(
        (
            Replica(base_url="http://a", max_concurrency=8),
            Replica(base_url="http://b", max_concurrency=4),
        )
    )
    assert pool.first_base_url == "http://a"
    assert pool.aggregate_capacity == 12
    index = pool.pick_sync()
    assert pool.base_url_for(index) in ("http://a", "http://b")
    pool.release(index)


def test_context_error_prompt_tokens_matches():
    body = "Error: prompt contains at least 4096 input tokens, please reduce."
    assert context_error_prompt_tokens(400, body) == 4096


def test_context_error_prompt_tokens_ignores_other_status():
    body = "prompt contains at least 4096 input tokens"
    assert context_error_prompt_tokens(500, body) is None


def test_context_error_prompt_tokens_no_match():
    assert context_error_prompt_tokens(400, "some other error") is None


def test_shrink_max_tokens_shrinks():
    payload = {"max_tokens": 512}
    adjusted = shrink_max_tokens(payload, available=100)
    assert adjusted == {"max_tokens": 100}
    assert payload == {"max_tokens": 512}  # original untouched


def test_shrink_max_tokens_handles_max_completion_tokens():
    payload = {"max_completion_tokens": 512, "other": 1}
    adjusted = shrink_max_tokens(payload, available=50)
    assert adjusted == {"max_completion_tokens": 50, "other": 1}


def test_shrink_max_tokens_no_budget():
    assert shrink_max_tokens({"max_tokens": 512}, available=0) is None
    assert shrink_max_tokens({"max_tokens": 512}, available=-5) is None


def test_shrink_max_tokens_no_shrinkable_key():
    assert shrink_max_tokens({"prompt": "x"}, available=100) is None


def test_shrink_max_tokens_already_smaller_than_available():
    # Nothing to shrink -- available is not tighter than current value.
    assert shrink_max_tokens({"max_tokens": 10}, available=100) is None


def test_context_length_exhausted_is_pool_error():
    assert issubclass(ContextLengthExhausted, PoolError)
