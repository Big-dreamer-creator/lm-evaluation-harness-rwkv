"""Cross-benchmark "shortest-remaining-rollout-first" admission ordering.

Used by :func:`lm_eval.evaluator.evaluate` to decide which benchmark should
be admitted next when more than one is allowed to run concurrently
(``[scheduler] max_concurrent_benchmarks``): the benchmark with the fewest
rollouts left to execute is admitted first, so short benchmarks finish (and
publish to the scoreboard) promptly instead of being stuck behind one very
large benchmark that happened to be listed first.

A "benchmark" here is a scheduling unit: either every leaf task of a
benchmark group (aggregated and published together), or a single standalone
task treated as its own one-leaf unit. Both are represented the same way --
a list of ``(task_name, task)`` pairs -- so this module never needs to know
which case it's looking at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from lm_eval.api.checkpoint import CheckpointStore
    from lm_eval.api.task import Task


def _pending_rollouts_for_task(
    task: "Task", task_name: str, checkpoint_store: "CheckpointStore | None"
) -> int:
    """Rollouts (repeated ``Instance`` draws) not yet completed for one task.

    A fresh run has no checkpoint data to consult, so the total is read
    straight out of ``task.instances`` (already populated by
    ``task.build_all_requests()`` before this is called). Resuming a run
    subtracts what the checkpoint store already has recorded as completed.
    """
    total = sum(instance.repeats for instance in task.instances)
    if checkpoint_store is not None:
        total -= checkpoint_store.count_completed(task_name)
    return max(total, 0)


def _benchmark_pending_rollouts(
    members: list[tuple[str, "Task"]],
    checkpoint_store: "CheckpointStore | None",
) -> int:
    """Sum of pending rollouts across every task in one scheduling unit."""
    return sum(
        _pending_rollouts_for_task(task, task_name, checkpoint_store)
        for task_name, task in members
    )


def admission_order(pending_by_benchmark: dict[str, int]) -> list[str]:
    """Benchmark names ordered by ascending pending-rollout count.

    Ties keep the original (insertion) order, since ``dict`` iteration order
    is preserved and :func:`sorted` is stable.
    """
    return [name for name, _ in sorted(pending_by_benchmark.items(), key=lambda kv: kv[1])]


def resolve_max_concurrent_benchmarks(
    requested: int,
    explicit: bool,
    world_size: int,
    warn: Any = None,
) -> int:
    """Reconcile the configured concurrency with a distributed ``world_size``.

    Concurrent admission of multiple benchmarks relies on each benchmark's
    ``run_requests``/``finish_benchmark`` calls issuing rank-synchronized
    collectives (``lm.barrier()``, ``lm.gather_object()``) in the same order
    across ranks; interleaving two benchmarks per-rank can't guarantee that,
    so it's only safe when ``world_size == 1`` (always true for HTTP/API
    backends, which is what this scheduler targets).

    - Left at the default (``explicit=False``) with ``world_size != 1``:
      silently downgrade to 1 and warn -- existing distributed runs that
      never touched ``[scheduler]`` must keep working unchanged.
    - Explicitly set to more than 1 with ``world_size != 1``: raise, since
      the user asked for something that isn't supported.
    """
    if world_size == 1 or requested <= 1:
        return requested
    if not explicit:
        if warn is not None:
            warn(
                "Detected a distributed run (world_size=%d): cross-benchmark "
                "concurrent scheduling has been automatically disabled "
                "(max_concurrent_benchmarks=1). Set [scheduler] "
                "max_concurrent_benchmarks explicitly to silence this.",
                world_size,
            )
        return 1
    raise NotImplementedError(
        f"[scheduler] max_concurrent_benchmarks={requested} is not supported "
        f"together with a distributed run (world_size={world_size}); "
        "concurrent benchmark admission cannot guarantee consistent "
        "cross-rank barrier ordering. Set max_concurrent_benchmarks = 1."
    )
