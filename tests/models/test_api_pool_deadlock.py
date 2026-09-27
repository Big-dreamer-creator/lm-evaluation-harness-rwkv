"""Regression tests for the capacity-leak deadlock diagnosed from a real
sustained-failure run against api.rwkv.rs: a task blocked in
``CapacityScheduler.acquire_async`` gets cancelled (e.g. because a sibling
task in the same batch failed and the caller cancelled the rest, or because
``asyncio.run()`` cancelled everything still pending at shutdown) while its
background thread is still inside the blocking ``acquire()``/``Condition.wait``
call. If that thread later succeeds and increments ``_in_flight`` after the
coroutine has already abandoned it, the slot is never released, and enough
of these leaks drain a replica's capacity to zero -- after which every future
``acquire()`` blocks forever.
"""

import asyncio
import threading
import time

import pytest

from lm_eval.models.api_pool import CapacityScheduler, Replica


@pytest.fixture
def single_slot_scheduler():
    return CapacityScheduler((Replica(base_url="http://replica-a", max_concurrency=1),))


def test_cancelling_acquire_async_releases_slot_that_completes_after_cancellation(
    single_slot_scheduler,
):
    """The exact race from the real deadlock: capacity is full, a coroutine
    is cancelled while waiting, and only afterwards does its background
    thread finally get a slot. That slot must not leak.
    """
    scheduler = single_slot_scheduler
    held_index = scheduler.acquire()  # fills the only slot

    async def scenario():
        task = asyncio.ensure_future(scheduler.acquire_async())
        # Give the background thread time to actually start blocking on
        # Condition.wait() before we cancel the awaiting coroutine.
        await asyncio.sleep(0.05)
        task.cancel()

        # A `run_in_executor` future that's already *running* (blocked in
        # Condition.wait(), in this case) can't actually be cancelled --
        # asyncio defers delivering the CancelledError into the coroutine
        # until the future genuinely completes. So `await task` would hang
        # forever unless something frees up capacity; do that here rather
        # than after, mirroring how a real sibling `release()` elsewhere in
        # the pool would eventually unblock this thread.
        scheduler.release(held_index)

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

        assert scheduler._in_flight[0] == 0, (
            "capacity leaked: the abandoned acquire's slot was never released"
        )

    asyncio.run(scenario())


def test_repeated_cancellation_does_not_exhaust_capacity(single_slot_scheduler):
    """Simulates the observed failure cascade: many requests are cancelled
    in quick succession while capacity is contended. Without the fix, each
    cancelled-then-completed acquire leaks a slot; enough of them and the
    pool can never acquire again.
    """
    scheduler = single_slot_scheduler

    async def scenario():
        held_index = scheduler.acquire()

        tasks = []
        for _ in range(20):
            task = asyncio.ensure_future(scheduler.acquire_async())
            await asyncio.sleep(0.01)
            task.cancel()
            tasks.append(task)

        # None of these can actually resolve (cancelled or not) until the
        # only slot frees up -- same as above, release before awaiting.
        scheduler.release(held_index)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Whichever cancelled acquire happened to win the race gets to keep
        # the slot released as part of its own cancellation handling above;
        # everyone else's should have been released automatically too.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if scheduler._in_flight[0] == 0:
                break

        assert scheduler._in_flight[0] == 0

        # The pool must still be usable afterwards -- this would hang
        # forever pre-fix if a slot had leaked.
        index = await asyncio.wait_for(scheduler.acquire_async(), timeout=2.0)
        scheduler.release(index)

    asyncio.run(scenario())


def test_cancellation_after_future_already_completed_still_releases(
    single_slot_scheduler,
):
    """Covers the narrower race where the background thread finishes
    *before* the cancellation is actually delivered into the coroutine --
    `future.done()` is already True inside the `except CancelledError`
    handler, which is the other branch of the fix (as opposed to the
    done-callback branch exercised above).
    """
    scheduler = single_slot_scheduler

    async def scenario():
        # No contention: the very first acquire_async() call resolves
        # almost immediately on its background thread, likely before
        # cancel() is even processed by the event loop.
        task = asyncio.ensure_future(scheduler.acquire_async())
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        for _ in range(50):
            await asyncio.sleep(0.02)
            if scheduler._in_flight[0] == 0:
                break

        assert scheduler._in_flight[0] == 0

    asyncio.run(scenario())
