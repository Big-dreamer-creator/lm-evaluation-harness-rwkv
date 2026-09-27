"""Multi-replica endpoint pools, capacity-aware load balancing, and
adaptive context-length-retry helpers for :class:`TemplateAPI` subclasses.

A pool is either:
- a single-replica pool built implicitly from a plain ``base_url`` string
  (the default, fully backward compatible), or
- a multi-replica pool described by a standalone TOML "pool manifest" file
  referenced via ``model_args.pool_manifest``.

Both paths produce the same ``tuple[Replica, ...]`` and go through the same
:class:`CapacityScheduler`/:class:`ReplicaPool`, so callers never need to
special-case the single-replica case.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


class PoolError(RuntimeError):
    """Raised for pool configuration or scheduling failures."""


class ContextLengthExhausted(PoolError):
    """Raised when a request's prompt tokens leave no positive token budget
    for a completion, even after shrinking the requested max tokens to fit.
    """


# Status codes worth failing over to another replica for (no point shrinking
# the payload, but also no point giving up on the whole request).
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

_CONTEXT_LENGTH_ERROR = re.compile(r"prompt contains at least (\d+) input tokens")
_MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")


@dataclass(frozen=True)
class Replica:
    base_url: str
    max_concurrency: int


def _normalize_replicas(
    base_url: Optional[str],
    replicas: Optional[Tuple[Replica, ...]],
    num_concurrent: int,
) -> Tuple[Replica, ...]:
    """Single entry point for turning either config surface into replicas.

    Exactly one of ``base_url``/``replicas`` must be given. A plain
    ``base_url`` becomes a single-replica pool whose capacity is
    ``num_concurrent`` -- i.e. "single base_url = single-replica pool", so
    existing configs go through the exact same code path as pool configs.
    """
    if replicas is not None:
        if base_url is not None:
            raise PoolError(
                "model_args: specify either `base_url` or `pool_manifest`, not both."
            )
        if not replicas:
            raise PoolError("Replica pool must contain at least one replica.")
        return tuple(replicas)
    if not base_url:
        raise PoolError("model_args: `base_url` is required when `pool_manifest` is not set.")
    return (Replica(base_url=base_url, max_concurrency=max(1, int(num_concurrent))),)


class PoolManifest:
    """Parses and validates a standalone pool-manifest TOML file."""

    @staticmethod
    def read(path: str) -> Tuple[Replica, ...]:
        manifest_path = Path(path).expanduser()
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
        except OSError as e:
            raise PoolError(f"Could not read pool manifest {path!r}: {e}") from e
        try:
            data = tomllib.loads(raw_text)
        except tomllib.TOMLDecodeError as e:
            raise PoolError(f"Malformed pool manifest {path!r}: {e}") from e

        schema_version = data.get("schema_version")
        if schema_version != 1:
            raise PoolError(
                f"Pool manifest {path!r}: unsupported schema_version={schema_version!r}, expected 1."
            )
        raw_replicas = data.get("replicas")
        if not raw_replicas or not isinstance(raw_replicas, list):
            raise PoolError(
                f"Pool manifest {path!r} must define a non-empty [[replicas]] array."
            )

        replicas = []
        seen_urls = set()
        for i, entry in enumerate(raw_replicas):
            if not isinstance(entry, dict):
                raise PoolError(f"Pool manifest {path!r}: replicas[{i}] must be a table.")
            base_url = entry.get("base_url")
            if not base_url or not isinstance(base_url, str):
                raise PoolError(
                    f"Pool manifest {path!r}: replicas[{i}].base_url is required and must be a string."
                )
            if base_url in seen_urls:
                raise PoolError(
                    f"Pool manifest {path!r}: duplicate base_url {base_url!r} in replicas[{i}]."
                )
            seen_urls.add(base_url)
            max_concurrency = entry.get("max_concurrency", 1)
            if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency <= 0:
                raise PoolError(
                    f"Pool manifest {path!r}: replicas[{i}].max_concurrency must be a positive integer."
                )
            replicas.append(Replica(base_url=base_url, max_concurrency=max_concurrency))
        return tuple(replicas)


class CapacityScheduler:
    """Thread-safe (not asyncio-only) capacity-aware replica scheduler.

    Selection minimizes normalized load (``in_flight / max_concurrency``),
    then raw ``in_flight``, then index -- so replicas with more capacity get
    proportionally more of the traffic instead of an equal share.
    """

    def __init__(self, replicas: Tuple[Replica, ...]):
        if not replicas:
            raise PoolError("CapacityScheduler requires at least one replica.")
        self._replicas = replicas
        self._in_flight = [0] * len(replicas)
        self._cond = threading.Condition()
        # Dedicated (not asyncio's shared default) executor for
        # `acquire_async`'s background `acquire()` calls, which block
        # indefinitely on `self._cond` while capacity is exhausted. Sized
        # generously so heavy concurrent-benchmark configs (many benchmark
        # worker threads x num_concurrent each, all potentially waiting on
        # this same pool at once) don't queue behind asyncio's small default
        # pool (min(32, cpu_count+4)), which would just make the blocked-time
        # -- and thus the window for the cancellation race handled in
        # `acquire_async` -- worse.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=256, thread_name_prefix="capacity-scheduler"
        )
        # Nothing calls a `close()`/`shutdown()` on `TemplateAPI` today, so a
        # process that builds more than one (e.g. repeated `simple_evaluate()`
        # calls from a notebook/library caller, each constructing its own
        # pool) would otherwise leak a fresh 256-thread pool per instance with
        # no way to reclaim it before process exit. `weakref.finalize` doesn't
        # hold a strong reference back to `self`, so this fires as soon as
        # this scheduler (and everything referencing it) is garbage
        # collected, regardless of whether any caller remembers to clean up.
        weakref.finalize(self, self._executor.shutdown, wait=False)

    @property
    def replicas(self) -> Tuple[Replica, ...]:
        return self._replicas

    def _pick_locked(self, excluded: FrozenSet[int]) -> Optional[int]:
        best_index = None
        best_key = None
        for i, replica in enumerate(self._replicas):
            if i in excluded or self._in_flight[i] >= replica.max_concurrency:
                continue
            key = (self._in_flight[i] / replica.max_concurrency, self._in_flight[i], i)
            if best_key is None or key < best_key:
                best_key, best_index = key, i
        return best_index

    def acquire(self, excluded: FrozenSet[int] = frozenset()) -> int:
        if len(excluded) >= len(self._replicas):
            raise PoolError("All replicas are excluded; nothing left to acquire.")
        with self._cond:
            while True:
                index = self._pick_locked(excluded)
                if index is not None:
                    self._in_flight[index] += 1
                    return index
                self._cond.wait()

    async def acquire_async(self, excluded: FrozenSet[int] = frozenset()) -> int:
        # `asyncio.to_thread`/`loop.run_in_executor` can't actually stop the
        # background OS thread once it's inside the blocking
        # `Condition.wait()` in `acquire()`. Cancelling the *asyncio* future
        # returned by `run_in_executor` transitions it to cancelled
        # synchronously regardless of whether the underlying
        # `concurrent.futures.Future` actually stopped -- if it hasn't (it's
        # still blocked waiting for capacity), asyncio's own future-chaining
        # silently discards whatever that thread eventually returns once the
        # wrapper is already cancelled. If that eventual result is a real
        # acquired slot, it would never be `release()`d -- a leak that,
        # repeated enough times, drains a replica's capacity to zero and
        # deadlocks every future `acquire()` on it.
        #
        # So: submit to our own executor directly and watch the *raw*
        # `concurrent.futures.Future` (not the asyncio wrapper) so we can
        # release on its behalf if it completes after we've stopped waiting
        # on it, no matter how much later that turns out to be.
        raw_future = self._executor.submit(self.acquire, excluded)
        try:
            return await asyncio.wrap_future(raw_future)
        except asyncio.CancelledError:
            raw_future.add_done_callback(self._release_orphaned_acquire)
            raise

    def _release_orphaned_acquire(self, raw_future: "concurrent.futures.Future") -> None:
        if not raw_future.cancelled() and raw_future.exception() is None:
            self.release(raw_future.result())

    def release(self, index: int) -> None:
        with self._cond:
            self._in_flight[index] -= 1
            self._cond.notify_all()


class ReplicaPool:
    """Thin façade over :class:`CapacityScheduler` that `TemplateAPI` holds."""

    def __init__(self, replicas: Tuple[Replica, ...]):
        self._scheduler = CapacityScheduler(replicas)

    @property
    def replicas(self) -> Tuple[Replica, ...]:
        return self._scheduler.replicas

    @property
    def first_base_url(self) -> str:
        """Stable representative URL for code that only knows about one
        endpoint (checkpoint fingerprinting, tokenizer probing, ...).
        """
        return self._scheduler.replicas[0].base_url

    @property
    def aggregate_capacity(self) -> int:
        return sum(r.max_concurrency for r in self._scheduler.replicas)

    def base_url_for(self, index: int) -> str:
        return self._scheduler.replicas[index].base_url

    def pick_sync(self, excluded: FrozenSet[int] = frozenset()) -> int:
        return self._scheduler.acquire(excluded)

    async def pick_async(self, excluded: FrozenSet[int] = frozenset()) -> int:
        return await self._scheduler.acquire_async(excluded)

    def release(self, index: int) -> None:
        self._scheduler.release(index)


def context_error_prompt_tokens(status: int, body_text: str) -> Optional[int]:
    """Extract "prompt contains at least N input tokens" from a vLLM-style
    400/422 error body, or ``None`` if this isn't a context-length error.
    """
    if status not in (400, 422) or not body_text:
        return None
    match = _CONTEXT_LENGTH_ERROR.search(body_text)
    return int(match.group(1)) if match else None


def shrink_max_tokens(payload: dict, available: int) -> Optional[dict]:
    """Return a copy of ``payload`` with its max-tokens field shrunk to fit
    ``available`` tokens, or ``None`` if there's no positive budget left or
    no max-tokens field to shrink (caller should treat both as "exhausted").
    """
    if available <= 0:
        return None
    for key in _MAX_TOKEN_KEYS:
        if key not in payload or payload[key] is None:
            continue
        new_value = min(int(payload[key]), available)
        if new_value <= 0:
            return None
        if new_value == payload[key]:
            return None
        adjusted = dict(payload)
        adjusted[key] = new_value
        return adjusted
    return None
