"""Maps a TOML-requested benchmark name to its leaf tasks and aggregation strategy.

Tags never produce a ``Group`` object inside lm-eval (see
``lm_eval.tasks._factory._build_tag``), so a tag-based benchmark family (e.g.
``model_written_evals``) has no runtime object recording which leaves belong
to it. This module captures that mapping at config-resolution time, before
``TaskManager.load()`` flattens everything into bare tasks/groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Strategy = Literal["group_aggregate", "group_fallback", "tag_fallback", "standalone"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """One entry from the TOML ``benchmarks`` list, classified for publication.

    ``leaf_task_names`` is precomputed for ``tag_fallback``/``standalone`` (the
    tag/task index has no live object to query later). For ``group_aggregate``/
    ``group_fallback`` it is left ``None``: the authoritative leaf list is
    read from the live ``Group.get_all_tasks()`` once ``TaskManager.load()``
    has run, to avoid duplicating group-membership resolution logic.
    """

    name: str
    strategy: Strategy
    leaf_task_names: Optional[tuple[str, ...]] = None


# Rollout budget for the avg@k evaluation mode, mirroring lighteval-rwkv's
# src/lighteval/models/rwkv/pipeline.py `_evaluation_plan`: pick the
# power-of-two `k` (samples per doc) whose total rollout count
# (`k * num_docs`) falls in [_MIN_COMPLETIONS, _MAX_COMPLETIONS], closest to
# _TARGET_COMPLETIONS. Every RWKV benchmark is scored the same way regardless
# of its native output_type, so this lives outside any single task's config.
_TARGET_COMPLETIONS = 4096
_MIN_COMPLETIONS = 3000
_MAX_COMPLETIONS = 6000


def avg_at_k_plan(num_docs: int, limited: bool) -> tuple[int, str]:
    """Return ``(k, metric_name)`` for the avg@k evaluation mode.

    ``num_docs`` must be the benchmark's full, pre-``--limit`` doc count --
    the whole point of the budget is to size `k` off of the real campaign,
    not a smoke-test slice. Per lighteval-rwkv, whenever a run is limited
    (``--limit``/``max_samples`` truncates the docs actually evaluated) the
    formula is bypassed entirely and `k` is pinned to 1: a limited run is a
    smoke test, not a real avg@k measurement.
    """
    if limited or num_docs == 0:
        return 1, "avg@1"
    candidates = []
    k = 1
    while k * num_docs <= _MAX_COMPLETIONS:
        if k * num_docs >= _MIN_COMPLETIONS:
            candidates.append(k)
        k *= 2
    if candidates:
        k = min(
            candidates, key=lambda candidate: (abs(candidate * num_docs - _TARGET_COMPLETIONS), candidate)
        )
    return k, f"avg@{k}"
