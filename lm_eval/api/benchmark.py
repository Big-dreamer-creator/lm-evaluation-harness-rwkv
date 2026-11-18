"""Maps a TOML-requested benchmark name to its leaf tasks and aggregation strategy.

Tags never produce a ``Group`` object inside lm-eval (see
``lm_eval.tasks._factory._build_tag``), so a tag-based benchmark family (e.g.
``model_written_evals``) has no runtime object recording which leaves belong
to it. This module captures that mapping at config-resolution time, before
``TaskManager.load()`` flattens everything into bare tasks/groups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


eval_logger = logging.getLogger(__name__)

Strategy = Literal["group_aggregate", "group_fallback", "tag_fallback", "standalone"]
BenchmarkField = Literal[
    "knowledge",
    "reasoning",
    "maths",
    "coding",
    "instruction_following",
    "agentic",
    "vision",
    "reading_comprehension",
]
BENCHMARK_FIELDS = frozenset(
    {
        "knowledge",
        "reasoning",
        "maths",
        "coding",
        "instruction_following",
        "agentic",
        "vision",
        "reading_comprehension",
    }
)
DEFAULT_BENCHMARK_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "benchmarks.toml"
)


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
    field: BenchmarkField
    strategy: Strategy
    leaf_task_names: Optional[tuple[str, ...]] = None


def read_benchmark_fields(
    path: str | Path = DEFAULT_BENCHMARK_CONFIG,
) -> dict[str, BenchmarkField]:
    """Read the explicit ``benchmark selector -> field`` configuration.

    Selectors are opaque identifiers here. Their spelling, prefixes, and
    expanded lm-eval tasks never participate in field assignment.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid benchmark config {config_path}: {error}") from error

    fields: dict[str, BenchmarkField] = {}
    for field, selectors in raw.items():
        if field not in BENCHMARK_FIELDS or not isinstance(selectors, list):
            raise ValueError(f"{config_path}: invalid benchmark field {field!r}")
        for selector in selectors:
            if (
                not isinstance(selector, str)
                or not selector
                or selector != selector.strip()
            ):
                raise ValueError(
                    f"{config_path}: benchmark selectors must be non-empty "
                    "trimmed strings"
                )
            if selector in fields:
                raise ValueError(
                    f"{config_path}: duplicate benchmark selector {selector!r}"
                )
            fields[selector] = field
    if not fields:
        raise ValueError(f"{config_path}: no benchmarks configured")
    return fields


DEFAULT_BENCHMARK_MODES = (
    Path(__file__).resolve().parent.parent / "config" / "benchmark_modes.toml"
)


def read_benchmark_modes(path: str | Path | None = None) -> dict[str, str]:
    """Prototype: read_benchmark_fields; one authoritative RWKV mode inventory."""
    config_path = Path(path or DEFAULT_BENCHMARK_MODES).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid benchmark modes {config_path}: {error}") from error
    modes = {"no_think", "fake_think", "open_think"}
    if set(raw) != {"default", *modes} or raw.get("default") not in modes:
        raise ValueError(f"{config_path}: invalid mode sections or default")
    result = {"default": raw["default"]}
    for mode in sorted(modes):
        section = raw[mode]
        if not isinstance(section, dict) or set(section) != {"benchmarks"}:
            raise ValueError(f"{config_path}: invalid {mode} section")
        selectors = section["benchmarks"]
        if not isinstance(selectors, list):
            raise ValueError(f"{config_path}: {mode}.benchmarks must be a list")
        for selector in selectors:
            if not isinstance(selector, str) or not selector.strip() or selector != selector.strip():
                raise ValueError(f"{config_path}: selectors must be non-empty trimmed strings")
            if selector in result:
                raise ValueError(f"{config_path}: duplicate mode selector {selector!r}")
            result[selector] = mode
    return result


def rwkv_task_mode(task_name: str, output_type: str, selectors=()) -> str:
    """Prototype: RWKV7Remote.configure_task; request wiring stays authoritative."""
    modes = read_benchmark_modes()
    mode = next(
        (modes[name] for name in (task_name, *reversed(selectors)) if name in modes),
        modes["default"],
    )
    if output_type in {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}:
        if mode != "no_think":
            raise ValueError(f"{task_name}: native likelihood requests require no_think")
        return "no_think"
    return mode


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


@dataclass(frozen=True)
class AvgAtKPlan:
    """One benchmark's avg@k sizing, as seen from one of its leaf tasks.

    A benchmark is sized once, from the documents all of its leaves add up
    to, and every leaf then samples that same ``k`` times per doc. Sizing
    each leaf separately instead multiplies the benchmark's own question
    count by a per-leaf ``k`` -- a 57-subtask benchmark of ~250 docs each
    picked ``k=16`` per subject, i.e. ~225k rollouts for a 14k-question
    benchmark -- which is exactly what the rollout budget exists to bound.
    """

    benchmark: str
    leaves: tuple[str, ...]
    num_docs: int
    k: int
    metric_name: str


def _leaf_names(item: Any, tasks: dict[str, Any]) -> tuple[str, ...]:
    """Leaf task names of one group, restricted to the tasks in this run."""
    names: list[str] = []
    for task_obj in item.get_all_tasks(recursive=True):
        name = task_obj.task_name
        if name in tasks and name not in names:
            names.append(name)
    return tuple(names)


def benchmark_leaves(
    tasks: dict[str, Any],
    groups: dict[str, Any],
    benchmark_map: Optional[dict[str, BenchmarkSpec]] = None,
) -> dict[str, tuple[str, ...]]:
    """Map every benchmark in this run to the leaf tasks that carry it.

    A benchmark is whatever the run asked for: an entry of the eval config's
    ``benchmarks`` list (``benchmark_map``), a group the request resolved
    through, or -- for a task that stands alone -- the task itself. Group
    levels (``mmlu`` and ``mmlu_humanities``) claim the same leaves, so a
    leaf covered by several benchmarks is attributed to the largest one when
    the plan is built; sizing a benchmark off a superset of its leaves only
    lowers ``k``, which is the safe direction for the budget.

    ``leaf_task_names`` on a spec is used as-is when present (tags and
    standalone entries were resolved from the task index at config time, and
    a tag has no runtime group object to ask).
    """
    candidates: dict[str, tuple[str, ...]] = {}
    for bname, spec in (benchmark_map or {}).items():
        if spec.leaf_task_names is not None:
            leaves = tuple(name for name in spec.leaf_task_names if name in tasks)
        elif (group_obj := groups.get(bname)) is not None:
            leaves = _leaf_names(group_obj, tasks)
        else:
            leaves = ()
        if leaves:
            candidates[bname] = leaves
    for gname, group_obj in groups.items():
        if gname in candidates:
            continue
        leaves = _leaf_names(group_obj, tasks)
        if leaves:
            candidates[gname] = leaves
    for name in tasks:
        if not any(name in leaves for leaves in candidates.values()):
            candidates[name] = (name,)
    return candidates


def plan_benchmark_samples(
    tasks: dict[str, Any],
    groups: dict[str, Any],
    benchmark_map: dict[str, BenchmarkSpec],
    limit: int,
) -> tuple[dict[str, list[int]], tuple[str, ...]]:
    """Give every requested benchmark selector at most ``limit`` documents.

    Native lm-eval limits apply independently to every leaf task. That is the
    wrong scope for a benchmark selector that expands to dozens of leaves.
    This planner spends one document budget across the selector's leaves:

    * at most ``limit`` leaves are selected, spread across the full selector
      when it has more leaves than the budget;
    * selected leaves receive deterministic, evenly spaced document indices,
      allocated round-robin so the total never exceeds ``limit``;
    * tasks outside the selected leaves are omitted from the returned task set
      by the caller, while the benchmark remains one aggregate publication.

    The selected indices are intentionally deterministic. A limited run is a
    reproducible sample, not an avg@k measurement, and resuming it must select
    exactly the same documents.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("benchmark-scoped limit must be a positive integer")
    if not benchmark_map:
        raise ValueError("benchmark-scoped limit requires a benchmark map")

    allocations: dict[str, list[int]] = {}
    claimed_by: dict[str, str] = {}

    for benchmark_name, spec in benchmark_map.items():
        if spec.leaf_task_names is not None:
            leaves = list(spec.leaf_task_names)
        else:
            group_obj = groups.get(benchmark_name)
            leaves = (
                [task.task_name for task in group_obj.get_all_tasks(recursive=True)]
                if group_obj is not None
                else []
            )

        leaves = [
            leaf
            for leaf in dict.fromkeys(leaves)
            if leaf in tasks and len(tasks[leaf].eval_docs) > 0
        ]
        if not leaves:
            continue

        for leaf in leaves:
            owner = claimed_by.setdefault(leaf, benchmark_name)
            if owner != benchmark_name:
                raise ValueError(
                    f"benchmark-scoped limit cannot share leaf {leaf!r} "
                    f"between {owner!r} and {benchmark_name!r}"
                )

        total_docs = sum(len(tasks[leaf].eval_docs) for leaf in leaves)
        if total_docs <= limit:
            counts = {leaf: len(tasks[leaf].eval_docs) for leaf in leaves}
        elif len(leaves) > limit:
            # Centre the selected leaves across the full selector rather than
            # silently dropping its tail.
            selected_leaves = [
                leaves[((2 * index + 1) * len(leaves)) // (2 * limit)]
                for index in range(limit)
            ]
            counts = {leaf: 1 for leaf in selected_leaves}
        else:
            # Allocate the shared budget evenly while respecting each leaf's
            # capacity. Short leaves release their unused share back to the
            # remaining leaves, so the selector still spends its full budget.
            counts = {leaf: 0 for leaf in leaves}
            remaining = limit
            active = list(leaves)
            while remaining and active:
                share, extra = divmod(remaining, len(active))
                if share == 0:
                    for leaf in active[:remaining]:
                        counts[leaf] += 1
                    break
                progressed = 0
                for index, leaf in enumerate(active):
                    want = share + (index < extra)
                    capacity = len(tasks[leaf].eval_docs) - counts[leaf]
                    added = min(want, capacity)
                    counts[leaf] += added
                    remaining -= added
                    progressed += added
                active = [
                    leaf
                    for leaf in active
                    if counts[leaf] < len(tasks[leaf].eval_docs)
                ]
                if not progressed:
                    break

        for leaf, count in counts.items():
            num_docs = len(tasks[leaf].eval_docs)
            allocations[leaf] = (
                list(range(num_docs))
                if count == num_docs
                else [
                    ((2 * index + 1) * num_docs) // (2 * count)
                    for index in range(count)
                ]
            )

    unclaimed = [name for name in tasks if name not in claimed_by]
    if unclaimed:
        raise ValueError(
            "benchmark-scoped limit found tasks without a benchmark selector: "
            + ", ".join(unclaimed)
        )

    selected = tuple(
        name for name in tasks if name in allocations
    )
    return allocations, selected


def _without_re_scored_parents(
    leaves: tuple[str, ...], doc_counts: dict[str, int]
) -> tuple[str, ...]:
    """Drop a leaf whose documents are exactly its siblings' put together.

    Groups such as winogender's score an ``all`` task next to the disjoint
    female/male/neutral subsets it contains; ``winogender_all`` itself is a
    leaf of that group, so counting it as well would double the benchmark's
    documents (and halve its ``k``). The publish path drops the same leaf
    before counting (see ``lm_eval.loggers.scoreboard._parent_leaves``).

    Two-leaf benchmarks are left alone: there, "equals the sum of the other
    leaves" also describes two ordinary halves of one dataset, and dropping a
    half would double the rollouts the benchmark takes.
    """
    if len(leaves) < 3:
        return leaves
    kept = tuple(
        leaf
        for leaf in leaves
        if not (
            doc_counts.get(leaf)
            and doc_counts[leaf]
            == sum(doc_counts.get(other, 0) for other in leaves if other != leaf)
        )
    )
    if len(kept) != len(leaves):
        eval_logger.info(
            "avg@k: benchmark leaves %s re-score the documents of %s; the "
            "parent leaf is left out of the benchmark's document count.",
            sorted(set(leaves) - set(kept)),
            sorted(kept),
        )
    return kept


def plan_avg_at_k(
    tasks: dict[str, Any],
    groups: dict[str, Any],
    benchmark_map: Optional[dict[str, BenchmarkSpec]] = None,
    *,
    limited: bool = False,
    sampled_benchmarks: Optional[set[str]] = None,
) -> dict[str, AvgAtKPlan]:
    """Size ``avg@k`` once per benchmark, keyed by each of its leaf tasks.

    The benchmark's document count is the sum over its leaves, so a benchmark
    spends its rollout budget in one place: ``k`` lands its whole rollout
    count in ``[_MIN_COMPLETIONS, _MAX_COMPLETIONS]`` when its question count
    allows it, and degrades to ``k=1`` (never inflating the benchmark beyond
    its own size) when it does not.
    """
    candidates = benchmark_leaves(tasks, groups, benchmark_map)
    doc_counts = {name: len(task_obj.eval_docs) for name, task_obj in tasks.items()}

    totals: dict[str, int] = {}
    for bname, leaves in candidates.items():
        totals[bname] = sum(
            doc_counts[name] for name in _without_re_scored_parents(leaves, doc_counts)
        )

    claimants: dict[str, list[str]] = {name: [] for name in tasks}
    for bname, leaves in candidates.items():
        for name in leaves:
            claimants[name].append(bname)

    sampled_benchmarks = sampled_benchmarks or set()
    plans: dict[str, AvgAtKPlan] = {}
    for name, bnames in claimants.items():
        if not bnames:
            continue
        benchmark = max(bnames, key=lambda candidate: (totals[candidate], candidate))
        k, metric_name = avg_at_k_plan(
            totals[benchmark],
            limited=limited or benchmark in sampled_benchmarks,
        )
        plans[name] = AvgAtKPlan(
            benchmark=benchmark,
            leaves=candidates[benchmark],
            num_docs=totals[benchmark],
            k=k,
            metric_name=metric_name,
        )
    return plans
