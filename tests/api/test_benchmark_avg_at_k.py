"""Unit tests for the avg@k rollout-sizing formula.

Mirrors lighteval-rwkv's ``src/lighteval/models/rwkv/pipeline.py``
``_evaluation_plan``: pick the power-of-two ``k`` whose total rollout count
(``k * num_docs``) lands in ``[3000, 6000]``, closest to 4096; a limited run
always degrades to ``k=1``.
"""

import pytest

from lm_eval.api.benchmark import avg_at_k_plan
from lm_eval.filters import build_filter_ensemble


@pytest.mark.parametrize(
    "num_docs, expected_k",
    [
        (30, 128),
        (500, 8),
        (4096, 1),
        (10000, 1),
    ],
)
def test_avg_at_k_plan_unlimited(num_docs, expected_k):
    k, metric_name = avg_at_k_plan(num_docs, limited=False)
    assert k == expected_k
    assert metric_name == f"avg@{expected_k}"


def test_avg_at_k_plan_limited_forces_k_1():
    k, metric_name = avg_at_k_plan(30, limited=True)
    assert k == 1
    assert metric_name == "avg@1"


def test_avg_at_k_plan_zero_docs():
    k, metric_name = avg_at_k_plan(0, limited=False)
    assert k == 1
    assert metric_name == "avg@1"


def test_avg_at_k_plan_stays_within_budget():
    for num_docs in (1, 2, 7, 17, 100, 999, 3001, 4095, 4097):
        k, _ = avg_at_k_plan(num_docs, limited=False)
        total = k * num_docs
        assert k == 1 or 3000 <= total <= 6000


def _default_filters():
    """The single-response-set pipeline a task without a `filter_list` gets."""
    return [build_filter_ensemble("none", [["take_first", None]])]


class _FakeTask:
    def __init__(self, output_type, process_results, filters=None):
        self.OUTPUT_TYPE = output_type
        self.process_results = process_results
        self._aggregation_list = {"acc": "mean_fn", "exact_match": "mean_fn"}
        self._higher_is_better = {"acc": True, "exact_match": True}
        self._filters = _default_filters() if filters is None else filters


class _FakeLM:
    def __init__(self):
        self._avg_k_samples = {}


def test_avg_at_k_survives_a_checkpoint_restored_doc():
    """A doc whose requests were restored instead of re-generated never
    reaches the model's ``loglikelihood()`` again, so ``lm._avg_k_samples``
    has no entry for it and it used to drop out of the benchmark's average
    entirely. The per-rollout record carried by the responses themselves
    keeps it in both the average and the counted completions.
    """
    from lm_eval.evaluator import _install_avg_at_k_patch
    from lm_eval.models.rwkv_choice import RWKVLoglikelihoodResponse

    def process_results(doc, results):
        (ll, is_greedy) = results[0]
        return {"acc": 1.0 if is_greedy else 0.0}

    task = _FakeTask("multiple_choice", process_results)
    lm = _FakeLM()  # nothing recorded in memory: this doc was restored
    _install_avg_at_k_patch(task, lm, k=4, metric_name="avg@4")

    rollouts = [
        {"index": 0, "truncated": False},
        {"index": 1, "truncated": False},
        {"index": 0, "truncated": True},
        {"index": 0, "truncated": False},
    ]
    responses = [
        RWKVLoglikelihoodResponse(
            0.0 if choice == 0 else -1e6, choice == 0, rollouts=rollouts
        )
        for choice in (0, 1)
    ]

    task._avg_k_current_doc_id = 7
    metrics = task.process_results({}, responses)

    assert metrics["acc"] == 1.0
    assert metrics["acc_avg@4"] == pytest.approx(3 / 4)
    assert task._rollout_stats[7] == {"completions": 4, "truncated": 1}


def test_install_avg_at_k_patch_multiple_choice_averages_samples():
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        (ll, is_greedy) = results[0]
        return {"acc": 1.0 if is_greedy else 0.0}

    task = _FakeTask("multiple_choice", process_results)
    lm = _FakeLM()
    lm._avg_k_samples[42] = [
        [(0.0, True), (-1e6, False)],
        [(-1e6, False), (0.0, True)],
        [(0.0, True), (-1e6, False)],
    ]

    _install_avg_at_k_patch(task, lm, k=3, metric_name="avg@3")

    task._avg_k_current_doc_id = 42
    metrics = task.process_results({}, [(0.0, True)])

    assert metrics["acc"] == 1.0
    assert metrics["acc_avg@3"] == pytest.approx(2 / 3)
    # doc_id popped so it can't leak into a later doc
    assert 42 not in lm._avg_k_samples


def test_install_avg_at_k_patch_generate_until_averages_k_generations():
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        return {"exact_match": 1.0 if results[0] == doc["gold"] else 0.0}

    task = _FakeTask("generate_until", process_results)
    lm = _FakeLM()

    _install_avg_at_k_patch(task, lm, k=4, metric_name="avg@4")

    # the task's own pipeline is kept, not replaced
    assert task._filters[0].name == "none"

    doc = {"gold": "42"}
    metrics = task.process_results(doc, [["42", "7", "42", "42"]])

    assert metrics["exact_match"] == 1.0
    assert metrics["exact_match_avg@4"] == pytest.approx(3 / 4)


def test_per_repeat_filter_runs_a_collapsing_pipeline_once_per_repeat():
    from lm_eval.filters.selection import PerRepeatFilter

    ensemble = build_filter_ensemble(
        "regex_match",
        [("regex", {"regex_pattern": r"([A-D])"}), ("take_first", None)],
    )

    filtered = PerRepeatFilter(ensemble).apply(
        [["reasoning... answer: A", "reasoning... answer: B", "answer: C", "answer: D"]],
        [{}],
    )

    # one value per repeat, each shaped like a `repeats: 1` result
    assert filtered == [["A", "B", "C", "D"]]


def test_per_repeat_filter_keeps_per_repeat_extractors_wrapped():
    from lm_eval.filters.selection import PerRepeatFilter

    ensemble = build_filter_ensemble(
        "albatross_mcq", [["albatross_mcq", {"choice_labels": "ABCD"}]]
    )

    filtered = PerRepeatFilter(ensemble).apply(
        [["<think>hmm</think>The answer is A", "The answer is B", "no letter", "D"]],
        [{}],
    )

    # `albatross_mcq` already yields a per-response list, so the wrapper adds
    # the repeat dimension without flattening it
    assert filtered == [[["A"], ["B"], ["[invalid]"], ["D"]]]


def test_install_avg_at_k_patch_generate_until_scores_the_extracted_answer():
    """Regression: the avg@k patch used to swap in `take_first_k`, so
    `process_results` saw raw completions and mutual-style lettered tasks
    scored 0 on every doc."""
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        predicted = results[0][0] if results and results[0] else None
        return {"acc": int(predicted == doc["gold"])}

    task = _FakeTask(
        "generate_until",
        process_results,
        filters=[
            build_filter_ensemble(
                "albatross_mcq", [["albatross_mcq", {"choice_labels": "ABCD"}]]
            )
        ],
    )
    lm = _FakeLM()

    _install_avg_at_k_patch(task, lm, k=4, metric_name="avg@4")

    # filter name (and therefore the metric key) survives the patch
    assert [ensemble.name for ensemble in task._filters] == ["albatross_mcq"]

    # `finish_benchmark` hands over one instance's filtered_resps: the
    # per-repeat extracted answers, not the k raw completions
    doc = {"gold": "B"}
    metrics = task.process_results(doc, [[["A"], ["B"], ["C"], ["B"]]])

    assert metrics["acc"] == 0.0
    assert metrics["acc_avg@4"] == pytest.approx(0.5)


def test_install_avg_at_k_patch_noop_when_k_is_one():
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        return {"acc": 1.0}

    task = _FakeTask("multiple_choice", process_results)
    lm = _FakeLM()

    _install_avg_at_k_patch(task, lm, k=1, metric_name="avg@1")

    assert task.process_results is process_results
