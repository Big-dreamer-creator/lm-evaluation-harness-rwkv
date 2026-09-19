"""Unit tests for the avg@k rollout-sizing formula.

Mirrors lighteval-rwkv's ``src/lighteval/models/rwkv/pipeline.py``
``_evaluation_plan``: pick the power-of-two ``k`` whose total rollout count
(``k * num_docs``) lands in ``[3000, 6000]``, closest to 4096; a limited run
always degrades to ``k=1``.
"""

import pytest

from lm_eval.api.benchmark import avg_at_k_plan


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


class _FakeTask:
    def __init__(self, output_type, process_results):
        self.OUTPUT_TYPE = output_type
        self.process_results = process_results
        self._aggregation_list = {"acc": "mean_fn", "exact_match": "mean_fn"}
        self._higher_is_better = {"acc": True, "exact_match": True}
        self._filters = None


class _FakeLM:
    def __init__(self):
        self._avg_k_samples = {}


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

    # switched to a filter that keeps all k generations instead of take_first
    assert task._filters[0].name == "none"

    doc = {"gold": "42"}
    metrics = task.process_results(doc, [["42", "7", "42", "42"]])

    assert metrics["exact_match_avg@4"] == pytest.approx(3 / 4)


def test_install_avg_at_k_patch_noop_when_k_is_one():
    from lm_eval.evaluator import _install_avg_at_k_patch

    def process_results(doc, results):
        return {"acc": 1.0}

    task = _FakeTask("multiple_choice", process_results)
    lm = _FakeLM()

    _install_avg_at_k_patch(task, lm, k=1, metric_name="avg@1")

    assert task.process_results is process_results
