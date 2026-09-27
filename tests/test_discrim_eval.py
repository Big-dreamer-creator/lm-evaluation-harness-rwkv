import datasets

from lm_eval.tasks.discrim_eval.utils import (
    agg_age_bias_slope,
    agg_demographic_bias_diff,
    process_results,
    shuffle_scenarios,
)


def _make_doc(race, gender, age, template_id):
    return {"race": race, "gender": gender, "age": age, "decision_question_id": template_id}


def _make_results(chose_yes: bool):
    # [(yes, is_greedy), (Yes, is_greedy), (no, is_greedy), (No, is_greedy)]
    if chose_yes:
        return [(0.0, True), (-1e6, False), (-1e6, False), (-1e6, False)]
    return [(-1e6, False), (-1e6, False), (0.0, True), (-1e6, False)]


def test_process_results_extracts_hard_choice_from_is_greedy():
    doc = _make_doc("white", "male", 30.0, 0)
    out = process_results(doc, _make_results(chose_yes=True))
    assert out["black_bias"] == (("white", "male", 30.0, 0), "black_bias", 1)

    out = process_results(doc, _make_results(chose_yes=False))
    assert out["black_bias"] == (("white", "male", 30.0, 0), "black_bias", 0)


def test_agg_demographic_bias_diff_recovers_injected_bias():
    # Scenario 0: black always chosen "yes", white always chosen "no" ->
    # a clean +1.0 bias favoring black over the white reference.
    items = []
    for _ in range(3):
        items.append(
            (("black", "male", 30.0, 0), "black_bias", 1)
        )
        items.append(
            (("white", "male", 30.0, 0), "black_bias", 0)
        )
    assert agg_demographic_bias_diff(items) == 1.0


def test_agg_demographic_bias_diff_averages_across_scenarios():
    items = [
        (("black", "male", 30.0, 0), "black_bias", 1),
        (("white", "male", 30.0, 0), "black_bias", 0),
        (("black", "male", 30.0, 1), "black_bias", 0),
        (("white", "male", 30.0, 1), "black_bias", 0),
    ]
    # scenario 0 diff = 1.0, scenario 1 diff = 0.0 -> mean 0.5
    assert agg_demographic_bias_diff(items) == 0.5


def test_agg_demographic_bias_diff_skips_scenarios_missing_a_group():
    items = [
        # scenario 0 has no white rows at all -> skipped
        (("black", "male", 30.0, 0), "black_bias", 1),
        (("black", "male", 30.0, 1), "black_bias", 1),
        (("white", "male", 30.0, 1), "black_bias", 0),
    ]
    assert agg_demographic_bias_diff(items) == 1.0


def test_agg_demographic_bias_diff_empty_input_is_zero():
    assert agg_demographic_bias_diff([]) == 0.0


def test_agg_age_bias_slope_recovers_positive_trend():
    items = [
        (("white", "male", 20.0, 0), "age_bias", 0),
        (("white", "male", 40.0, 0), "age_bias", 0),
        (("white", "male", 60.0, 0), "age_bias", 1),
        (("white", "male", 80.0, 0), "age_bias", 1),
    ]
    assert agg_age_bias_slope(items) > 0.0


def test_agg_age_bias_slope_empty_input_is_zero():
    assert agg_age_bias_slope([]) == 0.0


def test_agg_age_bias_slope_skips_degenerate_scenarios():
    items = [
        # single row: not enough to compute a slope
        (("white", "male", 20.0, 0), "age_bias", 1),
        # all same age: zero variance, must be skipped rather than div-by-zero
        (("white", "male", 30.0, 1), "age_bias", 1),
        (("white", "female", 30.0, 1), "age_bias", 0),
    ]
    assert agg_age_bias_slope(items) == 0.0


def test_shuffle_scenarios_preserves_rows_and_scenario_contiguity():
    rows = []
    for template_id in range(5):
        for i in range(3):
            rows.append({"decision_question_id": template_id, "value": f"{template_id}-{i}"})
    dataset = datasets.Dataset.from_list(rows)

    shuffled = shuffle_scenarios(dataset)

    assert sorted(shuffled["value"]) == sorted(dataset["value"])

    # Every scenario's rows must stay contiguous (blocks reordered, not
    # interleaved) and the block's internal row order preserved.
    ids = shuffled["decision_question_id"]
    seen_blocks = []
    i = 0
    while i < len(ids):
        j = i
        while j < len(ids) and ids[j] == ids[i]:
            j += 1
        seen_blocks.append(ids[i])
        assert shuffled["value"][i:j] == [f"{ids[i]}-{k}" for k in range(j - i)]
        i = j
    assert sorted(seen_blocks) == list(range(5))


def test_shuffle_scenarios_is_deterministic():
    rows = [
        {"decision_question_id": template_id, "value": template_id}
        for template_id in range(10)
        for _ in range(2)
    ]
    dataset = datasets.Dataset.from_list(rows)
    assert shuffle_scenarios(dataset)["value"] == shuffle_scenarios(dataset)["value"]
