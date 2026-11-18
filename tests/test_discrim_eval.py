import datasets
import numpy as np

from lm_eval.tasks.discrim_eval.utils import (
    agg_demographic_bias_regression,
    process_results,
    shuffle_scenarios,
    stratify_by_scenario,
)


def _make_doc(race, gender, age, template_id):
    return {"race": race, "gender": gender, "age": age, "decision_question_id": template_id}


def _logit(p: float) -> float:
    return float(np.log(p) - np.log1p(-p))


def test_process_results_computes_logit_yes_from_four_way_likelihoods():
    doc = _make_doc("white", "male", 30.0, 0)
    # yes=Yes=no=No equally likely -> pnorm(yes) = 0.5 -> logit 0.0
    out = process_results(doc, [(0.0, True), (0.0, True), (0.0, True), (0.0, True)])
    demographics, bias_name, logit_yes = out["black_bias"]
    assert demographics == ("white", "male", 30.0, 0)
    assert bias_name == "black_bias"
    assert logit_yes == 0.0


def test_process_results_combines_upper_and_lower_case_candidates():
    doc = _make_doc("black", "female", 45.0, 1)
    # yes/Yes overwhelmingly more likely than no/No.
    out = process_results(doc, [(-0.01, True), (-0.01, True), (-10.0, False), (-10.0, False)])
    _, _, logit_yes = out["age_bias"]
    assert logit_yes > 0


def test_agg_demographic_bias_regression_empty_input_is_zero():
    assert agg_demographic_bias_regression([]) == 0.0


def test_agg_demographic_bias_regression_too_few_rows_is_zero():
    items = [(("white", "male", 30.0, 0), "black_bias", 1.0)]
    assert agg_demographic_bias_regression(items) == 0.0


def test_agg_demographic_bias_regression_skips_rows_with_missing_demographics():
    # Every row is missing a demographic field, so none can enter the
    # regression -- must fall back to 0.0 rather than raising.
    items = [
        ((None, "male", 30.0, 0), "black_bias", 1.0),
        (("white", None, 30.0, 0), "black_bias", 0.0),
    ]
    assert agg_demographic_bias_regression(items) == 0.0


def test_agg_demographic_bias_regression_recovers_injected_bias():
    # A balanced race x gender x age cross per scenario, with black
    # consistently scored higher logit_yes than white -> a clean positive
    # coefficient for black vs. the white reference.
    items = []
    ages = [20.0, 40.0, 60.0, 80.0]
    for scenario in range(3):
        for age in ages:
            items.append((("black", "male", age, scenario), "black_bias", _logit(0.9)))
            items.append((("white", "male", age, scenario), "black_bias", _logit(0.1)))
            items.append((("black", "female", age, scenario), "black_bias", _logit(0.9)))
            items.append((("white", "female", age, scenario), "black_bias", _logit(0.1)))
    coef = agg_demographic_bias_regression(items)
    assert coef > 0


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


def test_stratify_by_scenario_preserves_rows_and_covers_every_scenario_first():
    rows = []
    for template_id in range(4):
        for race in ["white", "black"]:
            for gender in ["male", "female"]:
                for age in [20.0, 40.0]:
                    rows.append(
                        {
                            "decision_question_id": template_id,
                            "race": race,
                            "gender": gender,
                            "age": age,
                        }
                    )
    dataset = datasets.Dataset.from_list(rows)

    stratified = stratify_by_scenario(dataset, per_scenario=2)

    assert len(stratified) == len(dataset)
    first_block_ids = set(stratified["decision_question_id"][: 4 * 2])
    assert first_block_ids == {0, 1, 2, 3}
