from lm_eval.models.rwkv_choice import (
    build_choice_prompt,
    choice_letters,
    parse_choice_index,
)

LETTERS = ["A", "B", "C", "D"]
CHOICES = ["cats", "dogs", "birds", "fish"]


def test_build_choice_prompt_and_letters():
    prompt = build_choice_prompt("Which pet?", CHOICES)
    assert "A. cats" in prompt
    assert "D. fish" in prompt
    assert choice_letters(CHOICES) == LETTERS


def test_build_choice_prompt_closes_dangling_gen_prefix_quote():
    # A context ending in an unclosed quote (e.g. discrim_eval's gen_prefix
    # "...my answer would be \"") must not get a fresh menu tacked directly
    # onto it -- each context+choice completion should be rendered as its
    # own option instead.
    context = 'If I had to choose my answer would be "'
    prompt = build_choice_prompt(context, ["yes", "no"])
    assert f'A. {context}yes' in prompt
    assert f'B. {context}no' in prompt
    assert not prompt.startswith(f"{context}\n\n")


def test_parses_boxed_answer():
    text = "Reasoning...\nSo the answer is \\boxed{B}"
    assert parse_choice_index(text, LETTERS, CHOICES) == 1


def test_parses_boxed_text_wrapped_answer():
    text = "\\boxed{\\text{(C)}}"
    assert parse_choice_index(text, LETTERS, CHOICES) == 2


def test_parses_explicit_answer_is_phrasing():
    text = "After some thought, the final answer is: C"
    assert parse_choice_index(text, LETTERS, CHOICES) == 2


def test_parses_x_is_the_answer_phrasing():
    text = "D is the correct answer here."
    assert parse_choice_index(text, LETTERS, CHOICES) == 3


def test_falls_back_to_fallback_tier_phrasing():
    # explicit tier only covers "answer is/:" phrasing, so this exercises the
    # looser fallback tier ("choose/select/pick") instead.
    text = "I would choose: B"
    assert parse_choice_index(text, LETTERS, CHOICES) == 1


def test_parses_bare_trailing_letter_line():
    text = "Some reasoning about pets.\n\nB."
    assert parse_choice_index(text, LETTERS, CHOICES) == 1


def test_concluding_segment_after_think_takes_priority_over_reasoning_mentions():
    text = (
        "<think>Maybe A? No wait, let's reconsider... actually C seems right "
        "but let's double check.</think>\nThe answer is B"
    )
    assert parse_choice_index(text, LETTERS, CHOICES) == 1


def test_falls_back_to_think_segment_when_conclusion_is_truncated():
    text = "<think>After careful analysis, the answer is C</think>"
    assert parse_choice_index(text, LETTERS, CHOICES) == 2


def test_last_match_wins_within_a_tier():
    text = "Could be A, could be B, but actually the answer is C, no the answer is D"
    assert parse_choice_index(text, LETTERS, CHOICES) == 3


def test_falls_back_to_choice_text_substring_match():
    text = "I think the pet that barks and is loyal, dogs, is the one."
    assert parse_choice_index(text, LETTERS, CHOICES) == 1


def test_returns_negative_one_when_nothing_can_be_parsed():
    text = "This is a completely unrelated response with no letters at all here."
    assert parse_choice_index(text, LETTERS, CHOICES) == -1


def test_strips_leading_gt_artifact_before_parsing():
    assert parse_choice_index(">A", LETTERS, CHOICES) == 0
    assert parse_choice_index(">B", LETTERS, CHOICES) == 1
