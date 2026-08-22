from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


_ALBATROSS_ANSWER_MARKUP = re.compile(r"\*\*|__|`")


def _albatross_last_answer(
    text: str, patterns: tuple[re.Pattern[str], ...]
) -> str | None:
    matches = [
        (match.start(), match.group(1).upper())
        for pattern in patterns
        for match in pattern.finditer(text)
    ]
    return max(matches, key=lambda item: item[0])[1] if matches else None


def extract_albatross_mcq_answer(
    text: str,
    *,
    choice_labels: str = "ABCD",
    require_think_close: bool = False,
) -> str | None:
    """Extract the final letter using Albatross GPQA-style precedence."""
    if not isinstance(text, str):
        return None
    labels = "".join(dict.fromkeys(choice_labels.upper()))
    if not labels or any(not label.isalpha() or len(label) != 1 for label in labels):
        raise ValueError("choice_labels must contain unique alphabetic characters")
    if require_think_close and "</think>" not in text:
        return None

    label_class = re.escape(labels)
    answer_patterns = (
        re.compile(
            rf"\\boxed\s*\{{\s*(?:\\(?:text|mathrm)\s*\{{\s*)?"
            rf"\(?\s*([{label_class}])\s*\)?\s*\}}?\s*\}}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?i:(?:final\s+answer|correct\s+answer|answer)\s*"
            rf"(?:(?:choice|option)\s*)?(?:is\s*|[:=]\s*)"
            rf"(?:(?:choice|option)\s*)?\(?\s*)([{label_class}])"
            rf"(?i:\s*\)?)"
        ),
        re.compile(
            rf"(?i:(?:(?:choice|option)\s*)?\(?\s*)([{label_class}])"
            rf"(?i:\s*\)?\s+is\s+(?:the\s+)?"
            rf"(?:final\s+|correct\s+)?answer)"
        ),
    )
    fallback_patterns = (
        re.compile(
            rf"(?i:\b(?:choose|select|pick)\s+"
            rf"(?:(?:choice|option|answer)\s*)?[:=]?\s*\(?\s*)"
            rf"([{label_class}])(?i:\s*\)?\b)"
        ),
        re.compile(
            rf"(?i:\b(?:corresponds?|maps?)\s+to\s+"
            rf"(?:(?:choice|option|answer)\s*)?\(?\s*)"
            rf"([{label_class}])(?i:\s*\)?\b)"
        ),
        re.compile(
            rf"(?i:\b(?:therefore|thus|hence|so|consequently)[,:]?\s+"
            rf"(?:the\s+)?(?:(?:correct|final)\s+)?"
            rf"(?:answer|choice|option)\s+(?:is|would\s+be)\s+\(?\s*)"
            rf"([{label_class}])(?i:\s*\)?\b)"
        ),
        re.compile(
            rf"(?i:\b\(?\s*)([{label_class}])"
            rf"(?i:\s*\)?\s+(?:is|would\s+be)\s+(?:the\s+)?"
            rf"(?:best|correct|final)\s+(?:answer|choice|option)\b)"
        ),
    )
    think_final_pattern = re.compile(
        rf"(?i:\b(?:final\s+answer|correct\s+(?:answer|choice|option))\s*"
        rf"(?:is\s*|[:=]\s*)?(?:\\boxed\s*\{{\s*)?"
        rf"(?:\\(?:text|mathrm)\s*\{{\s*)?\(?\s*)"
        rf"([{label_class}])(?i:\s*\)?)"
    )

    before_think, think_closed, answer_text = text.rpartition("</think>")
    answer_text = _ALBATROSS_ANSWER_MARKUP.sub(
        "", answer_text if think_closed else text
    )
    answer = _albatross_last_answer(answer_text, answer_patterns)
    if answer is None:
        answer = _albatross_last_answer(answer_text, fallback_patterns)
    if answer is not None:
        return answer
    for line in reversed(answer_text.splitlines()):
        match = re.fullmatch(
            rf"\s*(?:final\s+answer\s*[:=]?\s*)?"
            rf"[\[(]?([{label_class}])[\])]?[.!]?\s*",
            line,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
    if think_closed:
        matches = list(
            think_final_pattern.finditer(_ALBATROSS_ANSWER_MARKUP.sub("", before_think))
        )
        if matches:
            return matches[-1].group(1).upper()
    return None


@register_filter("albatross_mcq")
class AlbatrossMultiChoiceFilter(Filter):
    """Apply the authoritative Albatross final-answer extraction order."""

    def __init__(
        self,
        choice_labels: str = "ABCD",
        require_think_close: bool = False,
        fallback: str = "[invalid]",
        regex_pattern: str | None = None,
    ) -> None:
        if regex_pattern:
            legacy_labels = re.search(r"\[([A-Z]+)\]", regex_pattern)
            if legacy_labels:
                choice_labels = legacy_labels.group(1)
        self.choice_labels = choice_labels
        self.require_think_close = require_think_close
        self.fallback = fallback
        self.regex_pattern = (
            re.compile(regex_pattern, re.IGNORECASE) if regex_pattern else None
        )

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        return [
            [
                (
                    _albatross_last_answer(response, (self.regex_pattern,))
                    if self.regex_pattern is not None
                    and (not self.require_think_close or "</think>" in response)
                    else None
                )
                or extract_albatross_mcq_answer(
                    response,
                    choice_labels=self.choice_labels,
                    require_think_close=self.require_think_close,
                )
                or self.fallback
                for response in response_set
            ]
            for response_set in resps
        ]


@register_filter("regex")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """Compile `regex_pattern` and set the fallback for non-matches.

        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> Iterable[list[str]]:
        def filter_set(inst: Sequence[str]) -> list[str]:
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        return [filter_set(x) for x in resps]


@register_filter("regex_pos")
class POSFilter(Filter):
    """Extract part-of-speech tags from model responses."""

    def __init__(
        self,
        regex_pattern: str = r"\['(.*?)'\]",
        group_select=0,
        fallback=None,
    ) -> None:
        """Compile `regex_pattern` and set the fallback for non-matches.

        `fallback` defines the output returned if no matches for the regex are located.
        """
        if fallback is None:
            fallback = ["invalid"]
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        def extract_tagged_tokens(text):
            # Extract tagged tokens list from text input using regex
            tokens = re.findall(r"\('([^']*)', '([^']*)'\)", text)
            return [(token, pos) for token, pos in tokens]

        def extract_pos_tags(result):
            pos_tags = []
            if isinstance(result, str):
                result = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result)
            return pos_tags or self.fallback

        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = extract_pos_tags(resp)
                filtered.append(match)
            return filtered

        return (filter_set(x) for x in resps)


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filters out leading and trailing whitespace from responses."""

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        def filter_set(inst: Sequence[str]) -> list[str]:
            return [resp.strip() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """Extract a model's answer on multiple choice questions with letter answers.

    Assumes each document has a "choices" field containing the list of answer choices
    and that the answer label symbols are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        regexes_to_ignore: list[str] | None = None,
    ) -> None:
        r"""Configure the multi-choice regex filter.

        Args:
            regex_pattern: The basic regex pattern to use. If it fails to match,
                a customized procedure is used:
                step 1 — parse choices between ([A-Z])s and search in the response.
                step 2 — parse with regex ``r'\s*([A-?])'``, where ``?`` varies by
                number of choices.
            group_select: Selects the (group_select)th match from the findall result.
            ignore_case: Ignore case during step 1 matching.
            ignore_punctuation: Remove punctuation during step 1 matching.
            regexes_to_ignore: Remove these regexes during step 1 matching.
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        import unicodedata

        def find_match(regex, resp, convert_dict: dict[str, str] | None = None):
            if convert_dict is None:
                convert_dict = {}
            if not isinstance(resp, str):
                resp = ""
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    non_empty = [m for m in match if m]
                    if not non_empty:
                        return ""
                    match = non_empty[0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs, strict=True):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps
