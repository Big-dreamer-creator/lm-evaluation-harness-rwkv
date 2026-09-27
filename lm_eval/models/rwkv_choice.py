"""Shared "loglikelihood-via-generation" helpers for RWKV backends without
native echo/logprob support (or, for avg@k, that deliberately route
``multiple_choice`` scoring through generation to exercise real sampling).

Extracted from the original ``RWKV7HTTP.loglikelihood``/``_resolve_choice``
implementation so ``RWKV7Remote`` can reuse the same lettered-choice prompt
and answer-parsing logic for its avg@k mode.

The answer-extraction cascade below mirrors the approach used by RWKV's own
GPQA-Diamond generative-scoring eval (which separates a "final answer"
segment from the model's chain-of-thought, tries several tiers of
progressively looser regexes against it, and falls back to the
chain-of-thought text itself before giving up): letter mentioned earlier in
a long chain-of-thought must not shadow the letter the model actually
settles on at the end.
"""

import re
from typing import Any, List, Optional, Tuple

_CHOICE_BOXED_RES = (
    re.compile(r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}"),
    re.compile(r"\\boxed\{\s*\\(?:text|mathrm)\{\s*\(?([A-Za-z])\)?\s*\}\s*\}"),
)
_CHOICE_EXPLICIT_RES = (
    re.compile(
        r"(?:final\s+answer|correct\s+answer|the\s+answer|answer)\s*(?:is\s*[:\-]?\s*|[:\-]\s*)\(?([A-Za-z])\)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\(?([A-Za-z])\)?\s+is\s+(?:the\s+)?(?:final\s+|correct\s+)?answer",
        re.IGNORECASE,
    ),
)
_CHOICE_FALLBACK_RES = (
    re.compile(r"\b(?:choose|select|pick)\s*[:\-]?\s*\(?([A-Za-z])\)?\b", re.IGNORECASE),
    re.compile(r"\b(?:corresponds?|maps?)\s+to\s*\(?([A-Za-z])\)?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:therefore|thus|hence|so|consequently)[,:]?\s+(?:the\s+)?(?:correct\s+|final\s+)?(?:answer|choice|option)\s+(?:is|would\s+be)\s*\(?([A-Za-z])\)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\(?([A-Za-z])\)?\s+(?:is|would\s+be)\s+(?:the\s+)?(?:best|correct|final)\s+(?:answer|choice|option)\b",
        re.IGNORECASE,
    ),
)
_CHOICE_BARE_LINE_RE = re.compile(r"^\s*\(?([A-Za-z])\)?[.\s]*$")


class RWKVLoglikelihoodResponse(tuple):
    """A ``(loglikelihood, is_greedy)`` 2-tuple that also carries the real
    generation this score was derived from.

    ``RWKV7Remote``/``RWKV7RemoteChat`` resolve ``loglikelihood`` requests by
    generating a real answer (``generate_until``) rather than echo/logprobs,
    so a genuine ``prompt`` (the ``build_choice_prompt()`` text actually
    sent) and ``raw_response``/``finish_reason``/``output_token_ids`` exist
    for every score -- unlike a bare ``(ll, is_greedy)`` tuple, this type
    keeps that evidence attached all the way through to Scoreboard
    publication instead of forcing downstream code to reconstruct a fake
    prompt/completion from the score. ``raw_response``/``finish_reason``/
    ``output_token_ids``/``truncated`` are picked up by ``evaluator.py``'s
    ``response_evidence()`` via ``getattr(response, "raw_response", None)``;
    ``prompt`` is a Scoreboard-specific addition consumed directly (see
    below).

    Subclasses ``tuple`` (rather than adding extra ``NamedTuple`` fields) so
    it unpacks, ``zip()``s, and ``isinstance(x, tuple)``-checks exactly like
    a bare 2-tuple everywhere the harness already expects one (e.g.
    ``task.py``'s ``ll, is_greedy = results`` and
    ``zip(*results, strict=True)``, ``checkpoint.py``'s
    ``isinstance(response, tuple)``) -- no existing consumer needs to
    change. The evidence lives in extra instance attributes, mirroring how
    ``_RWKVCompletion`` carries ``raw_response`` alongside its ``str``
    payload for ``generate_until``.
    """

    def __new__(
        cls,
        loglikelihood: float,
        is_greedy: bool,
        *,
        prompt: Optional[str] = None,
        raw_response: Optional[Any] = None,
        finish_reason: Optional[str] = None,
        output_token_ids: Optional[List[int]] = None,
        truncated: bool = False,
    ) -> "RWKVLoglikelihoodResponse":
        self = super().__new__(cls, (loglikelihood, is_greedy))
        self.prompt = prompt
        self.raw_response = raw_response
        self.finish_reason = finish_reason
        self.output_token_ids = output_token_ids
        self.truncated = truncated
        return self


def build_choice_prompt(context: str, choices: List[str]) -> str:
    """Render a lettered multiple-choice prompt for generative resolution.

    ``context`` is the echo-style loglikelihood context, which for tasks
    defining a ``gen_prefix`` (e.g. ``discrim_eval``) already ends mid-sentence
    (a dangling open quote meant to be continued directly by one of
    ``choices``). Appending a fresh lettered menu after that produces a
    malformed prompt -- the model is left with an unclosed quote immediately
    followed by an unrelated instruction block -- so in that case each
    ``context + choice`` is rendered as its own complete option instead.
    """
    letters = choice_letters(choices)
    if _expects_direct_continuation(context):
        options = "\n".join(
            f"{letter}. {context}{choice}" for letter, choice in zip(letters, choices)
        )
        return (
            f"Which of the following best completes the passage below?\n\n"
            f"{options}\n\nAnswer with only the letter of the correct option."
        )
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    return f"{context}\n\n{options}\n\nAnswer with only the letter of the correct option."


def _expects_direct_continuation(context: str) -> bool:
    """Whether ``context`` ends mid-sentence (e.g. a dangling open quote left
    by a task's ``gen_prefix``) such that a choice is meant to continue it
    directly, rather than ``context`` being a complete, freestanding
    question ready for a fresh instruction to follow."""
    stripped = context.rstrip()
    if not stripped:
        return False
    return stripped[-1] in "\"'"


def choice_letters(choices: List[str]) -> List[str]:
    return [chr(ord("A") + i) for i in range(len(choices))]


def _last_match(text: str, patterns) -> str:
    """Pool every match from every pattern in ``patterns`` and return the
    letter belonging to the one that starts latest in ``text`` (i.e. the
    final thing the model said), or ``None`` if nothing matched."""
    matches: List[Tuple[int, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.group(1).upper()))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _last_bare_line_letter(text: str) -> str:
    """Scan lines from the end for one that is just a bare (optionally
    parenthesized/dotted) letter, e.g. a trailing "B." on its own line."""
    for line in reversed(text.splitlines()):
        match = _CHOICE_BARE_LINE_RE.match(line)
        if match:
            return match.group(1).upper()
    return None


def _substring_fallback(text: str, choices: List[str]) -> int:
    lowered = text.lower()
    best_index, best_len = 0, -1
    for index, choice in enumerate(choices):
        choice_lower = choice.lower()
        if choice_lower and choice_lower in lowered and len(choice_lower) > best_len:
            best_index, best_len = index, len(choice_lower)
    return best_index if best_len >= 0 else -1


def parse_choice_index(generated: str, letters: List[str], choices: List[str]) -> int:
    """Best-effort extraction of which lettered choice a generation names.

    Returns the chosen index, or ``-1`` if no letter could be extracted with
    any confidence -- callers decide how to treat an unparseable answer
    (e.g. excluding it from a majority vote) rather than this function
    silently guessing.
    """
    text = generated.strip()

    # Chain-of-thought models may wrap their reasoning in <think>...</think>
    # before stating a final answer; parse the concluding segment first so an
    # earlier, since-corrected mention of a wrong letter can't shadow the
    # letter the model actually settles on.
    think_close = text.rfind("</think>")
    if think_close >= 0:
        think_text = text[:think_close]
        answer_text = text[think_close + len("</think>") :]
    else:
        think_text = ""
        answer_text = text

    for tier in (_CHOICE_BOXED_RES, _CHOICE_EXPLICIT_RES, _CHOICE_FALLBACK_RES):
        letter = _last_match(answer_text, tier)
        if letter is not None and letter in letters:
            return letters.index(letter)

    letter = _last_bare_line_letter(answer_text)
    if letter is not None and letter in letters:
        return letters.index(letter)

    # The concluding segment didn't contain a recognizable answer (e.g. the
    # generation was truncated by max_gen_toks before it finished its final
    # statement) -- fall back to the same tiers against the chain-of-thought
    # itself, which may already state the intended answer.
    if think_text:
        for tier in (_CHOICE_BOXED_RES, _CHOICE_EXPLICIT_RES, _CHOICE_FALLBACK_RES):
            letter = _last_match(think_text, tier)
            if letter is not None and letter in letters:
                return letters.index(letter)

    return _substring_fallback(text, choices)
