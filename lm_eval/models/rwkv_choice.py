"""Shared "loglikelihood-via-generation" helpers for RWKV backends without
native echo/logprob support (or, for avg@k, that deliberately route
``multiple_choice`` scoring through generation to exercise real sampling).

Extracted from the original ``RWKV7HTTP.loglikelihood``/``_resolve_choice``
implementation so ``RWKV7Remote`` can reuse the same lettered-choice prompt
and answer-parsing logic for its avg@k mode.
"""

import re
from typing import List

_CHOICE_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}")
_CHOICE_EXPLICIT_RE = re.compile(
    r"(?:answer|answer is|the answer is|answer:)\s*[:\-]?\s*\(?([A-Za-z])\)?",
    re.IGNORECASE,
)
_CHOICE_BARE_RE = re.compile(r"\(?([A-Za-z])\)?[.\s]*$")


def build_choice_prompt(context: str, choices: List[str]) -> str:
    """Render a lettered multiple-choice prompt for generative resolution."""
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    return f"{context}\n\n{options}\n\nAnswer with only the letter of the correct option."


def choice_letters(choices: List[str]) -> List[str]:
    return [chr(ord("A") + i) for i in range(len(choices))]


def parse_choice_index(generated: str, letters: List[str], choices: List[str]) -> int:
    """Best-effort extraction of which lettered choice a generation names."""
    text = generated.strip()
    for pattern in (_CHOICE_BOXED_RE, _CHOICE_EXPLICIT_RE, _CHOICE_BARE_RE):
        match = pattern.search(text)
        if match:
            letter = match.group(1).upper()
            if letter in letters:
                return letters.index(letter)

    lowered = text.lower()
    best_index, best_len = 0, -1
    for index, choice in enumerate(choices):
        choice_lower = choice.lower()
        if choice_lower and choice_lower in lowered and len(choice_lower) > best_len:
            best_index, best_len = index, len(choice_lower)
    return best_index if best_len >= 0 else -1
