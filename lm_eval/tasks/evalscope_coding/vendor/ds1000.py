# Adapted from ModelScope/EvalScope, Apache-2.0; see ../NOTICE.md.
import re


_THINK_RE = re.compile("<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(
    "```[ \\t]*(?:python|py)?[ \\t]*\\n(.*?)```", re.IGNORECASE | re.DOTALL
)
_STOP_MARKERS = ("</code>", "\nEND SOLUTION", "# SOLUTION END")


def extract_code(completion: str) -> str:
    """Return the solution code of a completion.

    The DS-1000 prompt ends with ``BEGIN SOLUTION\\n<code>`` and the official harness keeps the text up to
    ``</code>`` (or ``END SOLUTION`` / ``# SOLUTION END``), dropping ``<code>`` tags and markdown fences.
    Chat models often write prose around the code, so the code after an opening ``<code>`` tag, or else
    the first fenced block, is used when present.
    """
    text = _THINK_RE.sub("", completion or "")
    if "<code>" in text:
        text = text.split("<code>", 1)[1]
    for marker in _STOP_MARKERS:
        text = text.split(marker, 1)[0]
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    elif "```" in text:
        text = re.sub(
            "^\\s*```[ \\t]*(?:python|py)?[ \\t]*\\n?", "", text, flags=re.IGNORECASE
        ).split("```", 1)[0]
    return text.replace("<code>", "")


def build_test_program(code_context: str, code: str) -> str:
    """The official test program: context, the solution as a string, and the test calls."""
    program = f"{code_context}\ncode = {code!r}\ntest_execution(code)\n"
    if "test_string(" in code_context:
        program += "test_string(code)\n"
    return program
