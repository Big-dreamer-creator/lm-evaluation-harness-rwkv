import re


_FINAL_NUMBER = re.compile(r"(?i)final\s+answer\s*[:=]\s*\[?([^\]\n]+)")


def process_gsm8k_platinum_results(doc: dict, results: list[str]) -> dict[str, int]:
    from math_verify import parse, verify

    completion = results[0]
    answer = doc["answer"].split("####")[-1].strip()
    prediction = _FINAL_NUMBER.findall(completion)
    exact_match = int(bool(prediction) and prediction[-1].strip(" .$,") == answer)
    try:
        gold = parse(f"$\\boxed{{{answer}}}$")
        candidate = parse(completion)
        math_verify = int(bool(candidate) and verify(gold, candidate, strict=False))
    except Exception:
        math_verify = 0
    return {"exact_match": exact_match, "math_verify": math_verify}
