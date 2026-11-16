"""Dataset conversion and prompts for the native Coding task family."""

from __future__ import annotations

import base64
import io
import json
import pickle
import re
import zlib
from pathlib import Path

from lm_eval.tasks.evalscope_coding import prompts
from lm_eval.tasks.evalscope_coding.vendor import bird, mtbench, scicode_prompts


CATALOG = json.loads((Path(__file__).parent / "catalog.json").read_text())["benchmarks"]


class _TestCaseUnpickler(pickle.Unpickler):  # noqa: S301 - find_class forbids executable globals.
    def find_class(self, module, name):
        raise ValueError(
            "LiveCodeBench test data must not contain executable pickle globals"
        )


def lcb_sample(doc: dict) -> str:
    """Decode both public and private tests; reject executable pickle payloads."""
    public = json.loads(doc["public_test_cases"])
    try:
        private = json.loads(doc["private_test_cases"])
    except json.JSONDecodeError:
        blob = zlib.decompress(base64.b64decode(doc["private_test_cases"]))
        private = json.loads(_TestCaseUnpickler(io.BytesIO(blob)).load())
    cases = public + private
    if not cases:
        raise ValueError("LiveCodeBench problem has no tests")
    metadata = json.loads(doc["metadata"])
    return json.dumps(
        {
            "inputs": [case["input"] for case in cases],
            "outputs": [case["output"] for case in cases],
            "fn_name": metadata.get("func_name"),
        }
    )


def load_dataset(
    path: str, subset: str | None, benchmark: str, metadata: dict, kwargs: dict
):
    """Load with datasets or the ModelScope hub SDK; EvalScope is never imported."""
    import datasets

    if (
        metadata.get("hub") == "modelscope"
        and path not in {"json", "parquet", "csv", "arrow"}
        and not Path(path).exists()
    ):
        from modelscope.msdatasets import MsDataset

        split = CATALOG[benchmark]["eval_split"]
        dataset = MsDataset.load(
            path,
            subset_name=subset if subset != "default" else None,
            split=split,
            **kwargs,
        )
        if not isinstance(dataset, datasets.Dataset):
            dataset = dataset.to_hf_dataset()
        result = datasets.DatasetDict({split: dataset})
    else:
        result = datasets.load_dataset(path, name=subset, **kwargs)
    if benchmark.startswith("swe_bench") and "text" not in result["test"].column_names:
        # Oracle context is a separate inference input, never the gold patch.
        oracle_kwargs = dict(metadata.get("oracle_dataset_kwargs", {}))
        oracle = datasets.load_dataset(
            metadata.get("oracle_dataset", "princeton-nlp/SWE-bench_oracle"),
            split="test",
            **oracle_kwargs,
        )
        contexts = {row["instance_id"]: row["text"] for row in oracle}
        result["test"] = result["test"].map(
            lambda row: {"text": contexts[row["instance_id"]]}
        )
    return result


def select_docs(dataset, benchmark: str, metadata: dict):
    """Subset selection is explicit and retains the original problem population."""
    subset = metadata.get("subset")
    if benchmark == "ds1000":
        dataset = dataset.filter(lambda row: row["metadata"]["library"] == subset)
    elif benchmark == "bird_sql":
        dataset = dataset.filter(lambda row: row["difficulty"] == subset)
    elif benchmark == "mt_bench":
        dataset = dataset.filter(lambda row: row["category"] == "coding")
    elif benchmark == "live_code_bench":
        start, end = metadata.get("start_date"), metadata.get("end_date")
        if start or end:
            dataset = dataset.filter(
                lambda row: (
                    (not start or str(row["contest_date"])[:10] >= start)
                    and (not end or str(row["contest_date"])[:10] <= end)
                )
            )
    if not len(dataset):
        raise ValueError(f"{benchmark}/{subset}: empty evaluation population")
    return dataset


def bird_database(doc: dict, metadata: dict) -> Path:
    root = metadata.get("db_root")
    if not root:
        raise ValueError("BIRD-SQL requires metadata.db_root pointing to dev_databases")
    if not re.fullmatch(r"[\w-]+", doc["db_id"]):
        raise ValueError("Invalid BIRD database identifier")
    path = bird.db_file(root, doc["db_id"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def initial_messages(
    doc: dict, benchmark: str, metadata: dict, shots: int = 0
) -> list[dict]:
    """Build the exact upstream user/system content before a model's chat template."""
    spec = CATALOG[benchmark]
    system = spec["system_prompt"]
    if benchmark in {"humaneval", "humaneval_plus"}:
        query = spec["prompt_template"].format(question=doc["prompt"])
    elif benchmark in {"mbpp", "mbpp_plus"}:
        if shots not in {0, 3} or (benchmark == "mbpp_plus" and shots):
            raise ValueError("MBPP supports 0 or fixed 3 shots; MBPP+ supports 0")
        template = prompts.FEWSHOT_PROMPT if shots == 3 else prompts.PROMPT
        query = template.format(
            question=doc["text"] if "text" in doc else doc["prompt"],
            tests="\n".join(doc["test_list"]),
        )
    elif benchmark.startswith("multiple_"):
        from .vendor.multiple import normalize_languages

        language = doc.get("language") or metadata["subset"].split("-", 1)[1]
        extract_lang, _ = normalize_languages(language)
        query = (
            f"```{extract_lang}\n{doc['prompt']}\n```\n\n"
            "Please complete the above code according to the requirements in the docstring. "
            "Write the complete code and wrap it in markdown fenced code. The code should not contain `Main` function."
        )
    elif benchmark.startswith("bigcodebench"):
        query = doc[metadata.get("bcb_split", "instruct") + "_prompt"]
    elif benchmark == "live_code_bench":
        constants = prompts.CodeGenerationPromptConstants
        if doc.get("starter_code"):
            formatting = f"### Format: {constants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n```python\n{doc['starter_code']}\n```\n\n"
        else:
            formatting = f"### Format: {constants.FORMATTING_WITHOUT_STARTER_CODE}\n```python\n# YOUR CODE HERE\n```\n\n"
        query = spec["prompt_template"].format(
            question_content=doc["question_content"], format_prompt=formatting
        )
    elif benchmark == "scicode":
        if not doc["sub_steps"]:
            raise ValueError("SciCode problem has no subproblems")
        background = metadata.get("provide_background", False)
        template = (
            scicode_prompts.INITIAL_PROMPT_PROVIDE_BACKGROUND
            if background
            else scicode_prompts.INITIAL_PROMPT
        )
        system = template.format(required_dependencies=doc["required_dependencies"])
        query = scicode_turn(doc, metadata, 0)
    elif benchmark == "mt_bench":
        if len(doc["prompt"]) != 2 or len(mtbench_references(doc)) != 2:
            raise ValueError(
                "MT-Bench Coding requires two turns and two reference answers"
            )
        query = doc["prompt"][0]
    elif benchmark.startswith("swe_bench"):
        query = doc["text"]
    elif benchmark == "ds1000":
        query = doc["prompt"]
    elif benchmark == "bird_sql":
        schema = bird.schema_prompt(
            bird_database(doc, metadata), metadata.get("sample_rows", 0)
        )
        query = bird.build_prompt(schema, doc["question"], doc.get("evidence") or None)
    else:
        raise ValueError(benchmark)
    return ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": query}
    ]


def scicode_turn(doc: dict, metadata: dict, turn: int) -> str:
    template = (
        scicode_prompts.SUBPROBLEM_PROMPT_PROVIDE_BACKGROUND
        if metadata.get("provide_background")
        else scicode_prompts.SUBPROBLEM_PROMPT
    )
    return template.format(**doc["sub_steps"][turn])


def mtbench_references(doc: dict) -> list[str]:
    return doc.get("reference") or mtbench.FALLBACK_REFERENCES.get(doc["prompt_id"], [])


def target(doc: dict, benchmark: str) -> str:
    if benchmark.startswith("multiple_"):
        return doc["tests"]
    if benchmark.startswith("swe_bench"):
        return doc["patch"]
    if benchmark == "bird_sql":
        return doc["SQL"]
    if benchmark == "mt_bench":
        return json.dumps(mtbench_references(doc), ensure_ascii=False)
    if benchmark == "scicode":
        return json.dumps(
            [step["test_cases"] for step in doc["sub_steps"]], ensure_ascii=False
        )
    if benchmark == "live_code_bench":
        return lcb_sample(doc)
    return doc.get("canonical_solution", doc.get("reference_code", doc.get("code", "")))
