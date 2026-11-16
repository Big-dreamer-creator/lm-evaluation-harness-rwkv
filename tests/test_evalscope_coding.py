"""Native Coding contract tests; fixtures require neither datasets nor a live model."""

import base64
import copy
import json
import os
import pickle
import sqlite3
import sys
import zlib
from types import ModuleType
from unittest.mock import Mock

import datasets
import pytest

from lm_eval.api.model import LM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import TaskManager
from lm_eval.tasks.evalscope_coding import data, runtime
from lm_eval.tasks.evalscope_coding.task import Coding, subproblem_pass_rate
from lm_eval.tasks.evalscope_coding.vendor import multiple


@pytest.fixture
def records(tmp_path):
    database = tmp_path / "tiny"
    database.mkdir()
    with sqlite3.connect(database / "tiny.sqlite") as conn:
        conn.execute("create table numbers(value integer)")
        conn.executemany("insert into numbers values(?)", [(2,), (3,)])
    return {
        "task_id": "test/0",
        "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
        "canonical_solution": "    return a + b\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "text": "Write a function add(a, b) that adds two numbers.",
        "code": "def add(a,b): return a+b",
        "test_list": ["assert add(2,3) == 5"],
        "test_imports": [],
        "test_setup_code": "",
        "source_file": "fixture",
        "tests": "assert add(2,3) == 5",
        "language": "py",
        "complete_prompt": "def add(a, b):\n",
        "instruct_prompt": "Implement add(a,b).",
        "code_prompt": "def add(a, b):",
        "question_content": "Add the given numbers.",
        "starter_code": "",
        "public_test_cases": '[{"input":"2 3\\n","output":"5\\n"}]',
        "private_test_cases": '[{"input":"4 5\\n","output":"9\\n"}]',
        "contest_date": "2025-01-01T00:00:00",
        "metadata": "{}",
        "reference_code": "result = 5",
        "code_context": "def test_execution(code):\n    ns = {}\n    exec(code,ns)\n    assert ns['result'] == 5",
        "db_id": "tiny",
        "SQL": "select sum(value) from numbers",
        "question": "What is their sum?",
        "question_id": 0,
        "evidence": "",
        "difficulty": "simple",
        "problem_id": "1",
        "required_dependencies": "",
        "sub_steps": [
            {
                "step_number": "1.1",
                "step_description_prompt": "Add numbers.",
                "function_header": "def add(a,b):",
                "return_line": "return result",
                "test_cases": ["assert add(2,3)==5"],
                "step_background": "Addition",
            },
            {
                "step_number": "1.2",
                "step_description_prompt": "Double numbers using add.",
                "function_header": "def twice(a):",
                "return_line": "return result",
                "test_cases": ["assert twice(2)==4"],
                "step_background": "Multiplication",
            },
        ],
        "instance_id": "repo__issue-1",
        "patch": "diff --git a/a b/a\n",
        "problem_statement": "Fix a bug",
    }


def task_config(benchmark, **metadata):
    spec = data.CATALOG[benchmark]
    return {
        "task": "evalscope_" + benchmark,
        "dataset_path": "json",
        "dataset_kwargs": {},
        "metadata": {
            "benchmark": benchmark,
            "subset": spec["subset_list"][0],
            **metadata,
        },
        "num_fewshot": spec["few_shot_num"],
    }


def build_task(monkeypatch, records, benchmark, **metadata):
    row = copy.deepcopy(records)
    if benchmark == "ds1000":
        row["metadata"] = {
            "library": "Pandas",
            "problem_id": 0,
            "perturbation_type": "Origin",
        }
    if benchmark == "mt_bench":
        row.update(
            prompt=["Write add(a,b)", "Now double using add"],
            prompt_id=0,
            category="coding",
            reference=["add code", "twice code"],
        )
    if benchmark == "mbpp_plus":
        row["test"] = "assert add(-3, 2) == -1"
    split = data.CATALOG[benchmark]["eval_split"]
    monkeypatch.setattr(
        data,
        "load_dataset",
        lambda *a, **kw: datasets.DatasetDict(
            {split: datasets.Dataset.from_list([row])}
        ),
    )
    return Coding(task_config(benchmark, **metadata))


@pytest.mark.parametrize("benchmark", list(data.CATALOG))
def test_every_non_agent_benchmark_is_a_native_task(
    monkeypatch, records, tmp_path, benchmark
):
    task = build_task(monkeypatch, records, benchmark, db_root=str(tmp_path))
    assert task.OUTPUT_TYPE == "generate_until"
    assert task.UNSAFE_CODE
    assert task.doc_to_text(task.eval_docs[0])
    assert not any(
        name == "evalscope" or name.startswith("evalscope.") for name in sys.modules
    )


def test_catalog_and_all_selectors_load_without_network(monkeypatch, records, tmp_path):
    manager = TaskManager()
    selected = [name for name in manager.all_subtasks if name.startswith("evalscope_")]
    assert len(selected) == 87
    for name in selected:
        entry = manager._factory._load_full_config(manager.task_index[name], None)
        benchmark = entry["metadata"]["benchmark"]
        task = build_task(monkeypatch, records, benchmark, db_root=str(tmp_path))
        # Use the exact YAML through TaskManager, substituting only an in-memory dataset.
        if benchmark in {"ds1000", "bird_sql"}:
            entry["metadata"]["subset"] = task.options["subset"]
        entry["metadata"]["db_root"] = str(tmp_path)
        loaded = manager.load([entry])["tasks"][name]
        assert loaded.config.task == name


def test_mbpp_fixed_three_shot_and_zero_shot(monkeypatch, records):
    task = build_task(monkeypatch, records, "mbpp")
    context = task.fewshot_context(records, 3)
    assert context.count("[DONE]") == 3
    assert context.endswith("[BEGIN]\n")
    assert "[DONE]" not in task.fewshot_context(records, 0)
    with pytest.raises(ValueError):
        task.fewshot_context(records, 1)


def test_private_lcb_tests_are_not_lost(records):
    cases = json.loads(data.lcb_sample(records))
    assert len(cases["inputs"]) == 2
    records["private_test_cases"] = base64.b64encode(
        zlib.compress(pickle.dumps(records["private_test_cases"]))
    ).decode()
    assert json.loads(data.lcb_sample(records)) == cases
    records["private_test_cases"] = base64.b64encode(
        zlib.compress(b"cos\nsystem\n.")
    ).decode()
    with pytest.raises(ValueError, match="globals"):
        data.lcb_sample(records)


def test_pass_k_uses_all_repeats(monkeypatch, records):
    task = build_task(monkeypatch, records, "humaneval", pass_k=[1, 2])
    monkeypatch.setattr(
        task, "score_one", lambda doc, response: {"passed": response == "yes"}
    )
    assert task.process_results(records, [["yes", "no", "no", "yes"]]) == pytest.approx(
        {"pass@1": 0.5, "pass@2": 5 / 6}
    )
    with pytest.raises(ValueError):
        task.process_results(records, [["yes"]])


def test_plus_suites_include_augmented_tests(monkeypatch, records):
    task = build_task(monkeypatch, records, "mbpp_plus")
    captured = []
    monkeypatch.setattr(
        task,
        "_python",
        lambda code, tests: captured.append((code, tests)) or {"passed": True},
    )
    task.score_one(task.eval_docs[0], "```python\ndef add(a,b): return a+b\n```")
    assert "assert add(-3, 2) == -1" in captured[0][1]
    assert "assert add(2,3) == 5" in captured[0][1]


def test_open_think_requires_close_and_preserves_code_literals(monkeypatch, records):
    task = build_task(monkeypatch, records, "humaneval", thinking="open_think")
    assert task.score_one(records, ">reasoning") == {
        "passed": False,
        "answer": None,
        "status": "unfinished_thinking",
    }
    assert task._final('>reason</think>code = "</think>"') == 'code = "</think>"'


def test_bird_is_readonly_and_scores_execution(monkeypatch, records, tmp_path):
    task = build_task(monkeypatch, records, "bird_sql", db_root=str(tmp_path))
    assert task.score_one(records, "```sql\nselect 5\n```")["passed"]
    assert not task.score_one(records, "delete from numbers")["passed"]
    assert task.score_one(records, "select sum(value) from numbers")["passed"]
    records["SQL"] = "select missing from numbers"
    with pytest.raises(runtime.JudgeError):
        task.score_one(records, "select 5")


def test_sandbox_infrastructure_errors_are_not_wrong_answers(monkeypatch):
    response = Mock()
    response.json.return_value = {"status": "SandboxError"}
    monkeypatch.setattr(runtime.requests, "post", Mock(return_value=response))
    with pytest.raises(runtime.JudgeError):
        runtime.multiple_judge("bad", "cpp", {"sandbox_url": "http://localhost:8123"})
    response.json.return_value = {
        "status": "Failed",
        "compile_result": {"status": "Finished", "return_code": 1},
    }
    assert not runtime.multiple_judge(
        "bad", "cpp", {"sandbox_url": "http://localhost:8123"}
    )["passed"]


def test_multipl_java_keeps_methods_and_tests_inside_class():
    code = multiple.build_full_code(
        "```java\npublic class Problem { public static int add(int a,int b) { return a+b; } }\n```",
        "java",
        [],
        "} public static void main(String[] args) { assert add(2,3)==5; } }",
    )
    assert "return a+b" in code and "assert add(2,3)==5" in code
    assert "class Main" in code


class ScriptedLM(LM):
    def __init__(self, replies):
        super().__init__()
        self.replies = iter(replies)
        self.calls = []

    def generate_until(self, requests, **kwargs):
        self.calls.extend(requests)
        return [next(self.replies) for _ in requests]

    def loglikelihood(self, requests):
        raise AssertionError("Coding must generate")

    def loglikelihood_rolling(self, requests):
        raise AssertionError("Coding must generate")

    def apply_chat_template(self, history, **kwargs):
        return json.dumps(history)

    @property
    def tokenizer_name(self):
        return "fixture"


def test_native_evaluator_retains_multiturn_and_details(monkeypatch, records):
    task = build_task(monkeypatch, records, "mt_bench")
    monkeypatch.setattr(task, "validate_runtime", lambda: None)
    monkeypatch.setattr(
        runtime,
        "mtbench_judge",
        lambda turns, answers, references, meta: {
            "judge_score": 8.5,
            "first_turn_judge_score": 8,
            "second_turn_judge_score": 9,
            "judge_evidence": answers,
        },
    )
    model = ScriptedLM(["first code", "second code"])
    result = evaluate(
        model,
        {"tasks": {task.config.task: task}},
        bootstrap_iters=0,
        apply_chat_template=True,
        log_samples=True,
        confirm_run_unsafe_code=True,
    )
    assert len(model.calls) == 2
    assert "first code" in model.calls[1].args[0]
    assert model.calls[1].idx == 1
    samples = result["samples"][task.config.task]
    assert samples[0]["scoring_details"][0]["judge_evidence"] == [
        "first code",
        "second code",
    ]
    assert result["results"][task.config.task]["judge_score,all"] == 8.5


def test_scicode_weighted_subproblem_aggregation():
    assert subproblem_pass_rate([(1, 2), (1, 8)]) == 0.2


@pytest.mark.parametrize(
    "benchmark",
    [
        "humaneval",
        "humaneval_plus",
        "mbpp",
        "mbpp_plus",
        "bigcodebench",
        "live_code_bench",
        "ds1000",
    ],
)
def test_docker_known_pass_and_fail(monkeypatch, records, benchmark):
    task = build_task(monkeypatch, records, benchmark, docker_image="python:3.11-slim")
    if benchmark == "bigcodebench":
        records["test"] = (
            "import unittest\nclass TestCases(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(2,3), 5)"
        )
    if benchmark == "live_code_bench":
        # The official LCB grader imports numpy even for stdio fixtures.
        task.options["docker_image"] = os.environ.get(
            "LMEVAL_CODING_TEST_NUMPY_IMAGE", "lm-eval-coding-numpy:1"
        )
        good, bad = (
            "```python\na,b=map(int,input().split());print(a+b)\n```",
            "```python\nprint(0)\n```",
        )
    elif benchmark == "ds1000":
        good, bad = "<code>result = 5</code>", "<code>result = 0</code>"
    else:
        good, bad = (
            "```python\ndef add(a,b): return a+b\n```",
            "```python\ndef add(a,b): return 0\n```",
        )
    if benchmark == "mbpp_plus":
        records["test"] = "assert add(-3, 2) == -1"
    try:
        task.validate_runtime()
    except runtime.JudgeError as error:
        pytest.skip(str(error))
    assert task.score_one(records, good)["passed"]
    assert not task.score_one(records, bad)["passed"]


def test_multiturn_partial_checkpoint_resumes_only_missing_turn(
    monkeypatch, records, tmp_path
):
    from lm_eval.api.checkpoint import CheckpointStore, serialize_response
    from lm_eval.evaluator import _build_checkpointed_dispatch

    store = CheckpointStore(str(tmp_path / "resume.sqlite3"), "test")
    task = build_task(monkeypatch, records, "mt_bench")
    task.build_all_requests(
        rank=0,
        world_size=1,
        apply_chat_template=True,
        chat_template=json.dumps,
        tokenizer_name="fixture",
    )
    dispatched = []

    def dispatch(requests, fail_second=False):
        pending, keys = _build_checkpointed_dispatch(
            store,
            task.config.task,
            "task-fp",
            "model-fp",
            requests["generate_until"],
            "generate_until",
        )
        for request, key in zip(pending, keys, strict=True):
            if fail_second and request.idx > 0:
                raise RuntimeError("simulated connection failure")
            reply = f"answer for turn {request.idx}"
            dispatched.append(request.idx)
            store.mark_items_completed([(key.item_id, serialize_response(reply))])
            request.resps[key.repeat_ord] = reply

    with pytest.raises(RuntimeError):
        task.run_generation_rounds(
            lambda requests: dispatch(requests, fail_second=True)
        )
    assert dispatched == [0]
    task.run_generation_rounds(dispatch)
    assert dispatched == [0, 1]
    conversation = json.loads(task.instances[0].resps[0])
    assert conversation["answers"] == ["answer for turn 0", "answer for turn 1"]
    assert "answer for turn 0" in conversation["evidence"][1]["input"]
    store.close()


def test_multiturn_repeats_keep_separate_histories(monkeypatch, records):
    task = build_task(monkeypatch, records, "mt_bench")
    task.set_config("repeats", 2)
    task.build_all_requests(
        rank=0,
        world_size=1,
        apply_chat_template=True,
        chat_template=json.dumps,
        tokenizer_name="fixture",
    )
    requests_seen = []

    def dispatch(requests):
        for request in requests["generate_until"]:
            requests_seen.append(request)
            request.resps = [
                f"answer-{request.idx}-{i}" for i in range(request.repeats)
            ]

    task.run_generation_rounds(dispatch)
    assert [request.idx for request in requests_seen] == [0, 2, 3]
    assert (
        "answer-0-0" in requests_seen[1].args[0]
        and "answer-0-1" not in requests_seen[1].args[0]
    )
    assert (
        "answer-0-1" in requests_seen[2].args[0]
        and "answer-0-0" not in requests_seen[2].args[0]
    )


def test_scicode_previous_code_is_in_later_grader(monkeypatch, records):
    task = build_task(monkeypatch, records, "scicode")
    programs = []
    monkeypatch.setattr(
        task, "_python", lambda code, tests: programs.append(tests) or {"passed": True}
    )
    response = json.dumps(
        {
            "answers": [
                "```python\ndef add(a,b): return a+b\n```",
                "```python\ndef twice(a): return add(a,a)\n```",
            ],
            "evidence": [{}, {}],
        }
    )
    score = task.score_conversation(records, response)
    assert score["main_problem_pass_rate"] == 1
    assert "def add" in programs[1] and "def twice" in programs[1]
    assert "process_hdf5_to_tuple('1.2', 1)" in programs[1]


def test_hub_loader_accepts_native_dataset(monkeypatch, records):
    from modelscope.msdatasets import MsDataset

    ds = datasets.Dataset.from_list([records])
    monkeypatch.setattr(MsDataset, "load", lambda *args, **kwargs: ds)
    result = data.load_dataset(
        "evalscope/humanevalplus",
        "default",
        "humaneval_plus",
        {"hub": "modelscope"},
        {},
    )
    assert result["test"][0]["test"] == records["test"]


def test_missing_unsafe_optin_never_generates(monkeypatch, records):
    task = build_task(monkeypatch, records, "humaneval")
    model = ScriptedLM([])
    with pytest.raises(ValueError, match="unsafe"):
        evaluate(model, {"tasks": {task.config.task: task}}, bootstrap_iters=0)
    assert model.calls == []


def test_native_rwkv_path_does_not_wrap_pass_k(monkeypatch, records):
    from lm_eval.evaluator import simple_evaluate

    task = build_task(monkeypatch, records, "humaneval", pass_k=[1])
    monkeypatch.setattr(task, "validate_runtime", lambda: None)
    monkeypatch.setattr(
        task, "score_one", lambda doc, answer: {"passed": answer == "yes"}
    )
    model = ScriptedLM(["yes"])
    model.SUPPORTS_AVG_AT_K = True
    model.EVALUATION_PROTOCOL_VERSION = "test"
    result = simple_evaluate(
        model=model,
        tasks=[task],
        bootstrap_iters=0,
        confirm_run_unsafe_code=True,
        log_samples=True,
    )
    assert result["results"][task.config.task]["pass@1,all"] == 1
    assert task.config.repeats == 1
    assert task._filters[0].filters == []


@pytest.mark.parametrize(
    "language,wire", [("ts", "typescript"), ("r", "R"), ("d_ut", "D_ut")]
)
def test_sandbox_language_wire_names(monkeypatch, language, wire):
    response = Mock()
    response.json.return_value = {
        "status": "Success",
        "run_result": {"status": "Finished", "return_code": 0},
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(runtime.requests, "post", post)
    assert runtime.multiple_judge(
        "control", language, {"sandbox_url": "http://localhost:8123"}
    )["passed"]
    assert post.call_args.kwargs["json"]["language"] == wire
    assert post.call_args.kwargs["json"]["run_timeout"] == 30


@pytest.mark.parametrize("status", ["SandboxError", "InternalError", "Finished"])
def test_sandbox_missing_execution_verdict_is_infrastructure(monkeypatch, status):
    response = Mock()
    response.json.return_value = {
        "status": "Failed",
        "run_result": {"status": status, "return_code": None},
    }
    monkeypatch.setattr(runtime.requests, "post", Mock(return_value=response))
    with pytest.raises(runtime.JudgeError):
        runtime.multiple_judge("code", "cpp", {"sandbox_url": "http://localhost:8123"})


@pytest.mark.parametrize(
    "outcome", ["pass", "fail", "patch_rejected", "test_timeout", "infrastructure"]
)
def test_swe_official_verdicts_and_incomplete_runs(monkeypatch, tmp_path, outcome):
    modules = {}
    for name in (
        "docker",
        "swebench",
        "swebench.harness",
        "swebench.harness.constants",
        "swebench.harness.run_evaluation",
        "swebench.harness.test_spec",
        "swebench.harness.test_spec.test_spec",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    constants = modules["swebench.harness.constants"]
    constants.RUN_EVALUATION_LOG_DIR = tmp_path
    constants.LOG_INSTANCE = "run_instance.log"
    constants.LOG_TEST_OUTPUT = "test_output.txt"
    constants.APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
    client = Mock()
    modules["docker"].from_env = Mock(return_value=client)
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    make_spec = Mock(return_value="spec")
    modules["swebench.harness.test_spec.test_spec"].make_test_spec = make_spec
    calls = []

    def run_instance(spec, pred, **kwargs):
        assert spec == "spec"
        assert kwargs["timeout"] == 1800
        assert not kwargs["rm_image"] and not kwargs["force_rebuild"]
        calls.append(kwargs["run_id"])
        log_dir = tmp_path / kwargs["run_id"] / "lm-eval" / pred["instance_id"]
        log_dir.mkdir(parents=True)
        if outcome == "patch_rejected":
            (log_dir / constants.LOG_INSTANCE).write_text(constants.APPLY_PATCH_FAIL)
        elif outcome == "test_timeout":
            (log_dir / constants.LOG_TEST_OUTPUT).write_text(
                "tests started\n\nTimeout error: 1800 seconds exceeded."
            )
        return {"completed": outcome in {"pass", "fail"}, "resolved": outcome == "pass"}

    modules["swebench.harness.run_evaluation"].run_instance = run_instance
    doc = {"instance_id": "repo__issue-1", "test_patch": "original tests"}
    if outcome == "infrastructure":
        with pytest.raises(runtime.JudgeError, match="could not complete"):
            runtime.swe_judge(doc, "patch", {})
    else:
        result = runtime.swe_judge(doc, "patch", {})
        assert result["passed"] is (outcome == "pass")
        if outcome in {"patch_rejected", "test_timeout"}:
            assert result["status"] == outcome
        runtime.swe_judge(doc, "different patch", {})
        runtime.swe_judge({**doc, "test_patch": "different tests"}, "patch", {})
        assert len(set(calls)) == 3
    make_spec.assert_called_with(
        doc
        if outcome == "infrastructure"
        else {**doc, "test_patch": "different tests"},
        namespace="swebench",
    )


@pytest.mark.parametrize("pass_k", [[], [0], [True], [2]])
def test_invalid_sampling_contract_rejected_before_grading(
    monkeypatch, records, pass_k
):
    task = build_task(monkeypatch, records, "humaneval", pass_k=pass_k)
    with pytest.raises(ValueError, match="pass_k"):
        task.validate_runtime()
