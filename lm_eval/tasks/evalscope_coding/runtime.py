"""Execution backends for native Coding tasks (never imports EvalScope)."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import requests

from lm_eval.rwkv_coding import DockerJudge, JudgeError
from lm_eval.utils import handle_non_serializable


JUDGE_FORMAT = (
    "\n\nReply with a single JSON object and no other text, containing these required keys "
    "(additional keys are allowed):\n"
    '- "explanation": a string\n- "score": an integer >= 1 and <= 10\n'
    "Use valid JSON escaping for backslashes, double quotes, and newlines inside strings."
)

IMAGES = {
    "humaneval": "python:3.11-slim",
    "humaneval_plus": "lm-eval-coding-numpy:1",
    "mbpp": "python:3.11-slim",
    "mbpp_plus": "lm-eval-coding-numpy:1",
    "bigcodebench": "bigcodebench/bigcodebench-evaluate:latest",
    "bigcodebench_hard": "bigcodebench/bigcodebench-evaluate:latest",
    "live_code_bench": "lm-eval-coding-numpy:1",
    "scicode": "lm-eval-scicode:1",
    "ds1000": "lm-eval-ds1000:1",
}
TIMEOUTS = {
    "humaneval": 4,
    "humaneval_plus": 300,
    "mbpp": 20,
    "mbpp_plus": 60,
    "bigcodebench": 240,
    "bigcodebench_hard": 240,
    "live_code_bench": 6,
    "multiple_humaneval": 30,
    "multiple_mbpp": 30,
    "scicode": 300,
    "ds1000": 120,
}


def python_judge(
    benchmark: str, metadata: dict, timeout: float | None = None
) -> DockerJudge:
    """Reuse the existing restricted Docker judge and its infrastructure error contract."""
    return DockerJudge(
        SimpleNamespace(
            docker_image=metadata.get("docker_image", IMAGES[benchmark]),
            memory_mb=metadata.get("memory_mb", 4096),
            test_timeout=timeout or TIMEOUTS[benchmark],
        )
    )


def multiple_judge(code: str, language: str, metadata: dict) -> dict:
    """Use Sandbox Fusion's multilingual compiler/runner with the upstream 30s run limit."""
    url = metadata.get("sandbox_url") or os.environ.get("LMEVAL_CODING_SANDBOX_URL")
    if not url:
        raise JudgeError(
            "MultiPL-E requires metadata.sandbox_url or LMEVAL_CODING_SANDBOX_URL"
        )
    # EvalScope executor keys differ from Sandbox Fusion's wire enum.
    language = {"ts": "typescript", "r": "R", "d_ut": "D_ut"}.get(language, language)
    try:
        response = requests.post(
            url.rstrip("/") + "/run_code",
            json={"code": code, "language": language, "run_timeout": 30},
            timeout=(10, metadata.get("sandbox_wait_timeout", 3600)),
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        raise JudgeError("Sandbox Fusion transport failed; no model verdict") from error
    if result.get("status") not in {"Success", "Failed"}:
        raise JudgeError(
            f"Sandbox Fusion returned no execution verdict: {result.get('status')!r}"
        )
    run = result.get("run_result")
    compile_result = result.get("compile_result")
    if compile_result and compile_result.get("status") == "TimeLimitExceeded":
        raise JudgeError("Compiler timed out; no completed grading result")
    if compile_result and compile_result.get("status") != "Finished":
        raise JudgeError("Compiler returned no completed grading result")
    if run and run.get("status") not in {"Finished", "TimeLimitExceeded"}:
        raise JudgeError("Sandbox runner returned no completed grading result")
    if run and run.get("status") == "Finished" and run.get("return_code") is None:
        raise JudgeError("Sandbox runner returned no process exit code")
    if run is None and not (
        compile_result and compile_result.get("return_code") not in (0, None)
    ):
        raise JudgeError("Sandbox Fusion did not execute or reject the program")
    return {
        "passed": result["status"] == "Success"
        and run is not None
        and run.get("status") == "Finished"
        and run.get("return_code") == 0,
        "execution_result": result,
    }


def swe_judge(doc: dict, patch: str, metadata: dict) -> dict:
    """Run the separately packaged, official SWE-bench 4.1.0 Docker harness."""
    import docker
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        LOG_INSTANCE,
        LOG_TEST_OUTPUT,
        RUN_EVALUATION_LOG_DIR,
    )
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    if not patch.strip():
        return {"passed": False, "status": "empty_patch"}
    # Include both tests and prediction: official run_instance caches by run ID.
    identity = hashlib.sha256(
        json.dumps(
            [doc, patch], sort_keys=True, default=handle_non_serializable
        ).encode()
    ).hexdigest()
    run_id = "lm-eval-coding-" + identity
    spec = make_test_spec(doc, namespace=metadata.get("swe_namespace", "swebench"))
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / "lm-eval" / doc["instance_id"]
    with closing(docker.from_env()) as client:
        result = run_instance(
            spec,
            {
                "instance_id": doc["instance_id"],
                "model_patch": patch,
                "model_name_or_path": "lm-eval",
            },
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id=run_id,
            timeout=1800,
        )
    if not result or not result.get("completed"):
        # The official harness also reports completed=False for model failures.
        # Its own log markers distinguish those from image/daemon/test setup errors.
        log_path = log_dir / LOG_INSTANCE
        log = log_path.read_text() if log_path.is_file() else ""
        test_path = log_dir / LOG_TEST_OUTPUT
        test_log = test_path.read_text() if test_path.is_file() else ""
        status = (
            "patch_rejected"
            if APPLY_PATCH_FAIL in log
            else "test_timeout"
            if test_log.endswith("\n\nTimeout error: 1800 seconds exceeded.")
            else None
        )
        if status:
            return {"passed": False, "status": status, "run_id": run_id}
        raise JudgeError(
            f"SWE-bench could not complete {doc['instance_id']}; inspect {log_dir}"
        )
    return {
        "passed": bool(result["resolved"]),
        "execution_result": result,
        "run_id": run_id,
    }


def lcb_test_program(code: str, sample: str, timeout: int = 6) -> str:
    """Run the attributed LiveCodeBench grader inside Docker, including its per-test alarms."""
    source = (Path(__file__).parent / "vendor" / "lcb_testing.py").read_text()
    return source + (
        f"\n_result = run_test({{'input_output': {sample!r}}}, test={code!r}, timeout={timeout})\n"
        "assert _result is not None and len(_result[0]) > 0\n"
        "assert all(bool(x > 0) for x in _result[0]), repr(_result)\n"
    )


def mtbench_judge(
    turns: list[str], answers: list[str], references: list[str], metadata: dict
) -> dict:
    """Preserve the upstream reference-guided two-turn JSON rating protocol."""
    from openai import OpenAI

    from .vendor.mtbench import first_turn_prompt, second_turn_prompt

    settings = metadata.get("judge", {})
    client = OpenAI(
        api_key=os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY")),
        base_url=settings.get("base_url"),
        timeout=settings.get("timeout", 120),
        max_retries=0,
    )
    prompts = [
        first_turn_prompt(turns[0], answers[0], references[0]),
        second_turn_prompt(turns, answers, references, True),
    ]
    values, evidence = [], []
    try:
        for prompt, system in prompts:
            response = client.chat.completions.create(
                model=settings.get("model", "gpt-4"),
                temperature=0,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt + JUDGE_FORMAT},
                ],
            )
            raw = response.choices[0].message.content
            verdict = json.loads(raw)
            value = verdict["score"]
            if type(value) is not int or not 1 <= value <= 10:
                raise ValueError("MT-Bench judge score must be an integer from 1 to 10")
            values.append(value)
            evidence.append(
                {"messages": [system, prompt], "response": raw, "model": response.model}
            )
    except Exception as error:
        # A missing judge rating is never a zero or a fabricated complete score.
        raise JudgeError("MT-Bench judge did not return two valid ratings") from error
    finally:
        client.close()
    return {
        "judge_score": sum(values) / 2,
        "first_turn_judge_score": values[0],
        "second_turn_judge_score": values[1],
        "judge_evidence": evidence,
    }
