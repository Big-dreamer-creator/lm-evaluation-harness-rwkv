#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(
    os.environ.get("RWKV_BACKEND_ROOT", "/home/creator/code/vllm-rwkv")
)
PYTHON = ROOT / ".venv/bin/python"
MODEL = "rwkv7-g1i-1.5b-20260805-ctx16384"
BASE_URL = "http://127.0.0.1:8000"
CONFIG = ROOT / (
    "configs/eval/"
    "rwkv7_g1i_1_5b_20260805_ctx16384_"
    "moral_haerae_jsonschema_gsm8k_platinum_aexams.toml"
)
RESULT_ROOT = ROOT / "results/formal-rwkv-five-benchmarks-20260821"
CACHE_ROOT = Path("/home/creator/.cache/lm-eval-rwkv/five-benchmarks-20260821")
LOG_ROOT = CACHE_ROOT / "logs"
REQUEST_ROOT = CACHE_ROOT / "requests"
TMP_ROOT = CACHE_ROOT / "tmp"
WEIGHTS = Path("/mnt/e/code/Weights") / MODEL
VLLM = BACKEND_ROOT / ".venv/bin/vllm"
BENCHMARKS = ("moral_stories", "haerae", "jsonschema_bench", "gsm8k_platinum", "aexams")
SELECTORS = {
    "moral_stories": "rwkv7_g1i_1_5b_20260805_ctx16384_moral_stories",
    "haerae": "rwkv7_g1i_1_5b_20260805_ctx16384_haerae",
    "jsonschema_bench": "rwkv7_g1i_1_5b_20260805_ctx16384_jsonschema_bench",
    "gsm8k_platinum": "rwkv7_g1i_1_5b_20260805_ctx16384_gsm8k_platinum",
    "aexams": "rwkv7_g1i_1_5b_20260805_ctx16384_aexams",
}
INCLUDE_DIRS = {
    "moral_stories": "moral_stories",
    "haerae": "haerae",
    "jsonschema_bench": "jsonschema_bench",
    "gsm8k_platinum": "gsm8k_platinum",
    "aexams": "aexams",
}
DATASET_REVISIONS = {
    "moral_stories": "b830cf56eb00bc4edd1860dd544a192216eb3587",
    "haerae": "b480e81024913f27783d2b05d2f0b10089db19ad",
    "jsonschema_bench": "5bd0f4640badc6f3f02df796421d21cb0ca0b141",
    "gsm8k_platinum": "e762492455a1cf7967de89f05b6bef72fc713b66",
    "aexams": "bc7a29346dbcaa16a8cd883b1f3e681ab2b7ff2a",
}
GEN_KWARGS = {
    "jsonschema_bench": ("max_gen_toks=2048", "do_sample=true"),
    "gsm8k_platinum": ("max_gen_toks=512", "do_sample=true"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=900)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def result_dir(benchmark: str) -> Path:
    return RESULT_ROOT / MODEL / benchmark


def log_path(benchmark: str) -> Path:
    return LOG_ROOT / benchmark / "run.log"


def result_exists(benchmark: str) -> bool:
    return any(result_dir(benchmark).glob("**/results_*.json"))


def write_milestone(name: str, **fields: Any) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULT_ROOT / "milestones.json"
    current: dict[str, Any] = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    current[name] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def evaluation_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for path in (CACHE_ROOT, LOG_ROOT, REQUEST_ROOT, TMP_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "HF_HOME": "/home/creator/.cache/huggingface",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "LM_HARNESS_CACHE_PATH": str(REQUEST_ROOT),
            "TMPDIR": str(TMP_ROOT),
            "TMP": str(TMP_ROOT),
            "TEMP": str(TMP_ROOT),
        }
    )
    environment.setdefault("HTTP_PROXY", "http://172.17.32.1:7897")
    environment.setdefault("HTTPS_PROXY", "http://172.17.32.1:7897")
    local_hosts = {"127.0.0.1", "localhost"}
    local_hosts.update(
        host.strip() for host in environment.get("NO_PROXY", "").split(",") if host.strip()
    )
    environment["NO_PROXY"] = ",".join(sorted(local_hosts))
    environment["no_proxy"] = environment["NO_PROXY"]
    source_path = str(ROOT)
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_path, environment["PYTHONPATH"]]
        if environment.get("PYTHONPATH")
        else [source_path]
    )
    return environment


def server_environment() -> dict[str, str]:
    environment = evaluation_environment()
    environment.update(
        {
            "CUDA_HOME": "/usr/local/cuda-13.0",
            "RWKV_MAX_NUM_SEQS": "24",
            "RWKV_GPU_MEMORY_UTILIZATION": "0.85",
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_RWKV7_WKV_MODE": "fp32io16",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_RAPID_SAMPLER": "1",
        }
    )
    environment["PATH"] = os.pathsep.join(
        [str(BACKEND_ROOT / ".venv/bin"), environment.get("PATH", "")]
    )
    return environment


def server_command() -> list[str]:
    return [
        str(VLLM),
        "serve",
        str(WEIGHTS),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        MODEL,
        "--chat-template",
        str(WEIGHTS / "chat_template.jinja"),
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "24",
        "--max-num-batched-tokens",
        "16384",
        "--gpu-memory-utilization",
        "0.85",
        "--enable-tokenizer-info-endpoint",
    ]


def request_json(path: str) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        if response.status != 200:
            raise OSError(f"HTTP {response.status} from {path}")
        return json.loads(response.read())
    finally:
        connection.close()


def running_model() -> str | None:
    try:
        models = request_json("/v1/models").get("data", [])
    except (OSError, json.JSONDecodeError):
        return None
    if len(models) != 1:
        raise RuntimeError(f"Expected one served model, got {models}")
    return models[0].get("id")


def wait_for_server(process: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vllm-rwkv exited with status {process.returncode}")
        if running_model() == MODEL:
            return
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for {MODEL}")


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def evaluation_command(benchmark: str, *, force: bool = False) -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "lm_eval",
        "run",
        "--config",
        str(CONFIG),
        "--tasks",
        SELECTORS[benchmark],
        "--output_path",
        str(result_dir(benchmark)),
        "--use_cache",
        str(REQUEST_ROOT / benchmark),
        "--cache_requests",
        "refresh" if force else "true",
        "--metadata",
        json.dumps(
            {
                "benchmark_name": benchmark,
                "model_name": MODEL,
                "dataset_revision": DATASET_REVISIONS[benchmark],
                "campaign_name": "formal-rwkv-five-benchmarks-20260821",
                "cot_mode": "fake_think",
                "prompt_template": "assistant",
                "wkv_mode": "fp32io16",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "--verbosity",
        "INFO",
        "--include_path",
        os.pathsep.join(
            [
                str(ROOT / "lm_eval/tasks" / INCLUDE_DIRS[benchmark]),
                str(ROOT / "lm_eval/tasks/rwkv7_g1i_1_5b_20260805_ctx16384"),
            ]
        ),
    ]
    generation_kwargs = GEN_KWARGS.get(benchmark, ())
    if generation_kwargs:
        command.extend(["--gen_kwargs", *generation_kwargs])
    return command


def validate() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not VLLM.is_file():
        raise FileNotFoundError(VLLM)
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    template = WEIGHTS / "chat_template.jinja"
    if not template.is_file():
        raise FileNotFoundError(template)
    model_path = WEIGHTS.parent / f"{MODEL}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    check = subprocess.run(
        [str(PYTHON), "-c", "import math_verify, jsonschema"],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        raise RuntimeError("The project .venv lacks math_verify and jsonschema")


def write_manifest() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_name": "formal-rwkv-five-benchmarks-20260821",
        "model_name": MODEL,
        "harness_commit": git_commit(ROOT),
        "vllm_rwkv_commit": git_commit(BACKEND_ROOT),
        "config_path": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "chat_template_sha256": sha256(WEIGHTS / "chat_template.jinja"),
        "dataset_revisions": DATASET_REVISIONS,
        "prompt_template": "assistant",
        "cot_mode": "fake_think",
        "decoding": {"temperature": 1.0, "top_p": 0.28, "top_k": 32},
        "wkv_mode": "fp32io16",
        "num_concurrent": 25,
        "rwkv_max_num_seqs": 24,
        "rwkv_gpu_memory_utilization": 0.85,
        "benchmarks": BENCHMARKS,
    }
    (RESULT_ROOT / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_benchmark(benchmark: str, force: bool) -> None:
    if result_exists(benchmark) and not force:
        write_milestone(f"{benchmark}_complete", skipped=True, result_dir=str(result_dir(benchmark)))
        return
    result_dir(benchmark).mkdir(parents=True, exist_ok=True)
    path = log_path(benchmark)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = evaluation_command(benchmark, force=force)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=Path("/home/creator"),
            env=evaluation_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"{benchmark} failed with status {completed.returncode}")
    if not result_exists(benchmark):
        raise RuntimeError(f"{benchmark} produced no results JSON")
    write_milestone(
        f"{benchmark}_complete",
        result_dir=str(result_dir(benchmark)),
        log_path=str(path),
    )


def run(args: argparse.Namespace) -> int:
    benchmarks = list(dict.fromkeys(args.benchmark or BENCHMARKS))
    validate()
    write_manifest()
    if args.dry_run:
        print(json.dumps({benchmark: evaluation_command(benchmark) for benchmark in benchmarks}, indent=2))
        return 0
    current = running_model()
    if current is not None:
        raise RuntimeError(f"Port 8000 already serves {current}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    server_log = LOG_ROOT / "server.log"
    handle = server_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        server_command(),
        cwd=BACKEND_ROOT,
        env=server_environment(),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_milestone("server_started", pid=process.pid, model_name=MODEL)
    try:
        wait_for_server(process, args.server_timeout)
        write_milestone("server_ready", model_name=MODEL)
        for benchmark in benchmarks:
            run_benchmark(benchmark, args.force)
        write_milestone("evaluation_complete", benchmarks=benchmarks)
    except Exception as exc:
        write_milestone("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_process(process)
        handle.close()
        write_milestone("server_stopped")
    write_milestone("supervisor_exit_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
