#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib


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
WKV_MODES = ("fp16", "fp32io16")
DEFAULT_WKV_MODE = "fp32io16"
CAMPAIGN_NAME = "formal-rwkv-five-benchmarks-20260821"
# The producer is intentionally local-only until scoreboard-rwkv's lm-eval
# publication mapping is finalized.  Do not label these payloads as LightEval
# DTOs: the artifacts retain richer lm-eval/RWKV evidence and are converted by
# the future uploader at the scoreboard boundary.
CAMPAIGN_SCHEMA = "rwkv-producer-campaign-v1"
TASK_SCHEMA = "rwkv-producer-task-v1"
LIGHTEVAL_VERSION = "0.13.0"
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
SCOREBOARD_MAX_GEN_TOKS = 8192
SCOREBOARD_SAMPLING = {
    "temperature": 0.96,
    "top_p": 0.76,
    "top_k": 32,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.1,
    "repetition_penalty": 1.0,
    "penalty_decay": 0.988,
    "max_new_tokens": SCOREBOARD_MAX_GEN_TOKS,
    "ignore_eos": False,
    "stop": ["\nUser:"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument(
        "--wkv-mode",
        choices=("both", *WKV_MODES),
        default="both",
        help="Run one WKV mode or the complete fp16 + fp32io16 pair.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-mode task results (the default behavior).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify local publication artifacts after the run (no network).",
    )
    parser.add_argument(
        "--digest",
        action="store_true",
        help="Print local publication digests after the run (no network).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing local publication artifacts without inference.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check lockfile, backend/GPU compatibility, and service state only.",
    )
    parser.add_argument(
        "--scoreboard-compatible",
        action="store_true",
        help=(
            "Use scoreboard-rwkv's current LightEval publication contract: "
            "open_think sampling and an 8192-token generation budget."
        ),
    )
    parser.add_argument("--server-timeout", type=int, default=900)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def hash_path(path: Path) -> str | None:
    """Hash a checkpoint file, or a directory deterministically if needed."""

    try:
        if path.is_file():
            return sha256(path)
        if not path.is_dir():
            return None
        digest = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with child.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def weight_path() -> Path | None:
    candidates = [
        WEIGHTS.parent / f"{MODEL}.pth",
        WEIGHTS / f"{MODEL}.pth",
        WEIGHTS / "model.safetensors",
        WEIGHTS / "model.bin",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def optional_git_commit(path: Path) -> str | None:
    try:
        return git_commit(path)
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_snapshot() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "/usr/lib/wsl/lib/nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []
    result = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            result.append(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": fields[3],
                }
            )
    return result


def dependency_snapshot() -> dict[str, Any]:
    packages = {}
    for name in (
        "lm_eval",
        "datasets",
        "requests",
        "aiohttp",
        "tenacity",
        "jsonschema",
        "math-verify",
        "lighteval",
        "transformers",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock_path = ROOT / "uv.lock"
    return {
        "python": subprocess.run(
            [str(PYTHON), "--version"],
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()
        if PYTHON.is_file()
        else None,
        "packages": packages,
        "uv_lock_sha256": hash_path(lock_path),
    }


def backend_gpu_compatibility() -> dict[str, Any]:
    """Check the installed FlashRWKV2 binary against the local GPU.

    The vllm-rwkv model has no CPU/product fallback.  A prebuilt extension for
    another SM is guaranteed to fail only after the engine allocates memory, so
    detect that mismatch before starting the HTTP server.
    """

    probe = [
        str(BACKEND_ROOT / ".venv/bin/python"),
        "-c",
        (
            "import importlib.metadata, pathlib, torch, flashrwkv2; "
            "p=next(pathlib.Path(flashrwkv2.__file__).parent.glob('_C*.so')); "
            "print(torch.cuda.get_device_capability()[0], torch.cuda.get_device_capability()[1]); "
            "print(p); "
            "print(importlib.metadata.version('FlashRWKV2'))"
        ),
    ]
    try:
        completed = subprocess.run(
            probe, capture_output=True, check=False, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unknown", "error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "status": "unknown",
            "error": completed.stderr.strip() or "backend GPU probe failed",
        }
    lines = completed.stdout.splitlines()
    if len(lines) < 3:
        return {"status": "unknown", "error": "malformed backend GPU probe"}
    try:
        capability = [int(value) for value in lines[0].split()]
    except ValueError:
        return {"status": "unknown", "error": "malformed CUDA capability"}
    extension = Path(lines[1])
    version = lines[2].strip()
    try:
        strings = subprocess.run(
            ["strings", str(extension)],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        strings = ""
    compiled_arches = sorted(
        {
            int(value)
            for value in re.findall(r"csrc/sm(\d+)", strings)
            if value.isdigit()
        }
    )
    requested = capability[0] * 10 + capability[1]
    result: dict[str, Any] = {
        "status": "compatible" if not compiled_arches or requested in compiled_arches else "incompatible",
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "gpu_sm": requested,
        "flashrwkv2_version": version,
        "flashrwkv2_extension": str(extension),
        "compiled_arches": compiled_arches,
    }
    if result["status"] == "incompatible":
        result["error"] = (
            f"FlashRWKV2 extension has kernels for SM {compiled_arches}, "
            f"but the GPU is SM {requested}"
        )
    return result


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def result_dir(benchmark: str, wkv_mode: str = DEFAULT_WKV_MODE) -> Path:
    return RESULT_ROOT / wkv_mode / MODEL / benchmark


def log_path(benchmark: str, wkv_mode: str = DEFAULT_WKV_MODE) -> Path:
    return LOG_ROOT / wkv_mode / benchmark / "run.log"


def result_exists(benchmark: str, wkv_mode: str = DEFAULT_WKV_MODE) -> bool:
    return any(result_dir(benchmark, wkv_mode).glob("**/results_*.json"))


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


def write_preflight(
    wkv_modes: tuple[str, ...], *, status: str, error: str | None = None
) -> dict[str, Any]:
    """Persist service/backend checks even when startup fails before a campaign."""

    preflight_path = RESULT_ROOT / "preflight.json"
    previous: dict[str, Any] | None = None
    if preflight_path.is_file():
        try:
            loaded = json.loads(preflight_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("status") == "failed":
                previous = loaded
        except (OSError, ValueError):
            previous = None
    record: dict[str, Any] = {
        "schema_version": "rwkv-producer-preflight-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL,
        "wkv_modes": list(wkv_modes),
        "status": status,
        "service": service_probe(),
        "backend_gpu_compatibility": backend_gpu_compatibility(),
        "backend_root": str(BACKEND_ROOT),
        "backend_commit": optional_git_commit(BACKEND_ROOT),
    }
    if error is not None:
        record["error"] = error
    if previous is not None and status != "failed":
        record["previous_failure"] = previous
    write_atomic(
        preflight_path,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return record


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


def server_environment(wkv_mode: str = DEFAULT_WKV_MODE) -> dict[str, str]:
    environment = evaluation_environment()
    environment.update(
        {
            "CUDA_HOME": "/usr/local/cuda-13.0",
            "RWKV_MAX_NUM_SEQS": "24",
            "RWKV_GPU_MEMORY_UTILIZATION": "0.85",
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_RWKV7_WKV_MODE": wkv_mode,
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_RAPID_SAMPLER": "1",
        }
    )
    environment["PATH"] = os.pathsep.join(
        [str(BACKEND_ROOT / ".venv/bin"), environment.get("PATH", "")]
    )
    return environment


def server_command(wkv_mode: str = DEFAULT_WKV_MODE) -> list[str]:
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


def service_probe() -> dict[str, Any]:
    """Inspect the independent HTTP service without starting or mutating it."""

    result: dict[str, Any] = {"reachable": False, "model": None}
    try:
        models = request_json("/v1/models")
        result["reachable"] = True
        result["models"] = models.get("data", [])
        result["model"] = running_model()
    except (OSError, ValueError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    try:
        tokenizer_info = request_json("/tokenizer_info")
        result["tokenizer_info"] = {
            "has_chat_template": isinstance(
                tokenizer_info.get("chat_template"), str
            ),
            "chat_template_sha256": (
                hashlib.sha256(
                    tokenizer_info["chat_template"].encode("utf-8")
                ).hexdigest()
                if isinstance(tokenizer_info.get("chat_template"), str)
                else None
            ),
        }
    except (OSError, ValueError):
        result["tokenizer_info"] = {"reachable": False}
    return result


def service_wkv_mode() -> str | None:
    """Read the mode of a local listener when it is inspectable.

    vLLM's public model metadata does not currently expose RWKV's WKV mode. A
    reused service is therefore accepted only when its process environment can
    be inspected; an unknown mode must not be mistaken for fp16/fp32io16.
    """

    try:
        completed = subprocess.run(
            ["ss", "-ltnp"], capture_output=True, check=False, text=True
        )
    except OSError:
        return None
    pid = None
    for line in completed.stdout.splitlines():
        if ":8000 " not in line and ":8000," not in line:
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            pid = match.group(1)
            break
    if pid is None:
        return None
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = b"VLLM_RWKV7_WKV_MODE="
    for item in environment:
        if item.startswith(prefix):
            return item[len(prefix) :].decode("utf-8", errors="replace")
    return None


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


def evaluation_command(
    benchmark: str,
    *,
    force: bool = False,
    wkv_mode: str = DEFAULT_WKV_MODE,
    scoreboard_compatible: bool = False,
) -> list[str]:
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
        str(result_dir(benchmark, wkv_mode)),
        "--use_cache",
        str(REQUEST_ROOT / wkv_mode / benchmark),
        "--cache_requests",
        "refresh" if force else "true",
        "--metadata",
        json.dumps(
            {
                "benchmark_name": benchmark,
                "model_name": MODEL,
                "dataset_revision": DATASET_REVISIONS[benchmark],
                "campaign_name": "formal-rwkv-five-benchmarks-20260821",
                "cot_mode": "open_think" if scoreboard_compatible else "fake_think",
                "prompt_template": "assistant",
                "wkv_mode": wkv_mode,
                "scoreboard_compatible": scoreboard_compatible,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "--verbosity",
        "INFO",
        "--model_args",
        "model=" + MODEL,
        "base_url=http://127.0.0.1:8000/v1/completions",
        "rwkv_prompt_template=assistant",
        "rwkv_generation_prompt="
        + ("open_think" if scoreboard_compatible else "fake_think"),
        "rwkv_sampling_mode=profile",
        "num_concurrent=25",
        "max_length=16384",
        "record_evidence=true",
        "--include_path",
        os.pathsep.join(
            [
                str(ROOT / "lm_eval/tasks" / INCLUDE_DIRS[benchmark]),
                str(ROOT / "lm_eval/tasks/rwkv7_g1i_1_5b_20260805_ctx16384"),
            ]
        ),
    ]
    generation_kwargs = (
        ("max_gen_toks=8192", "do_sample=true")
        if scoreboard_compatible
        else GEN_KWARGS.get(benchmark, ())
    )
    if generation_kwargs:
        command.extend(["--gen_kwargs", *generation_kwargs])
    return command


def validate(
    *, require_backend: bool = True, scoreboard_compatible: bool = False
) -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    lock_path = ROOT / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"{lock_path} is required; run uv lock and commit the lockfile"
        )
    if require_backend and not VLLM.is_file():
        raise FileNotFoundError(VLLM)
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    template = WEIGHTS / "chat_template.jinja"
    model_path = weight_path()
    if require_backend and not template.is_file():
        raise FileNotFoundError(template)
    if require_backend and model_path is None:
        raise FileNotFoundError(WEIGHTS.parent / f"{MODEL}.pth")
    if require_backend:
        compatibility = backend_gpu_compatibility()
        if compatibility.get("status") == "incompatible":
            raise RuntimeError(
                "vllm-rwkv backend/GPU incompatibility: "
                + str(compatibility.get("error"))
            )
    if require_backend:
        check = subprocess.run(
            [str(PYTHON), "-c", "import jsonschema"],
            check=False,
            capture_output=True,
            text=True,
        )
        if check.returncode:
            raise RuntimeError("The project .venv lacks jsonschema")
    if scoreboard_compatible:
        try:
            lighteval_version = importlib.metadata.version("lighteval")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "--scoreboard-compatible requires the real LightEval 0.13.0 "
                "runtime because scoreboard-rwkv's DTO records LightEval provenance"
            ) from error
        if lighteval_version != "0.13.0":
            raise RuntimeError(
                "--scoreboard-compatible requires lighteval==0.13.0, "
                f"found {lighteval_version}"
            )


def task_identity(wkv_mode: str, benchmark: str) -> str:
    return f"{MODEL}:{wkv_mode}:{benchmark}"


def task_artifact_name(wkv_mode: str, benchmark: str) -> str:
    return f"{wkv_mode}__{benchmark}"


def write_manifest(
    wkv_modes: tuple[str, ...] = (DEFAULT_WKV_MODE,),
    scoreboard_compatible: bool = False,
) -> dict[str, Any]:
    """Write one campaign-level provenance document and return its contents."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULT_ROOT / "campaign_manifest.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, ValueError):
            previous = {}
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    profile = config.get("rwkv_profile", {})
    weight = weight_path()
    backend_commit = optional_git_commit(BACKEND_ROOT)
    if backend_commit is None:
        raise RuntimeError(
            f"cannot record vllm-rwkv backend commit from {BACKEND_ROOT}"
        )
    generation_prompt = "open_think" if scoreboard_compatible else profile.get(
        "generation_prompt", "fake_think"
    )
    generation_configs = {
        benchmark: {
            "max_gen_toks": (
                SCOREBOARD_MAX_GEN_TOKS
                if scoreboard_compatible
                else int(
                    next(
                        (
                            value.split("=", 1)[1]
                            for value in GEN_KWARGS.get(benchmark, ())
                            if value.startswith("max_gen_toks=")
                        ),
                        "256",
                    )
                )),
            "do_sample": (
                True if scoreboard_compatible else benchmark in GEN_KWARGS
            ),
        }
        for benchmark in BENCHMARKS
    }
    base = {
        "schema_version": "rwkv-producer-v1",
        "campaign_name": CAMPAIGN_NAME,
        "model_name": MODEL,
        "weight_path": str(weight) if weight else None,
        "weight_sha256": hash_path(weight) if weight else None,
        "backend": "vllm-rwkv",
        "backend_root": str(BACKEND_ROOT),
        "backend_commit": backend_commit,
        "vllm_rwkv_commit": backend_commit,
        "harness_commit": optional_git_commit(ROOT),
        "config_path": str(CONFIG),
        "config_sha256": hash_path(CONFIG),
        "preflight_sha256": hash_path(RESULT_ROOT / "preflight.json"),
        "dataset_revisions": DATASET_REVISIONS,
        "prompt": {
            "template": profile.get("prompt_template", "assistant"),
            "generation_prompt": generation_prompt,
            "chat_template_sha256": hash_path(WEIGHTS / "chat_template.jinja"),
        },
        "scoreboard_compatible": scoreboard_compatible,
        "generation_configs": generation_configs,
        "sampling_config": (
            dict(SCOREBOARD_SAMPLING)
            if scoreboard_compatible
            else {
                "temperature": 1.0 if generation_prompt == "fake_think" else 0.96,
                "top_p": 0.28 if generation_prompt == "fake_think" else 0.76,
                "top_k": 32,
                **(
                    {
                        "presence_penalty": 1.0,
                        "frequency_penalty": 0.1,
                        "penalty_decay": 0.988,
                    }
                    if generation_prompt == "open_think"
                    else {}
                ),
            }
        ),
        "sampling": {
            "mode": profile.get("sampling_mode", "profile"),
            "fake_think": {"temperature": 1.0, "top_p": 0.28, "top_k": 32},
            "open_think": {
                "temperature": 0.96,
                "top_p": 0.76,
                "top_k": 32,
                "presence_penalty": 1.0,
                "frequency_penalty": 0.1,
                "penalty_decay": 0.988,
            },
        },
        "runtime": {
            "max_length": 16384,
            "max_num_batched_tokens": 16384,
            "num_concurrent": 25,
            "rwkv_max_num_seqs": 24,
            "rwkv_gpu_memory_utilization": 0.85,
            "gpu": gpu_snapshot(),
            "backend_gpu_compatibility": backend_gpu_compatibility(),
        },
        "dependencies": dependency_snapshot(),
        "benchmarks": list(BENCHMARKS),
    }
    stable = {
        key: value
        for key, value in base.items()
        if key not in {"runtime", "dependencies", "preflight_sha256"}
    }
    manifest = {
        **base,
        "created_at": previous.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        ),
        "modes": list(dict.fromkeys((*previous.get("modes", []), *wkv_modes))),
        "run_key": value_digest(stable),
        "mode_runs": previous.get("mode_runs", {}),
    }
    for mode in wkv_modes:
        manifest["mode_runs"].setdefault(
            mode,
            {
                "wkv_mode": mode,
                "status": "pending",
                "task_identities": [task_identity(mode, benchmark) for benchmark in BENCHMARKS],
            },
        )
        mode_dir = RESULT_ROOT / mode / MODEL
        mode_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(
            mode_dir / "provenance.json",
            json.dumps(
                {**manifest, "wkv_mode": mode},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    write_atomic(
        path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def run_benchmark(
    benchmark: str,
    force: bool,
    wkv_mode: str,
    scoreboard_compatible: bool = False,
) -> None:
    if result_exists(benchmark, wkv_mode) and not force:
        write_milestone(
            f"{wkv_mode}_{benchmark}_complete",
            skipped=True,
            result_dir=str(result_dir(benchmark, wkv_mode)),
        )
        return
    result_dir(benchmark, wkv_mode).mkdir(parents=True, exist_ok=True)
    path = log_path(benchmark, wkv_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = evaluation_command(
        benchmark,
        force=force,
        wkv_mode=wkv_mode,
        scoreboard_compatible=scoreboard_compatible,
    )
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
        raise RuntimeError(f"{wkv_mode}/{benchmark} produced no results JSON")
    write_milestone(
        f"{wkv_mode}_{benchmark}_complete",
        result_dir=str(result_dir(benchmark, wkv_mode)),
        log_path=str(path),
    )


def _latest_file(root: Path, pattern: str) -> Path | None:
    files = [path for path in root.glob(f"**/{pattern}") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as error:
                raise RuntimeError(f"invalid sample JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"sample record is not an object at {path}:{line_number}")
            records.append(value)
    return records


def _normalize_evidence(
    evidence: Any,
    *,
    sample: dict[str, Any],
    benchmark: str,
    wkv_mode: str,
) -> list[dict[str, Any]]:
    values: list[Any] = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, list):
                values.extend(item)
            else:
                values.append(item)
    if not values:
        raise RuntimeError(
            f"{wkv_mode}/{benchmark} sample lacks model response evidence "
            f"(doc_id={sample.get('doc_id')!r})"
        )
    normalized = []
    for item in values:
        value = dict(item) if isinstance(item, dict) else {}
        value.setdefault("prompt", None)
        value.setdefault("input_token_ids", None)
        value.setdefault("output_token_ids", None)
        value.setdefault("raw_response", None)
        value.setdefault("reasoning", None)
        value.setdefault("post_processed_answer", None)
        value.setdefault("finish_reason", None)
        value.setdefault("truncation", False)
        value.setdefault("metrics", {})
        hashes = value.setdefault("hashes", {})
        if not isinstance(hashes, dict):
            hashes = {}
            value["hashes"] = hashes
        hashes.setdefault("prompt_sha256", value_digest(value["prompt"]))
        hashes.setdefault(
            "answer_sha256", value_digest(value["post_processed_answer"])
        )
        hashes.setdefault("raw_response_sha256", value_digest(value["raw_response"]))
        hashes["benchmark"] = benchmark
        hashes["wkv_mode"] = wkv_mode
        normalized.append(value)
    return normalized


def _task_config_from_results(
    raw_results: dict[str, Any],
    *,
    benchmark: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Extract document accounting and generation budget from lm-eval output."""

    n_samples = raw_results.get("n-samples")
    if not isinstance(n_samples, dict) or not n_samples:
        raise RuntimeError(
            f"{benchmark} results lack n-samples accounting required for publication"
        )
    original = 0
    effective = 0
    for value in n_samples.values():
        if not isinstance(value, dict):
            raise RuntimeError(f"{benchmark} n-samples entry is malformed")
        row_original = value.get("original")
        row_effective = value.get("effective")
        if (
            isinstance(row_original, bool)
            or not isinstance(row_original, int)
            or row_original < 0
            or isinstance(row_effective, bool)
            or not isinstance(row_effective, int)
            or row_effective < 0
        ):
            raise RuntimeError(f"{benchmark} n-samples accounting is invalid")
        original += row_original
        effective += row_effective
    if original <= 0 or effective <= 0 or effective > original:
        raise RuntimeError(f"{benchmark} n-samples accounting is inconsistent")
    generation_config = manifest.get("generation_configs", {}).get(benchmark)
    if not isinstance(generation_config, dict):
        raise RuntimeError(f"{benchmark} generation configuration was not recorded")
    return {
        "generation_size": generation_config.get("max_gen_toks"),
        "original_num_docs": original,
        "effective_num_docs": effective,
        "skipped_multiselect_docs": original - effective,
        "generation_config": generation_config,
    }


def materialize_task_artifacts(
    wkv_mode: str, benchmark: str, manifest: dict[str, Any]
) -> dict[str, Any] | None:
    """Convert lm-eval's timestamped output into stable producer artifacts."""

    source_root = result_dir(benchmark, wkv_mode)
    result_source = _latest_file(source_root, "results_*.json")
    if result_source is None:
        return None
    try:
        raw_results = json.loads(result_source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot load result JSON {result_source}") from error
    if not isinstance(raw_results, dict):
        raise RuntimeError(f"result JSON is not an object: {result_source}")

    sample_sources = sorted(source_root.glob("**/samples_*.jsonl"))
    samples: list[dict[str, Any]] = []
    for sample_source in sample_sources:
        samples.extend(_read_jsonl(sample_source))
    identity = task_identity(wkv_mode, benchmark)
    for sample in samples:
        evidence = _normalize_evidence(
            sample.get("response_evidence"),
            sample=sample,
            benchmark=benchmark,
            wkv_mode=wkv_mode,
        )
        sample["response_evidence"] = evidence
        sample["producer"] = {
            "schema_version": "rwkv-sample-v1",
            "identity": identity,
            "model_name": MODEL,
            "wkv_mode": wkv_mode,
            "dataset_revision": DATASET_REVISIONS[benchmark],
            "doc_id": sample.get("doc_id"),
        }
        sample_hash_input = dict(sample)
        sample_hash_input.pop("hashes", None)
        sample["hashes"] = {
            **(sample.get("hashes") if isinstance(sample.get("hashes"), dict) else {}),
            "sample_sha256": value_digest(sample_hash_input),
        }

    samples_text = "".join(
        json.dumps(sample, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        for sample in samples
    )
    artifact_dir = RESULT_ROOT / "tasks" / task_artifact_name(wkv_mode, benchmark)
    samples_path = artifact_dir / "samples.jsonl"
    results_path = artifact_dir / "results.json"
    write_atomic(samples_path, samples_text)
    metrics = raw_results.get("results", raw_results)
    truncation_count = sum(
        1
        for sample in samples
        for evidence in sample.get("response_evidence", [])
        if evidence.get("truncation")
    )
    evidence_count = sum(
        len(sample.get("response_evidence", [])) for sample in samples
    )
    result_artifact = {
        "schema_version": "rwkv-task-results-v1",
        "identity": identity,
        "task": benchmark,
        "model_name": MODEL,
        "wkv_mode": wkv_mode,
        "dataset_revision": DATASET_REVISIONS[benchmark],
        "sample_count": len(samples),
        "evidence_count": evidence_count,
        "truncated_samples": truncation_count,
        "truncation_rate": (
            truncation_count / evidence_count if evidence_count else None
        ),
        "metrics": metrics,
        "raw_results": raw_results,
        "source_results_sha256": hash_path(result_source),
        "samples_sha256": hash_path(samples_path),
        "provenance_digest": manifest["run_key"],
        "task_config": _task_config_from_results(
            raw_results, benchmark=benchmark, manifest=manifest
        ),
    }
    write_atomic(
        results_path,
        json.dumps(result_artifact, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )
    task_payload = {
        "schema_version": TASK_SCHEMA,
        "publication_contract": "rwkv-producer-v1",
        "scoreboard_upload_ready": False,
        "campaign_id": None,
        "task": {
            "identity": identity,
            "task_name": benchmark,
            "model_name": MODEL,
            "wkv_mode": wkv_mode,
            "dataset_revision": DATASET_REVISIONS[benchmark],
            "results": result_artifact,
            "samples": samples,
            "task_config": result_artifact["task_config"],
            "artifacts": {
                "results": str(results_path.relative_to(RESULT_ROOT)),
                "samples": str(samples_path.relative_to(RESULT_ROOT)),
                "results_sha256": hash_path(results_path),
                "samples_sha256": hash_path(samples_path),
            },
            "provenance": {
                "model_name": MODEL,
                "weight_sha256": manifest.get("weight_sha256"),
                "backend_commit": manifest.get("backend_commit"),
                "config_sha256": manifest.get("config_sha256"),
                "prompt": manifest.get("prompt"),
                "sampling": manifest.get("sampling"),
                "sampling_config": manifest.get("sampling_config"),
                "generation_config": manifest.get("generation_configs", {}).get(
                    benchmark, {}
                ),
                "scoreboard_compatible": manifest.get("scoreboard_compatible", False),
                "gpu": manifest.get("runtime", {}).get("gpu", []),
                "runtime": manifest.get("runtime", {}),
                "dependencies": manifest.get("dependencies", {}),
                "weight_path": manifest.get("weight_path"),
            },
        },
    }
    publication_task_path = (
        RESULT_ROOT / "publication" / "tasks" / f"{task_artifact_name(wkv_mode, benchmark)}.json"
    )
    write_atomic(
        publication_task_path,
        json.dumps(task_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )
    return {
        "identity": identity,
        "results_path": str(results_path),
        "samples_path": str(samples_path),
        "publication_path": str(publication_task_path),
        "sample_count": len(samples),
        "results_sha256": hash_path(results_path),
        "samples_sha256": hash_path(samples_path),
    }


def materialize_publication(manifest: dict[str, Any]) -> dict[str, Any]:
    """Write campaign publication JSON and a replayable local upload spool."""

    publication_root = RESULT_ROOT / "publication"
    task_root = publication_root / "tasks"
    expected = [
        {"identity": task_identity(mode, benchmark), "task": benchmark, "wkv_mode": mode}
        for mode in manifest.get("modes", [])
        for benchmark in BENCHMARKS
    ]
    campaign = {
        "schema_version": CAMPAIGN_SCHEMA,
        "publication_contract": "rwkv-producer-v1",
        "scoreboard_upload_ready": False,
        "scoreboard_mapping": "pending-scoreboard-rwkv-lm-eval-contract",
        "scoreboard_compatible": manifest.get("scoreboard_compatible", False),
        "run_key": manifest["run_key"],
        "config_digest": manifest.get("config_sha256") or value_digest(manifest),
        "registry_digest": value_digest(manifest.get("dataset_revisions", {})),
        "eval_contract_digest": value_digest(
            {"producer_schema": "rwkv-producer-v1", "task_schema": TASK_SCHEMA}
        ),
        "lighteval_version": LIGHTEVAL_VERSION,
        "campaign_name": manifest["campaign_name"],
        "model_name": MODEL,
        "weight_sha256": manifest.get("weight_sha256"),
        "backend_commit": manifest.get("backend_commit"),
        "configured_selectors": list(BENCHMARKS),
        "resolved_selectors": list(BENCHMARKS),
        "skipped_selectors": [],
        "expected_tasks": expected,
        "provenance": manifest,
    }
    campaign_path = publication_root / "campaign.json"
    write_atomic(
        campaign_path,
        json.dumps(campaign, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )
    spool: list[dict[str, Any]] = [
        {
            "schema_version": "rwkv-upload-spool-v1",
            "kind": "campaign",
            "path": str(campaign_path.relative_to(RESULT_ROOT)),
            "content_digest": value_digest(campaign),
        }
    ]
    for expected_task in expected:
        path = task_root / f"{task_artifact_name(expected_task['wkv_mode'], expected_task['task'])}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        spool.append(
            {
                "schema_version": "rwkv-upload-spool-v1",
                "kind": "task",
                "identity": expected_task["identity"],
                "path": str(path.relative_to(RESULT_ROOT)),
                "content_digest": value_digest(payload),
            }
        )
    spool_path = publication_root / "upload_spool.jsonl"
    write_atomic(
        spool_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in spool
        ),
    )
    return {
        "campaign_path": str(campaign_path),
        "spool_path": str(spool_path),
        "expected_task_count": len(expected),
        "published_task_count": len(spool) - 1,
        "campaign_digest": value_digest(campaign),
    }


def verify_publication() -> dict[str, Any]:
    """Validate every local publication reference and return stable digests."""

    publication_root = RESULT_ROOT / "publication"
    campaign_path = publication_root / "campaign.json"
    if not campaign_path.is_file():
        raise RuntimeError(f"missing publication campaign: {campaign_path}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise RuntimeError("publication campaign schema mismatch")
    expected = campaign.get("expected_tasks")
    if not isinstance(expected, list) or not expected:
        raise RuntimeError("publication campaign has no expected tasks")
    identities = [item.get("identity") for item in expected if isinstance(item, dict)]
    if len(identities) != len(expected) or len(set(identities)) != len(identities):
        raise RuntimeError("publication expected task identities are invalid")

    verified_tasks = []
    for item in expected:
        mode = item.get("wkv_mode")
        benchmark = item.get("task")
        if mode not in WKV_MODES or benchmark not in BENCHMARKS:
            raise RuntimeError(f"invalid expected task descriptor: {item}")
        path = publication_root / "tasks" / f"{task_artifact_name(mode, benchmark)}.json"
        if not path.is_file():
            raise RuntimeError(f"missing publication task: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = payload.get("task")
        if not isinstance(task, dict) or task.get("identity") != item["identity"]:
            raise RuntimeError(f"task identity mismatch: {path}")
        artifacts = task.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise RuntimeError(f"task artifacts missing: {path}")
        result_path = RESULT_ROOT / artifacts.get("results", "")
        samples_path = RESULT_ROOT / artifacts.get("samples", "")
        if not result_path.is_file() or not samples_path.is_file():
            raise RuntimeError(f"task artifact path missing: {path}")
        if artifacts.get("results_sha256") != hash_path(result_path):
            raise RuntimeError(f"results digest mismatch: {result_path}")
        if artifacts.get("samples_sha256") != hash_path(samples_path):
            raise RuntimeError(f"samples digest mismatch: {samples_path}")
        sample_count = 0
        with samples_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sample = json.loads(line)
                    if not isinstance(sample, dict):
                        raise RuntimeError(f"sample is not an object: {samples_path}")
                    if "response_evidence" not in sample:
                        raise RuntimeError(f"sample evidence missing: {samples_path}")
                    sample_count += 1
        if task.get("results", {}).get("sample_count") != sample_count:
            raise RuntimeError(f"sample count mismatch: {path}")
        verified_tasks.append(
            {"identity": item["identity"], "sample_count": sample_count, "digest": value_digest(payload)}
        )

    spool_path = publication_root / "upload_spool.jsonl"
    if not spool_path.is_file():
        raise RuntimeError(f"missing upload spool: {spool_path}")
    spool = [json.loads(line) for line in spool_path.read_text(encoding="utf-8").splitlines() if line]
    if not spool or spool[0].get("kind") != "campaign":
        raise RuntimeError("upload spool has no campaign record")
    if spool[0].get("content_digest") != value_digest(campaign):
        raise RuntimeError("upload spool campaign digest mismatch")
    for entry in spool[1:]:
        path = RESULT_ROOT / entry.get("path", "")
        if not path.is_file():
            raise RuntimeError(f"upload spool path missing: {path}")
        if entry.get("content_digest") != value_digest(
            json.loads(path.read_text(encoding="utf-8"))
        ):
            raise RuntimeError(f"upload spool digest mismatch: {path}")
    return {
        "campaign_digest": value_digest(campaign),
        "campaign_sha256": hash_path(campaign_path),
        "task_count": len(verified_tasks),
        "sample_count": sum(item["sample_count"] for item in verified_tasks),
        "spool_count": len(spool),
    }


def run(args: argparse.Namespace) -> int:
    benchmarks = list(dict.fromkeys(args.benchmark or BENCHMARKS))
    wkv_modes = WKV_MODES if args.wkv_mode == "both" else (args.wkv_mode,)
    if args.verify_only and args.preflight_only:
        raise ValueError("--verify-only and --preflight-only cannot be combined")
    if args.verify_only:
        verification = verify_publication()
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        write_preflight(
            wkv_modes,
            status="dry_run" if args.dry_run else "checking",
        )
        validate(
            require_backend=not args.dry_run,
            scoreboard_compatible=args.scoreboard_compatible,
        )
        if not args.dry_run:
            write_preflight(wkv_modes, status="ready")
    except Exception as error:
        write_preflight(
            wkv_modes,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    if args.preflight_only:
        print(
            json.dumps(
                json.loads((RESULT_ROOT / "preflight.json").read_text(encoding="utf-8")),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "wkv_modes": list(wkv_modes),
                    "service": service_probe(),
                    "backend_gpu_compatibility": backend_gpu_compatibility(),
                    "evaluations": {
                        mode: {
                            benchmark: evaluation_command(
                                benchmark,
                                wkv_mode=mode,
                                scoreboard_compatible=args.scoreboard_compatible,
                            )
                            for benchmark in benchmarks
                        }
                        for mode in wkv_modes
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = write_manifest(wkv_modes, args.scoreboard_compatible)
    for wkv_mode in wkv_modes:
        current = running_model()
        if current is not None and current != MODEL:
            raise RuntimeError(f"Port 8000 already serves {current}")
        if current == MODEL:
            observed_mode = service_wkv_mode()
            if observed_mode != wkv_mode:
                raise RuntimeError(
                    "Port 8000 already serves the requested model, but its "
                    f"RWKV WKV mode is {observed_mode!r}; expected {wkv_mode!r}. "
                    "Stop that service or rerun with the managed launcher."
                )
        process: subprocess.Popen[bytes] | None = None
        handle = None
        if current is None:
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            server_log = LOG_ROOT / wkv_mode / "server.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)
            handle = server_log.open("a", encoding="utf-8")
            process = subprocess.Popen(
                server_command(wkv_mode),
                cwd=BACKEND_ROOT,
                env=server_environment(wkv_mode),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_milestone(
                f"{wkv_mode}_server_started", pid=process.pid, model_name=MODEL
            )
        else:
            write_milestone(
                f"{wkv_mode}_server_reused",
                model_name=MODEL,
                service=service_probe(),
            )
        manifest["mode_runs"][wkv_mode].update(
            {"status": "running", "service_reused": process is None}
        )
        write_atomic(
            RESULT_ROOT / "campaign_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        try:
            if process is not None:
                wait_for_server(process, args.server_timeout)
            write_milestone(f"{wkv_mode}_server_ready", model_name=MODEL)
            for benchmark in benchmarks:
                run_benchmark(
                    benchmark,
                    args.force,
                    wkv_mode,
                    args.scoreboard_compatible,
                )
                materialize_task_artifacts(wkv_mode, benchmark, manifest)
                materialize_publication(manifest)
            manifest["mode_runs"][wkv_mode].update(
                {"status": "complete", "benchmarks": list(benchmarks)}
            )
            write_milestone(
                f"{wkv_mode}_evaluation_complete", benchmarks=benchmarks
            )
        except Exception as exc:
            manifest["mode_runs"][wkv_mode].update(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
            write_milestone(
                f"{wkv_mode}_failed", error=f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            stop_process(process)
            if handle is not None:
                handle.close()
            write_milestone(f"{wkv_mode}_server_stopped")
            write_atomic(
                RESULT_ROOT / "campaign_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )
    publication = materialize_publication(manifest)
    if args.verify:
        publication["verification"] = verify_publication()
    if args.digest:
        print(json.dumps(publication, ensure_ascii=False, indent=2, sort_keys=True))
    write_milestone("supervisor_exit_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
