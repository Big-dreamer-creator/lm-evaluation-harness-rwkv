#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import tomllib
from jinja2 import Environment


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path("/home/creator/code/vllm-rwkv")
WEIGHTS_ROOT = Path(os.environ.get("LM_EVAL_WEIGHTS_ROOT", "/mnt/e/code/Weights"))
RESULT_ROOT = ROOT / "results/formal-five-benchmarks-20260818"
CACHE_ROOT = Path("/home/creator/.cache/lm-eval-rwkv/formal-five-benchmarks-20260818")
ACTIVE_RESULT_ROOT = CACHE_ROOT / "results"
LOG_ROOT = CACHE_ROOT / "logs"
EVALUATION_CWD = Path("/home/creator")
RWKV_MODEL = "rwkv7-g1i-1.5b-20260805-ctx16384"
QWEN_MODEL = "Qwen3.5-2B"
BASE_URL = "http://127.0.0.1:8000"
WSL_PROXY = "http://172.17.32.1:7897"
GIT = "/usr/bin/git"
NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"
BENCHMARKS = ("graphwalks", "multiblimp", "logiqa2", "tmmluplus", "mmlu_prox")
MMLU_PROX_LANGUAGES = (
    "af",
    "ar",
    "bn",
    "cs",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "mr",
    "ne",
    "pt",
    "ru",
    "sr",
    "sw",
    "te",
    "th",
    "uk",
    "ur",
    "vi",
    "wo",
    "yo",
    "zh",
    "zu",
)
MMLU_PROX_SAMPLES_PER_LANGUAGE = 11759
MMLU_PROX_MAX_GEN_TOKS = 512
RWKV_MAX_NUM_SEQS = 24
RWKV_GPU_MEMORY_UTILIZATION = "0.85"
RWKV_FAKE_THINK_DECODING = {
    "temperature": 1.0,
    "top_p": 0.28,
    "top_k": 32,
}
DATASET_REVISIONS = {
    "graphwalks": "f338bb265735a56a79f4b0f5def722c9c3268ead",
    "multiblimp": "de923efa8d2483d6b13364ee68e65308e990a991",
    "logiqa2": "4e294f9878e845c7ed2ca1f9c4f6aa4fe0693786",
    "tmmluplus": "0d61a3eb2087c21f4f63f199bca5f225ddaf03ac",
    "mmlu_prox": "8e6106a6c6ce1c5027e66cc338143cf997b2aa09",
}


@dataclass(frozen=True)
class Stage:
    name: str
    model_name: str
    config_path: Path
    selectors: dict[str, str]
    server_command: tuple[str, ...]


def stage_definitions() -> dict[str, Stage]:
    rwkv_config = ROOT / (
        "configs/eval/"
        "rwkv7_g1i_1_5b_20260805_ctx16384_"
        "graphwalks_multiblimp_logiqa2_tmmluplus_mmlu_prox.toml"
    )
    qwen_config = ROOT / (
        "configs/eval/qwen3_5_2b_graphwalks_multiblimp_logiqa2_tmmluplus_mmlu_prox.toml"
    )
    rwkv_server = BACKEND_ROOT / (
        "tools/rwkv_profile/serve_rwkv7_g1i_1_5b_20260805_ctx16384.sh"
    )
    qwen_server = (
        str(BACKEND_ROOT / ".venv/bin/vllm"),
        "serve",
        str(WEIGHTS_ROOT / QWEN_MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        QWEN_MODEL,
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "16384",
        "--gpu-memory-utilization",
        "0.85",
        "--language-model-only",
        "--enable-tokenizer-info-endpoint",
    )
    return {
        "rwkv": Stage(
            name="rwkv",
            model_name=RWKV_MODEL,
            config_path=rwkv_config,
            selectors={
                benchmark: f"rwkv7_g1i_1_5b_20260805_ctx16384_{benchmark}"
                for benchmark in BENCHMARKS
            },
            server_command=(str(rwkv_server),),
        ),
        "qwen": Stage(
            name="qwen",
            model_name=QWEN_MODEL,
            config_path=qwen_config,
            selectors={
                "graphwalks": "graphwalks_128k",
                "multiblimp": "multiblimp",
                "logiqa2": "logiqa2",
                "tmmluplus": "tmmluplus",
                "mmlu_prox": "mmlu_prox",
            },
            server_command=qwen_server,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "rwkv", "qwen"), default="all")
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-request-cache",
        action="store_true",
        help="Reuse previously built task request caches when resuming a campaign.",
    )
    parser.add_argument("--server-timeout", type=int, default=900)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        [GIT, "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        shell=False,
        text=True,
    ).stdout.strip()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def validate_stage(stage: Stage) -> None:
    if not stage.config_path.is_file():
        raise FileNotFoundError(stage.config_path)
    for executable in (ROOT / ".venv/bin/python", Path(stage.server_command[0])):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    config = load_toml(stage.config_path)
    selectors = config.get("benchmarks", config.get("tasks"))
    if selectors != list(stage.selectors.values()):
        raise ValueError(
            f"{stage.config_path} selectors do not match the declared campaign order"
        )
    if stage.name == "rwkv" and config.get("model_name") != RWKV_MODEL:
        raise ValueError("RWKV config does not select the exact RWKV checkpoint")
    if stage.name == "qwen" and config.get("model_args", {}).get("model") != QWEN_MODEL:
        raise ValueError("Qwen config does not select the exact Qwen checkpoint")


def result_dir(stage: Stage, benchmark: str, shard: str | None = None) -> Path:
    root = ACTIVE_RESULT_ROOT if benchmark == "mmlu_prox" and shard is not None else RESULT_ROOT
    path = root / stage.model_name / benchmark
    return path / "shards" / shard if shard is not None else path


def legacy_result_dir(stage: Stage, benchmark: str, shard: str | None = None) -> Path:
    path = RESULT_ROOT / stage.model_name / benchmark
    return path / "shards" / shard if shard is not None else path


def run_log_path(stage: Stage, benchmark: str, shard: str | None = None) -> Path:
    path = LOG_ROOT / stage.model_name / benchmark
    return path / "shards" / shard / "run.log" if shard is not None else path / "run.log"


def benchmark_shards(benchmark: str) -> tuple[str | None, ...]:
    if benchmark == "mmlu_prox":
        return MMLU_PROX_LANGUAGES
    return (None,)


def has_shard_result(stage: Stage, benchmark: str, shard: str | None) -> bool:
    paths = [result_dir(stage, benchmark, shard)]
    legacy_path = legacy_result_dir(stage, benchmark, shard)
    if legacy_path != paths[0]:
        paths.append(legacy_path)
    return any(
        result_path
        for path in paths
        for result_path in path.glob("**/results_*.json")
    )


def has_result(stage: Stage, benchmark: str) -> bool:
    if benchmark != "mmlu_prox":
        return has_shard_result(stage, benchmark, None)
    return all(
        has_shard_result(stage, benchmark, shard)
        for shard in MMLU_PROX_LANGUAGES
    )


def require_rwkv_results(benchmarks: list[str]) -> None:
    rwkv = stage_definitions()["rwkv"]
    missing = [benchmark for benchmark in benchmarks if not has_result(rwkv, benchmark)]
    if missing:
        raise RuntimeError(
            "Qwen stage requires completed RWKV results for: " + ", ".join(missing)
        )


def evaluation_command(
    stage: Stage,
    benchmark: str,
    *,
    shard: str | None = None,
    reuse_request_cache: bool = False,
) -> list[str]:
    if shard is not None and (
        benchmark != "mmlu_prox" or shard not in MMLU_PROX_LANGUAGES
    ):
        raise ValueError(f"Unsupported {benchmark} shard: {shard}")
    output_dir = result_dir(stage, benchmark, shard)
    cache_path = CACHE_ROOT / "responses" / stage.model_name / benchmark
    if shard is not None:
        cache_path /= shard
    task_selector = stage.selectors[benchmark]
    if shard is not None:
        task_selector = (
            f"{task_selector}_{shard}"
            if stage.name == "rwkv"
            else f"mmlu_prox_{shard}"
        )
    metadata = {
        "benchmark_name": benchmark,
        "model_name": stage.model_name,
        "n_samples": (
            MMLU_PROX_SAMPLES_PER_LANGUAGE
            if shard is not None
            else 350 if benchmark == "graphwalks" else "full"
        ),
        "dataset_revision": DATASET_REVISIONS[benchmark],
        "campaign_name": "formal-five-benchmarks-20260818",
        "evaluation_scope": (
            "full_language_shard"
            if shard is not None
            else "128k_and_shorter_ctx16384_subset"
            if benchmark == "graphwalks"
            else "full"
        ),
    }
    if shard is not None:
        metadata["benchmark_shard"] = shard
        metadata["max_gen_toks"] = MMLU_PROX_MAX_GEN_TOKS
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "lm_eval",
        "run",
        "--config",
        str(stage.config_path),
        "--tasks",
        task_selector,
        "--output_path",
        str(output_dir),
        "--use_cache",
        str(cache_path),
        "--cache_requests",
        "true" if reuse_request_cache else "refresh",
        "--metadata",
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        "--verbosity",
        "INFO",
    ]
    configured_include_path = load_toml(stage.config_path).get("include_path", [])
    if isinstance(configured_include_path, str):
        configured_include_path = [configured_include_path]
    adapter_dir_name = (
        stage.selectors[benchmark].removesuffix(f"_{benchmark}")
        if stage.name == "rwkv"
        else ""
    )
    include_paths = []
    for path in configured_include_path:
        include_path = Path(path)
        if not include_path.is_absolute():
            include_path = ROOT / include_path
        if include_path.name not in {benchmark, adapter_dir_name}:
            continue
        if shard is not None and include_path.name == "mmlu_prox":
            include_path /= shard
        include_paths.append(str(include_path))
    absolute_include_path = os.pathsep.join(include_paths)
    if absolute_include_path:
        command.extend(["--include_path", absolute_include_path])
    config = load_toml(stage.config_path)
    model_overrides = (
        config.get("model_overrides", {})
        or config.get("metadata", {}).get("model_overrides", {})
    )
    model_overrides = model_overrides.get(benchmark, {})
    if model_overrides:
        if stage.name == "rwkv":
            profile = config["rwkv_profile"]
            model_args = {
                "model": config["model_name"],
                "base_url": config["base_url"],
                "rwkv_prompt_template": profile["prompt_template"],
                "rwkv_generation_prompt": profile["generation_prompt"],
                "rwkv_sampling_mode": profile["sampling_mode"],
                "num_concurrent": config["num_concurrent"],
                "max_length": config["max_length"],
            }
        else:
            model_args = dict(config["model_args"])
        model_args.update(model_overrides)
        command.extend(
            [
                "--model_args",
                *[
                    f"{name}={json.dumps(value)}"
                    for name, value in model_args.items()
                ],
            ]
        )
    task_overrides = (
        config.get("task_overrides", {})
        or config.get("metadata", {}).get("task_overrides", {})
    )
    task_overrides = task_overrides.get(benchmark, {})
    if stage.name == "rwkv" and benchmark == "mmlu_prox":
        task_overrides = {"max_gen_toks": MMLU_PROX_MAX_GEN_TOKS, **task_overrides}
    if task_overrides:
        command.extend(
            [
                "--gen_kwargs",
                *[
                    f"{name}={json.dumps(value)}"
                    for name, value in task_overrides.items()
                ],
            ]
        )
    return command


def write_mmlu_prox_shard_index(stage: Stage) -> None:
    records = []
    for shard in MMLU_PROX_LANGUAGES:
        paths = []
        for root in (
            result_dir(stage, "mmlu_prox", shard),
            legacy_result_dir(stage, "mmlu_prox", shard),
        ):
            paths.extend(root.glob("**/results_*.json"))
        paths = sorted(paths, key=lambda path: path.stat().st_mtime_ns)
        if not paths:
            raise RuntimeError(f"Missing MMLU-ProX result shard: {shard}")
        records.append(
            {
                "language": shard,
                "expected_samples": MMLU_PROX_SAMPLES_PER_LANGUAGE,
                "result_path": str(paths[-1]),
            }
        )
    index = {
        "benchmark_name": "mmlu_prox",
        "model_name": stage.model_name,
        "dataset_revision": DATASET_REVISIONS["mmlu_prox"],
        "n_shards": len(records),
        "expected_samples": len(records) * MMLU_PROX_SAMPLES_PER_LANGUAGE,
        "shards": records,
    }
    path = result_dir(stage, "mmlu_prox") / "shard_index.json"
    path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def request_json(path: str, timeout: float = 3) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=timeout)
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
        payload = request_json("/v1/models")
    except (OSError, json.JSONDecodeError):
        return None
    models = payload.get("data", [])
    if len(models) != 1 or not isinstance(models[0].get("id"), str):
        raise RuntimeError(f"Expected exactly one served model, got: {models}")
    return models[0]["id"]


def wait_for_server(process: subprocess.Popen, model_name: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with status {process.returncode}")
        if running_model() == model_name:
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {model_name}")


def server_environment(stage: Stage) -> dict[str, str]:
    env = os.environ.copy()
    temp_dir = CACHE_ROOT / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda-13.0",
            "TMPDIR": str(temp_dir),
            "TMP": str(temp_dir),
            "TEMP": str(temp_dir),
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    if stage.name == "rwkv":
        env["RWKV_MAX_NUM_SEQS"] = str(RWKV_MAX_NUM_SEQS)
        env["RWKV_GPU_MEMORY_UTILIZATION"] = RWKV_GPU_MEMORY_UTILIZATION
    return env


def evaluation_environment() -> dict[str, str]:
    env = os.environ.copy()
    temp_dir = CACHE_ROOT / "tmp"
    request_cache_dir = CACHE_ROOT / "requests"
    temp_dir.mkdir(parents=True, exist_ok=True)
    request_cache_dir.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HF_HOME": "/home/creator/.cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_XET": "1",
            "LM_HARNESS_CACHE_PATH": str(request_cache_dir),
            "TMPDIR": str(temp_dir),
            "TMP": str(temp_dir),
            "TEMP": str(temp_dir),
        }
    )
    for variable in ("HTTP_PROXY", "HTTPS_PROXY"):
        if not env.get(variable):
            env[variable] = WSL_PROXY
    local_hosts = {"127.0.0.1", "localhost"}
    no_proxy = env.get("NO_PROXY", "")
    local_hosts.update(host.strip() for host in no_proxy.split(",") if host.strip())
    env["NO_PROXY"] = ",".join(sorted(local_hosts))
    env["no_proxy"] = env["NO_PROXY"]
    source_path = str(ROOT)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    return env


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def publish_file(source: Path, destination: Path) -> None:
    try:
        shutil.copy2(source, destination)
    except PermissionError:
        shutil.copyfile(source, destination)


def start_server(stage: Stage) -> tuple[subprocess.Popen | None, TextIO | None]:
    current = running_model()
    if current == stage.model_name:
        return None, None
    if current is not None:
        raise RuntimeError(
            f"Port 8000 already serves {current}; expected {stage.model_name}"
        )
    server_dir = LOG_ROOT / stage.model_name / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (server_dir / "server.log").open("a", encoding="utf-8")
    log_handle.write(json.dumps({"command": stage.server_command}) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        stage.server_command,
        cwd=BACKEND_ROOT,
        env=server_environment(stage),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=True,
    )
    return process, log_handle


def telemetry_loop(path: Path, stop_event: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if path.stat().st_size == 0:
            writer.writerow(
                [
                    "timestamp",
                    "gpu_index",
                    "utilization_gpu",
                    "memory_used",
                    "memory_total",
                ]
            )
        while not stop_event.is_set():
            completed = subprocess.run(
                [
                    NVIDIA_SMI,
                    "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=False,
                shell=False,
                text=True,
            )
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    writer.writerow([value.strip() for value in line.split(",")])
                handle.flush()
            stop_event.wait(2)


def validate_rwkv_prompt_template(path: Path) -> dict[str, str]:
    template = Environment().from_string(path.read_text(encoding="utf-8"))
    expected = {
        "assistant": "User: hello\n\nAssistant: <think></think>\n",
        "bot": "User✿hello✿\nBot✿<think></think>\n",
        "function_calling": "### User\nhello\n### Assistant\n<think></think>\n",
    }
    rendered = {}
    for mode in expected:
        output = template.render(
            messages=[{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            rwkv_prompt_template=mode,
            rwkv_generation_prompt="fake_think",
        )
        if output.endswith("<think></think"):
            output += ">\n"
        rendered[mode] = output
    if rendered != expected:
        raise RuntimeError("RWKV official prompt-template validation failed")
    return rendered


def write_manifest(stage: Stage) -> None:
    model_dir = RESULT_ROOT / stage.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    backend_requirements = BACKEND_ROOT / "requirements" / f"{stage.name}.txt"
    config = load_toml(stage.config_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": asdict(stage) | {"config_path": str(stage.config_path)},
        "harness_commit": git_commit(ROOT),
        "vllm_rwkv_commit": git_commit(BACKEND_ROOT),
        "config_sha256": sha256(stage.config_path),
        "backend_requirements": str(backend_requirements),
        "backend_requirements_sha256": sha256(backend_requirements),
        "dataset_revisions": DATASET_REVISIONS,
        "registry_audit": {
            "lighteval_commit": "932e1f2f4c5af3e926534f12b2a84a3ae18d6d3f",
            "evalscope_commit": "ec0f2b3fb7b7d7493345c6d0917b7dad7e98f57c",
            "exact_requested_benchmarks_present": False,
        },
    }
    if stage.name == "rwkv":
        template = WEIGHTS_ROOT / RWKV_MODEL / "chat_template.jinja"
        profile = config["rwkv_profile"]
        manifest["chat_template_sha256"] = sha256(template)
        manifest["wkv_mode"] = "fp32io16"
        manifest["rwkv_max_num_seqs"] = RWKV_MAX_NUM_SEQS
        manifest["rwkv_gpu_memory_utilization"] = RWKV_GPU_MEMORY_UTILIZATION
        manifest["rwkv_prompt_template_validation"] = {
            "supported_modes": list(validate_rwkv_prompt_template(template)),
            "selected_mode": profile["prompt_template"],
            "generation_prompt": profile["generation_prompt"],
            "decoding": RWKV_FAKE_THINK_DECODING,
        }
        manifest["benchmark_model_overrides"] = (
            config.get("model_overrides", {})
            or config.get("metadata", {}).get("model_overrides", {})
        )
    else:
        template = WEIGHTS_ROOT / QWEN_MODEL / "chat_template.jinja"
        manifest["chat_template_sha256"] = sha256(template)
        manifest["language_model_only"] = True
    manifest["mmlu_prox_max_gen_toks"] = MMLU_PROX_MAX_GEN_TOKS
    (model_dir / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_benchmark(
    stage: Stage,
    benchmark: str,
    *,
    force: bool = False,
    reuse_request_cache: bool = False,
) -> None:
    for shard in benchmark_shards(benchmark):
        if has_shard_result(stage, benchmark, shard) and not force:
            continue
        output_dir = result_dir(stage, benchmark, shard)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = evaluation_command(
            stage,
            benchmark,
            shard=shard,
            reuse_request_cache=reuse_request_cache,
        )
        log_path = run_log_path(stage, benchmark, shard)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=EVALUATION_CWD,
                env=evaluation_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
            )
        publish_file(log_path, output_dir / "run.log")
        if completed.returncode != 0:
            shard_label = f"/{shard}" if shard is not None else ""
            raise RuntimeError(
                f"{stage.model_name} {benchmark}{shard_label} failed with "
                f"status {completed.returncode}"
            )
        if not has_shard_result(stage, benchmark, shard):
            raise RuntimeError(
                f"{stage.model_name} {benchmark}/{shard} produced no results JSON"
            )
    if benchmark == "mmlu_prox":
        write_mmlu_prox_shard_index(stage)
    if not has_result(stage, benchmark):
        raise RuntimeError(f"{stage.model_name} {benchmark} produced no results JSON")


def run_stage(
    stage: Stage,
    benchmarks: list[str],
    force: bool,
    timeout: int,
    reuse_request_cache: bool = False,
) -> None:
    validate_stage(stage)
    write_manifest(stage)
    process, log_handle = start_server(stage)
    telemetry_stop = threading.Event()
    telemetry = threading.Thread(
        target=telemetry_loop,
        args=(LOG_ROOT / stage.model_name / "gpu_telemetry.csv", telemetry_stop),
        daemon=True,
    )
    telemetry.start()
    try:
        if process is not None:
            wait_for_server(process, stage.model_name, timeout)
        for benchmark in benchmarks:
            if has_result(stage, benchmark) and not force:
                continue
            run_benchmark(
                stage,
                benchmark,
                force=force,
                reuse_request_cache=reuse_request_cache,
            )
    finally:
        telemetry_stop.set()
        telemetry.join(timeout=5)
        stop_process(process)
        if log_handle is not None:
            log_handle.close()
        telemetry_path = LOG_ROOT / stage.model_name / "gpu_telemetry.csv"
        if telemetry_path.is_file():
            publish_file(
                telemetry_path,
                RESULT_ROOT / stage.model_name / "gpu_telemetry.csv",
            )
        server_log = LOG_ROOT / stage.model_name / "server" / "server.log"
        if server_log.is_file():
            published_server_dir = RESULT_ROOT / stage.model_name / "server"
            published_server_dir.mkdir(parents=True, exist_ok=True)
            publish_file(server_log, published_server_dir / "server.log")


def main() -> int:
    args = parse_args()
    stages = stage_definitions()
    benchmarks = list(dict.fromkeys(args.benchmark or BENCHMARKS))
    stage_names = ["rwkv", "qwen"] if args.stage == "all" else [args.stage]
    for stage_name in stage_names:
        validate_stage(stages[stage_name])
    if args.dry_run:
        print(
            json.dumps(
                {
                    stage_name: {
                        "server": stages[stage_name].server_command,
                        "evaluations": [
                            evaluation_command(
                                stages[stage_name],
                                benchmark,
                                shard=shard,
                                reuse_request_cache=args.reuse_request_cache,
                            )
                            for benchmark in benchmarks
                            for shard in benchmark_shards(benchmark)
                        ],
                    }
                    for stage_name in stage_names
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    for stage_name in stage_names:
        if stage_name == "qwen":
            require_rwkv_results(benchmarks)
        run_stage(
            stages[stage_name],
            benchmarks,
            args.force,
            args.server_timeout,
            args.reuse_request_cache,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
