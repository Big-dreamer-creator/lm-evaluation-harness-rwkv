import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/run_rwkv_qwen_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_rwkv_qwen_campaign", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage_selectors_keep_rwkv_and_qwen_routes_separate():
    stages = MODULE.stage_definitions()

    assert stages["rwkv"].selectors == {
        benchmark: f"rwkv7_g1i_1_5b_20260805_ctx16384_{benchmark}"
        for benchmark in MODULE.BENCHMARKS
    }
    assert stages["qwen"].selectors["graphwalks"] == "graphwalks_128k"
    assert stages["rwkv"].config_path != stages["qwen"].config_path
    assert stages["rwkv"].server_command != stages["qwen"].server_command


def test_evaluation_command_preserves_graphwalks_explicit_samples():
    stage = MODULE.stage_definitions()["rwkv"]

    command = MODULE.evaluation_command(stage, "graphwalks")

    assert command[command.index("--tasks") + 1] == (
        "rwkv7_g1i_1_5b_20260805_ctx16384_graphwalks"
    )
    assert "--limit" not in command
    assert command[command.index("--cache_requests") + 1] == "refresh"
    metadata = command[command.index("--metadata") + 1]
    assert '"n_samples":350' in metadata
    include_path = command[command.index("--include_path") + 1]
    assert "lm_eval/tasks/graphwalks" in include_path
    assert "lm_eval/tasks/multiblimp" not in include_path
    assert "rwkv7_g1i_1_5b_20260805_ctx16384" in include_path
    model_args_index = command.index("--model_args")
    gen_kwargs_index = command.index("--gen_kwargs")
    model_args = command[model_args_index + 1 : gen_kwargs_index]
    assert 'rwkv_prompt_template="assistant"' in model_args
    assert "rwkv_generation_prompt=\"fake_think\"" in model_args
    assert command[gen_kwargs_index + 1 :] == [
        "max_gen_toks=512",
        'until=["\\n"]',
    ]

    qwen_command = MODULE.evaluation_command(
        MODULE.stage_definitions()["qwen"], "graphwalks"
    )
    assert qwen_command[qwen_command.index("--gen_kwargs") + 1] == "max_gen_toks=1024"

    resumed_command = MODULE.evaluation_command(
        stage, "graphwalks", reuse_request_cache=True
    )
    assert resumed_command[resumed_command.index("--cache_requests") + 1] == "true"


def test_mmlu_prox_uses_native_language_children_under_the_declared_parent():
    stages = MODULE.stage_definitions()

    rwkv_command = MODULE.evaluation_command(
        stages["rwkv"], "mmlu_prox", shard="af"
    )
    qwen_command = MODULE.evaluation_command(
        stages["qwen"], "mmlu_prox", shard="af"
    )

    assert stages["rwkv"].selectors["mmlu_prox"] == (
        "rwkv7_g1i_1_5b_20260805_ctx16384_mmlu_prox"
    )
    assert rwkv_command[rwkv_command.index("--tasks") + 1] == (
        "rwkv7_g1i_1_5b_20260805_ctx16384_mmlu_prox_af"
    )
    assert qwen_command[qwen_command.index("--tasks") + 1] == "mmlu_prox_af"
    assert rwkv_command[rwkv_command.index("--config") + 1] != (
        qwen_command[qwen_command.index("--config") + 1]
    )
    assert "shards/af" in rwkv_command[rwkv_command.index("--output_path") + 1]
    assert rwkv_command[rwkv_command.index("--gen_kwargs") + 1] == (
        "max_gen_toks=512"
    )
    assert qwen_command[qwen_command.index("--gen_kwargs") + 1] == (
        "max_gen_toks=512"
    )


def test_mmlu_prox_completion_requires_every_language_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RESULT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "ACTIVE_RESULT_ROOT", tmp_path)
    stage = MODULE.stage_definitions()["rwkv"]

    first = MODULE.MMLU_PROX_LANGUAGES[0]
    partial = MODULE.result_dir(stage, "mmlu_prox", first) / "model"
    partial.mkdir(parents=True)
    (partial / "results_partial.json").write_text("{}", encoding="utf-8")
    assert not MODULE.has_result(stage, "mmlu_prox")

    for shard in MODULE.MMLU_PROX_LANGUAGES[1:]:
        path = MODULE.result_dir(stage, "mmlu_prox", shard) / "model"
        path.mkdir(parents=True)
        (path / f"results_{shard}.json").write_text("{}", encoding="utf-8")
    assert MODULE.has_result(stage, "mmlu_prox")


def test_qwen_requires_rwkv_results(monkeypatch):
    monkeypatch.setattr(
        MODULE, "has_result", lambda _stage, benchmark: benchmark == "logiqa2"
    )

    with pytest.raises(RuntimeError, match="graphwalks"):
        MODULE.require_rwkv_results(["graphwalks", "logiqa2"])


def test_configured_graphwalks_indices_are_balanced_and_identical():
    rwkv = MODULE.load_toml(MODULE.stage_definitions()["rwkv"].config_path)
    qwen = MODULE.load_toml(MODULE.stage_definitions()["qwen"].config_path)
    rwkv_indices = rwkv["samples"]["rwkv7_g1i_1_5b_20260805_ctx16384_graphwalks"]
    qwen_indices = qwen["samples"]["graphwalks_128k"]

    assert rwkv_indices == qwen_indices
    assert rwkv_indices == [*range(200), *range(400, 550)]


def test_rwkv_graphwalks_prompt_compacts_duplicate_edges():
    from lm_eval.tasks.graphwalks.utils import (
        RWKV_PARENT_QUERY_REPEATS,
        doc_to_text_rwkv,
        process_results_rwkv,
    )

    prompt = doc_to_text_rwkv(
        {
            "prompt": (
                "example text\nHere is the graph to operate on:\n"
                "The graph has the following edges:\n"
                "alpha -> beta\nalpha -> beta\nalpha -> gamma\n"
                "\n\nOperation:\nFind the parents of node beta.\n\n"
                "You should reason through the operation step by step."
            )
        }
    )

    assert "example text" not in prompt
    assert "alpha -> beta" not in prompt
    assert prompt.count("Query record: node 1 has parent labels [0].") == (
        RWKV_PARENT_QUERY_REPEATS
    )
    assert prompt.endswith("Final Answer: [comma-separated labels]")
    assert process_results_rwkv(
        {
            "prompt": (
                "Here is the graph to operate on:\n"
                "alpha -> beta\nalpha -> beta\nalpha -> gamma\n"
                "\n\nOperation:\nFind the parents of node beta.\n\n"
                "You should reason through the operation step by step."
            ),
            "answer_nodes": ["alpha"],
        },
        ["Final Answer: [0]"],
    ) == {"f1": 1.0, "flexible_f1": 1.0}

    bfs_doc = {
        "prompt": (
            "Here is the graph to operate on:\n"
            "alpha -> alpha\nalpha -> beta\nalpha -> gamma\n"
            "\n\nOperation:\nPerform a BFS from node alpha with depth 1.\n\n"
            "You should reason through the operation step by step."
        ),
        "answer_nodes": ["beta", "gamma"],
    }
    bfs_prompt = doc_to_text_rwkv(bfs_doc)
    assert bfs_prompt.count(
        "Query record: node 0 has outgoing labels [1, 2]."
    ) == RWKV_PARENT_QUERY_REPEATS
    assert process_results_rwkv(bfs_doc, ["Final Answer: [1, 2]"]) == {
        "f1": 1.0,
        "flexible_f1": 1.0,
    }


def test_campaign_uses_startable_memory_target_and_wsl_proxy(monkeypatch):
    stages = MODULE.stage_definitions()
    qwen_command = stages["qwen"].server_command
    assert qwen_command[qwen_command.index("--gpu-memory-utilization") + 1] == "0.85"
    assert "--language-model-only" in qwen_command

    rwkv_environment = MODULE.server_environment(stages["rwkv"])
    assert rwkv_environment["RWKV_GPU_MEMORY_UTILIZATION"] == "0.85"
    assert rwkv_environment["RWKV_MAX_NUM_SEQS"] == "24"

    monkeypatch.setenv("HTTP_PROXY", "")
    monkeypatch.setenv("HTTPS_PROXY", "")
    evaluation_environment = MODULE.evaluation_environment()
    assert evaluation_environment["HTTP_PROXY"] == MODULE.WSL_PROXY
    assert evaluation_environment["HTTPS_PROXY"] == MODULE.WSL_PROXY
    assert evaluation_environment["LM_HARNESS_CACHE_PATH"] == str(
        MODULE.CACHE_ROOT / "requests"
    )
    assert "127.0.0.1" in evaluation_environment["NO_PROXY"].split(",")
    assert "localhost" in evaluation_environment["NO_PROXY"].split(",")


def test_high_frequency_logs_stay_on_linux_storage():
    stage = MODULE.stage_definitions()["rwkv"]

    log_path = MODULE.run_log_path(stage, "multiblimp")

    assert log_path.is_relative_to(MODULE.CACHE_ROOT)
    assert not log_path.is_relative_to(MODULE.RESULT_ROOT)


def test_official_rwkv_template_supports_all_prompt_modes():
    template = Path(
        "/mnt/e/code/Weights/"
        "rwkv7-g1i-1.5b-20260805-ctx16384/chat_template.jinja"
    )
    try:
        template_available = template.is_file()
    except OSError:
        template_available = False
    if not template_available:
        pytest.skip("local RWKV checkpoint is unavailable")

    rendered = MODULE.validate_rwkv_prompt_template(template)

    assert list(rendered) == ["assistant", "bot", "function_calling"]


def test_official_qwen_template_produces_tokenized_requests():
    from transformers import AutoTokenizer

    checkpoint = Path("/mnt/e/code/Weights/Qwen3.5-2B")
    try:
        checkpoint_available = checkpoint.is_dir()
    except OSError:
        checkpoint_available = False
    if not checkpoint_available:
        pytest.skip("local Qwen checkpoint is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    single = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
    ).input_ids
    batch = tokenizer(
        [rendered, rendered],
        add_special_tokens=False,
        return_attention_mask=False,
    ).input_ids

    assert rendered.endswith("<think>\n\n</think>\n\n")
    assert single and all(isinstance(token_id, int) for token_id in single)
    assert batch == [single, single]
