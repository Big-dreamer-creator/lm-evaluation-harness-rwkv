import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/run_rwkv_five_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_rwkv_five_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_campaign_declares_exact_requested_scope_and_protocol():
    assert MODULE.BENCHMARKS == (
        "moral_stories",
        "haerae",
        "jsonschema_bench",
        "gsm8k_platinum",
        "aexams",
    )
    assert MODULE.MODEL == "rwkv7-g1i-1.5b-20260805-ctx16384"
    assert MODULE.DATASET_REVISIONS["aexams"] == "bc7a29346dbcaa16a8cd883b1f3e681ab2b7ff2a"


def test_evaluation_commands_keep_include_paths_narrow_and_generation_limits():
    command = MODULE.evaluation_command("jsonschema_bench")
    assert command[command.index("--tasks") + 1].endswith("_jsonschema_bench")
    include_path = command[command.index("--include_path") + 1]
    assert "lm_eval/tasks/jsonschema_bench" in include_path
    assert "lm_eval/tasks/moral_stories" not in include_path
    assert command[-2:] == ["max_gen_toks=2048", "do_sample=true"]


def test_milestones_are_atomic_and_recoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RESULT_ROOT", tmp_path)
    MODULE.write_milestone("server_ready", model_name=MODULE.MODEL)
    MODULE.write_milestone("moral_stories_complete", n_samples=12000)
    milestones = json.loads((tmp_path / "milestones.json").read_text())
    assert milestones["server_ready"]["model_name"] == MODULE.MODEL
    assert milestones["moral_stories_complete"]["n_samples"] == 12000
    assert not (tmp_path / "milestones.json.tmp").exists()
