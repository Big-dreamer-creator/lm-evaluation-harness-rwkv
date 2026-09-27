import pytest

from lm_eval.config.evaluate_config import EvaluatorConfig


_BASE_VERSIONED_TOML = """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1i-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["race"]
output_dir = "results/rwkv7-g1i-1.5b-remote"
max_length = 16384

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
"""


def _write(tmp_path, extra: str):
    config = tmp_path / "config.toml"
    config.write_text(_BASE_VERSIONED_TOML + extra, encoding="utf-8")
    return config


def test_scheduler_defaults_to_empty_when_absent(tmp_path):
    cfg = EvaluatorConfig.from_config(_write(tmp_path, ""))
    assert cfg.scheduler == {}


def test_scheduler_max_concurrent_benchmarks_is_normalized(tmp_path):
    config = _write(
        tmp_path,
        "\n[scheduler]\nmax_concurrent_benchmarks = 5\n",
    )
    cfg = EvaluatorConfig.from_config(config)
    assert cfg.scheduler == {"max_concurrent_benchmarks": 5}


def test_scheduler_rejects_unknown_field(tmp_path):
    config = _write(
        tmp_path,
        "\n[scheduler]\nbogus_field = 1\n",
    )
    with pytest.raises(ValueError, match="Unknown scheduler fields"):
        EvaluatorConfig.from_config(config)


def test_scheduler_rejects_non_table(tmp_path):
    # A bare `scheduler = true` must come before any `[table]` header in TOML,
    # otherwise it's parsed as a key inside the preceding table.
    config = tmp_path / "config.toml"
    config.write_text(
        "scheduler = true\n" + _BASE_VERSIONED_TOML, encoding="utf-8"
    )
    with pytest.raises(TypeError, match="scheduler must be a table"):
        EvaluatorConfig.from_config(config)


@pytest.mark.parametrize(
    "toml_literal", ["0", "-1", "1.5", "true", '"3"']
)
def test_scheduler_rejects_non_positive_int_max_concurrent_benchmarks(
    tmp_path, toml_literal
):
    config = _write(
        tmp_path,
        f"\n[scheduler]\nmax_concurrent_benchmarks = {toml_literal}\n",
    )
    with pytest.raises(ValueError, match="max_concurrent_benchmarks"):
        EvaluatorConfig.from_config(config)
