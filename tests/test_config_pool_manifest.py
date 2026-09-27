import pytest

from lm_eval.config.evaluate_config import EvaluatorConfig


_HEAD = """
schema_version = 1
backend = "rwkv7-remote"
model_name = "rwkv7-g1i-1.5b"
benchmarks = ["race"]
output_dir = "results/rwkv7-g1i-1.5b-remote"
max_length = 16384
"""

_PROFILE = """
[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
"""


def _write(tmp_path, extra: str):
    # `extra` must be inserted before the `[rwkv_profile]` table header --
    # bare top-level keys after a `[table]` header are parsed as belonging to
    # that table in TOML, not as new top-level keys.
    config = tmp_path / "config.toml"
    config.write_text(_HEAD + extra + _PROFILE, encoding="utf-8")
    return config


def test_pool_manifest_used_instead_of_base_url(tmp_path):
    config = _write(tmp_path, '\npool_manifest = "pools/rwkv7_pool.toml"\n')
    cfg = EvaluatorConfig.from_config(config)

    assert cfg.model_args == {
        "model": "rwkv7-g1i-1.5b",
        "pool_manifest": "pools/rwkv7_pool.toml",
        "num_concurrent": 5,
        "max_length": 16384,
        "rwkv_prompt_template": "assistant",
        "rwkv_generation_prompt": "fake_think",
    }


def test_requires_exactly_one_of_base_url_or_pool_manifest(tmp_path):
    config = _write(tmp_path, "")
    with pytest.raises(
        ValueError, match="Exactly one of base_url or pool_manifest must be set"
    ):
        EvaluatorConfig.from_config(config)


def test_rejects_both_base_url_and_pool_manifest(tmp_path):
    config = _write(
        tmp_path,
        '\nbase_url = "https://api.rwkv.rs/v1/completions"\n'
        'pool_manifest = "pools/rwkv7_pool.toml"\n',
    )
    with pytest.raises(
        ValueError, match="Exactly one of base_url or pool_manifest must be set"
    ):
        EvaluatorConfig.from_config(config)


def test_pool_manifest_must_be_non_empty_string(tmp_path):
    config = _write(tmp_path, '\npool_manifest = ""\n')
    with pytest.raises(ValueError, match="pool_manifest"):
        EvaluatorConfig.from_config(config)
