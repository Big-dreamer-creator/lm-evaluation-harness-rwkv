# Configuration Guide

This guide explains how to use TOML or YAML configuration files with `lm-eval` to define reusable evaluation settings. Task and benchmark definitions remain YAML.

## Overview

Instead of passing many CLI arguments, define evaluation parameters in a TOML file:

```bash
# Instead of:
lm-eval run --model hf --model_args pretrained=gpt2,dtype=float32 --tasks hellaswag arc_easy --num_fewshot 5 --batch_size 8 --device cuda:0

# Use:
lm-eval run --config eval_config.toml
```

CLI arguments override config file values, so you can set defaults in a config file and override specific settings:

```bash
lm-eval run --config eval_config.toml --tasks mmlu --limit 100
```

## Quick Reference

All configuration keys correspond directly to CLI arguments. See the [CLI Reference](interface.md#lm-eval-run) for detailed descriptions of each option.

### RWKV7 complete run

Start the independent `vllm-rwkv` HTTP service from its own repository and uv environment:

```bash
cd /path/to/vllm-rwkv
# Start the repository's RWKV7 server launcher here.
```

Configure the launcher to pass `--enable-tokenizer-info-endpoint`. The service
must expose `/tokenizer_info` so lm-eval receives the same official RWKV chat
template used by the inference backend.

Then run every benchmark adapted for `rwkv7-g1i-1.5b-20260805-ctx16384` with the repository config:

```bash
uv run --no-sync python -m lm_eval run -C lm-eval-rwkv.toml
```

The TOML owns the model endpoint, concurrency, complete task set, and result location. Each benchmark YAML owns its prompt, generation limits, filters, and metrics. The deprecated `temp/` launch scripts are not used.

## Config Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"hf"` | Model type/provider |
| `model_args` | dict | `{}` | Model constructor arguments |
| `tasks` | list/string | required | Tasks to evaluate |
| `num_fewshot` | int/null | `null` | Few-shot example count |
| `batch_size` | int/string | `1` | Batch size or "auto" |
| `max_batch_size` | int/null | `null` | Max batch size for auto |
| `device` | string/null | `"cuda:0"` | Device to use |
| `limit` | float/null | `null` | Example limit per task |
| `samples` | dict/null | `null` | Specific sample indices |
| `use_cache` | string/null | `null` | Response cache path |
| `cache_requests` | string/dict | `{}` | Request cache settings |
| `output_path` | string/null | `null` | Results output path |
| `log_samples` | bool | `false` | Save model I/O |
| `predict_only` | bool | `false` | Skip metrics |
| `apply_chat_template` | bool/string | `false` | Chat template |
| `system_instruction` | string/null | `null` | System prompt |
| `fewshot_as_multiturn` | bool/null | `null` | Multi-turn few-shot |
| `include_path` | string/null | `null` | External tasks path |
| `gen_kwargs` | dict | `{}` | Generation arguments |
| `wandb_args` | dict | `{}` | W&B init arguments |
| `hf_hub_log_args` | dict | `{}` | HF Hub logging |
| `seed` | list/int | `[0,1234,1234,1234]` | Random seeds |
| `trust_remote_code` | bool | `false` | Trust remote code |
| `metadata` | dict | `{}` | Task metadata |

---

## Example

```toml
# basic_eval.toml
model = "hf"
tasks = ["hellaswag", "arc_easy"]
num_fewshot = 0
batch_size = "auto"
device = "cuda:0"
output_path = "./results/gpt2/"
log_samples = true

[model_args]
pretrained = "gpt2"
dtype = "float32"
```

## Inheriting a run config

Small or smoke runs can include a formal config and override only their scope:

```toml
include = "lm-eval-rwkv.toml"
tasks = ["rwkv7_g1i_1_5b_20260805_ctx16384_race"]
limit = 10
output_path = "results/smoke/race"
```

Included paths are relative to the including file. Nested tables are merged, and local values override included values. CLI arguments still have the highest priority.

Generation settings, prompts, filters, and metrics should be tuned in each benchmark's task YAML, not in a global run config. This preserves the native task protocol for non-RWKV backends.

---

## Programmatic Usage

For loading config files in Python, see the [Python API Guide](python-api.md#using-evaluatorconfig).

---

## Validation

Validate your configuration before running:

```bash
# Check that tasks exist
lm-eval validate --tasks hellaswag,arc_easy

# With external tasks
lm-eval validate --tasks my_task --include_path /path/to/tasks
```

---

## Tips

1. **Start simple**: Begin with minimal config and add options as needed
2. **Use CLI overrides**: Set defaults in config, override with CLI for experiments
3. **Separate concerns**: Create different configs for different model families or task sets
4. **Version control**: Commit config files alongside results for reproducibility
5. **Use comments**: TOML and YAML support `#` comments to document your choices
