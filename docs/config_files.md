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

Unversioned configuration keys correspond directly to CLI arguments. The
versioned RWKV preset uses the smaller public schema documented below. See the
[CLI Reference](interface.md#lm-eval-run) for detailed descriptions of native
options.

### RWKV7 complete run

Start the independent `vllm-rwkv` HTTP service from its own repository and uv environment:

```bash
cd /path/to/vllm-rwkv
# Start the repository's RWKV7 server launcher here.
```

Configure the launcher to pass `--enable-tokenizer-info-endpoint`. The service
must expose `/tokenizer_info` so lm-eval receives the same official RWKV chat
template used by the inference backend.

`configs/eval/lm_eval.toml` is the versioned, validated preset for
`rwkv7-g1i-1.5b-20260805-ctx16384`. Run it directly with:

```bash
uv run --no-sync python -m lm_eval run -C configs/eval/lm_eval.toml
```

The `rwkv7-http` backend itself accepts any RWKV7 served model name. For another
checkpoint size or revision, copy the preset and change `model_name`,
`max_length`, `benchmarks`, and `output_dir`. The loader derives lm-eval's
`model_args` and result metadata from this public schema, removing duplicated
model and profile fields. `rwkv_profile.wkv_mode` remains a declared server-side
fact and must be checked against the vllm-rwkv startup log. The TOML owns the
exact model version, endpoint, ordered benchmark selectors, RWKV profile,
concurrency, and result location. The runner preserves the `benchmarks` array
order. For `rwkv7-http`, these ordinary names are resolved through the
model-specific task adapter declared in the adapted task YAML; internal RWKV
task names never appear in this file. Each benchmark YAML owns its prompt,
answer extraction, metrics, stop conditions, and `max_gen_toks`. See the
[RWKV evaluation guide](../RWKV_EVALUATION_GUIDE.md) for the complete server and
evaluation workflow. The deprecated `temp/` launch scripts are not used.

The preset currently exposes 15 benchmark families in configured run order:
RACE, DROP, XQuAD, BABILong, InfiniteBench, RULER, WMDP, CRUXEval,
Inverse Scaling Prize, Model-Written Evals, HumanEval-Infilling, MuTual,
MC-TACO, Discrim-Eval, and Winogender. PALOMA remains excluded because dataset
access is still blocked; it must not be represented as a completed benchmark.

## Config Schema

The repository preset follows the same strict configuration approach as the
Helicopter LightEval runner: `schema_version = 1`, an allowlist of public fields,
required-field checks, duplicate benchmark rejection, and derived internal
arguments. Unknown fields fail before model or dataset initialization.

| Versioned RWKV field | Type | Default | Description |
|----------------------|------|---------|-------------|
| `schema_version` | int | required | Must be `1` |
| `backend` | string | required | Must be `"rwkv7-http"` |
| `model_name` | string | required | Complete served RWKV7 version name |
| `base_url` | string | required | HTTP(S) `/v1/completions` endpoint |
| `benchmarks` | list | required | Unique public benchmark names in execution order |
| `output_dir` | string | required | Result directory |
| `max_length` | int | required | Positive service context limit |
| `batch_size` | int | `1` | lm-eval request batch size |
| `num_concurrent` | int | `5` | HTTP request concurrency |
| `device` | string | `"cpu"` | Evaluation-side device |
| `rwkv_profile` | table | required | Complete RWKV execution profile |
| `rwkv_profile.prompt_template` | string | required | Official RWKV template mode |
| `rwkv_profile.generation_prompt` | string | required | `fake_think` or `open_think` |
| `rwkv_profile.sampling_mode` | string | required | RWKV profile or task-native sampling |
| `rwkv_profile.wkv_mode` | string | required | Recorded server WKV mode |
| `apply_chat_template` | bool | `true` | Render the service's official template |
| `fewshot_as_multiturn` | bool | chat setting | Render few-shot examples as turns |
| `log_samples` | bool | `true` | Save inputs and model outputs |
| `seed` | list[int] | `[0,1234,1234,1234]` | Four lm-eval random seeds |
| `limit` | number | unset | Smoke-only sample limit |
| `use_cache` | string | unset | Response cache path |

Unversioned TOML and YAML files remain compatible with upstream lm-eval. Their
keys correspond directly to CLI arguments:

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
include = "lm_eval.toml"
benchmarks = ["race"]
limit = 10
output_dir = "results/smoke/race"
```

Included paths are relative to the including file. Nested tables are merged, and local values override included values. CLI arguments still have the highest priority.

The array order is the run order; do not hide a campaign sequence in a
model-specific group YAML. Benchmark prompt construction (`doc_to_text`), answer
extraction and filters, metrics, stop conditions (`until`), and `max_gen_toks`
belong in each task YAML and are rejected by the versioned global schema. For
RWKV, task YAML additionally declares `metadata.task_adapter` and
`metadata.benchmark_name`; the generic task manager uses those declarations to
resolve each public name to one or more adapted tasks. The `rwkv_profile` table selects
only the official model template and RWKV sampling profile. The Python CLI
runner only loads this configuration, selects the model adapter, and executes
lm-eval; it contains no benchmark-specific strategy.

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
