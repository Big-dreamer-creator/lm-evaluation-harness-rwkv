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
exact model version, endpoint, benchmark selectors, served endpoint capacity
(`max_num_seqs`), RWKV profile, and result location. The `benchmarks` array
keeps its configured order as the stable tie-break, but benchmarks execute one
at a time in ascending pending-rollout order (see
`lm_eval/api/benchmark_scheduler.py`). For `rwkv7-http`, these ordinary names are resolved through the
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
| `model_name` | string | required | Complete served RWKV7 version name (top level for a single-model file, per `[[models]]` entry for a campaign) |
| `base_url` | string | required | HTTP(S) `/v1/completions` endpoint |
| `benchmarks` | list | required | Unique public benchmark names; configured order is the stable tie-break for the runtime ascending-pending-rollout order |
| `output_dir` | string | required | Result directory (top level for a single-model file, per `[[models]]` entry for a campaign) |
| `max_length` | int | required | Positive service context limit |
| `max_gen_toks` | int | `8192` | Remote RWKV choice-generation output budget; preserve the full input and reduce output to the remaining context window |
| `batch_size` | int | `1` | lm-eval request batch size |
| `max_num_seqs` | int | required | Served endpoint's own `--max-num-seqs`; the evaluator's flat request budget is derived from it (no hand-tuned request concurrency) |
| `device` | string | `"cpu"` | Evaluation-side device |
| `rwkv_profile` | table | required | Complete RWKV execution profile |
| `rwkv_profile.prompt_template` | string | required | Official RWKV template mode |
| `rwkv_profile.generation_prompt` | string | required | `fake_think` or `open_think` |
| `rwkv_profile.wkv_mode` | string | required | Recorded server WKV mode |
| `models` | array of tables | unset | The model ladder of a campaign; see [One file per benchmark set](#one-file-per-benchmark-set) |
| `model_overrides` | table | `{}` | Versioned fields forced onto every `[[models]]` entry last; the launch-time overlay hook |
| `apply_chat_template` | bool | `true` | Render the service's official template |
| `fewshot_as_multiturn` | bool | chat setting | Render few-shot examples as turns |
| `log_samples` | bool | `true` | Save inputs and model outputs |
| `seed` | list[int] | `[0,1234,1234,1234]` | Four lm-eval random seeds |
| `limit` | number | unset | Smoke-only sample limit |
| `use_cache` | string | unset | Response cache path |
| `weight_revision` | string | unset | Exact weight identity, including `ctx<number>` for Scoreboard; separate from the evaluation's `max_length` |

### Remote RWKV evaluation protocol

`rwkv7-remote` renders the official local tokenizer's template and sends token
IDs to `/v1/completions`. Set `tokenizer_name` to the same official tokenizer
used by the server and keep `apply_chat_template = true`. The native
open-thinking prefix ends in `<think` intentionally. A completed `</think>`
is required before final-answer grading; raw responses remain in the artifacts.
Without a tokenizer, this backend only passes raw text through and cannot apply
a chat template. `rwkv7-remote-chat` delegates template rendering to the server.

RWKV sampling presets are sent as top-level vLLM fields. The custom server must
support `penalty_decay` and its required sampler. Each document's repeats use
`seed[2] + repeat_ord`, including after resume, and share one concurrency budget.
The full input is preserved. `max_gen_toks` defaults to 8192 for choice rollouts;
set it explicitly when comparing models with different reasoning lengths.
A prompt that exhausts the context window aborts rather than receiving a score
for an empty completion. Multi-leaf benchmarks share one avg@k sizing budget.

The protocol identity is `rwkv-generative-v2`. Use a separate output/checkpoint
directory for runs made before this version. Old raw answers need a versioned
replay before their scores can be published using the new protocol.


## One file per benchmark set

A full campaign evaluates the same benchmark set on every RWKV7 size, so the
per-size TOMLs differ only in `model_name`, `tokenizer_name`, `output_dir`, and
the endpoint capacity each size is served with. Declare those once as
`[[models]]` entries and keep everything else in the shared top-level table:

```toml
# configs/eval/rwkv7_g1k_campaign_five_benchmarks_full_scoreboard.toml
schema_version = 1
backend = "rwkv7-remote"

base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["mc_taco", "mutual", "race", "winogender", "wmdp"]
include_path = ["configs/eval/offline_six_tasks"]
max_length = 25600
apply_chat_template = true
fewshot_as_multiturn = true
log_samples = true

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "open_think"
wkv_mode = "fp32io16"

[[models]]
model_name = "rwkv7-g1k-1.5b"
tokenizer_name = "/home/link/weight/rwkv-g1k-tokenizer/1.5b"
max_num_seqs = 64
output_dir = "results/rwkv7-g1k-1.5b-...-full"

[[models]]
model_name = "rwkv7-g1k-2.9b"
tokenizer_name = "/home/link/weight/rwkv-g1k-tokenizer/2.9b"
max_num_seqs = 64
output_dir = "results/rwkv7-g1k-2.9b-...-full"

# ... one entry per size, in run order
```

The top-level table holds what the models share; an entry holds what is specific
to one model and may carry any versioned field. Precedence, lowest to highest:
top-level table, `[[models]]` entry, `[model_overrides]`. Each entry must name
its own `model_name` and `output_dir`, and the resolved `output_dir`s and
`model_name`s must be distinct so that no two models write into the same result
directory or publish under the same name.

`lm-eval run -C <campaign.toml>` then evaluates every declared model: one child
process per model, running at the same time by default, because a campaign only
keeps the served endpoint busy when the whole ladder has requests in flight.
Each child re-reads the same file with `--only-model <model_name>`, so it keeps
its own request budget, checkpoint, result directory, and publication settings
— and resolves to exactly the config the equivalent single-model file
produced, checkpoint fingerprint included, so an in-progress run stays
resumable across the switch to one file. Per-model console output is appended to
`<output_dir>/console.log`. `--jobs N` caps how many models run at once;
`--only-model <model_name>` runs one model without fanning out (that is what the
children do, and it is also the way to re-run a single size by hand).

Launch-time overlays keep working: because the overlay cannot know which
per-model values it should beat, put campaign-wide launch knobs under
`[model_overrides]`.

```toml
# local launch overlay (gitignored temp/, one file for the whole ladder)
schema_version = 1
include = ["../configs/eval/rwkv7_g1k_campaign_five_benchmarks_full_scoreboard.toml"]

[model_overrides]
max_num_seqs = 200
timeout = 1800
max_retries = 6
```

The campaign exits non-zero if any model failed, after reporting a one-line
summary per model, so a batch launcher can tell success from partial failure.
`lm-eval publish -C <campaign.toml>` reads the top-level `[publication]` table;
pass `--rerun_reason` when a single model's note has to differ from the shared
one.

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
belong in each task YAML and are rejected by the versioned global schema. Each
`benchmarks` entry is a native lm-eval-harness task, group, or tag name,
resolved directly by `TaskManager` — there is no RWKV-specific adapter or
renaming layer. The `rwkv_profile` table selects only the official model
template and RWKV generation profile; sampling parameters (temperature,
top_p/top_k, penalties) always come from that profile and cannot be
overridden by task YAML. The Python CLI runner only loads this configuration,
selects the model adapter, and executes lm-eval; it contains no
benchmark-specific strategy.

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
