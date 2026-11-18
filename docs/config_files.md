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

`rwkv7-remote` is the only RWKV HTTP backend. It speaks standard OpenAI-style
`/v1/completions` to any vLLM deployment — a self-hosted `vllm-rwkv` instance
or a hosted endpoint (e.g. `https://api.rwkv.rs`) are the same backend, differing
only in `base_url`. To self-host, start the independent `vllm-rwkv` HTTP
service from its own repository and uv environment:

```bash
cd /path/to/vllm-rwkv
# Start the repository's RWKV7 server launcher here.
```

[configs/eval/rwkv7_g1j_1.5b_remote_race_scoreboard_example.toml](../configs/eval/rwkv7_g1j_1.5b_remote_race_scoreboard_example.toml)
is a validated example preset. Run it directly with:

```bash
uv run --no-sync python -m lm_eval run -C configs/eval/rwkv7_g1j_1.5b_remote_race_scoreboard_example.toml
```

The `rwkv7-remote` backend itself accepts any RWKV7 served model name. For another
checkpoint size or revision, copy the preset and change `model_name`,
`max_length`, `benchmarks`, and `output_dir`. The loader derives lm-eval's
`model_args` and result metadata from this public schema, removing duplicated
model and profile fields. `rwkv_profile.wkv_mode` remains a declared server-side
fact and must be checked against the vllm-rwkv startup log. The TOML owns the
exact model version, endpoint, benchmark selectors, served endpoint capacity
(`max_num_seqs`), RWKV profile, and result location. The `benchmarks` array
keeps its configured order as the stable tie-break, but benchmarks execute one
at a time in ascending pending-rollout order (see
`lm_eval/api/benchmark_scheduler.py`). These ordinary names resolve directly as
native lm-eval-harness task/group/tag names via `TaskManager`, with no adapter
indirection. Each benchmark YAML owns its prompt, answer extraction, metrics,
stop conditions, and `max_gen_toks`. See the
[RWKV evaluation guide](../RWKV_EVALUATION_GUIDE.md) for the complete server and
evaluation workflow. The deprecated `temp/` launch scripts are not used.

## Config Schema

The repository preset follows the same strict configuration approach as the
Helicopter LightEval runner: `schema_version = 1`, an allowlist of public fields,
required-field checks, duplicate benchmark rejection, and derived internal
arguments. Unknown fields fail before model or dataset initialization.

| Versioned RWKV field | Type | Default | Description |
|----------------------|------|---------|-------------|
| `schema_version` | int | required | Must be `1` |
| `backend` | string | required | `rwkv7-remote` |
| `model_name` | string | required | Model id sent to the inference service as the literal API `model=` value (top level for a single-model file, per `[[models]]` entry for a campaign); normally the short serving alias, and may carry the full release stem for older configs |
| `weight_revision` | string | unset | Full served weight filename stem, including the release date and `ctx<N>` (e.g. `rwkv7-g1j-1.5b-20260831-ctx16384`); recorded in run metadata and used by scoreboard publication to derive `ctx_len` without putting it in the API-facing `model_name` |
| `base_url` | string | required | HTTP(S) endpoint, always `/v1/completions` |
| `benchmarks` | list | required | Unique public benchmark names; configured order is the stable tie-break for the runtime ascending-pending-rollout order |
| `benchmark_config` | string | `lm_eval/config/benchmarks.toml` | Explicit top-level field-to-selector mapping; relative paths are resolved from the run config |
| `output_dir` | string | required | Result directory (top level for a single-model file, per `[[models]]` entry for a campaign) |
| `max_length` | int | required | Positive service context limit |
| `max_gen_toks` | int | `8192` | Remote RWKV choice-generation output budget; sent to the backend unchanged regardless of prompt length -- it is a completion length, not a total-context budget, and is never shrunk by subtracting prompt tokens |
| `batch_size` | int | `1` | lm-eval request batch size |
| `max_num_seqs` | int | required | Served endpoint's own `--max-num-seqs`; the evaluator's request concurrency is derived from it |
| `device` | string | `"cpu"` | Evaluation-side device |
| `rwkv_profile` | table | required | Complete RWKV execution profile |
| `rwkv_profile.prompt_template` | string | required | Official RWKV template mode |
| `rwkv_profile.generation_prompt` | string | deprecated | Ignored with a warning; each RWKV task selects its mode |
| `rwkv_profile.wkv_mode` | string | required | Recorded server WKV mode |
| `models` | array of tables | unset | The model ladder of a campaign; see [One file per benchmark set](#one-file-per-benchmark-set) |
| `model_overrides` | table | `{}` | Versioned fields forced onto every `[[models]]` entry last; the launch-time overlay hook |
| `apply_chat_template` | bool | `true` | Render the service's official template |
| `fewshot_as_multiturn` | bool | chat setting | Render few-shot examples as turns |
| `log_samples` | bool | `true` | Save inputs and model outputs |
| `seed` | list[int] | `[0,1234,1234,1234]` | Four lm-eval random seeds |
| `limit` | number | unset | Smoke-only sample limit |
| `limit_scope` | string | `task` | Apply `limit` per leaf task (`task`, native lm-eval behavior) or once per configured benchmark selector (`benchmark`) |
| `skip_benchmarks` | list[string] | `[]` | Configured selectors to omit before task loading; useful when a canonical dataset or runtime is unavailable |
| `use_cache` | string | unset | Response cache path |
| `weight_revision` | string | unset | Exact weight identity, including `ctx<number>` for Scoreboard; separate from the evaluation's `max_length` |

### Benchmark fields

Benchmark domains are configured only in `lm_eval/config/benchmarks.toml`. Each
top-level key is one supported field and its array contains opaque benchmark
selectors:

```toml
knowledge = ["wmdp"]
reasoning = ["race", "drop"]
coding = ["cruxeval", "humaneval_infilling"]
```

The evaluator looks up the requested selector before `TaskManager` expands it
into tasks, groups, or tags. It never derives a field from the selector text or
from an expanded leaf name. A missing selector, unknown field, or selector
listed under two fields fails configuration validation.

The selected field is carried by `BenchmarkSpec`, saved in each aggregate
artifact as `benchmark.field`, and read back from that artifact during
Scoreboard publication. `[publication.tasks.<selector>.field]` is no longer a
source of truth; when present for compatibility, it may only agree with the
saved field and cannot override it.

`[publication].is_preview` is the explicit Scoreboard model-identity flag. It
defaults to `false`; set it to `true` only for a preview release. Producer
clients must not infer this flag from the data-version label.

`limit_scope = "benchmark"` gives every configured selector one shared
document budget. When a selector expands to more leaf tasks than the budget,
the first and last leaf tasks are included with the remaining leaves spread
evenly across the selector. Selected leaves receive deterministic leading
document indices in round-robin order. This is an explicitly sampled score,
not full benchmark coverage; omitting the field preserves lm-eval's native
per-leaf `limit` behavior.

`skip_benchmarks` is an operational exclusion list. Its entries must also
appear in `benchmarks`; skipped selectors are absent from results and
publication rather than being scored with substituted or truncated inputs.
Skip decisions are excluded from checkpoint fingerprints so an interrupted run
can resume without discarding completed rollouts.

### Remote RWKV evaluation protocol

RWKV mode selection follows `RWKVLightevalTask.cot_mode` and
`RWKVLightevalTaskConfig` in `lighteval-rwkv`. The local prototypes are
`TaskFactory._build_task`, the evaluator's per-task override loop, and
`RWKV7Remote.__init__` / `get_model_info`.

The executable task contract is:

- The RWKV factory applies existing `metadata.rwkv_no_think` task variants before
  constructing the native Task. Native `multiple_choice` / likelihood tasks
  select `no_think`; actual `Instance.request_type` remains `loglikelihood`.
- `lm_eval/config/benchmark_modes.toml` is the authoritative RWKV mode inventory.
  Its `no_think`, `fake_think`, and `open_think` sections list benchmark selectors;
  `default` handles unlisted tasks. Native TaskManager expands group/tag selectors,
  and an exact leaf entry overrides its enclosing selectors. Generative tasks
  retain direct-answer `no_think` unless explicitly listed otherwise. Task YAML
  and run metadata no longer select the mode. This does not turn generation into likelihood.
- Dataset versions, fields, splits, targets, choices, scoring functions,
  aggregation and task generation budgets retain their existing definitions.
  No benchmark is added by this change. Native non-RWKV task loading is unchanged.
- The evaluator activates the task profile before request construction and again
  before dispatch. This controls chat rendering, stops, fixed sampling presets,
  final-answer filtering, avg@k eligibility and checkpoint model fingerprints.
  Direct-answer NoCoT generation stays `generate_until` with its native stops
  and budgets, and is excluded from automatic avg@k planning. Only CoT/FakeCoT
  generation remains eligible for RWKV automatic repeated sampling. Native
  multiple-choice and likelihood tasks stay deterministic sequence scoring and
  are excluded because they have no independent samples to average. NoCoT does
  not imply loglikelihood for open-ended QA.
- Each leaf and benchmark artifact records its actual mode, protocol, sampling
  parameters and template. The complete run records `config.task_execution`;
  one global sampling block cannot describe a mixed run. Mixed-mode benchmark
  leaves cannot be published as one mode.
- Publication derives `cot_mode` from saved execution artifacts, regardless of an
  obsolete `publication.cot_mode`. The field/category still comes exclusively
  from `BenchmarkSpec.field` / saved `benchmark.field`, and model identity still
  comes from the saved model and exact weight revision. Historical artifacts
  retain their existing protocol checks; their modes are never guessed from a
  newly edited run config.
- Legacy versioned presets may contain `rwkv_profile.generation_prompt`, but it
  is ignored with a warning. Maintained presets omit it and `publication.cot_mode`.
  Explicit low-level adapter construction remains compatible for historical
  replay; normal versioned runs use task modes. Local coding owns its CoT mode.

Acceptance requires the hermetic HTTP/mixed-mode/publication/resume regressions
and ten-question 1.5B + 2.9B smoke artifacts with actual LOGPROBS / GENERATIVE
logs before production publication. Missing endpoint manifests or credentials do
not count as successful live validation.


RWKV multiple-choice tasks automatically select NoCoT and use
native `multiple_choice` requests and real sequence loglikelihoods from
`/v1/completions`. The matching `tokenizer_name` is mandatory and requests
send Token IDs. Native task prompts take precedence over the global chat
template for these tasks; a missing standalone `\nAnswer: ` is appended in
the RWKV adapter. Candidate strings, targets and metrics are preserved.
NoCoT likelihood tasks do not use generative avg@k planning. Missing prompt
logprobs abort evaluation, and the chat and transformers-only backends reject
this workload. Protocol `rwkv-loglikelihood-v1` separates checkpoints and
response caches from historical generative NoCoT scores. Old scores require
real likelihood requests; generated letters cannot be converted to logprobs.
See the current [NoCoT review](../RWKV_EVALUATION_GUIDE.md#nocot-似然修复审查2026-11-01).

The following generation-specific description applies to generation tasks and
the historical CoT / FakeCoT choice path:

`rwkv7-remote` renders the official local tokenizer's template and sends token
IDs to `/v1/completions`. Set `tokenizer_name` to the same official tokenizer
used by the server and keep `apply_chat_template = true`. The native
open-thinking prefix ends in `<think` intentionally. A completed `</think>`
is required before final-answer grading; raw responses remain in the artifacts.
Without a tokenizer, this backend only passes raw text through and cannot apply
a chat template.

RWKV sampling presets are sent as top-level vLLM fields. The custom server must
support `penalty_decay` and its required sampler. Each document's repeats use
`seed[2] + repeat_ord`, including after resume, and share one concurrency budget.
The full input is preserved. `max_gen_toks` defaults to 8192 for choice rollouts;
set it explicitly when comparing models with different reasoning lengths.
A prompt that exhausts the context window aborts rather than receiving a score
for an empty completion. Multi-leaf benchmarks share one avg@k sizing budget.
The evaluator materializes one request window at a time. That window is bounded
by the endpoint concurrency pool and by `STREAM_WINDOW_MAX_PROMPT_BYTES`
(2 MiB of native prompt text in `RWKV7Remote`), so a long-context leaf cannot
retain the entire pool's prompt strings and token IDs before dispatch.
With local sample output enabled, each completed window is appended to the
task's JSONL file before its sample objects are released. Task finalization
reuses that file without loading or rewriting earlier windows. A write failure
stops evaluation before publication; checkpointed responses remain available
for replay. Distributed execution, request caching, and Hugging Face sample
uploads retain the existing sample-saving path. Metric inputs remain resident
until task aggregation because task-defined aggregators may need every value.
This reuses `EvaluationTracker.save_results_samples` serialization and
`save_task_artifacts` metadata writing; the default overwrite behavior remains
unchanged for callers outside windowed evaluation. The hermetic long-prompt
regression uses 256 documents with 32 KiB questions and checks that the windowed
Python allocation peak stays below half the native path's peak. This measures
producer allocations, not server memory or production process RSS.

The protocol identity is `rwkv-generative-v2`. Use a separate output/checkpoint
directory for runs made before this version. Old raw answers need a versioned
replay before their scores can be published using the new protocol.

### Local RWKV coding

`lm-eval coding -C configs/eval/rwkv7_coding_local.toml` uses the same validated
RWKV model/profile fields and adds a strict `[coding]` table. This entry point
reads a local JSONL problem suite, generates over HTTP, and executes Python tests
in local Docker containers. It does not resolve `rwkv_coding_local` through the
ordinary task registry. Model metadata and samples are always saved locally;
automatic publication and generic run-only task/checkpoint overrides are rejected.

| Coding field | Default | Meaning |
| --- | --- | --- |
| `coding.problems` | required | Local JSONL or JSONL.gz file; relative to the config |
| `coding.num_samples` | `8` | Fixed independent generations per problem |
| `coding.pass_k` | `[1, num_samples]` | Values of k, each no greater than num_samples |
| `coding.workers` | `2` | Local grading workers; separate from inference concurrency |
| `coding.test_timeout` | `5.0` | Per-candidate execution timeout in seconds |
| `coding.memory_mb` | `4096` | Memory limit per container |
| `coding.docker_image` | `python:3.11-slim` | Existing local image, resolved to an immutable ID |

The exact `weight_revision` must include `ctx<number>`. Raw completions require
the official local `tokenizer_name`; the chat backend delegates the native RWKV
template to its server. Both require `apply_chat_template = true`. `output_dir`
also resolves relative to the config. Use `--phase generate` without Docker or
`--phase score --samples ... --output-dir ...` for offline replay into an empty
directory. Only an incomplete generation may use `--resume`; scoring identities
and full problem/repeat coverage are verified before accepting saved answers.


## One file per benchmark set

A full campaign evaluates the same benchmark set on every RWKV7 size, so the
per-size TOMLs differ only in `model_name`, `weight_revision`, `tokenizer_name`,
`output_dir`, and the endpoint capacity each size is served with. Declare those
once as `[[models]]` entries and keep everything else in the shared top-level
table:

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
wkv_mode = "fp32io16"

[[models]]
model_name = "rwkv7-g1k-1.5b"
weight_revision = "rwkv7-g1k-1.5b-20260930-ctx25600"
tokenizer_name = "/home/link/weight/rwkv-g1k-tokenizer/1.5b"
max_num_seqs = 64
output_dir = "results/rwkv7-g1k-1.5b-...-full"

[[models]]
model_name = "rwkv7-g1k-2.9b"
weight_revision = "rwkv7-g1k-2.9b-20260930-ctx25600"
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
directory or publish under the same name. Give every published entry its own
`weight_revision` (or keep the full release stem in `output_dir`, which
publication also reads); it carries the exact release date and `ctx_len`, which
are part of the scoreboard's model identity even when the API alias is stable
across releases.

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
include = "base.toml"
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
template and declared server WKV mode. Tasks select the fixed RWKV generation
profile; individual temperature, top_p/top_k and penalty overrides remain unsupported. The Python CLI runner only loads this configuration,
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

### Native mode audit (2026-11-13)

Mode selection must follow the selected benchmark variant rather than assuming
all generation is CoT. The upstream [DROP task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/drop/default.yaml)
uses direct answers and EM/F1; [XQuAD](https://github.com/google-deepmind/xquad)
uses SQuAD-style extractive answers, and its [upstream task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/xquad/xquad_common_yaml)
stops at newline with greedy generation. Therefore DROP/XQuAD remain NoCoT
generation with the existing 64-token budget, extraction and scorer. No budget
is enlarged to accommodate reasoning. Other native direct-answer generation
variants inherit this same policy.

[CRUXEval](https://github.com/facebookresearch/cruxeval) exposes both direct
and chain-of-thought variants. Only `cruxeval_input_cot` and
`cruxeval_output_cot` explicitly select `open_think` in the central inventory; the direct variants
retain `no_think`. Their native prompts, repeats, budgets and metrics are
preserved. A mixed-mode CRUXEval selector cannot be uploaded as one mode;
it needs an explicit, separately reviewed publication split.

The publisher refuses a legacy upload API that requires only `benchmark_name`:
that interface cannot distinguish same-name benchmarks with different fields or
sample contracts. It must accept the registered `benchmark_id`; no name-only
retry is sent. This keeps historical rows intact and prevents a correct local
`reading_comprehension` field from becoming an older `reasoning` row remotely.

Live validation on 2026-11-13 used the current gateway model registry and pool
capacities (1.5B: 128; 2.9B: 512 + 2048), running both models independently.
RACE, DROP and XQuAD each populated ten-question artifacts per model; RACE had
40 real likelihood candidate requests, and each QA benchmark had ten generation
requests. Resumption restored their completed checkpoints. CRUXEval input/output
CoT variants each populated ten questions with ten native repeats for both models;
the original 512/1024-token limits caused substantial truncation, so these are
protocol validation artifacts, not production acceptance for those variants.
See `results/task-native-mode-smoke-20261113/validation_summary.json` locally.

The provided test Scoreboard at `127.0.0.1:7862` exposed a name-only upload
schema. Its first compatibility retry associated RACE with historical benchmark
id 1 (`reasoning`) instead of newly registered id 29
(`reading_comprehension`); the publisher detected and expired only that new
ambiguous score. The fallback has been removed, and a subsequent real API check
confirmed rejection on HTTP 422 before writing another score. Test-dashboard
publication acceptance remains blocked until the external deployment supports
registered benchmark ids. No Scoreboard source or unrelated scores were changed.
