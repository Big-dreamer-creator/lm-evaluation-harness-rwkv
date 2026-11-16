# EvalScope Coding protocols in native lm-eval

These are native `ConfigurableTask` implementations. Use `lm-eval run --tasks ...` with any supported generation model, including `rwkv7-remote` and `rwkv7-remote-chat`. They do **not** import, launch, or require EvalScope. The `evalscope_` prefix identifies the source protocol and avoids changing the existing harness tasks of the same benchmark names.

Source inventory: [ModelScope/EvalScope at d9b7287](https://github.com/modelscope/evalscope/tree/d9b7287e90c6bb9ddca1432ebcce88025895e211). All 16 non-agent benchmarks carrying the Coding tag are covered. MT-Bench includes only its Coding subset. Agent adapters (SWE-bench Agentic, DeepSWE, SWE-Atlas-QnA, TerminalBench) are outside this port.

## Installation and execution

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[api,coding]'
docker pull python:3.11-slim
docker build -t lm-eval-coding-numpy:1 lm_eval/tasks/evalscope_coding/docker/python-numpy

# Set the exact served model ID and a chat completion endpoint in your shell.
.venv/bin/python -m lm_eval run \
  --model rwkv7-remote-chat \
  --model_args "model=${SERVED_MODEL},base_url=${CHAT_COMPLETIONS_URL},rwkv_generation_prompt=fake_think,num_concurrent=8" \
  --tasks evalscope_humaneval,evalscope_humaneval_plus,evalscope_mbpp,evalscope_mbpp_plus \
  --apply_chat_template --confirm_run_unsafe_code \
  --log_samples --output_path outputs/coding
```

API credentials belong in the model backend's environment variables, never in the task YAML, metadata, or command-line arguments. `--limit 10` is useful for validation; remove it for complete benchmark scores. These task registrations do not enable automatic publication or start inference services.

## Task inventory

| Selector | Population / variants | Default grading |
| --- | --- | --- |
| `evalscope_humaneval` | 164 Python problems | 4s, full dataset tests, pass@1 |
| `evalscope_humaneval_plus` | 164 problems with expanded executable tests | 300s, pass@1 |
| `evalscope_mbpp` | MBPP full/test, fixed 3-shot prompt (0-shot also supported) | 20s, pass@1 |
| `evalscope_mbpp_plus` | Sanitized MBPP+ and expanded tests, 0-shot | 60s, pass@1 |
| `evalscope_multiple_humaneval` | All 18 language suites | Sandbox Fusion, 30s run timeout, pass@1 |
| `evalscope_multiple_mbpp` | All 18 language suites | Same multilingual protocol |
| `evalscope_bigcodebench` | Instruct, v0.1.4; `_complete` and `_instruct` selectable | 240s, official `TestCases`, pass@1 |
| `evalscope_bigcodebench_hard` | Hard Instruct, v0.1.4; both modes selectable | Same BigCodeBench grader |
| `evalscope_live_code_bench` | `release_latest`; all 28 source release/window selectors available | Original call/stdio grader, 6s per test, pass@1 |
| `evalscope_scicode` | Ordered subproblem conversations | 300s; main-problem and weighted subproblem pass rates |
| `evalscope_ds1000` | Pandas, Numpy, Matplotlib, Sklearn, Scipy, Pytorch, Tensorflow | 120s; `test_execution` and `test_string`, accuracy |
| `evalscope_bird_sql` | Simple, moderate, challenging | Read-only SQLite, 30s, execution accuracy |
| `evalscope_swe_bench_lite` | 300 oracle-context patch-generation cases | Official SWE-bench harness, 1800s, resolved accuracy |
| `evalscope_swe_bench_verified` | 500 oracle-context cases | Same SWE-bench harness |
| `evalscope_swe_bench_verified_mini` | 50 verified cases | Same SWE-bench harness |
| `evalscope_mt_bench_coding` | 10 two-turn Coding conversations | Reference-guided judge, two 1–10 ratings and their mean |

The umbrella group `evalscope_coding` expands to the 16 rows above. It intentionally has **no overall score**, because these benchmarks have different metrics and populations. Family groups weight constituent suites by sample count. Language selectors have suffixes such as `_cpp`, `_java`, `_ts`, `_jl`; DS-1000 uses `_pandas`, `_numpy`, etc. LiveCodeBench selectors have suffixes such as `_release_v6`, `_v5_v6`. The default group runs only `release_latest`, avoiding repeated overlapping releases. Pin a release and dataset revision for reproducibility.

## Data and protocol

`catalog.json` records the source revision and original dataset/subset/split/prompt declarations. Hugging Face supplies the original datasets where available; ModelScope's hub SDK supplies mirrors with the executable Plus test suites, SciCode data, all LiveCodeBench windows, BIRD Mini-Dev, and MT-Bench reference answers. The ModelScope SDK is a data dependency, not an EvalScope runtime.

Local dataset overrides use standard task YAML:

```yaml
include: /absolute/path/to/lm_eval/tasks/evalscope_coding/evalscope_humaneval_plus.yaml
dataset_path: json
dataset_kwargs:
  data_files:
    test: /absolute/path/to/humanevalplus.jsonl
```

HumanEval(+) retains the source prompt-plus-completion construction and `check(entry_point)`. MBPP+ executes `test_imports`, the basic `test_list`, **and the expanded `test` field**; an HF table without that field is not an equivalent substitute. MultiPL-E retains language-specific extraction, test suffixes and Java class assembly. BigCodeBench explicitly invokes `TestCases`, including failure/error checks, instead of accepting a program that merely defines tests. LiveCodeBench decodes both public and private test cases and executes its original grader inside Docker; its legacy pickle format cannot load executable globals.

Generation defaults are explicit: 4096 tokens, 24576 for LiveCodeBench, 1024 for MT-Bench; task `generation_kwargs` or CLI `--gen_kwargs` can override budgets for a separately identified run. No newline/function-definition stop is injected. MT-Bench's Coding temperature is 0.0. Configure sampling for the served model explicitly; generation defaults are not claims of score equivalence across model configurations.

Use `--apply_chat_template` to preserve system/user roles and server-side RWKV templates. It is required for SciCode and MT-Bench. Their subsequent turns include the actual earlier assistant answers, use native `generate_until`, and retain per-turn input, raw response and truncation evidence. Native checkpoint-capable backends save each turn separately and resume missing turns. These multi-turn tasks currently require one model process (`world_size=1`); conversations and HTTP calls can still run concurrently.

Leading `<think>...</think>` is removed before the benchmark's own extractor; unfinished reasoning and known truncated completions fail. RWKV open-think prefill is detected from the selected template renderer. For saved outputs or another backend with an open prefix, set task metadata `thinking: open_think`; use `thinking: none` when reasoning extraction must be disabled. Tags appearing inside generated code remain literal code.

## Repeated sampling and evidence

Set `repeats` and `metadata.pass_k` together in a task override, for example:

```yaml
include: /absolute/path/to/lm_eval/tasks/evalscope_coding/evalscope_humaneval.yaml
repeats: 8
metadata:
  pass_k: [1, 2, 8]
```

All repeats are retained. Per-problem pass@k uses the unbiased estimator `1 - C(n-c,k)/C(n,k)` and rejects `k > n`. These tasks own repeated-sampling scoring and opt out of RWKV's automatic avg@k wrapper. SciCode weights subproblems by their actual counts; it does not average problem-level subproblem percentages.

`--log_samples` records native inputs/responses/targets plus `scoring_details`: extracted code, execution verdicts, immutable Docker image IDs, subproblem outcomes or judge replies. A transport, missing-image, failed reference query, compiler timeout or missing judge verdict raises an error; it does not fabricate a failed model answer or a complete aggregate. Generated code that completes grading and fails tests scores zero. Completed responses remain available through the existing checkpoint and sample mechanisms.

## Additional runtimes

- **MultiPL-E:** run the Sandbox Fusion image `volcengine/sandbox-fusion:server-20250609` separately, then set `LMEVAL_CODING_SANDBOX_URL` to its HTTP origin, or `metadata.sandbox_url`. Compiler policy is left to the pinned runtime; the run limit remains 30s. `metadata.sandbox_wait_timeout` (default 3600s) changes transport waiting, not the execution limit.
- **BigCodeBench:** pre-pull `bigcodebench/bigcodebench-evaluate:latest`. Prefer an immutable image tag/digest via `metadata.docker_image` in reproducible runs. `metadata.calibrate` defaults to true; `metadata.bcb_split` is `instruct` or `complete` (the named selectors already set it).
- **SciCode:** obtain the benchmark's `test_data.h5`, copy it into `docker/scicode/`, then `docker build -t lm-eval-scicode:1 lm_eval/tasks/evalscope_coding/docker/scicode`. The Dockerfile bundles the original comparison/HDF5 helpers. `metadata.provide_background: true` selects the source background prompt variant. Do not commit the downloaded data.
- **DS-1000:** `docker build -t lm-eval-ds1000:1 lm_eval/tasks/evalscope_coding/docker/ds1000`. This pins the source scientific library versions. Tensorflow requires linux/amd64, as in the upstream image.
- **BIRD-SQL:** download the [official Mini-Dev databases](https://github.com/bird-bench/mini_dev) and set `metadata.db_root` to the extracted `dev_databases` directory. No SQL modifications, extensions or attached databases are allowed. `metadata.sample_rows` defaults to the official value of zero.
- **SWE-bench:** install `.[coding-swe]`; this pins the official `swebench==4.1.0` grader. Docker and the repository-specific images are required. `metadata.swe_namespace` defaults to `swebench`; null permits local builds. Oracle context comes from `princeton-nlp/SWE-bench_oracle` and is joined by `instance_id`. The official harness's patch-rejection and 1800s test-timeout markers count as unsuccessful model predictions; other incomplete runs raise infrastructure errors. Logs stay under the official `logs/run_evaluation/` tree, with a hash of the problem and prediction in the run ID.
- **MT-Bench:** the default judge is `gpt-4` using `OPENAI_API_KEY`. `metadata.judge` accepts `model`, `base_url`, `api_key_env` and `timeout`. A different judge must be identified in the run and must not be presented as the default GPT-4 protocol.

Standard Python images must exist before generation; the Python tasks do not pull them implicitly. SWE-bench uses its official harness's repository-image pull/build behavior. Standard Python grading reuses the existing restricted Docker judge (network disabled, read-only root, process/memory/time limits). `metadata.docker_image` and `metadata.memory_mb` configure each Python runtime.

## Validation and production acceptance

The port includes offline catalog/loader/prompt tests, native evaluator integration, repeated sampling and checkpoint tests, and Docker known-pass/known-fail controls. Run:

```bash
.venv/bin/python -m pytest tests/test_evalscope_coding.py -q
```

Docker tests skip when the required local images are absent. `LMEVAL_CODING_TEST_NUMPY_IMAGE` can select an already-installed NumPy image for the LiveCodeBench control.

Full benchmark scores, live RWKV 1.5B/2.9B acceptance runs and Scoreboard publication are separate production gates. Registering tasks or passing fixture tests does not certify a model score, authorize a production campaign or satisfy the repository's live publication audit.
