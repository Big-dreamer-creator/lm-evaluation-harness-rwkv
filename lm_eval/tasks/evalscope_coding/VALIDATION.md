# Native Coding port validation

Validated on 2026-11-16 against `feature/adapt-new-api` at
`b5e555ad42d6665eaeaec878d9609b8234443fdc`. The source protocol revision is
`d9b7287e90c6bb9ddca1432ebcce88025895e211`; see `catalog.json` and `NOTICE.md`.

## Automated checks

- **64/64 native Coding tests pass**, including seven real Docker positive/negative
  fixture tests, all registrations, prompt/data conversion, executable Plus/private
  tests, repeated sampling, native evaluator integration, multi-turn checkpoint
  resume, unsafe opt-in and infrastructure-error classification.
- The related regression selection passes **232 tests**, with **one pre-existing
  failure** and no skips:

  ```bash
  LMEVAL_CODING_TEST_NUMPY_IMAGE=python3.11-numpy:19be8de754b25de6d87a5ee3 \
  .venv/bin/python -m pytest \
    tests/test_evalscope_coding.py \
    tests/test_evaluator_checkpoint.py tests/test_checkpoint.py \
    tests/test_rwkv_coding.py tests/test_evaluate_config_benchmark.py \
    tests/api/test_benchmark_avg_at_k.py \
    tests/models/test_rwkv_remote_chat.py tests/models/test_rwkv_remote.py -q
  ```

  `test_coding_config_rejects_invalid_evaluation_contract[generation_prompt = "open_think"-generation_prompt = "no_think"]`
  expects `no_think` to be rejected although the target implementation accepts it.
  The same failure was reproduced in an unmodified detached worktree at the base
  SHA above. That test case and configuration validation are unchanged.
- Ruff lint/format checks pass for the task directory and its tests; `git diff
  --check` passes. Existing evaluator lint diagnostics match the base branch.
- Native CLI discovery exposes all **95 selectors: 87 leaves and 8 groups**.
- The wheel contains all 95 YAML files, catalog, license, Python graders and Docker
  resources. It excludes bytecode and the downloaded SciCode HDF5 file **even when
  that file is present in the source tree**. Its runtime files match the tested
  source. No EvalScope dependency is declared or installed in the test environment.

## Real model and data acceptance

Both official `rwkv7-g1l-{1.5b,2.9b}-20261031-ctx34304` checkpoints were exercised
through the native `Coding` task, `simple_evaluate`, sample logging and SQLite
checkpoint path. Each model completed ten source documents from each of the
**16 benchmark families**: **32 task/model runs, 320 documents and 440 generation
turns**. SciCode accounts for 60 subproblem turns per model; MT-Bench accounts for
20 turns per model. These are limited acceptance runs, not full benchmark scores.

The representative selectors were HumanEval(+), MBPP(+), C++ for both MultiPL-E
families, BigCodeBench Instruct and Hard Instruct, LiveCodeBench release_v1,
SciCode, DS-1000 Pandas, BIRD moderate, the three SWE-bench variants, and MT-Bench
Coding. BIRD's selected source has only eight simple documents, so the ten-document
smoke uses its moderate subset. All other subset registrations are covered by
loader/config tests; this does not claim ten real-model samples for every leaf.

Official PTH SHA-256 values were freshly checked before serving:

| Model | SHA-256 |
| --- | --- |
| 1.5B | `b0753acefe6c9ea0b4e2e9d60a3c2faa76d712a1d607db187301aae13dadff41` |
| 2.9B | `0e6015bc133ea1b8287ddecb1fa22ca73365f3b8c2097ecda4c845c99a9f844e` |

The isolated inference runtime was Alic-Li/rwkv_lightning_cuda v1.8.12,
commit `bd1eecf56a21dd9791a6312388de46b8bedda998`, with fp32 state and CMix sparsity
disabled. A validation-only transport adapter mapped native raw completions to
its `/v1/batch/completions` endpoint, retained the exact rendered prompts and raw
wire evidence, mapped EOS token 0 to its text-stop interface, and mapped MT-Bench
`temperature=0` to exact greedy `top_k=1`. A long LCB response used the runtime's
SSE interface to avoid its non-streaming idle disconnect; the task budget stayed
24576. This adapter is acceptance tooling, not a new supported harness backend.

All tasks used the official FakeCoT template and original task budgets: 4096,
24576 for LCB, and 1024 for MT-Bench. These are GENERATIVE tasks, not logprob
multiple-choice tasks. The 34304 context value records weight metadata, not a
measured server capacity. The native harness assigns seed 42; this temporary
runtime does not expose seed control, so these smokes **do not certify seeded
reproducibility**. Actual wire requests omit unsupported seed fields.

Early runs using an incorrect transport EOS mapping were rejected and retained
separately. Subsequent transport failures resumed only missing answers; SWE
serialization/grading retries reused saved predictions. No failed infrastructure
case was silently counted as an incorrect model answer.

## Real graders and resources

- Actual reference-solution and negative controls pass for HumanEval(+), MBPP(+),
  BigCodeBench and BigCodeBench Hard using their executable dataset tests.
- All **18 MultiPL-E languages** pass real Sandbox Fusion positive/negative
  controls (36 executions): C++, TypeScript, Bash, C#, Go, Java, Lua, JavaScript,
  PHP, Perl, Racket, R, Rust, Scala, Swift, Ruby, D and Julia.
- LiveCodeBench's actual first problem (`1873_A`) passes a known-correct solution
  and rejects a wrong solution, including public/private tests.
- SciCode uses the full official HDF5 data and original helpers. Real subproblem
  `77.1` passes the correct periodic-wrap implementation and rejects a wrong one.
- DS-1000's complete pinned scientific image passes reference/negative controls
  in **all seven libraries**, including PyTorch and TensorFlow.
- BIRD uses real Mini-Dev databases; reference SQL passes and a wrong query fails.
- SWE-bench uses `swebench==4.1.0` and actual repository-instance images. The gold
  patch for `astropy__astropy-12907` resolves its official tests; a malformed patch
  is classified as a model patch failure. All 28 images needed by the smoke
  populations were checked against official Docker Hub manifest digests.
- **MT-Bench live-judge acceptance was explicitly waived by the user.** Both
  models generated all ten two-turn Coding conversations; the unmodified OpenAI
  judge client sent reference-guided prompts to a local protocol fixture and
  parsed its JSON ratings. Those ratings are synthetic and establish API/schema
  behavior only. They are not GPT-4 judgments or model-quality scores.

## Score/detail round trip

An isolated current Scoreboard service backed by real PostgreSQL accepted
**32 task records and 320 full sample records**. An explicit acceptance exporter
mapped native results and scoring evidence to its producer-neutral `scoreboard-v1`
contract. Every score, task setting, prompt, extracted answer, reference, raw
completion and diagnostic field was read back and compared exactly. Publication
idempotency, outcome-category counts and filtered sample queries were checked;
the campaign finalized with no missing tasks.

This exercises native evidence against the current database contract. It does not
claim that the harness's separate legacy `/api/upload` publisher supports the new
Scoreboard API. Only the isolated acceptance database was written; no production
score was published or expired. MT fixture records are explicitly tagged and use
an undetermined outcome rather than fabricated correctness.

## Fixes found by acceptance

- Serializable task configuration no longer retains a bound scorer that copies
  live model/thread locks when computing checkpoint fingerprints.
- Both RWKV HTTP adapters supply the official template's `fake_cot_mode` flag.
  The chat protocol version changes to prevent reuse of old template checkpoints.
- MT-Bench's explicit greedy temperature survives RWKV's default sampling profile;
  other tasks retain their existing profile behavior.
- SWE-bench correctly closes Docker SDK clients and hashes timestamp-valued dataset
  fields using the native serializer.
- The DS-1000 image uses a compatible pip version for its pinned CPU wheels and
  limits TensorFlow/OpenMP pools to the grader's CPU allocation. The correct
  TensorFlow reference previously failed under the container's process limit.
- Recursive package exclusions keep downloaded HDF5 data out of the wheel.

## Evidence and limits

The full local receipt, source locks, immutable-image identities, data snapshots,
wire requests/responses, native checkpoints/results, official SWE logs and
PostgreSQL readbacks are retained at:
`/home/chase/eval-live/lm-eval-pr5-acceptance-20261116`.
The completion receipt is `completion.json`; the database receipt is
`publication/acceptance.json`. Validation scripts are preserved in `scripts/`.
Temporary services are released after the evidence audit; unrelated workloads
and historical evaluation campaigns are untouched.

Full population evaluation, deterministic-seed certification of the temporary
inference runtime, live MT-Bench judge quality, and production publication remain
outside this acceptance. The port remains native and has no EvalScope runtime.
