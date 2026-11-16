# Native Coding port validation

Validated locally on 2026-11-16, against target branch `feature/adapt-new-api` at
`b5e555ad42d6665eaeaec878d9609b8234443fdc`. The upstream protocol revision is
`d9b7287e90c6bb9ddca1432ebcce88025895e211`; see `catalog.json` and `NOTICE.md`.

## Automated checks

- `tests/test_evalscope_coding.py`: **55 passed**, including seven real Docker
  known-pass/known-fail fixture tests. These cover all native task registrations,
  prompt/data conversion, full Plus/private tests, code assembly, read-only SQL,
  native evaluator integration, repeated sampling, multi-turn checkpoint resume,
  unsafe opt-in, and model-failure versus infrastructure-error classification.
- Related regression selection: **189 passed, 1 pre-existing failure**:

  ```bash
  .venv/bin/python -m pytest \
    tests/test_evalscope_coding.py \
    tests/test_evaluator_checkpoint.py tests/test_checkpoint.py \
    tests/test_rwkv_coding.py tests/test_evaluate_config_benchmark.py \
    tests/api/test_benchmark_avg_at_k.py tests/models/test_rwkv_remote_chat.py -q
  ```

  The failure is
  `test_coding_config_rejects_invalid_evaluation_contract[generation_prompt = "open_think"-generation_prompt = "no_think"]`.
  It expects `no_think` to be rejected, although the target implementation accepts
  that mode. The same failure was reproduced in a detached, unmodified worktree at
  the target SHA above. This PR does not modify that test or its implementation.
- Ruff lint/format checks pass for the new task directory and tests.
  The three existing Ruff diagnostics in `lm_eval/evaluator.py` are identical on
  the target SHA and this branch.
- Native CLI discovery lists all **95 selectors: 87 leaf tasks and 8 groups**.
- A clean wheel builds successfully. Its task package contains all 95 YAML files,
  catalog, license, Python graders and Docker build resources; it contains no
  bytecode or downloaded SciCode HDF5 data. Wheel metadata has no EvalScope
  dependency, and EvalScope was not installed in the validation environment.

## Additional real data/runtime controls

The native loader successfully read the following ModelScope datasets:

| Dataset | Rows checked in loaded split |
| --- | ---: |
| HumanEval+ | 164, expanded executable `test` field present |
| MBPP+ | 378, expanded `test` and `test_imports` present |
| SciCode | 65, subproblem/test fields present |
| MT-Bench | 80 total, 10 Coding conversations with references |
| LiveCodeBench release_v1 | 400, first record's public/private tests decoded |
| BIRD Mini-Dev | 500, SQL/database/difficulty fields present |

An isolated local `volcengine/sandbox-fusion:server-20250609` container ran
known-pass and known-fail programs in all **18 MultiPL-E languages** (36 executions):
C++, TypeScript, Bash, C#, Go, Java, Lua, JavaScript, PHP, Perl, Racket, R, Rust,
Scala, Swift, Ruby, D and Julia. All controls produced the expected verdicts.
This exposed and verified the TypeScript/R/D wire-enum mappings. The temporary
container was stopped and removed after validation.

Docker fixture tests used an existing Python 3.11 image and a NumPy image selected
through `LMEVAL_CODING_TEST_NUMPY_IMAGE`; they are small correctness controls, not
full benchmark environments or measured model scores.

## Remaining production acceptance

No live RWKV requests or production score writes were made for this port.
The repository's live RWKV 1.5B/2.9B ten-question runs and test-database score/detail
readback remain unperformed. Full SciCode/DS-1000/BigCodeBench scientific images,
official SWE-bench repository-instance runs, and a real MT-Bench judge call were
not exercised here. Their integration contracts are covered by fixtures/source
inspection; these checks do not establish full benchmark or model acceptance.
