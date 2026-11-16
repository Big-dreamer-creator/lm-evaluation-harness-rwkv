# Source and license notices

This directory ports the non-agent Coding benchmark protocols from
[ModelScope/EvalScope](https://github.com/modelscope/evalscope/tree/d9b7287e90c6bb9ddca1432ebcce88025895e211),
revision `d9b7287e90c6bb9ddca1432ebcce88025895e211`.

Copyright (c) Alibaba, Inc. and its affiliates. Source-derived code is distributed under Apache License 2.0; a copy is included in `LICENSE-APACHE-2.0`. The repository's MIT license continues to apply to its existing code.

- `vendor/code_utils.py` and `vendor/multiple.py`: source code extraction and MultiPL-E assembly helpers. Imports use local modules/standard logging, unused v2 extraction functions are omitted, and formatting follows this repository.
- `vendor/lcb_testing.py`, `vendor/lcb_extract.py`, and LiveCodeBench prompt constants in `prompts.py`: source LiveCodeBench helpers (original copyright LiveCodeBench, 2024), adapted to remove EvalScope imports. The grader runs only inside a restricted container.
- `vendor/ds1000.py`: source extraction and test-program construction only; the local host execution path is omitted.
- `vendor/bird.py`: source read-only SQLite extraction/prompt/execution helpers. Its header attributes the BIRD Mini-Dev reference protocol (CC BY-SA 4.0); this port retains that attribution. Dataset and reference-project licenses remain applicable to their own content.
- `vendor/mtbench.py`, `vendor/scicode_prompts.py`, and MBPP constants in `prompts.py`: original prompt builders/text; framework adapter classes are replaced by native lm-eval tasks.
- `docker/`: source Dockerfiles and SciCode comparison/HDF5 helpers. SciCode helper headers retain their original Apache-2.0 attribution and source links.
- `catalog.json`: a compact inventory derived from upstream benchmark metadata, with source revision and original helper hashes.

Integration prototypes: existing `lm_eval/tasks/humaneval`, `lm_eval/tasks/mbpp`, `ConfigurableTask`, `Instance`, the native evaluator's checkpoint dispatch, and `lm_eval/rwkv_coding.py::DockerJudge`. EvalScope's adapter implementations supply benchmark-specific protocols; no EvalScope package is loaded at runtime.

Port-specific behavior is documented in `README.md`: native task names and aggregation, explicit generation budgets, restricted Python containers, safe private-test decoding, and fail-closed infrastructure errors. Existing tasks are not replaced. These implementation changes are not retrospective score corrections.
