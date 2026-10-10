import json
import random
from pathlib import Path
from urllib.error import HTTPError

import pytest

from lm_eval import evaluator
from lm_eval.api.benchmark import BenchmarkSpec
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.loggers.scoreboard import (
    MAX_SAMPLES_PER_OUTCOME,
    PublicationConfig,
    ScoreboardError,
    ScoreboardPublisher,
    _detail,
    _messages,
    _rwkv_model,
    _weight_ctx_len,
    build_score,
)
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


_TEST_DATA = [
    {"question": "What is 2+2?", "answer": "lol"},
    {"question": "What is 3+3?", "answer": "not-lol"},
    {"question": "What is 4+4?", "answer": "not-lol"},
    {"question": "What is 5+5?", "answer": "not-lol"},
]

_LEAF_TEMPLATE = """\
task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data_path}
output_type: generate_until
doc_to_text: "{{{{question}}}}"
doc_to_target: answer
test_split: test
num_fewshot: 0
generation_kwargs:
  until: []
  max_gen_toks: 32
metric_list:
  - metric: {metric}
    aggregation: mean
    higher_is_better: true
metadata:
  field: test-field
  benchmark_name: {task_name}
"""

_MODEL_NAME = "rwkv7-g1j-1.5b"
# Scoreboard records the *weight's* context window as part of the model
# identity, so run output lives under a directory named after the weight
# release (see `_weight_ctx_len`). Keep this fixture dir conventions-complete.
_RUN_DIR_NAME = "rwkv7-g1j-1.5b-20260101-ctx4096-remote-unit"
_MODEL_ARGS = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "penalty_decay": 0.0,
    "seed": 42,
}


def _write_data_and_leaves(fixtures_dir: Path, leaves: list[tuple[str, str]]) -> Path:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    data_path = fixtures_dir / "test_data.json"
    data_path.write_text(json.dumps(_TEST_DATA), encoding="utf-8")
    for task_name, metric in leaves:
        (fixtures_dir / f"{task_name}.yaml").write_text(
            _LEAF_TEMPLATE.format(
                task_name=task_name, data_path=str(data_path), metric=metric
            ),
            encoding="utf-8",
        )
    return data_path


def _publish_config(tasks: dict[str, dict] | None = None) -> PublicationConfig:
    return PublicationConfig(
        base_url="https://scoreboard.example",
        token_env="SCOREBOARD_PUBLICATION_TOKEN",  # noqa: S106 - env var name, not a secret
        cot_mode="NoCoT",
        tasks=tasks or {},
        timeout=60,
        control_timeout=30,
        retries=0,
        retry_delay=0,
        rerun_reason=None,
        dry_run=True,
    )


def _callback_context() -> dict:
    return {
        "config": {
            "model": _MODEL_NAME,
            "model_args": dict(_MODEL_ARGS),
            "max_length": 4096,
        },
        "lm_eval_version": "test",
    }


def _run_benchmark(
    fixtures_dir: Path,
    group_yaml: str,
    benchmark_name: str,
    strategy: str,
    tmp_path: Path,
    seed: int = 0,
) -> dict:
    """Runs a real `evaluate()` pass and returns the aggregated `results.json`
    document for the benchmark, exactly as `ScoreboardTaskCallback` reads it
    in production.
    """
    (fixtures_dir / "group.yaml").write_text(group_yaml, encoding="utf-8")
    task_manager = TaskManager(include_path=str(fixtures_dir), include_defaults=False)
    task_dict = task_manager.load([benchmark_name])

    random.seed(seed)
    lm = DummyLM()
    evaluation_tracker = EvaluationTracker(output_path=str(tmp_path / _RUN_DIR_NAME))

    captured: dict = {}

    def callback(task_name, artifact_paths):
        captured["task_name"] = task_name
        captured["artifact_paths"] = artifact_paths

    evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=0,
        evaluation_tracker=evaluation_tracker,
        task_callback=callback,
        task_callback_context=_callback_context(),
        benchmark_map={
            benchmark_name: BenchmarkSpec(
                name=benchmark_name,
                strategy=strategy,
                leaf_task_names=(benchmark_name,) if strategy == "standalone" else None,
            )
        },
    )

    return json.loads(Path(captured["artifact_paths"]["results"]).read_text("utf-8"))


def test_standalone_benchmark_produces_score_with_real_metric(tmp_path):
    fixtures_dir = tmp_path / "fixtures_standalone"
    _write_data_and_leaves(fixtures_dir, [("solo_leaf", "exact_match")])
    group_yaml = "group: unused\ntask:\n  - solo_leaf\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "solo_leaf", "standalone", tmp_path
    )

    config = _publish_config({"solo_leaf": {"field": "reasoning", "avg_k": 1}})
    benchmark_info, score = build_score(results_doc, config)

    assert benchmark_info == {
        "name": "solo_leaf",
        "field": "reasoning",
        "num_samples": 4,
        "avg_k": 1,
    }
    assert score["model"] == {
        "arch_version": "rwkv7",
        "data_version": "G1j",
        "param_size": "1.5b",
        "ctx_len": 4096,
    }
    assert score["cot_mode"] == "NoCoT"
    assert score["score"] == 0.25
    assert len(score["passed_details"]) == 1
    assert len(score["wrong_details"]) == 3
    assert score["passed_details"][0]["is_passed"] is True
    assert score["passed_details"][0]["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "lol"},
    ]


def test_group_fallback_merges_leaves_into_one_score(tmp_path):
    fixtures_dir = tmp_path / "fixtures_shared"
    _write_data_and_leaves(
        fixtures_dir, [("shared_leaf_a", "exact_match"), ("shared_leaf_b", "exact_match")]
    )
    group_yaml = "group: bench_shared\ntask:\n  - shared_leaf_a\n  - shared_leaf_b\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "bench_shared", "group_fallback", tmp_path
    )

    config = _publish_config({"bench_shared": {"field": "reasoning", "avg_k": 1}})
    benchmark_info, score = build_score(results_doc, config)

    assert benchmark_info["num_samples"] == 8
    assert len(score["passed_details"]) == 2
    assert len(score["wrong_details"]) == 6


def test_field_missing_from_policy_raises(tmp_path):
    fixtures_dir = tmp_path / "fixtures_missing_field"
    _write_data_and_leaves(fixtures_dir, [("no_field_leaf", "exact_match")])
    group_yaml = "group: unused\ntask:\n  - no_field_leaf\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "no_field_leaf", "standalone", tmp_path
    )

    config = _publish_config({"no_field_leaf": {"avg_k": 1}})
    with pytest.raises(ScoreboardError, match="field"):
        build_score(results_doc, config)


def test_avg_k_missing_from_policy_raises(tmp_path):
    fixtures_dir = tmp_path / "fixtures_missing_avg_k"
    _write_data_and_leaves(fixtures_dir, [("no_avg_k_leaf", "exact_match")])
    group_yaml = "group: unused\ntask:\n  - no_avg_k_leaf\n"
    results_doc = _run_benchmark(
        fixtures_dir, group_yaml, "no_avg_k_leaf", "standalone", tmp_path
    )

    config = _publish_config({"no_avg_k_leaf": {"field": "reasoning"}})
    with pytest.raises(ScoreboardError, match="avg_k"):
        build_score(results_doc, config)


def test_publication_config_validates_cot_mode():
    with pytest.raises(ScoreboardError, match="cot_mode"):
        PublicationConfig.from_mapping(
            {"base_url": "https://scoreboard.example", "cot_mode": "not-a-mode"}
        )


def test_rwkv_model_parses_name_with_explicit_ctx_len():
    assert _rwkv_model("rwkv7-g1j-1.5b", 4096) == {
        "arch_version": "rwkv7",
        "data_version": "G1j",
        "param_size": "1.5b",
        "ctx_len": 4096,
    }
    assert _rwkv_model("rwkv7-g1i-1.5b-20260805-ctx16384", 16384) == {
        "arch_version": "rwkv7",
        "data_version": "G1i",
        "param_size": "1.5b",
        "ctx_len": 16384,
    }


def test_weight_ctx_len_reads_the_weight_revision_not_the_eval_budget():
    """ctx_len identifies the weight, so it never comes from max_length.

    Evaluating one weight at a shorter harness `max_length` must not fork it
    into a second model on the board, so the revision is read from
    ``model_name`` when present and otherwise from the run directory, which is
    named after the served weight.
    """
    assert (
        _weight_ctx_len(
            "rwkv7-g1k-13.3b-20260930-ctx25600",
            "results/rwkv7-g1k-13.3b-20260930-ctx25600-remote-x/task_artifacts/a/results.json",
        )
        == 25600
    )
    # model_name without a revision: fall back to the artifact path.
    assert (
        _weight_ctx_len(
            "rwkv7-g1k-13.3b",
            "results/rwkv7-g1k-13.3b-20260930-ctx25600-remote-x/task_artifacts/a/results.json",
        )
        == 25600
    )
    # A revision in the name wins over a directory that disagrees.
    assert (
        _weight_ctx_len(
            "rwkv7-g1k-13.3b-20260930-ctx25600", "results/stale-ctx8192-remote/x/a.json"
        )
        == 25600
    )


def test_weight_ctx_len_refuses_unidentifiable_weight():
    with pytest.raises(ScoreboardError, match="ctx<number>"):
        _weight_ctx_len("rwkv7-g1j-1.5b", "output/a/results.json")


def test_rwkv_model_rejects_unknown_shape():
    with pytest.raises(ScoreboardError):
        _rwkv_model("not-a-known-model-name", 4096)


def test_messages_parses_json_chat_str_on_disk_shape():
    arguments = {
        "gen_args_0": {
            "arg_0": ['[{"role": "user", "content": "hi"}]'],
            "arg_1": {"until": []},
        }
    }
    assert _messages(arguments) == [{"role": "user", "content": "hi"}]


def test_messages_falls_back_to_plain_string_prompt():
    arguments = {"gen_args_0": {"arg_0": "plain prompt text", "arg_1": {}}}
    assert _messages(arguments) == [{"role": "user", "content": "plain prompt text"}]


def _generative_row(
    *,
    answer: str,
    target: str,
    evidence_text: str | None,
    filtered_answer: str | None = None,
) -> dict:
    # `filtered_resps` is one list of filtered strings (one per repeat) per
    # document request -- [[answer]] mirrors the real on-disk shape for a
    # single-repeat generate_until task (see samples_<task>_*.jsonl), not a
    # flat [answer]. `filtered_answer` defaults to `answer` so existing
    # callers that don't care about filtering behave as before.
    return {
        "arguments": {"gen_args_0": {"arg_0": "the prompt", "arg_1": {}}},
        "filtered_resps": [[filtered_answer if filtered_answer is not None else answer]],
        "resps": [[answer]],
        "target": target,
        "response_evidence": (
            [{"text": evidence_text, "finish_reason": "stop"}]
            if evidence_text is not None
            else []
        ),
        "exact_match": 1.0 if answer == target else 0.0,
    }


def test_detail_buckets_passed_wrong_and_failed():
    passed_detail, passed_outcome, _ = _detail(
        _generative_row(answer="4", target="4", evidence_text="4"), "exact_match"
    )
    assert passed_outcome == "passed"
    assert passed_detail["is_passed"] is True
    assert passed_detail["messages"] == [
        {"role": "user", "content": "the prompt"},
        {"role": "assistant", "content": "4"},
    ]

    wrong_detail, wrong_outcome, _ = _detail(
        _generative_row(answer="5", target="4", evidence_text="5"), "exact_match"
    )
    assert wrong_outcome == "wrong"
    assert wrong_detail["is_passed"] is False
    assert wrong_detail["messages"][-1] == {"role": "assistant", "content": "5"}

    failed_detail, failed_outcome, _ = _detail(
        _generative_row(answer="", target="4", evidence_text=""), "exact_match"
    )
    assert failed_outcome == "failed"
    assert failed_detail["is_passed"] is False
    assert failed_detail["messages"][-1] == {"role": "assistant", "content": ""}


def test_detail_answer_uses_filtered_resps_not_raw_completion():
    """A `filter_list` (e.g. `yes_no`/`albatross_mcq`) exists to reduce a
    long chain-of-thought completion down to the clean value actually used
    for scoring -- `answer` must show that distilled value, not the raw
    completion the filter exists to cut through (the bug: Scoreboard was
    showing the entire multi-KB reasoning dump as "the answer" for every
    `generate_until` task that uses a filter).
    """
    row = _generative_row(
        answer="C",
        target="C",
        evidence_text="<think>lots of reasoning...</think>C. the full option text",
        filtered_answer="C",
    )
    detail, _outcome, _truncated = _detail(row, "exact_match")
    assert detail["answer"] == "C"
    assert detail["messages"][-1]["content"].startswith("<think>")


def test_detail_answer_falls_back_to_dash_when_filter_could_not_extract():
    """The `yes_no`/`albatross_mcq` filters' fallback sentinel
    ("[invalid]") means no answer could be extracted at all -- that must
    show as "-", not the literal sentinel string nor the raw completion.
    """
    row = _generative_row(
        answer="whatever",
        target="C",
        evidence_text="rambling with no clear final answer",
        filtered_answer="[invalid]",
    )
    detail, _outcome, _truncated = _detail(row, "exact_match")
    assert detail["answer"] == "-"


def test_detail_tuple_valued_metric_with_usable_answer_passes():
    """Tasks like discrim_eval score each doc with an aggregation-input
    tuple (`(demographics, bias_name, chose_yes)`), not a binary 0/1 --
    there's no "wrong answer" concept there (any yes/no is legitimate), so
    a doc with a real extracted answer must be "passed", not blanket
    "failed" just because the metric value isn't a plain scalar.
    """
    row = _generative_row(
        answer="yes",
        target="yes",
        evidence_text="<think>reasoning</think>yes",
        filtered_answer="yes",
    )
    row["exact_match"] = [["black", "female", 30.0, 1], "black_bias", 1]
    detail, outcome, _truncated = _detail(row, "exact_match")
    assert outcome == "passed"
    assert detail["is_passed"] is True
    assert detail["answer"] == "yes"


def test_detail_tuple_valued_metric_with_unparseable_answer_fails():
    """Same tuple-valued metric shape, but the `yes_no` filter's fallback
    sentinel means no answer could be extracted at all -- that is a real
    failure, unlike every other doc that did get a usable answer.
    """
    row = _generative_row(
        answer="rambling with no verdict",
        target="yes",
        evidence_text="rambling with no verdict",
        filtered_answer="[invalid]",
    )
    row["exact_match"] = [["black", "female", 30.0, 1], "black_bias", None]
    detail, outcome, _truncated = _detail(row, "exact_match")
    assert outcome == "failed"
    assert detail["is_passed"] is False
    assert detail["answer"] == "-"


def test_detail_answer_falls_back_to_raw_completion_without_filtered_resps():
    """A task with no `filter_list` at all (just the implicit `take_first`
    pass-through) has no meaningfully different filtered value -- the raw
    completion is already the genuine answer there.
    """
    row = _generative_row(answer="4", target="4", evidence_text="4")
    del row["filtered_resps"]
    detail, _outcome, _truncated = _detail(row, "exact_match")
    assert detail["answer"] == "4"


def test_samples_cap_is_shared_across_leaves_of_one_benchmark(tmp_path):
    """A benchmark's per-outcome cap must apply across ALL of its leaves
    combined, not be re-applied independently per leaf. Otherwise a
    benchmark with N sub-tasks uploads up to N * MAX_SAMPLES_PER_OUTCOME
    "passed" samples instead of MAX_SAMPLES_PER_OUTCOME total.
    """
    from lm_eval.loggers.scoreboard import _details

    num_leaves = 7
    docs_per_leaf = MAX_SAMPLES_PER_OUTCOME
    shared_uploaded_counts: dict[str, int] = dict.fromkeys(
        ("passed", "wrong", "failed"), 0
    )
    total_uploaded = 0
    for leaf_index in range(num_leaves):
        rows = [
            _generative_row(answer="4", target="4", evidence_text="4")
            for _ in range(docs_per_leaf)
        ]
        samples_path = tmp_path / f"leaf_{leaf_index}.jsonl"
        samples_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        buckets, _total, _truncation_rate = _details(
            samples_path, "exact_match", shared_uploaded_counts
        )
        total_uploaded += len(buckets["passed"])

    assert total_uploaded == MAX_SAMPLES_PER_OUTCOME
    assert shared_uploaded_counts["passed"] == MAX_SAMPLES_PER_OUTCOME


def _http_error(code: str, score_id: int = 4) -> HTTPError:
    import io

    body = json.dumps({"detail": {"code": code, "score_id": score_id}}).encode()
    error = HTTPError(
        url="https://scoreboard.example/api/upload",
        code=409,
        msg="Conflict",
        hdrs=None,
        fp=io.BytesIO(body),
    )
    return error


def _publisher(monkeypatch, rerun_reason: str | None) -> ScoreboardPublisher:
    import dataclasses

    monkeypatch.setenv("SCOREBOARD_PUBLICATION_TOKEN", "test-token")
    config = dataclasses.replace(
        _publish_config(), rerun_reason=rerun_reason, dry_run=False
    )
    return ScoreboardPublisher(config)


def test_upload_duplicate_score_without_rerun_reason_is_idempotent_replay(
    monkeypatch,
):
    """Without --rerun-reason, a duplicate_score conflict is a harmless
    no-op replay of the existing score -- no expire, no retry.
    """
    publisher = _publisher(monkeypatch, rerun_reason=None)
    monkeypatch.setattr(
        "lm_eval.loggers.scoreboard.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error("duplicate_score")),
    )

    receipt = publisher._upload({"benchmark_id": 1})

    assert receipt["replayed"] is True
    assert receipt["score_id"] == 4


def test_upload_duplicate_score_with_rerun_reason_expires_and_reuploads(
    monkeypatch,
):
    """With --rerun-reason set, a duplicate_score conflict must NOT be
    silently replayed: the stale score is expired and the new payload
    (e.g. containing a bugfix that doesn't change the numeric score) is
    genuinely re-uploaded, not swallowed.
    """
    publisher = _publisher(monkeypatch, rerun_reason="fixed messages bug")
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if request.full_url.endswith("/api/upload") and len(calls) == 1:
            raise _http_error("duplicate_score")
        if request.full_url.endswith("/api/expire"):
            return _fake_response({"success": True})
        return _fake_response({"success": True, "score_id": 99, "message": "ok"})

    monkeypatch.setattr("lm_eval.loggers.scoreboard.urlopen", fake_urlopen)

    receipt = publisher._upload({"benchmark_id": 1})

    assert calls == [
        "https://scoreboard.example/api/upload",
        "https://scoreboard.example/api/expire",
        "https://scoreboard.example/api/upload",
    ]
    assert receipt.get("replayed") is not True
    assert receipt["score_id"] == 99


def _fake_response(payload: dict):
    import io
    from contextlib import contextmanager

    @contextmanager
    def cm():
        yield io.BytesIO(json.dumps(payload).encode())

    return cm()
