from unittest.mock import patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model
from lm_eval.models.rwkv_remote import RWKV7Remote


def _build(**kwargs):
    kwargs.setdefault("model", "rwkv7-g1i-1.5b")
    kwargs.setdefault("num_concurrent", 1)
    return RWKV7Remote(**kwargs)


def test_rwkv7_remote_construction_defaults(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.base_url == RWKV7Remote.DEFAULT_BASE_URL
    assert model.tokenizer_backend is None
    assert model.tokenizer is None


def test_rwkv7_remote_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("RWKV_API_KEY", raising=False)
    model = _build()

    with pytest.raises(ValueError, match="RWKV_API_KEY"):
        _ = model.api_key


def test_rwkv7_remote_api_key_used_in_header(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "sk-test-fixture-key")
    model = _build()

    assert model.api_key == "sk-test-fixture-key"
    assert model.header == {"Authorization": "Bearer sk-test-fixture-key"}


def test_rwkv7_remote_registered_in_model_mapping(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    assert get_model("rwkv7-remote") is RWKV7Remote


def test_rwkv7_remote_tokenizer_enables_tokenized_requests(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        model = _build(tokenizer="some/repo")

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.args[0] == "some/repo"
    assert model.tokenizer_backend == "huggingface"
    assert model.tokenized_requests is True


def test_rwkv7_remote_without_tokenizer_keeps_defaults(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.tokenizer_backend is None
    assert model.tokenized_requests is False


def test_rwkv7_remote_without_tokenizer_reports_fixed_sampling_config(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.get_model_info() == {
        "sampling_config": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "do_sample": True,
            "stop": ["✿", "\nUser:", "\n### User"],
        }
    }


def test_rwkv7_remote_get_model_info_reports_sampling_config_with_tokenizer(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 0
        bos_token_id = 1
        chat_template = "{{ messages }}"

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()
    ):
        model = _build(tokenizer="some/repo")

    assert model.get_model_info() == {
        "sampling_config": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "do_sample": True,
            "stop": ["✿", "\nUser:", "\n### User"],
        }
    }


def test_rwkv7_remote_open_think_uses_its_own_fixed_sampling_config(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_generation_prompt="open_think")

    assert model.get_model_info() == {
        "sampling_config": {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
            "do_sample": True,
            "stop": ["✿", "\nUser:", "\n### User"],
        }
    }


def test_rwkv7_remote_parse_generations_reports_finish_reason():
    outputs = {
        "choices": [
            {"index": 0, "text": "hello", "finish_reason": "length"},
        ]
    }

    (generation,) = RWKV7Remote.parse_generations(outputs)

    assert str(generation) == "hello"
    assert generation.finish_reason == "length"
    assert generation.truncated is True


def _mc_requests(context, choices, doc_id=0, repeats=1):
    # Mirrors evaluator.py's `cloned_reqs = [req for req in reqs for _ in
    # range(req.repeats)]`: avg@k's k is read off the actual number of
    # cloned instances per choice, not off the `repeats` metadata value
    # alone, so tests must clone here too.
    return [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(context, choice),
            idx=idx,
            metadata=("some_task", doc_id, repeats),
        )
        for idx, choice in enumerate(choices)
        for _ in range(repeats)
    ]


def test_rwkv7_remote_avg_at_k_samples_k_generations_via_generate_until(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    calls = []

    def fake_generate_until(instances):
        calls.append(instances)
        return ["A", "B", "A", "A"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests("What is 1+1?", ["one", "two"], doc_id=7, repeats=4)
    results = model.loglikelihood(requests)

    # avg@k exercises real generation (not echo/logprobs), so the sampling
    # profile in _create_payload actually applies -- all k samples for a doc
    # are dispatched as one generate_until call (not k blocking calls) so
    # the harness's existing concurrency parallelizes the rollouts.
    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert "A. one" in calls[0][0].args[0]
    assert "B. two" in calls[0][0].args[0]

    # k=4 sample encodings recorded for the process_results avg@k patch,
    # one full per-choice (ll, is_greedy) list per independent generation.
    sample_encodings = model._avg_k_samples[7]
    assert len(sample_encodings) == 4
    assert [enc[0][1] for enc in sample_encodings] == [True, False, True, True]
    assert [enc[1][1] for enc in sample_encodings] == [False, True, False, False]
    assert model._avg_k_parsed[7] == [True, True, True, True]

    # The single (ll, is_greedy) list returned still encodes a majority-vote
    # winner (choice "A" won 3/4), so acc/acc_norm work without any patch.
    # `requests` has 4 cloned instances per choice, so `results` mirrors that
    # shape: 4 outcomes for choice "one" (the winner) then 4 for "two".
    assert results == [(0.0, True)] * 4 + [(-1e6, False)] * 4


def test_rwkv7_remote_avg_at_k_ignores_unparsed_rollouts_in_majority_vote(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    def fake_generate_until(instances):
        # Only one of four rollouts names a valid letter; the rest can't be
        # parsed into a choice at all.
        return ["gibberish", "B", "gibberish", "gibberish"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests("What is 1+1?", ["one", "two"], doc_id=9, repeats=4)
    results = model.loglikelihood(requests)

    # The majority vote is decided purely by the single valid rollout ("B"),
    # not diluted/overruled by unparseable ones silently defaulting to A.
    assert results == [(-1e6, False)] * 4 + [(0.0, True)] * 4
    assert model._avg_k_parsed[9] == [False, True, False, False]


def test_rwkv7_remote_avg_at_k_defaults_to_one_sample(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    monkeypatch.setattr(model, "generate_until", lambda instances: ["B"])

    requests = _mc_requests("ctx", ["one", "two"], doc_id=1, repeats=1)
    results = model.loglikelihood(requests)

    assert len(model._avg_k_samples[1]) == 1
    assert results == [(-1e6, False), (0.0, True)]


def test_rwkv7_remote_loglikelihood_attaches_real_generation_as_evidence(monkeypatch):
    """The Scoreboard-published assembled_prompt/raw_completion must come
    from the real generative rollout, not be reconstructed from the score
    (see RWKVLoglikelihoodResponse) -- k=1, the only mode currently used in
    production, has exactly one real generation per doc, so this checks it
    is threaded through unmodified."""
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    real_generation = RWKV7Remote.parse_generations(
        {
            "choices": [
                {"index": 0, "text": "The answer is B.", "finish_reason": "stop"}
            ]
        }
    )[0]
    monkeypatch.setattr(model, "generate_until", lambda instances: [real_generation])

    requests = _mc_requests("ctx", ["one", "two"], doc_id=1, repeats=1)
    results = model.loglikelihood(requests)

    assert results == [(-1e6, False), (0.0, True)]
    for outcome in results:
        assert outcome.raw_response is real_generation.raw_response
        assert outcome.finish_reason == "stop"
        assert "Answer with only the letter" in outcome.prompt
        assert "A. one" in outcome.prompt
        assert "B. two" in outcome.prompt
