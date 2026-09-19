import json
from unittest.mock import patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.rwkv_choice import build_choice_prompt
from lm_eval.models.rwkv_remote_chat import RWKV7RemoteChat


def _build(**kwargs):
    kwargs.setdefault("model", "rwkv7-g1i-1.5b")
    kwargs.setdefault("num_concurrent", 1)
    return RWKV7RemoteChat(**kwargs)


def test_rwkv7_remote_chat_construction_defaults(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    assert model.base_url == RWKV7RemoteChat.DEFAULT_BASE_URL
    assert model.base_url.endswith("/v1/chat/completions")
    assert model.tokenizer_backend is None
    assert model.tokenizer is None


def test_rwkv7_remote_chat_rejects_bad_prompt_template(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    with pytest.raises(ValueError, match="rwkv_prompt_template"):
        _build(rwkv_prompt_template="nope")


def test_rwkv7_remote_chat_rejects_bad_generation_prompt(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    with pytest.raises(ValueError, match="rwkv_generation_prompt"):
        _build(rwkv_generation_prompt="nope")


def test_rwkv7_remote_chat_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("RWKV_API_KEY", raising=False)
    model = _build()

    with pytest.raises(ValueError, match="RWKV_API_KEY"):
        _ = model.api_key


def test_rwkv7_remote_chat_registered_in_model_mapping(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")

    assert get_model("rwkv7-remote-chat") is RWKV7RemoteChat


def test_rwkv7_remote_chat_get_model_info_reports_fixed_sampling_config(monkeypatch):
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


def test_rwkv7_remote_chat_tokenizer_name_has_no_tokenizer_dependency(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_prompt_template="bot", rwkv_generation_prompt="open_think")

    assert model.tokenizer_name == "rwkv7-g1i-1.5b:bot:open_think"


def test_rwkv7_remote_chat_create_payload_sends_chat_template_kwargs(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build(rwkv_prompt_template="bot", rwkv_generation_prompt="open_think")

    payload = model._create_payload(
        [{"role": "user", "content": "hi"}],
        generate=True,
        gen_kwargs={"until": [], "max_gen_toks": 32},
    )

    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["chat_template_kwargs"] == {
        "rwkv_prompt_template": "bot",
        "rwkv_generation_prompt": "open_think",
    }
    # Sampling knobs are folded into the server-side generation_config blob
    # (mirrors RWKV7Remote); they must not leak as top-level payload fields.
    for name in ("top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay"):
        assert name not in payload
    generation_config = json.loads(payload["generation_config"])
    assert generation_config["top_p"] == 0.76
    assert generation_config["penalty_decay"] == 0.988


def test_rwkv7_remote_chat_create_payload_passthrough_for_non_generate(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    payload = model._create_payload([{"role": "user", "content": "hi"}], generate=False)

    assert "chat_template_kwargs" not in payload


def test_rwkv7_remote_chat_parse_generations_reads_message_content():
    outputs = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            },
        ]
    }

    (generation,) = RWKV7RemoteChat.parse_generations(outputs)

    assert str(generation) == "hello"
    assert generation.finish_reason == "stop"
    assert generation.raw_response == outputs


def _mc_requests(context, choices, doc_id=0, repeats=1):
    return [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(context, choice),
            idx=idx,
            metadata=("some_task", doc_id, repeats),
        )
        for idx, choice in enumerate(choices)
    ]


def test_rwkv7_remote_chat_resolve_choice_wraps_prompt_as_chat_message(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    calls = []

    def fake_generate_until(instances):
        calls.append(instances[0])
        return ["A"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    prompt = build_choice_prompt("What is 1+1?", ["one", "two"])
    index = model._resolve_choice(prompt, ["A", "B"], ["one", "two"])

    assert index == 0
    sent = calls[0].args[0]
    # A bare string would blow up LocalChatCompletion's `_create_payload`
    # assertion, so `_resolve_choice` must hand generate_until a JsonChatStr
    # that unwraps into a proper [{"role", "content"}, ...] message list.
    assert isinstance(sent, JsonChatStr)
    messages = json.loads(sent.prompt)
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    assert "A. one" in messages[0]["content"]
    assert "B. two" in messages[0]["content"]


def test_rwkv7_remote_chat_avg_at_k_samples_k_generations_via_generate_until(monkeypatch):
    monkeypatch.setenv("RWKV_API_KEY", "test-key")
    model = _build()

    generated_answers = iter(["A", "B", "A", "A"])
    calls = []

    def fake_generate_until(instances):
        calls.append(instances[0])
        return [next(generated_answers)]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    requests = _mc_requests("What is 1+1?", ["one", "two"], doc_id=7, repeats=4)
    results = model.loglikelihood(requests)

    assert len(calls) == 4
    sample_encodings = model._avg_k_samples[7]
    assert len(sample_encodings) == 4
    assert [enc[0][1] for enc in sample_encodings] == [True, False, True, True]
    assert results == [(0.0, True), (-1e6, False)]
