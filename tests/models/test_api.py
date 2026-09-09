import asyncio
import hashlib
import json
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.api_models import create_image_prompt
from lm_eval.models.openai_completions import RWKV7HTTP, LocalCompletionsAPI


@pytest.fixture
def api():
    return LocalCompletionsAPI(
        base_url="http://test-url.com", tokenizer_backend=None, model="gpt-3.5-turbo"
    )


@pytest.fixture
def api_tokenized():
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        model="EleutherAI/pythia-1b",
        tokenizer_backend="huggingface",
    )


@pytest.fixture
def api_batch_ssl_tokenized():
    return LocalCompletionsAPI(
        base_url="https://test-url.com",
        model="EleutherAI/pythia-1b",
        verify_certificate=False,
        num_concurrent=2,
        tokenizer_backend="huggingface",
    )


def test_create_payload_generate(api):
    messages = ["Generate a story"]
    gen_kwargs = {
        "max_tokens": 100,
        "temperature": 0.7,
        "until": ["The End"],
        "do_sample": True,
        "seed": 1234,
    }
    payload = api._create_payload(messages, generate=True, gen_kwargs=gen_kwargs)

    assert payload == {
        "prompt": ["Generate a story"],
        "model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["The End"],
        "seed": 1234,
    }


def test_create_payload_loglikelihood(api):
    messages = ["The capital of France is"]
    payload = api._create_payload(messages, generate=False, gen_kwargs=None)

    assert payload == {
        "model": "gpt-3.5-turbo",
        "prompt": ["The capital of France is"],
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
        "temperature": 0,
        "seed": 1234,
    }


def test_local_completions_preserves_finish_reason(api):
    generations = api.parse_generations(
        {
            "choices": [
                {"index": 0, "text": "complete", "finish_reason": "stop"},
                {"index": 1, "text": "cut off", "finish_reason": "length"},
            ]
        }
    )

    assert generations == ["complete", "cut off"]
    assert generations[0].finish_reason == "stop"
    assert generations[1].finish_reason == "length"

    restored = pickle.loads(pickle.dumps(generations))
    assert restored == generations
    assert restored[0].finish_reason == "stop"
    assert restored[1].finish_reason == "length"


def _build_rwkv7_http(monkeypatch, profile=None, chat_template_source="{{ messages }}", **kwargs):
    profile = profile or {
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 40,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.2,
        "penalty_decay": 0.99,
    }

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 0
        bos_token_id = 1
        chat_template = chat_template_source

    class FakeGenerationConfig:
        def to_dict(self):
            return dict(profile)

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kw: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "transformers.GenerationConfig.from_pretrained",
        lambda *args, **kw: FakeGenerationConfig(),
    )
    kwargs.setdefault("model", "rwkv7-g1h-test")
    kwargs.setdefault("base_url", "http://test-url.com/v1/completions")
    kwargs.setdefault("num_concurrent", 1)
    return RWKV7HTTP(**kwargs), profile


def _mc_instance(doc_id, idx, context, continuation):
    return Instance(
        request_type="loglikelihood",
        doc={},
        arguments=(context, continuation),
        idx=idx,
        metadata=("task", doc_id, 1),
    )


def test_rwkv7_http_construction_defaults(monkeypatch):
    model, profile = _build_rwkv7_http(monkeypatch)

    assert model.model == "rwkv7-g1h-test"
    assert model.rwkv_prompt_template == "assistant"
    assert model.rwkv_generation_prompt == "fake_think"
    assert model._batch_size == 1
    assert model.tokenized_requests is False
    assert model._generation_profile == profile

    expected_sha = hashlib.sha256("{{ messages }}".encode("utf-8")).hexdigest()
    assert model.tokenizer_name == f"rwkv7-g1h-test:{expected_sha}:assistant:fake_think"


def test_rwkv7_http_rejects_tokenized_requests(monkeypatch):
    with pytest.raises(ValueError, match="tokenized_requests=False"):
        RWKV7HTTP(model="rwkv7-g1h-test", tokenized_requests=True)


def test_rwkv7_http_apply_chat_template(monkeypatch):
    template = (
        "{{ rwkv_prompt_template }}|{{ rwkv_generation_prompt }}|"
        "{{ messages[0]['content'] }}|{{ add_generation_prompt }}"
    )
    model, _ = _build_rwkv7_http(monkeypatch, chat_template_source=template)

    assert (
        model.apply_chat_template([{"role": "user", "content": "hello"}])
        == "assistant|fake_think|hello|True"
    )


def test_rwkv7_http_apply_chat_template_appends_fake_think_close(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch, chat_template_source="{{ '<think></think' }}")

    assert (
        model.apply_chat_template([{"role": "user", "content": "hello"}])
        == "<think></think>\n"
    )
    assert (
        model.apply_chat_template(
            [{"role": "assistant", "content": "hi"}], add_generation_prompt=False
        )
        == "<think></think"
    )


def test_rwkv7_http_create_payload_generate(monkeypatch):
    model, profile = _build_rwkv7_http(monkeypatch)

    payload = model._create_payload(
        ["prompt"],
        generate=True,
        gen_kwargs={"max_gen_toks": 4, "temperature": 0.1, "top_p": 0.99},
    )

    assert payload["prompt"] == ["prompt"]
    assert payload["model"] == model.model
    assert payload["max_tokens"] == 4
    assert payload["temperature"] == profile["temperature"]
    assert payload["stop"] == ["\nUser:"]
    assert payload["generation_config"] == json.dumps(
        profile, separators=(",", ":"), sort_keys=True
    )
    for key in ("top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay"):
        assert key not in payload


def test_rwkv7_http_create_payload_generate_preserves_custom_stop(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch)

    payload = model._create_payload(
        ["prompt"], generate=True, gen_kwargs={"max_gen_toks": 4, "until": ["STOP"]}
    )

    assert payload["stop"] == ["STOP"]


def test_rwkv7_http_create_payload_loglikelihood_not_implemented(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch)

    with pytest.raises(NotImplementedError):
        model._create_payload(["prompt"], generate=False, gen_kwargs=None)


def test_rwkv7_http_parse_generations(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch)

    generations = model.parse_generations(
        {"choices": [{"index": 0, "text": "answer", "finish_reason": "length"}]}
    )

    assert generations == ["answer"]
    assert generations[0].finish_reason == "length"
    assert model.prefix_token_id == 0


def test_rwkv7_http_loglikelihood_groups_by_doc_and_scores_argmax(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch)

    doc0 = ["Paris", "London", "Berlin"]
    doc1 = ["Cat", "Dog", "Fish"]
    requests = [
        _mc_instance(1, 2, "Context for doc1", doc1[2]),
        _mc_instance(0, 0, "Context for doc0", doc0[0]),
        _mc_instance(1, 0, "Context for doc1", doc1[0]),
        _mc_instance(0, 2, "Context for doc0", doc0[2]),
        _mc_instance(1, 1, "Context for doc1", doc1[1]),
        _mc_instance(0, 1, "Context for doc0", doc0[1]),
    ]

    def fake_generate_until(instances):
        prompt = instances[0].args[0]
        return ["The answer is B."] if "doc0" in prompt else ["\\boxed{A}"]

    monkeypatch.setattr(model, "generate_until", fake_generate_until)

    results = model.loglikelihood(requests)

    assert len(results) == len(requests)
    expected_true = {5, 2}
    for position, result in enumerate(results):
        if position in expected_true:
            assert result == (0.0, True)
        else:
            assert result == (-1e6, False)


@pytest.mark.parametrize(
    "generated, expected_index",
    [
        ("I think the answer is B, based on the clues.", 1),
        ("Therefore \\boxed{C} is correct.", 2),
        ("My final choice is (B)", 1),
        ("The correct one is C.", 2),
    ],
)
def test_rwkv7_http_parse_choice_index_regex_tiers(monkeypatch, generated, expected_index):
    model, _ = _build_rwkv7_http(monkeypatch)
    letters = ["A", "B", "C"]
    choices = ["Paris", "London", "Berlin"]

    assert model._parse_choice_index(generated, letters, choices) == expected_index


def test_rwkv7_http_parse_choice_index_substring_fallback(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch)
    letters = ["A", "B", "C"]
    choices = ["Paris", "London", "Berlin"]
    generated = "The city famed for the Eiffel Tower is Paris, and it fits nicely."

    assert model._parse_choice_index(generated, letters, choices) == 0


def test_rwkv7_http_parse_choice_index_default_and_warns(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch)
    warning_mock = MagicMock()
    monkeypatch.setattr(
        "lm_eval.models.openai_completions.eval_logger.warning", warning_mock
    )
    letters = ["A", "B", "C"]
    choices = ["Paris", "London", "Berlin"]
    generated = "I have no idea what to say here."

    assert model._parse_choice_index(generated, letters, choices) == 0
    warning_mock.assert_called_once()


def test_rwkv7_http_loglikelihood_skips_unconditional_context(monkeypatch):
    model, _ = _build_rwkv7_http(monkeypatch)
    warning_mock = MagicMock()
    monkeypatch.setattr(
        "lm_eval.models.openai_completions.eval_logger.warning", warning_mock
    )

    requests = [
        _mc_instance(0, 0, "", "unconditional"),
        _mc_instance(0, 1, "Real context", "Paris"),
        _mc_instance(0, 2, "Real context", "London"),
        _mc_instance(0, 3, "Real context", "Berlin"),
    ]

    generate_until_mock = MagicMock(return_value=["The answer is A."])
    monkeypatch.setattr(model, "generate_until", generate_until_mock)

    results = model.loglikelihood(requests)

    assert results[0] == (0.0, False)
    assert results[1] == (0.0, True)
    assert results[2] == (-1e6, False)
    assert results[3] == (-1e6, False)
    generate_until_mock.assert_called_once()
    warning_mock.assert_called_once()


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            ["Hello, how are"],
            True,
            {"max_gen_toks": 100, "temperature": 0.7, "until": ["hi"]},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "temperature": 0.7,
                "stop": ["hi"],
                "seed": 1234,
            },
        ),
        (
            ["Hello, how are", "you"],
            True,
            {},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 256,
                "temperature": 0,
                "stop": [],
                "seed": 1234,
            },
        ),
    ],
)
def test_model_generate_call_usage(
    api, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            [[1, 2, 3, 4, 5]],
            False,
            None,
            {
                "model": "EleutherAI/pythia-1b",
                "prompt": [[1, 2, 3, 4, 5]],
                "max_tokens": 1,
                "logprobs": 1,
                "echo": True,
                "seed": 1234,
                "temperature": 0,
            },
        ),
    ],
)
def test_model_tokenized_call_usage(
    api_tokenized, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api_tokenized.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(
            "LocalChatCompletion",
            id="local-chat-completions",
        ),
        pytest.param(
            "OpenAIChatCompletion",
            id="openai-chat-completions",
        ),
    ],
)
def test_chat_template_payload_does_not_add_top_level_type(model_cls):
    from lm_eval.models import openai_completions

    model = getattr(openai_completions, model_cls)(
        base_url="http://test-url.com",
        model="test-model",
    )
    chat = [{"role": "user", "content": "Reply with one word: hello"}]

    messages = model.create_message((model.apply_chat_template(chat),))
    payload = model._create_payload(messages, generate=True, gen_kwargs={})

    assert payload["messages"] == chat
    assert "type" not in payload["messages"][0]


@pytest.mark.parametrize("include_legacy_type", [False, True])
def test_create_image_prompt_uses_content_parts_without_top_level_type(
    include_legacy_type,
):
    class DummyImage:
        def save(self, buf, format):
            buf.write(b"image-bytes")

    chat = [{"role": "user", "content": "Describe this image"}]
    if include_legacy_type:
        chat[0]["type"] = "text"

    messages = create_image_prompt([DummyImage()], json.loads(json.dumps(chat)))

    assert "type" not in messages[-1]
    assert messages[-1]["content"][0]["type"] == "image_url"
    assert messages[-1]["content"][1] == {
        "type": "text",
        "text": "Describe this image",
    }


class DummyAsyncContextManager:
    def __init__(self, result):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.parametrize(
    "expected_inputs, expected_ctxlens, expected_cache_keys",
    [
        (
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            [3, 3, 3, 3],
            ["cache_key1", "cache_key2", "cache_key3", "cache_key4"],
        ),
    ],
)
def test_get_batched_requests_with_no_ssl(
    api_batch_ssl_tokenized, expected_inputs, expected_ctxlens, expected_cache_keys
):
    with (
        patch(
            "lm_eval.models.api_models.TCPConnector", autospec=True
        ) as mock_connector,
        patch(
            "lm_eval.models.api_models.ClientSession", autospec=True
        ) as mock_client_session,
        patch(
            "lm_eval.models.openai_completions.LocalCompletionsAPI.parse_logprobs",
            autospec=True,
        ) as mock_parse,
    ):
        mock_session_instance = AsyncMock()
        mock_post_response = AsyncMock()
        mock_post_response.status = 200
        mock_post_response.ok = True
        mock_post_response.json = AsyncMock(return_value={"mocked": "response"})
        mock_post_response.raise_for_status = lambda: None
        mock_session_instance.post = lambda *args, **kwargs: DummyAsyncContextManager(
            mock_post_response
        )
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        mock_parse.return_value = [(1.23, True), (4.56, False)]

        async def run():
            return await api_batch_ssl_tokenized.get_batched_requests(
                expected_inputs,
                expected_cache_keys,
                generate=False,
                ctxlens=expected_ctxlens,
            )

        result_batches = asyncio.run(run())

        mock_connector.assert_called_with(limit=2, ssl=False)
        assert result_batches


def test_local_completionsapi_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token="secure-token",
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == "secure-token"


def test_local_completionsapi_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None


def test_localchatcompletion_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token="secure-token",
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == "secure-token"


def test_localchatcompletion_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None
