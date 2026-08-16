import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.models.api_models import create_image_prompt
from lm_eval.models.openai_completions import (
    RWKV7HTTP,
    LocalCompletionsAPI,
    _VLLMRWKVTokenizer,
)


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


def test_rwkv7_http_template_and_sampling_profiles(monkeypatch):
    class FakeRemoteTokenizer:
        eos_token = "<eos>"
        bos_token = "<bos>"
        eos_token_id = 0
        bos_token_id = 1
        tokenizer_info = {
            "chat_template": (
                "{{ rwkv_prompt_template }}|{{ rwkv_generation_prompt }}|"
                "{{ messages[0]['content'] }}|{{ add_generation_prompt }}"
            )
        }

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, text):
            return [1]

        def batch_decode(self, tokens):
            return ["" for _ in tokens]

    monkeypatch.setattr(
        "lm_eval.models.openai_completions._VLLMRWKVTokenizer",
        FakeRemoteTokenizer,
    )
    model = RWKV7HTTP(
        model=RWKV7HTTP.DEFAULT_MODEL,
        base_url="http://test-url.com/v1/completions",
        num_concurrent=1,
    )

    assert model.apply_chat_template(
        [{"role": "user", "content": "hello"}]
    ) == "assistant|open_think|hello|True"
    assert model._create_payload(
        ["prompt"], generate=True, gen_kwargs={"max_gen_toks": 4, "temperature": 0.1}
    ) == {
        "prompt": ["prompt"],
        "model": RWKV7HTTP.DEFAULT_MODEL,
        "max_tokens": 4,
        "temperature": 0.96,
        "stop": ["\nUser:"],
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "penalty_decay": 0.988,
    }

    model.rwkv_generation_prompt = "fake_think"
    model._chat_template_source = (
        "{{ '<think></think' if rwkv_generation_prompt == 'fake_think' "
        "else '<think' }}"
    )
    assert model.apply_chat_template(
        [{"role": "user", "content": "hello"}]
    ) == "<think></think>\n"
    fake_payload = model._create_payload(
        ["prompt"], generate=True, gen_kwargs={"max_gen_toks": 4}
    )
    assert {
        key: fake_payload[key]
        for key in ("temperature", "top_p", "top_k")
    } == {"temperature": 1.0, "top_p": 0.28, "top_k": 32}

    logprobs_payload = model._create_payload(
        [[1, 2]], generate=False, gen_kwargs=None
    )
    assert logprobs_payload["temperature"] == 1
    assert logprobs_payload["top_k"] == 1
    assert logprobs_payload["echo"] is True
    assert model.prefix_token_id == 0
    assert model.tokenized_requests is True

    generations = model.parse_generations(
        {
            "choices": [
                {"index": 0, "text": "answer", "finish_reason": "length"}
            ]
        }
    )
    assert generations == ["answer"]
    assert generations[0].finish_reason == "length"


def test_vllm_rwkv_tokenizer_uses_native_routes_and_strips_bos(monkeypatch):
    responses = []

    def request(method, url, **kwargs):
        response = MagicMock()
        if url.endswith("/tokenizer_info"):
            response.json.return_value = {
                "tokenizer_class": "RWKVTokenizer",
                "chat_template": "{{ messages }}",
            }
        elif url.endswith("/tokenize"):
            response.json.return_value = {"tokens": [0, 10, 11]}
        else:
            response.json.return_value = {"prompt": "decoded"}
        responses.append((method, url, kwargs.get("json")))
        return response

    session = MagicMock()
    session.request.side_effect = request
    monkeypatch.setattr("lm_eval.utils.requests.Session", lambda: session)
    tokenizer = _VLLMRWKVTokenizer(
        "http://test-url.com/v1/completions",
        RWKV7HTTP.DEFAULT_MODEL,
    )

    assert tokenizer.encode("hello") == [10, 11]
    assert tokenizer.decode([0, 10, 11]) == "decoded"
    assert responses[-2][1:] == (
        "http://test-url.com/tokenize",
        {
            "model": RWKV7HTTP.DEFAULT_MODEL,
            "prompt": "hello",
            "add_special_tokens": False,
        },
    )
    assert responses[-1][1:] == (
        "http://test-url.com/detokenize",
        {"model": RWKV7HTTP.DEFAULT_MODEL, "tokens": [0, 10, 11]},
    )


def test_rwkv7_http_likelihood_request_adds_one_bos(monkeypatch):
    class BoundaryTokenizer:
        tokenizer_info = {"chat_template": "{{ messages }}"}

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, text):
            return {
                "hello": [10],
                "hello world": [12, 13],
            }[text]

        def batch_decode(self, tokens):
            return ["" for _ in tokens]

    monkeypatch.setattr(
        "lm_eval.models.openai_completions._VLLMRWKVTokenizer",
        BoundaryTokenizer,
    )
    model = RWKV7HTTP(
        num_concurrent=1,
    )
    context, continuation = model._encode_pair("hello", " world")
    inputs, context_lengths, _ = model.batch_loglikelihood_requests(
        [[(("hello", " world"), context, continuation)]]
    )

    assert context == []
    assert continuation == [12, 13]
    assert inputs == [[0, 12, 13]]
    assert context_lengths == [1]

    empty_inputs, empty_context_lengths, _ = model.batch_loglikelihood_requests(
        [[(("", "hello"), [0], [10])]]
    )
    assert empty_inputs == [[0, 10]]
    assert empty_context_lengths == [1]


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
