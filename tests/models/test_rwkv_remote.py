from unittest.mock import patch

import pytest

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
