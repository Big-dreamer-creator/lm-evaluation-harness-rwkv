"""Integration-style tests wiring the replica pool and adaptive retry into
``TemplateAPI.model_call``/``amodel_call``/``generate_until`` via a real
``LocalCompletionsAPI`` instance, with only the HTTP transport mocked.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.api_pool import ContextLengthExhausted, PoolError
from lm_eval.models.openai_completions import LocalCompletionsAPI


class DummyAsyncContextManager:
    def __init__(self, result):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.fixture
def pool_manifest_path(tmp_path):
    manifest = tmp_path / "pool.toml"
    manifest.write_text(
        """
schema_version = 1
[[replicas]]
base_url = "http://replica-a/v1/completions"
max_concurrency = 1
[[replicas]]
base_url = "http://replica-b/v1/completions"
max_concurrency = 1
"""
    )
    return str(manifest)


@pytest.fixture
def pooled_api(pool_manifest_path):
    return LocalCompletionsAPI(
        pool_manifest=pool_manifest_path,
        tokenizer_backend=None,
        model="test-model",
    )


def test_pooled_api_builds_pool_from_manifest(pooled_api):
    assert pooled_api.base_url == "http://replica-a/v1/completions"
    assert [r.base_url for r in pooled_api._pool.replicas] == [
        "http://replica-a/v1/completions",
        "http://replica-b/v1/completions",
    ]


def test_pooled_api_rejects_base_url_and_manifest_together(pool_manifest_path):
    with pytest.raises(PoolError):
        LocalCompletionsAPI(
            base_url="http://also-set",
            pool_manifest=pool_manifest_path,
            tokenizer_backend=None,
            model="test-model",
        )


def test_model_call_routes_to_selected_replica(pooled_api):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_response

        with (
            patch.object(pooled_api._pool, "pick_sync", return_value=1) as mock_pick,
            patch.object(pooled_api._pool, "release") as mock_release,
        ):
            result = pooled_api.model_call(["hello"], generate=False)

        assert result == {"result": "ok"}
        mock_pick.assert_called_once()
        mock_release.assert_called_once_with(1)
        called_url = mock_post.call_args.args[0]
        assert called_url == "http://replica-b/v1/completions"


def test_model_call_shrinks_max_tokens_on_context_overflow(pooled_api):
    overflow_response = MagicMock()
    overflow_response.ok = False
    overflow_response.status_code = 400
    overflow_response.text = (
        "prompt contains at least 2040 input tokens, please reduce."
    )

    success_response = MagicMock()
    success_response.ok = True
    success_response.json.return_value = {"result": "shrunk-ok"}

    with patch(
        "requests.post", side_effect=[overflow_response, success_response]
    ) as mock_post:
        result = pooled_api.model_call(
            ["hello"], generate=True, gen_kwargs={"max_tokens": 256, "until": []}
        )

    assert result == {"result": "shrunk-ok"}
    assert mock_post.call_count == 2
    first_payload = mock_post.call_args_list[0].kwargs["json"]
    second_payload = mock_post.call_args_list[1].kwargs["json"]
    assert first_payload["max_tokens"] == 256
    assert second_payload["max_tokens"] == pooled_api.max_length - 2040


def test_model_call_raises_context_length_exhausted_when_unshrinkable(pooled_api):
    overflow_response = MagicMock()
    overflow_response.ok = False
    overflow_response.status_code = 400
    overflow_response.text = (
        f"prompt contains at least {pooled_api.max_length + 100} input tokens."
    )

    with patch("requests.post", return_value=overflow_response):
        with pytest.raises(ContextLengthExhausted):
            pooled_api.model_call(
                ["hello"], generate=True, gen_kwargs={"max_tokens": 256, "until": []}
            )


def test_model_call_fails_over_to_other_replica_on_retryable_status(pooled_api):
    retryable_response = MagicMock()
    retryable_response.ok = False
    retryable_response.status_code = 503
    retryable_response.text = "service unavailable"

    success_response = MagicMock()
    success_response.ok = True
    success_response.json.return_value = {"result": "failover-ok"}

    with patch(
        "requests.post", side_effect=[retryable_response, success_response]
    ) as mock_post:
        result = pooled_api.model_call(["hello"], generate=False)

    assert result == {"result": "failover-ok"}
    urls_called = [call.args[0] for call in mock_post.call_args_list]
    assert urls_called == [
        "http://replica-a/v1/completions",
        "http://replica-b/v1/completions",
    ]


def test_generate_until_converts_exhaustion_to_placeholder(pooled_api):
    overflow_response = MagicMock()
    overflow_response.ok = False
    overflow_response.status_code = 400
    overflow_response.text = (
        f"prompt contains at least {pooled_api.max_length + 100} input tokens."
    )

    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=("hello", {"until": [], "max_tokens": 16}),
        idx=0,
    )

    with patch("requests.post", return_value=overflow_response):
        results = pooled_api.generate_until([instance])

    assert len(results) == 1
    assert str(results[0]) == ""
    assert results[0].finish_reason == "length"
    assert results[0].truncated is True


def _async_response(*, ok, status=200, text="", json_body=None):
    resp = AsyncMock()
    resp.ok = ok
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    resp.json = AsyncMock(return_value=json_body or {})
    return resp


def test_amodel_call_shrinks_on_same_replica_no_failover(pooled_api):
    overflow = _async_response(
        ok=False, status=400, text="prompt contains at least 2040 input tokens."
    )
    success = _async_response(
        ok=True, json_body={"choices": [{"index": 0, "text": "ok"}]}
    )

    post_calls = []

    def fake_post(url, json, headers):
        post_calls.append((url, dict(json)))
        return DummyAsyncContextManager(overflow if len(post_calls) == 1 else success)

    session = MagicMock()
    session.post = fake_post
    sem = asyncio.Semaphore(1)

    async def run():
        return await pooled_api.amodel_call(
            session,
            sem,
            ["hello"],
            generate=True,
            gen_kwargs={"max_tokens": 256, "until": []},
        )

    with patch.object(LocalCompletionsAPI, "parse_generations", return_value=["ok"]):
        answers = asyncio.run(run())

    assert answers == ["ok"]
    assert len(post_calls) == 2
    # Same replica retried in place (no failover) since this is a context-shrink retry.
    assert post_calls[0][0] == post_calls[1][0]
    assert post_calls[1][1]["max_tokens"] == pooled_api.max_length - 2040


def test_amodel_call_fails_over_using_sessions_map(pooled_api):
    retryable = _async_response(ok=False, status=503, text="unavailable")
    success = _async_response(
        ok=True, json_body={"choices": [{"index": 0, "text": "ok"}]}
    )

    calls_by_url = {}

    def make_post(tag, response):
        def _post(url, json, headers):
            calls_by_url.setdefault(url, []).append(tag)
            return DummyAsyncContextManager(response)

        return _post

    session_a = MagicMock()
    session_a.post = make_post("a", retryable)
    session_b = MagicMock()
    session_b.post = make_post("b", success)
    sessions = {0: session_a, 1: session_b}
    sem = asyncio.Semaphore(1)

    async def run():
        return await pooled_api.amodel_call(
            session_a,
            sem,
            ["hello"],
            generate=True,
            gen_kwargs={"max_tokens": 16, "until": []},
            sessions=sessions,
        )

    with patch.object(LocalCompletionsAPI, "parse_generations", return_value=["ok"]):
        answers = asyncio.run(run())

    assert answers == ["ok"]
    assert calls_by_url["http://replica-a/v1/completions"] == ["a"]
    assert calls_by_url["http://replica-b/v1/completions"] == ["b"]


def test_get_batched_requests_cancels_siblings_when_one_task_fails_hard(pooled_api):
    """When one batch hits a non-retryable, non-context-length error,
    `get_batched_requests` used to let `tqdm_asyncio.gather` raise and leave
    every other in-flight `amodel_call` task running -- orphaned against
    `ClientSession`s the enclosing `async with` was about to close, and, if
    one of them was blocked in `pool.pick_async()`, capable of leaking pool
    capacity (see ``test_api_pool_deadlock.py``). It must now cancel the
    other tasks itself before propagating the error.
    """
    pooled_api.max_retries = 1  # avoid tenacity's exponential backoff delay
    still_running = []
    cancelled = []

    async def fake_amodel_call(self, session, sem, messages, **kwargs):
        # `get_batched_requests` chunks `requests` with `batch_size=1`, so
        # each call receives a single-element list wrapping one of our
        # original request entries, e.g. `[["fail"]]` for the `["fail"]`
        # request.
        if messages == [["fail"]]:
            raise PoolError("HTTP 404: Unknown model")
        still_running.append(messages)
        try:
            await asyncio.sleep(5)
            return ["should not get here"]
        except asyncio.CancelledError:
            cancelled.append(messages)
            raise

    with patch.object(
        type(pooled_api), "amodel_call", side_effect=fake_amodel_call, autospec=True
    ):
        with pytest.raises(PoolError):
            asyncio.run(
                asyncio.wait_for(
                    pooled_api.get_batched_requests(
                        [["fail"], ["slow-1"], ["slow-2"]],
                        cache_keys=[None, None, None],
                        generate=True,
                        gen_kwargs={"until": []},
                    ),
                    timeout=3,
                )
            )

    assert sorted(still_running) == [[["slow-1"]], [["slow-2"]]]
    assert sorted(cancelled) == [[["slow-1"]], [["slow-2"]]]
