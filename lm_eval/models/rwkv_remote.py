import os
from functools import cached_property
from typing import Dict, List, Union

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI, _RWKVCompletion


@register_model("rwkv7-remote")
class RWKV7Remote(LocalCompletionsAPI):
    """Adapter for RWKV's hosted, standard OpenAI-format HTTP service (e.g. api.rwkv.rs).

    Unlike ``rwkv7-http`` (a local, unauthenticated ``transformers serve`` process with
    no echo/logprobs), this targets a real OpenAI-compatible ``/v1/completions``
    endpoint assumed to support ``echo``+``logprobs`` natively, so loglikelihood scoring
    uses LocalCompletionsAPI's inherited echo-based path rather than a generation-based
    letter-parsing fallback.

    By default no local/HF-Hub tokenizer is required: prompts are sent through as-is
    (``tokenized_requests=False``), which is sufficient for ``generate_until`` tasks but
    not for loglikelihood/multiple-choice tasks (those require a local tokenizer to
    compute context lengths). Pass ``tokenizer=`` (a local directory or HF Hub repo id
    for the same official RWKV tokenizer the remote service was built with) to enable
    those tasks: this switches to ``tokenizer_backend="huggingface"`` and
    ``tokenized_requests=True``, so requests are sent as token-id arrays rather than
    text -- avoiding any local/remote tokenization mismatch since the server never
    re-tokenizes the given ids.
    """

    DEFAULT_BASE_URL = "https://api.rwkv.rs/v1/completions"

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        tokenizer=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        **kwargs,
    ):
        if tokenizer is not None and tokenizer_backend is None:
            tokenizer_backend = "huggingface"
        if tokenized_requests is None:
            tokenized_requests = tokenizer is not None
        super().__init__(
            base_url=base_url,
            tokenizer=tokenizer,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self) -> str:
        key = os.environ.get("RWKV_API_KEY")
        if not key:
            raise ValueError(
                "API key not found. Please set the `RWKV_API_KEY` environment variable."
            )
        return key

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations = []
        for output in outputs:
            choices = [None] * len(output["choices"])
            for choice in output["choices"]:
                choices[choice["index"]] = _RWKVCompletion(
                    choice.get("text", ""),
                    choice.get("finish_reason"),
                    raw_response=output,
                )
            generations.extend(choices)
        return generations
