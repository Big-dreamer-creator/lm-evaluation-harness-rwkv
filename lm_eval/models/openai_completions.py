import hashlib
import json
import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import RemoteTokenizer


eval_logger = logging.getLogger(__name__)


class _VLLMRWKVTokenizer(RemoteTokenizer):
    """Tokenizer client for vllm-rwkv's native tokenizer endpoints."""

    def __init__(self, base_url: str, model: str, *args, **kwargs):
        self.model = model
        super().__init__(base_url, *args, **kwargs)

    def _validate_server(self):
        chat_template = self.tokenizer_info.get("chat_template")
        if not isinstance(chat_template, str) or not chat_template:
            raise RuntimeError(
                "vllm-rwkv /tokenizer_info did not provide a chat template. "
                "Start the server with --enable-tokenizer-info-endpoint."
            )
        self.encode("")
        self.decode([0])

    @property
    def eos_token(self) -> None:
        return None

    @property
    def bos_token(self) -> None:
        return None

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> List[int]:
        response = self._request_with_retries(
            "POST",
            f"{self.base_url}/tokenize",
            json={
                "model": self.model,
                "prompt": text,
                "add_special_tokens": False,
            },
        )
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise RuntimeError("Malformed response from vllm-rwkv /tokenize endpoint.")
        return tokens[1:] if tokens and tokens[0] == 0 else tokens

    def decode(self, tokens: List[int]) -> str:
        response = self._request_with_retries(
            "POST",
            f"{self.base_url}/detokenize",
            json={"model": self.model, "tokens": tokens},
        )
        prompt = response.json().get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError(
                "Malformed response from vllm-rwkv /detokenize endpoint."
            )
        return prompt


class _CompletionGeneration(str):
    def __new__(
        cls,
        text: str,
        finish_reason: str | None = None,
        *,
        raw_response: dict | None = None,
        prompt_token_ids: list[int] | None = None,
        output_token_ids: list[int] | None = None,
        reasoning: str | None = None,
    ):
        value = super().__new__(cls, text)
        value.finish_reason = finish_reason
        value.raw_response = raw_response
        value.prompt_token_ids = prompt_token_ids
        value.output_token_ids = output_token_ids
        value.reasoning = reasoning
        value.truncated = finish_reason in {"length", "max_tokens"}
        return value


class _LogLikelihoodEvidence(tuple):
    """Tuple-compatible loglikelihood result with optional HTTP evidence."""

    def __new__(
        cls,
        logprob: float,
        is_greedy: bool,
        *,
        raw_response: dict | None = None,
        prompt_token_ids: list[int] | None = None,
        output_token_ids: list[int] | None = None,
    ):
        value = super().__new__(cls, (logprob, is_greedy))
        value.raw_response = raw_response
        value.prompt_token_ids = prompt_token_ids
        value.output_token_ids = output_token_ids
        value.finish_reason = None
        value.reasoning = None
        value.truncated = False
        return value


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="auto",
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        # Auto-detect tokenizer backend
        if tokenizer_backend == "auto":
            if base_url:
                from lm_eval.utils import check_remote_tokenizer_support

                if check_remote_tokenizer_support(
                    base_url,
                    verify_certificate=verify_certificate,
                    ca_cert_path=ca_cert_path,
                    auth_token=auth_token,
                ):
                    eval_logger.info(
                        "Auto-detected remote tokenizer support. Using remote tokenizer backend."
                    )
                    tokenizer_backend = "remote"
                else:
                    eval_logger.info(
                        "Remote tokenizer not supported. Using huggingface tokenizer backend."
                    )
                    tokenizer_backend = "huggingface"
            else:
                eval_logger.warning(
                    "No base_url provided. Using huggingface tokenizer backend."
                )
                tokenizer_backend = "huggingface"

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                prompt_ids = choice.get("prompt_token_ids")
                all_token_ids = choice.get("token_ids") or []
                output_ids = (
                    all_token_ids[ctxlen:]
                    if isinstance(all_token_ids, list)
                    else None
                )
                res.append(
                    _LogLikelihoodEvidence(
                        logprobs,
                        is_greedy,
                        raw_response=out,
                        prompt_token_ids=prompt_ids,
                        output_token_ids=output_ids,
                    )
                )
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choice in out["choices"]:
                tmp[choice["index"]] = _CompletionGeneration(
                    choice["text"], choice.get("finish_reason")
                )
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("rwkv7-http")
class RWKV7HTTP(LocalCompletionsAPI):
    """HTTP adapter for native RWKV7 completion endpoints.

    The inference process is intentionally outside this repository. The default
    vllm-rwkv transport sends token ids and supports prompt logprobs. The
    transformers-rwkv transport sends rendered strings to ``transformers serve``
    and supports generation tasks only.
    """

    TASK_ADAPTER = "rwkv7-http"
    DEFAULT_MODEL = "rwkv7-g1i-1.5b-20260805-ctx16384"
    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/completions"
    PROMPT_TEMPLATES = {"assistant", "bot", "function_calling"}
    PROMPT_STOPS = {
        "assistant": "\nUser:",
        "bot": "✿",
        "function_calling": "\n### User",
    }
    GENERATION_PROMPTS = {"open_think", "fake_think"}
    SAMPLING_MODES = {"profile", "task"}
    SERVICE_BACKENDS = {"transformers", "vllm"}
    TRANSFORMERS_PROFILE_FILES = {
        "open_think": "generation_config.json",
        "fake_think": "fake_think_generation_config.json",
    }
    SAMPLING_PROFILES = {
        "open_think": {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
        },
        "fake_think": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
        },
    }

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        model=None,
        pretrained=None,
        tokenizer=None,
        tokenizer_backend=None,
        service_backend="vllm",
        rapid_sampling=True,
        num_concurrent=5,
        batch_size=1,
        max_length=16384,
        rwkv_prompt_template="assistant",
        rwkv_generation_prompt="open_think",
        rwkv_sampling_mode="profile",
        rwkv_system_prompt=None,
        rwkv_system_prompt_pattern=None,
        cot_mode=None,
        record_evidence=False,
        tokenized_requests=None,
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        timeout=300,
        max_retries=3,
        **kwargs,
    ):
        model = model or pretrained
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "rwkv7-http requires model= or pretrained= with the complete "
                "served model name."
            )
        if cot_mode is not None:
            rwkv_generation_prompt = cot_mode
        if service_backend not in self.SERVICE_BACKENDS:
            raise ValueError(
                "service_backend must be one of: "
                + ", ".join(sorted(self.SERVICE_BACKENDS))
            )
        if rwkv_prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError(
                "rwkv_prompt_template must be one of: "
                + ", ".join(sorted(self.PROMPT_TEMPLATES))
            )
        if rwkv_generation_prompt not in self.GENERATION_PROMPTS:
            raise ValueError(
                "rwkv_generation_prompt/cot_mode must be one of: "
                + ", ".join(sorted(self.GENERATION_PROMPTS))
            )
        if rwkv_sampling_mode not in self.SAMPLING_MODES:
            raise ValueError(
                "rwkv_sampling_mode must be one of: "
                + ", ".join(sorted(self.SAMPLING_MODES))
            )
        if tokenizer_backend is None:
            tokenizer_backend = (
                "remote" if service_backend == "vllm" else "huggingface"
            )
        if tokenized_requests is None:
            tokenized_requests = service_backend == "vllm"
        if service_backend == "vllm":
            if tokenizer_backend != "remote":
                raise ValueError(
                    "vllm service_backend requires tokenizer_backend=remote."
                )
            if not tokenized_requests:
                raise ValueError(
                    "vllm service_backend requires tokenized_requests=True."
                )
            if not rapid_sampling:
                raise ValueError(
                    "vllm service_backend requires rapid_sampling=True."
                )
        else:
            if tokenizer_backend != "huggingface":
                raise ValueError(
                    "transformers service_backend requires "
                    "tokenizer_backend=huggingface."
                )
            if tokenized_requests:
                raise ValueError(
                    "transformers service_backend requires "
                    "tokenized_requests=False because transformers serve accepts "
                    "string prompts."
                )
            if record_evidence:
                raise ValueError(
                    "transformers service_backend does not expose completion token "
                    "IDs, so record_evidence=True is unsupported."
                )

        self.service_backend = service_backend
        self.rwkv_prompt_template = rwkv_prompt_template
        self.rwkv_generation_prompt = rwkv_generation_prompt
        self.rwkv_sampling_mode = rwkv_sampling_mode
        self.rwkv_system_prompt = rwkv_system_prompt
        self.rwkv_system_prompt_pattern = rwkv_system_prompt_pattern
        self.record_evidence = bool(record_evidence)

        if service_backend == "vllm":
            super().__init__(
                base_url=base_url,
                model=model,
                tokenizer_backend=None,
                num_concurrent=num_concurrent,
                batch_size=batch_size,
                max_length=max_length,
                tokenized_requests=False,
                verify_certificate=verify_certificate,
                ca_cert_path=ca_cert_path,
                auth_token=auth_token,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
            self.tokenizer_backend = "remote"
            self.tokenized_requests = True
            self.tokenizer = _VLLMRWKVTokenizer(
                base_url,
                model,
                timeout=timeout,
                verify_certificate=verify_certificate,
                ca_cert_path=ca_cert_path,
                auth_token=auth_token,
                max_retries=max_retries,
            )
            self._chat_template_source = self.tokenizer.tokenizer_info[
                "chat_template"
            ]
            self._transformers_generation_config = None
        else:
            tokenizer = tokenizer or model
            super().__init__(
                base_url=base_url,
                model=model,
                tokenizer=tokenizer,
                tokenizer_backend=tokenizer_backend,
                num_concurrent=num_concurrent,
                batch_size=batch_size,
                max_length=max_length,
                tokenized_requests=tokenized_requests,
                verify_certificate=verify_certificate,
                ca_cert_path=ca_cert_path,
                auth_token=auth_token,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
            if self._batch_size != 1:
                raise ValueError(
                    "transformers service_backend requires batch_size=1; use "
                    "num_concurrent for parallel HTTP requests."
                )
            self._chat_template_source = self.tokenizer.chat_template
            if not isinstance(self._chat_template_source, str) or not (
                self._chat_template_source
            ):
                raise RuntimeError(
                    "transformers service_backend tokenizer does not provide a "
                    "chat template."
                )
            from transformers import GenerationConfig

            profile_file = self.TRANSFORMERS_PROFILE_FILES[
                self.rwkv_generation_prompt
            ]
            generation_config = GenerationConfig.from_pretrained(
                tokenizer,
                config_file_name=profile_file,
                local_files_only=os.path.isdir(tokenizer),
            )
            self._transformers_generation_config = generation_config.to_dict()
        self._chat_template_sha256 = hashlib.sha256(
            self._chat_template_source.encode("utf-8")
        ).hexdigest()

    @property
    def tokenizer_name(self) -> str:
        system_prompt_sha256 = hashlib.sha256(
            (
                f"{self.rwkv_system_prompt or ''}\0"
                f"{self.rwkv_system_prompt_pattern or ''}"
            ).encode("utf-8")
        ).hexdigest()
        return (
            f"{self.model}:{self._chat_template_sha256}:"
            f"{self.rwkv_prompt_template}:{self.rwkv_generation_prompt}:"
            f"{self.rwkv_sampling_mode}:{system_prompt_sha256}:"
            f"evidence={int(self.record_evidence)}"
        )

    @cached_property
    def eot_token_id(self) -> int:
        return 0

    @cached_property
    def prefix_token_id(self) -> int:
        return 0

    @cached_property
    def eos_string(self) -> None:
        return None

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        return self._chat_template_source

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        from lm_eval.utils import env

        use_system_prompt = bool(self.rwkv_system_prompt) and (
            not self.rwkv_system_prompt_pattern
            or any(
                self.rwkv_system_prompt_pattern.casefold()
                in str(message.get("content", "")).casefold()
                for message in chat_history
            )
        )
        if use_system_prompt:
            chat_history = [dict(message) for message in chat_history]
            if chat_history and chat_history[0]["role"] == "system":
                chat_history[0]["content"] = (
                    f"{chat_history[0]['content'].rstrip()}\n\n"
                    f"{self.rwkv_system_prompt}"
                )
            else:
                chat_history.insert(
                    0,
                    {"role": "system", "content": self.rwkv_system_prompt},
                )
        if (
            not add_generation_prompt
            and self.rwkv_generation_prompt == "fake_think"
            and chat_history
            and chat_history[-1]["role"] == "assistant"
            and not chat_history[-1]["content"].startswith("<think></think>")
        ):
            chat_history = [*chat_history]
            chat_history[-1] = {
                **chat_history[-1],
                "content": f"<think></think>\n{chat_history[-1]['content']}",
            }
        render_kwargs = {
            "tools": kwargs.pop("tools", None),
            "rwkv_prompt_template": self.rwkv_prompt_template,
            "rwkv_generation_prompt": self.rwkv_generation_prompt,
            **kwargs,
        }
        rendered = env.from_string(self._chat_template_source).render(
            messages=chat_history,
            add_generation_prompt=add_generation_prompt,
            **render_kwargs,
        )
        if (
            add_generation_prompt
            and self.rwkv_generation_prompt == "fake_think"
            and rendered.endswith("<think></think")
        ):
            rendered += ">\n"
        return rendered

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        if not context:
            raise ValueError("context cannot be empty")
        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]

        context_tokens = self.tok_encode(context)
        boundary_characters = min(1024, max(1, self.max_length // 4))
        boundary_context = context[-boundary_characters:]
        boundary_context_tokens = self.tok_encode(boundary_context)
        boundary_whole_tokens = self.tok_encode(boundary_context + continuation)
        common_length = 0
        for context_token, whole_token in zip(
            boundary_context_tokens, boundary_whole_tokens
        ):
            if context_token != whole_token:
                break
            common_length += 1
        replaced_context_tokens = len(boundary_context_tokens) - common_length
        if replaced_context_tokens > len(context_tokens):
            raise ValueError("RWKV tokenizer boundary exceeds encoded context")
        stable_context = (
            context_tokens[:-replaced_context_tokens]
            if replaced_context_tokens
            else context_tokens
        )
        return stable_context, boundary_whole_tokens[common_length:]

    def batch_loglikelihood_requests(
        self, chunks
    ) -> Tuple[List[List[int]], List[int], List[Tuple[str, str]]]:
        inputs = []
        context_lengths = []
        cache_keys = []
        for chunk in chunks:
            for cache_key, context_tokens, continuation_tokens in chunk:
                scoring_context = (
                    context_tokens[1:]
                    if context_tokens
                    and context_tokens[0] == self.prefix_token_id
                    else context_tokens
                )
                combined = scoring_context + continuation_tokens
                available_tokens = self.max_length - 1
                overflow = max(0, len(combined) - available_tokens)
                truncated = combined[-available_tokens:] if overflow else combined
                inputs.append([self.prefix_token_id] + truncated)
                context_lengths.append(
                    1 + max(0, len(scoring_context) - overflow)
                )
                cache_keys.append(cache_key)
        return inputs, context_lengths, cache_keys

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        generations = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for output in outputs:
            choices = [None] * len(output["choices"])
            for choice in output["choices"]:
                text = choice.get("text", "")
                reasoning = choice.get("reasoning_content")
                if reasoning is None and "<think>" in text:
                    _, remainder = text.split("<think>", 1)
                    if "</think>" in remainder:
                        reasoning, _ = remainder.split("</think>", 1)
                choices[choice["index"]] = _CompletionGeneration(
                    text,
                    choice.get("finish_reason"),
                    raw_response=output,
                    prompt_token_ids=choice.get("prompt_token_ids")
                    or output.get("prompt_token_ids"),
                    output_token_ids=choice.get("token_ids")
                    or choice.get("output_token_ids"),
                    reasoning=reasoning,
                )
            generations.extend(choices)
        return generations

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        gen_kwargs = dict(gen_kwargs or {})
        do_sample = gen_kwargs.get("do_sample")
        if self.service_backend == "transformers" and not generate:
            raise NotImplementedError(
                "transformers serve does not expose echo prompt logprobs; "
                "multiple_choice and other loglikelihood tasks are unsupported."
            )
        payload = super()._create_payload(
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )
        if self.service_backend == "transformers":
            generation_config = dict(self._transformers_generation_config)
            sampling_parameters = {
                name: gen_kwargs[name]
                for name in self.SAMPLING_PROFILES["open_think"]
                if name in gen_kwargs
            }
            sampling_parameters.update(
                {
                    name: gen_kwargs[name]
                    for name in self.SAMPLING_PROFILES["fake_think"]
                    if name in gen_kwargs
                }
            )
            if self.rwkv_sampling_mode == "task":
                generation_config.update(sampling_parameters)
                if do_sample is not None:
                    generation_config["do_sample"] = bool(do_sample)
                if do_sample is False:
                    generation_config.update(
                        {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": 1,
                            "presence_penalty": 0.0,
                            "frequency_penalty": 0.0,
                        }
                    )
            payload["temperature"] = generation_config.get("temperature", 1.0)
            payload["generation_config"] = json.dumps(
                generation_config, separators=(",", ":"), sort_keys=True
            )
            for name in (
                "top_p",
                "top_k",
                "presence_penalty",
                "frequency_penalty",
                "penalty_decay",
            ):
                payload.pop(name, None)
            if not payload["stop"]:
                payload["stop"] = [self.PROMPT_STOPS[self.rwkv_prompt_template]]
            return payload

        payload.pop("seed", None)
        if generate:
            # vllm-rwkv exposes both prompt and completion IDs in the native
            # OpenAI-compatible response when this flag is enabled. Keeping it
            # on the request makes the producer's per-sample evidence complete.
            if self.record_evidence:
                payload["return_token_ids"] = True
            for name in (
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "frequency_penalty",
                "penalty_decay",
            ):
                if do_sample is False or self.rwkv_sampling_mode == "profile":
                    payload.pop(name, None)
            if self.rwkv_sampling_mode == "profile":
                payload.update(
                    self.SAMPLING_PROFILES[self.rwkv_generation_prompt]
                )
            elif do_sample is False:
                payload["temperature"] = 1
                payload["top_k"] = 1
            if not payload["stop"]:
                payload["stop"] = [self.PROMPT_STOPS[self.rwkv_prompt_template]]
        else:
            if self.record_evidence:
                payload["return_token_ids"] = True
            payload["temperature"] = 1
            payload["top_k"] = 1
        return payload


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    """
    Minimal chat-completions wrapper.
    - Only accepts messages as list[dict].
    - No tokenization or template logic.
    - Use with --apply_chat_template or ensure upstream formats messages correctly.
    """

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert isinstance(messages, list) and all(
            isinstance(m, dict) for m in messages
        ), (
            "LocalChatCompletion expects messages as list[dict]. "
            "If you see this error, ensure --apply_chat_template is set or upstream code formats messages correctly."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choices in out["choices"]:
                    tmp[choices["index"]] = choices["message"]["content"]
            except Exception as e:
                # account for cases that generation is blocked by content filter,
                # which is common for Azure OpenAI Service,
                # not sure if need to account for multiple choices
                eval_logger.warning(f"Could not parse generations: {e}")
                tmp = [""]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if (
            "o1" in self.model
            or "5" in self.model
            or "o3" in self.model
            or "o4" in self.model
        ):
            output.pop("stop")
            output["temperature"] = 1
        return output


@register_model("azure-openai-chat-completions")
class AzureOpenaiChatCompletionsLM(OpenAIChatCompletion):
    def __init__(
        self,
        model: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        base_url: str = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        truncate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        self.truncate = truncate
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url, api_version=api_version, api_key=self.api_key
        )

    @cached_property
    def api_key(self):
        key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `AZURE_OPENAI_API_KEY` environment variable."
            )
        return key
