import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.checkpoint import ItemKey, serialize_response
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import (
    LocalCompletionsAPI,
    _RWKVCompletion,
    _rwkv_sampling_config,
)
from lm_eval.models.rwkv_choice import build_choice_prompt, choice_letters, parse_choice_index


eval_logger = logging.getLogger(__name__)

# Mirrors lighteval-rwkv's `http_model.MAX_NEW_TOKENS`: open_think/fake_think
# generations think before answering, so the generic API-model default
# (`self._max_gen_toks`, 256) truncates the chain before a choice letter is
# ever produced. Avg@k choice resolution always uses this instead.
MAX_NEW_TOKENS = 8192


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

    When ``tokenizer`` is provided and the caller applies the harness's chat
    template (``apply_chat_template=True``), this adapter renders the official
    RWKV chat template itself -- mirroring ``rwkv7-http`` -- so ``rwkv_prompt_template``
    actually takes effect for the remote backend too, both point at the same
    underlying RWKV7 inference stack. Without a tokenizer, chat-template rendering
    is skipped and that option is a no-op (matching this class's prior behavior).

    ``rwkv_generation_prompt`` (``fake_think``/``open_think``) always selects one of
    two fixed sampling presets (``SAMPLING_PARAMETERS``) regardless of whether a
    tokenizer is configured -- these are not read from a per-checkpoint
    ``generation_config.json`` and cannot be overridden via ``gen_kwargs``.
    """

    # Signals to the evaluator's per-task override loop (lm_eval/evaluator.py)
    # that this backend implements avg@k: multiple_choice tasks are scored by
    # sampling `Instance.repeats` independent generations (through the real
    # open_think/fake_think sampling profile) and averaging correctness over
    # them, rather than the echo/logprob loglikelihood path used by default.
    SUPPORTS_AVG_AT_K = True

    DEFAULT_BASE_URL = "https://api.rwkv.rs/v1/completions"
    PROMPT_TEMPLATES = {"assistant", "bot", "function_calling"}
    PROMPT_STOPS = {
        "assistant": "\nUser:",
        "bot": "✿",
        "function_calling": "\n### User",
    }
    GENERATION_PROMPTS = {"open_think", "fake_think"}
    # Fixed sampling presets for the two RWKV7 generation modes, mirroring the RWKV
    # weight family's own defaults (and lighteval-rwkv's SAMPLING_PARAMETERS). These
    # are shipped in source rather than read from a per-checkpoint generation_config.json
    # so the reported sampling_config is a stable, auditable constant independent of
    # which tokenizer directory (if any) a given run happens to point at.
    SAMPLING_PARAMETERS = {
        "open_think": {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
            "do_sample": True,
            "stop_strings": ["✿", "\nUser:", "\n### User"],
        },
        "fake_think": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "do_sample": True,
            "stop_strings": ["✿", "\nUser:", "\n### User"],
        },
    }

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        tokenizer=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        rwkv_prompt_template="assistant",
        rwkv_generation_prompt="fake_think",
        cot_mode=None,
        **kwargs,
    ):
        if tokenizer is not None and tokenizer_backend is None:
            tokenizer_backend = "huggingface"
        if tokenized_requests is None:
            tokenized_requests = tokenizer is not None
        if cot_mode is not None:
            rwkv_generation_prompt = cot_mode
        if rwkv_prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError("rwkv_prompt_template must be assistant, bot, or function_calling.")
        if rwkv_generation_prompt not in self.GENERATION_PROMPTS:
            raise ValueError("rwkv_generation_prompt must be open_think or fake_think.")

        self.rwkv_prompt_template = rwkv_prompt_template
        self.rwkv_generation_prompt = rwkv_generation_prompt
        super().__init__(
            base_url=base_url,
            tokenizer=tokenizer,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

        self._chat_template_source: Optional[str] = None
        self._chat_template_sha256: Optional[str] = None
        self._generation_profile: dict = dict(
            self.SAMPLING_PARAMETERS[rwkv_generation_prompt]
        )
        # Per-doc list of k independently-sampled per-choice (ll, is_greedy)
        # encodings, populated by loglikelihood() and consumed by the avg@k
        # process_results patch installed in lm_eval/evaluator.py. Cleared as
        # each doc is consumed so it never leaks across evaluation runs.
        self._avg_k_samples: Dict[Any, List[List[Tuple[float, bool]]]] = {}
        if tokenizer is not None:
            self._chat_template_source = self.tokenizer.chat_template
            if not isinstance(self._chat_template_source, str) or not self._chat_template_source:
                raise RuntimeError("The RWKV tokenizer must provide the official chat template.")
            self._chat_template_sha256 = hashlib.sha256(
                self._chat_template_source.encode("utf-8")
            ).hexdigest()

    @cached_property
    def api_key(self) -> str:
        key = os.environ.get("RWKV_API_KEY")
        if not key:
            raise ValueError(
                "API key not found. Please set the `RWKV_API_KEY` environment variable."
            )
        return key

    @property
    def tokenizer_name(self) -> str:
        if self._chat_template_sha256 is None:
            return super().tokenizer_name
        return (
            f"{self.model}:{self._chat_template_sha256}:"
            f"{self.rwkv_prompt_template}:{self.rwkv_generation_prompt}"
        )

    def get_model_info(self) -> dict:
        return {"sampling_config": _rwkv_sampling_config(self._generation_profile)}

    def loglikelihood(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
        item_keys: Optional[List[Optional[ItemKey]]] = None,
    ) -> List[Tuple[float, bool]]:
        """Score ``multiple_choice`` docs via avg@k generation instead of echo/logprobs.

        Unlike the echo-based loglikelihood this class inherits from
        ``LocalCompletionsAPI``/``TemplateAPI`` (which never issues a
        ``generate=True`` request, so the open_think/fake_think sampling
        profile in ``_create_payload`` never applies to multiple_choice
        tasks), this asks the model to generate a lettered answer
        ``Instance.repeats`` times per doc -- exercising the real sampling
        profile and protocol -- and records the k per-sample outcomes for
        the avg@k ``process_results`` patch (installed in
        ``lm_eval/evaluator.py``) to average against gold. The single
        ``(loglikelihood, is_greedy)`` list returned here still encodes a
        majority-vote winner per doc, so ``acc``/``acc_norm`` keep working
        unmodified if no avg@k patch is installed.

        ``item_keys`` (checkpoint identity, parallel to ``requests``) is
        accepted because the evaluator's checkpoint-aware dispatch always
        passes it to backends with ``_supports_checkpoint_items = True``
        (which this class inherits); since scoring here bypasses the base
        class's echo-based ``_loglikelihood_tokens`` entirely, completion is
        marked directly against ``self.checkpoint_store`` instead.
        """
        results: List[Optional[Tuple[float, bool]]] = [None] * len(requests)
        groups: Dict[Any, List[int]] = defaultdict(list)
        for position, request in enumerate(requests):
            groups[(request.task_name, request.doc_id)].append(position)

        for positions in groups.values():
            positions = sorted(positions, key=lambda p: requests[p].idx)
            group = [requests[p] for p in positions]
            contexts = {req.args[0] for req in group}
            if "" in contexts and len(contexts) > 1:
                eval_logger.warning(
                    "RWKV7Remote's avg@k generative scoring does not support "
                    "the unconditional likelihood acc_mutual_info needs; "
                    "acc_mutual_info results for this task are not meaningful."
                )
                kept_positions, kept_group = [], []
                for position, request in zip(positions, group):
                    if request.args[0] == "":
                        results[position] = (0.0, False)
                    else:
                        kept_positions.append(position)
                        kept_group.append(request)
                positions, group = kept_positions, kept_group
                if not group:
                    continue

            # `Instance.repeats` (avg@k's k) duplicates each *choice* Instance
            # k times consecutively (see evaluator.py's `cloned_reqs`), so this
            # group is [choice_0 x k, choice_1 x k, ...] rather than one entry
            # per real choice -- cluster consecutive same-`idx` positions back
            # into the real, distinct choices before building the prompt.
            choice_positions: List[List[int]] = []
            choice_texts: List[str] = []
            last_idx = None
            for position, request in zip(positions, group):
                if last_idx is None or request.idx != last_idx:
                    choice_positions.append([])
                    choice_texts.append(request.args[1].strip())
                    last_idx = request.idx
                choice_positions[-1].append(position)

            context = group[0].args[0]
            choices = choice_texts
            k = min(len(cg) for cg in choice_positions)
            letters = choice_letters(choices)
            prompt = build_choice_prompt(context, choices)

            sample_encodings: List[List[Tuple[float, bool]]] = []
            chosen_indices: List[int] = []
            for _ in range(k):
                index = self._resolve_choice(prompt, letters, choices)
                chosen_indices.append(index)
                sample_encodings.append(
                    [
                        (0.0, True) if choice_idx == index else (-1e6, False)
                        for choice_idx in range(len(choices))
                    ]
                )
            doc_id = group[0].doc_id
            self._avg_k_samples[doc_id] = sample_encodings

            majority_index = Counter(chosen_indices).most_common(1)[0][0]
            for choice_idx, cg in enumerate(choice_positions):
                outcome = (0.0, True) if choice_idx == majority_index else (-1e6, False)
                for position in cg:
                    results[position] = outcome

        if item_keys and self.checkpoint_store is not None:
            completed = [
                (key.item_id, serialize_response(res))
                for key, res in zip(item_keys, results)
                if key is not None and res is not None
            ]
            if completed:
                self.checkpoint_store.mark_items_completed(completed)

        return results

    def _resolve_choice(self, prompt: str, letters: List[str], choices: List[str]) -> int:
        gen_kwargs = {"until": [], "max_gen_toks": MAX_NEW_TOKENS}
        instance = Instance(
            request_type="generate_until",
            doc={},
            arguments=(prompt, gen_kwargs),
            idx=0,
        )
        generated = str(self.generate_until([instance])[0])
        index = parse_choice_index(generated, letters, choices)
        if index < 0:
            eval_logger.warning(
                "RWKV7Remote could not parse a choice letter from the generated "
                f"answer {generated.strip()!r}; defaulting to option {letters[0]!r}."
            )
            return 0
        return index

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        if self._chat_template_source is None:
            return super().apply_chat_template(
                chat_history, add_generation_prompt=add_generation_prompt, **kwargs
            )

        from lm_eval.utils import env

        rendered = env.from_string(self._chat_template_source).render(
            messages=chat_history,
            add_generation_prompt=add_generation_prompt,
            rwkv_prompt_template=self.rwkv_prompt_template,
            rwkv_generation_prompt=self.rwkv_generation_prompt,
            tools=kwargs.pop("tools", None),
            **kwargs,
        )
        if (
            add_generation_prompt
            and self.rwkv_generation_prompt == "fake_think"
            and rendered.endswith("<think></think")
        ):
            rendered += ">\n"
        return rendered

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        payload = super()._create_payload(
            messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
        )
        if not generate:
            return payload

        profile = dict(self._generation_profile)
        payload["temperature"] = profile.get("temperature", 1.0)
        payload["generation_config"] = json.dumps(
            profile, separators=(",", ":"), sort_keys=True
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
