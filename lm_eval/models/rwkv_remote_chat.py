import json
import logging
import os
from collections import Counter, defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.checkpoint import ItemKey, serialize_response
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import (
    LocalChatCompletion,
    _RWKVCompletion,
    _rwkv_sampling_config,
)
from lm_eval.models.rwkv_choice import build_choice_prompt, choice_letters, parse_choice_index
from lm_eval.models.rwkv_remote import MAX_NEW_TOKENS, RWKV7Remote


eval_logger = logging.getLogger(__name__)


@register_model("rwkv7-remote-chat")
class RWKV7RemoteChat(LocalChatCompletion):
    """Adapter for RWKV's hosted ``/v1/chat/completions`` endpoint.

    Unlike ``rwkv7-remote`` (which targets ``/v1/completions`` and must render the
    RWKV chat template itself using a local tokenizer), this class sends unrendered
    ``messages`` plus ``chat_template_kwargs={"rwkv_prompt_template", "rwkv_generation_prompt"}``
    and lets the vLLM-backed server apply its own chat template -- mirroring
    lighteval-rwkv's ``RWKVHttpModel``. No local/HF-Hub tokenizer is required or
    accepted; the server has been confirmed (against api.rwkv.rs) to apply the
    matching per-template stop sequence (e.g. ``✿`` for ``bot``) on its own, so this
    adapter does not send an explicit ``stop`` for the RWKV template either.
    """

    SUPPORTS_AVG_AT_K = True

    DEFAULT_BASE_URL = "https://api.rwkv.rs/v1/chat/completions"
    PROMPT_TEMPLATES = RWKV7Remote.PROMPT_TEMPLATES
    GENERATION_PROMPTS = RWKV7Remote.GENERATION_PROMPTS
    SAMPLING_PARAMETERS = RWKV7Remote.SAMPLING_PARAMETERS

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        rwkv_prompt_template="assistant",
        rwkv_generation_prompt="fake_think",
        cot_mode=None,
        **kwargs,
    ):
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
            tokenizer_backend=None,
            tokenized_requests=False,
            **kwargs,
        )

        self._generation_profile: dict = dict(
            self.SAMPLING_PARAMETERS[rwkv_generation_prompt]
        )
        # Per-doc list of k independently-sampled per-choice (ll, is_greedy)
        # encodings, populated by loglikelihood() and consumed by the avg@k
        # process_results patch installed in lm_eval/evaluator.py. Cleared as
        # each doc is consumed so it never leaks across evaluation runs.
        self._avg_k_samples: Dict[Any, List[List[Tuple[float, bool]]]] = {}

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
        return f"{self.model}:{self.rwkv_prompt_template}:{self.rwkv_generation_prompt}"

    def get_model_info(self) -> dict:
        return {"sampling_config": _rwkv_sampling_config(self._generation_profile)}

    def loglikelihood(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
        item_keys: Optional[List[Optional[ItemKey]]] = None,
    ) -> List[Tuple[float, bool]]:
        """Score ``multiple_choice`` docs via avg@k generation (see ``RWKV7Remote.loglikelihood``).

        Identical strategy to ``RWKV7Remote``: this backend has no echo/logprob
        path to fall back on (chat-completions APIs don't expose prompt logprobs),
        so every loglikelihood request is resolved by generating a lettered answer
        ``Instance.repeats`` times per doc and averaging correctness over them.
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
                    "RWKV7RemoteChat's avg@k generative scoring does not support "
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
        # `generate_until` unwraps a `JsonChatStr` into the [{"role","content"}, ...]
        # list this class's `_create_payload` requires; a bare string (fine for the
        # raw-completions `rwkv7-remote`) would fail LocalChatCompletion's assertion.
        chat_prompt = JsonChatStr(
            json.dumps([{"role": "user", "content": prompt}], ensure_ascii=False)
        )
        instance = Instance(
            request_type="generate_until",
            doc={},
            arguments=(chat_prompt, gen_kwargs),
            idx=0,
        )
        generated = str(self.generate_until([instance])[0])
        index = parse_choice_index(generated, letters, choices)
        if index < 0:
            eval_logger.warning(
                "RWKV7RemoteChat could not parse a choice letter from the generated "
                f"answer {generated.strip()!r}; defaulting to option {letters[0]!r}."
            )
            return 0
        return index

    def _create_payload(
        self,
        messages: List[Dict],
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
        payload["chat_template_kwargs"] = {
            "rwkv_prompt_template": self.rwkv_prompt_template,
            "rwkv_generation_prompt": self.rwkv_generation_prompt,
        }
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
                    choice.get("message", {}).get("content", ""),
                    choice.get("finish_reason"),
                    raw_response=output,
                )
            generations.extend(choices)
        return generations
