"""Native ConfigurableTask integration for EvalScope's non-agent Coding protocols."""

from __future__ import annotations

import copy
import json
import re
from contextlib import closing

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.api.task import ConfigurableTask
from lm_eval.rwkv_coding import JudgeError, pass_at_k
from lm_eval.tasks.evalscope_coding import data, runtime
from lm_eval.tasks.evalscope_coding.vendor import (
    bird,
    code_utils,
    ds1000,
    lcb_extract,
    multiple,
)


class Coding(ConfigurableTask):
    """Uses native loading, generation, checkpointing, sample logs and aggregation."""

    HANDLES_REPEATS = True
    HANDLES_FINAL_ANSWER = True

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.pop("class", None)
        self.benchmark = config["metadata"]["benchmark"]
        self.requires_multi_turn = self.benchmark in {"scicode", "mt_bench"}
        self._judge = None
        self.scoring_details = []
        self._render_chat = None
        self._extra_system = None
        keys = (
            ["main_problem_pass_rate", "subproblem_pass_rate"]
            if self.benchmark == "scicode"
            else ["judge_score", "first_turn_judge_score", "second_turn_judge_score"]
            if self.benchmark == "mt_bench"
            else [f"pass@{k}" for k in config["metadata"].get("pass_k", [1])]
        )
        if (
            self.benchmark == "ds1000"
            or self.benchmark == "bird_sql"
            or self.benchmark.startswith("swe_bench")
        ):
            keys = ["accuracy"]
        config.update(
            output_type="generate_until",
            unsafe_code=True,
            # Like Task.override_metric, name the class override without retaining
            # the live task/model (which can hold locks) in serializable config.
            process_results="process_results",
            metric_list=[
                {"metric": key, "aggregation": "mean", "higher_is_better": True}
                for key in keys
            ],
        )
        config.setdefault("num_fewshot", 3 if self.benchmark == "mbpp" else 0)
        config.setdefault("repeats", 1)
        config.setdefault(
            "generation_kwargs", {"until": [], "max_gen_toks": 4096, "do_sample": True}
        )
        config.setdefault("test_split", data.CATALOG[self.benchmark]["eval_split"])
        super().__init__(config=config)
        # Keep every repeat for unbiased pass@k. No majority vote or take-first collapse.
        self._filters = [FilterEnsemble("all", [])]

    @property
    def options(self):
        return self.config.metadata

    def download(self, dataset_kwargs=None, **kwargs):
        self.dataset = data.load_dataset(
            self.config.dataset_path,
            self.config.dataset_name,
            self.benchmark,
            self.options,
            dict(dataset_kwargs or {}),
        )
        split = self.config.test_split
        self.dataset[split] = data.select_docs(
            self.dataset[split], self.benchmark, self.options
        )

    def build_all_requests(self, **kwargs):
        # Request-cache hits bypass fewshot_context; retain the renderer for later turns.
        self._render_chat = (
            kwargs.get("chat_template") if kwargs.get("apply_chat_template") else None
        )
        self._extra_system = kwargs.get("system_instruction")
        if self.requires_multi_turn and not self._render_chat:
            raise ValueError("Multi-turn Coding requires --apply_chat_template")
        super().build_all_requests(**kwargs)

    def doc_to_text(self, doc):
        messages = data.initial_messages(
            doc, self.benchmark, self.options, self.config.num_fewshot
        )
        return "\n\n".join(message["content"] for message in messages)

    def doc_to_target(self, doc):
        return data.target(doc, self.benchmark)

    def fewshot_context(
        self,
        doc,
        num_fewshot,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        gen_prefix=None,
    ):
        if gen_prefix:
            raise ValueError(
                "Coding tasks preserve their benchmark prompt; do not override gen_prefix"
            )
        self._extra_system = system_instruction
        self._render_chat = chat_template if apply_chat_template else None
        if self.requires_multi_turn and not self._render_chat:
            raise ValueError(
                "SciCode and MT-Bench require --apply_chat_template to preserve conversation roles"
            )
        messages = self.messages(doc, num_fewshot)
        return self.render(messages)

    def messages(self, doc, shots=None):
        messages = data.initial_messages(
            doc,
            self.benchmark,
            self.options,
            self.config.num_fewshot if shots is None else shots,
        )
        if self._extra_system:
            if messages[0]["role"] == "system":
                messages[0]["content"] = (
                    self._extra_system + "\n\n" + messages[0]["content"]
                )
            else:
                messages.insert(0, {"role": "system", "content": self._extra_system})
        return messages

    def render(self, messages):
        if self._render_chat:
            return self._render_chat(messages)
        return "\n\n".join(message["content"] for message in messages)

    def construct_requests(self, doc, ctx, **kwargs):
        request = super().construct_requests(doc, ctx, **kwargs)
        if self.benchmark == "mt_bench":
            args = dict(request.arguments[1])
            args["temperature"] = 0.0  # Official Coding category temperature.
            owner = getattr(self._render_chat, "__self__", None)
            if getattr(owner, "SUPPORTS_TASK_TEMPERATURE", False):
                args["rwkv_task_temperature"] = 0.0
            request.arguments = (request.arguments[0], args)
        return request

    def validate_runtime(self):
        """Fail on missing runtime resources before spending generation requests."""
        ks = self.options.get("pass_k", [1])
        if (
            not isinstance(ks, list)
            or not ks
            or any(type(k) is not int or k < 1 or k > self.config.repeats for k in ks)
        ):
            raise ValueError(
                "pass_k must contain positive integers no larger than repeats"
            )
        if self.benchmark in runtime.IMAGES:
            self._judge = runtime.python_judge(self.benchmark, self.options)
        elif self.benchmark.startswith("multiple_"):
            result = runtime.multiple_judge("print(1)", "python", self.options)
            if not result["passed"]:
                raise JudgeError("Sandbox Fusion failed its known-pass control")
        elif self.benchmark.startswith("swe_bench"):
            import docker
            import swebench.harness.run_evaluation  # noqa: F401

            with closing(docker.from_env()) as client:
                client.ping()
        elif self.benchmark == "mt_bench":
            import os

            import openai  # noqa: F401

            if not os.environ.get(
                self.options.get("judge", {}).get("api_key_env", "OPENAI_API_KEY")
            ):
                raise ValueError(
                    "MT-Bench requires a judge API key in the configured environment variable"
                )

    def run_generation_rounds(self, dispatch):
        """Sequential turns, parallel independent conversations; each turn uses native checkpoints."""
        initial = list(self.instances)
        dispatch({"generate_until": initial})
        conversations = []
        for request in initial:
            for ordinal, response in enumerate(request.resps):
                history = self.messages(request.doc)
                history.append({"role": "assistant", "content": str(response)})
                conversations.append(
                    {
                        "request": request,
                        "ordinal": ordinal,
                        "history": history,
                        "answers": [str(response)],
                        "evidence": [self.evidence(request, response)],
                        "count": len(request.doc["sub_steps"])
                        if self.benchmark == "scicode"
                        else 2,
                    }
                )
        for turn in range(1, max((item["count"] for item in conversations), default=0)):
            active, requests = [], []
            for item in conversations:
                if turn >= item["count"]:
                    continue
                base = item["request"]
                prompt = (
                    data.scicode_turn(base.doc, self.options, turn)
                    if self.benchmark == "scicode"
                    else base.doc["prompt"][turn]
                )
                item["history"].append({"role": "user", "content": prompt})
                request = Instance(
                    "generate_until",
                    base.doc,
                    (self.render(item["history"]), copy.deepcopy(base.args[1])),
                    idx=turn * base.repeats + item["ordinal"],
                    metadata=(base.task_name, base.doc_id, 1),
                )
                requests.append(request)
                active.append(item)
            dispatch({"generate_until": requests})
            for item, request in zip(active, requests, strict=True):
                response = request.resps[0]
                item["answers"].append(str(response))
                item["history"].append({"role": "assistant", "content": str(response)})
                item["evidence"].append(self.evidence(request, response))
        for item in conversations:
            item["request"].resps[item["ordinal"]] = json.dumps(
                {
                    "answers": item["answers"],
                    "messages": item["history"],
                    "evidence": item["evidence"],
                },
                ensure_ascii=False,
            )

    @staticmethod
    def evidence(request, response):
        return {
            "input": request.args[0],
            "generation_kwargs": request.args[1],
            "response": str(response),
            "finish_reason": getattr(response, "finish_reason", None),
            "truncated": bool(getattr(response, "truncated", False)),
            "raw_response": getattr(response, "raw_response", None),
        }

    def _final(self, response):
        if getattr(response, "truncated", False):
            return None
        text = str(response)
        mode = self.options.get("thinking", "auto")
        owner = getattr(self._render_chat, "__self__", None)
        if (
            mode == "auto"
            and getattr(owner, "rwkv_generation_prompt", None) == "open_think"
        ):
            mode = "open_think"
        if mode not in {"auto", "open_think", "none"}:
            raise ValueError("metadata.thinking must be auto, open_think, or none")
        # Strip only a leading reasoning wrapper; never tags inside generated code.
        if mode == "open_think" or (
            mode == "auto" and text.lstrip().startswith("<think>")
        ):
            if "</think>" not in text:
                return None
            text = text.split("</think>", 1)[1]
        return text

    def _python(self, code, tests, timeout=None):
        judge = self._judge
        if judge is None or timeout is not None:
            judge = runtime.python_judge(self.benchmark, self.options, timeout)
        result = judge.check(code, tests)
        return {**result, "image_id": judge.image_id}

    def score_one(self, doc, response):
        final = self._final(response)
        if final is None:
            return {"passed": False, "answer": None, "status": "unfinished_thinking"}
        name = self.benchmark
        if name in {"humaneval", "humaneval_plus"}:
            blocks = re.findall(r"```\w*\n(.*?)```", final, re.DOTALL)
            code = blocks[0] if blocks else final
            result = self._python(
                doc["prompt"] + code, doc["test"] + f"\ncheck({doc['entry_point']})"
            )
        elif name in {"mbpp", "mbpp_plus"}:
            code, _ = code_utils.extract_code_from_freeform_completion(
                final.split("[DONE]", 1)[0], "python", first_block_only=True
            )
            for stop in ("\nassert", '\n"""'):
                code = code.split(stop, 1)[0]
            if name == "mbpp":
                code = doc["test_setup_code"] + "\n" + code
            code = code.split("if __name__ ==", 1)[0]
            tests = "\n".join(doc["test_list"])
            if name == "mbpp_plus":
                tests = "\n".join([*doc["test_imports"], tests, doc["test"]])
            result = self._python(code, tests)
        elif name.startswith("bigcodebench"):
            code, _ = code_utils.extract_code_from_freeform_completion(
                final, "python", first_block_only=True
            )
            code = code if code.strip() else final
            executable = (
                doc["code_prompt"] + "\n    pass\n"
                if self.options.get("calibrate", True)
                else ""
            ) + code
            tests = (
                doc["test"]
                + "\nimport unittest\n_result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestCases))\nassert _result.wasSuccessful()\n"
            )
            result = self._python(executable, tests)
        elif name.startswith("multiple_"):
            language = doc.get("language") or self.options["subset"].split("-", 1)[1]
            code = multiple.build_full_code(
                final, language, doc.get("stop_tokens", []), doc["tests"]
            )
            result = runtime.multiple_judge(
                code, multiple.normalize_languages(language)[1], self.options
            )
        elif name == "live_code_bench":
            code = lcb_extract.extract_code_generation(final)
            sample = data.lcb_sample(doc)
            count = len(json.loads(sample)["inputs"])
            result = self._python(
                "", runtime.lcb_test_program(code, sample), timeout=6 * (count + 1) + 10
            )
        elif name == "ds1000":
            code = ds1000.extract_code(final)
            program = ds1000.build_test_program(doc["code_context"], code)
            result = self._python(
                "import os\nos.environ.update(TF_CPP_MIN_LOG_LEVEL='3', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg')",
                program,
            )
        elif name == "bird_sql":
            code = bird.extract_sql(final)
            path = data.bird_database(doc, self.options)
            gold = bird.execute_sql(path, doc["SQL"], 30)
            if not gold.ok:
                raise JudgeError(f"BIRD reference SQL could not run: {gold.error}")
            pred = bird.execute_sql(path, code, 30)
            result = {
                "passed": pred.ok and set(pred.rows) == set(gold.rows),
                "pred_error": pred.error,
            }
        elif name.startswith("swe_bench"):
            from swebench.inference.make_datasets.utils import extract_diff

            code = extract_diff(final)
            result = runtime.swe_judge(doc, code, self.options)
        else:
            raise ValueError(name)
        return {**result, "answer": code}

    def score_conversation(self, doc, raw):
        conversation = json.loads(raw)
        answers = [
            None if evidence.get("truncated") else self._final(answer)
            for answer, evidence in zip(
                conversation["answers"], conversation["evidence"], strict=True
            )
        ]
        if self.benchmark == "mt_bench":
            return runtime.mtbench_judge(
                doc["prompt"],
                [answer or "" for answer in answers],
                data.mtbench_references(doc),
                self.options,
            )
        generated, verdicts = [], []
        for step, answer in zip(doc["sub_steps"], answers, strict=True):
            code = answer or ""
            if "```" in code:
                fence = "```python" if "```python" in code else "```"
                code = code.split(fence, 1)[1].split("```", 1)[0]
            generated.append(code.strip())
            sections = [
                "import sys\nsys.path.insert(0, '/workspace')",
                doc["required_dependencies"],
                "from test_util import are_dicts_close, cmp_tuple_or_list",
                *generated,
                "from process_data import process_hdf5_to_tuple",
                f"targets = process_hdf5_to_tuple({step['step_number']!r}, {len(step['test_cases'])})",
            ]
            for i, test in enumerate(step["test_cases"]):
                sections.extend([f"target = targets[{i}]", test])
            program = (
                "\n".join(sections)
                .replace("from scicode.compare.cmp import cmp_tuple_or_list", "")
                .replace("from scicode.compare.cmp import are_dicts_close", "")
            )
            verdict = (
                self._python("", program) if answer is not None else {"passed": False}
            )
            verdicts.append({**verdict, "answer": code, "step": step["step_number"]})
        passed = sum(item["passed"] for item in verdicts)
        return {
            "main_problem_pass_rate": float(passed == len(verdicts)),
            "subproblem_pass_rate": (passed, len(verdicts)),
            "steps": verdicts,
        }

    def process_results(self, doc, results):
        responses = results[0]
        if not isinstance(responses, list):
            responses = [responses]
        if not responses:
            raise ValueError("Coding received no responses")
        verdicts = [
            (
                self.score_conversation(doc, response)
                if self.requires_multi_turn
                else self.score_one(doc, response)
            )
            for response in responses
        ]
        self.scoring_details = verdicts
        if self.requires_multi_turn:
            if self.benchmark == "scicode":
                return {
                    "main_problem_pass_rate": mean(
                        [v["main_problem_pass_rate"] for v in verdicts]
                    ),
                    "subproblem_pass_rate": (
                        sum(v["subproblem_pass_rate"][0] for v in verdicts),
                        sum(v["subproblem_pass_rate"][1] for v in verdicts),
                    ),
                }
            return {
                key: mean([v[key] for v in verdicts]) for key in self._metric_fn_list
            }
        n, c = len(verdicts), sum(v["passed"] for v in verdicts)
        if "accuracy" in self._metric_fn_list:
            return {"accuracy": c / n}
        return {
            f"pass@{k}": pass_at_k(n, c, k) for k in self.options.get("pass_k", [1])
        }

    def aggregation(self):
        result = super().aggregation()
        if self.benchmark == "scicode":
            result["subproblem_pass_rate"] = subproblem_pass_rate
        return result


def subproblem_pass_rate(items):
    return sum(item[0] for item in items) / sum(item[1] for item in items)
