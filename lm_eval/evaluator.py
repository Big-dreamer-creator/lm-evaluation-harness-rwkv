from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import random
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import lm_eval.api.model
import lm_eval.api.registry
from lm_eval.api.benchmark import avg_at_k_plan
from lm_eval.api.benchmark_scheduler import (
    _benchmark_pending_rollouts,
    admission_order,
    resolve_max_concurrent_benchmarks,
)
from lm_eval.api.checkpoint import (
    ItemKey,
    compute_idempotency_key,
    compute_item_id,
    compute_request_fingerprint,
    compute_task_fingerprint,
    deserialize_response,
)
from lm_eval.api.checkpoint import model_fingerprint as compute_model_fingerprint
from lm_eval.api.metrics import aggregate_benchmark_fallback, mean
from lm_eval.caching.cache import delete_cache
from lm_eval.defaults import DEFAULT_OTHER_SEED, DEFAULT_RANDOM_SEED, LMEVAL_HASHMM
from lm_eval.evaluator_utils import (
    ResultAcc,
    _compute_task_aggregations,
    _handle_back_comp,
    _log_selected_tasks,
    _process_results,
    get_sample_size,
    print_writeout,
    run_task_tests,
)
from lm_eval.filters import build_filter_ensemble
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import TaskManager
from lm_eval.utils import (
    handle_non_serializable,
    hash_dict_images,
    hash_string,
    positional_deprecated,
    set_torch_seed,
    setup_logging,
    simple_parse_args_string,
    wrap_text,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.api.benchmark import BenchmarkSpec
    from lm_eval.api.checkpoint import CheckpointStore
    from lm_eval.api.group import Group
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.result_schema import EvalResults
    from lm_eval.tasks.manager import TaskDict

    _NestedDict = dict[Group, dict[str, Task] | Group] | dict[str, Task]

eval_logger = logging.getLogger(__name__)


def _build_checkpointed_dispatch(
    checkpoint_store: CheckpointStore,
    task_name: str,
    task_fingerprint: str,
    model_fp: str,
    reqs: list[Any],
    reqtype: str,
) -> tuple[list[Any], list[ItemKey | None]]:
    """Build the checkpoint-aware dispatch list for one (task, reqtype) pair.

    Every ``repeat_ord`` already ``completed`` in the checkpoint store is
    restored directly into ``req.resps`` (by index) and excluded from
    dispatch; only the remaining repeats are cloned into the returned
    dispatch list, each paired with the ``ItemKey`` identity that
    ``amodel_call``/``_loglikelihood_tokens`` use to mark it complete once the
    response comes back.
    """
    existing_items = checkpoint_store.get_items(task_name)
    dispatch_reqs: list[Any] = []
    dispatch_keys: list[ItemKey | None] = []
    to_register: list[ItemKey] = []
    for req in reqs:
        repeats = req.repeats
        if len(req.resps) != repeats:
            req.resps = [None] * repeats
        for repeat_ord in range(repeats):
            item_id = compute_item_id(
                task_name, req.doc_id, req.idx, reqtype, repeat_ord
            )
            request_fp = compute_request_fingerprint(
                item_id, req.args, task_fingerprint, model_fp
            )
            existing = existing_items.get(item_id)
            if existing is not None and existing["status"] == "completed":
                if existing["request_fingerprint"] == request_fp:
                    req.resps[repeat_ord] = deserialize_response(
                        existing["result_json"]
                    )
                    continue
                eval_logger.warning(
                    "Checkpoint item %s (task=%s) changed content since the "
                    "last run; redispatching instead of trusting the stored "
                    "result.",
                    item_id,
                    task_name,
                )
            key = ItemKey(
                item_id=item_id,
                request_fingerprint=request_fp,
                idempotency_key=compute_idempotency_key(item_id, request_fp),
                task_name=task_name,
                doc_id=req.doc_id,
                instance_idx=req.idx,
                request_type=reqtype,
                repeat_ord=repeat_ord,
            )
            dispatch_reqs.append(req)
            dispatch_keys.append(key)
            to_register.append(key)
    if to_register:
        checkpoint_store.register_pending_items(to_register)
    return dispatch_reqs, dispatch_keys


def _install_avg_at_k_patch(
    task_obj: Task, lm: LM, k: int, metric_name: str
) -> None:
    """Wrap ``task_obj.process_results`` to average correctness over ``k``
    independently-sampled generations per doc, surfacing the result under a
    new ``<metric>_avg@k`` key alongside the task's normal (single-sample)
    metrics -- so ``acc``/``exact_match`` etc. keep working unmodified for
    any code that doesn't yet know about avg@k.

    Only two ``OUTPUT_TYPE``s are handled:
    - ``multiple_choice``: relies on ``lm._avg_k_samples[doc_id]`` (populated
      by a ``SUPPORTS_AVG_AT_K`` model's own ``loglikelihood()`` override,
      e.g. ``RWKV7Remote``) holding the ``k`` independently-generated
      per-choice encodings for this doc; the task's own (unmodified)
      ``process_results`` is invoked once per encoding and the results
      averaged.
    - ``generate_until``: relies on the task's filter having been switched
      to ``take_first_k`` (below) so ``results[0]`` is the full list of
      ``k`` generations rather than a single winner; the task's own
      ``process_results`` is invoked once per generation and averaged.
    Any other ``OUTPUT_TYPE`` (loglikelihood/loglikelihood_rolling) is left
    untouched -- avg@k has no meaning for perplexity-style scoring.
    """
    if k <= 1:
        return
    original_process_results = task_obj.process_results
    output_type = task_obj.OUTPUT_TYPE

    def _average_metrics(per_sample: list[dict]) -> dict:
        metrics = dict(per_sample[0])
        for key in per_sample[0]:
            values = [m[key] for m in per_sample if key in m]
            if values and all(isinstance(v, (int, float)) for v in values):
                avg_key = f"{key}_{metric_name}"
                metrics[avg_key] = sum(values) / len(values)
                task_obj._aggregation_list[avg_key] = task_obj._aggregation_list.get(
                    key, mean
                )
                task_obj._higher_is_better[avg_key] = task_obj._higher_is_better.get(
                    key, True
                )
        return metrics

    if output_type == "multiple_choice":

        def patched(doc, results):
            metrics = original_process_results(doc, results)
            doc_id = getattr(task_obj, "_avg_k_current_doc_id", None)
            sample_encodings = lm._avg_k_samples.pop(doc_id, None)
            parsed_flags = getattr(lm, "_avg_k_parsed", {}).pop(doc_id, None)
            if sample_encodings:
                # A sample whose generation couldn't be parsed into a choice
                # letter didn't actually "guess wrong" -- it never named a
                # choice at all. Score it at chance level (1/num_choices)
                # instead of whatever placeholder index it was encoded with,
                # so avg@k measures real correctness rather than being
                # dragged down by answer-format misses.
                chance = 1.0 / len(sample_encodings[0])
                per_sample = []
                for index, enc in enumerate(sample_encodings):
                    sample_metrics = original_process_results(doc, enc)
                    if parsed_flags is not None and not parsed_flags[index]:
                        sample_metrics = {
                            key: (chance if isinstance(value, (int, float)) else value)
                            for key, value in sample_metrics.items()
                        }
                    per_sample.append(sample_metrics)
                metrics.update(_average_metrics(per_sample))
            return metrics
    elif output_type == "generate_until":
        task_obj._filters = [build_filter_ensemble("none", [["take_first_k", {"k": k}]])]

        def patched(doc, results):
            generations = results[0]
            per_sample = [
                original_process_results(doc, [generation])
                for generation in generations
            ]
            return _average_metrics(per_sample)
    else:
        return

    task_obj.process_results = patched


@positional_deprecated
def simple_evaluate(
    model: str | LM,
    model_args: str | dict[str, str | int | float] | None = None,
    tasks: list[str | dict[str, Any] | Task] | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str | None = None,
    max_batch_size: int | None = None,
    device: str | None = None,
    use_cache: str | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: int | float | None = None,
    samples: dict[str, list[int]] | None = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: EvaluationTracker | None = None,
    system_instruction: str | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = True,
    gen_kwargs: str | dict[str, str | float | int] | None = None,
    task_manager: TaskManager | None = None,
    verbosity=None,
    predict_only: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    numpy_random_seed: int = DEFAULT_OTHER_SEED,
    torch_random_seed: int = DEFAULT_OTHER_SEED,
    fewshot_random_seed: int = DEFAULT_OTHER_SEED,
    confirm_run_unsafe_code: bool = False,
    metadata: dict[str, Any] | None = None,
    task_callback: Callable[..., Any] | None = None,
    checkpoint_store: CheckpointStore | None = None,
    benchmark_map: dict[str, BenchmarkSpec] | None = None,
    max_concurrent_benchmarks: int = 3,
    max_concurrent_benchmarks_explicit: bool = False,
) -> EvalResults | None:
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        model (str | LM): Name of model or LM object. See
            lm_eval.models.__init__.py for available aliases.
        model_args: String or dict arguments for each model class, e.g.,
            "pretrained=EleutherAI/pythia-1.3B,revision=main" or {"pretrained": "EleutherAI/pythia-1.3B"}.
            Ignored if ``model`` argument is a LM object.
        tasks (list[str | dict | Task]): List of task names or Task objects.
            Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined
            and type(task).__name__ otherwise.
        num_fewshot (int): Number of examples in few-shot context.
        batch_size (int | str | None): Batch size for model.
        max_batch_size (int | None): Maximal batch size to try with automatic
            batch size detection.
        device (str | None): PyTorch device (e.g. "cpu" or "cuda:0") for running
            models.
        use_cache (str | None): A path to a sqlite db file for caching model
            responses. `None` if not caching.
        cache_requests (bool): Speed up evaluation by caching the building of
            dataset requests (inputs). `None` if not caching.
        rewrite_requests_cache (bool): Rewrites all the request cache if set to
            `True`. `None` if not desired.
        delete_requests_cache (bool): Deletes all the request cache if set to
            `True`. `None` if not desired.
        limit (int | float | None): Limit the number of examples per task (only
            use this for testing). If <1, limit is a percentage of the total
            number of examples.
        samples (dict | None): Dictionary indicating which examples should be
            tested in each task, e.g.,
            {"mmlu_astronomy": [0, 3, 6], "mmlu_anatomy": [1, 4, 7, 10]}.
            Incompatible with `limit`.
        bootstrap_iters (int): Number of iterations for bootstrap statistics, used
            when calculating stderrs. Set to 0 for no stderr calculations to be
            performed.
        check_integrity (bool): Whether to run the relevant part of the test suite
            for the tasks.
        write_out (bool): If True, write out an example document and model input
            for checking task integrity.
        log_samples (bool): If True, write out all model outputs and documents for
            per-sample measurement and post-hoc analysis.
        evaluation_tracker (EvaluationTracker | None): Tracker for logging
            experiment configuration and results.
        system_instruction (str | None): System instruction to be applied to the
            prompt.
        apply_chat_template (bool | str): Specifies whether to apply a chat
            template to the prompt. If set to True, the default chat template is
            applied. If set to a string, applies the specified chat template by
            name. Defaults to False (no chat template applied).
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as a
            multiturn conversation or a single user turn.
        gen_kwargs (dict | str | None): Arguments for model generation. Ignored
            for all tasks with loglikelihood output_type.
        task_manager (TaskManager | None): Task manager instance to use.
        verbosity (str | None): Verbosity level for logging.
        predict_only (bool): If True, only model outputs will be generated and
            returned. Metrics will not be evaluated.
        random_seed (int): Random seed for python's random module. If set to None,
            the seed will not be set.
        numpy_random_seed (int): Random seed for numpy. If set to None, the seed
            will not be set.
        torch_random_seed (int): Random seed for torch. If set to None, the seed
            will not be set.
        fewshot_random_seed (int): Random seed for fewshot sampler random generator.
            If set to None, the seed of generator will be set to None.
        confirm_run_unsafe_code (bool): Whether to confirm running tasks marked
            as unsafe (e.g. code execution tasks).
        metadata (dict | None): Additional metadata to be added to the task
            manager. Will get passed to the download function of the task.

    Returns:
        dict | None: Dictionary of results, or None if not on rank 0.
    """
    if verbosity is not None:
        eval_logger.info("Setting verbosity through simple_evaluate is deprecated.")
    start_date = time.time()

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )

    _NEEDS_CHAT_TEMPLATE = ("inst", "chat")
    if (
        (
            isinstance(model_args, str)
            and any(kw in model_args.lower() for kw in _NEEDS_CHAT_TEMPLATE)
        )
        or (
            isinstance(model_args, dict)
            and any(
                any(kw in str(v).lower() for kw in _NEEDS_CHAT_TEMPLATE)
                for v in model_args.values()
            )
        )
    ) and not apply_chat_template:
        eval_logger.warning(
            wrap_text(
                f"""pretrained={model_args.get("pretrained") if isinstance(model_args, dict) else model_args} appears to be an
                instruct or chat variant but chat template is not applied.
                Recommend setting `apply_chat_template` (optionally `fewshot_as_multiturn`).""",
            )
        )

    if delete_requests_cache:
        eval_logger.info("Deleting requests cache...")
        delete_cache()

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        set_torch_seed(torch_random_seed)

    if fewshot_random_seed is not None:
        seed_message.append(f"Setting fewshot manual seed to {fewshot_random_seed}")

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs:
        if isinstance(gen_kwargs, str):
            gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            f"generation_kwargs: {gen_kwargs} specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if not gen_kwargs:
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            eval_logger.warning("model_args not specified. Using defaults.")
            model_args = ""

        if isinstance(model_args, dict):
            eval_logger.info(
                f"Initializing {model} model, with arguments: {model_args}"
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )

        else:
            eval_logger.info(
                wrap_text(
                    f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
                )
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, lm_eval.api.model.LM):
            raise TypeError(
                f"The value of `model` passed to simple_evaluate() was of type {type(model)}, but is required to be a subclass of lm_eval.api.model.LM . This may be because you are passing an initialized Hugging Face PreTrainedModel without having wrapped it in `lm_eval.models.huggingface.HFLM(pretrained=my_model)` first."
            )
        eval_logger.info("Using pre-initialized model")
        lm = model

    # Under TP launchers (torchrun) every rank reports lm.rank==0; fall back to
    # LOCAL_RANK so each process gets its own cache db and only LOCAL_RANK==0
    # performs final result aggregation / I/O.
    cache_rank = lm.rank or int(os.environ.get("LOCAL_RANK", "0"))
    if use_cache is not None:
        eval_logger.info(
            f"Using cache at {use_cache + '_rank' + str(cache_rank) + '.db'}"
        )
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(cache_rank)
            + ".db",
        )

    if checkpoint_store is not None:
        if isinstance(lm, lm_eval.api.model.CachingLM):
            raise ValueError(
                "checkpoint_store and use_cache cannot be combined: CachingLM's "
                "cache-check wrapper does not forward checkpoint item identity."
            )
        lm.set_checkpoint_store(checkpoint_store)

    if task_manager is None:
        metadata = (
            simple_parse_args_string(model_args)
            if isinstance(model_args, str)
            else model_args
            if isinstance(model_args, dict)
            else {}
        ) | (metadata or {})
        task_manager = TaskManager(metadata=metadata)

    # Load tasks - returns {"tasks":.., "groups":..}
    loaded = task_manager.load(tasks)

    # Log selected tasks with hierarchy
    _log_selected_tasks(loaded["tasks"], loaded["groups"], task_manager)

    # Apply config overrides to tasks
    for task_name, task_obj in loaded["tasks"].items():
        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.set_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )
            eval_logger.info(
                f"{task_obj.config.task}: Using gen_kwargs: {task_obj.config.generation_kwargs}"
            )

        if predict_only:
            eval_logger.info(
                f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
            )
            # we have to change the class properties post-hoc. This is pretty hacky.
            task_obj.override_metric(metric_name="bypass")

        # override tasks' fewshot values to the provided num_fewshot arg value
        # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
        if num_fewshot is not None:
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.set_config(key="num_fewshot", value=num_fewshot)
        else:
            # if num_fewshot not provided, and the task does not define a default one, default to 0
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                task_obj.set_config(key="num_fewshot", value=0)
        # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
        task_obj.set_fewshot_seed(seed=fewshot_random_seed)

        # avg@k: only for backends that declare support (RWKV7Remote), and only
        # for tasks that haven't already opted into their own repeated-sampling
        # scoring (e.g. cruxeval's native repeats=10 pass@k) -- clobbering those
        # would break their bespoke process_results.
        if getattr(lm, "SUPPORTS_AVG_AT_K", False) and task_obj.get_config(
            "repeats"
        ) in (None, 1):
            num_docs = len(task_obj.eval_docs)
            k, metric_name = avg_at_k_plan(num_docs, limited=limit is not None)
            task_obj.set_config(key="repeats", value=k, update=False)
            _install_avg_at_k_patch(task_obj, lm, k, metric_name)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model if isinstance(model, str) else "CUSTOM",
            model_args=model_args or "",
            system_instruction=system_instruction,
            chat_template=lm.chat_template(apply_chat_template)
            if apply_chat_template
            else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    _callback_model_args = (
        simple_parse_args_string(model_args)
        if isinstance(model_args, str)
        else dict(model_args or {})
    )
    callback_context: dict[str, Any] = {
        "config": {
            "model": model if isinstance(model, str) else type(model).__name__,
            "model_args": _callback_model_args,
            "batch_size": batch_size,
            "device": device,
            "gen_kwargs": gen_kwargs,
            "max_length": _callback_model_args.get("max_length"),
        },
        "lm_eval_version": lm_eval.__version__,
        "git_hash": get_git_commit_hash(),
    }
    if hasattr(lm, "get_model_info"):
        callback_context["config"].update(lm.get_model_info())  # type: ignore

    results = evaluate(
        lm=lm,
        task_dict=loaded,
        limit=limit,
        samples=samples,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        task_callback=task_callback,
        task_callback_context=callback_context,
        evaluation_tracker=evaluation_tracker,
        checkpoint_store=checkpoint_store,
        benchmark_map=benchmark_map,
        max_concurrent_benchmarks=max_concurrent_benchmarks,
        max_concurrent_benchmarks_explicit=max_concurrent_benchmarks_explicit,
    )
    if verbosity is not None:
        setup_logging(verbosity=verbosity)

    # `lm.rank == 0` covers DP / single-process; `LOCAL_RANK == 0` covers TP
    # (torchrun), where every rank reports rank==0 but only one process should
    # build/return results so callers don't duplicate file writes.
    if lm.rank == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available
        if hasattr(lm, "get_model_info"):
            results["config"].update(lm.get_model_info())  # type: ignore
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []  # type: ignore
                ),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = start_date
        add_env_info(results)  # additional environment info to results
        add_tokenizer_info(results, lm)  # additional info about tokenizer
        return results
    else:
        return None


@positional_deprecated
def evaluate(
    lm: LM,
    task_dict: TaskDict | _NestedDict,
    limit: int | None = None,
    samples: dict[str, list[int]] | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: int | None = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: str | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    confirm_run_unsafe_code: bool = False,
    task_callback: Callable[..., Any] | None = None,
    task_callback_context: dict[str, Any] | None = None,
    evaluation_tracker: EvaluationTracker | None = None,
    checkpoint_store: CheckpointStore | None = None,
    benchmark_map: dict[str, BenchmarkSpec] | None = None,
    max_concurrent_benchmarks: int = 3,
    max_concurrent_benchmarks_explicit: bool = False,
) -> EvalResults | None:
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        lm (LM): Language Model.
        task_dict (TaskDict): Dictionary returned by TaskManager.load() containing
            'tasks', 'groups', and 'group_map' entries.
        limit (int | None): Limit the number of examples per task (only use this
            for testing).
        samples (dict | None): Dictionary indicating which examples should be
            tested in each task, e.g.,
            {"mmlu_astronomy": [0, 3, 6], "mmlu_anatomy": [1, 4, 7, 10]}.
        cache_requests (bool): Speed up evaluation by caching the building of
            dataset requests.
        rewrite_requests_cache (bool): Rewrites all the request cache if set to
            `True`.
        bootstrap_iters (int | None): Number of iterations for bootstrap
            statistics, used when calculating stderr. Set to 0 for skipping all
            stderr calculations.
        write_out (bool): If True, write out an example document and model input
            for checking task integrity.
        log_samples (bool): If True, write out all model outputs and documents
            for per-sample measurement and post-hoc analysis.
        system_instruction (str | None): System instruction to be applied to the
            prompt.
        apply_chat_template (bool | str): Specifies whether to apply a chat
            template to the prompt. If set to True, the default chat template is
            applied. If set to a string, applies the specified chat template by
            name. Defaults to False (no chat template applied).
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as a
            multiturn conversation or a single user turn.
        verbosity (str): Verbosity level for logging. (no-op, deprecated)
        confirm_run_unsafe_code (bool): Whether to confirm running tasks marked
            as unsafe (e.g code execution tasks).

    Returns:
        dict | None: Dictionary of results, or None if not on rank 0.
    """

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )
    if samples is not None:
        eval_logger.info(f"Evaluating examples for tasks {list(samples.keys())}")
    # tracks all Instances/requests a model must generate output on.
    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)
    task_padding_requests: dict[str, dict[str, int]] = {}

    # Initialize groups and tasks
    # handle back_comp. Assume if "tasks" not present, then using old nested.
    if "tasks" not in task_dict:
        groups, eval_tasks = _handle_back_comp(cast("_NestedDict", task_dict))
    else:
        task_dict = cast("TaskDict", task_dict)
        groups, eval_tasks = task_dict.get("groups", {}), task_dict.get("tasks", {})

    # Group leaf tasks by the benchmark they were requested under, so that
    # scoreboard publication happens once per benchmark (as soon as its own
    # leaves are all done), never once per leaf task.
    benchmark_trackers: dict[str, dict[str, Any]] = {}
    leaf_to_benchmark: dict[str, str] = {}
    if benchmark_map and (task_callback is not None or checkpoint_store is not None):
        for bname, spec in benchmark_map.items():
            if spec.leaf_task_names is not None:
                leaves = list(spec.leaf_task_names)
            else:
                group_obj = groups.get(bname)
                leaves = (
                    [t.task_name for t in group_obj.get_all_tasks(recursive=True)]
                    if group_obj is not None
                    else []
                )
            leaves = [leaf for leaf in leaves if leaf in eval_tasks]
            if not leaves:
                continue
            benchmark_trackers[bname] = {
                "spec": spec,
                "leaf_names": leaves,
                "remaining": len(leaves),
                "collected_metrics": {},
                "collected_artifacts": {},
            }
            for leaf in leaves:
                leaf_to_benchmark[leaf] = bname

    # Initialize accumulators for per-sample metrics and logged samples
    eval_results_acc: dict[str, ResultAcc] = {
        task_name: {
            "task": task_obj,
            "raw_metrics": defaultdict(list),
            "logged_samples": [],
        }
        for task_name, task_obj in eval_tasks.items()
    }
    if not log_samples and not all(
        "bypass" not in getattr(task_obj, "_metric_fn_list", {})
        for task_obj in eval_tasks.values()
    ):
        raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    # validation checks:
    # 1.are we running code that is marked as unsafe.
    # 2.are we running multimodal task <-> non-multimodal model class, or vice-versa.
    incompatible_tasks = []
    for task_name, task in eval_tasks.items():
        if getattr(task, "UNSAFE_CODE", False) and not confirm_run_unsafe_code:
            raise ValueError(
                f"Attempted to run task: {task_name} which is marked as unsafe. Set confirm_run_unsafe_code=True to run this task."
            )

        if getattr(task, "MULTIMODAL", False) and not getattr(lm, "MULTIMODAL", False):
            incompatible_tasks.append(task_name)
    if len(incompatible_tasks) > 0 and not getattr(lm, "MULTIMODAL", False):
        raise ValueError(
            f"Attempted to run tasks: {incompatible_tasks} which require multimodal input, but the selected model type does not currently implement this. Multimodal support is currently restricted to the ['hf-multimodal', 'vllm-vlm'] model type."
        )
    # end validation check

    # Cache the limit arg.
    limit_arg = limit
    limits = []
    for task_name, task in eval_tasks.items():
        limit = get_sample_size(task, limit_arg)
        limits.append(limit)
        task.build_all_requests(
            limit=limit,
            samples=samples.get(task_name, None) if samples is not None else samples,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=bool(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template", None)
            if apply_chat_template
            else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "")
            if apply_chat_template
            else "",
        )
        eval_logger.debug(
            f"Task: {task_name}; number of requests on this rank: {len(task.instances)}"
        )
        if write_out:
            print_writeout(task)
        if task_callback is None and checkpoint_store is None:
            for instance in task.instances:
                requests[instance.request_type].append(instance)

        if lm.world_size > 1:
            import torch

            instances_rnk = torch.tensor(
                len(task._instances) if task._instances else 0, device=lm.device
            )
            gathered_item = lm.all_gather(instances_rnk).cpu().detach().numpy().tolist()
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = (
                "loglikelihood"
                if task.OUTPUT_TYPE == "multiple_choice"
                else task.OUTPUT_TYPE
            )
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad
            task_padding_requests[task_name] = {reqtype: numpad}

    RANK = lm.rank
    WORLD_SIZE = lm.world_size

    checkpoint_capable = checkpoint_store is not None and getattr(
        lm, "_supports_checkpoint_items", False
    )
    model_fp = compute_model_fingerprint(lm) if checkpoint_capable else None

    def run_requests(
        task_requests: dict[str, list[Any]],
        task_padding: dict[str, int],
        task_name: str | None = None,
        task_fingerprint: str | None = None,
    ) -> None:
        checkpoint_active = (
            checkpoint_capable and task_name is not None and task_fingerprint is not None
        )
        for reqtype, reqs in task_requests.items():
            eval_logger.info(f"Running {reqtype} requests")
            if checkpoint_active:
                cloned_reqs, item_keys = _build_checkpointed_dispatch(
                    checkpoint_store,
                    task_name,
                    task_fingerprint,
                    model_fp,
                    reqs,
                    reqtype,
                )
            else:
                cloned_reqs = [req for req in reqs for _ in range(req.repeats)]
                item_keys = None
            for _ in range(task_padding.get(reqtype, 0)):
                cloned_reqs.extend([reqs[-1]] * reqs[-1].repeats)
                if item_keys is not None:
                    item_keys.extend([None] * reqs[-1].repeats)
            if cloned_reqs:
                if item_keys is not None:
                    resps = getattr(lm, reqtype)(cloned_reqs, item_keys=item_keys)
                    for response, request, key in zip(
                        resps, cloned_reqs, item_keys, strict=True
                    ):
                        if key is not None:
                            request.resps[key.repeat_ord] = response
                        else:
                            request.resps.append(response)
                else:
                    resps = getattr(lm, reqtype)(cloned_reqs)
                    for response, request in zip(resps, cloned_reqs, strict=True):
                        request.resps.append(response)
            if WORLD_SIZE > 1:
                lm.barrier()

    def response_evidence(responses: list[Any]) -> list[dict[str, Any]]:
        """Fields Scoreboard consumes from a raw model response.

        Must run here, on the live response objects, before evaluation_tracker
        serializes ``resps`` to plain strings and this data is gone for good.
        """
        evidence = []
        for response in responses:
            raw_response = getattr(response, "raw_response", None)
            if raw_response is None:
                continue
            choice = next(iter(raw_response.get("choices") or []), {})
            message = choice.get("message") or {}
            evidence.append(
                {
                    "text": choice.get("text") or message.get("content"),
                    "prompt": getattr(response, "prompt", None),
                    "tool_calls": message.get("tool_calls") or [],
                    "finish_reason": getattr(response, "finish_reason", None)
                    or choice.get("finish_reason"),
                    "truncated": bool(getattr(response, "truncated", False)),
                    "output_token_ids": getattr(response, "output_token_ids", None)
                    or [],
                }
            )
        return evidence

    def finish_benchmark(
        task_name: str,
        acc: ResultAcc,
        limit: int | None,
    ) -> None:
        task = acc["task"]
        task.apply_filters()
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        for filter_key in task.instances[0].filtered_resps:
            indices = samples.get(task_name, None) if samples is not None else None
            doc_iterator = task.doc_iterator(
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
                samples=indices,
            )
            for doc_id, doc in doc_iterator:
                doc_id_true = indices[doc_id] if indices else doc_id
                document_requests = instances_by_doc_id[doc_id]
                # Read by the avg@k process_results patch (see
                # _install_avg_at_k_patch) to look up this doc's
                # independently-sampled per-choice encodings; a harmless
                # no-op for any task that isn't patched.
                task._avg_k_current_doc_id = doc_id
                metrics = task.process_results(
                    doc,
                    [req.filtered_resps[filter_key] for req in document_requests],
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id_true,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in document_requests],
                        "resps": [req.resps for req in document_requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in document_requests
                        ],
                        "filter": filter_key,
                        "response_evidence": response_evidence(
                            [
                                resp
                                for req in document_requests
                                for resp in req.resps
                            ]
                        ),
                        "metrics": list(metrics.keys()),
                        "doc_hash": hash_string(
                            json.dumps(
                                document_requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(document_requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    acc["logged_samples"].append(example)
                for metric, value in metrics.items():
                    acc["raw_metrics"][(metric, filter_key)].append(value)

        if WORLD_SIZE > 1:
            if log_samples:
                all_samples = lm.gather_object(
                    {task_name: acc["logged_samples"]}, dst=0
                )
                if RANK == 0:
                    acc["logged_samples"] = list(
                        itertools.chain.from_iterable(
                            rank_data[task_name]
                            for rank_data in all_samples  # type: ignore
                        )
                    )

            all_metrics = lm.gather_object({task_name: dict(acc["raw_metrics"])}, dst=0)
            if RANK == 0:
                for metric_key in acc["raw_metrics"]:
                    acc["raw_metrics"][metric_key] = list(
                        itertools.chain.from_iterable(
                            rank_data[task_name][metric_key]
                            for rank_data in all_metrics  # type: ignore
                        )
                    )

        if RANK == 0 and (task_callback is not None or checkpoint_store is not None):
            aggregated_metrics, sample_len = _compute_task_aggregations(
                task, acc["raw_metrics"], bootstrap_iters
            )
            task_config = dict(task.dump_config())
            callback_results = {
                **(task_callback_context or {}),
                "results": {
                    task_name: {
                        "name": task_name,
                        "alias": task_config.get("task_alias", task_name),
                        "sample_len": sample_len,
                        **aggregated_metrics,
                    }
                },
                "configs": {task_name: task_config},
                "versions": {task_name: task.VERSION},
                "n-samples": {
                    task_name: {
                        "original": len(task.eval_docs),
                        "effective": sample_len,
                    }
                },
            }
            artifact_paths = None
            if evaluation_tracker is not None:
                try:
                    artifact_paths = evaluation_tracker.save_task_artifacts(
                        task_name, callback_results, acc["logged_samples"]
                    )
                except (OSError, TypeError, ValueError) as error:
                    eval_logger.warning(
                        "Could not persist task artifacts for %s: %s", task_name, error
                    )
            if checkpoint_store is not None:
                checkpoint_store.mark_benchmark_completed(
                    task_name,
                    (artifact_paths or {}).get("results"),
                    (artifact_paths or {}).get("metadata"),
                )
            # Leaf tasks are never published to the scoreboard on their own:
            # only once every leaf of the benchmark it belongs to is done do
            # we publish one aggregated entry for that benchmark.
            bname = leaf_to_benchmark.get(task_name)
            if bname is not None:
                tracker = benchmark_trackers[bname]
                tracker["collected_metrics"][task_name] = callback_results["results"][
                    task_name
                ]
                tracker["collected_artifacts"][task_name] = artifact_paths
                tracker["remaining"] -= 1
                if tracker["remaining"] == 0:
                    _finish_and_publish_benchmark(bname)
        if WORLD_SIZE > 1:
            lm.barrier()

    def _finish_and_publish_benchmark(bname: str) -> None:
        tracker = benchmark_trackers[bname]
        spec = tracker["spec"]
        leaves = tracker["leaf_names"]
        metrics = tracker["collected_metrics"]

        if spec.strategy == "standalone":
            leaf = leaves[0]
            aggregated = {**metrics[leaf], "name": bname, "alias": bname}
            primary_metric_hint = None
        elif spec.strategy == "group_aggregate":
            group_obj = groups[bname]
            aggregated = group_obj.aggregate(metrics)
            primary_metric_hint = group_obj.aggregate_metric_list[0].metric
        else:  # group_fallback / tag_fallback
            aggregated, primary_metric_hint = aggregate_benchmark_fallback(
                bname, leaves, metrics
            )

        n_original = sum(
            len(eval_tasks[leaf].eval_docs) for leaf in leaves if leaf in eval_tasks
        )
        n_effective = sum(
            metrics[leaf].get("sample_len", 0) for leaf in leaves if leaf in metrics
        )

        fingerprint = compute_task_fingerprint(
            {"leaves": sorted(leaves), "strategy": spec.strategy}
        )
        skip_publish = False
        if checkpoint_store is not None:
            existing = checkpoint_store.get_benchmark(bname)
            skip_publish = (
                existing is not None and existing["publish_status"] == "succeeded"
            )
            checkpoint_store.start_benchmark(bname, fingerprint)

        results_doc = {
            **(task_callback_context or {}),
            "results": {bname: aggregated},
            "configs": (
                {bname: eval_tasks[leaves[0]].dump_config()}
                if spec.strategy == "standalone"
                else {}
            ),
            "versions": (
                {bname: eval_tasks[leaves[0]].VERSION}
                if spec.strategy == "standalone"
                else {}
            ),
            "n-samples": {
                bname: {"original": n_original, "effective": n_effective}
            },
            "strategy": spec.strategy,
            "primary_metric_hint": primary_metric_hint,
            "leaf_artifact_paths": dict(tracker["collected_artifacts"]),
        }
        artifact_paths = None
        if evaluation_tracker is not None:
            try:
                artifact_paths = evaluation_tracker.save_benchmark_artifacts(
                    bname, results_doc
                )
            except (OSError, TypeError, ValueError) as error:
                eval_logger.warning(
                    "Could not persist benchmark artifacts for %s: %s", bname, error
                )

        if checkpoint_store is not None:
            checkpoint_store.mark_benchmark_completed(
                bname,
                (artifact_paths or {}).get("results"),
                (artifact_paths or {}).get("metadata"),
            )
        if task_callback is not None and not skip_publish:
            try:
                callback_status = task_callback(
                    task_name=bname, artifact_paths=artifact_paths
                )
                publish_ok = True
                if isinstance(callback_status, dict):
                    publish_ok = callback_status.get("publication") not in (
                        "failed",
                        None,
                    )
                    eval_logger.log(
                        logging.INFO if publish_ok else logging.WARNING,
                        "Benchmark %s publication=%s uploaded=%s%s",
                        bname,
                        callback_status.get("publication", "unknown"),
                        callback_status.get("uploaded", False),
                        f" error={callback_status['error']}"
                        if callback_status.get("error")
                        else "",
                    )
                if checkpoint_store is not None:
                    checkpoint_store.set_publish_status(
                        bname, "succeeded" if publish_ok else "failed"
                    )
            except Exception as error:  # noqa: BLE001
                eval_logger.warning("Task callback failed for %s: %s", bname, error)
                if checkpoint_store is not None:
                    checkpoint_store.set_publish_status(bname, "failed")

    def _run_one_task(
        task_name: str, acc: ResultAcc, task_limit: int | None
    ) -> None:
        task_fingerprint = None
        if checkpoint_store is not None:
            task_fingerprint = compute_task_fingerprint(dict(acc["task"].dump_config()))
            checkpoint_store.start_benchmark(task_name, task_fingerprint)
        benchmark_requests = defaultdict(list)
        for instance in acc["task"].instances:
            benchmark_requests[instance.request_type].append(instance)
        run_requests(
            benchmark_requests,
            task_padding_requests.get(task_name, {}),
            task_name=task_name,
            task_fingerprint=task_fingerprint,
        )

    if task_callback is None and checkpoint_store is None:
        run_requests(requests, padding_requests)
        for (task_name, acc), task_limit in zip(
            eval_results_acc.items(), limits, strict=True
        ):
            finish_benchmark(task_name, acc, task_limit)
    else:
        # Group leaf tasks into schedulable units: every leaf of a benchmark
        # group runs (sequentially, within one worker thread) as a single
        # unit so they still aggregate/publish together once all their
        # leaves are done; a task that isn't part of any benchmark group is
        # its own one-task unit. Insertion order here matches the original
        # (pre-scheduler) task order.
        schedule_units: dict[str, list[tuple[str, ResultAcc, int | None]]] = {}
        for (task_name, acc), task_limit in zip(
            eval_results_acc.items(), limits, strict=True
        ):
            bname = leaf_to_benchmark.get(task_name, task_name)
            schedule_units.setdefault(bname, []).append((task_name, acc, task_limit))

        concurrency = resolve_max_concurrent_benchmarks(
            requested=max_concurrent_benchmarks,
            explicit=max_concurrent_benchmarks_explicit,
            world_size=WORLD_SIZE,
            warn=eval_logger.warning,
        )

        if concurrency <= 1:
            # No extra threads, original strictly-sequential task order --
            # this is both the world_size != 1 path and what users get back
            # by explicitly setting max_concurrent_benchmarks = 1.
            for bname in schedule_units:
                for task_name, acc, task_limit in schedule_units[bname]:
                    _run_one_task(task_name, acc, task_limit)
                    finish_benchmark(task_name, acc, task_limit)
        else:
            pending_by_unit = {
                bname: _benchmark_pending_rollouts(
                    [(task_name, acc["task"]) for task_name, acc, _ in members],
                    checkpoint_store,
                )
                for bname, members in schedule_units.items()
            }
            pending_order = admission_order(pending_by_unit)

            scoring_queue: queue.Queue = queue.Queue()
            stop_event = threading.Event()
            active_threads: dict[str, threading.Thread] = {}
            first_error: BaseException | None = None

            def _run_benchmark_unit(bname: str) -> None:
                try:
                    for task_name, acc, task_limit in schedule_units[bname]:
                        if stop_event.is_set():
                            break
                        _run_one_task(task_name, acc, task_limit)
                    scoring_queue.put((bname, None))
                except BaseException as exc:  # noqa: BLE001
                    scoring_queue.put((bname, exc))

            def _admit_next() -> None:
                while (
                    pending_order
                    and not stop_event.is_set()
                    and len(active_threads) < concurrency
                ):
                    bname = pending_order.pop(0)
                    thread = threading.Thread(
                        target=_run_benchmark_unit,
                        args=(bname,),
                        name=f"benchmark-{bname}",
                    )
                    active_threads[bname] = thread
                    thread.start()

            _admit_next()
            while active_threads:
                bname, error = scoring_queue.get()
                thread = active_threads.pop(bname, None)
                if thread is not None:
                    thread.join()
                if error is not None:
                    if first_error is None:
                        first_error = error
                    stop_event.set()
                    eval_logger.error(
                        "Benchmark %s failed during concurrent scheduling: %s",
                        bname,
                        error,
                    )
                elif not stop_event.is_set():
                    for task_name, acc, task_limit in schedule_units[bname]:
                        finish_benchmark(task_name, acc, task_limit)
                _admit_next()

            if first_error is not None:
                raise first_error

    if RANK == 0:
        res = _process_results(eval_results_acc, groups, bootstrap_iters)

        samples = None
        if log_samples:
            samples = res.samples
            if LMEVAL_HASHMM and hasattr(lm, "MULTIMODAL"):
                samples = hash_dict_images(samples)

        return res._to_eval_results(samples=samples)
    else:
        return None
