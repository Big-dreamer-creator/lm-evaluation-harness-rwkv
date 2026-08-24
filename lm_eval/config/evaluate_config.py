from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import yaml


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from lm_eval.utils import simple_parse_args_string


if TYPE_CHECKING:
    from argparse import Namespace

    from lm_eval.tasks import TaskManager

eval_logger = logging.getLogger(__name__)
DICT_KEYS = [
    "wandb_args",
    "wandb_config_args",
    "hf_hub_log_args",
    "metadata",
    "model_args",
    "gen_kwargs",
]
VERSIONED_CONFIG_FIELDS = {
    "apply_chat_template",
    "backend",
    "base_url",
    "batch_size",
    "benchmarks",
    "device",
    "fewshot_as_multiturn",
    "include_defaults",
    "include_path",
    "limit",
    "log_samples",
    "max_length",
    "model_overrides",
    "model_name",
    "num_concurrent",
    "output_dir",
    "publication",
    "publish",
    "scoreboard",
    "rwkv_profile",
    "samples",
    "schema_version",
    "seed",
    "task_overrides",
    "use_cache",
}
VERSIONED_REQUIRED_FIELDS = {
    "backend",
    "base_url",
    "benchmarks",
    "max_length",
    "model_name",
    "output_dir",
    "rwkv_profile",
    "schema_version",
}
RWKV_PROFILE_FIELDS = {
    "generation_prompt",
    "prompt_template",
    "sampling_mode",
    "wkv_mode",
}
RWKV_PROMPT_TEMPLATES = {"assistant", "bot", "function_calling"}
RWKV_GENERATION_PROMPTS = {"fake_think", "open_think"}
RWKV_SAMPLING_MODES = {"profile", "task"}
RWKV_WKV_MODES = {"fp16", "fp32io16"}
PUBLICATION_FIELDS = {
    "enabled",
    "base_url",
    "token_env",
    "timeout",
    "finalize",
    "model_sha256",
    "model_revision",
    "task_metadata",
}


@dataclass(slots=True)
class EvaluatorConfig:
    """Configuration for language model evaluation runs.

    This dataclass contains all parameters for configuring model evaluations via
    `simple_evaluate()` or the CLI. It supports initialization from:
    - CLI arguments (via `from_cli()`)
    - TOML or YAML configuration files (via `from_config()`)
    - Direct instantiation with keyword arguments

    The configuration handles argument parsing, validation, and preprocessing
    to ensure properly structured and validated.

    Example:
        # From CLI arguments
        config = EvaluatorConfig.from_cli(args)

        # From a TOML file
        config = EvaluatorConfig.from_config("eval_config.toml")

        # Direct instantiation
        config = EvaluatorConfig(
            model="hf",
            model_args={"pretrained": "gpt2"},
            tasks=["hellaswag", "arc_easy"],
            num_fewshot=5
        )

      See individual field documentation for detailed parameter descriptions.
    """

    # Core evaluation parameters
    config: str | None = field(
        default=None, metadata={"help": "Path to TOML or YAML config file"}
    )
    model: str = field(default="hf", metadata={"help": "Name of model e.g. 'hf'"})
    model_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for model initialization"}
    )
    tasks: str | list[str] = field(
        default_factory=list,
        metadata={"help": "Comma-separated list of task names to evaluate"},
    )

    # Few-shot and batching
    num_fewshot: int | None = field(
        default=None, metadata={"help": "Number of examples in few-shot context"}
    )
    batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_batch_size: int | None = field(
        default=None, metadata={"help": "Maximum batch size for auto batching"}
    )

    # Device
    device: str | None = field(
        default="cuda:0", metadata={"help": "Device to use (e.g. cuda, cuda:0, cpu)"}
    )

    # Data sampling and limiting
    limit: float | None = field(
        default=None, metadata={"help": "Limit number of examples per task"}
    )
    samples: str | dict | None = field(
        default=None,
        metadata={"help": "dict, JSON string or path to JSON file with doc indices"},
    )

    # Caching
    use_cache: str | None = field(
        default=None,
        metadata={"help": "Path to sqlite db file for caching model outputs"},
    )
    cache_requests: dict = field(
        default_factory=dict,
        metadata={"help": "Cache dataset requests: true/refresh/delete"},
    )

    # Output and logging flags
    check_integrity: bool = field(
        default=False, metadata={"help": "Run test suite for tasks"}
    )
    write_out: bool = field(
        default=False, metadata={"help": "Print prompts for first few documents"}
    )
    log_samples: bool = field(
        default=False, metadata={"help": "Save model outputs and inputs"}
    )
    output_path: str | None = field(
        default=None, metadata={"help": "Dir path where result metrics will be saved"}
    )
    predict_only: bool = field(
        default=False,
        metadata={
            "help": "Only save model outputs, don't evaluate metrics. Use with log_samples."
        },
    )

    # Chat and instruction handling
    system_instruction: str | None = field(
        default=None, metadata={"help": "Custom System instruction to add"}
    )
    apply_chat_template: bool | str = field(
        default=False,
        metadata={
            "help": "Apply chat template to prompt. Either True, or a string identifying the tokenizer template."
        },
    )
    fewshot_as_multiturn: bool | None = field(
        default=None,
        metadata={
            "help": "Use fewshot as multi-turn conversation. Defaults to True when apply_chat_template is set."
        },
    )

    # Configuration display
    show_config: bool = field(
        default=False, metadata={"help": "Show full config at end of evaluation"}
    )

    # External tasks and generation
    include_path: str | list[str] | None = field(
        default=None, metadata={"help": "Additional dir path for external tasks"}
    )
    include_defaults: bool = field(
        default=True, metadata={"help": "Include built-in task definitions"}
    )
    gen_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for model generation. Will update Task defaults"},
    )

    # Logging and verbosity
    verbosity: str | None = field(
        default=None, metadata={"help": "Logging verbosity level"}
    )

    # External integrations
    wandb_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for wandb.init"}
    )
    wandb_config_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for wandb.config.update"}
    )
    hf_hub_log_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for HF Hub logging"}
    )
    trackio_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for trackio logging"}
    )

    # Reproducibility
    seed: list = field(
        default_factory=lambda: [0, 1234, 1234, 1234],
        metadata={"help": "Seeds for random, numpy, torch, fewshot (random)"},
    )

    # Security
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code for HF datasets"}
    )
    confirm_run_unsafe_code: bool = field(
        default=False,
        metadata={
            "help": "Confirm understanding of unsafe code risks (for code tasks that executes arbitrary Python)"
        },
    )

    # Internal metadata
    metadata: dict = field(
        default_factory=dict,
        metadata={"help": "Additional metadata for tasks that require it"},
    )

    # Dashboard publication.  Evaluation remains useful without a configured
    # publisher; when enabled, the run writes a complete local publication
    # spool before attempting any network request.
    publication: dict = field(
        default_factory=dict,
        metadata={"help": "Scoreboard publication settings"},
    )

    @classmethod
    def from_cli(cls, namespace: Namespace) -> EvaluatorConfig:
        """Build an EvaluationConfig by merging with a simple precedence.

        CLI args > config file > built-in defaults.
        """
        # Start with built-in defaults
        config = asdict(cls())

        # Load and merge TOML/YAML config if provided
        if used_config := getattr(namespace, "config", None):
            config.update(cls.load_config(cast("str", used_config)))

        # Override with CLI args (only truthy values or 0, exclude non-config args)
        excluded_args = {"command", "func"}  # argparse internal args
        cli_args = {
            k: v
            for k, v in vars(namespace).items()
            if (v or v == 0) and k not in excluded_args
        }
        config.update(cli_args)

        # Create an instance and validate
        instance = cls(**config)._parse_dict_args()
        instance._configure()

        if used_config:
            cli_args.pop("config", None)
            eval_logger.info(
                "CLI args %s will override config file", cli_args
            ) if cli_args else None
            print(textwrap.dedent(f"""{instance}"""))

        return instance

    @classmethod
    def from_config(cls, config_path: str | Path) -> EvaluatorConfig:
        """Build an EvaluationConfig from a TOML or YAML config file.

        Merges with built-in defaults and validates.
        """
        file_config = cls.load_config(config_path)
        return cls(**file_config)._configure()

    @classmethod
    def load_config(
        cls,
        config_path: str | Path,
        *,
        _seen: set[Path] | None = None,
    ) -> dict[str, Any]:
        """Load and validate a TOML or YAML evaluation config file."""
        config_data = cls._load_config_mapping(config_path, _seen=_seen)
        return cls._normalize_versioned_config(config_data)

    @classmethod
    def _load_config_mapping(
        cls,
        config_path: str | Path,
        *,
        _seen: set[Path] | None = None,
    ) -> dict[str, Any]:
        _config_path = Path(config_path).expanduser().resolve()
        if not _config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {_config_path}")
        if _seen is None:
            _seen = set()
        if _config_path in _seen:
            raise ValueError(f"Config include cycle at {_config_path}")
        _seen.add(_config_path)

        try:
            if _config_path.suffix.lower() == ".toml":
                config_data = tomllib.loads(_config_path.read_text(encoding="utf-8"))
            elif _config_path.suffix.lower() in {".yaml", ".yml"}:
                config_data = yaml.safe_load(_config_path.read_text(encoding="utf-8"))
            else:
                raise ValueError(
                    f"Unsupported config format for {_config_path}. "
                    "Use a .toml, .yaml, or .yml file."
                )
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML in {_config_path}: {e}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {_config_path}: {e}") from e
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Could not read config file {_config_path}: {e}") from e

        if not isinstance(config_data, dict):
            raise TypeError(
                f"Config root must be a mapping in {_config_path.resolve()}, "
                f"got {type(config_data).__name__}"
            )

        includes = config_data.pop("include", [])
        if not isinstance(includes, (str, list)):
            raise TypeError("Config include must be a path or list of paths")

        merged: dict[str, Any] = {}
        for include in [includes] if isinstance(includes, str) else includes:
            if not isinstance(include, str):
                raise TypeError("Every config include must be a path string")
            include_path = Path(include)
            if not include_path.is_absolute():
                include_path = _config_path.parent / include_path
            included = cls._load_config_mapping(include_path, _seen=_seen)
            merged = cls._merge_config(merged, included)

        _seen.remove(_config_path)
        return cls._merge_config(merged, config_data)

    @classmethod
    def _normalize_versioned_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Translate the repository's public eval schema to lm-eval arguments."""
        if "schema_version" not in config:
            return config

        unknown = sorted(set(config) - VERSIONED_CONFIG_FIELDS)
        if unknown:
            raise ValueError("Unknown versioned config fields: " + ", ".join(unknown))
        missing = sorted(VERSIONED_REQUIRED_FIELDS - set(config))
        if missing:
            raise ValueError("Missing versioned config fields: " + ", ".join(missing))
        if isinstance(config["schema_version"], bool) or config["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        if config["backend"] != "rwkv7-http":
            raise ValueError("backend must be rwkv7-http for schema_version = 1")

        model_name = cls._non_empty_string(config["model_name"], "model_name")
        base_url = cls._completion_url(config["base_url"])
        benchmarks = cls._string_list(config["benchmarks"], "benchmarks")
        output_dir = cls._non_empty_string(config["output_dir"], "output_dir")
        max_length = cls._positive_int(config["max_length"], "max_length")
        batch_size = cls._positive_int(config.get("batch_size", 1), "batch_size")
        num_concurrent = cls._positive_int(
            config.get("num_concurrent", 5), "num_concurrent"
        )

        profile = config["rwkv_profile"]
        if not isinstance(profile, dict):
            raise TypeError("rwkv_profile must be a table")
        unknown_profile = sorted(set(profile) - RWKV_PROFILE_FIELDS)
        if unknown_profile:
            raise ValueError(
                "Unknown RWKV profile fields: " + ", ".join(unknown_profile)
            )
        missing_profile = sorted(RWKV_PROFILE_FIELDS - set(profile))
        if missing_profile:
            raise ValueError(
                "Missing RWKV profile fields: " + ", ".join(missing_profile)
            )

        prompt_template = cls._non_empty_string(
            profile["prompt_template"], "rwkv_profile.prompt_template"
        )
        if prompt_template not in RWKV_PROMPT_TEMPLATES:
            raise ValueError(
                "rwkv_profile.prompt_template must be one of: "
                + ", ".join(sorted(RWKV_PROMPT_TEMPLATES))
            )
        generation_prompt = cls._non_empty_string(
            profile["generation_prompt"], "rwkv_profile.generation_prompt"
        )
        if generation_prompt not in RWKV_GENERATION_PROMPTS:
            raise ValueError(
                "rwkv_profile.generation_prompt must be one of: "
                + ", ".join(sorted(RWKV_GENERATION_PROMPTS))
            )
        sampling_mode = cls._non_empty_string(
            profile["sampling_mode"], "rwkv_profile.sampling_mode"
        )
        if sampling_mode not in RWKV_SAMPLING_MODES:
            raise ValueError(
                "rwkv_profile.sampling_mode must be one of: "
                + ", ".join(sorted(RWKV_SAMPLING_MODES))
            )
        wkv_mode = cls._non_empty_string(profile["wkv_mode"], "rwkv_profile.wkv_mode")
        if wkv_mode not in RWKV_WKV_MODES:
            raise ValueError(
                "rwkv_profile.wkv_mode must be one of: "
                + ", ".join(sorted(RWKV_WKV_MODES))
            )

        apply_chat_template = cls._boolean(
            config.get("apply_chat_template", True), "apply_chat_template"
        )
        fewshot_as_multiturn = cls._boolean(
            config.get("fewshot_as_multiturn", apply_chat_template),
            "fewshot_as_multiturn",
        )
        if fewshot_as_multiturn and not apply_chat_template:
            raise ValueError("fewshot_as_multiturn requires apply_chat_template = true")
        log_samples = cls._boolean(config.get("log_samples", True), "log_samples")
        include_defaults = cls._boolean(
            config.get("include_defaults", True), "include_defaults"
        )
        device = cls._non_empty_string(config.get("device", "cpu"), "device")
        seed = config.get("seed", [0, 1234, 1234, 1234])
        if (
            not isinstance(seed, list)
            or len(seed) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in seed
            )
        ):
            raise ValueError("seed must contain four integers")

        publication_value = config.get("publication")
        if publication_value is None:
            # Accept the short aliases for hand-written TOMLs while exposing a
            # single normalized field to the execution pipeline.
            publication_value = config.get("scoreboard", config.get("publish", {}))
        if isinstance(publication_value, bool):
            publication_value = {"enabled": publication_value}
        if publication_value is None:
            publication_value = {}
        if not isinstance(publication_value, dict):
            raise TypeError("publication must be a table or boolean")
        unknown_publication = sorted(set(publication_value) - PUBLICATION_FIELDS)
        if unknown_publication:
            raise ValueError(
                "Unknown publication fields: " + ", ".join(unknown_publication)
            )
        publication = dict(publication_value)
        publication["enabled"] = cls._boolean(
            publication.get("enabled", False), "publication.enabled"
        )
        if "base_url" in publication:
            publication["base_url"] = cls._non_empty_string(
                publication["base_url"], "publication.base_url"
            )
            parsed_publication_url = urlsplit(publication["base_url"])
            if (
                parsed_publication_url.scheme not in {"http", "https"}
                or not parsed_publication_url.netloc
                or parsed_publication_url.username is not None
                or parsed_publication_url.password is not None
                or parsed_publication_url.query
                or parsed_publication_url.fragment
            ):
                raise ValueError(
                    "publication.base_url must be an absolute HTTP(S) URL without credentials, query, or fragment"
                )
        if "token_env" in publication:
            publication["token_env"] = cls._non_empty_string(
                publication["token_env"], "publication.token_env"
            )
        if "timeout" in publication:
            timeout = publication["timeout"]
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or timeout <= 0
            ):
                raise ValueError("publication.timeout must be positive")
        if "finalize" in publication:
            publication["finalize"] = cls._boolean(
                publication["finalize"], "publication.finalize"
            )
        if "model_sha256" in publication:
            model_sha256 = cls._non_empty_string(
                publication["model_sha256"], "publication.model_sha256"
            )
            if len(model_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in model_sha256
            ):
                raise ValueError("publication.model_sha256 must be 64 lowercase hex characters")
            publication["model_sha256"] = model_sha256
        if "model_revision" in publication:
            publication["model_revision"] = cls._non_empty_string(
                publication["model_revision"], "publication.model_revision"
            )
        if "task_metadata" in publication:
            task_metadata = publication["task_metadata"]
            if not isinstance(task_metadata, dict):
                raise TypeError("publication.task_metadata must be a table")
            for task_name, value in task_metadata.items():
                if not isinstance(task_name, str) or not task_name.strip():
                    raise ValueError("publication.task_metadata keys must be non-empty strings")
                if not isinstance(value, dict):
                    raise TypeError(
                        f"publication.task_metadata.{task_name} must be a table"
                    )
        if publication["enabled"] and not log_samples:
            raise ValueError(
                "publication.enabled requires log_samples = true so per-sample evidence is retained"
            )

        normalized: dict[str, Any] = {
            "model": "rwkv7-http",
            "tasks": benchmarks,
            "batch_size": batch_size,
            "device": device,
            "apply_chat_template": apply_chat_template,
            "fewshot_as_multiturn": fewshot_as_multiturn,
            "log_samples": log_samples,
            "include_defaults": include_defaults,
            "output_path": output_dir,
            "seed": seed,
            "model_args": {
                "model": model_name,
                "base_url": base_url,
                "rwkv_prompt_template": prompt_template,
                "rwkv_generation_prompt": generation_prompt,
                "rwkv_sampling_mode": sampling_mode,
                "num_concurrent": num_concurrent,
                "max_length": max_length,
            },
            "metadata": {
                "model_name": model_name,
                "wkv_mode": wkv_mode,
                "cot_mode": generation_prompt,
                "prompt_template": prompt_template,
            },
            "publication": publication,
        }
        if publication["enabled"]:
            # The HTTP backend keeps raw responses and token IDs only when
            # evidence recording is enabled.  Do not change legacy configs'
            # model-argument shape unless they opt into publication.
            normalized["model_args"]["record_evidence"] = True
        if "limit" in config:
            limit = config["limit"]
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or limit <= 0
            ):
                raise ValueError("limit must be a positive number")
            normalized["limit"] = limit
        if "samples" in config:
            samples = config["samples"]
            if not isinstance(samples, dict) or not samples:
                raise TypeError("samples must be a non-empty table")
            normalized_samples: dict[str, list[int]] = {}
            for task_name, indices in samples.items():
                if (
                    not isinstance(task_name, str)
                    or not task_name
                    or task_name != task_name.strip()
                ):
                    raise ValueError(
                        "samples task names must be non-empty trimmed strings"
                    )
                if (
                    not isinstance(indices, list)
                    or not indices
                    or any(
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index < 0
                        for index in indices
                    )
                ):
                    raise ValueError(
                        f"samples.{task_name} must be a non-empty array of "
                        "non-negative integers"
                    )
                if len(indices) != len(set(indices)):
                    raise ValueError(f"samples.{task_name} contains duplicate indices")
                normalized_samples[task_name] = indices
            normalized["samples"] = normalized_samples
        if "include_path" in config:
            include_path = config["include_path"]
            if isinstance(include_path, str):
                normalized["include_path"] = cls._non_empty_string(
                    include_path, "include_path"
                )
            else:
                normalized["include_path"] = cls._string_list(
                    include_path, "include_path"
                )
        if not include_defaults and "include_path" not in normalized:
            raise ValueError("include_defaults = false requires include_path")
        if "limit" in normalized and "samples" in normalized:
            raise ValueError("limit and samples are mutually exclusive")
        if "use_cache" in config:
            normalized["use_cache"] = cls._non_empty_string(
                config["use_cache"], "use_cache"
            )
        return normalized

    @staticmethod
    def _non_empty_string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a non-empty trimmed string")
        return value

    @classmethod
    def _completion_url(cls, value: Any) -> str:
        url = cls._non_empty_string(value, "base_url")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/v1/completions"
        ):
            raise ValueError("base_url must be an HTTP(S) /v1/completions endpoint")
        return url

    @staticmethod
    def _string_list(value: Any, name: str) -> list[str]:
        if (
            not isinstance(value, list)
            or not value
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in value
            )
        ):
            raise ValueError(
                f"{name} must be a non-empty array of non-empty trimmed strings"
            )
        duplicates = sorted(item for item in set(value) if value.count(item) > 1)
        if duplicates:
            raise ValueError(f"duplicate {name}: " + ", ".join(duplicates))
        return value

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _boolean(value: Any, name: str) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
        return value

    @staticmethod
    def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = EvaluatorConfig._merge_config(merged[key], value)
            else:
                merged[key] = value
        return merged

    load_yaml_config = load_config

    def _parse_dict_args(self):
        # Parse string arguments that should be dictionaries
        for f in fields(self):
            if f.type is dict and isinstance(getattr(self, f.name), str):
                setattr(self, f.name, simple_parse_args_string(getattr(self, f.name)))
        return self

    def _configure(self):
        """Validate configuration and preprocess fields after creation."""
        self._validate_arguments()._process_arguments()._set_trust_remote_code()

        return self

    def _validate_arguments(self):
        """Validate configuration arguments and cross-field constraints."""
        # tasks are required
        if self.tasks is None:
            raise ValueError("Need to specify task to evaluate.")

        if self.limit:
            eval_logger.warning(
                "--limit SHOULD ONLY BE USED FOR TESTING. "
                "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
            )

        # predict_only implies log_samples
        if self.predict_only:
            self.log_samples = True

        # log_samples or predict_only requires output_path
        if (self.log_samples or self.predict_only) and not self.output_path:
            raise ValueError(
                "Specify --output_path if providing --log_samples or --predict_only"
            )

        # Handle fewshot_as_multiturn logic:
        # - If None and apply_chat_template is set, default to True
        # - If explicitly True, require apply_chat_template
        # - If explicitly False, keep it False
        if self.fewshot_as_multiturn is None and self.apply_chat_template:
            eval_logger.info("Using default fewshot_as_multiturn=True.")
            self.fewshot_as_multiturn = bool(self.apply_chat_template)
        elif self.fewshot_as_multiturn is True and not self.apply_chat_template:
            raise ValueError(
                "When `fewshot_as_multiturn` is True, `apply_chat_template` must be set."
            )

        # samples and limit are mutually exclusive
        if self.samples and self.limit is not None:
            raise ValueError("If --samples is not None, then --limit must be None.")

        return self

    def _process_arguments(self):
        """Process samples argument - load from a file if needed."""
        if self.samples:
            if isinstance(self.samples, dict):
                self.samples = self.samples
            elif isinstance(self.samples, str):
                try:
                    self.samples = json.loads(self.samples)
                except json.JSONDecodeError:
                    if (samples_path := Path(cast("str", self.samples))).is_file():
                        self.samples = json.loads(samples_path.read_text())

        # Set up metadata by merging model_args and metadata.
        if self.model_args is None:
            self.model_args = {}
        if self.metadata is None:
            self.metadata = {}

        self.metadata = self.model_args | self.metadata

        return self

    def process_tasks(self, metadata: dict | None = None) -> TaskManager:
        """Process and validate tasks, return resolved task names.

        Handles:
        - Task names (e.g., "hellaswag", "arc_easy")
        - Custom YAML config files (e.g., "/path/to/task.yaml")
        - Glob patterns (e.g., "/path/to/*.yaml")
        - Directories of YAML files
        """
        import glob
        import itertools

        from lm_eval.tasks import TaskManager
        from lm_eval.tasks._yaml_loader import load_yaml

        # if metadata manually passed use that:
        self.metadata = metadata or self.metadata

        task_manager = TaskManager(
            include_path=self.include_path,
            include_defaults=self.include_defaults,
            metadata=self.metadata or {},
        )

        # Normalize tasks to a list
        # We still allow tasks in the form task1,task2
        task_list = (
            self.tasks.split(",")
            if isinstance(self.tasks, str)
            else [t for task in self.tasks for t in task.split(",")]
        )

        # Handle directory input
        if len(task_list) == 1 and Path(task_list[0]).is_dir():
            task_names = []
            yaml_path = Path(task_list[0]) / "*.yaml"
            for yaml_file in glob.glob(str(yaml_path)):
                config = load_yaml(yaml_file, resolve_func=False)
                task_names.append(config)
            self.tasks = task_names
            return task_manager

        import lm_eval.models  # noqa: F401
        from lm_eval.api.registry import get_model

        model_cls = get_model(self.model)
        task_adapter = getattr(model_cls, "TASK_ADAPTER", None)
        if task_adapter is not None:
            task_list = task_manager.resolve_adapter_tasks(task_list, task_adapter)

        # Normalize paths and deduplicate
        task_list = [
            str(Path(task).absolute()) if task.endswith(".yaml") else task
            for task in task_list
        ]
        match_dict = dict.fromkeys(task_list)  # deduplicate file paths

        # Match each task
        for task in match_dict:
            if not task.endswith(".yaml"):
                # Standard task name - match via task manager
                matches = task_manager.match_tasks([task])
            else:
                # Custom config file(s) - support glob patterns
                matches = []
                for yaml_file in glob.glob(task):
                    config = load_yaml(yaml_file, resolve_func=False)
                    matches.append(config)
            match_dict[task] = matches

        # Flatten and deduplicate results
        task_names = []
        for task in itertools.chain.from_iterable(match_dict.values()):
            if task not in task_names:
                task_names.append(task)

        # Check for missing tasks
        task_missing = [task for task, matches in match_dict.items() if not matches]
        if task_missing:
            missing = ", ".join(task_missing)
            raise ValueError(f"Tasks not found: {missing}")

        # Update tasks with resolved names
        self.tasks = task_names
        return task_manager

    def _set_trust_remote_code(self):
        """Apply the trust_remote_code setting if enabled."""
        if self.trust_remote_code:
            # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
            # because it's already been determined based on the prior env var before launching our
            # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
            import datasets

            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

            # Add to model_args for the actual model initialization
            if self.model_args is None:
                self.model_args = {}
            self.model_args["trust_remote_code"] = True

        return self
