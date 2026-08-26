from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import random
import socket
import subprocess
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable, Protocol, Sequence


INTERNAL_WEIGHT = "__selection_weight__"
APP_VERSION = "0.11.0"
_STARTED_AT = time.monotonic()
_EVENT_OUTPUT_ENABLED = True


def _require(module: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"Install {install_hint} to use this stage") from exc


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dependency_versions() -> dict[str, str]:
    packages = (
        "catboost",
        "numpy",
        "nvidia-ml-py",
        "polars",
        "psutil",
        "pyarrow",
        "PyYAML",
        "ray",
        "scikit-learn",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _event(name: str, **details: Any) -> None:
    if not _EVENT_OUTPUT_ENABLED:
        return
    suffix = " | ".join(
        f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in details.items()
    )
    message = f"[{time.monotonic() - _STARTED_AT:8.1f}s] {name.replace('_', ' ').upper()}"
    print(f"{message} | {suffix}" if suffix else message, flush=True)


def _set_event_output(enabled: bool) -> None:
    global _EVENT_OUTPUT_ENABLED
    _EVENT_OUTPUT_ENABLED = enabled


def _write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    Path(path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _clamp_subset(
    subset: Iterable[str],
    universe: Sequence[str],
    minimum: int,
    maximum: int,
    rng: random.Random,
    required: Iterable[str] = (),
) -> frozenset[str]:
    universe_set = set(universe)
    required_set = set(required) & universe_set
    if len(required_set) > maximum:
        raise ValueError("The required feature count exceeds max_features")
    selected = (set(subset) & universe_set) | required_set
    if len(selected) > maximum:
        optional = sorted(selected - required_set)
        selected = required_set | set(
            rng.sample(optional, maximum - len(required_set))
        )
    missing = [feature for feature in universe if feature not in selected]
    if len(selected) < minimum:
        selected.update(rng.sample(missing, min(minimum - len(selected), len(missing))))
    return frozenset(selected)


@dataclass(slots=True)
class DataConfig:
    train_path: str
    valid_path: str
    target: str
    oos_path: str | None = None
    oot_path: str | None = None
    test_path: str | None = None
    positive_label: Any = 1
    id_columns: list[str] = field(default_factory=list)
    sampling_key_columns: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    categorical_null_strategy: str = "fill"
    categorical_null_value: str = "__MISSING__"
    excluded_features: list[str] = field(default_factory=list)
    required_features: list[str] = field(default_factory=list)
    leakage_key_columns: list[str] = field(default_factory=list)
    bootstrap_key_columns: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SampleConfig:
    screen_train_positive_fraction: float = 1.0
    screen_train_negative_fraction: float = 0.05
    screen_valid_positive_fraction: float = 1.0
    screen_valid_negative_fraction: float = 0.10
    train_positive_fraction: float = 1.0
    train_negative_fraction: float = 0.15
    valid_positive_fraction: float = 1.0
    valid_negative_fraction: float = 0.30
    confirm_train_positive_fraction: float = 1.0
    confirm_train_negative_fraction: float = 0.50
    confirm_valid_positive_fraction: float = 1.0
    confirm_valid_negative_fraction: float = 1.0
    oos_positive_fraction: float = 1.0
    oos_negative_fraction: float = 1.0
    oot_positive_fraction: float = 1.0
    oot_negative_fraction: float = 1.0
    test_positive_fraction: float = 1.0
    test_negative_fraction: float = 1.0
    seed: int = 42


@dataclass(slots=True)
class ValidationConfig:
    check_split_schema: bool = True
    check_id_overlap: bool = False
    fail_on_id_overlap: bool = True


@dataclass(slots=True)
class FidelityConfig:
    name: str
    train_sample: str
    valid_sample: str
    iterations: int
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass(slots=True)
class SearchConfig:
    primary_metric: str = "average_precision"
    min_features: int = 10
    max_features: int = 100
    search_max_features: int | None = None
    random_subspaces: int = 60
    adaptive_subspaces: int = 40
    subspace_min_features: int = 20
    subspace_max_features: int = 150
    elite_fraction: float = 0.20
    population_size: int = 36
    generations: int = 15
    mutation_rate: float = 0.75
    crossover_rate: float = 0.80
    interaction_pairs: int = 100
    interaction_mutation_probability: float = 0.35
    coalition_top_pairs: int = 200
    coalition_top_triples: int = 40
    coalition_min_support: int = 2
    coalition_smoothing: float = 2.0
    coalition_pairs_per_subset: int = 2_000
    coalition_triples_per_subset: int = 200
    coalition_sampling_probability: float = 0.45
    coalition_mutation_probability: float = 0.35
    coalition_preservation_probability: float = 0.65
    coalition_refresh_interval: int = 3
    coalition_history_limit: int = 300
    interaction_prior_weight: float = 0.50
    local_candidates: int = 5
    local_rounds: int = 2
    max_primary_metric_gap: float | None = None
    max_gini_gap: float = 0.15
    # Backward-compatible alias for older configs; maps to primary-metric gap.
    max_gap: float | None = None
    allowed_metric_drop: float = 0.002
    successive_halving_enabled: bool = False
    promotion_fraction: float = 0.35
    min_promoted: int = 4
    promotion_strategy: str = "fixed"
    promotion_min_fraction: float = 0.20
    promotion_max_fraction: float = 0.60
    promotion_metric_band: float = 0.005
    restarts: int = 2
    seed: int = 42


@dataclass(slots=True)
class OutputConfig:
    directory: str = "interaction_selection_results"
    cache_directory: str = "interaction_selection_results/cache"
    save_trial_predictions: bool = False


@dataclass(slots=True)
class RobustValidationConfig:
    enabled: bool = True
    require_oos: bool = False
    evaluate_oot: bool = True
    evaluate_test: bool = True
    bootstrap_repeats: int = 500
    confidence_level: float = 0.95
    bootstrap_max_rows: int = 50_000
    bootstrap_seed: int = 2026
    bootstrap_mode: str = "auto"
    noninferiority_margin: float | None = None


@dataclass(slots=True)
class DecisionThresholdConfig:
    enabled: bool = False
    min_recall: float = 0.50
    min_precision: float = 0.20
    objective: str = "max_precision"
    require_oos_feasible: bool = True
    enforce_during_search: bool = True
    enforce_fidelities: list[str] = field(default_factory=lambda: ["search", "confirm"])


@dataclass(slots=True)
class ExecutionConfig:
    backend: str = "local"
    local_trial_mode: str = "process"
    process_start_method: str = "forkserver"
    parallel_trials: int = 1
    threads_per_trial: int = 0
    gpu_devices: list[str] = field(default_factory=list)
    trial_max_retries: int = 1
    trial_failure_policy: str = "continue"
    minimum_successful_fraction: float = 0.75
    retry_backoff_seconds: float = 0.0
    trial_timeout_seconds: float = 3_600.0
    heartbeat_interval_seconds: float = 2.0
    heartbeat_timeout_seconds: float = 30.0
    terminate_grace_seconds: float = 5.0
    show_progress: bool = True
    progress_interval_seconds: float = 10.0
    ray_address: str | None = None
    ray_namespace: str = "interaction-selector"
    ray_num_cpus_per_trial: float = 1.0
    ray_num_gpus_per_trial: float = 0.0
    ray_runtime_env: dict[str, Any] = field(default_factory=dict)
    checkpoint_enabled: bool = True
    resume_search: bool = True


@dataclass(slots=True)
class ResourceConfig:
    enabled: bool = True
    mode: str = "hybrid"
    cpu_cores: int | None = None
    reserve_cpu_cores: int = 2
    max_cpu_utilization: float = 0.85
    ram_gb: float | None = None
    reserve_ram_gb: float = 4.0
    max_ram_utilization: float = 0.80
    gpu_count: int | None = None
    gpu_total_memory_gb: float | None = None
    gpu_memory_by_device_gb: dict[str, float] = field(default_factory=dict)
    reserve_gpu_memory_gb: float = 2.0
    max_gpu_memory_utilization: float = 0.85
    adaptive_concurrency: bool = True
    monitoring_interval_seconds: float = 1.0
    resource_wait_timeout_seconds: float = 300.0
    calibration_enabled: bool = True
    calibration_safety_factor: float = 1.30
    estimated_ram_per_trial_gb: float | None = None
    estimated_gpu_memory_per_trial_gb: float | None = None
    hard_ram_per_trial_gb: float | None = None
    hard_gpu_memory_per_trial_gb: float | None = None
    hard_limit_multiplier: float = 2.0
    profile_learning_enabled: bool = True
    oom_concurrency_reduction: float = 0.50


@dataclass(slots=True)
class AppConfig:
    data: DataConfig
    sample: SampleConfig = field(default_factory=SampleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    robust_validation: RobustValidationConfig = field(
        default_factory=RobustValidationConfig
    )
    decision_threshold: DecisionThresholdConfig = field(
        default_factory=DecisionThresholdConfig
    )
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        if not isinstance(raw, dict):
            raise TypeError("Selector config must be a YAML mapping")
        allowed_sections = {
            "data",
            "sample",
            "validation",
            "search",
            "robust_validation",
            "decision_threshold",
            "execution",
            "resources",
            "output",
            "model_params",
        }
        unknown_sections = sorted(set(raw) - allowed_sections)
        if unknown_sections:
            raise ValueError(
                "Unknown selector config sections: "
                f"{unknown_sections}. The legacy 'audit' section was replaced "
                "by 'validation' in selector 0.10.0."
            )
        if "data" not in raw:
            raise ValueError("Selector config must contain a 'data' section")
        return cls(
            data=DataConfig(**raw["data"]),
            sample=SampleConfig(**raw.get("sample", {})),
            validation=ValidationConfig(**raw.get("validation", {})),
            search=SearchConfig(**raw.get("search", {})),
            robust_validation=RobustValidationConfig(
                **raw.get("robust_validation", {})
            ),
            decision_threshold=DecisionThresholdConfig(
                **raw.get("decision_threshold", {})
            ),
            execution=ExecutionConfig(**raw.get("execution", {})),
            resources=ResourceConfig(**raw.get("resources", {})),
            output=OutputConfig(**raw.get("output", {})),
            model_params=raw.get("model_params", {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        yaml = _require("yaml", "pyyaml")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    def validate(self) -> None:
        supported_metrics = {"average_precision", "pr_auc", "roc_auc", "auc", "gini", "logloss", "brier"}
        if self.search.primary_metric not in supported_metrics:
            raise ValueError(
                f"Unsupported primary_metric={self.search.primary_metric!r}; "
                f"choose one of {sorted(supported_metrics)}"
            )
        fractions = [
            self.sample.screen_train_positive_fraction,
            self.sample.screen_train_negative_fraction,
            self.sample.screen_valid_positive_fraction,
            self.sample.screen_valid_negative_fraction,
            self.sample.train_positive_fraction,
            self.sample.train_negative_fraction,
            self.sample.valid_positive_fraction,
            self.sample.valid_negative_fraction,
            self.sample.confirm_train_positive_fraction,
            self.sample.confirm_train_negative_fraction,
            self.sample.confirm_valid_positive_fraction,
            self.sample.confirm_valid_negative_fraction,
            self.sample.oos_positive_fraction,
            self.sample.oos_negative_fraction,
            self.sample.oot_positive_fraction,
            self.sample.oot_negative_fraction,
            self.sample.test_positive_fraction,
            self.sample.test_negative_fraction,
        ]
        if not all(0 < fraction <= 1 for fraction in fractions):
            raise ValueError("All sampling fractions must be in (0, 1]")
        if not self.data.sampling_key_columns:
            raise ValueError("sampling_key_columns are required for deterministic large-data sampling")
        if self.data.categorical_null_strategy not in {"fill", "error"}:
            raise ValueError(
                "data.categorical_null_strategy must be 'fill' or 'error'"
            )
        if (
            self.data.categorical_null_strategy == "fill"
            and not self.data.categorical_null_value
        ):
            raise ValueError(
                "data.categorical_null_value must be a non-empty string "
                "when categorical_null_strategy='fill'"
            )
        if self.search.min_features < 1:
            raise ValueError("min_features must be positive")
        if self.search.max_features < self.search.min_features:
            raise ValueError("max_features must be >= min_features")
        if self.search.search_max_features is None:
            self.search.search_max_features = self.search.max_features
        if self.search.search_max_features < self.search.max_features:
            raise ValueError("search_max_features must be >= max_features")
        if not 0 < self.search.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        probabilities = {
            "mutation_rate": self.search.mutation_rate,
            "crossover_rate": self.search.crossover_rate,
            "interaction_mutation_probability": self.search.interaction_mutation_probability,
            "coalition_sampling_probability": self.search.coalition_sampling_probability,
            "coalition_mutation_probability": self.search.coalition_mutation_probability,
            "coalition_preservation_probability": self.search.coalition_preservation_probability,
        }
        if not all(0 <= value <= 1 for value in probabilities.values()):
            raise ValueError(f"Search probabilities must be in [0, 1]: {probabilities}")
        if self.search.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.search.generations < 1 or self.search.restarts < 1:
            raise ValueError("generations and restarts must be positive")
        if self.search.allowed_metric_drop < 0:
            raise ValueError("allowed_metric_drop must be non-negative")
        if self.search.max_primary_metric_gap is None and self.search.max_gap is not None:
            self.search.max_primary_metric_gap = self.search.max_gap
        if (
            self.search.max_primary_metric_gap is not None
            and self.search.max_primary_metric_gap < 0
        ):
            raise ValueError("max_primary_metric_gap must be non-negative or null")
        if self.search.max_gap is not None and self.search.max_gap < 0:
            raise ValueError("max_gap must be non-negative or null")
        if self.search.max_gini_gap < 0:
            raise ValueError("max_gini_gap must be non-negative")
        if self.search.subspace_min_features < 1:
            raise ValueError("subspace_min_features must be positive")
        if self.search.subspace_max_features < self.search.subspace_min_features:
            raise ValueError("subspace_max_features must be >= subspace_min_features")
        coalition_counts = {
            "coalition_top_pairs": self.search.coalition_top_pairs,
            "coalition_top_triples": self.search.coalition_top_triples,
            "coalition_min_support": self.search.coalition_min_support,
            "coalition_pairs_per_subset": self.search.coalition_pairs_per_subset,
            "coalition_triples_per_subset": self.search.coalition_triples_per_subset,
            "coalition_refresh_interval": self.search.coalition_refresh_interval,
            "coalition_history_limit": self.search.coalition_history_limit,
        }
        if any(value < 0 for value in coalition_counts.values()):
            raise ValueError(f"Coalition counts must be non-negative: {coalition_counts}")
        if self.search.coalition_min_support < 1:
            raise ValueError("coalition_min_support must be positive")
        if self.search.coalition_refresh_interval < 1:
            raise ValueError("coalition_refresh_interval must be positive")
        if self.search.coalition_history_limit < 2:
            raise ValueError("coalition_history_limit must be at least 2")
        if self.search.coalition_smoothing <= 0:
            raise ValueError("coalition_smoothing must be positive")
        if self.search.interaction_prior_weight < 0:
            raise ValueError("interaction_prior_weight must be non-negative")
        if not 0 < self.search.promotion_fraction <= 1:
            raise ValueError("promotion_fraction must be in (0, 1]")
        if self.search.min_promoted < 1:
            raise ValueError("min_promoted must be positive")
        if self.search.promotion_strategy not in {"fixed", "adaptive"}:
            raise ValueError("promotion_strategy must be fixed or adaptive")
        if not (
            0 < self.search.promotion_min_fraction
            <= self.search.promotion_max_fraction
            <= 1
        ):
            raise ValueError(
                "promotion fractions must satisfy "
                "0 < promotion_min_fraction <= promotion_max_fraction <= 1"
            )
        if self.search.promotion_metric_band < 0:
            raise ValueError("promotion_metric_band must be non-negative")
        fidelity_iterations = {
            "screen_iterations": int(self.model_params.get("screen_iterations", 120)),
            "search_iterations": int(self.model_params.get("search_iterations", 350)),
            "confirm_iterations": int(self.model_params.get("confirm_iterations", 1000)),
        }
        if any(value < 1 for value in fidelity_iterations.values()):
            raise ValueError(
                f"Fidelity iterations must be positive: {fidelity_iterations}"
            )
        if self.search.successive_halving_enabled and not (
            fidelity_iterations["screen_iterations"]
            <= fidelity_iterations["search_iterations"]
            <= fidelity_iterations["confirm_iterations"]
        ):
            raise ValueError(
                "Successive-halving iterations must satisfy "
                "screen_iterations <= search_iterations <= confirm_iterations"
            )
        robust = self.robust_validation
        if robust.require_oos and not self.data.oos_path:
            raise ValueError("robust_validation.require_oos=true but data.oos_path is absent")
        if robust.bootstrap_repeats < 50:
            raise ValueError("bootstrap_repeats must be at least 50")
        if not 0.5 < robust.confidence_level < 1:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if robust.bootstrap_max_rows < 100:
            raise ValueError("bootstrap_max_rows must be at least 100")
        if robust.bootstrap_mode not in {"auto", "row", "cluster"}:
            raise ValueError("bootstrap_mode must be auto, row, or cluster")
        if robust.bootstrap_mode == "cluster" and not self.data.bootstrap_key_columns:
            raise ValueError(
                "bootstrap_mode=cluster requires data.bootstrap_key_columns"
            )
        if (
            robust.noninferiority_margin is not None
            and robust.noninferiority_margin < 0
        ):
            raise ValueError("noninferiority_margin must be non-negative")
        threshold = self.decision_threshold
        if not 0 <= threshold.min_recall <= 1:
            raise ValueError("decision_threshold.min_recall must be in [0, 1]")
        if not 0 <= threshold.min_precision <= 1:
            raise ValueError("decision_threshold.min_precision must be in [0, 1]")
        if threshold.objective not in {"max_precision", "max_f1"}:
            raise ValueError("decision_threshold.objective must be max_precision or max_f1")
        invalid_fidelities = set(threshold.enforce_fidelities) - {"screen", "search", "confirm"}
        if invalid_fidelities:
            raise ValueError(
                "decision_threshold.enforce_fidelities contains unsupported values: "
                f"{sorted(invalid_fidelities)}"
            )
        execution = self.execution
        if execution.backend not in {"local", "ray"}:
            raise ValueError("execution.backend must be local or ray")
        if execution.local_trial_mode not in {"process", "thread"}:
            raise ValueError(
                "execution.local_trial_mode must be process or thread"
            )
        if execution.process_start_method not in {"spawn", "forkserver"}:
            raise ValueError(
                "execution.process_start_method must be spawn or forkserver"
            )
        if execution.parallel_trials < 0:
            raise ValueError("execution.parallel_trials must be non-negative")
        if execution.parallel_trials == 0 and not self.resources.enabled:
            raise ValueError(
                "execution.parallel_trials=0 requires resources.enabled=true"
            )
        if execution.threads_per_trial < 0:
            raise ValueError("execution.threads_per_trial must be non-negative")
        if execution.trial_max_retries < 0:
            raise ValueError("execution.trial_max_retries must be non-negative")
        if execution.trial_failure_policy not in {"continue", "fail_fast"}:
            raise ValueError(
                "execution.trial_failure_policy must be continue or fail_fast"
            )
        if not 0 < execution.minimum_successful_fraction <= 1:
            raise ValueError(
                "execution.minimum_successful_fraction must be in (0, 1]"
            )
        if execution.retry_backoff_seconds < 0:
            raise ValueError("execution.retry_backoff_seconds must be non-negative")
        for name, value in {
            "trial_timeout_seconds": execution.trial_timeout_seconds,
            "heartbeat_interval_seconds": execution.heartbeat_interval_seconds,
            "heartbeat_timeout_seconds": execution.heartbeat_timeout_seconds,
            "terminate_grace_seconds": execution.terminate_grace_seconds,
            "progress_interval_seconds": execution.progress_interval_seconds,
        }.items():
            if value <= 0:
                raise ValueError(f"execution.{name} must be positive")
        if (
            execution.heartbeat_timeout_seconds
            <= execution.heartbeat_interval_seconds
        ):
            raise ValueError(
                "heartbeat_timeout_seconds must exceed heartbeat_interval_seconds"
            )
        if execution.ray_num_cpus_per_trial <= 0:
            raise ValueError("execution.ray_num_cpus_per_trial must be positive")
        if execution.ray_num_gpus_per_trial < 0:
            raise ValueError(
                "execution.ray_num_gpus_per_trial must be non-negative"
            )
        gpu_devices = [str(device) for device in execution.gpu_devices]
        if len(gpu_devices) != len(set(gpu_devices)):
            raise ValueError("execution.gpu_devices must be unique")
        task_type = str(self.model_params.get("task_type", "CPU")).upper()
        if gpu_devices and task_type != "GPU":
            raise ValueError("execution.gpu_devices requires model_params.task_type=GPU")
        if execution.backend == "ray" and gpu_devices:
            raise ValueError(
                "Ray assigns CUDA devices; leave execution.gpu_devices empty"
            )
        if (
            execution.backend == "ray"
            and task_type == "GPU"
            and execution.ray_num_gpus_per_trial <= 0
        ):
            raise ValueError(
                "Ray GPU trials require execution.ray_num_gpus_per_trial > 0"
            )
        if (
            execution.backend == "ray"
            and task_type != "GPU"
            and execution.ray_num_gpus_per_trial > 0
        ):
            raise ValueError(
                "Ray CPU trials require execution.ray_num_gpus_per_trial=0"
            )
        if (
            execution.backend == "local"
            and task_type == "GPU"
            and execution.parallel_trials > 1
        ):
            if len(gpu_devices) < execution.parallel_trials:
                raise ValueError(
                    "Parallel GPU trials require at least one distinct "
                    "execution.gpu_devices entry per worker"
                )
        resources = self.resources
        if resources.mode not in {"auto", "manual", "hybrid"}:
            raise ValueError("resources.mode must be auto, manual, or hybrid")
        if resources.cpu_cores is not None and resources.cpu_cores < 1:
            raise ValueError("resources.cpu_cores must be positive")
        if resources.reserve_cpu_cores < 0:
            raise ValueError("resources.reserve_cpu_cores must be non-negative")
        if resources.ram_gb is not None and resources.ram_gb <= 0:
            raise ValueError("resources.ram_gb must be positive")
        if resources.reserve_ram_gb < 0:
            raise ValueError("resources.reserve_ram_gb must be non-negative")
        if resources.gpu_count is not None and resources.gpu_count < 0:
            raise ValueError("resources.gpu_count must be non-negative")
        if (
            resources.gpu_total_memory_gb is not None
            and resources.gpu_total_memory_gb <= 0
        ):
            raise ValueError("resources.gpu_total_memory_gb must be positive")
        if (
            resources.gpu_total_memory_gb is not None
            and not resources.gpu_count
        ):
            raise ValueError(
                "resources.gpu_total_memory_gb requires a positive gpu_count"
            )
        if any(
            value <= 0 for value in resources.gpu_memory_by_device_gb.values()
        ):
            raise ValueError(
                "resources.gpu_memory_by_device_gb values must be positive"
            )
        for name, value in {
            "max_cpu_utilization": resources.max_cpu_utilization,
            "max_ram_utilization": resources.max_ram_utilization,
            "max_gpu_memory_utilization": resources.max_gpu_memory_utilization,
        }.items():
            if not 0 < value <= 1:
                raise ValueError(f"resources.{name} must be in (0, 1]")
        if resources.monitoring_interval_seconds <= 0:
            raise ValueError(
                "resources.monitoring_interval_seconds must be positive"
            )
        if resources.resource_wait_timeout_seconds <= 0:
            raise ValueError(
                "resources.resource_wait_timeout_seconds must be positive"
            )
        if resources.calibration_safety_factor < 1:
            raise ValueError(
                "resources.calibration_safety_factor must be at least 1"
            )
        if (
            resources.estimated_ram_per_trial_gb is not None
            and resources.estimated_ram_per_trial_gb <= 0
        ):
            raise ValueError(
                "resources.estimated_ram_per_trial_gb must be positive"
            )
        if (
            resources.estimated_gpu_memory_per_trial_gb is not None
            and resources.estimated_gpu_memory_per_trial_gb <= 0
        ):
            raise ValueError(
                "resources.estimated_gpu_memory_per_trial_gb must be positive"
            )
        for name, value in {
            "hard_ram_per_trial_gb": resources.hard_ram_per_trial_gb,
            "hard_gpu_memory_per_trial_gb": (
                resources.hard_gpu_memory_per_trial_gb
            ),
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"resources.{name} must be positive")
        if resources.hard_limit_multiplier <= 1:
            raise ValueError(
                "resources.hard_limit_multiplier must exceed 1"
            )
        if not 0 < resources.oom_concurrency_reduction < 1:
            raise ValueError(
                "resources.oom_concurrency_reduction must be in (0, 1)"
            )
        if resources.mode == "manual":
            if resources.cpu_cores is None or resources.ram_gb is None:
                raise ValueError(
                    "resources.mode=manual requires cpu_cores and ram_gb"
                )
            if task_type == "GPU" and not (
                resources.gpu_memory_by_device_gb
                or (
                    resources.gpu_count
                    and resources.gpu_total_memory_gb
                )
            ):
                raise ValueError(
                    "Manual GPU mode requires per-device memory or both "
                    "gpu_count and gpu_total_memory_gb"
                )


@dataclass(slots=True)
class Evaluation:
    subset: tuple[str, ...]
    fidelity: str
    valid_metric: float
    train_metric: float
    gap: float
    metric_std: float
    n_features: int
    runtime_seconds: float
    metrics: dict[str, float]
    constraint_violation: float = 0.0

    @property
    def key(self) -> str:
        return _stable_hash({"subset": self.subset, "fidelity": self.fidelity})


@dataclass(slots=True)
class CoalitionScore:
    members: tuple[str, ...]
    order: int
    support: int
    elite_support: int
    lift: float
    mean_metric: float
    metric_gain: float
    score: float
    source: str = "observed"


@dataclass(slots=True)
class SplitPredictions:
    split: str
    y_true: Any
    prediction: Any
    weight: Any
    seed_metrics: list[float]
    groups: Any | None = None


@dataclass(slots=True)
class RobustCandidate:
    subset: tuple[str, ...]
    split: str
    n_features: int
    point_metric: float
    ci_low: float
    ci_high: float
    bootstrap_std: float
    seed_metric_std: float
    n_rows: int
    bootstrap_rows: int
    bootstrap_unit: str
    n_clusters: int | None
    bootstrap_clusters: int | None
    reference_subset: tuple[str, ...]
    delta_to_reference: float
    delta_ci_low: float
    delta_ci_high: float
    eligible_for_selection: bool
    noninferior: bool


@dataclass(slots=True)
class ThresholdResult:
    subset: tuple[str, ...]
    threshold: float
    objective: str
    tuning_split: str
    tuning_precision: float
    tuning_recall: float
    tuning_f1: float
    tuning_feasible: bool
    evaluation_split: str | None
    evaluation_precision: float | None
    evaluation_recall: float | None
    evaluation_f1: float | None
    predicted_positive_rate: float | None
    evaluation_feasible: bool | None


@dataclass(slots=True)
class TrialDiagnostic:
    subset: tuple[str, ...]
    fidelity: str
    backend: str
    outcome: str
    attempts: int
    error_type: str
    error_message: str
    oom_detected: bool = False
    concurrency_limit: int | None = None
    termination_reason: str | None = None
    worker_pid: int | None = None
    peak_worker_rss_gb: float = 0.0
    peak_worker_gpu_gb: float = 0.0


@dataclass(slots=True)
class ResourceProfile:
    fidelity: str
    observations: int = 0
    peak_rss_gb: float = 0.0
    peak_gpu_gb: float = 0.0
    maximum_subset_size: int = 0
    maximum_runtime_seconds: float = 0.0


@dataclass(slots=True)
class PromotionBatch:
    batch: int
    strategy: str
    screened: int
    promoted: int
    actual_fraction: float
    pareto_candidates: int
    near_best_candidates: int
    metric_band: float


@dataclass(slots=True)
class GPUResource:
    device: str
    total_gb: float
    free_gb: float
    utilization: float
    source: str


@dataclass(slots=True)
class ResourceSnapshot:
    timestamp: float
    cpu_logical: int
    cpu_physical: int
    cpu_effective: int
    cpu_percent: float
    ram_total_gb: float
    ram_available_gb: float
    ram_percent: float
    process_rss_gb: float
    gpus: list[GPUResource]
    source: str


@dataclass(slots=True)
class TrialResourceEstimate:
    subset_size: int
    fidelity: str
    runtime_seconds: float
    ram_increment_gb: float
    gpu_increment_gb: float
    safety_factor: float
    ram_per_trial_gb: float
    gpu_memory_per_trial_gb: float


def _optional_import(module: str) -> Any | None:
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def _read_text_if_exists(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _cgroup_cpu_limit() -> int | None:
    cpu_max = _read_text_if_exists("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, period = cpu_max.split()[:2]
        if quota != "max" and float(period) > 0:
            return max(1, math.floor(float(quota) / float(period)))
    quota = _read_text_if_exists(
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    )
    period = _read_text_if_exists(
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    )
    if quota and period and float(quota) > 0 and float(period) > 0:
        return max(1, math.floor(float(quota) / float(period)))
    return None


def _cgroup_memory() -> tuple[int, int] | None:
    maximum = _read_text_if_exists("/sys/fs/cgroup/memory.max")
    current = _read_text_if_exists("/sys/fs/cgroup/memory.current")
    if maximum and current and maximum != "max":
        return int(maximum), max(0, int(maximum) - int(current))
    maximum = _read_text_if_exists(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    current = _read_text_if_exists(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )
    if maximum and current:
        limit = int(maximum)
        if limit < 1 << 60:
            return limit, max(0, limit - int(current))
    return None


def _detect_gpus() -> list[GPUResource]:
    gib = float(1024**3)
    pynvml = _optional_import("pynvml")
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            result = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                result.append(
                    GPUResource(
                        device=str(index),
                        total_gb=float(memory.total) / gib,
                        free_gb=float(memory.free) / gib,
                        utilization=float(utilization.gpu) / 100.0,
                        source="nvml",
                    )
                )
            pynvml.nvmlShutdown()
            return result
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    result = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        device, total_mib, free_mib, utilization = parts
        result.append(
            GPUResource(
                device=device,
                total_gb=float(total_mib) / 1024.0,
                free_gb=float(free_mib) / 1024.0,
                utilization=float(utilization) / 100.0,
                source="nvidia-smi",
            )
        )
    return result


def detect_resources() -> ResourceSnapshot:
    gib = float(1024**3)
    psutil = _optional_import("psutil")
    affinity_count = None
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    logical = int(
        affinity_count
        or (psutil.cpu_count(logical=True) if psutil else os.cpu_count())
        or 1
    )
    physical = int(
        (psutil.cpu_count(logical=False) if psutil else None) or logical
    )
    quota = _cgroup_cpu_limit()
    effective = min(logical, quota) if quota else logical
    if psutil:
        memory = psutil.virtual_memory()
        total_bytes = int(memory.total)
        available_bytes = int(memory.available)
        cpu_percent = float(psutil.cpu_percent(interval=None))
        try:
            process_rss = float(
                psutil.Process(os.getpid()).memory_info().rss
            ) / gib
        except Exception:
            process_rss = _process_tree_rss_gb(os.getpid())
        source = "psutil"
    else:
        total_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        available_bytes = total_bytes
        cpu_percent = 0.0
        process_rss = 0.0
        source = "stdlib"
    cgroup_memory = _cgroup_memory()
    if cgroup_memory and cgroup_memory[0] < total_bytes:
        total_bytes, available_bytes = cgroup_memory
        source += "+cgroup"
    total_gb = total_bytes / gib
    available_gb = min(available_bytes / gib, total_gb)
    return ResourceSnapshot(
        timestamp=time.time(),
        cpu_logical=logical,
        cpu_physical=physical,
        cpu_effective=max(1, effective),
        cpu_percent=cpu_percent,
        ram_total_gb=total_gb,
        ram_available_gb=available_gb,
        ram_percent=(1.0 - available_gb / total_gb) if total_gb else 0.0,
        process_rss_gb=process_rss,
        gpus=_detect_gpus(),
        source=source,
    )


def _is_oom_error(exc: BaseException | str) -> bool:
    text_value = str(exc).lower()
    markers = (
        "out of memory",
        "cuda_error_out_of_memory",
        "cannot allocate memory",
        "std::bad_alloc",
        "bad allocation",
        "memoryerror",
        "not enough memory",
    )
    return any(marker in text_value for marker in markers)


def _process_tree_rss_gb(pid: int) -> float:
    gib = float(1024**3)
    psutil = _optional_import("psutil")
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            processes = [process, *process.children(recursive=True)]
            return sum(item.memory_info().rss for item in processes) / gib
        except Exception:
            pass
    status = _read_text_if_exists(f"/proc/{pid}/status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            try:
                return float(line.split()[1]) * 1024.0 / gib
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def _process_memory_telemetry_available() -> bool:
    psutil = _optional_import("psutil")
    if psutil is not None:
        try:
            psutil.Process(os.getpid()).memory_info().rss
            return True
        except Exception:
            pass
    return Path(f"/proc/{os.getpid()}/status").exists()


def _process_gpu_memory_gb(pid: int) -> float:
    gib = float(1024**3)
    pynvml = _optional_import("pynvml")
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            used = 0
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                processes = []
                for getter_name in (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    getter = getattr(pynvml, getter_name, None)
                    if getter is not None:
                        try:
                            processes.extend(getter(handle))
                        except Exception:
                            pass
                used += sum(
                    int(item.usedGpuMemory)
                    for item in processes
                    if int(item.pid) == pid
                    and int(item.usedGpuMemory) >= 0
                )
            pynvml.nvmlShutdown()
            return used / gib
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return 0.0
    used_mib = 0.0
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                used_mib += float(parts[1])
            except ValueError:
                pass
    return used_mib / 1024.0


def _stop_process(process: Any, grace_seconds: float) -> None:
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    process.terminate()
    process.join(timeout=grace_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=grace_seconds)


def _resolve_process_start_method(cfg: AppConfig) -> str:
    method = cfg.execution.process_start_method
    if (
        cfg.execution.backend == "local"
        and cfg.execution.local_trial_mode == "process"
        and method == "forkserver"
    ):
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.close()
        except OSError:
            method = "spawn"
            cfg.execution.process_start_method = method
    return method


class ResourceManager:
    def __init__(
        self,
        cfg: AppConfig,
        output_directory: str | Path,
        snapshot_provider: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.config = cfg.resources
        self.execution = cfg.execution
        self.output = Path(output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        self.usage_path = self.output / "resource_usage.jsonl"
        self.usage_path.write_text("", encoding="utf-8")
        self._snapshot_provider = snapshot_provider or detect_resources
        self.initial_snapshot = self._apply_manual_limits(
            self._snapshot_provider()
        )
        self.task_type = str(
            cfg.model_params.get("task_type", "CPU")
        ).upper()
        self.requested_parallel_trials = self.execution.parallel_trials
        self.ram_per_trial_gb = float(
            self.config.estimated_ram_per_trial_gb or 2.0
        )
        self.gpu_memory_per_trial_gb = float(
            self.config.estimated_gpu_memory_per_trial_gb or 2.0
        )
        self.calibration: TrialResourceEstimate | None = None
        self.resource_profiles: dict[str, ResourceProfile] = {}
        self.process_memory_telemetry_available = (
            _process_memory_telemetry_available()
        )
        self.oom_events = 0
        self.timeout_events = 0
        self.heartbeat_events = 0
        self.hard_limit_events = 0
        self.throttle_events = 0
        self.max_active_trials = 0
        self.min_ram_available_gb = self.initial_snapshot.ram_available_gb
        self.max_ram_percent = self.initial_snapshot.ram_percent
        self.max_cpu_percent = self.initial_snapshot.cpu_percent
        self.min_gpu_free_gb = {
            item.device: item.free_gb for item in self.initial_snapshot.gpus
        }
        self.started_at = time.time()
        self._resolve_execution()
        self.max_parallel_trials = self._initial_parallel_limit()
        self.current_parallel_limit = self.max_parallel_trials

    def _apply_manual_limits(
        self, snapshot: ResourceSnapshot
    ) -> ResourceSnapshot:
        mode = self.config.mode
        if mode == "auto":
            return snapshot
        if self.config.cpu_cores is not None:
            cpu_effective = (
                self.config.cpu_cores
                if mode == "manual"
                else min(snapshot.cpu_effective, self.config.cpu_cores)
            )
        else:
            cpu_effective = snapshot.cpu_effective
        if self.config.ram_gb is not None:
            ram_total = (
                self.config.ram_gb
                if mode == "manual"
                else min(snapshot.ram_total_gb, self.config.ram_gb)
            )
            # A manual RAM value is a scheduling ceiling. Preserve the
            # observed utilization ratio inside that ceiling.
            ram_available = min(
                snapshot.ram_available_gb,
                ram_total * max(0.0, 1.0 - snapshot.ram_percent),
            )
        else:
            ram_total = snapshot.ram_total_gb
            ram_available = snapshot.ram_available_gb
        detected = {item.device: item for item in snapshot.gpus}
        manual_gpu_memory = {
            str(device): float(total)
            for device, total in self.config.gpu_memory_by_device_gb.items()
        }
        if (
            not manual_gpu_memory
            and self.config.gpu_count
            and self.config.gpu_total_memory_gb
        ):
            per_device = (
                self.config.gpu_total_memory_gb / self.config.gpu_count
            )
            manual_gpu_memory = {
                str(index): per_device
                for index in range(self.config.gpu_count)
            }
        if manual_gpu_memory:
            gpus = []
            for device, total in sorted(manual_gpu_memory.items()):
                observed = detected.get(device)
                effective_total = (
                    total
                    if mode == "manual" or observed is None
                    else min(total, observed.total_gb)
                )
                observed_used = (
                    observed.total_gb - observed.free_gb
                    if observed is not None
                    else 0.0
                )
                gpus.append(
                    GPUResource(
                        device=device,
                        total_gb=effective_total,
                        free_gb=max(0.0, effective_total - observed_used),
                        utilization=(observed.utilization if observed else 0.0),
                        source=(
                            "manual"
                            if observed is None
                            else f"{observed.source}+manual_cap"
                        ),
                    )
                )
        else:
            gpus = snapshot.gpus
        return ResourceSnapshot(
            timestamp=snapshot.timestamp,
            cpu_logical=snapshot.cpu_logical,
            cpu_physical=snapshot.cpu_physical,
            cpu_effective=max(1, int(cpu_effective)),
            cpu_percent=snapshot.cpu_percent,
            ram_total_gb=float(ram_total),
            ram_available_gb=float(ram_available),
            ram_percent=(
                1.0 - ram_available / ram_total if ram_total else 0.0
            ),
            process_rss_gb=snapshot.process_rss_gb,
            gpus=gpus,
            source=(snapshot.source if mode == "auto" else f"{snapshot.source}+{mode}"),
        )

    def snapshot(self) -> ResourceSnapshot:
        return self._apply_manual_limits(self._snapshot_provider())

    def _usable_cpu_cores(self) -> int:
        return max(
            1,
            math.floor(
                self.initial_snapshot.cpu_effective
                * self.config.max_cpu_utilization
            )
            - self.config.reserve_cpu_cores,
        )

    def _resolve_execution(self) -> None:
        usable_cpu = self._usable_cpu_cores()
        if self.execution.threads_per_trial == 0:
            configured_threads = int(
                self.cfg.model_params.get("thread_count", 0)
            )
            if configured_threads > 0:
                threads = configured_threads
            elif self.execution.backend == "ray":
                threads = max(
                    1, math.floor(self.execution.ray_num_cpus_per_trial)
                )
            elif self.task_type == "GPU":
                gpu_count = max(1, len(self.initial_snapshot.gpus))
                threads = max(1, usable_cpu // gpu_count)
            else:
                target_workers = (
                    self.execution.parallel_trials
                    if self.execution.parallel_trials > 0
                    else max(1, math.floor(math.sqrt(usable_cpu)))
                )
                threads = max(1, usable_cpu // target_workers)
            self.execution.threads_per_trial = min(threads, usable_cpu)
        if self.task_type == "GPU" and self.execution.backend == "local":
            requested = {str(item) for item in self.execution.gpu_devices}
            detected = self.initial_snapshot.gpus
            selected = [
                item
                for item in detected
                if not requested or item.device in requested
            ]
            eligible = [item.device for item in selected if self._gpu_fits(item)]
            if not eligible:
                raise RuntimeError(
                    "No GPU has enough free memory under the configured safety limits"
                )
            self.execution.gpu_devices = eligible

    def _gpu_fits(self, item: GPUResource) -> bool:
        used = item.total_gb - item.free_gb
        allowed_used = item.total_gb * self.config.max_gpu_memory_utilization
        return (
            item.free_gb - self.config.reserve_gpu_memory_gb
            >= self.gpu_memory_per_trial_gb
            and allowed_used - used >= self.gpu_memory_per_trial_gb
        )

    def gpu_device_available(self, device: str) -> bool:
        return any(
            item.device == str(device) and self._gpu_fits(item)
            for item in self.snapshot().gpus
        )

    def _ram_additional_slots(self, snapshot: ResourceSnapshot) -> int:
        used = snapshot.ram_total_gb - snapshot.ram_available_gb
        allowed_used = (
            snapshot.ram_total_gb * self.config.max_ram_utilization
        )
        headroom = max(
            0.0,
            min(
                snapshot.ram_available_gb - self.config.reserve_ram_gb,
                allowed_used - used,
            ),
        )
        return max(0, math.floor(headroom / self.ram_per_trial_gb))

    def _initial_parallel_limit(self) -> int:
        if self.execution.backend == "ray":
            return self.requested_parallel_trials or 256
        cpu_slots = max(
            1,
            self._usable_cpu_cores()
            // max(1, self.execution.threads_per_trial),
        )
        ram_slots = max(1, self._ram_additional_slots(self.initial_snapshot))
        limits = [cpu_slots, ram_slots]
        if self.task_type == "GPU":
            limits.append(len(self.execution.gpu_devices))
        if self.requested_parallel_trials > 0:
            limits.append(self.requested_parallel_trials)
        return max(1, min(limits))

    def allowed_concurrency(
        self,
        active_trials: int,
        pending_trials: int,
        event: str = "scheduler",
    ) -> int:
        if not self.config.enabled:
            return self.current_parallel_limit
        snapshot = self.snapshot()
        allowed = self.current_parallel_limit
        additional_ram = self._ram_additional_slots(snapshot)
        allowed = min(allowed, active_trials + additional_ram)
        if self.task_type == "GPU" and self.execution.backend == "local":
            selected = set(self.execution.gpu_devices)
            additional_gpu = sum(
                1
                for item in snapshot.gpus
                if item.device in selected and self._gpu_fits(item)
            )
            allowed = min(
                allowed,
                len(selected),
                active_trials + additional_gpu,
            )
        if (
            self.config.adaptive_concurrency
            and snapshot.cpu_percent
            >= self.config.max_cpu_utilization * 100.0
        ):
            allowed = min(allowed, active_trials)
        if allowed < self.current_parallel_limit and pending_trials:
            self.throttle_events += 1
        self.record_usage(
            snapshot,
            active_trials,
            pending_trials,
            max(0, allowed),
            event,
        )
        return max(0, allowed)

    def record_usage(
        self,
        snapshot: ResourceSnapshot,
        active_trials: int,
        pending_trials: int,
        allowed_trials: int,
        event: str,
    ) -> None:
        self.max_active_trials = max(self.max_active_trials, active_trials)
        self.min_ram_available_gb = min(
            self.min_ram_available_gb, snapshot.ram_available_gb
        )
        self.max_ram_percent = max(self.max_ram_percent, snapshot.ram_percent)
        self.max_cpu_percent = max(self.max_cpu_percent, snapshot.cpu_percent)
        for gpu in snapshot.gpus:
            self.min_gpu_free_gb[gpu.device] = min(
                self.min_gpu_free_gb.get(gpu.device, gpu.free_gb),
                gpu.free_gb,
            )
        row = {
            "timestamp": snapshot.timestamp,
            "event": event,
            "active_trials": active_trials,
            "pending_trials": pending_trials,
            "allowed_trials": allowed_trials,
            "parallel_limit": self.current_parallel_limit,
            "cpu_percent": snapshot.cpu_percent,
            "ram_total_gb": snapshot.ram_total_gb,
            "ram_available_gb": snapshot.ram_available_gb,
            "ram_percent": snapshot.ram_percent,
            "process_rss_gb": snapshot.process_rss_gb,
            "gpus": [asdict(item) for item in snapshot.gpus],
        }
        with self.usage_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def wait_for_single_trial(self, event: str) -> None:
        started = time.monotonic()
        while self.allowed_concurrency(0, 1, event) < 1:
            if (
                time.monotonic() - started
                >= self.config.resource_wait_timeout_seconds
            ):
                raise RuntimeError(
                    "Resource wait timeout: a single trial would exceed the "
                    "configured CPU/RAM/GPU safety limits"
                )
            time.sleep(self.config.monitoring_interval_seconds)

    def calibrate(
        self,
        operation: Any,
        subset_size: int,
        fidelity: str,
    ) -> Any:
        gc.collect()
        initial = self.snapshot()
        peak_rss = initial.process_rss_gb
        initial_gpu_used = {
            item.device: item.total_gb - item.free_gb
            for item in initial.gpus
        }
        peak_gpu_used = dict(initial_gpu_used)
        stop = threading.Event()

        def monitor() -> None:
            nonlocal peak_rss
            interval = min(0.20, self.config.monitoring_interval_seconds)
            while not stop.wait(interval):
                current = self.snapshot()
                peak_rss = max(peak_rss, current.process_rss_gb)
                for gpu in current.gpus:
                    peak_gpu_used[gpu.device] = max(
                        peak_gpu_used.get(gpu.device, 0.0),
                        gpu.total_gb - gpu.free_gb,
                    )

        thread = threading.Thread(target=monitor, daemon=True)
        started = time.perf_counter()
        thread.start()
        try:
            result = operation()
        finally:
            stop.set()
            thread.join(timeout=2.0)
        runtime = time.perf_counter() - started
        final = self.snapshot()
        peak_rss = max(peak_rss, final.process_rss_gb)
        ram_increment = max(0.05, peak_rss - initial.process_rss_gb)
        gpu_increment = max(
            [
                max(
                    0.0,
                    peak_gpu_used.get(device, used) - used,
                )
                for device, used in initial_gpu_used.items()
            ]
            or [0.0]
        )
        factor = self.config.calibration_safety_factor
        measured_ram = max(0.10, ram_increment * factor)
        measured_gpu = max(0.0, gpu_increment * factor)
        isolated_profile = self.resource_profiles.get(fidelity)
        if isolated_profile is not None:
            measured_ram = max(
                measured_ram, isolated_profile.peak_rss_gb * factor
            )
            measured_gpu = max(
                measured_gpu, isolated_profile.peak_gpu_gb * factor
            )
            ram_increment = max(
                ram_increment, isolated_profile.peak_rss_gb
            )
            gpu_increment = max(
                gpu_increment, isolated_profile.peak_gpu_gb
            )
        if self.config.estimated_ram_per_trial_gb is None:
            self.ram_per_trial_gb = measured_ram
        if (
            self.config.estimated_gpu_memory_per_trial_gb is None
            and self.task_type == "GPU"
        ):
            self.gpu_memory_per_trial_gb = max(0.25, measured_gpu)
        if self.task_type == "GPU" and self.execution.backend == "local":
            current = self.snapshot()
            selected = set(self.execution.gpu_devices)
            viable = [
                item.device
                for item in current.gpus
                if item.device in selected and self._gpu_fits(item)
            ]
            if not viable:
                raise RuntimeError(
                    "The calibrated GPU-memory estimate does not fit on any "
                    "selected device under the configured safety limits"
                )
            self.execution.gpu_devices = viable
        self.calibration = TrialResourceEstimate(
            subset_size=subset_size,
            fidelity=fidelity,
            runtime_seconds=runtime,
            ram_increment_gb=ram_increment,
            gpu_increment_gb=gpu_increment,
            safety_factor=factor,
            ram_per_trial_gb=self.ram_per_trial_gb,
            gpu_memory_per_trial_gb=self.gpu_memory_per_trial_gb,
        )
        self.max_parallel_trials = self._initial_parallel_limit()
        self.current_parallel_limit = self.max_parallel_trials
        self.record_usage(final, 0, 0, self.current_parallel_limit, "calibration")
        return result

    def report_oom(self) -> None:
        self.oom_events += 1
        self.current_parallel_limit = max(
            1,
            math.floor(
                self.current_parallel_limit
                * self.config.oom_concurrency_reduction
            ),
        )
        self.ram_per_trial_gb *= 1.25
        if self.task_type == "GPU":
            self.gpu_memory_per_trial_gb *= 1.25

    def hard_ram_limit_gb(self, fidelity: str) -> float:
        if self.config.hard_ram_per_trial_gb is not None:
            return self.config.hard_ram_per_trial_gb
        profile = self.resource_profiles.get(fidelity)
        observed = profile.peak_rss_gb if profile else 0.0
        estimated = max(
            self.ram_per_trial_gb + 0.50,
            self.ram_per_trial_gb * self.config.hard_limit_multiplier,
            observed * self.config.hard_limit_multiplier,
        )
        return min(
            estimated,
            max(
                0.10,
                self.initial_snapshot.ram_total_gb
                - self.config.reserve_ram_gb,
            ),
        )

    def hard_gpu_limit_gb(self, fidelity: str) -> float:
        if self.config.hard_gpu_memory_per_trial_gb is not None:
            return self.config.hard_gpu_memory_per_trial_gb
        profile = self.resource_profiles.get(fidelity)
        observed = profile.peak_gpu_gb if profile else 0.0
        return max(
            self.gpu_memory_per_trial_gb
            * self.config.hard_limit_multiplier,
            observed * self.config.hard_limit_multiplier,
        )

    def observe_isolated_trial(
        self,
        fidelity: str,
        subset_size: int,
        runtime_seconds: float,
        peak_rss_gb: float,
        peak_gpu_gb: float,
    ) -> None:
        if not self.config.profile_learning_enabled:
            return
        profile = self.resource_profiles.setdefault(
            fidelity, ResourceProfile(fidelity=fidelity)
        )
        profile.observations += 1
        profile.peak_rss_gb = max(profile.peak_rss_gb, peak_rss_gb)
        profile.peak_gpu_gb = max(profile.peak_gpu_gb, peak_gpu_gb)
        profile.maximum_subset_size = max(
            profile.maximum_subset_size, subset_size
        )
        profile.maximum_runtime_seconds = max(
            profile.maximum_runtime_seconds, runtime_seconds
        )
        factor = self.config.calibration_safety_factor
        if self.config.estimated_ram_per_trial_gb is None and peak_rss_gb > 0:
            self.ram_per_trial_gb = max(
                self.ram_per_trial_gb, peak_rss_gb * factor
            )
        if (
            self.task_type == "GPU"
            and self.config.estimated_gpu_memory_per_trial_gb is None
            and peak_gpu_gb > 0
        ):
            self.gpu_memory_per_trial_gb = max(
                self.gpu_memory_per_trial_gb, peak_gpu_gb * factor
            )
        updated_limit = self._initial_parallel_limit()
        self.max_parallel_trials = min(
            self.max_parallel_trials, updated_limit
        )
        self.current_parallel_limit = min(
            self.current_parallel_limit, self.max_parallel_trials
        )

    def report_termination(self, reason: str) -> None:
        if reason in {"ram_limit", "gpu_memory_limit"}:
            self.hard_limit_events += 1
            self.report_oom()
        elif reason == "timeout":
            self.timeout_events += 1
        elif reason == "heartbeat_timeout":
            self.heartbeat_events += 1
        self.record_usage(
            self.snapshot(),
            0,
            0,
            self.current_parallel_limit,
            f"worker_{reason}",
        )

    def _hardware_signature(self) -> dict[str, Any]:
        snapshot = self.initial_snapshot
        return {
            "cpu_effective": snapshot.cpu_effective,
            "ram_total_gb": round(snapshot.ram_total_gb, 3),
            "gpus": [
                {
                    "device": item.device,
                    "total_gb": round(item.total_gb, 3),
                }
                for item in snapshot.gpus
            ],
        }

    def restore_calibration(
        self,
        run_fingerprint: str,
        calibration_subset: Sequence[str],
        calibration_fidelity: str,
    ) -> bool:
        path = self.output / "resource_plan.json"
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration = payload["calibration"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            payload.get("selector_version") != APP_VERSION
            or payload.get("run_fingerprint") != run_fingerprint
            or payload.get("hardware_signature") != self._hardware_signature()
            or payload.get("calibration_subset") != list(calibration_subset)
            or payload.get("calibration_fidelity") != calibration_fidelity
            or not isinstance(calibration, dict)
        ):
            return False
        try:
            restored = TrialResourceEstimate(**calibration)
        except (TypeError, ValueError):
            return False
        self.calibration = restored
        self.resource_profiles = {
            name: ResourceProfile(**row)
            for name, row in payload.get("resource_profiles", {}).items()
        }
        if self.config.estimated_ram_per_trial_gb is None:
            self.ram_per_trial_gb = restored.ram_per_trial_gb
        if self.config.estimated_gpu_memory_per_trial_gb is None:
            self.gpu_memory_per_trial_gb = restored.gpu_memory_per_trial_gb
        self.max_parallel_trials = self._initial_parallel_limit()
        self.current_parallel_limit = self.max_parallel_trials
        self.record_usage(
            self.snapshot(),
            0,
            0,
            self.current_parallel_limit,
            "calibration_restored",
        )
        return True

    def write_plan(
        self,
        calibration_subset: Sequence[str],
        calibration_fidelity: str,
        run_fingerprint: str,
    ) -> None:
        snapshot = self.initial_snapshot
        payload = {
            "selector_version": APP_VERSION,
            "run_fingerprint": run_fingerprint,
            "hardware_signature": self._hardware_signature(),
            "mode": self.config.mode,
            "backend": self.execution.backend,
            "task_type": self.task_type,
            "requested_parallel_trials": self.requested_parallel_trials,
            "effective_parallel_limit": self.max_parallel_trials,
            "threads_per_trial": self.execution.threads_per_trial,
            "process_memory_telemetry_available": (
                self.process_memory_telemetry_available
            ),
            "detected": asdict(snapshot),
            "selected_gpu_devices": list(self.execution.gpu_devices),
            "ram_per_trial_gb": self.ram_per_trial_gb,
            "gpu_memory_per_trial_gb": self.gpu_memory_per_trial_gb,
            "calibration_subset": list(calibration_subset),
            "calibration_fidelity": calibration_fidelity,
            "calibration": (
                asdict(self.calibration) if self.calibration else None
            ),
            "resource_profiles": {
                name: asdict(profile)
                for name, profile in sorted(self.resource_profiles.items())
            },
            "limits": asdict(self.config),
        }
        _write_json(self.output / "resource_plan.json", payload)

    def write_summary(self) -> None:
        payload = {
            "selector_version": APP_VERSION,
            "elapsed_seconds": time.time() - self.started_at,
            "max_active_trials": self.max_active_trials,
            "final_parallel_limit": self.current_parallel_limit,
            "process_memory_telemetry_available": (
                self.process_memory_telemetry_available
            ),
            "minimum_ram_available_gb": self.min_ram_available_gb,
            "maximum_ram_utilization": self.max_ram_percent,
            "maximum_cpu_utilization": self.max_cpu_percent / 100.0,
            "minimum_gpu_free_gb": self.min_gpu_free_gb,
            "oom_events": self.oom_events,
            "timeout_events": self.timeout_events,
            "heartbeat_events": self.heartbeat_events,
            "hard_limit_events": self.hard_limit_events,
            "throttle_events": self.throttle_events,
            "ram_per_trial_gb": self.ram_per_trial_gb,
            "gpu_memory_per_trial_gb": self.gpu_memory_per_trial_gb,
            "resource_profiles": {
                name: asdict(profile)
                for name, profile in sorted(self.resource_profiles.items())
            },
        }
        _write_json(self.output / "resource_summary.json", payload)

    @property
    def ray_memory_bytes_per_trial(self) -> int:
        return max(1, int(self.ram_per_trial_gb * 1024**3))


class SubsetEvaluator(Protocol):
    primary_metric: str

    def evaluate(self, subset: Iterable[str], fidelity: str) -> Evaluation:
        ...


class EvaluationCache:
    def __init__(self, directory: str | Path, fingerprint: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def _path(self, subset: Sequence[str], fidelity: str) -> Path:
        key = _stable_hash(
            {"fingerprint": self.fingerprint, "subset": sorted(subset), "fidelity": fidelity}
        )
        return self.directory / f"{key}.json"

    def get(self, subset: Sequence[str], fidelity: str) -> Evaluation | None:
        path = self._path(subset, fidelity)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["subset"] = tuple(payload["subset"])
        return Evaluation(**payload)

    def put(self, evaluation: Evaluation) -> None:
        path = self._path(evaluation.subset, evaluation.fidelity)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(evaluation), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


def _parquet_sources(path: str) -> list[str] | str:
    source = Path(path)
    if source.is_dir():
        files = sorted(str(file) for file in source.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        return files
    if glob.has_magic(path):
        files = sorted(glob.glob(path, recursive=True))
        if not files:
            raise FileNotFoundError(f"No files match Parquet path pattern: {path}")
        return files
    return path


def _source_fingerprint(path: str) -> dict[str, Any]:
    resolved = _parquet_sources(path)
    files = [resolved] if isinstance(resolved, str) else resolved
    result = []
    for name in files:
        file = Path(name)
        if file.exists():
            stat = file.stat()
            result.append((str(file.resolve()), stat.st_size, stat.st_mtime_ns))
        else:
            result.append((name, None, None))
    return {"path": path, "files": result}


def scan_parquet(path: str) -> Any:
    pl = _require("polars", "polars>=1.30")
    return pl.scan_parquet(_parquet_sources(path))


def collect_feature_names(cfg: AppConfig) -> list[str]:
    train_schema = dict(scan_parquet(cfg.data.train_path).collect_schema())
    excluded = {
        cfg.data.target,
        *cfg.data.id_columns,
        *cfg.data.sampling_key_columns,
        *cfg.data.leakage_key_columns,
        *cfg.data.bootstrap_key_columns,
        *cfg.data.excluded_features,
        INTERNAL_WEIGHT,
    }
    control_columns = {
        cfg.data.target,
        *cfg.data.id_columns,
        *cfg.data.sampling_key_columns,
        *cfg.data.leakage_key_columns,
        *cfg.data.bootstrap_key_columns,
    }
    missing_control = sorted(control_columns - set(train_schema))
    if missing_control:
        raise ValueError(f"Required control columns are missing: {missing_control}")
    features = [column for column in train_schema if column not in excluded]
    missing_required = sorted(set(cfg.data.required_features) - set(features))
    if missing_required:
        raise ValueError(f"Required features are missing: {missing_required}")
    unknown_categorical = sorted(set(cfg.data.categorical_features) - set(features))
    if unknown_categorical:
        raise ValueError(f"Categorical features are missing: {unknown_categorical}")
    return features


def configured_split_paths(cfg: AppConfig) -> dict[str, str]:
    return {
        name: path
        for name, path in {
            "train": cfg.data.train_path,
            "valid": cfg.data.valid_path,
            "oos": cfg.data.oos_path,
            "oot": cfg.data.oot_path,
            "test": cfg.data.test_path,
        }.items()
        if path
    }


def validation_mode(cfg: AppConfig) -> str:
    has_oos = bool(cfg.data.oos_path)
    has_final_holdout = bool(cfg.data.oot_path or cfg.data.test_path)
    if has_oos and has_final_holdout:
        return "full"
    if has_oos:
        return "oos_only"
    if has_final_holdout:
        return "final_holdout_only"
    return "development_only"


def validate_inputs(cfg: AppConfig, features: Sequence[str]) -> None:
    """Fail fast on unusable inputs without changing or filtering features."""
    pl = _require("polars", "polars>=1.30")
    split_paths = configured_split_paths(cfg)
    schemas = {
        name: dict(scan_parquet(path).collect_schema())
        for name, path in split_paths.items()
    }
    train_schema = schemas["train"]

    for split_name, path in split_paths.items():
        required_columns = {
            *features,
            cfg.data.target,
            *cfg.data.id_columns,
            *cfg.data.sampling_key_columns,
            *cfg.data.bootstrap_key_columns,
        }
        if cfg.validation.check_id_overlap:
            required_columns.update(cfg.data.leakage_key_columns)
        missing = sorted(required_columns - set(schemas[split_name]))
        if missing:
            raise ValueError(f"{split_name} is missing columns: {missing}")
        if cfg.validation.check_split_schema and split_name != "train":
            mismatched = sorted(
                feature
                for feature in features
                if str(train_schema[feature]) != str(schemas[split_name][feature])
            )
            if mismatched:
                raise TypeError(f"{split_name} has dtype mismatches: {mismatched}")

        summary = (
            scan_parquet(path)
            .select(
                pl.len().alias("rows"),
                pl.col(cfg.data.target).null_count().alias("target_nulls"),
                pl.col(cfg.data.target).drop_nulls().n_unique().alias("classes"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
        if int(summary["rows"]) == 0:
            raise ValueError(f"{split_name} split is empty")
        if int(summary["target_nulls"]):
            raise ValueError(
                f"{split_name} target contains {int(summary['target_nulls'])} null values"
            )
        if int(summary["classes"]) != 2:
            raise ValueError(
                f"{split_name} target must contain exactly two non-null classes"
            )
        positive_present = bool(
            scan_parquet(path)
            .select(
                (pl.col(cfg.data.target) == cfg.data.positive_label)
                .any()
                .alias("positive_present")
            )
            .collect(engine="streaming")[0, "positive_present"]
        )
        if not positive_present:
            raise ValueError(
                f"positive_label={cfg.data.positive_label!r} is absent in "
                f"{split_name} target"
            )

    if cfg.validation.check_id_overlap and cfg.data.leakage_key_columns:
        keys = cfg.data.leakage_key_columns
        train_keys = scan_parquet(cfg.data.train_path).select(keys).unique()
        for split_name, path in split_paths.items():
            if split_name == "train":
                continue
            overlap = int(
                train_keys.join(
                    scan_parquet(path).select(keys).unique(),
                    on=keys,
                    how="inner",
                )
                .select(pl.len().alias("rows"))
                .collect(engine="streaming")[0, "rows"]
            )
            if overlap and cfg.validation.fail_on_id_overlap:
                raise ValueError(
                    f"Found {overlap} overlapping leakage keys between train "
                    f"and {split_name}"
                )


def _sample_expression(
    cfg: AppConfig,
    positive_fraction: float,
    negative_fraction: float,
) -> Any:
    pl = _require("polars", "polars>=1.30")
    key = pl.concat_str(
        [
            pl.col(column).cast(pl.String).fill_null("__NULL__")
            for column in cfg.data.sampling_key_columns
        ],
        separator="|",
    )
    uniform = (key.hash(seed=cfg.sample.seed) % 1_000_000).cast(pl.Float64) / 1_000_000
    probability = (
        pl.when(pl.col(cfg.data.target) == cfg.data.positive_label)
        .then(pl.lit(positive_fraction))
        .otherwise(pl.lit(negative_fraction))
    )
    return uniform, probability


def prepare_sample(
    cfg: AppConfig,
    source: str,
    output: str | Path,
    features: Sequence[str],
    positive_fraction: float,
    negative_fraction: float,
) -> None:
    pl = _require("polars", "polars>=1.30")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    uniform, probability = _sample_expression(cfg, positive_fraction, negative_fraction)
    columns = list(
        dict.fromkeys(
            [
                *features,
                cfg.data.target,
                *cfg.data.id_columns,
                *cfg.data.bootstrap_key_columns,
            ]
        )
    )
    query = (
        scan_parquet(source)
        .filter(uniform < probability)
        .with_columns((1.0 / probability).alias(INTERNAL_WEIGHT))
        .select([*columns, INTERNAL_WEIGHT])
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    query.sink_parquet(temporary, compression="zstd", mkdir=True)
    temporary.replace(output)


def prepare_samples(cfg: AppConfig, features: Sequence[str]) -> dict[str, Path]:
    dependency_versions = _dependency_versions()
    root = Path(cfg.output.directory) / "samples"
    specs = {
        "screen_train": (
            cfg.data.train_path,
            cfg.sample.screen_train_positive_fraction,
            cfg.sample.screen_train_negative_fraction,
        ),
        "screen_valid": (
            cfg.data.valid_path,
            cfg.sample.screen_valid_positive_fraction,
            cfg.sample.screen_valid_negative_fraction,
        ),
        "search_train": (
            cfg.data.train_path,
            cfg.sample.train_positive_fraction,
            cfg.sample.train_negative_fraction,
        ),
        "search_valid": (
            cfg.data.valid_path,
            cfg.sample.valid_positive_fraction,
            cfg.sample.valid_negative_fraction,
        ),
        "confirm_train": (
            cfg.data.train_path,
            cfg.sample.confirm_train_positive_fraction,
            cfg.sample.confirm_train_negative_fraction,
        ),
        "confirm_valid": (
            cfg.data.valid_path,
            cfg.sample.confirm_valid_positive_fraction,
            cfg.sample.confirm_valid_negative_fraction,
        ),
    }
    if not cfg.search.successive_halving_enabled:
        specs.pop("screen_train")
        specs.pop("screen_valid")
    if cfg.data.oos_path:
        specs["oos"] = (
            cfg.data.oos_path,
            cfg.sample.oos_positive_fraction,
            cfg.sample.oos_negative_fraction,
        )
    if cfg.data.oot_path:
        specs["oot"] = (
            cfg.data.oot_path,
            cfg.sample.oot_positive_fraction,
            cfg.sample.oot_negative_fraction,
        )
    if cfg.data.test_path:
        specs["test"] = (
            cfg.data.test_path,
            cfg.sample.test_positive_fraction,
            cfg.sample.test_negative_fraction,
        )
    outputs: dict[str, Path] = {}
    for name, (source, positive_fraction, negative_fraction) in specs.items():
        sample_hash = _stable_hash(
            {
                "source": _source_fingerprint(source),
                "selector_version": APP_VERSION,
                "polars_version": dependency_versions["polars"],
                "features": sorted(features),
                "target": cfg.data.target,
                "positive_label": cfg.data.positive_label,
                "keys": cfg.data.sampling_key_columns,
                "id_columns": cfg.data.id_columns,
                "bootstrap_key_columns": cfg.data.bootstrap_key_columns,
                "seed": cfg.sample.seed,
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
            }
        )[:12]
        destination = root / f"{name}_{sample_hash}.parquet"
        outputs[name] = destination
        if destination.exists():
            continue
        prepare_sample(
            cfg,
            source,
            destination,
            features,
            positive_fraction,
            negative_fraction,
        )
    return outputs


def _metric_value(name: str, y_true: Any, prediction: Any, weight: Any) -> float:
    metrics = _require("sklearn.metrics", "scikit-learn>=1.5")
    if name in {"average_precision", "pr_auc"}:
        return float(metrics.average_precision_score(y_true, prediction, sample_weight=weight))
    if name in {"roc_auc", "auc"}:
        return float(metrics.roc_auc_score(y_true, prediction, sample_weight=weight))
    if name == "gini":
        auc = metrics.roc_auc_score(y_true, prediction, sample_weight=weight)
        return float(2 * auc - 1)
    if name == "logloss":
        return float(-metrics.log_loss(y_true, prediction, sample_weight=weight, labels=[0, 1]))
    if name == "brier":
        return float(-metrics.brier_score_loss(y_true, prediction, sample_weight=weight))
    raise ValueError(f"Unsupported metric: {name}")


class CatBoostSubsetEvaluator:
    def __init__(
        self,
        cfg: AppConfig,
        fidelities: Sequence[FidelityConfig],
        cache: EvaluationCache,
        resource_manager: ResourceManager | None = None,
    ):
        self.cfg = cfg
        self.fidelities = {item.name: item for item in fidelities}
        self.cache = cache
        self.primary_metric = cfg.search.primary_metric
        self.parallel_trials = cfg.execution.parallel_trials
        self.resource_manager = resource_manager
        self._gpu_slots: Queue[str] | None = None
        if cfg.execution.gpu_devices:
            self._gpu_slots = Queue()
            for device in cfg.execution.gpu_devices:
                self._gpu_slots.put(str(device))

    @contextmanager
    def _training_slot(self) -> Iterable[str | None]:
        device = None
        if self._gpu_slots is not None:
            started = time.monotonic()
            while device is None:
                candidate = self._gpu_slots.get()
                if (
                    self.resource_manager is None
                    or self.resource_manager.gpu_device_available(candidate)
                ):
                    device = candidate
                    break
                self._gpu_slots.put(candidate)
                if (
                    time.monotonic() - started
                    >= self.resource_manager.config.resource_wait_timeout_seconds
                ):
                    raise RuntimeError(
                        "Resource wait timeout: no selected GPU has enough "
                        "free memory for this trial"
                    )
                time.sleep(
                    self.resource_manager.config.monitoring_interval_seconds
                )
        try:
            yield device
        finally:
            if self._gpu_slots is not None and device is not None:
                self._gpu_slots.put(device)

    def _prediction_cache_path(
        self,
        subset: Sequence[str],
        fidelity: str,
        holdout_samples: dict[str, str | Path],
    ) -> Path:
        key = _stable_hash(
            {
                "fingerprint": self.cache.fingerprint,
                "subset": sorted(subset),
                "fidelity": fidelity,
                "holdouts": {
                    name: str(path) for name, path in sorted(holdout_samples.items())
                },
            }
        )
        directory = self.cache.directory / "predictions"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}.npz"

    def _interaction_cache_path(
        self,
        subset: Sequence[str],
        fidelity: str,
        limit: int,
    ) -> Path:
        key = _stable_hash(
            {
                "fingerprint": self.cache.fingerprint,
                "subset": sorted(subset),
                "fidelity": fidelity,
                "limit": limit,
            }
        )
        directory = self.cache.directory / "interactions"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}.json"

    def _load(self, path: str, subset: Sequence[str]) -> tuple[Any, Any, Any]:
        pl = _require("polars", "polars>=1.30")
        columns = [*subset, self.cfg.data.target, INTERNAL_WEIGHT]
        frame = pl.read_parquet(path, columns=columns)
        y = frame[self.cfg.data.target].to_numpy()
        y = (y == self.cfg.data.positive_label).astype("int8")
        weight = frame[INTERNAL_WEIGHT].to_numpy()
        return self._model_frame(frame, subset), y, weight

    def _model_frame(self, frame: Any, subset: Sequence[str]) -> Any:
        """Prepare CatBoost categorical values without mutating source Parquet."""
        pl = _require("polars", "polars>=1.30")
        categorical = [
            feature
            for feature in subset
            if feature in self.cfg.data.categorical_features
        ]
        if not categorical:
            return frame.select(list(subset))
        if self.cfg.data.categorical_null_strategy == "error":
            nulls = {
                feature: int(frame[feature].null_count())
                for feature in categorical
                if frame[feature].null_count()
            }
            if nulls:
                raise ValueError(
                    "Categorical features contain null values: "
                    f"{nulls}. Use data.categorical_null_strategy='fill' "
                    "or prepare the source data upstream."
                )
            return frame.select(list(subset))
        replacement = self.cfg.data.categorical_null_value
        return frame.with_columns(
            [
                pl.col(feature)
                .cast(pl.String)
                .fill_null(replacement)
                .alias(feature)
                for feature in categorical
            ]
        ).select(list(subset))

    def _load_holdout(
        self, path: str, subset: Sequence[str]
    ) -> tuple[Any, Any, Any, Any | None]:
        pl = _require("polars", "polars>=1.30")
        group_columns = self.cfg.data.bootstrap_key_columns
        columns = [
            *subset,
            self.cfg.data.target,
            INTERNAL_WEIGHT,
            *group_columns,
        ]
        frame = pl.read_parquet(path, columns=list(dict.fromkeys(columns)))
        y = (frame[self.cfg.data.target].to_numpy() == self.cfg.data.positive_label).astype(
            "int8"
        )
        weight = frame[INTERNAL_WEIGHT].to_numpy()
        groups = None
        if group_columns:
            group_name = "__bootstrap_group__"
            frame = frame.with_columns(
                pl.struct(group_columns).hash(seed=71).alias(group_name)
            )
            groups = frame[group_name].to_numpy()
        return self._model_frame(frame, subset), y, weight, groups

    def _params(
        self,
        fidelity: FidelityConfig,
        seed: int,
        device: str | None = None,
    ) -> dict[str, Any]:
        eval_metric = {
            "average_precision": "PRAUC:type=Classic;use_weights=true",
            "pr_auc": "PRAUC:type=Classic;use_weights=true",
            "roc_auc": "AUC:type=Classic;use_weights=true",
            "auc": "AUC:type=Classic;use_weights=true",
            "gini": "AUC:type=Classic;use_weights=true",
            "logloss": "Logloss:use_weights=true",
            "brier": "BrierScore:use_weights=true",
        }[self.primary_metric]
        params = {
            "loss_function": "Logloss",
            "eval_metric": eval_metric,
            "iterations": fidelity.iterations,
            "learning_rate": 0.05,
            "depth": 7,
            "l2_leaf_reg": 10.0,
            "random_seed": seed,
            "logging_level": "Silent",
            "allow_writing_files": False,
            "thread_count": -1,
        }
        params.update(self.cfg.model_params)
        # primary_metric owns CatBoost early-stopping/eval metric. Keeping an
        # independent model_params.eval_metric would make feature ranking and
        # early stopping optimize different objectives.
        params["eval_metric"] = eval_metric
        params["iterations"] = fidelity.iterations
        params["random_seed"] = seed
        if self.cfg.execution.threads_per_trial:
            params["thread_count"] = self.cfg.execution.threads_per_trial
        if device is not None:
            params["devices"] = device
        return params

    def evaluate(
        self,
        subset: Iterable[str],
        fidelity: str,
        use_cache: bool = True,
    ) -> Evaluation:
        selected = tuple(sorted(set(subset)))
        if not selected:
            raise ValueError("Cannot evaluate an empty subset")
        cached = self.cache.get(selected, fidelity) if use_cache else None
        if cached:
            return cached
        if fidelity not in self.fidelities:
            raise KeyError(f"Unknown fidelity: {fidelity}")

        np = _require("numpy", "numpy>=1.26,<3")
        catboost = _require("catboost", "catboost>=1.2.8")
        level = self.fidelities[fidelity]
        categorical = [feature for feature in selected if feature in self.cfg.data.categorical_features]
        X_train, y_train, w_train = self._load(level.train_sample, selected)
        X_valid, y_valid, w_valid = self._load(level.valid_sample, selected)
        train_pool = catboost.Pool(
            X_train,
            label=y_train,
            weight=w_train,
            cat_features=categorical,
            feature_names=list(selected),
        )
        valid_pool = catboost.Pool(
            X_valid,
            label=y_valid,
            weight=w_valid,
            cat_features=categorical,
            feature_names=list(selected),
        )

        start = time.perf_counter()
        train_scores: list[float] = []
        valid_scores: list[float] = []
        train_gini_scores: list[float] = []
        valid_gini_scores: list[float] = []
        valid_prediction_sum = np.zeros(len(y_valid), dtype="float64")
        metric_rows: list[dict[str, float]] = []
        with self._training_slot() as device:
            for seed in level.seeds:
                model = catboost.CatBoostClassifier(
                    **self._params(level, seed, device)
                )
                model.fit(
                    train_pool,
                    eval_set=valid_pool,
                    use_best_model=True,
                    early_stopping_rounds=50,
                )
                pred_train = model.predict_proba(train_pool)[:, 1]
                pred_valid = model.predict_proba(valid_pool)[:, 1]
                train_primary = _metric_value(
                    self.primary_metric, y_train, pred_train, w_train
                )
                valid_primary = _metric_value(
                    self.primary_metric, y_valid, pred_valid, w_valid
                )
                train_scores.append(train_primary)
                valid_scores.append(valid_primary)
                train_gini_scores.append(_metric_value("gini", y_train, pred_train, w_train))
                valid_gini_scores.append(_metric_value("gini", y_valid, pred_valid, w_valid))
                valid_prediction_sum += pred_valid
                metric_rows.append(
                    {
                        "average_precision": _metric_value(
                            "average_precision", y_valid, pred_valid, w_valid
                        ),
                        "roc_auc": _metric_value(
                            "roc_auc", y_valid, pred_valid, w_valid
                        ),
                        "gini": _metric_value(
                            "gini", y_valid, pred_valid, w_valid
                        ),
                        "logloss": _metric_value(
                            "logloss", y_valid, pred_valid, w_valid
                        ),
                        "brier": _metric_value(
                            "brier", y_valid, pred_valid, w_valid
                        ),
                    }
                )

        train_metric = float(np.mean(train_scores))
        valid_metric = float(np.mean(valid_scores))
        gap = max(0.0, train_metric - valid_metric)
        train_gini = float(np.mean(train_gini_scores))
        valid_gini = float(np.mean(valid_gini_scores))
        gini_gap = abs(train_gini - valid_gini)
        primary_gap_violation = (
            max(0.0, gap - self.cfg.search.max_primary_metric_gap)
            if self.cfg.search.max_primary_metric_gap is not None
            else 0.0
        )
        gini_gap_violation = max(0.0, gini_gap - self.cfg.search.max_gini_gap)
        aggregate_metrics = {
            name: float(np.mean([row[name] for row in metric_rows])) for name in metric_rows[0]
        }
        aggregate_metrics.update({
            "train_gini": train_gini,
            "valid_gini": valid_gini,
            "gini_gap": gini_gap,
        })
        threshold_violation = 0.0
        if (
            self.cfg.decision_threshold.enabled
            and self.cfg.decision_threshold.enforce_during_search
            and fidelity in self.cfg.decision_threshold.enforce_fidelities
        ):
            averaged_valid_prediction = valid_prediction_sum / len(level.seeds)
            threshold_result = select_decision_threshold(
                selected,
                SplitPredictions(
                    split=level.valid_sample,
                    y_true=y_valid,
                    prediction=averaged_valid_prediction,
                    weight=w_valid,
                    seed_metrics=valid_scores,
                ),
                None,
                self.cfg.decision_threshold,
            )
            aggregate_metrics.update({
                "search_threshold": threshold_result.threshold,
                "search_threshold_precision": threshold_result.tuning_precision,
                "search_threshold_recall": threshold_result.tuning_recall,
                "search_threshold_f1": threshold_result.tuning_f1,
                "search_threshold_feasible": float(threshold_result.tuning_feasible),
            })
            if not threshold_result.tuning_feasible:
                threshold_violation = (
                    max(0.0, self.cfg.decision_threshold.min_precision - threshold_result.tuning_precision)
                    + max(0.0, self.cfg.decision_threshold.min_recall - threshold_result.tuning_recall)
                )
        violation = primary_gap_violation + gini_gap_violation + threshold_violation
        evaluation = Evaluation(
            subset=selected,
            fidelity=fidelity,
            valid_metric=valid_metric,
            train_metric=train_metric,
            gap=gap,
            metric_std=float(np.std(valid_scores)),
            n_features=len(selected),
            runtime_seconds=time.perf_counter() - start,
            metrics=aggregate_metrics,
            constraint_violation=violation,
        )
        self.cache.put(evaluation)
        return evaluation

    def evaluate_with_holdouts(
        self,
        subset: Iterable[str],
        fidelity: str,
        holdout_samples: dict[str, str | Path],
        force_save_predictions: bool = False,
    ) -> tuple[Evaluation, dict[str, SplitPredictions]]:
        selected = tuple(sorted(set(subset)))
        if not selected:
            raise ValueError("Cannot evaluate an empty subset")
        if fidelity not in self.fidelities:
            raise KeyError(f"Unknown fidelity: {fidelity}")

        np = _require("numpy", "numpy>=1.26,<3")
        prediction_cache = self._prediction_cache_path(
            selected, fidelity, holdout_samples
        )
        cached_evaluation = self.cache.get(selected, fidelity)
        if cached_evaluation is not None and prediction_cache.exists():
            with np.load(prediction_cache, allow_pickle=False) as cached:
                split_predictions = {
                    name: SplitPredictions(
                        split=name,
                        y_true=cached[f"{name}__y"],
                        prediction=cached[f"{name}__prediction"],
                        weight=cached[f"{name}__weight"],
                        seed_metrics=cached[f"{name}__seed_metrics"].tolist(),
                        groups=(
                            cached[f"{name}__groups"]
                            if f"{name}__groups" in cached.files
                            else None
                        ),
                    )
                    for name in holdout_samples
                }
            return cached_evaluation, split_predictions
        catboost = _require("catboost", "catboost>=1.2.8")
        level = self.fidelities[fidelity]
        categorical = [
            feature
            for feature in selected
            if feature in self.cfg.data.categorical_features
        ]
        X_train, y_train, w_train = self._load(level.train_sample, selected)
        X_valid, y_valid, w_valid = self._load(level.valid_sample, selected)
        train_pool = catboost.Pool(
            X_train,
            label=y_train,
            weight=w_train,
            cat_features=categorical,
            feature_names=list(selected),
        )
        valid_pool = catboost.Pool(
            X_valid,
            label=y_valid,
            weight=w_valid,
            cat_features=categorical,
            feature_names=list(selected),
        )
        holdouts = {
            name: self._load_holdout(str(path), selected)
            for name, path in holdout_samples.items()
        }
        prediction_sums = {
            name: np.zeros(len(y_true), dtype="float64")
            for name, (_, y_true, _, _) in holdouts.items()
        }
        holdout_seed_metrics: dict[str, list[float]] = {
            name: [] for name in holdouts
        }
        start = time.perf_counter()
        train_scores: list[float] = []
        valid_scores: list[float] = []
        train_gini_scores: list[float] = []
        valid_gini_scores: list[float] = []
        valid_prediction_sum = np.zeros(len(y_valid), dtype="float64")
        metric_rows: list[dict[str, float]] = []

        with self._training_slot() as device:
            for seed in level.seeds:
                model = catboost.CatBoostClassifier(
                    **self._params(level, seed, device)
                )
                model.fit(
                    train_pool,
                    eval_set=valid_pool,
                    use_best_model=True,
                    early_stopping_rounds=50,
                )
                pred_train = model.predict_proba(train_pool)[:, 1]
                pred_valid = model.predict_proba(valid_pool)[:, 1]
                train_scores.append(
                    _metric_value(
                        self.primary_metric, y_train, pred_train, w_train
                    )
                )
                valid_scores.append(
                    _metric_value(
                        self.primary_metric, y_valid, pred_valid, w_valid
                    )
                )
                train_gini_scores.append(_metric_value("gini", y_train, pred_train, w_train))
                valid_gini_scores.append(_metric_value("gini", y_valid, pred_valid, w_valid))
                valid_prediction_sum += pred_valid
                metric_rows.append(
                    {
                        "average_precision": _metric_value(
                            "average_precision", y_valid, pred_valid, w_valid
                        ),
                        "roc_auc": _metric_value(
                            "roc_auc", y_valid, pred_valid, w_valid
                        ),
                        "gini": _metric_value(
                            "gini", y_valid, pred_valid, w_valid
                        ),
                        "logloss": _metric_value(
                            "logloss", y_valid, pred_valid, w_valid
                        ),
                        "brier": _metric_value(
                            "brier", y_valid, pred_valid, w_valid
                        ),
                    }
                )
                for name, (
                    X_holdout,
                    y_holdout,
                    w_holdout,
                    _,
                ) in holdouts.items():
                    prediction = model.predict_proba(X_holdout)[:, 1]
                    prediction_sums[name] += prediction
                    holdout_seed_metrics[name].append(
                        _metric_value(
                            self.primary_metric,
                            y_holdout,
                            prediction,
                            w_holdout,
                        )
                    )

        train_metric = float(np.mean(train_scores))
        valid_metric = float(np.mean(valid_scores))
        gap = max(0.0, train_metric - valid_metric)
        train_gini = float(np.mean(train_gini_scores))
        valid_gini = float(np.mean(valid_gini_scores))
        gini_gap = abs(train_gini - valid_gini)
        primary_gap_violation = (
            max(0.0, gap - self.cfg.search.max_primary_metric_gap)
            if self.cfg.search.max_primary_metric_gap is not None
            else 0.0
        )
        gini_gap_violation = max(0.0, gini_gap - self.cfg.search.max_gini_gap)
        aggregate_metrics = {
            name: float(np.mean([row[name] for row in metric_rows]))
            for name in metric_rows[0]
        }
        aggregate_metrics.update({
            "train_gini": train_gini,
            "valid_gini": valid_gini,
            "gini_gap": gini_gap,
        })
        threshold_violation = 0.0
        if (
            self.cfg.decision_threshold.enabled
            and self.cfg.decision_threshold.enforce_during_search
            and fidelity in self.cfg.decision_threshold.enforce_fidelities
        ):
            threshold_result = select_decision_threshold(
                selected,
                SplitPredictions(
                    split=level.valid_sample,
                    y_true=y_valid,
                    prediction=valid_prediction_sum / len(level.seeds),
                    weight=w_valid,
                    seed_metrics=valid_scores,
                ),
                None,
                self.cfg.decision_threshold,
            )
            aggregate_metrics.update({
                "search_threshold": threshold_result.threshold,
                "search_threshold_precision": threshold_result.tuning_precision,
                "search_threshold_recall": threshold_result.tuning_recall,
                "search_threshold_f1": threshold_result.tuning_f1,
                "search_threshold_feasible": float(threshold_result.tuning_feasible),
            })
            if not threshold_result.tuning_feasible:
                threshold_violation = (
                    max(0.0, self.cfg.decision_threshold.min_precision - threshold_result.tuning_precision)
                    + max(0.0, self.cfg.decision_threshold.min_recall - threshold_result.tuning_recall)
                )
        evaluation = Evaluation(
            subset=selected,
            fidelity=fidelity,
            valid_metric=valid_metric,
            train_metric=train_metric,
            gap=gap,
            metric_std=float(np.std(valid_scores)),
            n_features=len(selected),
            runtime_seconds=time.perf_counter() - start,
            metrics=aggregate_metrics,
            constraint_violation=primary_gap_violation + gini_gap_violation + threshold_violation,
        )
        self.cache.put(evaluation)
        split_predictions = {
            name: SplitPredictions(
                split=name,
                y_true=y_true,
                prediction=prediction_sums[name] / len(level.seeds),
                weight=weight,
                seed_metrics=holdout_seed_metrics[name],
                groups=groups,
            )
            for name, (_, y_true, weight, groups) in holdouts.items()
        }
        if self.cfg.output.save_trial_predictions or force_save_predictions:
            arrays: dict[str, Any] = {}
            for name, item in split_predictions.items():
                arrays[f"{name}__y"] = item.y_true
                arrays[f"{name}__prediction"] = item.prediction
                arrays[f"{name}__weight"] = item.weight
                arrays[f"{name}__seed_metrics"] = np.asarray(
                    item.seed_metrics, dtype="float64"
                )
                if item.groups is not None:
                    arrays[f"{name}__groups"] = item.groups
            temporary = prediction_cache.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(prediction_cache)
        return evaluation, split_predictions

    def interaction_pairs(
        self, subset: Sequence[str], fidelity: str, limit: int
    ) -> list[tuple[str, str, float]]:
        selected = tuple(sorted(set(subset)))
        cache_path = self._interaction_cache_path(selected, fidelity, limit)
        if cache_path.exists():
            return [
                (row["first"], row["second"], float(row["strength"]))
                for row in json.loads(cache_path.read_text(encoding="utf-8"))
            ]
        catboost = _require("catboost", "catboost>=1.2.8")
        level = self.fidelities[fidelity]
        categorical = [feature for feature in selected if feature in self.cfg.data.categorical_features]
        X_train, y_train, w_train = self._load(level.train_sample, selected)
        X_valid, y_valid, w_valid = self._load(level.valid_sample, selected)
        train_pool = catboost.Pool(
            X_train,
            label=y_train,
            weight=w_train,
            cat_features=categorical,
            feature_names=list(selected),
        )
        valid_pool = catboost.Pool(
            X_valid,
            label=y_valid,
            weight=w_valid,
            cat_features=categorical,
            feature_names=list(selected),
        )
        with self._training_slot() as device:
            model = catboost.CatBoostClassifier(
                **self._params(level, level.seeds[0], device)
            )
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
                early_stopping_rounds=50,
            )
        raw = model.get_feature_importance(data=valid_pool, type="Interaction")
        pairs: list[tuple[str, str, float]] = []
        for first, second, strength in raw:
            pairs.append((selected[int(first)], selected[int(second)], float(strength)))
        result = sorted(pairs, key=lambda row: row[2], reverse=True)[:limit]
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                [
                    {"first": first, "second": second, "strength": strength}
                    for first, second, strength in result
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        return result


def _distributed_catboost_trial(payload: dict[str, Any]) -> dict[str, Any]:
    """Stateless Ray task; all durable state lives in the shared evaluation cache."""
    cfg = AppConfig.from_dict(payload["config"])
    cfg.execution.backend = "local"
    cfg.execution.parallel_trials = 1
    cfg.execution.gpu_devices = []
    fidelities = [FidelityConfig(**item) for item in payload["fidelities"]]
    subset = tuple(payload["subset"])
    fidelity = str(payload["fidelity"])
    errors: list[tuple[str, str]] = []
    attempts = int(payload["trial_max_retries"]) + 1
    for attempt in range(1, attempts + 1):
        try:
            evaluator = CatBoostSubsetEvaluator(
                cfg,
                fidelities,
                EvaluationCache(
                    payload["cache_directory"], payload["fingerprint"]
                ),
            )
            evaluation = evaluator.evaluate(subset, fidelity)
            diagnostic = None
            if errors:
                error_type, error_message = errors[-1]
                diagnostic = asdict(
                    TrialDiagnostic(
                        subset=tuple(sorted(subset)),
                        fidelity=fidelity,
                        backend="ray",
                        outcome="recovered",
                        attempts=attempt,
                        error_type=error_type,
                        error_message=error_message,
                        oom_detected=any(
                            _is_oom_error(message) for _, message in errors
                        ),
                    )
                )
            return {
                "evaluation": asdict(evaluation),
                "diagnostic": diagnostic,
            }
        except Exception as exc:  # task boundary must return a durable diagnostic
            errors.append((type(exc).__name__, str(exc)[:2_000]))
            if attempt < attempts and payload["retry_backoff_seconds"]:
                time.sleep(float(payload["retry_backoff_seconds"]))
    error_type, error_message = errors[-1]
    return {
        "evaluation": None,
        "diagnostic": asdict(
            TrialDiagnostic(
                subset=tuple(sorted(subset)),
                fidelity=fidelity,
                backend="ray",
                outcome="failed",
                attempts=attempts,
                error_type=error_type,
                error_message=error_message,
                oom_detected=any(
                    _is_oom_error(message) for _, message in errors
                ),
            )
        ),
    }


def parquet_inventory(path: str) -> dict[str, Any]:
    pyarrow_parquet = _require("pyarrow.parquet", "pyarrow>=16,<24")
    sources = _parquet_sources(path)
    files = [sources] if isinstance(sources, str) else list(sources)
    rows = 0
    bytes_total = 0
    columns: list[str] = []
    for index, item in enumerate(files):
        file_path = Path(item)
        metadata = pyarrow_parquet.ParquetFile(file_path).metadata
        rows += int(metadata.num_rows)
        bytes_total += file_path.stat().st_size
        if index == 0:
            columns = list(metadata.schema.names)
    return {
        "path": path,
        "files": len(files),
        "rows": rows,
        "size_gb": bytes_total / float(1024**3),
        "columns": len(columns),
        "column_names": columns,
    }


def _isolated_catboost_trial(
    payload: dict[str, Any], channel: Any
) -> None:
    """One CatBoost evaluation in a killable local worker process."""
    stop = threading.Event()
    heartbeat_interval = float(payload["heartbeat_interval_seconds"])

    def heartbeat() -> None:
        while not stop.wait(heartbeat_interval):
            channel.put(
                {
                    "kind": "heartbeat",
                    "timestamp": time.monotonic(),
                    "pid": os.getpid(),
                }
            )

    channel.put(
        {
            "kind": "started",
            "timestamp": time.monotonic(),
            "pid": os.getpid(),
        }
    )
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        cfg = AppConfig.from_dict(payload["config"])
        cfg.execution.backend = "local"
        cfg.execution.local_trial_mode = "thread"
        cfg.execution.parallel_trials = 1
        device = payload.get("gpu_device")
        cfg.execution.gpu_devices = [str(device)] if device is not None else []
        evaluator = CatBoostSubsetEvaluator(
            cfg,
            [FidelityConfig(**item) for item in payload["fidelities"]],
            EvaluationCache(
                payload["cache_directory"], payload["fingerprint"]
            ),
        )
        action = str(payload.get("action", "evaluate"))
        interactions = None
        if action == "evaluate":
            evaluation = evaluator.evaluate(
                payload["subset"],
                payload["fidelity"],
                use_cache=bool(payload.get("use_cache", True)),
            )
        elif action == "evaluate_with_holdouts":
            evaluation, _ = evaluator.evaluate_with_holdouts(
                payload["subset"],
                payload["fidelity"],
                payload["holdout_samples"],
                force_save_predictions=True,
            )
        elif action == "interaction_pairs":
            interactions = evaluator.interaction_pairs(
                payload["subset"],
                payload["fidelity"],
                int(payload["interaction_limit"]),
            )
            evaluation = None
        else:
            raise ValueError(f"Unknown isolated action: {action}")
        channel.put(
            {
                "kind": "result",
                "evaluation": (
                    asdict(evaluation) if evaluation is not None else None
                ),
                "interactions": interactions,
                "action": action,
                "error_type": None,
                "error_message": None,
                "oom_detected": False,
            }
        )
    except BaseException as exc:
        channel.put(
            {
                "kind": "result",
                "evaluation": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2_000],
                "oom_detected": _is_oom_error(exc),
            }
        )
    finally:
        stop.set()
        thread.join(timeout=min(heartbeat_interval, 1.0))


@dataclass(slots=True)
class IsolatedWorkerState:
    index: int
    subset: frozenset[str]
    attempt: int
    process: Any
    channel: Any
    started_at: float
    last_heartbeat: float
    gpu_device: str | None
    peak_rss_gb: float = 0.0
    peak_gpu_gb: float = 0.0
    worker_pid: int | None = None


def run_isolated_catboost_action(
    evaluator: CatBoostSubsetEvaluator,
    resource_manager: ResourceManager | None,
    action: str,
    subset: Sequence[str],
    fidelity: str,
    **action_payload: Any,
) -> dict[str, Any]:
    """Run one non-search CatBoost action behind the same watchdog."""
    execution = evaluator.cfg.execution
    if execution.process_start_method == "forkserver":
        mp.set_forkserver_preload(
            [
                "interaction_subset_selector",
                "catboost",
                "numpy",
                "polars",
                "pyarrow",
            ]
        )
    context = mp.get_context(execution.process_start_method)
    fidelities = []
    for item in evaluator.fidelities.values():
        row = asdict(item)
        row["train_sample"] = str(Path(row["train_sample"]).resolve())
        row["valid_sample"] = str(Path(row["valid_sample"]).resolve())
        fidelities.append(row)
    task_type = str(
        evaluator.cfg.model_params.get("task_type", "CPU")
    ).upper()
    gpu_device = None
    if task_type == "GPU" and execution.gpu_devices:
        eligible = [
            str(device)
            for device in execution.gpu_devices
            if resource_manager is None
            or resource_manager.gpu_device_available(str(device))
        ]
        if not eligible:
            raise RuntimeError(
                "No selected GPU currently satisfies the memory limits"
            )
        gpu_device = eligible[0]
    common = {
        "config": asdict(evaluator.cfg),
        "fidelities": fidelities,
        "cache_directory": str(evaluator.cache.directory.resolve()),
        "fingerprint": evaluator.cache.fingerprint,
        "fidelity": fidelity,
        "subset": sorted(set(subset)),
        "gpu_device": gpu_device,
        "heartbeat_interval_seconds": execution.heartbeat_interval_seconds,
        "action": action,
        **action_payload,
    }
    errors: list[str] = []
    for attempt in range(1, execution.trial_max_retries + 2):
        if resource_manager is not None:
            resource_manager.wait_for_single_trial(f"{action}_wait")
        channel = context.Queue()
        process = context.Process(
            target=_isolated_catboost_trial,
            args=(common, channel),
            daemon=False,
            name=f"feature-{action}-{attempt}",
        )
        started = time.monotonic()
        last_heartbeat = started
        peak_rss_gb = 0.0
        peak_gpu_gb = 0.0
        process.start()
        result: dict[str, Any] | None = None
        termination_reason = None
        interval = (
            resource_manager.config.monitoring_interval_seconds
            if resource_manager is not None
            else min(0.25, execution.heartbeat_interval_seconds)
        )
        while result is None and termination_reason is None:
            now = time.monotonic()
            try:
                while True:
                    message = channel.get_nowait()
                    if message.get("kind") in {"started", "heartbeat"}:
                        last_heartbeat = now
                    elif message.get("kind") == "result":
                        result = message
                        break
            except Empty:
                pass
            pid = process.pid
            peak_rss_gb = max(
                peak_rss_gb, _process_tree_rss_gb(pid)
            )
            if task_type == "GPU":
                peak_gpu_gb = max(
                    peak_gpu_gb, _process_gpu_memory_gb(pid)
                )
            if resource_manager is not None:
                if (
                    peak_rss_gb
                    > resource_manager.hard_ram_limit_gb(fidelity)
                ):
                    termination_reason = "ram_limit"
                elif (
                    task_type == "GPU"
                    and peak_gpu_gb
                    > resource_manager.hard_gpu_limit_gb(fidelity)
                ):
                    termination_reason = "gpu_memory_limit"
            if (
                termination_reason is None
                and now - started > execution.trial_timeout_seconds
            ):
                termination_reason = "timeout"
            if (
                termination_reason is None
                and now - last_heartbeat
                > execution.heartbeat_timeout_seconds
            ):
                termination_reason = "heartbeat_timeout"
            if result is None and termination_reason is None:
                if not process.is_alive():
                    try:
                        final_message = channel.get(timeout=0.10)
                        if final_message.get("kind") == "result":
                            result = final_message
                        else:
                            termination_reason = "unexpected_exit"
                    except Empty:
                        termination_reason = "unexpected_exit"
                else:
                    time.sleep(min(interval, 0.25))
        runtime = time.monotonic() - started
        if termination_reason is not None:
            _stop_process(process, execution.terminate_grace_seconds)
            if resource_manager is not None:
                resource_manager.report_termination(termination_reason)
            errors.append(termination_reason)
        else:
            process.join(timeout=0.25)
            if process.is_alive():
                _stop_process(process, execution.terminate_grace_seconds)
            if result is not None and result.get("error_type") is not None:
                errors.append(
                    f"{result['error_type']}: {result['error_message']}"
                )
                if result.get("oom_detected") and resource_manager is not None:
                    resource_manager.report_oom()
            elif result is not None:
                if resource_manager is not None:
                    resource_manager.observe_isolated_trial(
                        fidelity,
                        len(set(subset)),
                        runtime,
                        peak_rss_gb,
                        peak_gpu_gb,
                    )
                try:
                    channel.close()
                except (AttributeError, OSError):
                    pass
                return result
        try:
            channel.close()
        except (AttributeError, OSError):
            pass
        if attempt <= execution.trial_max_retries:
            if execution.retry_backoff_seconds:
                time.sleep(execution.retry_backoff_seconds)
            continue
    raise RuntimeError(
        f"Isolated {action} failed after {len(errors)} attempt(s): "
        f"{errors[-1] if errors else 'unknown worker failure'}"
    )


def _dominates(left: Evaluation, right: Evaluation) -> bool:
    if left.constraint_violation < right.constraint_violation:
        return True
    if left.constraint_violation > right.constraint_violation:
        return False
    no_worse = (
        left.valid_metric >= right.valid_metric
        and left.n_features <= right.n_features
        and left.metric_std <= right.metric_std
    )
    strictly_better = (
        left.valid_metric > right.valid_metric
        or left.n_features < right.n_features
        or left.metric_std < right.metric_std
    )
    return no_worse and strictly_better


def pareto_front(evaluations: Sequence[Evaluation]) -> list[Evaluation]:
    unique = {evaluation.subset: evaluation for evaluation in evaluations}.values()
    result = [
        candidate
        for candidate in unique
        if not any(
            _dominates(other, candidate)
            for other in unique
            if other.subset != candidate.subset
        )
    ]
    return sorted(result, key=lambda item: (-item.valid_metric, item.n_features, item.metric_std))


def _crowding(front: Sequence[Evaluation]) -> dict[tuple[str, ...], float]:
    distance = {item.subset: 0.0 for item in front}
    objectives = [
        (lambda item: item.valid_metric),
        (lambda item: -item.n_features),
        (lambda item: -item.metric_std),
    ]
    for objective in objectives:
        ordered = sorted(front, key=objective)
        distance[ordered[0].subset] = math.inf
        distance[ordered[-1].subset] = math.inf
        low, high = objective(ordered[0]), objective(ordered[-1])
        if high == low:
            continue
        for index in range(1, len(ordered) - 1):
            previous = objective(ordered[index - 1])
            following = objective(ordered[index + 1])
            distance[ordered[index].subset] += (following - previous) / (high - low)
    return distance


def _non_dominated_fronts(evaluations: Sequence[Evaluation]) -> list[list[Evaluation]]:
    remaining = list({evaluation.subset: evaluation for evaluation in evaluations}.values())
    fronts: list[list[Evaluation]] = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                _dominates(other, candidate)
                for other in remaining
                if other.subset != candidate.subset
            )
        ]
        fronts.append(front)
        selected = {item.subset for item in front}
        remaining = [item for item in remaining if item.subset not in selected]
    return fronts


def nsga_select(evaluations: Sequence[Evaluation], size: int) -> list[Evaluation]:
    result: list[Evaluation] = []
    for front in _non_dominated_fronts(evaluations):
        if len(result) + len(front) <= size:
            result.extend(front)
            continue
        crowding = _crowding(front)
        front = sorted(front, key=lambda item: crowding[item.subset], reverse=True)
        result.extend(front[: size - len(result)])
        break
    return result


def _bounded_combinations(
    members: Sequence[str],
    order: int,
    limit: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    values = tuple(sorted(set(members)))
    if len(values) < order or limit <= 0:
        return []
    total = math.comb(len(values), order)
    if total <= limit:
        return list(combinations(values, order))
    sampled: set[tuple[str, ...]] = set()
    attempts = 0
    while len(sampled) < limit and attempts < limit * 20:
        sampled.add(tuple(sorted(rng.sample(values, order))))
        attempts += 1
    return sorted(sampled)


def learn_coalitions(
    evaluations: Sequence[Evaluation],
    config: SearchConfig,
    seed: int,
) -> list[CoalitionScore]:
    unique = list({item.subset: item for item in evaluations}.values())
    if len(unique) < 2:
        return []
    ordered = sorted(
        unique,
        key=lambda item: (
            item.constraint_violation,
            -item.valid_metric,
            item.n_features,
            item.metric_std,
        ),
    )
    elite_size = max(1, math.ceil(len(ordered) * config.elite_fraction))
    elite_subsets = {item.subset for item in ordered[:elite_size]}
    global_metric = sum(item.valid_metric for item in unique) / len(unique)
    metric_scale = math.sqrt(
        sum((item.valid_metric - global_metric) ** 2 for item in unique) / len(unique)
    )
    metric_scale = max(metric_scale, 1e-12)
    rng = random.Random(seed)
    stats: dict[tuple[str, ...], list[float]] = {}

    for item in sorted(unique, key=lambda row: row.subset):
        coalitions = [
            *_bounded_combinations(
                item.subset, 2, config.coalition_pairs_per_subset, rng
            ),
            *_bounded_combinations(
                item.subset, 3, config.coalition_triples_per_subset, rng
            ),
        ]
        for coalition in coalitions:
            row = stats.setdefault(coalition, [0.0, 0.0, 0.0])
            row[0] += 1
            row[2] += item.valid_metric
            if item.subset in elite_subsets:
                row[1] += 1

    smoothing = config.coalition_smoothing
    result: list[CoalitionScore] = []
    for members, (raw_count, raw_elite_count, total_metric) in stats.items():
        count = int(raw_count)
        if count < config.coalition_min_support:
            continue
        elite_count = int(raw_elite_count)
        if elite_count == 0:
            continue
        elite_rate = (elite_count + smoothing) / (elite_size + 2 * smoothing)
        overall_rate = (count + smoothing) / (len(unique) + 2 * smoothing)
        lift = math.log(max(elite_rate, 1e-12) / max(overall_rate, 1e-12))
        mean_metric = (
            total_metric + smoothing * global_metric
        ) / (count + smoothing)
        metric_gain = mean_metric - global_metric
        normalized_gain = max(-5.0, min(5.0, metric_gain / metric_scale))
        confidence = count / (count + smoothing)
        score = max(0.0, lift + normalized_gain) * confidence
        if score <= 0:
            continue
        result.append(
            CoalitionScore(
                members=members,
                order=len(members),
                support=count,
                elite_support=elite_count,
                lift=lift,
                mean_metric=mean_metric,
                metric_gain=metric_gain,
                score=score,
            )
        )

    pairs = sorted(
        (item for item in result if item.order == 2),
        key=lambda item: (-item.score, -item.support, item.members),
    )[: config.coalition_top_pairs]
    triples = sorted(
        (item for item in result if item.order == 3),
        key=lambda item: (-item.score, -item.support, item.members),
    )[: config.coalition_top_triples]
    return [*pairs, *triples]


def _evaluation_from_payload(payload: dict[str, Any]) -> Evaluation:
    row = dict(payload)
    row["subset"] = tuple(row["subset"])
    return Evaluation(**row)


def _coalition_from_payload(payload: dict[str, Any]) -> CoalitionScore:
    row = dict(payload)
    row["members"] = tuple(row["members"])
    return CoalitionScore(**row)


def _trial_diagnostic_from_payload(payload: dict[str, Any]) -> TrialDiagnostic:
    row = dict(payload)
    row["subset"] = tuple(row["subset"])
    return TrialDiagnostic(**row)


def _promotion_batch_from_payload(payload: dict[str, Any]) -> PromotionBatch:
    return PromotionBatch(**payload)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


class InteractionAwareSearch:
    def __init__(
        self,
        features: Sequence[str],
        evaluator: SubsetEvaluator,
        config: SearchConfig,
        interaction_pairs: Sequence[tuple[str, str, float]] = (),
        required_features: Sequence[str] = (),
        execution: ExecutionConfig | None = None,
        resource_manager: ResourceManager | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_fingerprint: str | None = None,
    ):
        self.features = tuple(sorted(set(features)))
        self.feature_set = set(self.features)
        self.required_features = frozenset(required_features) & self.feature_set
        self.evaluator = evaluator
        self.config = config
        self.execution = execution or ExecutionConfig()
        self.resource_manager = resource_manager
        self.parallel_trials = max(
            1,
            (
                resource_manager.max_parallel_trials
                if resource_manager is not None
                else self.execution.parallel_trials
            ),
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.interaction_pairs = [
            (first, second, strength)
            for first, second, strength in interaction_pairs
            if first in self.feature_set and second in self.feature_set and first != second
        ]
        max_interaction = max(
            (strength for _, _, strength in self.interaction_pairs), default=0.0
        )
        self.interaction_priors = [
            CoalitionScore(
                members=tuple(sorted((first, second))),
                order=2,
                support=0,
                elite_support=0,
                lift=0.0,
                mean_metric=0.0,
                metric_gain=0.0,
                score=(
                    config.interaction_prior_weight * strength / max_interaction
                    if max_interaction > 0
                    else 0.0
                ),
                source="catboost",
            )
            for first, second, strength in self.interaction_pairs
        ]
        self.coalitions: list[CoalitionScore] = list(self.interaction_priors)
        self._coalitions_by_feature: dict[str, list[CoalitionScore]] = {}
        self._index_coalitions()
        self.history: list[Evaluation] = []
        self.screening_history: list[Evaluation] = []
        self.trial_diagnostics: list[TrialDiagnostic] = []
        self.promotion_batches: list[PromotionBatch] = []
        self.restart_winners: list[tuple[str, ...]] = []
        self._completed_restart_fronts: list[list[Evaluation]] = []
        self._active_checkpoint: dict[str, Any] | None = None
        self._checkpoint_complete_front: list[Evaluation] | None = None
        self._progress_restart = 0

    def _progress(self, name: str, **details: Any) -> None:
        if self.execution.show_progress:
            _event(name, **details)

    def _index_coalitions(self) -> None:
        index: dict[str, list[CoalitionScore]] = {}
        for item in self.coalitions:
            for feature in item.members:
                index.setdefault(feature, []).append(item)
        self._coalitions_by_feature = index

    def _refresh_coalitions(
        self, evaluations: Sequence[Evaluation], seed: int
    ) -> None:
        observed = learn_coalitions(
            evaluations[-self.config.coalition_history_limit :],
            self.config,
            seed,
        )
        merged = {item.members: item for item in observed}
        for prior in self.interaction_priors:
            current = merged.get(prior.members)
            if current is None:
                merged[prior.members] = prior
                continue
            merged[prior.members] = CoalitionScore(
                members=current.members,
                order=current.order,
                support=current.support,
                elite_support=current.elite_support,
                lift=current.lift,
                mean_metric=current.mean_metric,
                metric_gain=current.metric_gain,
                score=current.score + prior.score,
                source="observed+catboost",
            )
        self.coalitions = sorted(
            (item for item in merged.values() if item.score > 0),
            key=lambda item: (-item.score, -item.support, item.members),
        )
        self._index_coalitions()

    def _choose_coalition(
        self,
        rng: random.Random,
        selected: set[str],
        target_size: int,
    ) -> CoalitionScore | None:
        candidates = [
            item
            for item in self.coalitions
            if set(item.members) - selected
            and len(selected | set(item.members)) <= target_size
        ]
        if not candidates:
            return None
        return rng.choices(
            candidates,
            weights=[max(item.score, 1e-12) for item in candidates],
            k=1,
        )[0]

    def _coalition_affinity(self, feature: str, selected: set[str]) -> float:
        if not selected or not self.coalitions:
            return 0.0
        max_score = max(self.coalitions[0].score, 1e-12)
        affinity = 0.0
        for item in self._coalitions_by_feature.get(feature, []):
            partners = set(item.members) - {feature}
            overlap = len(partners & selected)
            if overlap:
                affinity += (item.score / max_score) * overlap / len(partners)
        return affinity

    def _random_subset(
        self,
        rng: random.Random,
        weights: dict[str, float] | None = None,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> frozenset[str]:
        minimum = minimum or self.config.subspace_min_features
        maximum = maximum or self.config.subspace_max_features
        maximum = min(maximum, (self.config.search_max_features or self.config.max_features), len(self.features))
        minimum = max(minimum, self.config.min_features, len(self.required_features))
        minimum = min(minimum, maximum)
        size = rng.randint(minimum, maximum)
        if not weights:
            optional = sorted(self.feature_set - self.required_features)
            selected = set(self.required_features)
            selected.update(rng.sample(optional, size - len(selected)))
            return frozenset(selected)
        selected = set(self.required_features)
        if self.coalitions and rng.random() < self.config.coalition_sampling_probability:
            coalition = self._choose_coalition(rng, selected, size)
            if coalition:
                selected.update(coalition.members)
        pool = sorted(self.feature_set - selected)
        while len(selected) < size:
            probabilities = [
                max(weights.get(feature, 0.0), 1e-6)
                * (1.0 + self._coalition_affinity(feature, selected))
                for feature in pool
            ]
            chosen = rng.choices(pool, weights=probabilities, k=1)[0]
            index = pool.index(chosen)
            selected.add(chosen)
            pool.pop(index)
        return frozenset(selected)

    def _unique_subsets(
        self, subsets: Iterable[Iterable[str]]
    ) -> list[frozenset[str]]:
        result: list[frozenset[str]] = []
        seen: set[frozenset[str]] = set()
        for subset in subsets:
            selected = frozenset(subset)
            if not selected or selected in seen:
                continue
            seen.add(selected)
            result.append(selected)
        return result

    def _evaluate_many_raw(
        self,
        subsets: Sequence[frozenset[str]],
        fidelity: str,
    ) -> list[Evaluation]:
        if not subsets:
            return []
        if self.execution.backend == "ray":
            outcomes = self._evaluate_many_ray(subsets, fidelity)
        elif (
            self.execution.local_trial_mode == "process"
            and isinstance(self.evaluator, CatBoostSubsetEvaluator)
        ):
            outcomes = self._evaluate_many_process(subsets, fidelity)
        else:
            outcomes = self._evaluate_many_local(subsets, fidelity)
        evaluations: list[Evaluation] = []
        failures: list[TrialDiagnostic] = []
        for evaluation, diagnostic in outcomes:
            if diagnostic is not None:
                self.trial_diagnostics.append(diagnostic)
                if diagnostic.outcome == "failed":
                    failures.append(diagnostic)
            if evaluation is not None:
                evaluations.append(evaluation)
        if failures and self.execution.trial_failure_policy == "fail_fast":
            first = failures[0]
            raise RuntimeError(
                f"Trial failed after {first.attempts} attempts for "
                f"{first.subset} at {first.fidelity}: "
                f"{first.error_type}: {first.error_message}"
            )
        minimum_successes = max(
            1,
            math.ceil(
                len(subsets) * self.execution.minimum_successful_fraction
            ),
        )
        if len(evaluations) < minimum_successes:
            raise RuntimeError(
                f"Only {len(evaluations)}/{len(subsets)} trials succeeded at "
                f"{fidelity}; at least {minimum_successes} are required"
            )
        if failures:
            _event(
                "trial_batch_degraded",
                fidelity=fidelity,
                submitted=len(subsets),
                succeeded=len(evaluations),
                failed=len(failures),
            )
        return evaluations

    def _evaluate_many_process(
        self,
        subsets: Sequence[frozenset[str]],
        fidelity: str,
        use_cache: bool = True,
    ) -> list[tuple[Evaluation | None, TrialDiagnostic | None]]:
        if not isinstance(self.evaluator, CatBoostSubsetEvaluator):
            raise TypeError(
                "Process isolation supports CatBoostSubsetEvaluator only"
            )
        if not subsets:
            return []
        if self.execution.process_start_method == "forkserver":
            mp.set_forkserver_preload(
                [
                    "interaction_subset_selector",
                    "catboost",
                    "numpy",
                    "polars",
                    "pyarrow",
                ]
            )
        context = mp.get_context(self.execution.process_start_method)
        fidelity_payloads = []
        for item in self.evaluator.fidelities.values():
            row = asdict(item)
            row["train_sample"] = str(Path(row["train_sample"]).resolve())
            row["valid_sample"] = str(Path(row["valid_sample"]).resolve())
            fidelity_payloads.append(row)
        common = {
            "config": asdict(self.evaluator.cfg),
            "fidelities": fidelity_payloads,
            "cache_directory": str(self.evaluator.cache.directory.resolve()),
            "fingerprint": self.evaluator.cache.fingerprint,
            "fidelity": fidelity,
            "use_cache": use_cache,
            "heartbeat_interval_seconds": (
                self.execution.heartbeat_interval_seconds
            ),
        }
        pending: deque[tuple[int, frozenset[str]]] = deque()
        running: dict[int, IsolatedWorkerState] = {}
        results: dict[
            int, tuple[Evaluation | None, TrialDiagnostic | None]
        ] = {}
        for index, subset in enumerate(subsets):
            cached = (
                self.evaluator.cache.get(tuple(sorted(subset)), fidelity)
                if use_cache
                else None
            )
            if cached is not None:
                results[index] = (cached, None)
            else:
                pending.append((index, subset))
        attempts: Counter[int] = Counter()
        errors: dict[int, tuple[str, str]] = {}
        oom_seen: set[int] = set()
        terminations: dict[int, str] = {}
        worker_pids: dict[int, int] = {}
        peak_rss: Counter[int] = Counter()
        peak_gpu: Counter[int] = Counter()
        task_type = str(
            self.evaluator.cfg.model_params.get("task_type", "CPU")
        ).upper()
        free_devices: deque[str] = deque(
            str(item) for item in self.execution.gpu_devices
        )
        requires_device = task_type == "GPU" and bool(free_devices)
        interval = (
            self.resource_manager.config.monitoring_interval_seconds
            if self.resource_manager is not None
            else min(0.25, self.execution.heartbeat_interval_seconds)
        )
        wait_timeout = (
            self.resource_manager.config.resource_wait_timeout_seconds
            if self.resource_manager is not None
            else 300.0
        )
        last_progress = time.monotonic()
        last_report = last_progress

        def take_device() -> str | None:
            if not requires_device:
                return None
            for _ in range(len(free_devices)):
                device = free_devices.popleft()
                if (
                    self.resource_manager is None
                    or self.resource_manager.gpu_device_available(device)
                ):
                    return device
                free_devices.append(device)
            return None

        def close_worker(state: IsolatedWorkerState) -> None:
            state.process.join(timeout=0.25)
            if state.process.is_alive():
                _stop_process(
                    state.process,
                    self.execution.terminate_grace_seconds,
                )
            if state.gpu_device is not None:
                free_devices.append(state.gpu_device)
            try:
                state.channel.close()
            except (AttributeError, OSError):
                pass

        def finish_failure(
            state: IsolatedWorkerState,
            error_type: str,
            error_message: str,
            oom: bool,
            termination_reason: str | None,
        ) -> None:
            index = state.index
            errors[index] = (error_type, error_message)
            peak_rss[index] = max(peak_rss[index], state.peak_rss_gb)
            peak_gpu[index] = max(peak_gpu[index], state.peak_gpu_gb)
            if state.worker_pid is not None:
                worker_pids[index] = state.worker_pid
            if oom:
                oom_seen.add(index)
            if termination_reason:
                terminations[index] = termination_reason
                if self.resource_manager is not None:
                    self.resource_manager.report_termination(
                        termination_reason
                    )
            elif oom and self.resource_manager is not None:
                self.resource_manager.report_oom()
            if self.resource_manager is not None:
                self.resource_manager.observe_isolated_trial(
                    fidelity,
                    len(state.subset),
                    time.monotonic() - state.started_at,
                    state.peak_rss_gb,
                    state.peak_gpu_gb,
                )
            close_worker(state)
            if attempts[index] <= self.execution.trial_max_retries:
                if self.execution.retry_backoff_seconds:
                    time.sleep(self.execution.retry_backoff_seconds)
                pending.append((index, state.subset))
                return
            results[index] = (
                None,
                TrialDiagnostic(
                    subset=tuple(sorted(state.subset)),
                    fidelity=fidelity,
                    backend="local_process",
                    outcome="failed",
                    attempts=attempts[index],
                    error_type=error_type,
                    error_message=error_message,
                    oom_detected=oom or index in oom_seen,
                    concurrency_limit=(
                        self.resource_manager.current_parallel_limit
                        if self.resource_manager is not None
                        else self.parallel_trials
                    ),
                    termination_reason=termination_reason,
                    worker_pid=worker_pids.get(index),
                    peak_worker_rss_gb=float(peak_rss[index]),
                    peak_worker_gpu_gb=float(peak_gpu[index]),
                ),
            )

        while pending or running:
            allowed = (
                self.resource_manager.allowed_concurrency(
                    len(running), len(pending), "process_scheduler"
                )
                if self.resource_manager is not None
                else self.parallel_trials
            )
            submitted = False
            while pending and len(running) < allowed:
                device = take_device()
                if requires_device and device is None:
                    break
                index, subset = pending.popleft()
                attempts[index] += 1
                channel = context.Queue()
                process = context.Process(
                    target=_isolated_catboost_trial,
                    args=(
                        {
                            **common,
                            "subset": sorted(subset),
                            "gpu_device": device,
                        },
                        channel,
                    ),
                    daemon=False,
                    name=f"feature-trial-{index}-{attempts[index]}",
                )
                started = time.monotonic()
                process.start()
                running[index] = IsolatedWorkerState(
                    index=index,
                    subset=subset,
                    attempt=attempts[index],
                    process=process,
                    channel=channel,
                    started_at=started,
                    last_heartbeat=started,
                    gpu_device=device,
                    worker_pid=process.pid,
                )
                submitted = True
            if submitted and self.resource_manager is not None:
                self.resource_manager.allowed_concurrency(
                    len(running), len(pending), "process_submitted"
                )
            if not running:
                if time.monotonic() - last_progress >= wait_timeout:
                    raise RuntimeError(
                        "Resource wait timeout: no safe isolated-worker slot "
                        "became available"
                    )
                time.sleep(min(interval, 0.25))
                continue

            now = time.monotonic()
            finished: list[int] = []
            for index, state in list(running.items()):
                messages: list[dict[str, Any]] = []
                while True:
                    try:
                        messages.append(state.channel.get_nowait())
                    except Empty:
                        break
                for message in messages:
                    kind = message.get("kind")
                    if kind in {"started", "heartbeat"}:
                        state.last_heartbeat = now
                        state.worker_pid = int(
                            message.get("pid") or state.process.pid
                        )
                        continue
                    if kind != "result":
                        continue
                    peak_rss[index] = max(
                        peak_rss[index], state.peak_rss_gb
                    )
                    peak_gpu[index] = max(
                        peak_gpu[index], state.peak_gpu_gb
                    )
                    if state.worker_pid is not None:
                        worker_pids[index] = state.worker_pid
                    evaluation_payload = message.get("evaluation")
                    if evaluation_payload is None:
                        finish_failure(
                            state,
                            str(message.get("error_type") or "WorkerError"),
                            str(message.get("error_message") or "unknown worker error"),
                            bool(message.get("oom_detected")),
                            None,
                        )
                    else:
                        evaluation = _evaluation_from_payload(
                            evaluation_payload
                        )
                        if self.resource_manager is not None:
                            self.resource_manager.observe_isolated_trial(
                                fidelity,
                                len(state.subset),
                                time.monotonic() - state.started_at,
                                state.peak_rss_gb,
                                state.peak_gpu_gb,
                            )
                        diagnostic = None
                        if index in errors:
                            error_type, error_message = errors[index]
                            diagnostic = TrialDiagnostic(
                                subset=tuple(sorted(state.subset)),
                                fidelity=fidelity,
                                backend="local_process",
                                outcome="recovered",
                                attempts=attempts[index],
                                error_type=error_type,
                                error_message=error_message,
                                oom_detected=index in oom_seen,
                                concurrency_limit=(
                                    self.resource_manager.current_parallel_limit
                                    if self.resource_manager is not None
                                    else self.parallel_trials
                                ),
                                termination_reason=terminations.get(index),
                                worker_pid=worker_pids.get(index),
                                peak_worker_rss_gb=float(peak_rss[index]),
                                peak_worker_gpu_gb=float(peak_gpu[index]),
                            )
                        close_worker(state)
                        results[index] = (evaluation, diagnostic)
                    finished.append(index)
                    last_progress = now
                    break
                if index in finished:
                    continue

                pid = state.worker_pid or state.process.pid
                state.peak_rss_gb = max(
                    state.peak_rss_gb, _process_tree_rss_gb(pid)
                )
                if task_type == "GPU":
                    state.peak_gpu_gb = max(
                        state.peak_gpu_gb, _process_gpu_memory_gb(pid)
                    )
                termination_reason = None
                if self.resource_manager is not None:
                    if (
                        state.peak_rss_gb
                        > self.resource_manager.hard_ram_limit_gb(fidelity)
                    ):
                        termination_reason = "ram_limit"
                    elif (
                        task_type == "GPU"
                        and state.peak_gpu_gb
                        > self.resource_manager.hard_gpu_limit_gb(fidelity)
                    ):
                        termination_reason = "gpu_memory_limit"
                if (
                    termination_reason is None
                    and now - state.started_at
                    > self.execution.trial_timeout_seconds
                ):
                    termination_reason = "timeout"
                if (
                    termination_reason is None
                    and now - state.last_heartbeat
                    > self.execution.heartbeat_timeout_seconds
                ):
                    termination_reason = "heartbeat_timeout"
                if termination_reason is not None:
                    _stop_process(
                        state.process,
                        self.execution.terminate_grace_seconds,
                    )
                    finish_failure(
                        state,
                        "WorkerTerminated",
                        f"Isolated worker terminated: {termination_reason}",
                        termination_reason
                        in {"ram_limit", "gpu_memory_limit"},
                        termination_reason,
                    )
                    finished.append(index)
                    last_progress = now
                elif not state.process.is_alive():
                    try:
                        message = state.channel.get(timeout=0.10)
                    except Empty:
                        message = None
                    if message and message.get("kind") == "result":
                        if message.get("evaluation") is not None:
                            evaluation = _evaluation_from_payload(
                                message["evaluation"]
                            )
                            close_worker(state)
                            results[index] = (evaluation, None)
                        else:
                            finish_failure(
                                state,
                                str(message.get("error_type") or "WorkerError"),
                                str(message.get("error_message") or "unknown worker error"),
                                bool(message.get("oom_detected")),
                                None,
                            )
                    else:
                        finish_failure(
                            state,
                            "WorkerExit",
                            f"Worker exited with code {state.process.exitcode}",
                            state.process.exitcode in {-9, 137},
                            "unexpected_exit",
                        )
                    finished.append(index)
                    last_progress = now
            for index in finished:
                running.pop(index, None)
            if (
                self.execution.show_progress
                and now - last_report
                >= self.execution.progress_interval_seconds
            ):
                self._progress(
                    "trial_progress",
                    fidelity=fidelity,
                    completed=len(results),
                    total=len(subsets),
                    running=len(running),
                    queued=len(pending),
                )
                last_report = now
            if running:
                time.sleep(min(interval, 0.25))
        return [results[index] for index in range(len(subsets))]

    def _evaluate_many_local(
        self,
        subsets: Sequence[frozenset[str]],
        fidelity: str,
    ) -> list[tuple[Evaluation | None, TrialDiagnostic | None]]:
        if not subsets:
            return []
        pending: deque[tuple[int, frozenset[str]]] = deque(
            enumerate(subsets)
        )
        results: dict[
            int, tuple[Evaluation | None, TrialDiagnostic | None]
        ] = {}
        attempts: Counter[int] = Counter()
        errors: dict[int, tuple[str, str]] = {}
        oom_seen: set[int] = set()
        last_progress = time.monotonic()
        interval = (
            self.resource_manager.config.monitoring_interval_seconds
            if self.resource_manager is not None
            else 1.0
        )
        wait_timeout = (
            self.resource_manager.config.resource_wait_timeout_seconds
            if self.resource_manager is not None
            else 300.0
        )
        with ThreadPoolExecutor(max_workers=self.parallel_trials) as executor:
            running: dict[Any, tuple[int, frozenset[str]]] = {}
            while pending or running:
                allowed = (
                    self.resource_manager.allowed_concurrency(
                        len(running), len(pending), "local_scheduler"
                    )
                    if self.resource_manager is not None
                    else self.parallel_trials
                )
                submitted = False
                while pending and len(running) < allowed:
                    index, subset = pending.popleft()
                    attempts[index] += 1
                    future = executor.submit(
                        self.evaluator.evaluate, subset, fidelity
                    )
                    running[future] = (index, subset)
                    submitted = True
                if submitted and self.resource_manager is not None:
                    self.resource_manager.allowed_concurrency(
                        len(running), len(pending), "local_submitted"
                    )
                if not running:
                    if time.monotonic() - last_progress >= wait_timeout:
                        raise RuntimeError(
                            "Resource wait timeout: no safe CPU/RAM/GPU slot "
                            "became available for a new trial"
                        )
                    time.sleep(min(interval, 0.25))
                    continue
                completed, _ = wait(
                    running,
                    timeout=interval,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    continue
                for future in completed:
                    index, subset = running.pop(future)
                    try:
                        evaluation = future.result()
                        diagnostic = None
                        if index in errors:
                            error_type, error_message = errors[index]
                            diagnostic = TrialDiagnostic(
                                subset=tuple(sorted(subset)),
                                fidelity=fidelity,
                                backend="local",
                                outcome="recovered",
                                attempts=attempts[index],
                                error_type=error_type,
                                error_message=error_message,
                                oom_detected=index in oom_seen,
                                concurrency_limit=(
                                    self.resource_manager.current_parallel_limit
                                    if self.resource_manager is not None
                                    else self.parallel_trials
                                ),
                            )
                        results[index] = (evaluation, diagnostic)
                    except Exception as exc:
                        error = (type(exc).__name__, str(exc)[:2_000])
                        errors[index] = error
                        oom = _is_oom_error(exc)
                        if oom:
                            oom_seen.add(index)
                            if self.resource_manager is not None:
                                self.resource_manager.report_oom()
                        if attempts[index] <= self.execution.trial_max_retries:
                            if self.execution.retry_backoff_seconds:
                                time.sleep(
                                    self.execution.retry_backoff_seconds
                                )
                            pending.append((index, subset))
                        else:
                            results[index] = (
                                None,
                                TrialDiagnostic(
                                    subset=tuple(sorted(subset)),
                                    fidelity=fidelity,
                                    backend="local",
                                    outcome="failed",
                                    attempts=attempts[index],
                                    error_type=error[0],
                                    error_message=error[1],
                                    oom_detected=oom or index in oom_seen,
                                    concurrency_limit=(
                                        self.resource_manager.current_parallel_limit
                                        if self.resource_manager is not None
                                        else self.parallel_trials
                                    ),
                                ),
                            )
                    last_progress = time.monotonic()
        return [results[index] for index in range(len(subsets))]

    def _evaluate_many_ray(
        self,
        subsets: Sequence[frozenset[str]],
        fidelity: str,
    ) -> list[tuple[Evaluation | None, TrialDiagnostic | None]]:
        if not isinstance(self.evaluator, CatBoostSubsetEvaluator):
            raise TypeError(
                "The Ray backend currently supports CatBoostSubsetEvaluator only"
            )
        ray = _require("ray", 'ray[default]>=2.40,<3')
        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {
                "namespace": self.execution.ray_namespace,
                "log_to_driver": False,
            }
            if self.execution.ray_address:
                init_kwargs["address"] = self.execution.ray_address
            if self.execution.ray_runtime_env:
                init_kwargs["runtime_env"] = self.execution.ray_runtime_env
            ray.init(**init_kwargs)
        remote_options: dict[str, Any] = {
            "num_cpus": self.execution.ray_num_cpus_per_trial,
            "num_gpus": self.execution.ray_num_gpus_per_trial,
            "max_retries": self.execution.trial_max_retries,
            "retry_exceptions": False,
        }
        if self.resource_manager is not None:
            remote_options["memory"] = (
                self.resource_manager.ray_memory_bytes_per_trial
            )
        remote_trial = ray.remote(_distributed_catboost_trial).options(
            **remote_options
        )
        fidelity_payloads = []
        for item in self.evaluator.fidelities.values():
            row = asdict(item)
            row["train_sample"] = str(Path(row["train_sample"]).resolve())
            row["valid_sample"] = str(Path(row["valid_sample"]).resolve())
            fidelity_payloads.append(row)
        common = {
            "config": asdict(self.evaluator.cfg),
            "fidelities": fidelity_payloads,
            "cache_directory": str(self.evaluator.cache.directory.resolve()),
            "fingerprint": self.evaluator.cache.fingerprint,
            "fidelity": fidelity,
            "trial_max_retries": self.execution.trial_max_retries,
            "retry_backoff_seconds": self.execution.retry_backoff_seconds,
        }
        outcomes: list[
            tuple[Evaluation | None, TrialDiagnostic | None]
        ] = []
        batch_limit = self.parallel_trials
        if hasattr(ray, "available_resources"):
            try:
                available = ray.available_resources()
                slot_limits = [
                    math.floor(
                        float(available.get("CPU", 0.0))
                        / self.execution.ray_num_cpus_per_trial
                    )
                ]
                if self.execution.ray_num_gpus_per_trial > 0:
                    slot_limits.append(
                        math.floor(
                            float(available.get("GPU", 0.0))
                            / self.execution.ray_num_gpus_per_trial
                        )
                    )
                if self.resource_manager is not None:
                    available_memory = float(
                        available.get("memory", 0.0)
                    )
                    if available_memory > 0:
                        slot_limits.append(
                            math.floor(
                                available_memory
                                / self.resource_manager.ray_memory_bytes_per_trial
                            )
                        )
                positive_limits = [
                    limit for limit in slot_limits if limit > 0
                ]
                if positive_limits:
                    batch_limit = min(batch_limit, *positive_limits)
            except (AttributeError, TypeError, ValueError):
                pass
        batch_limit = max(1, batch_limit)
        for batch in _chunks(list(subsets), batch_limit):
            references = [
                remote_trial.remote({**common, "subset": sorted(subset)})
                for subset in batch
            ]
            for subset, reference in zip(batch, references):
                try:
                    payload = ray.get(reference)
                    evaluation = (
                        _evaluation_from_payload(payload["evaluation"])
                        if payload["evaluation"] is not None
                        else None
                    )
                    diagnostic = (
                        _trial_diagnostic_from_payload(payload["diagnostic"])
                        if payload["diagnostic"] is not None
                        else None
                    )
                except Exception as exc:
                    evaluation = None
                    oom = _is_oom_error(exc)
                    diagnostic = TrialDiagnostic(
                        subset=tuple(sorted(subset)),
                        fidelity=fidelity,
                        backend="ray",
                        outcome="failed",
                        attempts=self.execution.trial_max_retries + 1,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:2_000],
                        oom_detected=oom,
                        concurrency_limit=(
                            self.resource_manager.current_parallel_limit
                            if self.resource_manager is not None
                            else self.parallel_trials
                        ),
                    )
                if (
                    diagnostic is not None
                    and diagnostic.oom_detected
                    and self.resource_manager is not None
                ):
                    self.resource_manager.report_oom()
                outcomes.append((evaluation, diagnostic))
        return outcomes

    def _promotion_count(
        self, screened: Sequence[Evaluation]
    ) -> tuple[int, int, int]:
        total = len(screened)
        pareto = pareto_front(screened)
        feasible = [item for item in screened if item.constraint_violation == 0]
        pool = feasible or list(screened)
        best_metric = max(item.valid_metric for item in pool)
        near_best = [
            item
            for item in pool
            if item.valid_metric
            >= best_metric - self.config.promotion_metric_band
        ]
        if self.config.promotion_strategy == "fixed":
            promoted = max(
                self.config.min_promoted,
                math.ceil(total * self.config.promotion_fraction),
            )
        else:
            lower = max(
                self.config.min_promoted,
                math.ceil(total * self.config.promotion_min_fraction),
            )
            upper = max(
                lower,
                math.ceil(total * self.config.promotion_max_fraction),
            )
            ambiguous = {
                item.subset for item in [*pareto, *near_best]
            }
            promoted = min(upper, max(lower, len(ambiguous)))
        return min(total, promoted), len(pareto), len(near_best)

    def _evaluate_many(
        self, subsets: Iterable[Iterable[str]], fidelity: str
    ) -> list[Evaluation]:
        unique = self._unique_subsets(subsets)
        if not unique:
            return []
        if self.config.successive_halving_enabled and fidelity == "search":
            screened = self._evaluate_many_raw(unique, "screen")
            self.screening_history.extend(screened)
            promoted_count, pareto_count, near_best_count = (
                self._promotion_count(screened)
            )
            self.promotion_batches.append(
                PromotionBatch(
                    batch=len(self.promotion_batches) + 1,
                    strategy=self.config.promotion_strategy,
                    screened=len(screened),
                    promoted=promoted_count,
                    actual_fraction=promoted_count / len(screened),
                    pareto_candidates=pareto_count,
                    near_best_candidates=near_best_count,
                    metric_band=self.config.promotion_metric_band,
                )
            )
            promoted = nsga_select(screened, promoted_count)
            unique = [frozenset(item.subset) for item in promoted]
        result = self._evaluate_many_raw(unique, fidelity)
        self.history.extend(result)
        return result

    def _checkpoint_payload(
        self,
        status: str,
        active: dict[str, Any] | None = None,
        final_front: Sequence[Evaluation] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "selector_version": APP_VERSION,
            "fingerprint": self.checkpoint_fingerprint,
            "status": status,
            "features": list(self.features),
            "history": [asdict(item) for item in self.history],
            "screening_history": [
                asdict(item) for item in self.screening_history
            ],
            "trial_diagnostics": [
                asdict(item) for item in self.trial_diagnostics
            ],
            "promotion_batches": [
                asdict(item) for item in self.promotion_batches
            ],
            "coalitions": [asdict(item) for item in self.coalitions],
            "restart_winners": [list(item) for item in self.restart_winners],
            "completed_restart_fronts": [
                [asdict(item) for item in front]
                for front in self._completed_restart_fronts
            ],
            "active": active,
            "final_front": (
                [asdict(item) for item in final_front]
                if final_front is not None
                else None
            ),
        }

    def _save_checkpoint(
        self,
        status: str,
        active: dict[str, Any] | None = None,
        final_front: Sequence[Evaluation] | None = None,
    ) -> None:
        if not self.execution.checkpoint_enabled or self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                self._checkpoint_payload(status, active, final_front),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.checkpoint_path)

    def _load_checkpoint(self) -> None:
        if (
            not self.execution.checkpoint_enabled
            or not self.execution.resume_search
            or self.checkpoint_path is None
            or not self.checkpoint_path.exists()
        ):
            return
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.checkpoint_fingerprint:
            _event("search_checkpoint_ignored", reason="fingerprint_mismatch")
            return
        if tuple(payload.get("features", ())) != self.features:
            _event("search_checkpoint_ignored", reason="feature_mismatch")
            return
        self.history = [
            _evaluation_from_payload(item) for item in payload.get("history", [])
        ]
        self.screening_history = [
            _evaluation_from_payload(item)
            for item in payload.get("screening_history", [])
        ]
        self.trial_diagnostics = [
            _trial_diagnostic_from_payload(item)
            for item in payload.get("trial_diagnostics", [])
        ]
        self.promotion_batches = [
            _promotion_batch_from_payload(item)
            for item in payload.get("promotion_batches", [])
        ]
        self.coalitions = [
            _coalition_from_payload(item)
            for item in payload.get("coalitions", [])
        ] or list(self.interaction_priors)
        self._index_coalitions()
        self.restart_winners = [
            tuple(item) for item in payload.get("restart_winners", [])
        ]
        self._completed_restart_fronts = [
            [_evaluation_from_payload(item) for item in front]
            for front in payload.get("completed_restart_fronts", [])
        ]
        self._active_checkpoint = payload.get("active")
        if payload.get("status") == "complete" and payload.get("final_front"):
            self._checkpoint_complete_front = [
                _evaluation_from_payload(item)
                for item in payload["final_front"]
            ]
        _event(
            "search_checkpoint_loaded",
            status=payload.get("status"),
            completed_restarts=len(self._completed_restart_fronts),
            active_stage=(self._active_checkpoint or {}).get("stage"),
        )

    def explore_subspaces(self, fidelity: str, seed: int) -> list[Evaluation]:
        rng = random.Random(seed)
        self._progress(
            "subspace_exploration_started",
            restart=self._progress_restart,
            random=self.config.random_subspaces,
            adaptive=self.config.adaptive_subspaces,
        )
        initial = [self._random_subset(rng) for _ in range(self.config.random_subspaces)]
        evaluated = self._evaluate_many(initial, fidelity)
        self._refresh_coalitions(evaluated, seed)
        elite_count = max(1, math.ceil(len(evaluated) * self.config.elite_fraction))
        elite = sorted(
            evaluated,
            key=lambda item: (item.constraint_violation, -item.valid_metric, item.n_features),
        )[:elite_count]
        counts = Counter(feature for item in elite for feature in item.subset)
        floor = 1.0 / max(1, len(self.features))
        weights = {
            feature: floor + counts.get(feature, 0) / max(1, elite_count)
            for feature in self.features
        }
        adaptive = [
            self._random_subset(rng, weights=weights)
            for _ in range(self.config.adaptive_subspaces)
        ]
        combined = [*evaluated, *self._evaluate_many(adaptive, fidelity)]
        self._refresh_coalitions(combined, seed + 1)
        self._progress(
            "subspace_exploration_completed",
            restart=self._progress_restart,
            evaluated=len(combined),
            coalitions=len(self.coalitions),
        )
        return combined

    def _seed_population(
        self, explored: Sequence[Evaluation], rng: random.Random
    ) -> list[frozenset[str]]:
        candidates = sorted(
            explored,
            key=lambda item: (item.constraint_violation, -item.valid_metric, item.n_features),
        )
        population: list[frozenset[str]] = []
        for item in candidates[: self.config.population_size // 2]:
            population.append(
                _clamp_subset(
                    item.subset,
                    self.features,
                    self.config.min_features,
                    (self.config.search_max_features or self.config.max_features),
                    rng,
                    self.required_features,
                )
            )
        for coalition in self.coalitions[: self.config.population_size // 4]:
            base = set(
                self._random_subset(
                    rng,
                    minimum=self.config.min_features,
                    maximum=(self.config.search_max_features or self.config.max_features),
                )
            )
            base.update(coalition.members)
            population.append(
                _clamp_subset(
                    base,
                    self.features,
                    self.config.min_features,
                    (self.config.search_max_features or self.config.max_features),
                    rng,
                    self.required_features,
                )
            )
        while len(population) < self.config.population_size:
            population.append(
                self._random_subset(
                    rng,
                    minimum=self.config.min_features,
                    maximum=(self.config.search_max_features or self.config.max_features),
                )
            )
        return population[: self.config.population_size]

    def _mutate(self, subset: frozenset[str], rng: random.Random) -> frozenset[str]:
        selected = set(subset)
        available = sorted(self.feature_set - selected)
        operation = rng.choice(("add", "drop", "swap", "pair", "coalition"))
        if (
            operation == "coalition"
            and self.coalitions
            and rng.random() < self.config.coalition_mutation_probability
        ):
            coalition = self._choose_coalition(
                rng, selected, (self.config.search_max_features or self.config.max_features)
            )
            if coalition:
                selected.update(coalition.members)
        elif (
            operation == "pair"
            and self.interaction_pairs
            and rng.random() < self.config.interaction_mutation_probability
        ):
            first, second, _ = rng.choice(self.interaction_pairs)
            selected.update((first, second))
        elif operation == "add" and available:
            selected.add(rng.choice(available))
        elif operation == "drop" and len(selected) > self.config.min_features:
            droppable = sorted(selected - self.required_features)
            if droppable:
                selected.remove(rng.choice(droppable))
        elif operation == "swap" and available and selected:
            droppable = sorted(selected - self.required_features)
            if droppable:
                selected.remove(rng.choice(droppable))
                selected.add(rng.choice(available))
        return _clamp_subset(
            selected,
            self.features,
            self.config.min_features,
            (self.config.search_max_features or self.config.max_features),
            rng,
            self.required_features,
        )

    def _crossover(
        self, left: frozenset[str], right: frozenset[str], rng: random.Random
    ) -> frozenset[str]:
        union = sorted(left | right)
        child = {feature for feature in union if rng.random() < 0.5}
        child.update(left & right)
        for coalition in self.coalitions:
            members = set(coalition.members)
            inherited = members <= left or members <= right
            if (
                inherited
                and members & child
                and len(child | members) <= (self.config.search_max_features or self.config.max_features)
                and rng.random() < self.config.coalition_preservation_probability
            ):
                child.update(members)
        return _clamp_subset(
            child,
            self.features,
            self.config.min_features,
            (self.config.search_max_features or self.config.max_features),
            rng,
            self.required_features,
        )

    def evolve(
        self,
        explored: Sequence[Evaluation],
        fidelity: str,
        seed: int,
        initial_parents: Sequence[Evaluation] | None = None,
        start_generation: int = 0,
        rng_state: Any | None = None,
        checkpoint_callback: Any | None = None,
    ) -> list[Evaluation]:
        rng = random.Random(seed)
        if rng_state is not None:
            rng.setstate(_nested_tuple(rng_state))
        if initial_parents is None:
            population = self._seed_population(explored, rng)
            parents = self._evaluate_many(population, fidelity)
            self._refresh_coalitions(self.history, seed)
        else:
            parents = list(initial_parents)
        for generation in range(start_generation, self.config.generations):
            selected = nsga_select(parents, self.config.population_size)
            offspring: list[frozenset[str]] = []
            while len(offspring) < self.config.population_size:
                if len(selected) < 2:
                    selected = [*selected, *selected]
                first, second = rng.sample(selected, 2)
                if rng.random() < self.config.crossover_rate:
                    child = self._crossover(
                        frozenset(first.subset), frozenset(second.subset), rng
                    )
                else:
                    child = frozenset(first.subset)
                if rng.random() < self.config.mutation_rate:
                    child = self._mutate(child, rng)
                offspring.append(child)
            evaluated_offspring = self._evaluate_many(offspring, fidelity)
            parents = nsga_select(
                [*selected, *evaluated_offspring], self.config.population_size
            )
            if (generation + 1) % self.config.coalition_refresh_interval == 0:
                self._refresh_coalitions(self.history, seed + generation + 1)
            if checkpoint_callback is not None:
                checkpoint_callback(generation + 1, parents, rng.getstate())
            front = pareto_front(parents)
            self._progress(
                "search_generation_completed",
                restart=self._progress_restart,
                generation=generation + 1,
                generations=self.config.generations,
                best_metric=max(item.valid_metric for item in parents),
                pareto=len(front),
                full_trials=len(self.history),
                screen_trials=len(self.screening_history),
            )
        return parents

    def local_refine(
        self, candidates: Sequence[Evaluation], fidelity: str, seed: int
    ) -> list[Evaluation]:
        rng = random.Random(seed)
        self._progress(
            "local_refinement_started",
            restart=self._progress_restart,
            candidates=min(len(candidates), self.config.local_candidates),
        )
        refined: list[Evaluation] = list(candidates)
        for initial in candidates[: self.config.local_candidates]:
            current = initial
            for _ in range(self.config.local_rounds):
                subset = set(current.subset)
                proposals: list[frozenset[str]] = []
                for feature in rng.sample(
                    sorted(subset), min(12, len(subset))
                ):
                    if (
                        len(subset) > self.config.min_features
                        and feature not in self.required_features
                    ):
                        proposals.append(frozenset(subset - {feature}))
                available = sorted(self.feature_set - subset)
                for feature in rng.sample(available, min(12, len(available))):
                    proposals.append(
                        _clamp_subset(
                            subset | {feature},
                            self.features,
                            self.config.min_features,
                            (self.config.search_max_features or self.config.max_features),
                            rng,
                            self.required_features,
                        )
                    )
                for coalition in self.coalitions[:12]:
                    proposals.append(
                        _clamp_subset(
                            subset | set(coalition.members),
                            self.features,
                            self.config.min_features,
                            (self.config.search_max_features or self.config.max_features),
                            rng,
                            self.required_features,
                        )
                    )
                evaluated = self._evaluate_many(proposals, fidelity)
                choices = [current, *evaluated]
                best_metric = max(item.valid_metric for item in choices)
                eligible = [
                    item
                    for item in choices
                    if item.constraint_violation == 0
                    and item.valid_metric >= best_metric - self.config.allowed_metric_drop
                ]
                next_item = min(
                    eligible or choices,
                    key=lambda item: (
                        item.constraint_violation,
                        item.n_features,
                        -item.valid_metric,
                        item.metric_std,
                    ),
                )
                refined.append(next_item)
                if next_item.subset == current.subset:
                    break
                current = next_item
        self._progress(
            "local_refinement_completed",
            restart=self._progress_restart,
            evaluated=len(refined),
        )
        return refined

    def run(self, fidelity: str = "search") -> list[Evaluation]:
        self._load_checkpoint()
        if self._checkpoint_complete_front is not None:
            self._progress(
                "search_resume_completed",
                pareto=len(self._checkpoint_complete_front),
            )
            return list(self._checkpoint_complete_front)

        finalists = [
            item
            for front in self._completed_restart_fronts
            for item in front
        ]
        start_restart = len(self._completed_restart_fronts)
        active = self._active_checkpoint
        if active is not None and int(active["restart"]) != start_restart:
            raise RuntimeError("Search checkpoint has an inconsistent active restart")

        for restart in range(start_restart, self.config.restarts):
            self._progress_restart = restart + 1
            self._progress(
                "search_restart_started",
                restart=restart + 1,
                restarts=self.config.restarts,
            )
            seed = self.config.seed + restart * 10_000
            if active is not None and int(active["restart"]) == restart:
                explored = [
                    _evaluation_from_payload(item)
                    for item in active.get("explored", [])
                ]
                if not explored:
                    raise RuntimeError("Search checkpoint is missing explored candidates")
            else:
                explored = self.explore_subspaces(fidelity, seed)
                active = {
                    "restart": restart,
                    "stage": "explored",
                    "explored": [asdict(item) for item in explored],
                }
                self._save_checkpoint("running", active=active)

            if active is not None and active.get("stage") == "refine":
                evolved = [
                    _evaluation_from_payload(item)
                    for item in active.get("evolved", [])
                ]
            else:
                initial_parents = None
                start_generation = 0
                rng_state = None
                if active is not None and active.get("stage") == "evolve":
                    initial_parents = [
                        _evaluation_from_payload(item)
                        for item in active.get("parents", [])
                    ]
                    start_generation = int(active.get("next_generation", 0))
                    rng_state = active.get("rng_state")

                def checkpoint_generation(
                    next_generation: int,
                    parents: Sequence[Evaluation],
                    current_rng_state: Any,
                ) -> None:
                    generation_state = {
                        "restart": restart,
                        "stage": "evolve",
                        "explored": [asdict(item) for item in explored],
                        "parents": [asdict(item) for item in parents],
                        "next_generation": next_generation,
                        "rng_state": current_rng_state,
                    }
                    self._save_checkpoint("running", active=generation_state)

                evolved = self.evolve(
                    explored,
                    fidelity,
                    seed + 1,
                    initial_parents=initial_parents,
                    start_generation=start_generation,
                    rng_state=rng_state,
                    checkpoint_callback=checkpoint_generation,
                )
                active = {
                    "restart": restart,
                    "stage": "refine",
                    "explored": [asdict(item) for item in explored],
                    "evolved": [asdict(item) for item in evolved],
                }
                self._save_checkpoint("running", active=active)

            refined = self.local_refine(
                pareto_front([*explored, *evolved]), fidelity, seed + 2
            )
            restart_front = pareto_front(refined)
            restart_winner = choose_compact(
                restart_front, self.config.allowed_metric_drop
            )
            self.restart_winners.append(restart_winner.subset)
            finalists.extend(restart_front)
            self._completed_restart_fronts.append(restart_front)
            self._progress(
                "search_restart_completed",
                restart=restart + 1,
                pareto=len(restart_front),
                winner_features=restart_winner.n_features,
                winner_metric=restart_winner.valid_metric,
            )
            active = None
            self._save_checkpoint("running", active=None)
        self._refresh_coalitions(self.history, self.config.seed + 999_999)
        final_front = pareto_front(finalists)
        self._save_checkpoint("complete", active=None, final_front=final_front)
        return final_front


def synergy_score(
    evaluator: SubsetEvaluator,
    selected: Sequence[str],
    first: str,
    second: str,
    fidelity: str,
) -> dict[str, Any]:
    background = set(selected) - {first, second}
    if not background:
        raise ValueError("Synergy evaluation requires at least one background feature")
    q0 = evaluator.evaluate(background, fidelity).valid_metric
    qa = evaluator.evaluate(background | {first}, fidelity).valid_metric
    qb = evaluator.evaluate(background | {second}, fidelity).valid_metric
    qab = evaluator.evaluate(background | {first, second}, fidelity).valid_metric
    return {
        "first": first,
        "second": second,
        "background_size": len(background),
        "score_background": q0,
        "score_first": qa,
        "score_second": qb,
        "score_pair": qab,
        "synergy": qab - qa - qb + q0,
    }


def pairwise_jaccard(subsets: Sequence[Sequence[str]]) -> float:
    sets = [set(subset) for subset in subsets]
    if len(sets) < 2:
        return 1.0
    scores: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            union = sets[left_index] | sets[right_index]
            scores.append(len(sets[left_index] & sets[right_index]) / len(union) if union else 1.0)
    return sum(scores) / len(scores)


def choose_compact(front: Sequence[Evaluation], allowed_drop: float) -> Evaluation:
    feasible = [item for item in front if item.constraint_violation == 0]
    pool = feasible or list(front)
    best = max(item.valid_metric for item in pool)
    eligible = [item for item in pool if item.valid_metric >= best - allowed_drop]
    return min(eligible, key=lambda item: (item.n_features, item.metric_std, -item.valid_metric))


def choose_finalists(
    front: Sequence[Evaluation], limit: int, allowed_drop: float
) -> list[Evaluation]:
    if len(front) <= limit:
        return list(front)
    anchors = [
        max(front, key=lambda item: item.valid_metric),
        min(front, key=lambda item: item.n_features),
        choose_compact(front, allowed_drop),
    ]
    by_size = sorted(front, key=lambda item: (item.n_features, -item.valid_metric))
    if limit > len(anchors):
        step = (len(by_size) - 1) / max(1, limit - len(anchors) - 1)
        anchors.extend(by_size[round(index * step)] for index in range(limit - len(anchors)))
    unique = {item.subset: item for item in anchors}
    if len(unique) < limit:
        for item in front:
            unique.setdefault(item.subset, item)
            if len(unique) == limit:
                break
    return list(unique.values())[:limit]


def _bootstrap_base_indices(
    y_true: Any,
    weight: Any,
    max_rows: int,
    seed: int,
) -> tuple[Any, Any]:
    np = _require("numpy", "numpy>=1.26,<3")
    y = np.asarray(y_true, dtype="int8")
    weights = np.asarray(weight, dtype="float64")
    if len(y) != len(weights):
        raise ValueError("Target and weight lengths differ")
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    if any(len(indices) < 2 for indices in classes):
        raise ValueError("Robust validation requires at least two rows in each class")
    if len(y) <= max_rows:
        return np.arange(len(y)), weights.copy()

    rng = np.random.default_rng(seed)
    keep = [min(len(indices), max_rows // 2) for indices in classes]
    remaining = max_rows - sum(keep)
    while remaining:
        changed = False
        for class_index, indices in enumerate(classes):
            capacity = len(indices) - keep[class_index]
            if capacity <= 0:
                continue
            addition = min(capacity, remaining)
            keep[class_index] += addition
            remaining -= addition
            changed = True
            if remaining == 0:
                break
        if not changed:
            break

    sampled: list[Any] = []
    adjusted_weights: list[Any] = []
    for indices, size in zip(classes, keep):
        chosen = rng.choice(indices, size=size, replace=False)
        inclusion_probability = size / len(indices)
        sampled.append(chosen)
        adjusted_weights.append(weights[chosen] / inclusion_probability)
    return np.concatenate(sampled), np.concatenate(adjusted_weights)


def _threshold_metrics(
    y_true: Any,
    prediction: Any,
    weight: Any,
    threshold: float,
) -> tuple[float, float, float, float]:
    np = _require("numpy", "numpy>=1.26,<3")
    y = np.asarray(y_true, dtype="int8")
    scores = np.asarray(prediction, dtype="float64")
    weights = np.asarray(weight, dtype="float64")
    predicted = scores >= threshold
    true_positive = float(weights[(y == 1) & predicted].sum())
    false_positive = float(weights[(y == 0) & predicted].sum())
    false_negative = float(weights[(y == 1) & ~predicted].sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_positive_rate = float(weights[predicted].sum() / weights.sum())
    return precision, recall, f1, predicted_positive_rate


def select_decision_threshold(
    subset: Sequence[str],
    tuning: SplitPredictions,
    evaluation: SplitPredictions | None,
    config: DecisionThresholdConfig,
) -> ThresholdResult:
    np = _require("numpy", "numpy>=1.26,<3")
    metrics = _require("sklearn.metrics", "scikit-learn>=1.5")
    precision, recall, thresholds = metrics.precision_recall_curve(
        tuning.y_true,
        tuning.prediction,
        sample_weight=tuning.weight,
    )
    precision = np.asarray(precision[:-1], dtype="float64")
    recall = np.asarray(recall[:-1], dtype="float64")
    thresholds = np.asarray(thresholds, dtype="float64")
    if not len(thresholds):
        raise ValueError("Cannot select a threshold from constant/empty predictions")
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    feasible = (precision >= config.min_precision) & (recall >= config.min_recall)
    candidate_indices = np.flatnonzero(feasible)
    tuning_feasible = bool(len(candidate_indices))
    if not tuning_feasible:
        precision_shortfall = np.maximum(0.0, config.min_precision - precision)
        recall_shortfall = np.maximum(0.0, config.min_recall - recall)
        violation = precision_shortfall + recall_shortfall
        candidate_indices = np.flatnonzero(violation == violation.min())

    if config.objective == "max_precision":
        chosen = max(
            candidate_indices,
            key=lambda index: (
                precision[index],
                recall[index],
                f1[index],
                thresholds[index],
            ),
        )
    else:
        chosen = max(
            candidate_indices,
            key=lambda index: (
                f1[index],
                precision[index],
                recall[index],
                thresholds[index],
            ),
        )
    threshold = float(thresholds[chosen])
    if evaluation is None:
        evaluation_precision = None
        evaluation_recall = None
        evaluation_f1 = None
        positive_rate = None
        evaluation_feasible = None
    else:
        evaluation_precision, evaluation_recall, evaluation_f1, positive_rate = (
            _threshold_metrics(
                evaluation.y_true,
                evaluation.prediction,
                evaluation.weight,
                threshold,
            )
        )
        evaluation_feasible = (
            evaluation_precision >= config.min_precision
            and evaluation_recall >= config.min_recall
        )
    return ThresholdResult(
        subset=tuple(sorted(set(subset))),
        threshold=threshold,
        objective=config.objective,
        tuning_split=tuning.split,
        tuning_precision=float(precision[chosen]),
        tuning_recall=float(recall[chosen]),
        tuning_f1=float(f1[chosen]),
        tuning_feasible=tuning_feasible,
        evaluation_split=evaluation.split if evaluation is not None else None,
        evaluation_precision=evaluation_precision,
        evaluation_recall=evaluation_recall,
        evaluation_f1=evaluation_f1,
        predicted_positive_rate=positive_rate,
        evaluation_feasible=evaluation_feasible,
    )


def _cluster_bootstrap_base_indices(
    y_true: Any,
    weight: Any,
    groups: Any,
    max_rows: int,
    seed: int,
) -> tuple[Any, Any, Any, int, int]:
    np = _require("numpy", "numpy>=1.26,<3")
    y = np.asarray(y_true, dtype="int8")
    weights = np.asarray(weight, dtype="float64")
    group_values = np.asarray(groups)
    if len(y) != len(weights) or len(y) != len(group_values):
        raise ValueError("Target, weight, and cluster-key lengths differ")
    unique_groups, inverse = np.unique(group_values, return_inverse=True)
    group_count = len(unique_groups)
    if group_count < 4:
        raise ValueError("Cluster bootstrap requires at least four clusters")
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=group_count)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    positive_counts = np.bincount(inverse, weights=y, minlength=group_count)
    strata = [
        np.flatnonzero(positive_counts == 0),
        np.flatnonzero(positive_counts > 0),
    ]
    if any(len(stratum) < 2 for stratum in strata):
        raise ValueError(
            "Cluster bootstrap requires at least two positive-bearing and two negative clusters"
        )
    if len(y) <= max_rows:
        return (
            np.arange(len(y)),
            weights.copy(),
            group_values.copy(),
            group_count,
            group_count,
        )

    rng = np.random.default_rng(seed)
    selected_by_stratum: list[list[int]] = []
    target_rows = max_rows // 2
    for stratum in strata:
        shuffled = rng.permutation(stratum)
        selected: list[int] = []
        selected_rows = 0
        for group_index in shuffled:
            group_size = int(counts[group_index])
            if selected and selected_rows + group_size > target_rows:
                continue
            selected.append(int(group_index))
            selected_rows += group_size
            if selected_rows >= target_rows:
                break
        if len(selected) < 2:
            selected = [int(value) for value in shuffled[:2]]
        selected_by_stratum.append(selected)

    selected_rows: list[Any] = []
    adjusted_weights: list[Any] = []
    selected_group_values: list[Any] = []
    for stratum, selected in zip(strata, selected_by_stratum):
        inclusion_probability = len(selected) / len(stratum)
        for group_index in selected:
            start = int(starts[group_index])
            stop = start + int(counts[group_index])
            rows = order[start:stop]
            selected_rows.append(rows)
            adjusted_weights.append(weights[rows] / inclusion_probability)
            selected_group_values.append(group_values[rows])
    indices = np.concatenate(selected_rows)
    return (
        indices,
        np.concatenate(adjusted_weights),
        np.concatenate(selected_group_values),
        group_count,
        sum(len(selected) for selected in selected_by_stratum),
    )


def robust_paired_bootstrap(
    predictions: dict[tuple[str, ...], SplitPredictions],
    metric_name: str,
    confidence_level: float,
    repeats: int,
    max_rows: int,
    seed: int,
    noninferiority_margin: float,
    eligible_subsets: set[tuple[str, ...]] | None = None,
    bootstrap_mode: str = "auto",
) -> tuple[list[RobustCandidate], tuple[str, ...]]:
    if not predictions:
        raise ValueError("No predictions supplied for robust validation")
    np = _require("numpy", "numpy>=1.26,<3")
    subsets = sorted(predictions)
    if eligible_subsets is None:
        eligible_subsets = set(subsets)
    if not eligible_subsets <= set(subsets):
        raise ValueError("eligible_subsets contains a subset without predictions")
    reference_data = predictions[subsets[0]]
    y_full = np.asarray(reference_data.y_true, dtype="int8")
    weight_full = np.asarray(reference_data.weight, dtype="float64")
    split = reference_data.split
    groups_full = (
        np.asarray(reference_data.groups)
        if reference_data.groups is not None
        else None
    )
    for subset in subsets[1:]:
        item = predictions[subset]
        if item.split != split:
            raise ValueError("All candidates must belong to the same split")
        if not np.array_equal(y_full, np.asarray(item.y_true, dtype="int8")):
            raise ValueError("Candidate holdout targets are not row-aligned")
        if not np.allclose(weight_full, np.asarray(item.weight, dtype="float64")):
            raise ValueError("Candidate holdout weights are not row-aligned")
        if (groups_full is None) != (item.groups is None):
            raise ValueError("Candidate cluster keys are inconsistent")
        if groups_full is not None and not np.array_equal(
            groups_full, np.asarray(item.groups)
        ):
            raise ValueError("Candidate cluster keys are not row-aligned")

    if bootstrap_mode not in {"auto", "row", "cluster"}:
        raise ValueError("bootstrap_mode must be auto, row, or cluster")
    effective_mode = (
        "cluster"
        if bootstrap_mode == "cluster"
        or (bootstrap_mode == "auto" and groups_full is not None)
        else "row"
    )
    if effective_mode == "cluster" and groups_full is None:
        raise ValueError("Cluster bootstrap requested without cluster keys")

    point_scores = np.asarray(
        [
            _metric_value(
                metric_name,
                y_full,
                np.asarray(predictions[subset].prediction),
                weight_full,
            )
            for subset in subsets
        ],
        dtype="float64",
    )
    if effective_mode == "cluster":
        (
            base_indices,
            base_weights,
            base_groups,
            n_clusters,
            bootstrap_clusters,
        ) = _cluster_bootstrap_base_indices(
            y_full, weight_full, groups_full, max_rows, seed
        )
    else:
        base_indices, base_weights = _bootstrap_base_indices(
            y_full, weight_full, max_rows, seed
        )
        base_groups = None
        n_clusters = None
        bootstrap_clusters = None
    y_base = y_full[base_indices]
    prediction_matrix = np.vstack(
        [
            np.asarray(predictions[subset].prediction, dtype="float64")[base_indices]
            for subset in subsets
        ]
    )
    if effective_mode == "cluster":
        _, base_group_inverse = np.unique(base_groups, return_inverse=True)
        base_group_count = int(base_group_inverse.max()) + 1
        base_order = np.argsort(base_group_inverse, kind="stable")
        base_counts = np.bincount(base_group_inverse, minlength=base_group_count)
        base_starts = np.concatenate(([0], np.cumsum(base_counts)[:-1]))
        base_positive_counts = np.bincount(
            base_group_inverse, weights=y_base, minlength=base_group_count
        )
        bootstrap_strata = [
            np.flatnonzero(base_positive_counts == 0),
            np.flatnonzero(base_positive_counts > 0),
        ]
        rows_by_group = [
            base_order[
                int(base_starts[group_index]) : int(base_starts[group_index])
                + int(base_counts[group_index])
            ]
            for group_index in range(base_group_count)
        ]
    else:
        class_indices = [np.flatnonzero(y_base == value) for value in (0, 1)]
    rng = np.random.default_rng(seed + 1)
    bootstrap_scores = np.empty((repeats, len(subsets)), dtype="float64")
    for repeat in range(repeats):
        if effective_mode == "cluster":
            sampled_groups = [
                group_index
                for stratum in bootstrap_strata
                for group_index in rng.choice(
                    stratum, size=len(stratum), replace=True
                )
            ]
            bootstrap_indices = np.concatenate(
                [rows_by_group[int(group_index)] for group_index in sampled_groups]
            )
        else:
            bootstrap_indices = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices
                ]
            )
        y_bootstrap = y_base[bootstrap_indices]
        weight_bootstrap = base_weights[bootstrap_indices]
        for candidate_index in range(len(subsets)):
            bootstrap_scores[repeat, candidate_index] = _metric_value(
                metric_name,
                y_bootstrap,
                prediction_matrix[candidate_index, bootstrap_indices],
                weight_bootstrap,
            )

    alpha = (1.0 - confidence_level) / 2.0
    reference_index = int(np.argmax(point_scores))
    reference_subset = subsets[reference_index]
    result: list[RobustCandidate] = []
    for candidate_index, subset in enumerate(subsets):
        candidate_bootstrap = bootstrap_scores[:, candidate_index]
        centered = candidate_bootstrap - float(np.mean(candidate_bootstrap))
        ci_low = float(
            point_scores[candidate_index] + np.quantile(centered, alpha)
        )
        ci_high = float(
            point_scores[candidate_index] + np.quantile(centered, 1.0 - alpha)
        )
        delta = float(point_scores[candidate_index] - point_scores[reference_index])
        bootstrap_delta = (
            candidate_bootstrap - bootstrap_scores[:, reference_index]
        )
        centered_delta = bootstrap_delta - float(np.mean(bootstrap_delta))
        delta_ci_low = float(delta + np.quantile(centered_delta, alpha))
        delta_ci_high = float(delta + np.quantile(centered_delta, 1.0 - alpha))
        result.append(
            RobustCandidate(
                subset=subset,
                split=split,
                n_features=len(subset),
                point_metric=float(point_scores[candidate_index]),
                ci_low=ci_low,
                ci_high=ci_high,
                bootstrap_std=float(np.std(candidate_bootstrap, ddof=1)),
                seed_metric_std=float(
                    np.std(predictions[subset].seed_metrics, ddof=1)
                    if len(predictions[subset].seed_metrics) > 1
                    else 0.0
                ),
                n_rows=len(y_full),
                bootstrap_rows=len(base_indices),
                bootstrap_unit=effective_mode,
                n_clusters=n_clusters,
                bootstrap_clusters=bootstrap_clusters,
                reference_subset=reference_subset,
                delta_to_reference=delta,
                delta_ci_low=delta_ci_low,
                delta_ci_high=delta_ci_high,
                eligible_for_selection=subset in eligible_subsets,
                noninferior=delta_ci_low >= -noninferiority_margin,
            )
        )

    best_lcb = max(item.ci_low for item in result)
    selection_pool = [item for item in result if item.eligible_for_selection]
    if not selection_pool:
        raise ValueError("No candidate is eligible for robust selection")
    eligible = [
        item
        for item in selection_pool
        if item.noninferior
        and item.ci_low >= best_lcb - noninferiority_margin
    ]
    if eligible:
        winner = min(
            eligible,
            key=lambda item: (
                item.n_features,
                -item.ci_low,
                -item.point_metric,
            ),
        )
    else:
        winner = max(
            selection_pool,
            key=lambda item: (item.ci_low, item.point_metric, -item.n_features),
        )
    return sorted(result, key=lambda item: (-item.ci_low, item.n_features)), winner.subset


def write_results(
    output_directory: str | Path,
    baseline: Evaluation,
    interactions: Sequence[tuple[str, str, float]],
    coalitions: Sequence[CoalitionScore],
    history: Sequence[Evaluation],
    screening_history: Sequence[Evaluation],
    trial_diagnostics: Sequence[TrialDiagnostic],
    promotion_batches: Sequence[PromotionBatch],
    search_front: Sequence[Evaluation],
    confirmed: Sequence[Evaluation],
    winner: Evaluation,
    selection_basis: str,
    robust_winner: RobustCandidate | None,
    threshold_winner: ThresholdResult | None,
    restart_winners: Sequence[Sequence[str]],
) -> Evaluation:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "baseline.json", asdict(baseline))
    _write_json(
        output / "interactions.json",
        [
            {"first": first, "second": second, "strength": strength}
            for first, second, strength in interactions
        ],
    )
    _write_json(
        output / "coalitions.json", [asdict(item) for item in coalitions]
    )
    _write_jsonl(
        output / "trials.jsonl", (asdict(item) for item in history)
    )
    _write_jsonl(
        output / "screening_trials.jsonl",
        (asdict(item) for item in screening_history),
    )
    _write_jsonl(
        output / "trial_diagnostics.jsonl",
        (asdict(item) for item in trial_diagnostics),
    )
    _write_jsonl(
        output / "promotion_batches.jsonl",
        (asdict(item) for item in promotion_batches),
    )
    _write_json(
        output / "pareto_search.json", [asdict(item) for item in search_front]
    )
    _write_json(
        output / "pareto_confirmed.json", [asdict(item) for item in confirmed]
    )
    _write_json(
        output / "selected_features.json",
        {
            "features": list(winner.subset),
            "n_features": winner.n_features,
            "valid_metric": winner.valid_metric,
            "metric_std": winner.metric_std,
            "gap": winner.gap,
            "selection_basis": selection_basis,
            "robust_validation": asdict(robust_winner) if robust_winner else None,
            "decision_threshold": (
                asdict(threshold_winner) if threshold_winner else None
            ),
        },
    )
    _write_json(
        output / "selection_stability.json",
        {
            "restart_winners": [list(subset) for subset in restart_winners],
            "pairwise_jaccard": pairwise_jaccard(restart_winners),
        },
    )
    return winner


def run_preflight(config_path: str | Path) -> dict[str, Any]:
    cfg = AppConfig.from_yaml(config_path)
    cfg.validate()
    _set_event_output(cfg.execution.show_progress)
    _resolve_process_start_method(cfg)
    output = Path(cfg.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    manager = ResourceManager(cfg, output) if cfg.resources.enabled else None
    split_paths = configured_split_paths(cfg)
    inventories = {
        name: parquet_inventory(path) for name, path in split_paths.items()
    }
    feature_count = len(collect_feature_names(cfg))
    iterations = {
        "screen": int(cfg.model_params.get("screen_iterations", 120)),
        "search": int(cfg.model_params.get("search_iterations", 350)),
        "confirm": int(cfg.model_params.get("confirm_iterations", 1000)),
    }
    fractions = {
        "screen": (
            max(
                cfg.sample.screen_train_positive_fraction,
                cfg.sample.screen_train_negative_fraction,
            ),
            max(
                cfg.sample.screen_valid_positive_fraction,
                cfg.sample.screen_valid_negative_fraction,
            ),
        ),
        "search": (
            max(
                cfg.sample.train_positive_fraction,
                cfg.sample.train_negative_fraction,
            ),
            max(
                cfg.sample.valid_positive_fraction,
                cfg.sample.valid_negative_fraction,
            ),
        ),
        "confirm": (
            max(
                cfg.sample.confirm_train_positive_fraction,
                cfg.sample.confirm_train_negative_fraction,
            ),
            max(
                cfg.sample.confirm_valid_positive_fraction,
                cfg.sample.confirm_valid_negative_fraction,
            ),
        ),
    }
    workloads: dict[str, Any] = {}
    train_rows = inventories["train"]["rows"]
    valid_rows = inventories["valid"]["rows"]
    feature_budget = min(feature_count, (cfg.search.search_max_features or cfg.search.max_features))
    for fidelity, (train_fraction, valid_fraction) in fractions.items():
        estimated_train = math.ceil(train_rows * train_fraction)
        estimated_valid = math.ceil(valid_rows * valid_fraction)
        dense_gb = (
            (estimated_train + estimated_valid)
            * feature_budget
            * 8
            / float(1024**3)
        )
        workloads[fidelity] = {
            "maximum_train_rows": estimated_train,
            "maximum_valid_rows": estimated_valid,
            "feature_budget": feature_budget,
            "iterations": iterations[fidelity],
            "seeds": 3 if fidelity == "confirm" else 1,
            "dense_matrix_upper_bound_gb": dense_gb,
        }
    baseline_dense_gb = (
        (train_rows + valid_rows)
        * feature_count
        * 8
        / float(1024**3)
    )
    warnings = []
    if feature_count > (cfg.search.search_max_features or cfg.search.max_features):
        warnings.append(
            "The all-feature baseline exceeds search_max_features and may require "
            "substantially more memory than a search trial"
        )
    if (
        cfg.execution.backend == "local"
        and cfg.execution.local_trial_mode == "thread"
    ):
        warnings.append(
            "Thread mode cannot survive a process-level OOM; use process mode"
        )
    if (
        manager is not None
        and cfg.execution.local_trial_mode == "process"
        and not manager.process_memory_telemetry_available
    ):
        warnings.append(
            "Per-process RSS telemetry is unavailable; process isolation and "
            "global RAM throttling remain active, but the hard RSS watchdog "
            "cannot enforce its limit"
        )
    if (
        str(cfg.model_params.get("task_type", "CPU")).upper() == "GPU"
        and cfg.resources.gpu_total_memory_gb is not None
        and not cfg.resources.gpu_memory_by_device_gb
    ):
        warnings.append(
            "Aggregate GPU memory assumes equal cards; per-device values are safer"
        )
    mode = validation_mode(cfg)
    if mode in {"final_holdout_only", "development_only"}:
        warnings.append(
            "OOS is absent: finalists will be selected on confirm_valid without "
            "paired OOS bootstrap"
        )
    if mode in {"oos_only", "development_only"}:
        warnings.append(
            "No post-selection OOT/test holdout is configured"
        )
    if cfg.decision_threshold.enabled and not cfg.data.oos_path:
        warnings.append(
            "OOS is absent: the operating threshold will be selected and "
            "checked only on confirm_valid before frozen holdout evaluation"
        )
    payload = {
        "selector_version": APP_VERSION,
        "status": "plan_only",
        "training_started": False,
        "validation_mode": validation_mode(cfg),
        "active_splits": list(split_paths),
        "input_features": feature_count,
        "search_feature_budget": feature_budget,
        "inventories": inventories,
        "workloads": workloads,
        "baseline": {
            "features": feature_count,
            "dense_matrix_upper_bound_gb": baseline_dense_gb,
        },
        "execution": {
            "backend": cfg.execution.backend,
            "local_trial_mode": cfg.execution.local_trial_mode,
            "requested_parallel_trials": cfg.execution.parallel_trials,
            "effective_parallel_limit": (
                manager.max_parallel_trials
                if manager is not None
                else max(1, cfg.execution.parallel_trials)
            ),
            "threads_per_trial": cfg.execution.threads_per_trial,
            "trial_timeout_seconds": cfg.execution.trial_timeout_seconds,
        },
        "resources": (
            {
                "detected": asdict(manager.initial_snapshot),
                "ram_per_trial_gb": manager.ram_per_trial_gb,
                "gpu_memory_per_trial_gb": (
                    manager.gpu_memory_per_trial_gb
                ),
                "hard_ram_per_trial_gb": manager.hard_ram_limit_gb("search"),
                "hard_gpu_memory_per_trial_gb": (
                    manager.hard_gpu_limit_gb("search")
                ),
                "process_memory_telemetry_available": (
                    manager.process_memory_telemetry_available
                ),
            }
            if manager is not None
            else None
        ),
        "warnings": warnings,
    }
    _write_json(output / "preflight_plan.json", payload)
    return payload


def run_pipeline(config_path: str | Path) -> None:
    cfg = AppConfig.from_yaml(config_path)
    cfg.validate()
    _set_event_output(cfg.execution.show_progress)
    _resolve_process_start_method(cfg)
    output = Path(cfg.output.directory)
    output.mkdir(parents=True, exist_ok=True)

    _event("input_validation_started")
    features = collect_feature_names(cfg)
    validate_inputs(cfg, features)
    if len(features) < cfg.search.min_features:
        raise RuntimeError("Input contains fewer features than search.min_features")
    if len(cfg.data.required_features) > cfg.search.max_features:
        raise ValueError("The required feature count exceeds search.max_features")
    _event(
        "input_validation_completed",
        features=len(features),
        validation_mode=validation_mode(cfg),
    )

    _event("sampling_started")
    samples = prepare_samples(cfg, features)
    _event(
        "sampling_completed",
        splits=",".join(sorted(samples)),
        directory=str(next(iter(samples.values())).parent),
    )
    fidelities: list[FidelityConfig] = []
    if cfg.search.successive_halving_enabled:
        fidelities.append(
            FidelityConfig(
                name="screen",
                train_sample=str(samples["screen_train"]),
                valid_sample=str(samples["screen_valid"]),
                iterations=int(cfg.model_params.get("screen_iterations", 120)),
                seeds=[cfg.search.seed],
            )
        )
    fidelities.extend(
        [
            FidelityConfig(
                name="search",
                train_sample=str(samples["search_train"]),
                valid_sample=str(samples["search_valid"]),
                iterations=int(cfg.model_params.get("search_iterations", 350)),
                seeds=[cfg.search.seed],
            ),
            FidelityConfig(
                name="confirm",
                train_sample=str(samples["confirm_train"]),
                valid_sample=str(samples["confirm_valid"]),
                iterations=int(cfg.model_params.get("confirm_iterations", 1000)),
                seeds=[cfg.search.seed, cfg.search.seed + 1, cfg.search.seed + 2],
            ),
        ]
    )
    model_params = dict(cfg.model_params)
    model_params.pop("screen_iterations", None)
    model_params.pop("search_iterations", None)
    model_params.pop("confirm_iterations", None)
    cfg.model_params = model_params
    resource_manager = (
        ResourceManager(cfg, output) if cfg.resources.enabled else None
    )
    dependency_versions = _dependency_versions()
    fingerprint = _stable_hash(
        {
            "selector_version": APP_VERSION,
            "dependency_versions": dependency_versions,
            "config": asdict(cfg),
            "features": features,
            "samples": {name: str(path) for name, path in samples.items()},
            "sources": {
                name: _source_fingerprint(path)
                for name, path in configured_split_paths(cfg).items()
            },
        }
    )
    _write_json(
        output / "run_manifest.json",
        {
            "fingerprint": fingerprint,
            "selector_version": APP_VERSION,
            "dependency_versions": dependency_versions,
            "config": asdict(cfg),
            "features": len(features),
            "validation_mode": validation_mode(cfg),
            "samples": {name: str(path) for name, path in samples.items()},
        },
    )
    cache = EvaluationCache(cfg.output.cache_directory, fingerprint)
    evaluator = CatBoostSubsetEvaluator(
        cfg, fidelities, cache, resource_manager=resource_manager
    )
    isolate_local = (
        cfg.execution.backend == "local"
        and cfg.execution.local_trial_mode == "process"
    )

    def new_search(
        interaction_pairs: Sequence[tuple[str, str, float]] = (),
        checkpoint: bool = False,
    ) -> InteractionAwareSearch:
        return InteractionAwareSearch(
            features=features,
            evaluator=evaluator,
            config=cfg.search,
            interaction_pairs=interaction_pairs,
            required_features=cfg.data.required_features,
            execution=cfg.execution,
            resource_manager=resource_manager,
            checkpoint_path=(output / "search_state.json") if checkpoint else None,
            checkpoint_fingerprint=fingerprint if checkpoint else None,
        )

    calibration_subset = [
        feature
        for feature in dict.fromkeys(
            [
                *cfg.data.required_features,
                *cfg.data.categorical_features,
                *features,
            ]
        )
        if feature in features
    ][: (cfg.search.search_max_features or cfg.search.max_features)]
    calibration_fidelity = "search"
    if resource_manager is not None:
        resource_manager.wait_for_single_trial("calibration_wait")
        calibration_restored = resource_manager.restore_calibration(
            fingerprint,
            calibration_subset,
            calibration_fidelity,
        )
        if (
            cfg.resources.calibration_enabled
            and cfg.execution.backend == "local"
            and not calibration_restored
        ):
            _event(
                "resource_calibration_started",
                subset_size=len(calibration_subset),
                fidelity=calibration_fidelity,
            )
            calibration_runner = new_search()

            def isolated_calibration() -> Evaluation:
                if cfg.execution.local_trial_mode == "thread":
                    return evaluator.evaluate(
                        calibration_subset,
                        calibration_fidelity,
                        use_cache=False,
                    )
                evaluation, diagnostic = (
                    calibration_runner._evaluate_many_process(
                        [frozenset(calibration_subset)],
                        calibration_fidelity,
                        use_cache=False,
                    )[0]
                )
                if evaluation is None:
                    raise RuntimeError(
                        "Isolated resource calibration failed: "
                        f"{diagnostic}"
                    )
                return evaluation

            resource_manager.calibrate(
                isolated_calibration,
                subset_size=len(calibration_subset),
                fidelity=calibration_fidelity,
            )
            _event(
                "resource_calibration_completed",
                ram_per_trial_gb=resource_manager.ram_per_trial_gb,
                gpu_memory_per_trial_gb=(
                    resource_manager.gpu_memory_per_trial_gb
                ),
                parallel_limit=resource_manager.max_parallel_trials,
            )
        elif calibration_restored:
            _event(
                "resource_calibration_restored",
                ram_per_trial_gb=resource_manager.ram_per_trial_gb,
                gpu_memory_per_trial_gb=(
                    resource_manager.gpu_memory_per_trial_gb
                ),
                parallel_limit=resource_manager.max_parallel_trials,
            )
        elif cfg.resources.calibration_enabled:
            _event(
                "resource_calibration_skipped",
                reason="ray_workers_may_differ_from_driver",
                ram_per_trial_gb=resource_manager.ram_per_trial_gb,
                gpu_memory_per_trial_gb=(
                    resource_manager.gpu_memory_per_trial_gb
                ),
            )
        resource_manager.write_plan(
            calibration_subset, calibration_fidelity, fingerprint
        )
        # Calibration may remove GPUs that no longer satisfy the measured
        # per-trial VRAM estimate; rebuild the device-slot queue accordingly.
        evaluator = CatBoostSubsetEvaluator(
            cfg, fidelities, cache, resource_manager=resource_manager
        )

    _event("baseline_started", n_features=len(features))
    if resource_manager is not None:
        resource_manager.wait_for_single_trial("baseline_wait")
    if isolate_local:
        baseline_runner = new_search()
        baseline = baseline_runner._evaluate_many_raw(
            [frozenset(features)], "search"
        )[0]
    else:
        baseline = evaluator.evaluate(features, "search")
    _event("baseline_completed", valid_metric=baseline.valid_metric)
    interaction_cache = evaluator._interaction_cache_path(
        features, "search", cfg.search.interaction_pairs
    )
    if isolate_local and not interaction_cache.exists():
        interaction_result = run_isolated_catboost_action(
            evaluator,
            resource_manager,
            "interaction_pairs",
            features,
            "search",
            interaction_limit=cfg.search.interaction_pairs,
        )
        interactions = [
            (str(first), str(second), float(strength))
            for first, second, strength in interaction_result["interactions"]
        ]
    else:
        interactions = evaluator.interaction_pairs(
            features, "search", cfg.search.interaction_pairs
        )
    _event("interactions_discovered", pairs=len(interactions))
    search = new_search(interactions, checkpoint=True)
    _event("search_started", restarts=cfg.search.restarts)
    search_front = search.run("search")
    _event(
        "search_completed",
        trials=len(search.history),
        screening_trials=len(search.screening_history),
        trial_diagnostics=len(search.trial_diagnostics),
        promotion_batches=len(search.promotion_batches),
        pareto_size=len(search_front),
        learned_coalitions=len(search.coalitions),
    )

    def evaluate_holdouts(
        subset: Sequence[str],
        holdouts: dict[str, str | Path],
    ) -> tuple[Evaluation, dict[str, SplitPredictions]]:
        normalized = {
            name: Path(path).resolve() for name, path in holdouts.items()
        }
        if isolate_local and not evaluator._prediction_cache_path(
            subset, "confirm", normalized
        ).exists():
            run_isolated_catboost_action(
                evaluator,
                resource_manager,
                "evaluate_with_holdouts",
                subset,
                "confirm",
                holdout_samples={
                    name: str(path) for name, path in normalized.items()
                },
            )
        return evaluator.evaluate_with_holdouts(
            subset, "confirm", normalized
        )

    eligible_baselines = (
        [baseline] if baseline.n_features <= cfg.search.max_features else []
    )
    final_size_pool = [
        item for item in [*eligible_baselines, *search_front]
        if item.n_features <= cfg.search.max_features
    ]
    if not final_size_pool:
        raise RuntimeError(
            "Search found no candidate within final max_features; "
            "increase search/local refinement budget or max_features"
        )
    candidates = choose_finalists(
        pareto_front(final_size_pool),
        limit=max(10, cfg.search.local_candidates * 2),
        allowed_drop=cfg.search.allowed_metric_drop,
    )
    robust_rows: list[RobustCandidate] = []
    robust_winner: RobustCandidate | None = None
    threshold_rows: list[ThresholdResult] = []
    threshold_winner: ThresholdResult | None = None
    selection_basis = "confirm_compact"
    if cfg.robust_validation.enabled and "oos" in samples:
        confirmed_all: list[Evaluation] = []
        oos_predictions: dict[tuple[str, ...], SplitPredictions] = {}
        threshold_valid_predictions: dict[
            tuple[str, ...], SplitPredictions
        ] = {}
        validation_items = list(candidates)
        if baseline.subset not in {item.subset for item in validation_items}:
            validation_items.append(baseline)
        _event("oos_prediction_started", candidates=len(validation_items))
        candidate_subsets = {item.subset for item in candidates}
        holdout_samples = {"oos": samples["oos"]}
        if cfg.decision_threshold.enabled:
            holdout_samples["threshold_valid"] = samples["confirm_valid"]
        for item in validation_items:
            evaluation, split_predictions = evaluate_holdouts(
                item.subset, holdout_samples
            )
            if evaluation.subset in candidate_subsets:
                confirmed_all.append(evaluation)
            oos_predictions[evaluation.subset] = split_predictions["oos"]
            if cfg.decision_threshold.enabled:
                threshold_valid_predictions[evaluation.subset] = (
                    split_predictions["threshold_valid"]
                )
        confirmed = pareto_front(confirmed_all)
        confirmed_subsets = {item.subset for item in confirmed}
        oos_predictions = {
            subset: prediction
            for subset, prediction in oos_predictions.items()
            if subset in confirmed_subsets or subset == baseline.subset
        }
        threshold_valid_predictions = {
            subset: prediction
            for subset, prediction in threshold_valid_predictions.items()
            if subset in oos_predictions
        }
        eligible_subsets = set(confirmed_subsets)
        if cfg.decision_threshold.enabled:
            threshold_rows = [
                select_decision_threshold(
                    subset,
                    threshold_valid_predictions[subset],
                    oos_prediction,
                    cfg.decision_threshold,
                )
                for subset, oos_prediction in oos_predictions.items()
            ]
            threshold_by_subset = {
                item.subset: item for item in threshold_rows
            }
            eligible_subsets = {
                subset
                for subset in confirmed_subsets
                if threshold_by_subset[subset].tuning_feasible
                and (
                    threshold_by_subset[subset].evaluation_feasible
                    or not cfg.decision_threshold.require_oos_feasible
                )
            }
            _write_json(
                output / "threshold_validation.json",
                {
                    "status": "evaluated",
                    "threshold_selected_on": "confirm_valid",
                    "threshold_tested_on": "oos",
                    "min_recall": cfg.decision_threshold.min_recall,
                    "min_precision": cfg.decision_threshold.min_precision,
                    "objective": cfg.decision_threshold.objective,
                    "require_oos_feasible": (
                        cfg.decision_threshold.require_oos_feasible
                    ),
                    "eligible_subsets": [
                        list(subset) for subset in sorted(eligible_subsets)
                    ],
                    "candidates": [asdict(item) for item in threshold_rows],
                },
            )
            if not eligible_subsets:
                raise RuntimeError(
                    "No confirmed feature subset satisfies the configured "
                    "decision-threshold constraints on confirm_valid and OOS"
                )
        else:
            _write_json(
                output / "threshold_validation.json",
                {"status": "skipped", "reason": "decision_threshold_disabled"},
            )
        margin = (
            cfg.robust_validation.noninferiority_margin
            if cfg.robust_validation.noninferiority_margin is not None
            else cfg.search.allowed_metric_drop
        )
        robust_rows, winner_subset = robust_paired_bootstrap(
            oos_predictions,
            metric_name=cfg.search.primary_metric,
            confidence_level=cfg.robust_validation.confidence_level,
            repeats=cfg.robust_validation.bootstrap_repeats,
            max_rows=cfg.robust_validation.bootstrap_max_rows,
            seed=cfg.robust_validation.bootstrap_seed,
            noninferiority_margin=margin,
            eligible_subsets=eligible_subsets,
            bootstrap_mode=cfg.robust_validation.bootstrap_mode,
        )
        winner = next(item for item in confirmed if item.subset == winner_subset)
        robust_winner = next(
            item for item in robust_rows if item.subset == winner_subset
        )
        if cfg.decision_threshold.enabled:
            threshold_winner = threshold_by_subset[winner_subset]
        selection_basis = (
            "oos_threshold_constrained_paired_bootstrap_lcb"
            if cfg.decision_threshold.enabled
            else "oos_paired_bootstrap_lcb"
        )
        _write_json(
            output / "robust_oos.json",
            {
                "selection_split": "oos",
                "primary_metric": cfg.search.primary_metric,
                "confidence_level": cfg.robust_validation.confidence_level,
                "bootstrap_repeats": cfg.robust_validation.bootstrap_repeats,
                "bootstrap_mode_requested": cfg.robust_validation.bootstrap_mode,
                "bootstrap_unit_used": robust_winner.bootstrap_unit,
                "ci_method": (
                    "centered_basic_cluster_stratified_paired_bootstrap"
                    if robust_winner.bootstrap_unit == "cluster"
                    else "centered_basic_row_stratified_paired_bootstrap"
                ),
                "reference_rule": "highest_oos_point_metric",
                "noninferiority_margin": margin,
                "winner_subset": list(winner_subset),
                "winner_noninferior_to_reference": robust_winner.noninferior,
                "candidates": [asdict(item) for item in robust_rows],
            },
        )
        _event(
            "oos_validation_completed",
            candidates=len(robust_rows),
            winner_features=len(winner_subset),
            winner_lcb=robust_winner.ci_low,
        )
    else:
        if cfg.decision_threshold.enabled:
            confirmed_evaluations = []
            threshold_valid_predictions = {}
            for item in candidates:
                evaluation, split_predictions = evaluate_holdouts(
                    item.subset,
                    {"threshold_valid": samples["confirm_valid"]},
                )
                confirmed_evaluations.append(evaluation)
                threshold_valid_predictions[evaluation.subset] = (
                    split_predictions["threshold_valid"]
                )
        elif isolate_local:
            confirmed_evaluations = search._evaluate_many_raw(
                [frozenset(item.subset) for item in candidates],
                "confirm",
            )
        else:
            confirmed_evaluations = [
                evaluator.evaluate(item.subset, "confirm")
                for item in candidates
            ]
        confirmed = pareto_front(confirmed_evaluations)
        if cfg.decision_threshold.enabled:
            threshold_rows = [
                select_decision_threshold(
                    item.subset,
                    threshold_valid_predictions[item.subset],
                    None,
                    cfg.decision_threshold,
                )
                for item in confirmed
            ]
            threshold_by_subset = {
                item.subset: item for item in threshold_rows
            }
            eligible_subsets = {
                item.subset for item in threshold_rows if item.tuning_feasible
            }
            if not eligible_subsets:
                raise RuntimeError(
                    "No confirmed feature subset satisfies the configured "
                    "decision-threshold constraints on confirm_valid"
                )
            winner = choose_compact(
                [
                    item
                    for item in confirmed
                    if item.subset in eligible_subsets
                ],
                cfg.search.allowed_metric_drop,
            )
            threshold_winner = threshold_by_subset[winner.subset]
            selection_basis = "confirm_valid_threshold_constrained_compact"
            _write_json(
                output / "threshold_validation.json",
                {
                    "status": "evaluated_without_oos",
                    "threshold_selected_on": "confirm_valid",
                    "threshold_tested_on": None,
                    "min_recall": cfg.decision_threshold.min_recall,
                    "min_precision": cfg.decision_threshold.min_precision,
                    "objective": cfg.decision_threshold.objective,
                    "eligible_subsets": [
                        list(subset) for subset in sorted(eligible_subsets)
                    ],
                    "candidates": [asdict(item) for item in threshold_rows],
                },
            )
        else:
            winner = choose_compact(confirmed, cfg.search.allowed_metric_drop)
            _write_json(
                output / "threshold_validation.json",
                {"status": "skipped", "reason": "decision_threshold_disabled"},
            )
        _write_json(
            output / "robust_oos.json",
            {
                "status": "skipped",
                "reason": (
                    "robust_validation_disabled"
                    if not cfg.robust_validation.enabled
                    else "oos_path_absent"
                ),
            },
        )
    _event("confirmation_completed", finalists=len(confirmed))
    winner = write_results(
        output,
        baseline,
        interactions,
        search.coalitions,
        search.history,
        search.screening_history,
        search.trial_diagnostics,
        search.promotion_batches,
        search_front,
        confirmed,
        winner,
        selection_basis,
        robust_winner,
        threshold_winner,
        search.restart_winners,
    )

    def evaluate_final_holdout(
        split_name: str,
        enabled: bool,
        file_name: str,
        seed_offset: int,
    ) -> None:
        event_prefix = f"{split_name}_evaluation"
        if (
            cfg.robust_validation.enabled
            and enabled
            and split_name in samples
        ):
            _event(f"{event_prefix}_started", n_features=winner.n_features)
            _, predictions = evaluate_holdouts(
                winner.subset, {split_name: samples[split_name]}
            )
            rows, _ = robust_paired_bootstrap(
                {winner.subset: predictions[split_name]},
                metric_name=cfg.search.primary_metric,
                confidence_level=cfg.robust_validation.confidence_level,
                repeats=cfg.robust_validation.bootstrap_repeats,
                max_rows=cfg.robust_validation.bootstrap_max_rows,
                seed=cfg.robust_validation.bootstrap_seed + seed_offset,
                noninferiority_margin=0.0,
                bootstrap_mode=cfg.robust_validation.bootstrap_mode,
            )
            frozen_threshold = None
            if threshold_winner is not None:
                precision, recall, f1, positive_rate = _threshold_metrics(
                    predictions[split_name].y_true,
                    predictions[split_name].prediction,
                    predictions[split_name].weight,
                    threshold_winner.threshold,
                )
                frozen_threshold = {
                    "threshold": threshold_winner.threshold,
                    "selected_on": threshold_winner.tuning_split,
                    "previously_tested_on": threshold_winner.evaluation_split,
                    "frozen_for_holdout": True,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "predicted_positive_rate": positive_rate,
                    "feasible": (
                        precision >= cfg.decision_threshold.min_precision
                        and recall >= cfg.decision_threshold.min_recall
                    ),
                }
            _write_json(
                output / file_name,
                {
                    "status": "evaluated_after_selection",
                    "split": split_name,
                    "used_for_selection": False,
                    "primary_metric": cfg.search.primary_metric,
                    "ci_method": (
                        "centered_basic_cluster_stratified_bootstrap"
                        if rows[0].bootstrap_unit == "cluster"
                        else "centered_basic_row_stratified_bootstrap"
                    ),
                    "result": asdict(rows[0]),
                    "decision_threshold": frozen_threshold,
                },
            )
            _event(
                f"{event_prefix}_completed",
                metric=rows[0].point_metric,
            )
            return
        _write_json(
            output / file_name,
            {
                "status": "skipped",
                "split": split_name,
                "used_for_selection": False,
                "reason": (
                    "robust_validation_disabled"
                    if not cfg.robust_validation.enabled
                    else f"{split_name}_disabled_or_path_absent"
                ),
            },
        )

    evaluate_final_holdout(
        "oot",
        cfg.robust_validation.evaluate_oot,
        "oot_final.json",
        10_000,
    )
    evaluate_final_holdout(
        "test",
        cfg.robust_validation.evaluate_test,
        "test_final.json",
        20_000,
    )

    synergy_rows = []
    selected_pairs = [
        (first, second)
        for first, second, _ in interactions
        if first in winner.subset and second in winner.subset
    ][:10]
    for first, second in selected_pairs:
        if winner.n_features > 2:
            if isolate_local:
                background = set(winner.subset) - {first, second}
                subsets = [
                    frozenset(background),
                    frozenset(background | {first}),
                    frozenset(background | {second}),
                    frozenset(background | {first, second}),
                ]
                evaluated = search._evaluate_many_raw(subsets, "confirm")
                score = {
                    item.subset: item.valid_metric for item in evaluated
                }
                q0, qa, qb, qab = (
                    score[tuple(sorted(subset))] for subset in subsets
                )
                synergy_rows.append(
                    {
                        "first": first,
                        "second": second,
                        "background_size": len(background),
                        "score_background": q0,
                        "score_first": qa,
                        "score_second": qb,
                        "score_pair": qab,
                        "synergy": qab - qa - qb + q0,
                    }
                )
            else:
                synergy_rows.append(
                    synergy_score(
                        evaluator, winner.subset, first, second, "confirm"
                    )
                )
    _write_json(output / "synergy.json", synergy_rows)
    if resource_manager is not None:
        resource_manager.write_plan(
            calibration_subset, calibration_fidelity, fingerprint
        )
        resource_manager.write_summary()
    _event(
        "pipeline_completed",
        output=str(output),
        selected_features=winner.n_features,
        valid_metric=winner.valid_metric,
    )


class DeterministicEvaluator:
    """Dependency-free evaluator used by the built-in search self-test."""

    primary_metric = "synthetic"

    def __init__(self) -> None:
        self.cache: dict[tuple[str, ...], Evaluation] = {}

    def evaluate(self, subset: Iterable[str], fidelity: str) -> Evaluation:
        selected = tuple(sorted(set(subset)))
        if selected in self.cache:
            return self.cache[selected]
        values = set(selected)
        score = 0.50
        if {"xor_a", "xor_b"} <= values:
            score += 0.35
        if "main" in values:
            score += 0.04
        score -= 0.0005 * len(values - {"xor_a", "xor_b", "main"})
        result = Evaluation(
            subset=selected,
            fidelity=fidelity,
            valid_metric=score,
            train_metric=score + 0.01,
            gap=0.01,
            metric_std=0.0,
            n_features=len(selected),
            runtime_seconds=0.0,
            metrics={"synthetic": score},
            constraint_violation=0.0,
        )
        self.cache[selected] = result
        return result


def self_test() -> None:
    features = ["xor_a", "xor_b", "main", *[f"noise_{index}" for index in range(17)]]
    config = SearchConfig(
        min_features=2,
        max_features=5,
        random_subspaces=25,
        adaptive_subspaces=15,
        subspace_min_features=2,
        subspace_max_features=5,
        population_size=16,
        generations=6,
        local_candidates=3,
        local_rounds=1,
        restarts=1,
        seed=7,
    )
    evaluator = DeterministicEvaluator()
    search = InteractionAwareSearch(
        features,
        evaluator,
        config,
        interaction_pairs=[("xor_a", "xor_b", 1.0)],
    )
    front = search.run("search")
    winner = choose_compact(front, allowed_drop=0.001)
    assert {"xor_a", "xor_b"} <= set(winner.subset), winner
    synergy = synergy_score(evaluator, [*winner.subset, "main"], "xor_a", "xor_b", "search")
    assert synergy["synergy"] > 0.30, synergy
    assert 0 <= pairwise_jaccard([winner.subset, winner.subset]) <= 1
    print(
        json.dumps(
            {"status": "ok", "winner": winner.subset, "synergy": synergy["synergy"]},
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interaction-aware multi-objective feature subset selection for CatBoost"
    )
    parser.add_argument("--config", type=Path, help="YAML configuration")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Inspect Parquet metadata and resources without training",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.config:
        parser.error("--config is required unless --self-test is used")
    if args.plan_only:
        print(
            json.dumps(
                run_preflight(args.config),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
