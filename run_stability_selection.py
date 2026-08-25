#!/usr/bin/env python3
"""Run interaction selector 0.10.x on development folds and aggregate stability."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require(module: str, package: str) -> Any:
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency {module!r}. Install with: pip install {package}"
        ) from exc


def _read_yaml(path: str | Path) -> dict[str, Any]:
    yaml = _require("yaml", "pyyaml")
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    yaml = _require("yaml", "pyyaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


@dataclass(slots=True)
class StabilityConfig:
    selector_script: str
    selector_template_config: str
    selector_data_config: str
    folds_manifest: str
    output_directory: str = "stability_selection"
    resume: bool = True
    run_final: bool = False
    candidate_min_frequency: float = 0.34
    core_min_frequency: float = 0.67
    pair_min_frequency: float = 0.34
    max_candidate_features: int = 300
    max_reported_pairs: int = 500
    restrict_final_universe: bool = True
    force_stable_core: bool = False
    always_keep_features: list[str] = field(default_factory=list)
    fold_overrides: dict[str, Any] = field(default_factory=dict)
    final_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StabilityConfig":
        raw = _read_yaml(path)
        return cls(**raw)

    def validate(self) -> None:
        for name in (
            "candidate_min_frequency",
            "core_min_frequency",
            "pair_min_frequency",
        ):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.core_min_frequency < self.candidate_min_frequency:
            raise ValueError("core_min_frequency must be >= candidate_min_frequency")
        if self.max_candidate_features < 1 or self.max_reported_pairs < 1:
            raise ValueError("Feature and pair limits must be positive")
        protected_fold_sections = {
            "data",
            "validation",
            "robust_validation",
            "decision_threshold",
            "output",
        }
        conflicting = sorted(protected_fold_sections & set(self.fold_overrides))
        if conflicting:
            raise ValueError(
                "fold_overrides cannot change selector contract sections: "
                f"{conflicting}"
            )
        conflicting_final = sorted({"data", "output"} & set(self.final_overrides))
        if conflicting_final:
            raise ValueError(
                "final_overrides cannot change selector data/output sections: "
                f"{conflicting_final}"
            )
        for path in (
            self.selector_script,
            self.selector_template_config,
            self.selector_data_config,
            self.folds_manifest,
        ):
            if not Path(path).exists():
                raise FileNotFoundError(path)


def aggregate_subsets(
    subsets: list[list[str]],
    candidate_min_frequency: float,
    core_min_frequency: float,
    pair_min_frequency: float,
    max_candidate_features: int,
    max_reported_pairs: int,
    always_keep: list[str] | None = None,
) -> dict[str, Any]:
    if not subsets:
        raise ValueError("No fold subsets to aggregate")
    normalized = [tuple(sorted(set(subset))) for subset in subsets]
    runs = len(normalized)
    feature_counts = Counter(feature for subset in normalized for feature in subset)
    pair_counts = Counter(
        pair
        for subset in normalized
        for pair in itertools.combinations(subset, 2)
    )
    candidate_min_count = max(1, math.ceil(candidate_min_frequency * runs - 1e-12))
    core_min_count = max(1, math.ceil(core_min_frequency * runs - 1e-12))
    pair_min_count = max(1, math.ceil(pair_min_frequency * runs - 1e-12))

    ranked_features = sorted(feature_counts, key=lambda name: (-feature_counts[name], name))
    candidate_pool = [
        name for name in ranked_features if feature_counts[name] >= candidate_min_count
    ][:max_candidate_features]
    for feature in always_keep or []:
        if feature not in candidate_pool:
            candidate_pool.append(feature)
    stable_core = [
        name for name in ranked_features if feature_counts[name] >= core_min_count
    ]
    stable_pairs = [
        {
            "first": pair[0],
            "second": pair[1],
            "count": count,
            "frequency": count / runs,
        }
        for pair, count in sorted(
            pair_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= pair_min_count
    ][:max_reported_pairs]

    jaccards: list[float] = []
    for left, right in itertools.combinations((set(item) for item in normalized), 2):
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
    feature_rows = [
        {
            "feature": name,
            "count": feature_counts[name],
            "frequency": feature_counts[name] / runs,
            "stable_core": name in stable_core,
            "candidate_pool": name in candidate_pool,
        }
        for name in ranked_features
    ]
    sizes = [len(item) for item in normalized]
    return {
        "runs": runs,
        "candidate_min_count": candidate_min_count,
        "core_min_count": core_min_count,
        "pair_min_count": pair_min_count,
        "candidate_pool": candidate_pool,
        "stable_core": stable_core,
        "stable_pairs": stable_pairs,
        "feature_stability": feature_rows,
        "subset_size": {
            "minimum": min(sizes),
            "maximum": max(sizes),
            "mean": sum(sizes) / len(sizes),
        },
        "pairwise_jaccard": {
            "minimum": min(jaccards) if jaccards else 1.0,
            "mean": sum(jaccards) / len(jaccards) if jaccards else 1.0,
            "maximum": max(jaccards) if jaccards else 1.0,
        },
    }


class StabilityRunner:
    def __init__(self, cfg: StabilityConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.root = Path(cfg.output_directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.template = _read_yaml(cfg.selector_template_config)
        self.base_data = _read_yaml(cfg.selector_data_config)["data"]
        self.fold_manifest = json.loads(
            Path(cfg.folds_manifest).read_text(encoding="utf-8")
        )
        self.folds = self.fold_manifest["folds"]
        if not self.folds:
            raise ValueError("folds_manifest contains no folds")
        self._validate_contract()

    def _validate_contract(self) -> None:
        if "audit" in self.template:
            raise ValueError(
                "Legacy selector template detected: replace 'audit' with "
                "'validation' and use interaction selector 0.10.x"
            )
        if "validation" not in self.template:
            raise ValueError("Selector template must contain a 'validation' section")
        required_data = {
            "train_path",
            "valid_path",
            "target",
            "sampling_key_columns",
        }
        missing_data = sorted(required_data - set(self.base_data))
        if missing_data:
            raise ValueError(
                f"selector_data_config is missing data keys: {missing_data}"
            )
        for fold in self.folds:
            missing_fold = sorted(
                {"name", "train_path", "valid_path"} - set(fold)
            )
            if missing_fold:
                raise ValueError(
                    f"Fold entry is missing keys {missing_fold}: {fold}"
                )

    def log(self, message: str) -> None:
        print(f"[{time.monotonic() - self.started:8.1f}s] {message}", flush=True)

    def _fold_config(self, fold: dict[str, Any], output: Path) -> dict[str, Any]:
        config = copy.deepcopy(self.template)
        config.setdefault("data", {}).update(copy.deepcopy(self.base_data))
        _deep_update(config, self.cfg.fold_overrides)
        config["data"]["train_path"] = fold["train_path"]
        config["data"]["valid_path"] = fold["valid_path"]
        for holdout in ("oos_path", "oot_path", "test_path"):
            config["data"].pop(holdout, None)
        config.setdefault("robust_validation", {}).update(
            {
                "enabled": False,
                "require_oos": False,
                "evaluate_oot": False,
                "evaluate_test": False,
            }
        )
        config.setdefault("decision_threshold", {})["enabled"] = False
        config.setdefault("validation", {}).update(
            {
                "check_id_overlap": self.fold_manifest.get("strategy")
                in {"group", "group_temporal"},
                "fail_on_id_overlap": True,
            }
        )
        config.setdefault("output", {}).update(
            {
                "directory": str(output),
                "cache_directory": str(output / "cache"),
            }
        )
        return config

    def _run_config(self, config_path: Path) -> None:
        command = [sys.executable, str(Path(self.cfg.selector_script).resolve()), "--config", str(config_path)]
        subprocess.run(command, check=True)

    def _run_fold(self, fold: dict[str, Any]) -> dict[str, Any]:
        output = self.root / "fold_runs" / fold["name"]
        config = self._fold_config(fold, output)
        config_path = self.root / "fold_configs" / f"{fold['name']}.yaml"
        _write_yaml(config_path, config)
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()
        marker = output / "stability_run.json"
        selected_path = output / "selected_features.json"
        if self.cfg.resume and marker.exists() and selected_path.exists():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous.get("config_sha256") == fingerprint:
                self.log(f"resume {fold['name']}")
                return json.loads(selected_path.read_text(encoding="utf-8"))
        self.log(f"run {fold['name']}")
        self._run_config(config_path)
        if not selected_path.exists():
            raise FileNotFoundError(f"Selector did not create {selected_path}")
        marker.write_text(
            json.dumps(
                {"config_sha256": fingerprint, "config": str(config_path)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return json.loads(selected_path.read_text(encoding="utf-8"))

    def _feature_universe(self, data: dict[str, Any]) -> list[str]:
        ds = _require("pyarrow.dataset", "pyarrow>=15")
        schema = ds.dataset(data["train_path"], format="parquet").schema
        controls = {
            data["target"],
            *data.get("id_columns", []),
            *data.get("sampling_key_columns", []),
            *data.get("leakage_key_columns", []),
            *data.get("bootstrap_key_columns", []),
            *data.get("excluded_features", []),
        }
        return [name for name in schema.names if name not in controls]

    def _final_config(self, aggregation: dict[str, Any]) -> dict[str, Any]:
        config = copy.deepcopy(self.template)
        config.setdefault("data", {}).update(copy.deepcopy(self.base_data))
        data = config["data"]
        original_required = list(data.get("required_features", []))
        keep = set(aggregation["candidate_pool"]) | set(original_required) | set(
            self.cfg.always_keep_features
        )
        if self.cfg.restrict_final_universe:
            universe = self._feature_universe(data)
            data["excluded_features"] = sorted(
                set(data.get("excluded_features", []))
                | {feature for feature in universe if feature not in keep}
            )
            data["categorical_features"] = [
                feature
                for feature in data.get("categorical_features", [])
                if feature in keep
            ]
        if self.cfg.force_stable_core:
            required = list(
                dict.fromkeys([*original_required, *aggregation["stable_core"]])
            )
            maximum = int(config.get("search", {}).get("max_features", len(required)))
            if len(required) > maximum:
                raise ValueError(
                    f"Required stable core has {len(required)} features, "
                    f"but selector search.max_features={maximum}"
                )
            data["required_features"] = required
        output = self.root / "final_run"
        config.setdefault("output", {}).update(
            {
                "directory": str(output),
                "cache_directory": str(output / "cache"),
            }
        )
        _deep_update(config, self.cfg.final_overrides)
        return config

    def _ensure_minimum_candidate_pool(self, aggregation: dict[str, Any]) -> None:
        minimum = int(self.template.get("search", {}).get("min_features", 1))
        pool = aggregation["candidate_pool"]
        for row in aggregation["feature_stability"]:
            if len(pool) >= minimum:
                break
            if row["feature"] not in pool:
                pool.append(row["feature"])
                row["candidate_pool"] = True
        if len(pool) < minimum:
            raise ValueError(
                f"Only {len(pool)} fold-selected features are available, "
                f"but selector search.min_features={minimum}"
            )
        if self.cfg.force_stable_core:
            maximum = int(self.template.get("search", {}).get("max_features", len(pool)))
            if len(aggregation["stable_core"]) > maximum:
                raise ValueError(
                    "Stable core is larger than selector search.max_features; "
                    "raise core_min_frequency or max_features"
                )

    def _write_reports(self, aggregation: dict[str, Any], fold_results: list[dict[str, Any]]) -> None:
        report = {
            "version": 2,
            "fold_results": fold_results,
            **aggregation,
        }
        (self.root / "stability_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (self.root / "feature_stability.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["feature", "count", "frequency", "stable_core", "candidate_pool"],
            )
            writer.writeheader()
            writer.writerows(aggregation["feature_stability"])
        with (self.root / "pair_stability.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["first", "second", "count", "frequency"],
            )
            writer.writeheader()
            writer.writerows(aggregation["stable_pairs"])

    def run(self) -> dict[str, Any]:
        selected = []
        fold_results = []
        for fold in self.folds:
            result = self._run_fold(fold)
            features = list(result["features"])
            selected.append(features)
            fold_results.append(
                {
                    "fold": fold["name"],
                    "features": features,
                    "n_features": len(features),
                    "valid_metric": result.get("valid_metric"),
                    "gap": result.get("gap"),
                }
            )
        aggregation = aggregate_subsets(
            selected,
            candidate_min_frequency=self.cfg.candidate_min_frequency,
            core_min_frequency=self.cfg.core_min_frequency,
            pair_min_frequency=self.cfg.pair_min_frequency,
            max_candidate_features=self.cfg.max_candidate_features,
            max_reported_pairs=self.cfg.max_reported_pairs,
            always_keep=self.cfg.always_keep_features,
        )
        self._ensure_minimum_candidate_pool(aggregation)
        self._write_reports(aggregation, fold_results)
        final_config = self._final_config(aggregation)
        final_config_path = self.root / "config.final.yaml"
        _write_yaml(final_config_path, final_config)
        if self.cfg.run_final:
            active = [
                role
                for role in ("train", "valid", "oos", "oot", "test")
                if self.base_data.get(f"{role}_path")
            ]
            self.log(f"run final selector on base {'/'.join(active)}")
            self._run_config(final_config_path)
        self.log(
            f"done: {len(aggregation['candidate_pool'])} candidate features, "
            f"{len(aggregation['stable_core'])} stable-core features"
        )
        return aggregation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        StabilityRunner(StabilityConfig.from_yaml(args.config)).run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
