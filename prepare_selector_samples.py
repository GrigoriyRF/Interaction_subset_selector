#!/usr/bin/env python3
"""Split an already model-ready population for interaction_subset_selector.py.

This module never cleans, imputes, casts, deduplicates, or filters factors. The
selector owns its internal screen/search/confirm samples. This module owns only
base train/valid and optional OOS/OOT/test roles plus development folds.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _require(module: str, package: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise RuntimeError(
            f"Missing dependency {module!r}. Install with: pip install {package}"
        ) from exc


@dataclass(slots=True)
class InputConfig:
    path: str
    target: str
    positive_label: Any = 1
    format: str = "parquet"
    id_columns: list[str] = field(default_factory=list)
    sampling_key_columns: list[str] = field(default_factory=list)
    leakage_key_columns: list[str] = field(default_factory=list)
    bootstrap_key_columns: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    categorical_null_strategy: str = "fill"
    categorical_null_value: str = "__MISSING__"
    excluded_features: list[str] = field(default_factory=list)
    required_features: list[str] = field(default_factory=list)
    csv_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SplitConfig:
    mode: str = "auto"
    train_fraction: float = 0.70
    valid_fraction: float = 0.15
    oos_fraction: float = 0.15
    oot_fraction: float = 0.0
    test_fraction: float = 0.0
    seed: int = 42
    time_column: str | None = None
    group_columns: list[str] = field(default_factory=list)
    purge_final_holdout_groups_from_history: bool = True
    fail_on_null_time: bool = True
    fail_on_null_group: bool = True


@dataclass(slots=True)
class OutputConfig:
    directory: str = "split_parts"
    overwrite: bool = False
    compression: str = "zstd"


@dataclass(slots=True)
class ValidationConfig:
    require_binary_target_in_every_split: bool = True
    check_group_overlap: bool = True


@dataclass(slots=True)
class FoldsConfig:
    enabled: bool = False
    strategy: str = "auto"
    n_folds: int = 3
    repeats: int = 1
    seed_stride: int = 10_000
    temporal_min_train_fraction: float = 0.50
    min_positive_rows: int = 1


@dataclass(slots=True)
class Config:
    input: InputConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    folds: FoldsConfig = field(default_factory=FoldsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        yaml = _require("yaml", "pyyaml")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise TypeError("Sample-splitting config must be a YAML mapping")
        allowed_sections = {"input", "split", "output", "validation", "folds"}
        unknown_sections = sorted(set(raw) - allowed_sections)
        if unknown_sections:
            raise ValueError(
                f"Unknown sample-splitting config sections: {unknown_sections}"
            )
        if "input" not in raw:
            raise ValueError("Config must contain an 'input' section")
        return cls(
            input=InputConfig(**raw["input"]),
            split=SplitConfig(**raw.get("split", {})),
            output=OutputConfig(**raw.get("output", {})),
            validation=ValidationConfig(**raw.get("validation", {})),
            folds=FoldsConfig(**raw.get("folds", {})),
        )

    def resolved_mode(self) -> str:
        mode = self.split.mode.lower()
        if mode != "auto":
            return mode
        if self.split.time_column and self.group_columns:
            return "group_temporal"
        if self.split.time_column:
            return "temporal"
        if self.group_columns:
            return "group"
        return "stratified"

    @property
    def group_columns(self) -> list[str]:
        return self.split.group_columns or self.input.leakage_key_columns

    @property
    def downstream_leakage_keys(self) -> list[str]:
        return list(dict.fromkeys([*self.input.leakage_key_columns, *self.group_columns]))

    def validate(self) -> None:
        modes = {"random", "stratified", "group", "temporal", "group_temporal"}
        mode = self.resolved_mode()
        if mode not in modes:
            raise ValueError(f"Unsupported split.mode={self.split.mode!r}; choose {sorted(modes | {'auto'})}")
        if not self.input.sampling_key_columns:
            raise ValueError(
                "input.sampling_key_columns is required by the downstream selector"
            )
        if self.input.categorical_null_strategy not in {"fill", "error"}:
            raise ValueError(
                "input.categorical_null_strategy must be 'fill' or 'error'"
            )
        if (
            self.input.categorical_null_strategy == "fill"
            and not self.input.categorical_null_value
        ):
            raise ValueError(
                "input.categorical_null_value must be a non-empty string "
                "when categorical_null_strategy='fill'"
            )
        fractions = self.fractions()
        if any(value < 0 or value >= 1 for value in fractions.values()):
            raise ValueError(f"Each split fraction must be in [0, 1): {fractions}")
        if fractions["train"] <= 0 or fractions["valid"] <= 0:
            raise ValueError("train_fraction and valid_fraction must be positive")
        if abs(sum(fractions.values()) - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0: {fractions}")
        final_temporal_fraction = fractions["oot"] + fractions["test"]
        if mode in {"temporal", "group_temporal"}:
            if not self.split.time_column:
                raise ValueError(f"split.time_column is required for mode={mode}")
            if final_temporal_fraction <= 0:
                raise ValueError(
                    f"oot_fraction or test_fraction must be positive for mode={mode}"
                )
        elif fractions["oot"] != 0:
            raise ValueError("A random/group OOT is misleading; set oot_fraction=0 or use a temporal mode")
        if mode in {"group", "group_temporal"} and not self.group_columns:
            raise ValueError(f"group columns are required for mode={mode}")
        if self.input.format.lower() not in {"parquet", "csv"}:
            raise ValueError("input.format must be 'parquet' or 'csv'")
        if self.folds.enabled:
            if self.folds.n_folds < 2:
                raise ValueError("folds.n_folds must be at least 2")
            if self.folds.repeats < 1:
                raise ValueError("folds.repeats must be positive")
            if self.folds.min_positive_rows < 1:
                raise ValueError("folds.min_positive_rows must be positive")
            if not 0 < self.folds.temporal_min_train_fraction < 1:
                raise ValueError("folds.temporal_min_train_fraction must be in (0, 1)")
            fold_strategy = self.resolved_fold_strategy()
            if fold_strategy not in {
                "random",
                "stratified",
                "group",
                "temporal",
                "group_temporal",
            }:
                raise ValueError(f"Unsupported folds.strategy={self.folds.strategy!r}")
            if fold_strategy in {"temporal", "group_temporal"} and self.folds.repeats != 1:
                raise ValueError("Temporal folds do not support repeats; use repeats=1")
            if fold_strategy in {"temporal", "group_temporal"} and not self.split.time_column:
                raise ValueError(f"folds.strategy={fold_strategy} requires split.time_column")
            if fold_strategy in {"group", "group_temporal"} and not self.group_columns:
                raise ValueError(f"folds.strategy={fold_strategy} requires group columns")
        self._validate_path_layout()

    def resolved_fold_strategy(self) -> str:
        strategy = self.folds.strategy.lower()
        return self.resolved_mode() if strategy == "auto" else strategy

    def _validate_path_layout(self) -> None:
        if glob.has_magic(self.input.path):
            return
        source = Path(self.input.path).expanduser().resolve()
        output = Path(self.output.directory).expanduser().resolve()
        if source.is_dir() and (
            source == output
            or source.is_relative_to(output)
            or output.is_relative_to(source)
        ):
            raise ValueError(
                "Input and output directories must not contain one another; "
                "use a separate split_parts directory"
            )

    def fractions(self) -> dict[str, float]:
        return {
            "train": self.split.train_fraction,
            "valid": self.split.valid_fraction,
            "oos": self.split.oos_fraction,
            "oot": self.split.oot_fraction,
            "test": self.split.test_fraction,
        }


class SampleBuilder:
    def __init__(self, cfg: Config) -> None:
        cfg.validate()
        self.cfg = cfg
        self.pl = _require("polars", "polars>=1.30")
        self.started = time.monotonic()
        self.root = Path(cfg.output.directory).expanduser().resolve()
        self.source = self._scan_source()
        self.schema = dict(self.source.collect_schema())
        self.mode = cfg.resolved_mode()
        self._validate_columns()

    def log(self, message: str) -> None:
        print(f"[{time.monotonic() - self.started:8.1f}s] {message}", flush=True)

    def _scan_source(self) -> Any:
        path = self.cfg.input.path
        if self.cfg.input.format.lower() == "csv":
            return self.pl.scan_csv(path, **self.cfg.input.csv_options)
        source = Path(path)
        if source.is_dir():
            files = sorted(str(item) for item in source.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No Parquet files found under {source}")
            return self.pl.scan_parquet(files)
        return self.pl.scan_parquet(path)

    def _validate_columns(self) -> None:
        required = {
            self.cfg.input.target,
            *self.cfg.input.id_columns,
            *self.cfg.input.sampling_key_columns,
            *self.cfg.input.leakage_key_columns,
            *self.cfg.input.bootstrap_key_columns,
            *self.cfg.input.categorical_features,
            *self.cfg.input.required_features,
            *self.cfg.group_columns,
        }
        if self.cfg.split.time_column:
            required.add(self.cfg.split.time_column)
        missing = sorted(required - set(self.schema))
        if missing:
            raise ValueError(f"Input columns are missing: {missing}")
        target_nulls = self._scalar(
            self.source.select(self.pl.col(self.cfg.input.target).null_count().alias("value"))
        )
        if target_nulls:
            raise ValueError(f"Target contains {target_nulls} null values")
        if self.cfg.split.time_column and self.cfg.split.fail_on_null_time:
            nulls = self._scalar(
                self.source.select(
                    self.pl.col(self.cfg.split.time_column).null_count().alias("value")
                )
            )
            if nulls:
                raise ValueError(f"Time column contains {nulls} null values")
        if self.cfg.group_columns and self.cfg.split.fail_on_null_group:
            null_expr = self.pl.any_horizontal(
                [self.pl.col(column).is_null() for column in self.cfg.group_columns]
            ).sum().alias("value")
            nulls = self._scalar(self.source.select(null_expr))
            if nulls:
                raise ValueError(f"Group key contains nulls in {nulls} rows")

    def _scalar(self, query: Any) -> Any:
        return query.collect(engine="streaming")[0, "value"]

    def _hash_bucket(self, columns: Iterable[str], seed: int | None = None) -> Any:
        expressions = [self.pl.col(column) for column in columns]
        hashed = self.pl.struct(expressions).hash(
            seed=self.cfg.split.seed if seed is None else seed
        )
        return (hashed % 10_000_000).cast(self.pl.Float64) / 10_000_000.0

    def _hash_splits(self, source: Any, roles: list[str], columns: list[str]) -> dict[str, Any]:
        fractions = self.cfg.fractions()
        total = sum(fractions[role] for role in roles)
        if total <= 0:
            raise ValueError("No positive fractions for hash split")
        bucket = self._hash_bucket(columns)
        result: dict[str, Any] = {}
        lower = 0.0
        for index, role in enumerate(roles):
            upper = lower + fractions[role] / total
            if index == len(roles) - 1:
                predicate = bucket >= lower
            elif lower == 0:
                predicate = bucket < upper
            else:
                predicate = (bucket >= lower) & (bucket < upper)
            result[role] = source.filter(predicate)
            lower = upper
        return result

    def _time_cutoffs(self, roles: list[str], source: Any | None = None) -> dict[str, Any]:
        source = source if source is not None else self.source
        time_col = self.cfg.split.time_column
        assert time_col is not None
        fractions = self.cfg.fractions()
        cumulative = 0.0
        expressions = []
        for role in roles[:-1]:
            cumulative += fractions[role]
            expressions.append(
                self.pl.col(time_col)
                .quantile(cumulative, interpolation="nearest")
                .alias(role)
            )
        row = source.select(expressions).collect(engine="streaming").row(0, named=True)
        return dict(row)

    def _temporal_splits(self) -> tuple[dict[str, Any], dict[str, Any]]:
        roles = [role for role, value in self.cfg.fractions().items() if value > 0]
        cutoffs = self._time_cutoffs(roles)
        time_col = self.cfg.split.time_column
        assert time_col is not None
        result: dict[str, Any] = {}
        lower = None
        for index, role in enumerate(roles):
            if index == len(roles) - 1:
                predicate = self.pl.lit(True) if lower is None else self.pl.col(time_col) > lower
            else:
                upper = cutoffs[role]
                predicate = self.pl.col(time_col) <= upper
                if lower is not None:
                    predicate &= self.pl.col(time_col) > lower
                elif not self.cfg.split.fail_on_null_time:
                    predicate |= self.pl.col(time_col).is_null()
                lower = upper
            result[role] = self.source.filter(predicate)
        return result, cutoffs

    def _group_temporal_splits(self) -> tuple[dict[str, Any], dict[str, Any], Any]:
        time_col = self.cfg.split.time_column
        assert time_col is not None
        fractions = self.cfg.fractions()
        final_roles = [
            role for role in ("oot", "test") if fractions[role] > 0
        ]
        final_fraction = sum(fractions[role] for role in final_roles)
        history_cutoff = self._scalar(
            self.source.select(
                self.pl.col(time_col)
                .quantile(1.0 - final_fraction, interpolation="nearest")
                .alias("value")
            )
        )
        history_predicate = self.pl.col(time_col) <= history_cutoff
        if not self.cfg.split.fail_on_null_time:
            history_predicate |= self.pl.col(time_col).is_null()
        history = self.source.filter(history_predicate)
        final_splits: dict[str, Any] = {}
        cutoffs: dict[str, Any] = {"history_end": history_cutoff}
        lower = history_cutoff
        if fractions["oot"] > 0:
            if fractions["test"] > 0:
                oot_end = self._scalar(
                    self.source.select(
                        self.pl.col(time_col)
                        .quantile(
                            1.0 - fractions["test"],
                            interpolation="nearest",
                        )
                        .alias("value")
                    )
                )
                final_splits["oot"] = self.source.filter(
                    (self.pl.col(time_col) > lower)
                    & (self.pl.col(time_col) <= oot_end)
                )
                cutoffs["oot_end"] = oot_end
                lower = oot_end
            else:
                final_splits["oot"] = self.source.filter(
                    self.pl.col(time_col) > lower
                )
        if fractions["test"] > 0:
            final_splits["test"] = self.source.filter(
                self.pl.col(time_col) > lower
            )

        purged_frames: list[Any] = []
        if "oot" in final_splits and "test" in final_splits:
            test_groups = final_splits["test"].select(
                self.cfg.group_columns
            ).unique()
            purged_frames.append(
                final_splits["oot"].join(
                    test_groups,
                    on=self.cfg.group_columns,
                    how="semi",
                )
            )
            final_splits["oot"] = final_splits["oot"].join(
                test_groups,
                on=self.cfg.group_columns,
                how="anti",
            )

        if self.cfg.split.purge_final_holdout_groups_from_history:
            holdout_groups = self.pl.concat(
                [
                    frame.select(self.cfg.group_columns)
                    for frame in final_splits.values()
                ],
                how="vertical",
            ).unique()
            purged_frames.append(
                history.join(
                    holdout_groups,
                    on=self.cfg.group_columns,
                    how="semi",
                )
            )
            history = history.join(
                holdout_groups,
                on=self.cfg.group_columns,
                how="anti",
            )
        roles = [
            role
            for role in ("train", "valid", "oos")
            if fractions[role] > 0
        ]
        result = self._hash_splits(history, roles, self.cfg.group_columns)
        result.update(final_splits)
        purged = (
            self.pl.concat(purged_frames, how="vertical")
            if purged_frames
            else None
        )
        return result, cutoffs, purged

    def _build_queries(self) -> tuple[dict[str, Any], dict[str, Any], Any]:
        if self.mode == "temporal":
            splits, cutoffs = self._temporal_splits()
            return splits, cutoffs, None
        if self.mode == "group_temporal":
            return self._group_temporal_splits()
        roles = [
            role
            for role in ("train", "valid", "oos", "test")
            if self.cfg.fractions()[role] > 0
        ]
        if self.mode == "group":
            columns = self.cfg.group_columns
        elif self.mode == "stratified":
            columns = [*self.cfg.input.sampling_key_columns, self.cfg.input.target]
        else:
            columns = self.cfg.input.sampling_key_columns
        return self._hash_splits(self.source, roles, columns), {}, None

    def _prepare_output(self) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            if not self.cfg.output.overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {self.root}. Set output.overwrite=true explicitly."
                )
            for name in ("train", "valid", "oos", "oot", "test", "folds"):
                candidate = self.root / name
                if candidate.exists():
                    shutil.rmtree(candidate)
            for name in (
                "split_manifest.json",
                "selector_data.yaml",
                "folds_manifest.json",
            ):
                candidate = self.root / name
                if candidate.exists():
                    candidate.unlink()
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_splits(self, queries: dict[str, Any]) -> dict[str, str]:
        paths: dict[str, str] = {}
        for role, query in queries.items():
            destination = self.root / role
            self._write_query(query, destination, role)
            paths[role] = str(destination)
        return paths

    def _write_query(self, query: Any, destination: Path, label: str) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        file = destination / "part-00000.parquet"
        temporary = destination / "part-00000.parquet.tmp"
        if temporary.exists():
            temporary.unlink()
        self.log(f"writing {label} -> {file}")
        query.sink_parquet(
            temporary,
            compression=self.cfg.output.compression,
            mkdir=True,
        )
        temporary.replace(file)

    def _fold_hash_columns(self, strategy: str) -> list[str]:
        if strategy == "group":
            if not self.cfg.group_columns:
                raise ValueError("Group folds require group columns")
            return self.cfg.group_columns
        if strategy == "stratified":
            return [*self.cfg.input.sampling_key_columns, self.cfg.input.target]
        return self.cfg.input.sampling_key_columns

    def _hash_fold_queries(self, development: Any, strategy: str) -> list[dict[str, Any]]:
        folds: list[dict[str, Any]] = []
        columns = self._fold_hash_columns(strategy)
        for repeat in range(self.cfg.folds.repeats):
            seed = self.cfg.split.seed + repeat * self.cfg.folds.seed_stride
            bucket = self._hash_bucket(columns, seed=seed)
            fold_index = (bucket * self.cfg.folds.n_folds).floor().cast(self.pl.Int32)
            for fold in range(self.cfg.folds.n_folds):
                valid_predicate = fold_index == fold
                folds.append(
                    {
                        "name": f"repeat_{repeat:02d}_fold_{fold:02d}",
                        "repeat": repeat,
                        "fold": fold,
                        "seed": seed,
                        "train": development.filter(~valid_predicate),
                        "valid": development.filter(valid_predicate),
                        "cutoffs": {},
                    }
                )
        return folds

    def _temporal_fold_queries(
        self,
        development: Any,
        purge_groups: bool,
    ) -> list[dict[str, Any]]:
        time_col = self.cfg.split.time_column
        if not time_col:
            raise ValueError("Temporal folds require split.time_column")
        minimum = self.cfg.folds.temporal_min_train_fraction
        width = (1.0 - minimum) / self.cfg.folds.n_folds
        quantiles: list[float] = [minimum + index * width for index in range(self.cfg.folds.n_folds + 1)]
        expressions = [
            self.pl.col(time_col)
            .quantile(value, interpolation="nearest")
            .alias(f"q_{index}")
            for index, value in enumerate(quantiles)
        ]
        cutoffs = development.select(expressions).collect(engine="streaming").row(0, named=True)
        folds: list[dict[str, Any]] = []
        for fold in range(self.cfg.folds.n_folds):
            lower = cutoffs[f"q_{fold}"]
            upper = cutoffs[f"q_{fold + 1}"]
            train_predicate = self.pl.col(time_col) <= lower
            if not self.cfg.split.fail_on_null_time:
                train_predicate |= self.pl.col(time_col).is_null()
            valid_predicate = (self.pl.col(time_col) > lower) & (
                self.pl.col(time_col) <= upper
            )
            train = development.filter(train_predicate)
            valid = development.filter(valid_predicate)
            purged = None
            if purge_groups:
                if not self.cfg.group_columns:
                    raise ValueError("Group-temporal folds require group columns")
                valid_groups = valid.select(self.cfg.group_columns).unique()
                purged = train.join(valid_groups, on=self.cfg.group_columns, how="semi")
                train = train.join(valid_groups, on=self.cfg.group_columns, how="anti")
            folds.append(
                {
                    "name": f"repeat_00_fold_{fold:02d}",
                    "repeat": 0,
                    "fold": fold,
                    "seed": self.cfg.split.seed,
                    "train": train,
                    "valid": valid,
                    "purged": purged,
                    "cutoffs": {"train_end": lower, "valid_end": upper},
                }
            )
        return folds

    def _positive_rows(self, stats: dict[str, Any]) -> int:
        return next(
            (
                int(item["rows"])
                for item in stats["class_counts"]
                if item["label"] == self.cfg.input.positive_label
            ),
            0,
        )

    def _fold_overlap(self, train_path: Path, valid_path: Path) -> int:
        if not self.cfg.group_columns:
            return 0
        train = self.pl.scan_parquet(str(train_path / "*.parquet")).select(
            self.cfg.group_columns
        ).unique()
        valid = self.pl.scan_parquet(str(valid_path / "*.parquet")).select(
            self.cfg.group_columns
        ).unique()
        return int(
            self._scalar(
                train.join(valid, on=self.cfg.group_columns, how="inner")
                .select(self.pl.len().alias("value"))
            )
        )

    def _write_folds(self, base_queries: dict[str, Any]) -> dict[str, Any] | None:
        if not self.cfg.folds.enabled:
            return None
        development = self.pl.concat(
            [base_queries["train"], base_queries["valid"]],
            how="vertical",
        )
        strategy = self.cfg.resolved_fold_strategy()
        if strategy == "temporal":
            queries = self._temporal_fold_queries(development, purge_groups=False)
        elif strategy == "group_temporal":
            queries = self._temporal_fold_queries(development, purge_groups=True)
        else:
            queries = self._hash_fold_queries(development, strategy)

        root = self.root / "folds"
        rows: list[dict[str, Any]] = []
        for item in queries:
            fold_root = root / item["name"]
            train_path = fold_root / "train"
            valid_path = fold_root / "valid"
            self._write_query(item["train"], train_path, f"{item['name']}/train")
            self._write_query(item["valid"], valid_path, f"{item['name']}/valid")
            train_stats = self._split_stats(str(train_path))
            valid_stats = self._split_stats(str(valid_path))
            for role, stats in (("train", train_stats), ("valid", valid_stats)):
                if stats["rows"] == 0:
                    raise ValueError(f"{item['name']} {role} is empty")
                if len(stats["class_counts"]) != 2:
                    raise ValueError(
                        f"{item['name']} {role} must contain exactly two target classes"
                    )
                positives = self._positive_rows(stats)
                if positives < self.cfg.folds.min_positive_rows:
                    raise ValueError(
                        f"{item['name']} {role} has only {positives} positive rows; "
                        f"minimum is {self.cfg.folds.min_positive_rows}"
                    )
            overlap = 0
            if strategy in {"group", "group_temporal"}:
                overlap = self._fold_overlap(train_path, valid_path)
                if overlap:
                    raise ValueError(
                        f"{item['name']} has {overlap} overlapping groups between train and valid"
                    )
            purged_rows = 0
            if item.get("purged") is not None:
                purged_rows = int(
                    self._scalar(item["purged"].select(self.pl.len().alias("value")))
                )
            rows.append(
                {
                    "name": item["name"],
                    "repeat": item["repeat"],
                    "fold": item["fold"],
                    "seed": item["seed"],
                    "train_path": str(train_path),
                    "valid_path": str(valid_path),
                    "train_stats": train_stats,
                    "valid_stats": valid_stats,
                    "group_overlap": overlap,
                    "purged_train_rows": purged_rows,
                    "cutoffs": {
                        key: _json_value(value) for key, value in item["cutoffs"].items()
                    },
                }
            )
        result = {
            "version": 2,
            "strategy": strategy,
            "development_source": ["train", "valid"],
            "excluded_holdouts": [
                role
                for role in ("oos", "oot", "test")
                if role in base_queries
            ],
            "n_folds": self.cfg.folds.n_folds,
            "repeats": self.cfg.folds.repeats,
            "folds": rows,
        }
        path = self.root / "folds_manifest.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_value),
            encoding="utf-8",
        )
        return result

    def _split_stats(self, path: str) -> dict[str, Any]:
        frame = self.pl.scan_parquet(str(Path(path) / "*.parquet"))
        target = self.cfg.input.target
        rows = int(self._scalar(frame.select(self.pl.len().alias("value"))))
        counts_frame = (
            frame.group_by(target)
            .agg(self.pl.len().alias("rows"))
            .sort(target)
            .collect(engine="streaming")
        )
        class_counts = [
            {"label": item[target], "rows": int(item["rows"])}
            for item in counts_frame.to_dicts()
        ]
        stats: dict[str, Any] = {"rows": rows, "class_counts": class_counts}
        if self.cfg.split.time_column:
            time_col = self.cfg.split.time_column
            bounds = frame.select(
                self.pl.col(time_col).min().alias("minimum"),
                self.pl.col(time_col).max().alias("maximum"),
            ).collect(engine="streaming").row(0, named=True)
            stats["time_min"] = _json_value(bounds["minimum"])
            stats["time_max"] = _json_value(bounds["maximum"])
        if self.cfg.group_columns:
            groups = int(
                self._scalar(
                    frame.select(self.cfg.group_columns)
                    .unique()
                    .select(self.pl.len().alias("value"))
                )
            )
            stats["groups"] = groups
        return stats

    def _validate_outputs(self, paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, int]]:
        stats = {role: self._split_stats(path) for role, path in paths.items()}
        for role, item in stats.items():
            if item["rows"] == 0:
                raise ValueError(f"Generated split {role!r} is empty")
            if (
                self.cfg.validation.require_binary_target_in_every_split
                and len(item["class_counts"]) != 2
            ):
                raise ValueError(
                    f"Split {role!r} must contain exactly two target classes: {item['class_counts']}"
                )
            labels = [entry["label"] for entry in item["class_counts"]]
            if self.cfg.input.positive_label not in labels:
                raise ValueError(
                    f"positive_label={self.cfg.input.positive_label!r} is absent in {role}"
                )
        overlaps: dict[str, int] = {}
        if (
            self.cfg.validation.check_group_overlap
            and self.cfg.group_columns
            and self.mode in {"group", "group_temporal"}
        ):
            roles = list(paths)
            key_frames = {
                role: self.pl.scan_parquet(str(Path(path) / "*.parquet"))
                .select(self.cfg.group_columns)
                .unique()
                for role, path in paths.items()
            }
            for left_index, left in enumerate(roles):
                for right in roles[left_index + 1 :]:
                    overlap = int(
                        self._scalar(
                            key_frames[left]
                            .join(key_frames[right], on=self.cfg.group_columns, how="inner")
                            .select(self.pl.len().alias("value"))
                        )
                    )
                    overlaps[f"{left}__{right}"] = overlap
                    if overlap:
                        raise ValueError(
                            f"Found {overlap} overlapping groups between {left} and {right}"
                        )
        return stats, overlaps

    def _categorical_features(self) -> list[str]:
        if self.cfg.input.categorical_features:
            return sorted(set(self.cfg.input.categorical_features))
        controls = {
            self.cfg.input.target,
            *self.cfg.input.id_columns,
            *self.cfg.input.sampling_key_columns,
            *self.cfg.input.leakage_key_columns,
            *self.cfg.input.bootstrap_key_columns,
            *self.cfg.input.excluded_features,
            *self.cfg.group_columns,
        }
        return sorted(
            column
            for column, dtype in self.schema.items()
            if column not in controls
            and any(
                token in str(dtype).lower()
                for token in ("string", "utf8", "categorical", "enum", "bool")
            )
        )

    def _write_selector_config(self, paths: dict[str, str]) -> Path:
        yaml = _require("yaml", "pyyaml")
        data: dict[str, Any] = {
            "train_path": paths["train"],
            "valid_path": paths["valid"],
        }
        for role in ("oos", "oot", "test"):
            if role in paths:
                data[f"{role}_path"] = paths[role]
        data.update(
            {
                "target": self.cfg.input.target,
                "positive_label": self.cfg.input.positive_label,
                "id_columns": self.cfg.input.id_columns,
                "sampling_key_columns": self.cfg.input.sampling_key_columns,
                "categorical_features": self._categorical_features(),
                "categorical_null_strategy": (
                    self.cfg.input.categorical_null_strategy
                ),
                "categorical_null_value": self.cfg.input.categorical_null_value,
                "excluded_features": self.cfg.input.excluded_features,
                "required_features": self.cfg.input.required_features,
                "leakage_key_columns": self.cfg.downstream_leakage_keys,
                "bootstrap_key_columns": self.cfg.input.bootstrap_key_columns,
            }
        )
        output = self.root / "selector_data.yaml"
        output.write_text(
            yaml.safe_dump({"data": data}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return output

    def build(self) -> dict[str, Any]:
        self.log(f"mode={self.mode}; validating and planning split")
        self._prepare_output()
        queries, cutoffs, purged = self._build_queries()
        input_rows = int(self._scalar(self.source.select(self.pl.len().alias("value"))))
        paths = self._write_splits(queries)
        folds = self._write_folds(queries)
        self.log("validating generated samples")
        stats, overlaps = self._validate_outputs(paths)
        purged_rows = 0
        if purged is not None:
            purged_rows = int(self._scalar(purged.select(self.pl.len().alias("value"))))
        assigned_rows = sum(item["rows"] for item in stats.values())
        unaccounted_rows = input_rows - assigned_rows - purged_rows
        if unaccounted_rows:
            raise ValueError(
                "Row accounting failed: "
                f"input={input_rows}, assigned={assigned_rows}, "
                f"purged={purged_rows}, unaccounted={unaccounted_rows}"
            )
        selector_config = self._write_selector_config(paths)
        schema_text = "\n".join(f"{name}:{dtype}" for name, dtype in self.schema.items())
        manifest = {
            "version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "input_path": str(Path(self.cfg.input.path).expanduser()),
            "input_rows": input_rows,
            "fractions": self.cfg.fractions(),
            "seed": self.cfg.split.seed,
            "polars_version": getattr(self.pl, "__version__", "unknown"),
            "time_column": self.cfg.split.time_column,
            "group_columns": self.cfg.group_columns,
            "time_cutoffs": {key: _json_value(value) for key, value in cutoffs.items()},
            "purged_rows": purged_rows,
            "assigned_rows": assigned_rows,
            "unaccounted_rows": unaccounted_rows,
            "paths": paths,
            "stats": stats,
            "group_overlaps": overlaps,
            "schema_sha256": hashlib.sha256(schema_text.encode()).hexdigest(),
            "selector_config": str(selector_config),
            "folds_manifest": (
                str(self.root / "folds_manifest.json") if folds is not None else None
            ),
            "config": asdict(self.cfg),
        }
        manifest_path = self.root / "split_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_value),
            encoding="utf-8",
        )
        self.log(f"done: {manifest_path}")
        return manifest


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="YAML base-sample splitting config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = SampleBuilder(Config.from_yaml(args.config)).build()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "paths": manifest["paths"],
                "stats": manifest["stats"],
                "folds_manifest": manifest["folds_manifest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
