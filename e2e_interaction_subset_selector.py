from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from interaction_subset_selector import APP_VERSION, run_pipeline


REQUIRED_PACKAGES = ("catboost", "numpy", "polars", "pyarrow", "yaml")


def missing_dependencies() -> list[str]:
    return [
        package
        for package in REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]


def _make_split(
    split: str,
    rows: int,
    seed: int,
    positive_keep_probability: float,
) -> Any:
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(seed)
    pool_rows = rows * 12
    xor_a = rng.integers(0, 2, pool_rows, dtype="int8")
    xor_b = rng.integers(0, 2, pool_rows, dtype="int8")
    tri_a = rng.integers(0, 2, pool_rows, dtype="int8")
    tri_b = rng.integers(0, 2, pool_rows, dtype="int8")
    tri_c = rng.integers(0, 2, pool_rows, dtype="int8")
    triple_event = (
        (tri_a == 1)
        & (tri_b == 1)
        & (tri_c == 1)
        & (rng.random(pool_rows) < 0.40)
    )
    target = ((xor_a != xor_b) | triple_event).astype("int8")
    keep = (target == 0) | (rng.random(pool_rows) < positive_keep_probability)
    selected = np.flatnonzero(keep)
    if len(selected) < rows:
        raise RuntimeError("Synthetic candidate pool is too small")
    selected = rng.permutation(selected)[:rows]

    xor_a = xor_a[selected]
    xor_b = xor_b[selected]
    tri_a = tri_a[selected]
    tri_b = tri_b[selected]
    tri_c = tri_c[selected]
    target = target[selected]
    required_anchor = rng.normal(size=rows)
    nullable_noise = rng.normal(loc=0.25 * (seed % 3), size=rows)
    nullable_noise[rng.random(rows) < 0.12] = np.nan
    split_offset = {
        "train": 0,
        "valid": 1_000_000,
        "oos": 2_000_000,
        "oot": 3_000_000,
        "test": 4_000_000,
    }[split]
    application_id = split_offset + np.arange(rows, dtype="int64")
    customer_id = [f"{split}_customer_{index // 3}" for index in range(rows)]
    categories = np.asarray(["A", "B", "C"])[rng.integers(0, 3, rows)]

    return pl.DataFrame(
        {
            "application_id": application_id,
            "customer_id": customer_id,
            "target": target,
            "required_anchor": required_anchor,
            "xor_a": xor_a,
            "xor_b": xor_b,
            "tri_a": tri_a,
            "tri_b": tri_b,
            "tri_c": tri_c,
            "nullable_noise": nullable_noise,
            "category_noise": categories,
            "noise_0": rng.normal(loc=0.15 * (split == "oot"), size=rows),
            "noise_1": rng.normal(size=rows),
        }
    )


def _write_partitioned(frame: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    boundaries = (0, len(frame) // 3, 2 * len(frame) // 3, len(frame))
    for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        frame.slice(start, stop - start).write_parquet(
            directory / f"part-{index:03d}.parquet",
            compression="zstd",
        )


def _write_fixture(root: Path) -> dict[str, Path]:
    specs = {
        "train": (900, 1101, 0.25),
        "valid": (450, 2202, 0.25),
        "oos": (450, 3303, 0.22),
        "oot": (450, 4404, 0.18),
        "test": (450, 5505, 0.16),
    }
    paths: dict[str, Path] = {}
    for split, (rows, seed, keep_probability) in specs.items():
        paths[split] = root / "data" / split
        _write_partitioned(
            _make_split(split, rows, seed, keep_probability),
            paths[split],
        )
    return paths


def _write_config(root: Path, paths: dict[str, Path]) -> Path:
    import yaml

    output = root / "results"
    config = {
        "data": {
            "train_path": str(paths["train"]),
            "valid_path": str(paths["valid"]),
            "oos_path": str(paths["oos"]),
            "oot_path": str(paths["oot"]),
            "test_path": str(paths["test"]),
            "target": "target",
            "positive_label": 1,
            "id_columns": ["application_id", "customer_id"],
            "sampling_key_columns": ["application_id"],
            "categorical_features": ["category_noise"],
            "excluded_features": [],
            "required_features": ["required_anchor"],
            "leakage_key_columns": ["customer_id"],
            "bootstrap_key_columns": ["customer_id"],
        },
        "sample": {
            "screen_train_positive_fraction": 1.0,
            "screen_train_negative_fraction": 1.0,
            "screen_valid_positive_fraction": 1.0,
            "screen_valid_negative_fraction": 1.0,
            "train_positive_fraction": 1.0,
            "train_negative_fraction": 1.0,
            "valid_positive_fraction": 1.0,
            "valid_negative_fraction": 1.0,
            "confirm_train_positive_fraction": 1.0,
            "confirm_train_negative_fraction": 1.0,
            "confirm_valid_positive_fraction": 1.0,
            "confirm_valid_negative_fraction": 1.0,
            "oos_positive_fraction": 1.0,
            "oos_negative_fraction": 1.0,
            "oot_positive_fraction": 1.0,
            "oot_negative_fraction": 1.0,
            "test_positive_fraction": 1.0,
            "test_negative_fraction": 1.0,
            "seed": 73,
        },
        "validation": {
            "check_split_schema": True,
            "check_id_overlap": True,
            "fail_on_id_overlap": True,
        },
        "search": {
            "primary_metric": "average_precision",
            "min_features": 3,
            "max_features": 4,
            "random_subspaces": 20,
            "adaptive_subspaces": 10,
            "subspace_min_features": 3,
            "subspace_max_features": 4,
            "elite_fraction": 0.25,
            "population_size": 10,
            "generations": 3,
            "mutation_rate": 0.80,
            "crossover_rate": 0.80,
            "interaction_pairs": 20,
            "interaction_mutation_probability": 0.70,
            "coalition_top_pairs": 30,
            "coalition_top_triples": 10,
            "coalition_min_support": 1,
            "coalition_smoothing": 1.0,
            "coalition_pairs_per_subset": 100,
            "coalition_triples_per_subset": 30,
            "coalition_sampling_probability": 0.65,
            "coalition_mutation_probability": 0.65,
            "coalition_preservation_probability": 0.80,
            "coalition_refresh_interval": 2,
            "coalition_history_limit": 100,
            "interaction_prior_weight": 1.0,
            "local_candidates": 2,
            "local_rounds": 1,
            "max_gap": 0.50,
            "allowed_metric_drop": 0.01,
            "successive_halving_enabled": True,
            "promotion_fraction": 0.50,
            "min_promoted": 4,
            "promotion_strategy": "adaptive",
            "promotion_min_fraction": 0.30,
            "promotion_max_fraction": 0.60,
            "promotion_metric_band": 0.01,
            "restarts": 1,
            "seed": 91,
        },
        "robust_validation": {
            "enabled": True,
            "require_oos": True,
            "evaluate_oot": True,
            "evaluate_test": True,
            "bootstrap_repeats": 60,
            "confidence_level": 0.90,
            "bootstrap_max_rows": 240,
            "bootstrap_seed": 117,
            "bootstrap_mode": "auto",
            "noninferiority_margin": 0.01,
        },
        "decision_threshold": {
            "enabled": True,
            "min_recall": 0.50,
            "min_precision": 0.20,
            "objective": "max_precision",
            "require_oos_feasible": True,
        },
        "execution": {
            "backend": "local",
            "local_trial_mode": "process",
            "process_start_method": "forkserver",
            "parallel_trials": 0,
            "threads_per_trial": 1,
            "gpu_devices": [],
            "trial_max_retries": 1,
            "trial_failure_policy": "continue",
            "minimum_successful_fraction": 0.75,
            "retry_backoff_seconds": 0.0,
            "trial_timeout_seconds": 60.0,
            "heartbeat_interval_seconds": 0.10,
            "heartbeat_timeout_seconds": 10.0,
            "terminate_grace_seconds": 0.50,
            "show_progress": False,
            "progress_interval_seconds": 1.0,
            "checkpoint_enabled": True,
            "resume_search": True,
        },
        "resources": {
            "enabled": True,
            "mode": "hybrid",
            "cpu_cores": 2,
            "reserve_cpu_cores": 0,
            "max_cpu_utilization": 1.0,
            "ram_gb": 4.0,
            "reserve_ram_gb": 0.25,
            "max_ram_utilization": 0.95,
            "adaptive_concurrency": True,
            "monitoring_interval_seconds": 0.05,
            "resource_wait_timeout_seconds": 30.0,
            "calibration_enabled": True,
            "calibration_safety_factor": 1.2,
            "estimated_ram_per_trial_gb": 0.25,
            "hard_ram_per_trial_gb": 1.5,
            "hard_limit_multiplier": 2.0,
            "profile_learning_enabled": True,
            "oom_concurrency_reduction": 0.5,
        },
        "model_params": {
            "task_type": "CPU",
            "thread_count": 2,
            "depth": 4,
            "learning_rate": 0.12,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.2,
            "screen_iterations": 30,
            "search_iterations": 60,
            "confirm_iterations": 100,
        },
        "output": {
            "directory": str(output),
            "cache_directory": str(output / "cache"),
            "save_trial_predictions": True,
        },
    }
    path = root / "e2e-config.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_outputs(root: Path) -> dict[str, Any]:
    output = root / "results"
    required_outputs = {
        "baseline.json",
        "interactions.json",
        "coalitions.json",
        "trials.jsonl",
        "screening_trials.jsonl",
        "trial_diagnostics.jsonl",
        "promotion_batches.jsonl",
        "pareto_search.json",
        "pareto_confirmed.json",
        "threshold_validation.json",
        "robust_oos.json",
        "oot_final.json",
        "test_final.json",
        "selected_features.json",
        "selection_stability.json",
        "synergy.json",
        "run_manifest.json",
        "search_state.json",
        "resource_plan.json",
        "resource_usage.jsonl",
        "resource_summary.json",
    }
    missing = sorted(name for name in required_outputs if not (output / name).exists())
    assert not missing, f"Missing output files: {missing}"

    selected = _read_json(output / "selected_features.json")
    selected_set = set(selected["features"])
    assert {"required_anchor", "xor_a", "xor_b"} <= selected_set, selected
    assert len(selected_set) <= 4, selected
    assert selected["selection_basis"] == "oos_threshold_constrained_paired_bootstrap_lcb", selected

    threshold = selected["decision_threshold"]
    assert threshold["tuning_split"] == "threshold_valid", threshold
    assert threshold["evaluation_split"] == "oos", threshold
    assert threshold["tuning_feasible"], threshold
    assert threshold["evaluation_feasible"], threshold
    assert threshold["evaluation_recall"] >= 0.50, threshold
    assert threshold["evaluation_precision"] >= 0.20, threshold

    robust = _read_json(output / "robust_oos.json")
    assert robust["bootstrap_unit_used"] == "cluster", robust
    winner_row = next(
        row
        for row in robust["candidates"]
        if set(row["subset"]) == selected_set
    )
    assert winner_row["n_clusters"] == 150, winner_row
    assert winner_row["bootstrap_clusters"] is not None, winner_row
    assert winner_row["eligible_for_selection"], winner_row

    oot = _read_json(output / "oot_final.json")
    assert oot["used_for_selection"] is False, oot
    assert oot["decision_threshold"]["frozen_for_holdout"] is True, oot
    assert oot["decision_threshold"]["threshold"] == threshold["threshold"], oot
    assert set(oot["result"]["subset"]) == selected_set, oot
    test = _read_json(output / "test_final.json")
    assert test["status"] == "evaluated_after_selection", test
    assert test["used_for_selection"] is False, test
    assert test["decision_threshold"]["frozen_for_holdout"] is True, test
    assert test["decision_threshold"]["threshold"] == threshold["threshold"], test
    assert set(test["result"]["subset"]) == selected_set, test

    manifest = _read_json(output / "run_manifest.json")
    assert manifest["selector_version"] == APP_VERSION, manifest
    for package in ("catboost", "polars", "pyarrow"):
        assert manifest["dependency_versions"][package] != "missing", manifest
    assert manifest["config"]["execution"]["parallel_trials"] == 0, manifest
    assert manifest["config"]["execution"]["backend"] == "local", manifest
    assert manifest["config"]["execution"]["local_trial_mode"] == "process", manifest
    resource_plan = _read_json(output / "resource_plan.json")
    resource_summary = _read_json(output / "resource_summary.json")
    assert resource_plan["requested_parallel_trials"] == 0, resource_plan
    assert resource_plan["effective_parallel_limit"] == 2, resource_plan
    assert resource_plan["threads_per_trial"] == 1, resource_plan
    assert resource_plan["calibration"] is not None, resource_plan
    assert resource_summary["max_active_trials"] <= 2, resource_summary
    assert resource_summary["oom_events"] == 0, resource_summary
    assert resource_summary["hard_limit_events"] == 0, resource_summary
    assert resource_summary["timeout_events"] == 0, resource_summary
    assert resource_summary["resource_profiles"]["search"]["observations"] > 0
    assert resource_summary["resource_profiles"]["confirm"]["observations"] > 0
    if resource_summary["process_memory_telemetry_available"]:
        assert resource_summary["resource_profiles"]["search"]["peak_rss_gb"] > 0
    assert (output / "resource_usage.jsonl").read_text().strip(), resource_summary

    screening_trials = sum(
        1 for line in (output / "screening_trials.jsonl").read_text().splitlines()
        if line
    )
    promoted_trials = sum(
        1 for line in (output / "trials.jsonl").read_text().splitlines()
        if line
    )
    assert screening_trials > promoted_trials > 0, (
        screening_trials,
        promoted_trials,
    )
    promotion_batches = [
        json.loads(line)
        for line in (output / "promotion_batches.jsonl").read_text().splitlines()
        if line
    ]
    assert promotion_batches, "Promotion diagnostics are empty"
    assert all(row["strategy"] == "adaptive" for row in promotion_batches)
    assert all(
        0.30 <= row["actual_fraction"] <= 0.60
        or row["promoted"] == 4
        for row in promotion_batches
    ), promotion_batches
    search_state = _read_json(output / "search_state.json")
    assert search_state["status"] == "complete", search_state
    assert search_state["fingerprint"] == manifest["fingerprint"], search_state
    assert search_state["final_front"], search_state
    assert len(search_state["promotion_batches"]) == len(promotion_batches)
    assert search_state["trial_diagnostics"] == []
    interaction_cache_files = list(
        (output / "cache" / "interactions").glob("*.json")
    )
    assert interaction_cache_files, "CatBoost interaction cache is empty"

    return {
        "selector_version": APP_VERSION,
        "selected_features": selected["features"],
        "oos_average_precision": selected["robust_validation"]["point_metric"],
        "threshold": threshold["threshold"],
        "oos_precision": threshold["evaluation_precision"],
        "oos_recall": threshold["evaluation_recall"],
        "oot_precision": oot["decision_threshold"]["precision"],
        "oot_recall": oot["decision_threshold"]["recall"],
        "test_precision": test["decision_threshold"]["precision"],
        "test_recall": test["decision_threshold"]["recall"],
        "bootstrap_unit": robust["bootstrap_unit_used"],
        "oos_clusters": winner_row["n_clusters"],
        "screening_trials": screening_trials,
        "promoted_trials": promoted_trials,
        "promotion_batches": len(promotion_batches),
        "requested_parallel_trials": resource_plan["requested_parallel_trials"],
        "effective_parallel_limit": resource_plan["effective_parallel_limit"],
        "max_active_trials": resource_summary["max_active_trials"],
        "resource_oom_events": resource_summary["oom_events"],
        "checkpoint_status": search_state["status"],
        "interaction_cache_files": len(interaction_cache_files),
    }


def run_e2e(root: str | Path, verify_resume: bool = True) -> dict[str, Any]:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(
            "Install requirements-interaction-selector.txt before the E2E test; "
            f"missing: {missing}"
        )
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"E2E directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    paths = _write_fixture(root)
    config_path = _write_config(root, paths)

    started = time.perf_counter()
    run_pipeline(config_path)
    first_run_seconds = time.perf_counter() - started
    summary = _validate_outputs(root)

    cache = root / "results" / "cache"
    cache_snapshot = {
        path.relative_to(cache).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in cache.rglob("*")
        if path.is_file()
    }
    assert cache_snapshot, "Evaluation cache is empty"

    second_run_seconds = None
    if verify_resume:
        selected_before = _read_json(root / "results" / "selected_features.json")
        started = time.perf_counter()
        run_pipeline(config_path)
        second_run_seconds = time.perf_counter() - started
        selected_after = _read_json(root / "results" / "selected_features.json")
        cache_after = {
            path.relative_to(cache).as_posix(): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in cache.rglob("*")
            if path.is_file()
        }
        assert selected_after == selected_before, "Resume changed the selected result"
        assert cache_after == cache_snapshot, "Resume rewrote or added cached evaluations"
        resource_events = [
            json.loads(line)["event"]
            for line in (root / "results" / "resource_usage.jsonl")
            .read_text()
            .splitlines()
            if line
        ]
        assert "calibration_restored" in resource_events, resource_events

    summary.update(
        {
            "status": "ok",
            "workdir": str(root),
            "first_run_seconds": first_run_seconds,
            "second_run_seconds": second_run_seconds,
            "cache_files": len(cache_snapshot),
            "resume_verified": verify_resume,
        }
    )
    (root / "e2e-report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Parquet splits and run the full selector E2E test"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Empty destination directory; a unique temporary directory is used by default",
    )
    parser.add_argument(
        "--no-resume-check",
        action="store_true",
        help="Skip the second run that verifies cache reuse",
    )
    args = parser.parse_args()
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="interaction-selector-e2e-"))
    result = run_e2e(workdir, verify_resume=not args.no_resume_check)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
