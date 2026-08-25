import random
import importlib.util
import io
import tempfile
import threading
import time
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from interaction_subset_selector import (
    AppConfig,
    CatBoostSubsetEvaluator,
    CoalitionScore,
    DecisionThresholdConfig,
    DeterministicEvaluator,
    Evaluation,
    EvaluationCache,
    ExecutionConfig,
    GPUResource,
    InteractionAwareSearch,
    ResourceManager,
    ResourceSnapshot,
    SearchConfig,
    SplitPredictions,
    choose_compact,
    learn_coalitions,
    pairwise_jaccard,
    pareto_front,
    robust_paired_bootstrap,
    run_pipeline,
    run_preflight,
    select_decision_threshold,
    synergy_score,
    validation_mode,
    _event,
    _set_event_output,
)


class InteractionSelectorTests(unittest.TestCase):
    def test_categorical_null_contract(self):
        cfg = AppConfig.from_dict(
            {
                "data": {
                    "train_path": "/data/train",
                    "valid_path": "/data/valid",
                    "target": "target",
                    "sampling_key_columns": ["row_id"],
                    "categorical_null_strategy": "fill",
                    "categorical_null_value": "__NA__",
                }
            }
        )
        cfg.validate()
        self.assertEqual(cfg.data.categorical_null_value, "__NA__")
        cfg.data.categorical_null_strategy = "unknown"
        with self.assertRaisesRegex(ValueError, "categorical_null_strategy"):
            cfg.validate()

    @unittest.skipUnless(
        importlib.util.find_spec("polars") is not None,
        "Polars is required for categorical-null model-frame handling",
    )
    def test_categorical_nulls_are_filled_only_in_model_frame(self):
        import polars as pl

        cfg = AppConfig.from_dict(
            {
                "data": {
                    "train_path": "/data/train",
                    "valid_path": "/data/valid",
                    "target": "target",
                    "sampling_key_columns": ["row_id"],
                    "categorical_features": ["category"],
                    "categorical_null_strategy": "fill",
                    "categorical_null_value": "__NA__",
                }
            }
        )
        evaluator = object.__new__(CatBoostSubsetEvaluator)
        evaluator.cfg = cfg
        source = pl.DataFrame(
            {"category": ["A", None], "numeric": [1.0, None]}
        )
        model = evaluator._model_frame(source, ["category", "numeric"])
        self.assertEqual(model["category"].to_list(), ["A", "__NA__"])
        self.assertEqual(model["numeric"].null_count(), 1)
        self.assertEqual(source["category"].null_count(), 1)

    def test_optional_split_roles_select_validation_mode(self):
        base_data = {
            "train_path": "/data/train",
            "valid_path": "/data/valid",
            "target": "target",
            "sampling_key_columns": ["row_id"],
        }
        cases = [
            ({}, "development_only"),
            ({"oos_path": "/data/oos"}, "oos_only"),
            ({"oot_path": "/data/oot"}, "final_holdout_only"),
            ({"test_path": "/data/test"}, "final_holdout_only"),
            (
                {"oos_path": "/data/oos", "oot_path": "/data/oot"},
                "full",
            ),
        ]
        for optional_paths, expected in cases:
            cfg = AppConfig.from_dict(
                {"data": {**base_data, **optional_paths}}
            )
            cfg.validate()
            self.assertEqual(validation_mode(cfg), expected)

    def test_progress_output_is_readable_and_can_be_disabled(self):
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                _set_event_output(False)
                _event("hidden_stage", value=1)
                _set_event_output(True)
                _event("visible_stage", completed=3, total=10)
        finally:
            _set_event_output(True)
        text = output.getvalue()
        self.assertNotIn("HIDDEN STAGE", text)
        self.assertIn("VISIBLE STAGE", text)
        self.assertIn("completed=3", text)
        self.assertIn("total=10", text)

    def test_resource_manager_autosizes_and_throttles_on_live_ram(self):
        config = AppConfig.from_yaml("config.interaction-selector.example.yaml")
        config.model_params["task_type"] = "CPU"
        config.execution.parallel_trials = 0
        config.execution.threads_per_trial = 0
        config.execution.gpu_devices = []
        config.resources.mode = "auto"
        config.resources.reserve_cpu_cores = 2
        config.resources.max_cpu_utilization = 0.75
        config.resources.reserve_ram_gb = 4.0
        config.resources.max_ram_utilization = 0.80
        config.resources.estimated_ram_per_trial_gb = 4.0
        snapshots = [
            ResourceSnapshot(
                time.time(), 16, 8, 16, 10.0,
                64.0, 50.0, 14.0 / 64.0, 0.5, [], "fake",
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            manager = ResourceManager(
                config, directory, snapshot_provider=lambda: snapshots[0]
            )
            self.assertEqual(config.execution.threads_per_trial, 3)
            self.assertEqual(manager.max_parallel_trials, 3)
            snapshots[0] = ResourceSnapshot(
                time.time(), 16, 8, 16, 15.0,
                64.0, 10.0, 54.0 / 64.0, 0.6, [], "fake",
            )
            self.assertEqual(manager.allowed_concurrency(0, 5), 0)
            manager.report_oom()
            self.assertEqual(manager.current_parallel_limit, 1)
            self.assertEqual(manager.oom_events, 1)

    def test_gpu_memory_is_scheduled_per_device_not_as_a_pool(self):
        config = AppConfig.from_yaml("config.interaction-selector.example.yaml")
        config.execution.parallel_trials = 0
        config.execution.threads_per_trial = 0
        config.execution.gpu_devices = []
        config.resources.mode = "auto"
        config.resources.reserve_cpu_cores = 0
        config.resources.max_cpu_utilization = 1.0
        config.resources.reserve_ram_gb = 0.0
        config.resources.max_ram_utilization = 1.0
        config.resources.estimated_ram_per_trial_gb = 1.0
        config.resources.estimated_gpu_memory_per_trial_gb = 2.0
        snapshot = ResourceSnapshot(
            time.time(), 8, 4, 8, 5.0,
            32.0, 24.0, 0.25, 0.5,
            [
                GPUResource("0", 16.0, 3.0, 0.8, "fake"),
                GPUResource("1", 16.0, 12.0, 0.1, "fake"),
            ],
            "fake",
        )
        with tempfile.TemporaryDirectory() as directory:
            manager = ResourceManager(
                config, directory, snapshot_provider=lambda: snapshot
            )
        self.assertEqual(config.execution.gpu_devices, ["1"])
        self.assertEqual(manager.max_parallel_trials, 1)

    def test_dynamic_scheduler_obeys_live_concurrency_limit_and_order(self):
        class FakeManager:
            def __init__(self):
                self.max_parallel_trials = 4
                self.current_parallel_limit = 4
                self.config = SimpleNamespace(
                    monitoring_interval_seconds=0.01,
                    resource_wait_timeout_seconds=1.0,
                )
                self.max_active = 0

            def allowed_concurrency(self, active, pending, event):
                self.max_active = max(self.max_active, active)
                return min(2, self.current_parallel_limit)

            def report_oom(self):
                self.current_parallel_limit = max(
                    1, self.current_parallel_limit // 2
                )

        class ActiveEvaluator:
            primary_metric = "synthetic"

            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def evaluate(self, subset, fidelity):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                selected = tuple(sorted(subset))
                return Evaluation(
                    selected, fidelity, 0.7, 0.7, 0.0, 0.0,
                    len(selected), 0.02, {},
                )

        manager = FakeManager()
        evaluator = ActiveEvaluator()
        search = InteractionAwareSearch(
            [f"f{i}" for i in range(6)],
            evaluator,
            SearchConfig(
                min_features=1,
                max_features=1,
                successive_halving_enabled=False,
            ),
            execution=ExecutionConfig(parallel_trials=4),
            resource_manager=manager,
        )
        subsets = [{f"f{i}"} for i in range(6)]
        result = search._evaluate_many(subsets, "search")
        self.assertEqual(
            [item.subset for item in result],
            [(f"f{i}",) for i in range(6)],
        )
        self.assertEqual(evaluator.max_active, 2)

    def test_oom_retry_reduces_concurrency_and_records_recovery(self):
        class FakeManager:
            def __init__(self):
                self.max_parallel_trials = 4
                self.current_parallel_limit = 4
                self.oom_events = 0
                self.config = SimpleNamespace(
                    monitoring_interval_seconds=0.01,
                    resource_wait_timeout_seconds=1.0,
                )

            def allowed_concurrency(self, active, pending, event):
                return self.current_parallel_limit

            def report_oom(self):
                self.oom_events += 1
                self.current_parallel_limit = max(
                    1, self.current_parallel_limit // 2
                )

        class OOMOnceEvaluator:
            primary_metric = "synthetic"

            def __init__(self):
                self.calls = Counter()
                self.lock = threading.Lock()

            def evaluate(self, subset, fidelity):
                selected = tuple(sorted(subset))
                with self.lock:
                    self.calls[selected] += 1
                    attempt = self.calls[selected]
                if selected == ("a",) and attempt == 1:
                    raise MemoryError("out of memory")
                return Evaluation(
                    selected, fidelity, 0.7, 0.7, 0.0, 0.0,
                    len(selected), 0.0, {},
                )

        manager = FakeManager()
        search = InteractionAwareSearch(
            ["a", "b", "c"],
            OOMOnceEvaluator(),
            SearchConfig(
                min_features=1,
                max_features=1,
                successive_halving_enabled=False,
            ),
            execution=ExecutionConfig(
                parallel_trials=4, trial_max_retries=1
            ),
            resource_manager=manager,
        )
        result = search._evaluate_many([{"a"}, {"b"}, {"c"}], "search")
        self.assertEqual(len(result), 3)
        self.assertEqual(manager.oom_events, 1)
        self.assertEqual(manager.current_parallel_limit, 2)
        recovered = next(
            item for item in search.trial_diagnostics
            if item.subset == ("a",)
        )
        self.assertEqual(recovered.outcome, "recovered")
        self.assertTrue(recovered.oom_detected)

    def test_successive_halving_promotes_only_a_fraction(self):
        class FidelityEvaluator:
            primary_metric = "synthetic"

            def __init__(self):
                self.calls = []

            def evaluate(self, subset, fidelity):
                selected = tuple(sorted(subset))
                self.calls.append((selected, fidelity))
                score = sum(ord(feature[0]) for feature in selected) / 1000
                return Evaluation(
                    selected,
                    fidelity,
                    score,
                    score + 0.01,
                    0.01,
                    0.0,
                    len(selected),
                    0.0,
                    {},
                )

        evaluator = FidelityEvaluator()
        search = InteractionAwareSearch(
            ["a", "b", "c", "d", "e"],
            evaluator,
            SearchConfig(
                min_features=1,
                max_features=2,
                successive_halving_enabled=True,
                promotion_fraction=0.4,
                min_promoted=2,
            ),
        )
        result = search._evaluate_many(
            [{"a"}, {"b"}, {"c"}, {"d"}, {"e"}], "search"
        )
        self.assertEqual(len(search.screening_history), 5)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(search.history), 2)
        self.assertEqual(
            sum(fidelity == "screen" for _, fidelity in evaluator.calls), 5
        )
        self.assertEqual(
            sum(fidelity == "search" for _, fidelity in evaluator.calls), 2
        )

    def test_adaptive_promotion_expands_an_ambiguous_batch(self):
        class RankedEvaluator:
            primary_metric = "synthetic"

            def evaluate(self, subset, fidelity):
                selected = tuple(sorted(subset))
                score = {
                    "a": 1.00,
                    "b": 0.99,
                    "c": 0.98,
                    "d": 0.80,
                    "e": 0.70,
                }[selected[0]]
                return Evaluation(
                    selected,
                    fidelity,
                    score,
                    score,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    {},
                )

        search = InteractionAwareSearch(
            ["a", "b", "c", "d", "e"],
            RankedEvaluator(),
            SearchConfig(
                min_features=1,
                max_features=1,
                successive_halving_enabled=True,
                promotion_strategy="adaptive",
                promotion_min_fraction=0.40,
                promotion_max_fraction=0.80,
                promotion_metric_band=0.025,
                min_promoted=1,
            ),
        )
        promoted = search._evaluate_many(
            [{"a"}, {"b"}, {"c"}, {"d"}, {"e"}], "search"
        )
        self.assertEqual(len(promoted), 3)
        self.assertEqual(search.promotion_batches[0].near_best_candidates, 3)
        self.assertEqual(search.promotion_batches[0].strategy, "adaptive")

    def test_failed_trial_is_retried_and_recovers(self):
        class FlakyEvaluator:
            primary_metric = "synthetic"

            def __init__(self):
                self.calls = {}

            def evaluate(self, subset, fidelity):
                selected = tuple(sorted(subset))
                self.calls[selected] = self.calls.get(selected, 0) + 1
                if self.calls[selected] == 1:
                    raise RuntimeError("transient")
                return Evaluation(
                    selected, fidelity, 0.7, 0.7, 0.0, 0.0,
                    len(selected), 0.0, {},
                )

        search = InteractionAwareSearch(
            ["a", "b", "c"],
            FlakyEvaluator(),
            SearchConfig(
                min_features=1,
                max_features=1,
                successive_halving_enabled=False,
            ),
            execution=ExecutionConfig(
                parallel_trials=2,
                trial_max_retries=1,
            ),
        )
        result = search._evaluate_many([{"a"}, {"b"}, {"c"}], "search")
        self.assertEqual(len(result), 3)
        self.assertEqual(len(search.trial_diagnostics), 3)
        self.assertTrue(
            all(item.outcome == "recovered" for item in search.trial_diagnostics)
        )
        self.assertTrue(
            all(item.attempts == 2 for item in search.trial_diagnostics)
        )

    def test_failed_trial_can_be_quarantined(self):
        class PartlyBrokenEvaluator:
            primary_metric = "synthetic"

            def evaluate(self, subset, fidelity):
                selected = tuple(sorted(subset))
                if "bad" in selected:
                    raise ValueError("invalid candidate")
                return Evaluation(
                    selected, fidelity, 0.7, 0.7, 0.0, 0.0,
                    len(selected), 0.0, {},
                )

        search = InteractionAwareSearch(
            ["a", "b", "c", "bad"],
            PartlyBrokenEvaluator(),
            SearchConfig(
                min_features=1,
                max_features=1,
                successive_halving_enabled=False,
            ),
            execution=ExecutionConfig(
                trial_max_retries=0,
                trial_failure_policy="continue",
                minimum_successful_fraction=0.75,
            ),
        )
        result = search._evaluate_many(
            [{"a"}, {"b"}, {"c"}, {"bad"}], "search"
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(search.trial_diagnostics[0].outcome, "failed")

    def test_parallel_trial_executor_uses_multiple_workers(self):
        class ThreadRecordingEvaluator:
            primary_metric = "synthetic"

            def __init__(self):
                self.thread_ids = set()
                self.lock = threading.Lock()

            def evaluate(self, subset, fidelity):
                with self.lock:
                    self.thread_ids.add(threading.get_ident())
                time.sleep(0.02)
                selected = tuple(sorted(subset))
                return Evaluation(
                    selected,
                    fidelity,
                    0.7,
                    0.71,
                    0.01,
                    0.0,
                    len(selected),
                    0.02,
                    {},
                )

        evaluator = ThreadRecordingEvaluator()
        search = InteractionAwareSearch(
            [f"f{i}" for i in range(8)],
            evaluator,
            SearchConfig(
                min_features=1,
                max_features=1,
                successive_halving_enabled=False,
            ),
            execution=ExecutionConfig(parallel_trials=4),
        )
        search._evaluate_many([{f"f{i}"} for i in range(8)], "search")
        self.assertGreater(len(evaluator.thread_ids), 1)

    def test_generation_checkpoint_resumes_to_same_front(self):
        config = SearchConfig(
            min_features=2,
            max_features=4,
            random_subspaces=10,
            adaptive_subspaces=6,
            subspace_min_features=2,
            subspace_max_features=4,
            population_size=8,
            generations=3,
            local_candidates=2,
            local_rounds=1,
            restarts=1,
            successive_halving_enabled=False,
            seed=19,
        )
        execution = ExecutionConfig(
            checkpoint_enabled=True,
            resume_search=True,
        )
        features = ["xor_a", "xor_b", "main", *[f"noise_{i}" for i in range(5)]]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "search_state.json"
            interrupted = InteractionAwareSearch(
                features,
                DeterministicEvaluator(),
                config,
                checkpoint_path=checkpoint,
                checkpoint_fingerprint="test-fingerprint",
                execution=execution,
            )
            original_save = interrupted._save_checkpoint

            def save_then_interrupt(status, active=None, final_front=None):
                original_save(status, active, final_front)
                if (
                    active
                    and active.get("stage") == "evolve"
                    and active.get("next_generation") == 1
                ):
                    raise RuntimeError("planned interruption")

            interrupted._save_checkpoint = save_then_interrupt
            with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                interrupted.run("search")

            resumed = InteractionAwareSearch(
                features,
                DeterministicEvaluator(),
                config,
                checkpoint_path=checkpoint,
                checkpoint_fingerprint="test-fingerprint",
                execution=execution,
            )
            resumed_front = resumed.run("search")

            uninterrupted = InteractionAwareSearch(
                features,
                DeterministicEvaluator(),
                config,
                execution=ExecutionConfig(checkpoint_enabled=False),
            )
            uninterrupted_front = uninterrupted.run("search")

        self.assertEqual(resumed_front, uninterrupted_front)
        self.assertEqual(
            resumed.restart_winners,
            uninterrupted.restart_winners,
        )

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(package) is not None
            for package in ("catboost", "polars", "pyarrow")
        ),
        "CatBoost/Polars/PyArrow are required for the full E2E test",
    )
    def test_full_parquet_catboost_pipeline(self):
        from e2e_interaction_subset_selector import run_e2e

        with tempfile.TemporaryDirectory() as directory:
            result = run_e2e(directory, verify_resume=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["bootstrap_unit"], "cluster")
        self.assertTrue(result["resume_verified"])

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(package) is not None
            for package in ("catboost", "polars", "pyarrow", "yaml")
        ),
        "CatBoost/Polars/PyArrow/PyYAML are required for process isolation",
    )
    def test_plan_only_and_worker_timeout_do_not_kill_coordinator(self):
        import json
        import yaml

        from e2e_interaction_subset_selector import (
            _write_config,
            _write_fixture,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_config(root, _write_fixture(root))
            preflight = run_preflight(config_path)
            self.assertFalse(preflight["training_started"])
            self.assertEqual(preflight["inventories"]["train"]["rows"], 900)

            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            raw["execution"]["trial_max_retries"] = 0
            raw["execution"]["trial_timeout_seconds"] = 0.05
            raw["execution"]["heartbeat_interval_seconds"] = 0.01
            raw["execution"]["heartbeat_timeout_seconds"] = 1.0
            raw["resources"]["monitoring_interval_seconds"] = 0.01
            raw["resources"]["hard_ram_per_trial_gb"] = 8.0
            config_path.write_text(
                yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "Isolated resource calibration failed"
            ):
                run_pipeline(config_path)
            events = [
                json.loads(line)["event"]
                for line in (root / "results" / "resource_usage.jsonl")
                .read_text()
                .splitlines()
                if line
            ]
        self.assertIn("worker_timeout", events)

    def test_example_config_is_valid(self):
        config = AppConfig.from_yaml("config.interaction-selector.example.yaml")
        config.validate()
        self.assertEqual(config.robust_validation.bootstrap_mode, "auto")
        self.assertEqual(config.data.bootstrap_key_columns, ["customer_id"])
        self.assertTrue(config.decision_threshold.enabled)

    def test_legacy_audit_section_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy 'audit'"):
            AppConfig.from_dict(
                {
                    "data": {
                        "train_path": "/data/train",
                        "valid_path": "/data/valid",
                        "target": "target",
                        "sampling_key_columns": ["row_id"],
                    },
                    "audit": {"check_id_overlap": True},
                }
            )

    def test_parallel_gpu_config_requires_one_device_per_worker(self):
        config = AppConfig.from_yaml("config.interaction-selector.example.yaml")
        config.execution.parallel_trials = 2
        config.execution.gpu_devices = ["0"]
        with self.assertRaisesRegex(ValueError, "one distinct"):
            config.validate()

    def test_ray_gpu_config_uses_ray_resources_not_device_ids(self):
        config = AppConfig.from_yaml("config.interaction-selector.example.yaml")
        config.execution.backend = "ray"
        config.execution.gpu_devices = []
        with self.assertRaisesRegex(ValueError, "ray_num_gpus_per_trial"):
            config.validate()
        config.execution.ray_num_gpus_per_trial = 1.0
        config.execution.gpu_devices = ["0"]
        with self.assertRaisesRegex(ValueError, "leave execution.gpu_devices empty"):
            config.validate()

    def test_ray_backend_contract_preserves_order_and_resources(self):
        from interaction_subset_selector import (
            CatBoostSubsetEvaluator,
            FidelityConfig,
        )

        config = AppConfig.from_yaml("config.interaction-selector.example.yaml")
        config.execution.backend = "ray"
        config.execution.parallel_trials = 2
        config.execution.gpu_devices = []
        config.execution.ray_num_gpus_per_trial = 1.0
        fidelity = FidelityConfig(
            "search", "/tmp/train.parquet", "/tmp/valid.parquet", 10, [42]
        )
        evaluator = CatBoostSubsetEvaluator(
            config,
            [fidelity],
            EvaluationCache("/tmp/selector-ray-contract-cache", "contract"),
        )

        class FakeRemote:
            def __init__(self, owner):
                self.owner = owner

            def options(self, **options):
                self.owner.options = options
                return self

            def remote(self, payload):
                selected = tuple(payload["subset"])
                return {
                    "evaluation": {
                        "subset": selected,
                        "fidelity": payload["fidelity"],
                        "valid_metric": 0.8,
                        "train_metric": 0.81,
                        "gap": 0.01,
                        "metric_std": 0.0,
                        "n_features": len(selected),
                        "runtime_seconds": 0.1,
                        "metrics": {},
                        "constraint_violation": 0.0,
                    },
                    "diagnostic": None,
                }

        class FakeRay:
            def __init__(self):
                self.initialized = False
                self.init_kwargs = None
                self.options = None

            def is_initialized(self):
                return self.initialized

            def init(self, **kwargs):
                self.initialized = True
                self.init_kwargs = kwargs

            def remote(self, function):
                return FakeRemote(self)

            @staticmethod
            def get(reference):
                return reference

        fake_ray = FakeRay()
        resource_manager = SimpleNamespace(
            max_parallel_trials=2,
            current_parallel_limit=2,
            ray_memory_bytes_per_trial=3 * 1024**3,
            report_oom=lambda: None,
        )
        search = InteractionAwareSearch(
            ["a", "b", "c"],
            evaluator,
            SearchConfig(
                min_features=1,
                max_features=1,
                successive_halving_enabled=False,
            ),
            execution=config.execution,
            resource_manager=resource_manager,
        )
        subsets = [frozenset({"a"}), frozenset({"b"}), frozenset({"c"})]
        with patch("interaction_subset_selector._require", return_value=fake_ray):
            result = search._evaluate_many_raw(subsets, "search")
        self.assertEqual([item.subset for item in result], [("a",), ("b",), ("c",)])
        self.assertEqual(fake_ray.options["num_gpus"], 1.0)
        self.assertEqual(fake_ray.options["num_cpus"], 1.0)
        self.assertEqual(fake_ray.options["memory"], 3 * 1024**3)
        self.assertEqual(fake_ray.init_kwargs["namespace"], "interaction-selector")

    def test_successive_halving_iterations_are_ordered(self):
        config = AppConfig.from_yaml("config.interaction-selector.example.yaml")
        config.model_params["screen_iterations"] = 500
        config.model_params["search_iterations"] = 100
        with self.assertRaisesRegex(ValueError, "screen_iterations"):
            config.validate()

    def test_interaction_pair_is_retained(self):
        features = ["xor_a", "xor_b", "main", *[f"noise_{i}" for i in range(17)]]
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
        winner = choose_compact(search.run("search"), allowed_drop=0.001)
        self.assertTrue({"xor_a", "xor_b"} <= set(winner.subset))

    def test_required_features_and_global_size_limit_are_enforced(self):
        features = ["main", *[f"noise_{i}" for i in range(8)]]
        config = SearchConfig(
            min_features=2,
            max_features=3,
            random_subspaces=8,
            adaptive_subspaces=4,
            subspace_min_features=4,
            subspace_max_features=8,
            population_size=6,
            generations=2,
            local_candidates=2,
            local_rounds=1,
            restarts=1,
            seed=11,
        )
        search = InteractionAwareSearch(
            features,
            DeterministicEvaluator(),
            config,
            required_features=["main"],
        )
        search.run("search")
        self.assertTrue(search.history)
        self.assertTrue(all("main" in item.subset for item in search.history))
        self.assertTrue(all(item.n_features <= 3 for item in search.history))

    def test_successful_pair_is_learned_as_a_coalition(self):
        rows = [
            (("a", "b", "x"), 0.91),
            (("a", "b", "y"), 0.89),
            (("a", "c", "x"), 0.56),
            (("b", "c", "y"), 0.55),
            (("c", "d", "x"), 0.53),
            (("c", "d", "y"), 0.52),
        ]
        evaluations = [
            Evaluation(
                subset,
                "search",
                metric,
                metric + 0.01,
                0.01,
                0.0,
                len(subset),
                0.0,
                {},
            )
            for subset, metric in rows
        ]
        config = SearchConfig(
            min_features=2,
            max_features=3,
            coalition_top_pairs=10,
            coalition_top_triples=0,
            coalition_min_support=2,
            elite_fraction=1 / 3,
        )
        coalitions = learn_coalitions(evaluations, config, seed=3)
        self.assertTrue(coalitions)
        self.assertEqual(coalitions[0].members, ("a", "b"))
        self.assertGreater(coalitions[0].lift, 0)

    def test_successful_triple_is_learned_as_a_coalition(self):
        rows = [
            (("a", "b", "c", "x"), 0.95),
            (("a", "b", "c", "y"), 0.94),
            (("a", "b", "x", "z"), 0.56),
            (("a", "c", "y", "z"), 0.55),
            (("b", "c", "x", "y"), 0.54),
            (("d", "e", "x", "z"), 0.51),
        ]
        evaluations = [
            Evaluation(
                subset,
                "search",
                metric,
                metric + 0.01,
                0.01,
                0.0,
                len(subset),
                0.0,
                {},
            )
            for subset, metric in rows
        ]
        config = SearchConfig(
            min_features=3,
            max_features=4,
            coalition_top_pairs=0,
            coalition_top_triples=10,
            coalition_min_support=2,
            coalition_triples_per_subset=100,
            elite_fraction=1 / 3,
        )
        coalitions = learn_coalitions(evaluations, config, seed=3)
        triple = next(item for item in coalitions if item.members == ("a", "b", "c"))
        self.assertEqual(triple.order, 3)
        self.assertGreater(triple.score, 0)

    def test_adaptive_sampling_preserves_a_seed_coalition(self):
        features = ["a", "b", "c", "d"]
        config = SearchConfig(
            min_features=2,
            max_features=2,
            subspace_min_features=2,
            subspace_max_features=2,
            coalition_sampling_probability=1.0,
        )
        search = InteractionAwareSearch(features, DeterministicEvaluator(), config)
        search.coalitions = [
            CoalitionScore(
                members=("a", "b"),
                order=2,
                support=5,
                elite_support=4,
                lift=1.0,
                mean_metric=0.9,
                metric_gain=0.2,
                score=2.0,
            )
        ]
        subset = search._random_subset(
            random.Random(5), weights={feature: 1.0 for feature in features}
        )
        self.assertEqual(subset, frozenset({"a", "b"}))

    def test_paired_bootstrap_selects_compact_noninferior_subset(self):
        y_true = np.asarray([0, 1] * 100, dtype="int8")
        weight = np.ones(len(y_true), dtype="float64")
        good_prediction = np.where(y_true == 1, 0.9, 0.1)
        bad_prediction = np.linspace(0.0, 1.0, len(y_true))
        predictions = {
            ("0_full", "b", "c", "d"): SplitPredictions(
                "oos", y_true, good_prediction.copy(), weight, [1.0, 1.0, 1.0]
            ),
            ("a", "b", "c"): SplitPredictions(
                "oos", y_true, good_prediction, weight, [1.0, 1.0, 1.0]
            ),
            ("a",): SplitPredictions(
                "oos", y_true, good_prediction.copy(), weight, [1.0, 1.0, 1.0]
            ),
            ("noise",): SplitPredictions(
                "oos", y_true, bad_prediction, weight, [0.5, 0.5, 0.5]
            ),
        }
        rows, winner = robust_paired_bootstrap(
            predictions,
            metric_name="average_precision",
            confidence_level=0.95,
            repeats=100,
            max_rows=100,
            seed=13,
            noninferiority_margin=0.001,
            eligible_subsets={("a", "b", "c"), ("a",), ("noise",)},
        )
        self.assertEqual(winner, ("a",))
        compact = next(item for item in rows if item.subset == ("a",))
        noisy = next(item for item in rows if item.subset == ("noise",))
        full = next(item for item in rows if item.subset[0] == "0_full")
        self.assertTrue(compact.noninferior)
        self.assertFalse(noisy.noninferior)
        self.assertFalse(full.eligible_for_selection)
        self.assertEqual(compact.bootstrap_rows, 100)

    def test_cluster_bootstrap_resamples_whole_entities(self):
        groups = np.repeat(np.arange(40), 4)
        y_true = np.repeat(np.asarray([0] * 20 + [1] * 20, dtype="int8"), 4)
        weight = np.ones(len(y_true), dtype="float64")
        good_prediction = np.where(y_true == 1, 0.9, 0.1)
        bad_prediction = np.tile([0.8, 0.2, 0.7, 0.3], 40)
        predictions = {
            ("a",): SplitPredictions(
                "oos", y_true, good_prediction, weight, [1.0], groups
            ),
            ("noise",): SplitPredictions(
                "oos", y_true, bad_prediction, weight, [0.5], groups
            ),
        }
        rows, winner = robust_paired_bootstrap(
            predictions,
            metric_name="average_precision",
            confidence_level=0.95,
            repeats=60,
            max_rows=80,
            seed=31,
            noninferiority_margin=0.001,
            bootstrap_mode="auto",
        )
        self.assertEqual(winner, ("a",))
        self.assertTrue(all(item.bootstrap_unit == "cluster" for item in rows))
        self.assertTrue(all(item.n_clusters == 40 for item in rows))
        self.assertTrue(all(item.bootstrap_clusters == 20 for item in rows))

    def test_threshold_is_selected_on_validation_and_frozen_on_oos(self):
        tuning = SplitPredictions(
            "threshold_valid",
            np.asarray([0, 0, 0, 1, 1, 1], dtype="int8"),
            np.asarray([0.1, 0.2, 0.4, 0.35, 0.8, 0.9]),
            np.ones(6),
            [1.0],
        )
        oos = SplitPredictions(
            "oos",
            np.asarray([0, 0, 1, 1], dtype="int8"),
            np.asarray([0.1, 0.6, 0.75, 0.9]),
            np.ones(4),
            [1.0],
        )
        result = select_decision_threshold(
            ("a", "b"),
            tuning,
            oos,
            DecisionThresholdConfig(
                enabled=True,
                min_recall=0.5,
                min_precision=0.8,
                objective="max_precision",
            ),
        )
        self.assertAlmostEqual(result.threshold, 0.8)
        self.assertTrue(result.tuning_feasible)
        self.assertTrue(result.evaluation_feasible)
        self.assertAlmostEqual(result.evaluation_precision, 1.0)
        self.assertAlmostEqual(result.evaluation_recall, 0.5)

        without_oos = select_decision_threshold(
            ("a", "b"),
            tuning,
            None,
            DecisionThresholdConfig(
                enabled=True,
                min_recall=0.5,
                min_precision=0.8,
                objective="max_precision",
            ),
        )
        self.assertTrue(without_oos.tuning_feasible)
        self.assertIsNone(without_oos.evaluation_split)
        self.assertIsNone(without_oos.evaluation_feasible)

    def test_synergy_is_conditional_gain(self):
        evaluator = DeterministicEvaluator()
        result = synergy_score(
            evaluator,
            ["xor_a", "xor_b", "main"],
            "xor_a",
            "xor_b",
            "search",
        )
        self.assertAlmostEqual(result["synergy"], 0.35)

    def test_pareto_keeps_quality_and_compact_solutions(self):
        evaluations = [
            Evaluation(("a",), "x", 0.70, 0.71, 0.01, 0.0, 1, 0.0, {}),
            Evaluation(("a", "b"), "x", 0.80, 0.81, 0.01, 0.0, 2, 0.0, {}),
            Evaluation(("a", "b", "c"), "x", 0.79, 0.80, 0.01, 0.0, 3, 0.0, {}),
        ]
        front = pareto_front(evaluations)
        self.assertEqual({item.subset for item in front}, {("a",), ("a", "b")})

    def test_cache_roundtrip(self):
        evaluation = Evaluation(("a", "b"), "search", 0.8, 0.82, 0.02, 0.0, 2, 1.0, {})
        with tempfile.TemporaryDirectory() as directory:
            cache = EvaluationCache(Path(directory), "fingerprint")
            cache.put(evaluation)
            loaded = cache.get(evaluation.subset, evaluation.fidelity)
        self.assertEqual(loaded, evaluation)

    def test_jaccard(self):
        self.assertAlmostEqual(pairwise_jaccard([["a", "b"], ["a", "c"]]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
