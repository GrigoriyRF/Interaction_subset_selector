from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from prepare_selector_samples import (
    Config,
    FoldsConfig,
    InputConfig,
    OutputConfig,
    SampleBuilder,
    SplitConfig,
)


def config(**split_overrides):
    values = {
        "mode": "auto",
        "train_fraction": 0.70,
        "valid_fraction": 0.15,
        "oos_fraction": 0.15,
        "oot_fraction": 0.0,
        "test_fraction": 0.0,
    }
    values.update(split_overrides)
    return Config(
        input=InputConfig(
            path="input.parquet",
            target="target",
            sampling_key_columns=["row_id"],
            leakage_key_columns=["customer_id"],
        ),
        split=SplitConfig(**values),
        output=OutputConfig(directory="unused"),
    )


class ConfigTests(unittest.TestCase):
    def test_categorical_null_policy_is_forwarded_to_selector_data(self):
        cfg = Config(
            input=InputConfig(
                path="/data/source",
                target="target",
                sampling_key_columns=["row_id"],
                categorical_null_strategy="fill",
                categorical_null_value="__NA__",
            )
        )
        cfg.validate()
        with tempfile.TemporaryDirectory() as directory:
            builder = SampleBuilder.__new__(SampleBuilder)
            builder.cfg = cfg
            builder.root = Path(directory)
            builder.schema = {"target": "Int64", "row_id": "Int64"}
            output = builder._write_selector_config(
                {"train": "/splits/train", "valid": "/splits/valid"}
            )
            import yaml

            selector_data = yaml.safe_load(output.read_text(encoding="utf-8"))["data"]

        self.assertEqual(selector_data["categorical_null_strategy"], "fill")
        self.assertEqual(selector_data["categorical_null_value"], "__NA__")

    def test_auto_without_time_uses_group_from_leakage_keys(self):
        cfg = config()
        self.assertEqual(cfg.group_columns, ["customer_id"])
        self.assertEqual(cfg.downstream_leakage_keys, ["customer_id"])
        self.assertEqual(cfg.resolved_mode(), "group")
        cfg.validate()

    def test_explicit_group_is_protected_downstream(self):
        cfg = config(group_columns=["household_id"])
        self.assertEqual(
            cfg.downstream_leakage_keys,
            ["customer_id", "household_id"],
        )
        cfg.validate()

    def test_auto_time_and_group_uses_group_temporal(self):
        cfg = config(
            train_fraction=0.55,
            time_column="event_date",
            oot_fraction=0.15,
        )
        self.assertEqual(cfg.resolved_mode(), "group_temporal")
        cfg.validate()

    def test_temporal_test_can_be_the_only_final_holdout(self):
        cfg = config(
            train_fraction=0.55,
            time_column="event_date",
            test_fraction=0.15,
        )
        self.assertEqual(cfg.resolved_mode(), "group_temporal")
        cfg.validate()

    def test_temporal_requires_oot(self):
        cfg = config(mode="temporal", time_column="event_date")
        with self.assertRaisesRegex(ValueError, "oot_fraction"):
            cfg.validate()

    def test_random_oot_is_rejected(self):
        cfg = config(
            mode="random",
            train_fraction=0.55,
            oot_fraction=0.15,
        )
        with self.assertRaisesRegex(ValueError, "random/group OOT"):
            cfg.validate()

    def test_group_test_holdout_is_allowed(self):
        cfg = config(
            train_fraction=0.55,
            test_fraction=0.15,
        )
        self.assertEqual(cfg.resolved_mode(), "group")
        cfg.validate()

    def test_sampling_key_is_mandatory(self):
        cfg = config()
        cfg.input.sampling_key_columns = []
        with self.assertRaisesRegex(ValueError, "sampling_key_columns"):
            cfg.validate()

    def test_yaml_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                """
input:
  path: source.parquet
  target: y
  sampling_key_columns: [id]
split:
  mode: stratified
  train_fraction: 0.7
  valid_fraction: 0.15
  oos_fraction: 0.15
  oot_fraction: 0.0
  test_fraction: 0.0
""",
                encoding="utf-8",
            )
            cfg = Config.from_yaml(path)
            self.assertEqual(cfg.input.target, "y")
            self.assertEqual(cfg.resolved_mode(), "stratified")
            cfg.validate()

    def test_unknown_top_level_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                """
input:
  path: source.parquet
  target: y
  sampling_key_columns: [id]
unexpected: {}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown sample-splitting"):
                Config.from_yaml(path)

    def test_nested_input_and_output_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "raw"
            source.mkdir()
            cfg = config()
            cfg.input.path = str(source)
            cfg.output.directory = str(source / "split_parts")
            with self.assertRaisesRegex(ValueError, "must not contain"):
                cfg.validate()

    def test_group_folds_are_selected_for_group_split(self):
        cfg = config()
        cfg.folds = FoldsConfig(enabled=True, n_folds=3, repeats=2)
        self.assertEqual(cfg.resolved_fold_strategy(), "group")
        cfg.validate()

    def test_temporal_folds_reject_repeats(self):
        cfg = config(
            train_fraction=0.55,
            time_column="event_date",
            oot_fraction=0.15,
        )
        cfg.folds = FoldsConfig(enabled=True, n_folds=3, repeats=2)
        with self.assertRaisesRegex(ValueError, "repeats=1"):
            cfg.validate()

    @unittest.skipUnless(
        importlib.util.find_spec("polars") is not None,
        "Polars is required for the split integration test",
    )
    def test_test_holdout_is_written_but_excluded_from_folds(self):
        import polars as pl
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.parquet"
            rows = 240
            pl.DataFrame(
                {
                    "row_id": list(range(rows)),
                    "customer_id": [index // 2 for index in range(rows)],
                    "target": [index % 2 for index in range(rows)],
                    "feature": [float(index) for index in range(rows)],
                }
            ).write_parquet(source)
            cfg = Config(
                input=InputConfig(
                    path=str(source),
                    target="target",
                    sampling_key_columns=["row_id"],
                    leakage_key_columns=["customer_id"],
                ),
                split=SplitConfig(
                    mode="group",
                    train_fraction=0.50,
                    valid_fraction=0.20,
                    oos_fraction=0.15,
                    oot_fraction=0.0,
                    test_fraction=0.15,
                ),
                output=OutputConfig(directory=str(root / "splits")),
                folds=FoldsConfig(enabled=True, n_folds=3),
            )
            manifest = SampleBuilder(cfg).build()
            selector_data = yaml.safe_load(
                Path(manifest["selector_config"]).read_text(encoding="utf-8")
            )["data"]
            folds = json.loads(
                Path(manifest["folds_manifest"]).read_text(encoding="utf-8")
            )
            self.assertIn("test_path", selector_data)
            self.assertEqual(folds["excluded_holdouts"], ["oos", "test"])
            self.assertTrue(
                all(
                    "test" not in item["train_path"]
                    and "test" not in item["valid_path"]
                    for item in folds["folds"]
                )
            )


if __name__ == "__main__":
    unittest.main()
