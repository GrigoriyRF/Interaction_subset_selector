from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interaction_subset_selector import AppConfig
from run_stability_selection import StabilityConfig, StabilityRunner, aggregate_subsets


class AggregationTests(unittest.TestCase):
    def test_feature_pair_and_jaccard_aggregation(self):
        result = aggregate_subsets(
            [
                ["a", "b", "c"],
                ["a", "b", "d"],
                ["a", "b", "e"],
            ],
            candidate_min_frequency=0.34,
            core_min_frequency=0.67,
            pair_min_frequency=0.67,
            max_candidate_features=10,
            max_reported_pairs=10,
        )
        self.assertEqual(result["candidate_pool"], ["a", "b"])
        self.assertEqual(result["stable_core"], ["a", "b"])
        self.assertEqual(
            result["stable_pairs"],
            [{"first": "a", "second": "b", "count": 3, "frequency": 1.0}],
        )
        self.assertAlmostEqual(result["pairwise_jaccard"]["mean"], 0.5)

    def test_always_keep_bypasses_frequency_threshold(self):
        result = aggregate_subsets(
            [["a"], ["a"], ["b"]],
            candidate_min_frequency=0.66,
            core_min_frequency=1.0,
            pair_min_frequency=1.0,
            max_candidate_features=10,
            max_reported_pairs=10,
            always_keep=["required_anchor"],
        )
        self.assertEqual(result["candidate_pool"], ["a", "required_anchor"])
        self.assertEqual(result["stable_core"], [])

    def test_end_to_end_orchestration_with_fake_selector(self):
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selector = root / "fake_selector.py"
            selector.write_text(
                """
import argparse, json
from pathlib import Path
import yaml
p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
cfg = yaml.safe_load(Path(p.parse_args().config).read_text())
output = Path(cfg['output']['directory'])
output.mkdir(parents=True, exist_ok=True)
name = output.name
features = ['a', 'b'] if name.endswith('00') else ['a', 'c']
(output / 'selected_features.json').write_text(json.dumps({
    'features': features, 'valid_metric': 0.7, 'gap': 0.02
}))
""",
                encoding="utf-8",
            )
            template = root / "template.yaml"
            template.write_text(
                yaml.safe_dump(
                    {
                        "data": {},
                        "search": {"min_features": 1, "max_features": 5},
                        "validation": {},
                        "robust_validation": {},
                        "decision_threshold": {},
                        "output": {},
                    }
                ),
                encoding="utf-8",
            )
            data = root / "data.yaml"
            data.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "train_path": "base/train",
                            "valid_path": "base/valid",
                            "oos_path": "base/oos",
                            "oot_path": "base/oot",
                            "test_path": "base/test",
                            "target": "target",
                            "sampling_key_columns": ["id"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "folds.json"
            manifest.write_text(
                json.dumps(
                    {
                        "strategy": "group",
                        "folds": [
                            {"name": "repeat_00_fold_00", "train_path": "f0/train", "valid_path": "f0/valid"},
                            {"name": "repeat_00_fold_01", "train_path": "f1/train", "valid_path": "f1/valid"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "stability"
            cfg = StabilityConfig(
                selector_script=str(selector),
                selector_template_config=str(template),
                selector_data_config=str(data),
                folds_manifest=str(manifest),
                output_directory=str(output),
                candidate_min_frequency=0.5,
                core_min_frequency=1.0,
                pair_min_frequency=0.5,
                max_candidate_features=10,
                max_reported_pairs=10,
                restrict_final_universe=False,
            )
            result = StabilityRunner(cfg).run()
            self.assertEqual(result["stable_core"], ["a"])
            self.assertEqual(result["candidate_pool"], ["a", "b", "c"])
            final = yaml.safe_load((output / "config.final.yaml").read_text())
            self.assertEqual(final["data"]["oos_path"], "base/oos")
            self.assertEqual(final["data"]["test_path"], "base/test")
            fold_config = yaml.safe_load(
                (output / "fold_configs" / "repeat_00_fold_00.yaml").read_text()
            )
            self.assertNotIn("oos_path", fold_config["data"])
            self.assertNotIn("oot_path", fold_config["data"])
            self.assertNotIn("test_path", fold_config["data"])
            self.assertFalse(fold_config["robust_validation"]["enabled"])
            self.assertFalse(fold_config["robust_validation"]["evaluate_test"])
            self.assertFalse(fold_config["decision_threshold"]["enabled"])
            self.assertTrue(fold_config["validation"]["check_id_overlap"])
            effective = AppConfig.from_dict(fold_config)
            self.assertTrue(effective.validation.check_id_overlap)
            self.assertIsNone(effective.data.test_path)

    def test_fold_overrides_cannot_reenable_contract_sections(self):
        cfg = StabilityConfig(
            selector_script=__file__,
            selector_template_config=__file__,
            selector_data_config=__file__,
            folds_manifest=__file__,
            fold_overrides={"robust_validation": {"enabled": True}},
        )
        with self.assertRaisesRegex(ValueError, "cannot change selector contract"):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
