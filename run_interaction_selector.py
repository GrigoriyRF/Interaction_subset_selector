from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import yaml

from interaction_subset_selector import run_pipeline, run_preflight


# =============================================================================
# CONFIG: EDIT ONLY THIS BLOCK
# =============================================================================
PROJECT_ROOT = Path(
    "/home/datalab/nfs/Ипотека/262743_ЦЧБ_Application"
)
SPLITS_ROOT = PROJECT_ROOT / "split_parts"

TRAIN_PATH = SPLITS_ROOT / "train"
VALID_PATH = SPLITS_ROOT / "valid"
# Необязательные независимые выборки. Укажите None, если роли нет.
OOS_PATH: Path | None = SPLITS_ROOT / "oos"
OOT_PATH: Path | None = SPLITS_ROOT / "oot"
TEST_PATH: Path | None = None

TARGET = "del30_4"
POSITIVE_LABEL = 1

ID_COLUMNS = [
    "model_appl_num",
    "model_appl_dt",
    "model_cust_epk_sid",
]
SAMPLING_KEY_COLUMNS = ["model_appl_num"]
LEAKAGE_KEY_COLUMNS = ["model_cust_epk_sid"]
BOOTSTRAP_KEY_COLUMNS = ["model_cust_epk_sid"]

# Файл joblib/pkl со списком ИМЕН категориальных факторов. Если None,
# строковые/dictionary-столбцы будут найдены по схеме Parquet автоматически.
CAT_FEATURES_PATH: Path | None = None
FORCED_CATEGORICAL_FEATURES = [
    "cust_cred_max_ovr_bucket_fifo_3m_code",
]

EXCLUDED_FEATURES: list[str] = []
REQUIRED_FEATURES: list[str] = []

OUTPUT_DIR = PROJECT_ROOT / "interaction_selection_results_v10"
GENERATED_CONFIG = OUTPUT_DIR / "config.run.yaml"

# Сначала всегда строится preflight_plan.json. True останавливает запуск после
# плана ресурсов; False после плана запускает полный подбор комбинаций.
PLAN_ONLY = False

# Ресурсы. None/пустой словарь = автоопределение. Если задать хотя бы один
# лимит, включится hybrid: будет использован минимум из обнаруженного и лимита.
CPU_CORES: int | None = None
RAM_GB: float | None = None
GPU_MEMORY_BY_DEVICE_GB: dict[str, float] = {}
GPU_DEVICES: list[str] = []  # [] = все подходящие карты; пример: ["0", "1"]

TEMPLATE_CONFIG = Path(__file__).with_name(
    "config.interaction-selector.example.yaml"
)


def parquet_schema(path: Path) -> pa.Schema:
    if not path.exists():
        raise FileNotFoundError(f"Не найдена выборка: {path}")
    return ds.dataset(str(path), format="parquet").schema


def load_categorical_features(schema: pa.Schema) -> list[str]:
    categorical = {
        field.name
        for field in schema
        if (
            pa.types.is_string(field.type)
            or pa.types.is_large_string(field.type)
            or pa.types.is_dictionary(field.type)
            or pa.types.is_boolean(field.type)
        )
    }
    if CAT_FEATURES_PATH is not None:
        if not CAT_FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Не найден CAT_FEATURES_PATH: {CAT_FEATURES_PATH}"
            )
        import joblib

        loaded: Any = joblib.load(CAT_FEATURES_PATH)
        if isinstance(loaded, dict):
            for key in (
                "categorical_features",
                "cat_features",
                "CAT_FEATURES",
                "features",
            ):
                if key in loaded:
                    loaded = loaded[key]
                    break
        if not isinstance(loaded, (list, tuple, set)):
            raise TypeError(
                "CAT_FEATURES_PATH должен содержать список имен факторов"
            )
        if any(not isinstance(name, str) for name in loaded):
            raise TypeError(
                "Категориальные факторы должны быть сохранены именами, не индексами"
            )
        categorical.update(loaded)
    categorical.update(FORCED_CATEGORICAL_FEATURES)
    excluded = {TARGET, *ID_COLUMNS, *EXCLUDED_FEATURES}
    return sorted((categorical & set(schema.names)) - excluded)


def build_config() -> dict[str, Any]:
    if not TEMPLATE_CONFIG.exists():
        raise FileNotFoundError(f"Не найден шаблон: {TEMPLATE_CONFIG}")

    train_schema = parquet_schema(TRAIN_PATH)
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["data"].update(
        {
            "train_path": str(TRAIN_PATH),
            "valid_path": str(VALID_PATH),
            "oos_path": str(OOS_PATH) if OOS_PATH is not None else None,
            "oot_path": str(OOT_PATH) if OOT_PATH is not None else None,
            "test_path": str(TEST_PATH) if TEST_PATH is not None else None,
            "target": TARGET,
            "positive_label": POSITIVE_LABEL,
            "id_columns": ID_COLUMNS,
            "sampling_key_columns": SAMPLING_KEY_COLUMNS,
            "categorical_features": load_categorical_features(train_schema),
            "excluded_features": EXCLUDED_FEATURES,
            "required_features": REQUIRED_FEATURES,
            "leakage_key_columns": LEAKAGE_KEY_COLUMNS,
            "bootstrap_key_columns": BOOTSTRAP_KEY_COLUMNS,
        }
    )

    # Утвержденные ограничения и метрика отбора комбинаций.
    config["search"].update(
        {
            "primary_metric": "average_precision",
            "max_features": 100,
            "max_gap": 0.15,
        }
    )
    config["robust_validation"].update(
        {
            "require_oos": OOS_PATH is not None,
            "evaluate_oot": OOT_PATH is not None,
            "evaluate_test": TEST_PATH is not None,
            "bootstrap_repeats": 1000,
        }
    )
    config["decision_threshold"].update(
        {
            "min_recall": 0.50,
            "min_precision": 0.20,
            "objective": "max_precision",
            "require_oos_feasible": OOS_PATH is not None,
        }
    )

    config["execution"].update(
        {
            "backend": "local",
            "local_trial_mode": "process",
            "parallel_trials": 0,
            "threads_per_trial": 0,
            "gpu_devices": GPU_DEVICES,
        }
    )
    manual_limits = (
        CPU_CORES is not None
        or RAM_GB is not None
        or bool(GPU_MEMORY_BY_DEVICE_GB)
    )
    config["resources"].update(
        {
            "mode": "hybrid" if manual_limits else "auto",
            "cpu_cores": CPU_CORES,
            "ram_gb": RAM_GB,
            "gpu_count": None,
            "gpu_total_memory_gb": None,
            "gpu_memory_by_device_gb": GPU_MEMORY_BY_DEVICE_GB,
        }
    )
    config["model_params"].update(
        {
            "task_type": "GPU",
            "eval_metric": "PRAUC:type=Classic;use_weights=true",
            "learning_rate": 0.08844,
            "depth": 8,
            "l2_leaf_reg": 20.075,
            "grow_policy": "Lossguide",
            "auto_class_weights": "Balanced",
        }
    )
    config["output"].update(
        {
            "directory": str(OUTPUT_DIR),
            "cache_directory": str(OUTPUT_DIR / "cache"),
        }
    )
    return config


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config()
    GENERATED_CONFIG.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    plan = run_preflight(GENERATED_CONFIG)
    print(
        "\nПЛАН ЗАПУСКА:",
        f"режим проверки: {plan['validation_mode']}",
        f"выборки: {', '.join(plan['active_splits'])}",
        f"факторов на входе: {plan['input_features']}",
        f"параллельных процессов: {plan['execution']['effective_parallel_limit']}",
        f"потоков на процесс: {plan['execution']['threads_per_trial']}",
        sep="\n  ",
    )
    if plan["warnings"]:
        print("\nЗамечания плана:")
        print("\n".join(f"  - {item}" for item in plan["warnings"]))
    if PLAN_ONLY:
        print(f"\nПлан сохранен: {OUTPUT_DIR / 'preflight_plan.json'}")
        return

    run_pipeline(GENERATED_CONFIG)
    print(f"\nГотово. Результаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
