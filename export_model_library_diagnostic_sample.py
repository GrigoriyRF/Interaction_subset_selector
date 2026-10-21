# -*- coding: utf-8 -*-
"""
Диагностическая выгрузка связанного сэмпла из витрины Библиотеки моделей.

Цель: проверить фактические связи и заполнение полей, используемых в аудите карточек версий моделей.
Выгружаются НЕ агрегированные исходные строки для одних и тех же MODEL_VER_SID,
включая техническую историю (CTL_ACTION / START_DT / END_DT / CTL_DATECHANGE).

Результат: model_library_diagnostic_sample.xlsx
"""

from pathlib import Path
from functools import reduce

import pandas as pd
from pyspark.sql import functions as F, Window
from pyspark.sql.types import ArrayType, MapType, StructType

# -----------------------------------------------------------------------------
# 1. НАСТРОЙКИ
# -----------------------------------------------------------------------------

SOURCE_DB = "prx_pri_custom_ris_l_library_custom_risk_model_library"
OUTPUT_FILE = Path.cwd().resolve() / "model_library_diagnostic_sample.xlsx"

# Ориентир 40–60 версий обычно достаточно. Итог может быть меньше/больше из-за union групп.
N_PER_CASE = 6
N_PER_SIGNIFICANCE = 3
N_RANDOM = 10
MAX_VERSIONS = 70
MAX_ROWS_PER_SHEET = 100_000

TABLES = {
    "model": f"{SOURCE_DB}.t_model",
    "model_ver": f"{SOURCE_DB}.t_model_ver",
    "model_ver_anlt_dtl": f"{SOURCE_DB}.t_model_ver_anlt_dtl",
    "valid": f"{SOURCE_DB}.t_valid",
    "model_ver_x_montrg_manual": f"{SOURCE_DB}.t_model_ver_x_montrg_manual",
    "montrg_manual": f"{SOURCE_DB}.t_montrg_manual",
    "montrg_manual_rslt": f"{SOURCE_DB}.t_montrg_manual_rslt",
    "model_ver_x_montrg_auto": f"{SOURCE_DB}.t_model_ver_x_montrg_auto",
    "montrg_auto": f"{SOURCE_DB}.t_montrg_auto",
    "montrg_auto_rslt": f"{SOURCE_DB}.t_montrg_auto_rslt",
    "sample_data": f"{SOURCE_DB}.t_sample_data",
    "change_log": f"{SOURCE_DB}.t_ent_param_chg",
}

for name, table in TABLES.items():
    if not spark.catalog.tableExists(table):
        raise RuntimeError(f"Не найдена таблица {name}: {table}")

# -----------------------------------------------------------------------------
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------------------------------------------------------

def present_col(name):
    c = F.col(name)
    s = F.lower(F.trim(c.cast("string")))
    return c.isNotNull() & (~s.isin("", "null", "none", "nan", "[]", "{}", "$not_applicable$"))


def truthy_col(name):
    return F.lower(F.trim(F.col(name).cast("string"))).isin("true", "1", "yes", "да")


def existing(df, cols):
    """Оставляет только реально существующие в таблице колонки."""
    available = set(df.columns)
    return [c for c in cols if c in available]


def select_existing(df, cols):
    cols2 = existing(df, cols)
    return df.select(*cols2) if cols2 else df.limit(0)


def non_deleted(df):
    if "ctl_action" not in df.columns:
        return df
    return df.filter(
        F.col("ctl_action").isNull()
        | (F.upper(F.trim(F.col("ctl_action").cast("string"))) != "D")
    )


def current_slice(df, keys):
    """
    Только для ОТБОРА сэмпла, не для выгрузки исходных строк.
    1) ограничиваем действующим бизнес-интервалом на текущую дату;
    2) выбираем последнее состояние по START_DT + CTL_DATECHANGE;
    3) если последнее состояние D — сущность исключается.
    """
    x = df
    today = F.current_date()

    if "start_dt" in x.columns:
        x = x.filter(F.col("start_dt").isNull() | (F.to_date("start_dt") <= today))
    if "end_dt" in x.columns:
        x = x.filter(F.col("end_dt").isNull() | (F.to_date("end_dt") >= today))

    order = []
    if "start_dt" in x.columns:
        order.append(F.col("start_dt").desc_nulls_last())
    if "ctl_datechange" in x.columns:
        order.append(F.col("ctl_datechange").desc_nulls_last())
    if not order:
        raise RuntimeError("Для current_slice нет START_DT/CTL_DATECHANGE")

    w = Window.partitionBy(*keys).orderBy(*order)
    x = x.withColumn("__rn", F.row_number().over(w)).filter("__rn = 1").drop("__rn")

    if "ctl_action" in x.columns:
        x = x.filter(
            F.col("ctl_action").isNull()
            | (F.upper(F.trim(F.col("ctl_action").cast("string"))) != "D")
        )
    return x


def flag_from_presence(df, group_col, source_col, alias):
    if source_col not in df.columns:
        return df.select(F.col(group_col).cast("string").alias(group_col)).distinct().withColumn(alias, F.lit(0))
    return (
        df.withColumn(group_col, F.col(group_col).cast("string"))
          .groupBy(group_col)
          .agg(F.max(F.when(present_col(source_col), 1).otherwise(0)).alias(alias))
    )


def sample_condition(df, condition, label, n=N_PER_CASE):
    return (
        df.filter(condition)
          .orderBy(F.rand(seed=17))
          .limit(n)
          .withColumn("sample_reason", F.lit(label))
    )


def stringify_complex_for_excel(df):
    exprs = []
    for field in df.schema.fields:
        c = F.col(field.name)
        if isinstance(field.dataType, (ArrayType, MapType, StructType)):
            exprs.append(F.to_json(c).alias(field.name))
        else:
            exprs.append(c)
    return df.select(*exprs)


def to_pandas_limited(df, sheet_name):
    pdf = stringify_complex_for_excel(df).limit(MAX_ROWS_PER_SHEET + 1).toPandas()
    if len(pdf) > MAX_ROWS_PER_SHEET:
        raise RuntimeError(
            f"Лист {sheet_name}: больше {MAX_ROWS_PER_SHEET} строк. "
            "Уменьшите MAX_VERSIONS/N_PER_CASE или выгружайте этот лист в parquet."
        )
    return pdf


# -----------------------------------------------------------------------------
# 3. ЧТЕНИЕ ТАБЛИЦ
# -----------------------------------------------------------------------------

raw = {name: spark.table(table) for name, table in TABLES.items()}

# -----------------------------------------------------------------------------
# 4. ФОРМИРОВАНИЕ КАНДИДАТОВ ДЛЯ СЭМПЛА
# -----------------------------------------------------------------------------

# Текущая версия и текущая аналитическая детализация нужны только для осмысленного отбора.
ver_cur = current_slice(raw["model_ver"], ["model_ver_sid"])
anlt_cur = current_slice(raw["model_ver_anlt_dtl"], ["model_ver_sid"])

ver_cols = [
    "model_sid", "model_ver_sid", "model_ver_num", "model_ver_name",
    "model_ver_montrg_last_rslt_name", "model_ver_montrg_last_rslt_dttm", "model_ver_montrg_last_sid",
    "model_ver_montrg_auto_exclude_reason_array",
    "model_ver_repstry_link_txt", "model_ver_mlstorage_branch_link_txt",
    "model_ver_implm_repstry_link_txt", "model_ver_prom_repstry_commit_link_txt",
    "model_ver_signfcnt_ctgry_code", "model_ver_signfcnt_lvl_name",
]

anlt_cols = [
    "model_ver_sid", "model_ver_prom_expl_flag", "model_ver_not_valid_flag",
    "model_ver_signfcnt_ctgry_code", "model_ver_signfcnt_lvl_name",
    "model_ver_prom_actual_implm_first_dttm", "model_ver_prom_implm_first_dttm",
]

v = select_existing(ver_cur, ver_cols).withColumn("model_ver_sid", F.col("model_ver_sid").cast("string"))
a = select_existing(anlt_cur, anlt_cols).withColumn("model_ver_sid", F.col("model_ver_sid").cast("string"))

candidates = v.alias("v").join(a.alias("a"), "model_ver_sid", "inner")

# Периметр: версии в эксплуатации.
if "model_ver_prom_expl_flag" in candidates.columns:
    candidates = candidates.filter(truthy_col("model_ver_prom_expl_flag"))

# Валидации: наличие любого фактического результата в сырых не-D строках.
valid_nd = non_deleted(raw["valid"])
valid_flags = flag_from_presence(valid_nd, "model_ver_sid", "valid_rslt_name", "has_valid_result")
candidates = candidates.join(valid_flags, "model_ver_sid", "left").fillna({"has_valid_result": 0})

# Manual monitoring: версия -> мониторинг -> результат.
manual_links_nd = non_deleted(raw["model_ver_x_montrg_manual"])
manual_results_nd = non_deleted(raw["montrg_manual_rslt"])
manual_join = (
    manual_links_nd
    .select(
        F.col("model_ver_sid").cast("string").alias("model_ver_sid"),
        F.col("montrg_manual_sid").cast("string").alias("montrg_manual_sid"),
    )
    .join(
        manual_results_nd.select(
            F.col("montrg_manual_sid").cast("string").alias("montrg_manual_sid"),
            *([F.col("montrg_manual_rslt_name")] if "montrg_manual_rslt_name" in manual_results_nd.columns else []),
        ),
        "montrg_manual_sid", "left"
    )
)
manual_flags = flag_from_presence(manual_join, "model_ver_sid", "montrg_manual_rslt_name", "has_manual_result")
candidates = candidates.join(manual_flags, "model_ver_sid", "left").fillna({"has_manual_result": 0})

# Auto monitoring.
auto_links_nd = non_deleted(raw["model_ver_x_montrg_auto"])
auto_results_nd = non_deleted(raw["montrg_auto_rslt"])
auto_join = (
    auto_links_nd
    .select(
        F.col("model_ver_sid").cast("string").alias("model_ver_sid"),
        F.col("montrg_auto_sid").cast("string").alias("montrg_auto_sid"),
    )
    .join(
        auto_results_nd.select(
            F.col("montrg_auto_sid").cast("string").alias("montrg_auto_sid"),
            *([F.col("montrg_auto_rslt_name")] if "montrg_auto_rslt_name" in auto_results_nd.columns else []),
        ),
        "montrg_auto_sid", "left"
    )
)
auto_flags = flag_from_presence(auto_join, "model_ver_sid", "montrg_auto_rslt_name", "has_auto_result")
candidates = candidates.join(auto_flags, "model_ver_sid", "left").fillna({"has_auto_result": 0})

# Удобные диагностические признаки.
if "model_ver_montrg_last_rslt_name" in candidates.columns:
    candidates = candidates.withColumn(
        "card_has_monitor_result",
        F.when(present_col("model_ver_montrg_last_rslt_name"), 1).otherwise(0),
    )
else:
    candidates = candidates.withColumn("card_has_monitor_result", F.lit(0))

if "model_ver_not_valid_flag" in candidates.columns:
    candidates = candidates.withColumn(
        "validation_not_required",
        F.when(truthy_col("model_ver_not_valid_flag"), 1).otherwise(0),
    )
else:
    candidates = candidates.withColumn("validation_not_required", F.lit(0))

if "model_ver_montrg_auto_exclude_reason_array" in candidates.columns:
    candidates = candidates.withColumn(
        "auto_monitoring_excluded",
        F.when(present_col("model_ver_montrg_auto_exclude_reason_array"), 1).otherwise(0),
    )
else:
    candidates = candidates.withColumn("auto_monitoring_excluded", F.lit(0))

candidates = candidates.withColumn(
    "linked_monitor_result_present",
    F.when((F.col("has_manual_result") == 1) | (F.col("has_auto_result") == 1), 1).otherwise(0),
)

# -----------------------------------------------------------------------------
# 5. СТРАТИФИЦИРОВАННЫЙ ОТБОР
# -----------------------------------------------------------------------------

parts = []

parts.append(sample_condition(
    candidates,
    (F.col("card_has_monitor_result") == 1) & (F.col("linked_monitor_result_present") == 0),
    "CARD_MONITOR_RESULT_BUT_NO_LINKED_RESULT",
))
parts.append(sample_condition(
    candidates,
    (F.col("has_manual_result") == 1) & (F.col("has_auto_result") == 0),
    "MANUAL_ONLY",
))
parts.append(sample_condition(
    candidates,
    (F.col("has_manual_result") == 0) & (F.col("has_auto_result") == 1),
    "AUTO_ONLY",
))
parts.append(sample_condition(
    candidates,
    (F.col("has_manual_result") == 1) & (F.col("has_auto_result") == 1),
    "MANUAL_AND_AUTO",
))
parts.append(sample_condition(
    candidates,
    (F.col("linked_monitor_result_present") == 0),
    "NO_LINKED_MONITOR_RESULT",
))
parts.append(sample_condition(
    candidates,
    F.col("validation_not_required") == 1,
    "VALIDATION_NOT_REQUIRED",
))
parts.append(sample_condition(
    candidates,
    (F.col("validation_not_required") == 0) & (F.col("has_valid_result") == 1),
    "VALIDATION_WITH_RESULT",
))
parts.append(sample_condition(
    candidates,
    (F.col("validation_not_required") == 0) & (F.col("has_valid_result") == 0),
    "VALIDATION_WITHOUT_RESULT",
))
parts.append(sample_condition(
    candidates,
    F.col("auto_monitoring_excluded") == 1,
    "AUTO_MONITORING_EXCLUDED",
))

# Дополнительно берём по несколько версий каждой категории значимости.
sign_col = None
for candidate_col in ["model_ver_signfcnt_ctgry_code", "model_ver_signfcnt_lvl_name"]:
    if candidate_col in candidates.columns:
        sign_col = candidate_col
        break

if sign_col:
    w_sign = Window.partitionBy(sign_col).orderBy(F.rand(seed=29))
    sign_sample = (
        candidates
        .filter(present_col(sign_col))
        .withColumn("__rn_sign", F.row_number().over(w_sign))
        .filter(F.col("__rn_sign") <= N_PER_SIGNIFICANCE)
        .drop("__rn_sign")
        .withColumn("sample_reason", F.concat(F.lit("SIGNIFICANCE="), F.col(sign_col).cast("string")))
    )
    parts.append(sign_sample)

parts.append(
    candidates.orderBy(F.rand(seed=41)).limit(N_RANDOM).withColumn("sample_reason", F.lit("RANDOM"))
)

selected = reduce(lambda x, y: x.unionByName(y, allowMissingColumns=True), parts)

# Одна версия могла попасть в несколько групп. Сохраняем причины отбора.
reason_agg = (
    selected.groupBy("model_ver_sid")
    .agg(F.concat_ws(" | ", F.sort_array(F.collect_set("sample_reason"))).alias("sample_reason"))
)

selected_versions = (
    candidates.dropDuplicates(["model_ver_sid"])
    .join(reason_agg, "model_ver_sid", "inner")
    .orderBy("model_ver_sid")
    .limit(MAX_VERSIONS)
)

selected_ids = selected_versions.select("model_ver_sid").distinct()
selected_id_values = [r["model_ver_sid"] for r in selected_ids.collect()]

if not selected_id_values:
    raise RuntimeError("Не удалось отобрать ни одной версии модели")

print(f"Отобрано версий: {len(selected_id_values)}")
print("MODEL_VER_SID:", ", ".join(selected_id_values))

# -----------------------------------------------------------------------------
# 6. СВЯЗАННЫЕ СЫРЫЕ СТРОКИ
# -----------------------------------------------------------------------------

# MODEL_VER — ВСЯ доступная история по выбранным SID.
model_ver_exp = raw["model_ver"].filter(F.col("model_ver_sid").cast("string").isin(selected_id_values))

# Получаем MODEL_SID для тех же версий.
model_sid_values = [
    r["model_sid"] for r in model_ver_exp.select(F.col("model_sid").cast("string").alias("model_sid"))
    .where(F.col("model_sid").isNotNull()).distinct().collect()
]
model_exp = raw["model"].filter(F.col("model_sid").cast("string").isin(model_sid_values)) if model_sid_values else raw["model"].limit(0)

anlt_exp = raw["model_ver_anlt_dtl"].filter(F.col("model_ver_sid").cast("string").isin(selected_id_values))
valid_exp = raw["valid"].filter(F.col("model_ver_sid").cast("string").isin(selected_id_values))

manual_link_exp = raw["model_ver_x_montrg_manual"].filter(F.col("model_ver_sid").cast("string").isin(selected_id_values))
manual_sid_values = [
    r["sid"] for r in manual_link_exp.select(F.col("montrg_manual_sid").cast("string").alias("sid"))
    .where(F.col("sid").isNotNull()).distinct().collect()
]
manual_mon_exp = raw["montrg_manual"].filter(F.col("montrg_manual_sid").cast("string").isin(manual_sid_values)) if manual_sid_values else raw["montrg_manual"].limit(0)
manual_rslt_exp = raw["montrg_manual_rslt"].filter(F.col("montrg_manual_sid").cast("string").isin(manual_sid_values)) if manual_sid_values else raw["montrg_manual_rslt"].limit(0)

auto_link_exp = raw["model_ver_x_montrg_auto"].filter(F.col("model_ver_sid").cast("string").isin(selected_id_values))
auto_sid_values = [
    r["sid"] for r in auto_link_exp.select(F.col("montrg_auto_sid").cast("string").alias("sid"))
    .where(F.col("sid").isNotNull()).distinct().collect()
]
auto_mon_exp = raw["montrg_auto"].filter(F.col("montrg_auto_sid").cast("string").isin(auto_sid_values)) if auto_sid_values else raw["montrg_auto"].limit(0)
auto_rslt_exp = raw["montrg_auto_rslt"].filter(F.col("montrg_auto_sid").cast("string").isin(auto_sid_values)) if auto_sid_values else raw["montrg_auto_rslt"].limit(0)

sample_exp = raw["sample_data"].filter(F.col("model_ver_sid").cast("string").isin(selected_id_values))

# История изменений может относиться и к MODEL, и к MODEL_VER.
change_ids = list(set(selected_id_values + model_sid_values))
change_exp = raw["change_log"].filter(F.col("ent_sid").cast("string").isin(change_ids)) if change_ids else raw["change_log"].limit(0)

# -----------------------------------------------------------------------------
# 7. ОСТАВЛЯЕМ ПОЛЯ, НУЖНЫЕ ДЛЯ НАШЕЙ ПРОВЕРКИ
#    Технические SCD-поля сохраняем обязательно.
# -----------------------------------------------------------------------------

TECH = ["ctl_action", "ctl_datechange", "ctl_datecreate_dttm", "ctl_loading", "start_dt", "end_dt"]

exports = {
    "00_selected_versions": selected_versions,

    "01_model": select_existing(model_exp, [
        "model_sid", "model_code", "model_name", "model_other_docs_sid",
    ] + TECH),

    "02_model_ver": select_existing(model_ver_exp, [
        "model_sid", "model_ver_sid", "model_ver_num", "model_ver_name", "model_ver_crtn_dttm",
        "model_ver_dev_start_fact_dttm", "model_ver_dev_end_fact_dttm",
        "model_ver_prom_actual_implm_first_dttm", "model_ver_prom_implm_first_dttm",
        "model_ver_owner_block_name", "model_ver_owner_dprtmt_name", "model_ver_owner_cont_name",
        "model_ver_dev_block_name", "model_ver_dev_dprtmt_name", "model_ver_dev_dprtmt_cont_name",
        "model_ver_signfcnt_ctgry_code", "model_ver_signfcnt_lvl_name",
        "model_ver_repstry_link_txt", "model_ver_mlstorage_branch_link_txt",
        "model_ver_implm_repstry_link_txt", "model_ver_prom_repstry_commit_link_txt",
        "model_ver_repstry_non_link_reason_name", "model_ver_repstry_non_link_reason_cmnt_txt",
        "model_ver_other_docs_sid", "model_ver_dev_report_sid",
        "model_ver_montrg_last_rslt_name", "model_ver_montrg_last_rslt_dttm", "model_ver_montrg_last_sid",
        "model_ver_montrg_auto_exclude_reason_array",
    ] + TECH),

    "03_anlt_dtl": select_existing(anlt_exp, [
        "model_ver_sid", "model_ver_stts_name", "model_stts_name", "model_ver_prom_expl_flag",
        "model_ver_prom_actual_implm_first_dttm", "model_ver_prom_implm_first_dttm",
        "model_ver_prom_actual_expl_output_dttm", "model_ver_prom_expl_output_dttm",
        "model_ver_owner_block_name", "model_ver_owner_dprtmt_name", "model_ver_owner_cont_array",
        "model_ver_dev_block_name", "model_ver_dev_dprtmt_name", "model_ver_dev_dprtmt_cont_array",
        "model_ver_employer_block_name", "model_ver_employer_dprtmt_name",
        "model_ver_risk_sgmnt_name", "model_ver_busn_proc_code",
        "model_ver_signfcnt_ctgry_code", "model_ver_signfcnt_lvl_name",
        "model_ver_not_valid_flag",
    ] + TECH),

    "04_validation": select_existing(valid_exp, [
        "model_ver_sid", "valid_sid", "valid_type_name", "valid_stts_name",
        "valid_crtn_dttm", "valid_start_fact_dttm", "valid_end_fact_dttm", "valid_neg_dttm",
        "valid_rslt_name", "valid_report_sid", "valid_not_sample_data_flag",
        "valid_cmmt_txt", "valid_cmnt_txt",
    ] + TECH),

    "05_manual_link": select_existing(manual_link_exp, [
        "model_ver_sid", "montrg_manual_sid",
    ] + TECH),

    "06_manual_monitor": select_existing(manual_mon_exp, [
        "montrg_manual_sid", "montrg_type_name", "montrg_manual_type_name", "montrg_manual_stts_name",
        "montrg_manual_crtn_dttm", "montrg_manual_start_dttm", "montrg_manual_end_dttm",
        "montrg_manual_last_rslt_name", "montrg_manual_last_rslt_dttm", "montrg_manual_next_dttm",
        "montrg_manual_prev_sid", "montrg_manual_prev_name",
    ] + TECH),

    "07_manual_result": select_existing(manual_rslt_exp, [
        "montrg_manual_sid", "montrg_manual_rslt_sid", "montrg_manual_rslt_stts_name",
        "montrg_manual_rslt_crtn_dttm", "montrg_manual_rslt_end_dttm", "montrg_manual_rslt_neg_dttm",
        "montrg_manual_rslt_name", "montrg_manual_rslt_report_sid",
        "montrg_manual_rslt_valid_not_sample_data_flag",
    ] + TECH),

    "08_auto_link": select_existing(auto_link_exp, [
        "model_ver_sid", "montrg_auto_sid",
    ] + TECH),

    "09_auto_monitor": select_existing(auto_mon_exp, [
        "montrg_auto_sid", "montrg_auto_type_name", "montrg_auto_stts_name", "montrg_auto_src_name",
        "montrg_auto_crtn_dttm", "montrg_auto_start_dttm", "montrg_auto_end_dttm",
        "montrg_auto_rslt_last_name", "montrg_auto_rslt_last_dttm", "montrg_auto_next_dttm",
        "montrg_auto_prev_sid", "montrg_auto_prev_name",
    ] + TECH),

    "10_auto_result": select_existing(auto_rslt_exp, [
        "montrg_auto_sid", "montrg_auto_rslt_sid", "montrg_auto_rslt_stts_name",
        "montrg_auto_rslt_crtn_dttm", "montrg_auto_rslt_end_dttm", "montrg_auto_rslt_neg_dttm",
        "montrg_auto_rslt_name", "montrg_auto_rslt_report_sid",
        "montrg_auto_rslt_valid_not_sample_data_flag",
    ] + TECH),

    "11_sample_data": select_existing(sample_exp, [
        "sample_data_sid", "model_ver_sid", "valid_sid",
        "montrg_manual_rslt_sid", "montrg_auto_rslt_sid",
        "sample_data_type_name", "sample_data_prop_array", "sample_data_stts_name",
        "sample_data_addtnl_info_txt", "sample_data_not_metric_calc_flag",
        "sample_data_crtn_dttm",
    ] + TECH),

    "12_change_log": select_existing(change_exp, [
        "ent_sid", "ent_type_name", "ent_param_chg_sid", "ent_param_chg_name",
        "ent_param_chg_prev_val", "ent_param_chg_val", "ent_param_chg_type_code",
        "ent_param_chg_usr_sid", "ent_param_chg_usr_name",
        "start_dttm", "end_dttm",
    ] + TECH),
}

# -----------------------------------------------------------------------------
# 8. ВЫГРУЗКА В ОДИН XLSX
# -----------------------------------------------------------------------------

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for sheet_name, sdf in exports.items():
        print(f"Выгрузка {sheet_name} ...")
        pdf = to_pandas_limited(sdf, sheet_name)
        pdf.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        ws = writer.book[sheet_name[:31]]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

print(f"Готово: {OUTPUT_FILE}")
