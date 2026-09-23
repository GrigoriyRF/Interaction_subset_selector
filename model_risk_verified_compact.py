"""Контроли модельного риска на этапе разработки: 28 рабочих + 3 динамических.

Результаты:
  arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis
  arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis_details

Скрипт использует только модельный контур разработки и ретроспективные
результаты валидации. Мониторинг, внедрение и GenAI в расчёт не входят.
Три динамических контроля рассчитываются при наличии применимых записей.
"""

from functools import reduce
from typing import List
from pathlib import Path
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window, functions as F

# Запуск в ноутбуке с существующей сессией spark.


# =============================================================================
# 1. CONFIG
# =============================================================================

SRC_DB = "prx_pri_custom_ris_l_library_custom_risk_model_library"
OUT_TABLE = "arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis"
OUT_DETAIL_TABLE = "arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis_details"

SHOW_ROWS = 100
MIN_GROUP_FOR_P90 = 5

conf = (
    SparkConf()
    .setAppName("model_risk_development_28_plus_3")
    .setMaster("yarn")
    .set("spark.executor.cores", "2")
    .set("spark.executor.memory", "6g")
    .set("spark.executor.memoryOverhead", "1g")
    .set("spark.driver.memory", "6g")
    .set("spark.driver.maxResultSize", "4g")
    .set("spark.dynamicAllocation.enabled", "true")
    .set("spark.dynamicAllocation.initialExecutors", "3")
    .set("spark.dynamicAllocation.maxExecutors", "12")
    .set("spark.dynamicAllocation.executorIdleTimeout", "120s")
    .set("spark.dynamicAllocation.cachedExecutorIdleTimeout", "600s")
    .set("spark.shuffle.service.enabled", "true")
    .set("spark.sql.parquet.writeLegacyFormat", "true")
    .set("spark.sql.parquet.compression.codec", "snappy")
    .set("spark.sql.session.timeZone", "Europe/Moscow")
)

spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()


# =============================================================================
# 2. НОРМАЛИЗАЦИЯ И АКТУАЛЬНЫЕ ЗАПИСИ
# =============================================================================

NULL_TEXT = "'', 'null', 'none'"
ARTIFACT_NULL_TEXT = "'', 'null', 'none', '-', 'нет', 'n/a', 'na', '$not_applicable$', 'отсутствует'"

DEV_STATUS_RE = r"BACKLOG|DEVELOPMENT|TRAIN_VALIDATION|VERIFICATION|AWAITING_VALIDATION"
STARTED_STATUS_RE = r"DEVELOPMENT|TRAIN_VALIDATION|VERIFICATION|AWAITING_VALIDATION"
GATE_STATUS_RE = (
    r"TRAIN_VALIDATION|VERIFICATION|AWAITING_VALIDATION|DECISION_MAKING|"
    r"PILOTING_PERMITTED|EXPLOIT_PERMITTED"
)
CLOSED_MODEL_STATUS_RE = r"CANCEL|CANCELED|ARCHIV|REJECT"
FINAL_VALID_STATUS_RE = r"DONE|COMPLETE|RESULTS_APPROVE"
EXCLUDED_VALID_STATUS_RE = r"CANCEL|CANCELED|NOT_NEEDED|SUSPENDED|REJECT"
NEGATIVE_VALID_STATUS_RE = r"DONE_NEGATIVE"
RED_YELLOW_RE = r"КРАСН|Ж[ЕЁ]ЛТ|RED|YELLOW"

UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
UUID_LIST_RE = rf"^{UUID}(\s*,\s*{UUID})*$"


def ts_col(name: str):
    raw = F.col(name)
    text = raw.cast("string")
    return F.coalesce(
        raw.cast("timestamp"),
        F.to_timestamp(text, "dd.MM.yyyy H:mm:ss"),
        F.to_timestamp(text, "dd.MM.yyyy HH:mm:ss"),
        F.to_timestamp(text, "yyyy-MM-dd HH:mm:ss.SSSSSS"),
        F.to_timestamp(text, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(text, "yyyy-MM-dd"),
    )


def blank(name: str) -> str:
    return f"({name} IS NULL OR LOWER(TRIM(CAST({name} AS STRING))) IN ({NULL_TEXT}))"


def artifact_missing(name: str) -> str:
    return (
        f"({name} IS NULL OR LOWER(TRIM(CAST({name} AS STRING))) "
        f"IN ({ARTIFACT_NULL_TEXT}))"
    )


def truth(name: str) -> str:
    return (
        f"COALESCE(LOWER(TRIM(CAST({name} AS STRING))) "
        f"IN ('true','1','да','yes','y'), FALSE)"
    )


def real_date(name: str) -> str:
    return (
        f"COALESCE(DATE({name}) >= DATE '2000-01-01' "
        f"AND DATE({name}) < DATE '2090-01-01', FALSE)"
    )


def clean_dim(name: str) -> str:
    return (
        f"CASE WHEN {blank(name)} THEN '$NULL$' "
        f"ELSE REGEXP_REPLACE(TRIM(CAST({name} AS STRING)), '\\\\s+', ' ') END"
    )


def load_current(table: str, columns: List[str], key: str) -> DataFrame:
    """Отбирает бизнес-актуальные строки и последнюю запись каждого SID."""
    raw = spark.table(f"{SRC_DB}.{table}")
    existing = set(raw.columns)
    missing = sorted(set(columns + [key]) - existing)
    if missing:
        raise ValueError(f"{table}: отсутствуют поля: {missing}")

    df = raw
    if "ctl_action" in existing:
        df = df.filter(
            F.col("ctl_action").isNull()
            | (F.upper(F.trim(F.col("ctl_action").cast("string"))) != "D")
        )
    if "start_dt" in existing:
        df = df.filter(ts_col("start_dt").isNull() | (F.to_date(ts_col("start_dt")) <= F.current_date()))
    if "end_dt" in existing:
        df = df.filter(ts_col("end_dt").isNull() | (F.to_date(ts_col("end_dt")) >= F.current_date()))

    order_cols = []
    for candidate in ("ctl_datechange", "start_dt", "ctl_datecreate_dttm"):
        if candidate in existing:
            order_cols.append(ts_col(candidate).desc_nulls_last())
    if not order_cols:
        order_cols = [F.col(key).desc_nulls_last()]

    df = (
        df.withColumn("_rn", F.row_number().over(Window.partitionBy(key).orderBy(*order_cols)))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    projected = []
    for name in columns:
        if name.endswith("_dttm"):
            projected.append(ts_col(name).alias(name))
        else:
            projected.append(F.col(name))
    return df.select(*projected)


SOURCES = {
    "mv": (
        "t_model_ver",
        "model_ver_sid",
        [
            "model_ver_sid", "model_sid", "model_ver_stts_name",
            "model_ver_signfcnt_ctgry_code", "model_ver_owner_dprtmt_name",
            "model_ver_dev_dprtmt_name", "model_ver_method_name",
            "model_ver_data_type_name", "model_ver_dev_start_plan_dttm",
            "model_ver_dev_start_fact_dttm", "model_ver_dev_end_plan_dttm",
            "model_ver_dev_end_fact_dttm", "model_ver_dev_report_sid",
            "model_ver_prevalid_flag", "model_ver_prevalid_rslt_name",
            "model_ver_llm_flag", "model_ver_llm_descr_txt",
            "model_ver_proj_feature_flag", "model_ver_proj_feature_sid",
            "model_ver_repstry_link_txt", "model_ver_repstry_commit_sid",
            "model_ver_crtn_dttm",
        ],
    ),
    "sample": (
        "t_sample_data",
        "sample_data_sid",
        [
            "sample_data_sid", "model_ver_sid", "valid_sid",
            "sample_data_type_name", "sample_data_not_metric_calc_flag",
        ],
    ),
    "metric": (
        "t_metric",
        "metric_sid",
        ["metric_sid", "sample_data_sid", "metric_name", "metric_val"],
    ),
    "valid": (
        "t_valid",
        "valid_sid",
        [
            "valid_sid", "model_ver_sid", "valid_stts_name", "valid_rslt_name",
            "valid_crtn_dttm", "valid_start_fact_dttm", "valid_end_fact_dttm",
            "valid_dev_return_reason_txt", "valid_dev_return_reason_validator_cmnt_txt",
            "valid_dev_return_reason_developer_cmnt_txt", "valid_prblm_name",
        ],
    ),
}

cached: List[DataFrame] = []
for view, (table, key, columns) in SOURCES.items():
    frame = load_current(table, columns, key).persist(StorageLevel.MEMORY_AND_DISK)
    frame.createOrReplaceTempView(view)
    cached.append(frame)

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW mv_active AS
SELECT * FROM mv
WHERE UPPER(COALESCE(model_ver_stts_name,'')) NOT RLIKE '{CLOSED_MODEL_STATUS_RE}'
""")

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW mv_dev_scope AS
SELECT * FROM mv_active
WHERE UPPER(COALESCE(model_ver_stts_name,'')) RLIKE '{DEV_STATUS_RE}'
""")

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW mv_started_scope AS
SELECT * FROM mv_active
WHERE UPPER(COALESCE(model_ver_stts_name,'')) RLIKE '{STARTED_STATUS_RE}'
""")

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW mv_gate_scope AS
SELECT * FROM mv_active
WHERE UPPER(COALESCE(model_ver_stts_name,'')) RLIKE '{GATE_STATUS_RE}'
""")

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW valid_completed AS
SELECT * FROM valid
WHERE UPPER(COALESCE(valid_stts_name,'')) NOT RLIKE '{EXCLUDED_VALID_STATUS_RE}'
  AND (
      UPPER(COALESCE(valid_stts_name,'')) RLIKE '{FINAL_VALID_STATUS_RE}'
      OR {real_date('valid_end_fact_dttm')}
  )
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW sample_model_map AS
SELECT s.*,
       COALESCE(s.model_ver_sid, v.model_ver_sid) AS resolved_model_ver_sid
FROM sample s
LEFT JOIN valid v ON s.valid_sid = v.valid_sid
""")


# =============================================================================
# 3. ЕДИНЫЙ КОНСТРУКТОР КОНТРОЛЕЙ
# =============================================================================

summary_parts: List[DataFrame] = []
detail_parts: List[DataFrame] = []


def add_control(
    query_id: int,
    source_check_id: int,
    query_name: str,
    risk_domain: str,
    severity: str,
    dim_1_name: str,
    dim_2_name: str,
    source_tables: str,
    interpretation: str,
    empty_reason: str,
    sql_body: str,
) -> None:
    """sql_body: model_ver_sid, source_entity_type/sid, issue_name,
    dim_1_value, dim_2_value, is_issue. Одна строка = одна применимая сущность.
    """
    raw = spark.sql(sql_body)
    expected_columns = [
        "model_ver_sid", "source_entity_type", "source_entity_sid",
        "issue_name", "dim_1_value", "dim_2_value", "is_issue",
    ]
    if len(raw.columns) != len(expected_columns):
        raise ValueError(
            f"Контроль {query_id}: ожидалось 7 полей, получено {len(raw.columns)}"
        )
    base = raw.toDF(*expected_columns).select(
        F.col("model_ver_sid").cast("string").alias("model_ver_sid"),
        F.col("source_entity_type").cast("string").alias("source_entity_type"),
        F.col("source_entity_sid").cast("string").alias("source_entity_sid"),
        F.col("issue_name").cast("string").alias("issue_name"),
        F.coalesce(F.col("dim_1_value").cast("string"), F.lit("$NULL$")).alias("dim_1_value"),
        F.coalesce(F.col("dim_2_value").cast("string"), F.lit("$ALL$")).alias("dim_2_value"),
        F.col("is_issue").cast("int").alias("is_issue"),
    )

    grouped = base.groupBy("dim_1_value", "dim_2_value").agg(
        F.sum("is_issue").cast("long").alias("numerator"),
        F.count(F.lit(1)).cast("long").alias("denominator"),
    )
    grouped = grouped.select(
        F.lit(query_id).cast("int").alias("query_id"),
        F.lit(source_check_id).cast("int").alias("source_check_id"),
        F.lit(query_name).alias("query_name"),
        F.lit(risk_domain).alias("risk_domain"),
        F.lit("MODEL").alias("entity_type"),
        F.lit(severity).alias("severity"),
        F.lit("ASSESSABLE").alias("control_status"),
        F.lit(None).cast("string").alias("status_reason"),
        F.lit("risk_rate_pct").alias("metric_name"),
        F.lit(dim_1_name).alias("dim_1_name"),
        F.col("dim_1_value"),
        F.lit(dim_2_name).alias("dim_2_name"),
        F.col("dim_2_value"),
        (100.0 * F.col("numerator") / F.col("denominator")).cast("double").alias("metric_value"),
        F.col("numerator"),
        F.col("denominator"),
        F.current_date().alias("as_of_dt"),
        F.lit(source_tables).alias("source_tables"),
        F.lit(interpretation).alias("interpretation"),
        F.lit(0).alias("_placeholder"),
    )

    placeholder = spark.range(1).select(
        F.lit(query_id).cast("int").alias("query_id"),
        F.lit(source_check_id).cast("int").alias("source_check_id"),
        F.lit(query_name).alias("query_name"),
        F.lit(risk_domain).alias("risk_domain"),
        F.lit("MODEL").alias("entity_type"),
        F.lit(severity).alias("severity"),
        F.lit("NOT_ASSESSABLE").alias("control_status"),
        F.lit(empty_reason).alias("status_reason"),
        F.lit("risk_rate_pct").alias("metric_name"),
        F.lit(dim_1_name).alias("dim_1_name"),
        F.lit("$ALL$").alias("dim_1_value"),
        F.lit(dim_2_name).alias("dim_2_name"),
        F.lit("$ALL$").alias("dim_2_value"),
        F.lit(None).cast("double").alias("metric_value"),
        F.lit(None).cast("long").alias("numerator"),
        F.lit(0).cast("long").alias("denominator"),
        F.current_date().alias("as_of_dt"),
        F.lit(source_tables).alias("source_tables"),
        F.lit(interpretation).alias("interpretation"),
        F.lit(1).alias("_placeholder"),
    )
    summary_parts.append(grouped.unionByName(placeholder))

    detail_parts.append(
        base.filter(
            (F.col("is_issue") == 1)
            & F.col("model_ver_sid").isNotNull()
            & ~F.lower(F.trim(F.col("model_ver_sid"))).isin("", "null", "none")
        ).select(
            F.lit(query_id).cast("int").alias("query_id"),
            F.lit(source_check_id).cast("int").alias("source_check_id"),
            F.lit(query_name).alias("query_name"),
            F.lit(risk_domain).alias("risk_domain"),
            F.lit(severity).alias("severity"),
            F.lit("MODEL").alias("entity_type"),
            F.col("model_ver_sid"),
            F.col("source_entity_type"),
            F.col("source_entity_sid"),
            F.col("issue_name"),
            F.col("dim_1_value"),
            F.col("dim_2_value"),
            F.current_date().alias("as_of_dt"),
        ).dropDuplicates(["query_id", "source_entity_type", "source_entity_sid", "issue_name"])
    )


# =============================================================================
# 4. 28 ПОДТВЕРЖДЁННЫХ КОНТРОЛЕЙ
# =============================================================================

add_control(1, 1, "Плановое начало позже планового окончания", "development_timeline", "HIGH",
            "department", "status", "t_model_ver",
            "Нарушена последовательность плановых дат разработки.",
            "Нет версий с двумя содержательными плановыми датами.", f"""
SELECT model_ver_sid,'MODEL_VER' source_entity_type,model_ver_sid source_entity_sid,
       'planned_start_after_planned_end' issue_name,
       {clean_dim('model_ver_dev_dprtmt_name')} dim_1_value,
       {clean_dim('model_ver_stts_name')} dim_2_value,
       CASE WHEN model_ver_dev_start_plan_dttm > model_ver_dev_end_plan_dttm THEN 1 ELSE 0 END is_issue
FROM mv_dev_scope
WHERE {real_date('model_ver_dev_start_plan_dttm')} AND {real_date('model_ver_dev_end_plan_dttm')}
""")

add_control(2, 2, "Фактическое окончание раньше начала", "development_timeline", "HIGH",
            "department", "status", "t_model_ver",
            "Нарушена последовательность фактических дат разработки.",
            "Нет версий с двумя содержательными фактическими датами.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'actual_end_before_actual_start',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_stts_name')},
       CASE WHEN model_ver_dev_end_fact_dttm < model_ver_dev_start_fact_dttm THEN 1 ELSE 0 END
FROM mv_active
WHERE {real_date('model_ver_dev_start_fact_dttm')} AND {real_date('model_ver_dev_end_fact_dttm')}
""")

add_control(3, 3, "Фактическая дата разработки находится в будущем", "development_timeline", "HIGH",
            "department", "status", "t_model_ver",
            "Будущая фактическая дата указывает на ошибку заполнения.",
            "Нет версий с содержательными фактическими датами.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'future_actual_development_date',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_stts_name')},
       CASE WHEN ({real_date('model_ver_dev_start_fact_dttm')} AND model_ver_dev_start_fact_dttm>CURRENT_TIMESTAMP())
                  OR ({real_date('model_ver_dev_end_fact_dttm')} AND model_ver_dev_end_fact_dttm>CURRENT_TIMESTAMP())
            THEN 1 ELSE 0 END
FROM mv_dev_scope
WHERE {real_date('model_ver_dev_start_fact_dttm')} OR {real_date('model_ver_dev_end_fact_dttm')}
""")

add_control(4, 4, "Разработка открыта при заполненной дате окончания", "development_timeline", "MEDIUM",
            "department", "status", "t_model_ver",
            "Статус DEVELOPMENT противоречит фактическому окончанию разработки.",
            "Нет версий в статусе DEVELOPMENT.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'development_status_with_actual_end',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {real_date('model_ver_dev_end_fact_dttm')} THEN 1 ELSE 0 END
FROM mv_active
WHERE UPPER(COALESCE(model_ver_stts_name,'')) RLIKE 'MODEL_VERSION_DEVELOPMENT$'
""")

add_control(5, 5, "Постразработочный статус без даты окончания", "development_timeline", "HIGH",
            "department", "status", "t_model_ver",
            "Версия перешла контрольный шлюз без зафиксированного окончания разработки.",
            "Нет версий на постразработочных стадиях.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'post_development_status_without_actual_end',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_stts_name')},
       CASE WHEN NOT {real_date('model_ver_dev_end_fact_dttm')} THEN 1 ELSE 0 END
FROM mv_gate_scope
""")

add_control(6, 6, "Разработка начата позже плана", "development_timeline", "MEDIUM",
            "department", "status", "t_model_ver",
            "Фактический старт разработки позже планового.",
            "Нет версий с плановой и фактической датами начала.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'actual_start_after_plan',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_stts_name')},
       CASE WHEN model_ver_dev_start_fact_dttm > model_ver_dev_start_plan_dttm THEN 1 ELSE 0 END
FROM mv_dev_scope
WHERE {real_date('model_ver_dev_start_plan_dttm')} AND {real_date('model_ver_dev_start_fact_dttm')}
""")

add_control(7, 7, "Открытая разработка без планового окончания", "development_timeline", "MEDIUM",
            "department", "significance", "t_model_ver",
            "У открытой разработки отсутствует контролируемый срок завершения.",
            "Нет версий в статусе DEVELOPMENT.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'open_development_without_planned_end',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_signfcnt_ctgry_code')},
       CASE WHEN NOT {real_date('model_ver_dev_end_plan_dttm')} THEN 1 ELSE 0 END
FROM mv_active
WHERE UPPER(COALESCE(model_ver_stts_name,'')) RLIKE 'MODEL_VERSION_DEVELOPMENT$'
""")

add_control(8, 8, "Разработка завершена позже плана", "development_timeline", "HIGH",
            "department", "significance", "t_model_ver",
            "Фактическое окончание разработки превышает плановый срок.",
            "Нет версий с плановой и фактической датами окончания.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'development_completed_late',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_signfcnt_ctgry_code')},
       CASE WHEN model_ver_dev_end_fact_dttm > model_ver_dev_end_plan_dttm THEN 1 ELSE 0 END
FROM mv_active
WHERE {real_date('model_ver_dev_end_plan_dttm')} AND {real_date('model_ver_dev_end_fact_dttm')}
""")

add_control(9, 9, "Длительность разработки выше p90 подразделения", "development_timeline", "MEDIUM",
            "department", "status", "t_model_ver",
            f"Версия превышает p90 длительности подразделения; статистика считается при n>={MIN_GROUP_FOR_P90}.",
            "Недостаточно завершённых разработок для расчёта p90.", f"""
WITH durations AS (
  SELECT model_ver_sid,model_ver_stts_name,
         {clean_dim('model_ver_dev_dprtmt_name')} department,
         (UNIX_TIMESTAMP(model_ver_dev_end_fact_dttm)-UNIX_TIMESTAMP(model_ver_dev_start_fact_dttm))/86400.0 days
  FROM mv_active
  WHERE {real_date('model_ver_dev_start_fact_dttm')} AND {real_date('model_ver_dev_end_fact_dttm')}
), stats AS (
  SELECT department,PERCENTILE_APPROX(days,0.90) p90,COUNT(*) n
  FROM durations WHERE days>=0 GROUP BY department HAVING COUNT(*)>={MIN_GROUP_FOR_P90}
)
SELECT d.model_ver_sid,'MODEL_VER',d.model_ver_sid,'duration_above_department_p90',
       d.department,{clean_dim('d.model_ver_stts_name')},CASE WHEN d.days>s.p90 THEN 1 ELSE 0 END
FROM durations d JOIN stats s ON d.department=s.department WHERE d.days>=0
""")

add_control(10, 11, "Не определён тип данных", "development_metadata", "HIGH",
            "significance", "status", "t_model_ver",
            "После начала разработки должен быть определён тип входных данных.",
            "Нет версий, начавших разработку.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'data_type_missing',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {blank('model_ver_data_type_name')} THEN 1 ELSE 0 END
FROM mv_started_scope
""")

add_control(11, 12, "Не указан метод моделирования", "development_metadata", "HIGH",
            "significance", "status", "t_model_ver",
            "После начала разработки должен быть указан метод моделирования.",
            "Нет версий, начавших разработку.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'modeling_method_missing',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {blank('model_ver_method_name')} THEN 1 ELSE 0 END
FROM mv_started_scope
""")

add_control(12, 13, "Не определена значимость перед валидацией", "development_governance", "HIGH",
            "department", "status", "t_model_ver",
            "До передачи на валидацию должна быть определена категория значимости.",
            "Нет версий на контрольном шлюзе.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'significance_missing_at_gate',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {blank('model_ver_signfcnt_ctgry_code')} THEN 1 ELSE 0 END
FROM mv_gate_scope
""")

add_control(13, 14, "Не указано подразделение-владелец", "development_governance", "HIGH",
            "significance", "status", "t_model_ver",
            "У версии отсутствует ответственное подразделение-владелец.",
            "Нет версий в периметре разработки.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'owner_department_missing',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {blank('model_ver_owner_dprtmt_name')} THEN 1 ELSE 0 END
FROM mv_dev_scope
""")

add_control(14, 15, "Не указано подразделение разработки", "development_governance", "HIGH",
            "significance", "status", "t_model_ver",
            "После начала разработки должно быть определено подразделение-разработчик.",
            "Нет версий, начавших разработку.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'development_department_missing',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {blank('model_ver_dev_dprtmt_name')} THEN 1 ELSE 0 END
FROM mv_started_scope
""")

add_control(15, 17, "LLM-версия без описания", "development_metadata", "MEDIUM",
            "department", "status", "t_model_ver",
            "Для версии с llm_flag=true отсутствует описание применения LLM.",
            "Нет версий разработки с содержательным llm_flag=true.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'llm_description_missing',
       {clean_dim('model_ver_dev_dprtmt_name')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {blank('model_ver_llm_descr_txt')} THEN 1 ELSE 0 END
FROM mv_dev_scope WHERE {truth('model_ver_llm_flag')}
""")

add_control(16, 25, "Применимая выборка без метрики", "development_data", "HIGH",
            "sample_type", "significance", "t_sample_data,t_metric,t_valid,t_model_ver",
            "У тестовой/валидационной выборки нет метрики при отсутствии явного запрета расчёта.",
            "Нет связанных применимых выборок.", f"""
WITH metric_presence AS (
  SELECT sample_data_sid,MAX(CASE WHEN metric_sid IS NOT NULL AND NOT {blank('metric_val')} THEN 1 ELSE 0 END) has_metric
  FROM metric GROUP BY sample_data_sid
)
SELECT s.resolved_model_ver_sid,'SAMPLE_DATA',s.sample_data_sid,'applicable_sample_without_metric',
       {clean_dim('s.sample_data_type_name')},{clean_dim('v.model_ver_signfcnt_ctgry_code')},
       CASE WHEN COALESCE(m.has_metric,0)=0 THEN 1 ELSE 0 END
FROM sample_model_map s
JOIN mv_active v ON s.resolved_model_ver_sid=v.model_ver_sid
LEFT JOIN metric_presence m ON s.sample_data_sid=m.sample_data_sid
WHERE UPPER(COALESCE(s.sample_data_type_name,'')) RLIKE 'ТЕСТ|ВАЛИДАЦ'
  AND NOT ({truth('s.sample_data_not_metric_calc_flag')})
""")

add_control(17, 26, "Значение метрики не приводится к числу", "development_data", "MEDIUM",
            "metric", "sample_type", "t_metric,t_sample_data,t_valid,t_model_ver",
            "Текстовое значение метрики невозможно преобразовать в число.",
            "Нет связанных заполненных значений метрик.", f"""
SELECT s.resolved_model_ver_sid,'METRIC',m.metric_sid,'metric_value_not_numeric',
       {clean_dim('m.metric_name')},{clean_dim('s.sample_data_type_name')},
       CASE WHEN TRY_CAST(TRIM(CAST(m.metric_val AS STRING)) AS DOUBLE) IS NULL THEN 1 ELSE 0 END
FROM metric m JOIN sample_model_map s ON m.sample_data_sid=s.sample_data_sid
JOIN mv_active v ON s.resolved_model_ver_sid=v.model_ver_sid
WHERE NOT {blank('m.metric_val')}
""")

add_control(18, 28, "Повтор одной метрики для одной выборки", "development_data", "MEDIUM",
            "metric", "sample_type", "t_metric,t_sample_data,t_valid,t_model_ver",
            "После SCD-дедупликации одна метрика повторяется в рамках одной выборки.",
            "Нет связанных именованных метрик.", f"""
WITH x AS (
  SELECT s.resolved_model_ver_sid,s.sample_data_sid,{clean_dim('s.sample_data_type_name')} sample_type,
         UPPER(TRIM(CAST(m.metric_name AS STRING))) metric_key,COUNT(*) cnt
  FROM metric m JOIN sample_model_map s ON m.sample_data_sid=s.sample_data_sid
  JOIN mv_active v ON s.resolved_model_ver_sid=v.model_ver_sid
  WHERE NOT {blank('m.metric_name')}
  GROUP BY s.resolved_model_ver_sid,s.sample_data_sid,{clean_dim('s.sample_data_type_name')},
           UPPER(TRIM(CAST(m.metric_name AS STRING)))
)
SELECT resolved_model_ver_sid,'SAMPLE_METRIC',CONCAT(CAST(sample_data_sid AS STRING),':',metric_key),
       'duplicate_metric_for_sample',metric_key,sample_type,CASE WHEN cnt>1 THEN 1 ELSE 0 END
FROM x
""")

add_control(19, 35, "Нет отчёта о разработке перед валидацией", "development_evidence", "HIGH",
            "significance", "status", "t_model_ver",
            "На контрольном шлюзе отсутствует идентификатор отчёта о разработке.",
            "Нет версий на контрольном шлюзе.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'development_report_missing_at_gate',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {artifact_missing('model_ver_dev_report_sid')} THEN 1 ELSE 0 END
FROM mv_gate_scope
""")

add_control(20, 36, "Некорректный формат идентификатора отчёта", "development_evidence", "MEDIUM",
            "significance", "status", "t_model_ver",
            "Идентификатор отчёта должен быть UUID или списком UUID через запятую.",
            "Нет версий с заполненным отчётом о разработке.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'development_report_id_invalid',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN TRIM(CAST(model_ver_dev_report_sid AS STRING)) NOT RLIKE '{UUID_LIST_RE}' THEN 1 ELSE 0 END
FROM mv_gate_scope WHERE NOT {artifact_missing('model_ver_dev_report_sid')}
""")

add_control(21, 40, "Превалидация проведена без результата", "development_prevalidation", "MEDIUM",
            "significance", "status", "t_model_ver",
            "Для prevalid_flag=true отсутствует результат превалидации.",
            "Нет версий с содержательным prevalid_flag=true.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'prevalidation_result_missing',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {blank('model_ver_prevalid_rslt_name')} THEN 1 ELSE 0 END
FROM mv_started_scope WHERE {truth('model_ver_prevalid_flag')}
""")

add_control(22, 45, "FeatureStore SID заполнен без положительного флага", "source_data_quality", "MEDIUM",
            "significance", "status", "t_model_ver",
            "SID проекта FeatureStore заполнен, но флаг использования не равен true.",
            "Нет версий с заполненным SID FeatureStore.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'feature_store_sid_without_true_flag',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN NOT ({truth('model_ver_proj_feature_flag')}) THEN 1 ELSE 0 END
FROM mv_dev_scope WHERE NOT {artifact_missing('model_ver_proj_feature_sid')}
""")

RETURN_BASE = f"""
FROM valid x JOIN mv_active v ON x.model_ver_sid=v.model_ver_sid
WHERE NOT {blank('x.valid_dev_return_reason_txt')}
"""

add_control(23, 51, "Возврат из-за отсутствия данных", "development_outcome", "HIGH",
            "department", "significance", "t_valid,t_model_ver",
            "Ретроспективный сигнал: валидатор вернул версию из-за отсутствия данных.",
            "Нет валидаций с причиной возврата.", f"""
SELECT v.model_ver_sid,'VALIDATION',x.valid_sid,'validation_return_missing_data',
       {clean_dim('v.model_ver_dev_dprtmt_name')},{clean_dim('v.model_ver_signfcnt_ctgry_code')},
       CASE WHEN UPPER(CAST(x.valid_dev_return_reason_txt AS STRING)) RLIKE 'НЕ ПРЕДОСТАВЛЕНЫ ДАННЫЕ' THEN 1 ELSE 0 END
{RETURN_BASE}
""")

add_control(24, 52, "Возврат из-за отсутствия доступа к данным", "development_outcome", "HIGH",
            "department", "significance", "t_valid,t_model_ver",
            "Ретроспективный сигнал: валидатору не предоставлен доступ к данным.",
            "Нет валидаций с причиной возврата.", f"""
SELECT v.model_ver_sid,'VALIDATION',x.valid_sid,'validation_return_no_data_access',
       {clean_dim('v.model_ver_dev_dprtmt_name')},{clean_dim('v.model_ver_signfcnt_ctgry_code')},
       CASE WHEN UPPER(CAST(x.valid_dev_return_reason_txt AS STRING)) RLIKE 'НЕ ПРЕДОСТАВЛЕН ДОСТУП' THEN 1 ELSE 0 END
{RETURN_BASE}
""")

add_control(25, 53, "Возврат из-за недостаточной информации для выборок", "development_outcome", "HIGH",
            "department", "significance", "t_valid,t_model_ver",
            "Ретроспективный сигнал: не предоставлена информация для формирования выборок.",
            "Нет валидаций с причиной возврата.", f"""
SELECT v.model_ver_sid,'VALIDATION',x.valid_sid,'validation_return_insufficient_sample_information',
       {clean_dim('v.model_ver_dev_dprtmt_name')},{clean_dim('v.model_ver_signfcnt_ctgry_code')},
       CASE WHEN UPPER(CAST(x.valid_dev_return_reason_txt AS STRING))
                      RLIKE 'НЕ ПРЕДОСТАВЛЕНА ИНФОРМАЦИЯ.*ФОРМИРОВАНИЯ ВАЛИДАЦИОННЫХ ВЫБОРОК'
            THEN 1 ELSE 0 END
{RETURN_BASE}
""")

add_control(26, 54, "Повторная отрицательная валидация версии", "development_outcome", "HIGH",
            "department", "significance", "t_valid,t_model_ver",
            "Версия получила более одного отрицательного/красного завершённого результата.",
            "Нет версий с завершённой валидацией.", f"""
WITH x AS (
  SELECT v.model_ver_sid,{clean_dim('v.model_ver_dev_dprtmt_name')} department,
         {clean_dim('v.model_ver_signfcnt_ctgry_code')} significance,
         SUM(CASE WHEN UPPER(COALESCE(c.valid_stts_name,'')) RLIKE '{NEGATIVE_VALID_STATUS_RE}'
                       OR UPPER(COALESCE(c.valid_rslt_name,'')) RLIKE 'КРАСН|RED' THEN 1 ELSE 0 END) negative_cnt
  FROM mv_active v JOIN valid_completed c ON v.model_ver_sid=c.model_ver_sid
  GROUP BY v.model_ver_sid,{clean_dim('v.model_ver_dev_dprtmt_name')},
           {clean_dim('v.model_ver_signfcnt_ctgry_code')}
)
SELECT model_ver_sid,'MODEL_VER_VALIDATION',model_ver_sid,'repeated_negative_validation',
       department,significance,CASE WHEN negative_cnt>1 THEN 1 ELSE 0 END
FROM x
""")

add_control(27, 55, "Красные и жёлтые результаты по типам проблем", "development_outcome", "HIGH",
            "department", "result", "t_valid,t_model_ver",
            "Доля завершённых валидаций с красным/жёлтым результатом; тип проблемы сохраняется в детализации.",
            "Нет завершённых валидаций.", f"""
SELECT v.model_ver_sid,'VALIDATION',c.valid_sid,
       CASE WHEN {blank('c.valid_prblm_name')} THEN 'adverse_validation_without_problem_type'
            ELSE CONCAT('adverse_validation:',REGEXP_REPLACE(TRIM(CAST(c.valid_prblm_name AS STRING)),'\\\\s+',' ')) END,
       {clean_dim('v.model_ver_dev_dprtmt_name')},{clean_dim('c.valid_rslt_name')},
       CASE WHEN UPPER(COALESCE(c.valid_rslt_name,'')) RLIKE '{RED_YELLOW_RE}' THEN 1 ELSE 0 END
FROM valid_completed c JOIN mv_active v ON c.model_ver_sid=v.model_ver_sid
""")

add_control(28, 56, "Версии с негативным результатом разработки по подразделению", "development_outcome", "HIGH",
            "department", "significance", "t_valid,t_model_ver",
            "Доля валидированных версий с возвратом либо красным/жёлтым результатом.",
            "Нет версий с завершённой валидацией.", f"""
WITH x AS (
  SELECT v.model_ver_sid,{clean_dim('v.model_ver_dev_dprtmt_name')} department,
         {clean_dim('v.model_ver_signfcnt_ctgry_code')} significance,
         MAX(CASE WHEN NOT {blank('c.valid_dev_return_reason_txt')}
                       OR UPPER(COALESCE(c.valid_rslt_name,'')) RLIKE '{RED_YELLOW_RE}'
                  THEN 1 ELSE 0 END) has_bad_outcome
  FROM mv_active v JOIN valid_completed c ON v.model_ver_sid=c.model_ver_sid
  GROUP BY v.model_ver_sid,{clean_dim('v.model_ver_dev_dprtmt_name')},
           {clean_dim('v.model_ver_signfcnt_ctgry_code')}
)
SELECT model_ver_sid,'MODEL_VER_VALIDATION',model_ver_sid,'negative_development_outcome',
       department,significance,has_bad_outcome FROM x
""")


# =============================================================================
# 5. 3 КОНТРОЛЯ С ДИНАМИЧЕСКОЙ ОЦЕНИМОСТЬЮ
# =============================================================================

add_control(29, 33, "Репозиторий есть, идентификатора коммита нет", "development_reproducibility", "HIGH",
            "significance", "status", "t_model_ver",
            "Контроль активируется автоматически при появлении содержательных ссылок на репозиторий.",
            "NOT_ASSESSABLE: в текущем профиле нет содержательных ссылок на репозиторий.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'repository_without_commit',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {artifact_missing('model_ver_repstry_commit_sid')} THEN 1 ELSE 0 END
FROM mv_started_scope WHERE NOT {artifact_missing('model_ver_repstry_link_txt')}
""")

add_control(30, 34, "Идентификатор коммита есть, репозитория нет", "development_reproducibility", "HIGH",
            "significance", "status", "t_model_ver",
            "Контроль активируется автоматически при появлении содержательных идентификаторов коммита.",
            "NOT_ASSESSABLE: в текущем профиле нет содержательных идентификаторов коммита.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'commit_without_repository',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {artifact_missing('model_ver_repstry_link_txt')} THEN 1 ELSE 0 END
FROM mv_started_scope WHERE NOT {artifact_missing('model_ver_repstry_commit_sid')}
""")

add_control(31, 46, "FeatureStore включён, SID проекта отсутствует", "source_data_quality", "MEDIUM",
            "significance", "status", "t_model_ver",
            "Контроль активируется автоматически при появлении содержательного true во флаге FeatureStore.",
            "NOT_ASSESSABLE: в текущем профиле нет содержательного proj_feature_flag=true.", f"""
SELECT model_ver_sid,'MODEL_VER',model_ver_sid,'feature_store_true_without_project_sid',
       {clean_dim('model_ver_signfcnt_ctgry_code')},{clean_dim('model_ver_stts_name')},
       CASE WHEN {artifact_missing('model_ver_proj_feature_sid')} THEN 1 ELSE 0 END
FROM mv_dev_scope WHERE {truth('model_ver_proj_feature_flag')}
""")


# =============================================================================
# 6. СБОРКА И ЗАПИСЬ 31 ИСХОДНОГО КОНТРОЛЯ
# =============================================================================

summary_raw = reduce(lambda a, b: a.unionByName(b), summary_parts)
w = Window.partitionBy("query_id")
final_df = (
    summary_raw
    .withColumn("_min_placeholder", F.min("_placeholder").over(w))
    .filter(F.col("_placeholder") == F.col("_min_placeholder"))
    .drop("_placeholder", "_min_placeholder")
)

detail_df = reduce(lambda a, b: a.unionByName(b), detail_parts).dropDuplicates()

final_df.write.mode("overwrite").format("parquet").saveAsTable(OUT_TABLE)
detail_df.write.mode("overwrite").format("parquet").saveAsTable(OUT_DETAIL_TABLE)


# =============================================================================
# 7. ПЕРИМЕТР EXCEL: ПОСЛЕДНЯЯ АКТУАЛЬНАЯ ВЕРСИЯ МОДЕЛИ В ЭКСПЛУАТАЦИИ
# =============================================================================

EXCEL_PATH = Path.cwd() / "model_risk_latest_prod.xlsx"


def take_latest(df: DataFrame, keys: List[str], order_column: str) -> DataFrame:
    ww = Window.partitionBy(*keys).orderBy(F.col(order_column).desc_nulls_last())
    return df.withColumn("_rn", F.row_number().over(ww)).filter(F.col("_rn") == 1).drop("_rn")


model_raw = spark.table(f"{SRC_DB}.t_model")
model_ver_raw = spark.table(f"{SRC_DB}.t_model_ver")
anlt_raw = spark.table(f"{SRC_DB}.t_model_ver_anlt_dtl")

model_current = take_latest(model_raw, ["model_sid"], "start_dt").select(
    "model_sid", "model_name", "model_code"
)
model_ver_current = take_latest(model_ver_raw, ["model_sid"], "start_dt").select(
    "model_sid", "model_ver_sid", "model_ver_signfcnt_ctgry_code"
)
version_base = model_ver_current.alias("v").join(
    model_current.alias("m"), F.col("v.model_sid") == F.col("m.model_sid"), "inner"
).select(
    F.col("v.model_ver_sid"), F.col("v.model_sid"),
    F.col("m.model_name"), F.col("m.model_code"),
    F.col("v.model_ver_signfcnt_ctgry_code"),
)
version_with_status = version_base.alias("v").join(
    anlt_raw.alias("a"), F.col("v.model_ver_sid") == F.col("a.model_ver_sid"), "left"
).select(
    "v.*",
    F.col("a.model_ver_stts_name"), F.col("a.model_stts_name"),
    F.col("a.model_ver_prom_expl_flag"),
    F.col("a.start_dt").alias("__anlt_start_dt"),
)
version_context = (
    take_latest(version_with_status, ["model_ver_sid"], "__anlt_start_dt")
    .drop("__anlt_start_dt")
    .filter(F.col("model_ver_prom_expl_flag").cast("boolean") == F.lit(True))
    .select(
        F.col("model_ver_sid").cast("string").alias("model_ver_sid"),
        F.col("model_sid").cast("string").alias("model_sid"),
        "model_name", "model_code", "model_ver_signfcnt_ctgry_code",
        F.col("model_ver_stts_name").alias("current_model_ver_status"),
        F.col("model_stts_name").alias("current_model_status"),
    )
)


# =============================================================================
# 8. EXCEL №29: только наблюдаемые записи, не исторические попытки.
# T_VALID_HIST недоступна. Нельзя восстановить изменения внутри VALID_SID.
# =============================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW validation_attempt_stats AS
WITH attempts AS (
    SELECT
        CAST(model_ver_sid AS STRING) model_ver_sid,
        CAST(valid_sid AS STRING) valid_sid,
        COALESCE(valid_start_fact_dttm,valid_crtn_dttm) attempt_start,
        valid_end_fact_dttm attempt_end,
        IF(NOT {blank('valid_dev_return_reason_txt')},1,0) has_return_reason,
        IF(UPPER(COALESCE(valid_stts_name,'')) RLIKE '{NEGATIVE_VALID_STATUS_RE}'
           OR UPPER(COALESCE(valid_rslt_name,'')) RLIKE '{RED_YELLOW_RE}',1,0) negative_result
    FROM valid
    WHERE model_ver_sid IS NOT NULL AND valid_sid IS NOT NULL
), ordered AS (
    SELECT *, LEAD(attempt_start) OVER (
        PARTITION BY model_ver_sid ORDER BY attempt_start,valid_sid
    ) next_attempt_start
    FROM attempts
), intervals AS (
    SELECT *, CASE
        WHEN has_return_reason=1 AND attempt_end IS NOT NULL
         AND next_attempt_start >= attempt_end
        THEN (UNIX_TIMESTAMP(next_attempt_start)-UNIX_TIMESTAMP(attempt_end))/86400.0
        ELSE NULL
    END rework_days
    FROM ordered
)
SELECT model_ver_sid,
       COUNT(DISTINCT valid_sid) available_validation_records,
       SUM(has_return_reason) available_return_records,
       SUM(negative_result) negative_validation_records,
       COUNT(rework_days) confirmed_rework_intervals,
       ROUND(SUM(rework_days),2) confirmed_rework_days
FROM intervals GROUP BY model_ver_sid
""")
validation_stats = spark.table("validation_attempt_stats")

# Обогащение листов сроков: поля взяты из T_MODEL_VER.
DATE_FIELDS = {
    "model_ver_dev_start_plan_dttm": "Начало разработки — план",
    "model_ver_dev_start_fact_dttm": "Начало разработки — факт",
    "model_ver_dev_end_plan_dttm": "Окончание разработки — план",
    "model_ver_dev_end_fact_dttm": "Окончание разработки — факт",
}
DATE_BY_CHECK = {
    1: ["model_ver_dev_start_plan_dttm", "model_ver_dev_end_plan_dttm"],
    2: ["model_ver_dev_start_fact_dttm", "model_ver_dev_end_fact_dttm"],
    3: ["model_ver_dev_start_fact_dttm", "model_ver_dev_end_fact_dttm"],
    4: ["model_ver_dev_start_fact_dttm", "model_ver_dev_end_fact_dttm"],
    5: ["model_ver_dev_start_fact_dttm", "model_ver_dev_end_fact_dttm"],
    6: ["model_ver_dev_start_plan_dttm", "model_ver_dev_start_fact_dttm"],
    7: ["model_ver_dev_start_plan_dttm", "model_ver_dev_end_plan_dttm"],
    8: ["model_ver_dev_end_plan_dttm", "model_ver_dev_end_fact_dttm"],
    9: list(DATE_FIELDS),
}
def days_between(end,start):
    return F.round((F.unix_timestamp(end)-F.unix_timestamp(start))/86400.0,2)

dev_dates = (spark.table("mv")
    .select(F.col("model_ver_sid").cast("string").alias("model_ver_sid"),
            *DATE_FIELDS)
    .withColumn("Отклонение начала, дней",days_between(
        "model_ver_dev_start_fact_dttm","model_ver_dev_start_plan_dttm"))
    .withColumn("Отклонение окончания, дней",days_between(
        "model_ver_dev_end_fact_dttm","model_ver_dev_end_plan_dttm"))
    .withColumn("Плановая длительность, дней",days_between(
        "model_ver_dev_end_plan_dttm","model_ver_dev_start_plan_dttm"))
    .withColumn("Фактическая длительность, дней",days_between(
        "model_ver_dev_end_fact_dttm","model_ver_dev_start_fact_dttm")))

valid_dates = spark.table("valid").select(
    F.col("valid_sid").cast("string").alias("_valid_sid"),
    F.col("model_ver_sid").cast("string").alias("_valid_model_ver_sid"),
    F.col("valid_crtn_dttm").alias("Создание валидации"),
    F.col("valid_start_fact_dttm").alias("Начало валидации — факт"),
    F.col("valid_end_fact_dttm").alias("Окончание валидации — факт"),
    F.col("valid_stts_name").alias("Статус валидации"),
    F.col("valid_rslt_name").alias("Результат валидации"),
    F.col("valid_dev_return_reason_txt").alias("Причина возврата"),
    F.col("valid_dev_return_reason_validator_cmnt_txt").alias("Комментарий валидатора"),
    F.col("valid_dev_return_reason_developer_cmnt_txt").alias("Комментарий разработчика"),
)


# =============================================================================
# =============================================================================
# 9. ОБОГАЩЕНИЕ ДЕТАЛЕЙ №9, №17, №18
# =============================================================================

# №9: ожидаемые дни = P90 подразделения; фактические = длительность версии.
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW check9_values AS
WITH d AS (
    SELECT
        CAST(model_ver_sid AS STRING) model_ver_sid,
        {clean_dim('model_ver_dev_dprtmt_name')} department,
        ROUND((UNIX_TIMESTAMP(model_ver_dev_end_fact_dttm)-
               UNIX_TIMESTAMP(model_ver_dev_start_fact_dttm))/86400.0,2) actual_days
    FROM mv_active
    WHERE {real_date('model_ver_dev_start_fact_dttm')}
      AND {real_date('model_ver_dev_end_fact_dttm')}
), s AS (
    SELECT department,
           PERCENTILE_APPROX(actual_days,0.90) expected_days_p90
    FROM d
    WHERE actual_days>=0
    GROUP BY department
    HAVING COUNT(*)>={MIN_GROUP_FOR_P90}
)
SELECT d.model_ver_sid,
       ROUND(s.expected_days_p90,2) expected_days_p90,
       d.actual_days
FROM d JOIN s ON d.department=s.department
WHERE d.actual_days>=0
""")

# №18: число повторов одной нормализованной метрики в выборке.
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW check18_values AS
SELECT
    CAST(s.resolved_model_ver_sid AS STRING) model_ver_sid,
    CONCAT(CAST(s.sample_data_sid AS STRING),':',
           UPPER(TRIM(CAST(m.metric_name AS STRING)))) source_entity_sid,
    COUNT(*) duplicate_count
FROM metric m
JOIN sample_model_map s ON m.sample_data_sid=s.sample_data_sid
WHERE NOT {blank('m.metric_name')}
GROUP BY s.resolved_model_ver_sid, s.sample_data_sid,
         UPPER(TRIM(CAST(m.metric_name AS STRING)))
HAVING COUNT(*)>1
""")


# =============================================================================
# 10. EXCEL: СВОД + 29 ЛИСТОВ
# =============================================================================

out_detail = (
    spark.table(OUT_DETAIL_TABLE)
    .filter(F.col("query_id").between(1,28))
    .withColumn("model_ver_sid", F.col("model_ver_sid").cast("string"))
    .join(version_context.select("model_ver_sid", "model_name", "model_code"), "model_ver_sid", "inner")
)

flags = (
    out_detail.groupBy("model_ver_sid")
    .pivot("query_id", list(range(1,29)))
    .agg(F.max(F.lit(1)))
)

summary_excel = version_context.alias("v").join(
    flags.alias("f"), "model_ver_sid", "left"
).join(
    validation_stats.alias("h"), "model_ver_sid", "left"
).select(
    F.col("v.model_name").alias("Модель"),
    F.col("v.model_code").alias("Код модели"),
    F.col("model_ver_sid").alias("Последняя актуальная версия"),
    *[
        F.when(F.col(f"f.`{i}`")==1, F.lit("Есть")).otherwise(F.lit("Нет")).alias(f"Проверка {i:02d}")
        for i in range(1,29)
    ],
    F.col("h.available_validation_records").alias("Доступных записей валидации"),
    F.col("h.available_return_records").alias("Записей с причиной возврата"),
    F.col("h.negative_validation_records").alias("Записей с негативным результатом"),
    F.col("h.confirmed_rework_intervals").alias("Подтверждённых интервалов доработки"),
    F.col("h.confirmed_rework_days").alias("Подтверждённых дней доработки"),
).orderBy("Модель")

# Метаданные 1–28 берём из уже рассчитанной таблицы.
meta = (
    spark.table(OUT_TABLE)
    .filter(F.col("query_id").between(1,28))
    .select("query_id", "query_name", "interpretation")
    .dropDuplicates(["query_id"])
    .toPandas().set_index("query_id")
)

base_detail = out_detail.select(
    "query_id", "model_name", "model_code", "model_ver_sid",
    "source_entity_type", "source_entity_sid", "issue_name",
    "dim_1_value", "dim_2_value"
)

# №9
c9 = spark.table("check9_values").withColumn("model_ver_sid",F.col("model_ver_sid").cast("string"))
# №17: возвращаем исходное значение метрики.
m17 = metric.select(
    F.col("metric_sid").cast("string").alias("metric_sid"),
    F.col("metric_name").alias("metric_name_raw"),
    F.col("metric_val").cast("string").alias("metric_value_raw")
)
# №18
c18 = spark.table("check18_values").withColumn("model_ver_sid",F.col("model_ver_sid").cast("string"))

PASS = {
 1:"Плановое начало ≤ планового окончания.", 2:"Фактическое окончание ≥ фактического начала.",
 3:"Фактические даты разработки не находятся в будущем.",
 4:"При DEVELOPMENT фактическая дата окончания отсутствует.",
 5:"На постразработочной стадии фактическая дата окончания заполнена.",
 6:"Фактический старт ≤ планового старта.", 7:"Для открытой разработки плановое окончание заполнено.",
 8:"Фактическое окончание ≤ планового окончания.",
 9:f"Фактическая длительность ≤ P90 подразделения; P90 рассчитывается при n ≥ {MIN_GROUP_FOR_P90}.",
 10:"После начала разработки тип входных данных заполнен.",
 11:"После начала разработки метод моделирования заполнен.",
 12:"До передачи на валидацию категория значимости заполнена.",
 13:"Подразделение-владелец заполнено.", 14:"После начала разработки подразделение-разработчик заполнено.",
 15:"При llm_flag=true описание применения LLM заполнено.",
 16:"У тестовой/валидационной выборки есть метрика либо явно установлен запрет её расчёта.",
 17:"Заполненное значение метрики приводится к числу.",
 18:"Одна и та же метрика встречается в одной выборке не более одного раза.",
 19:"Перед валидацией идентификатор отчёта о разработке заполнен.",
 20:"Идентификатор отчёта соответствует UUID или списку UUID через запятую.",
 21:"При prevalid_flag=true результат превалидации заполнен.",
 22:"При заполненном FeatureStore SID флаг использования FeatureStore = true.",
 23:"Нет возврата валидации из-за отсутствия данных.",
 24:"Нет возврата валидации из-за отсутствия доступа к данным.",
 25:"Нет возврата из-за недостаточной информации для формирования валидационных выборок.",
 26:"У версии не более одного отрицательного/красного завершённого результата валидации.",
 27:"Нет завершённых валидаций с красным или жёлтым результатом.",
 28:"Нет завершённой валидации с возвратом либо красным/жёлтым результатом.",
 29:"Информационный показатель; порог прохождения не применяется."
}

summary_pdf = summary_excel.toPandas()

with pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter", datetime_format="dd.mm.yyyy hh:mm", date_format="dd.mm.yyyy") as writer:
    summary_pdf.to_excel(writer, "Свод", index=False, startrow=2)
    wb = writer.book
    head = wb.add_format({"bold":True,"border":1,"text_wrap":True,"valign":"top"})
    dt_fmt = wb.add_format({"num_format":"dd.mm.yyyy hh:mm"})
    bold = wb.add_format({"bold":True})
    wrap = wb.add_format({"text_wrap":True,"valign":"top"})

    ws = writer.sheets["Свод"]
    ws.write(0,0,"Последние актуальные версии моделей в эксплуатации. Проверки 01–28: наличие нарушения; №29: доступные записи T_VALID, не полная история попыток.",wrap)
    for j,c in enumerate(summary_pdf.columns): ws.write(2,j,c,head)
    ws.freeze_panes(3,3); ws.autofilter(2,0,len(summary_pdf)+2,len(summary_pdf.columns)-1)
    ws.set_column(0,0,35); ws.set_column(1,2,24); ws.set_column(3,len(summary_pdf.columns)-1,15)

    for i in range(1,29):
        d = base_detail.filter(F.col("query_id")==i)
        if i in DATE_BY_CHECK:
            d = d.join(dev_dates,"model_ver_sid","left")
            extra = [F.col(c).alias(DATE_FIELDS[c]) for c in DATE_BY_CHECK[i]]
            if i == 6: extra += [F.col("Отклонение начала, дней")]
            if i == 8: extra += [F.col("Отклонение окончания, дней")]
            if i == 9:
                d = d.join(c9,"model_ver_sid","left")
                extra += [F.col("Плановая длительность, дней"),
                          F.col("Фактическая длительность, дней"),
                          F.col("Отклонение окончания, дней"),
                          F.col("expected_days_p90").alias("Ожидаемая длительность, дней (P90)"),
                          F.col("actual_days").alias("Длительность по проверке, дней")]
            d = d.select("model_name","model_code","model_ver_sid",
                "source_entity_type","source_entity_sid","issue_name",*extra)
        elif i==17:
            d = d.join(m17,F.col("source_entity_sid")==F.col("metric_sid"),"left").select(
                "model_name","model_code","model_ver_sid","source_entity_type","source_entity_sid","issue_name",
                F.col("metric_name_raw").alias("Метрика"), F.col("metric_value_raw").alias("Значение метрики")
            )
        elif i==18:
            d = d.join(c18,["model_ver_sid","source_entity_sid"],"left").select(
                "model_name","model_code","model_ver_sid","source_entity_type","source_entity_sid","issue_name",
                F.col("dim_1_value").alias("Метрика"), F.col("duplicate_count").alias("Количество повторов")
            )
        elif i in (23,24,25):
            d = d.join(valid_dates,
                (F.col("source_entity_sid")==F.col("_valid_sid")) &
                (F.col("model_ver_sid")==F.col("_valid_model_ver_sid")),"left"
            ).select("model_name","model_code","model_ver_sid",
                "source_entity_type","source_entity_sid","issue_name",
                "Создание валидации","Начало валидации — факт",
                "Окончание валидации — факт","Статус валидации",
                "Результат валидации","Причина возврата",
                "Комментарий валидатора","Комментарий разработчика")
        else:
            d = d.select(
                "model_name","model_code","model_ver_sid","source_entity_type","source_entity_sid","issue_name",
                F.col("dim_1_value").alias("Показатель 1"), F.col("dim_2_value").alias("Показатель 2")
            )

        pdf = d.orderBy("model_name","model_ver_sid").toPandas()
        pdf.columns = ["Модель","Код модели","Версия","Тип объекта","SID объекта","Нарушение",*list(pdf.columns[6:])]
        sheet=f"Проверка_{i:02d}"
        pdf.to_excel(writer,sheet,index=False,startrow=4)
        ws=writer.sheets[sheet]
        ws.write(0,0,"Показатель",bold); ws.write(0,1,str(meta.loc[i,"query_name"]),wrap)
        ws.write(1,0,"Как определяется",bold); ws.write(1,1,str(meta.loc[i,"interpretation"]),wrap)
        ws.write(2,0,"Порог прохождения",bold); ws.write(2,1,PASS[i],wrap)
        for j,c in enumerate(pdf.columns): ws.write(4,j,c,head)
        ws.freeze_panes(5,3)
        if len(pdf): ws.autofilter(4,0,len(pdf)+4,len(pdf.columns)-1)
        ws.set_column(0,0,35); ws.set_column(1,2,24); ws.set_column(3,len(pdf.columns)-1,28)
        for j, colname in enumerate(pdf.columns):
            if colname in DATE_FIELDS.values() or colname in (
                "Создание валидации", "Начало валидации — факт", "Окончание валидации — факт"
            ):
                ws.set_column(j,j,25,dt_fmt)
        if i in (23,24,25):
            ws.write(3,1,"Окончание валидации не равно подтверждённой дате возврата; дата события возврата в T_VALID отсутствует.",wrap)

    # №29 — доступные записи, а не полная история событий валидации.
    d29 = version_context.alias("v").join(validation_stats.alias("h"),"model_ver_sid","left").select(
        F.col("v.model_name").alias("Модель"), F.col("v.model_code").alias("Код модели"),
        F.col("model_ver_sid").alias("Версия"),
        F.col("h.available_validation_records").alias("Доступных записей валидации"),
        F.col("h.available_return_records").alias("Записей с причиной возврата"),
        F.col("h.negative_validation_records").alias("Записей с негативным результатом"),
        F.col("h.confirmed_rework_intervals").alias("Подтверждённых интервалов доработки"),
        F.col("h.confirmed_rework_days").alias("Подтверждённых дней доработки")
    ).orderBy("Модель").toPandas()
    d29.to_excel(writer,"Проверка_29",index=False,startrow=4)
    ws=writer.sheets["Проверка_29"]
    ws.write(0,0,"Показатель",bold)
    ws.write(0,1,"Доступные записи валидации и подтверждённые интервалы доработки",wrap)
    ws.write(1,0,"Как определяется",bold)
    ws.write(1,1,
        "Число записей = уникальные VALID_SID в T_VALID; это не количество исторических "
        "попыток. Возвраты = записи с заполненной причиной. Дни = только наблюдаемые "
        "закрытые интервалы от окончания возвращённой валидации до начала следующей "
        "валидации той же версии. Без T_VALID_HIST изменения одного VALID_SID и "
        "полная длительность доработок не восстанавливаются. NULL не заменяется нулём.",wrap)
    ws.write(2,0,"Порог прохождения",bold); ws.write(2,1,PASS[29],wrap)
    for j,c in enumerate(d29.columns): ws.write(4,j,c,head)
    ws.freeze_panes(5,3)
    if len(d29): ws.autofilter(4,0,len(d29)+4,len(d29.columns)-1)
    ws.set_column(0,0,35); ws.set_column(1,2,24); ws.set_column(3,len(d29.columns)-1,28)

for frame in cached:
    frame.unpersist()

print(f"Готово: {OUT_TABLE}")
print(f"Готово: {OUT_DETAIL_TABLE}")
print(f"Готово: {EXCEL_PATH}")
