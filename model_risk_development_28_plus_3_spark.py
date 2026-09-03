"""Контроли модельного риска на этапе разработки: 28 рабочих + 3 динамических.

Результаты:
  arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis
  arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis_details

Скрипт использует только модельный контур разработки и ретроспективные
результаты валидации. Мониторинг, внедрение и GenAI в расчёт не входят.
Три контроля без фактического периметра возвращают NOT_ASSESSABLE; если данные
появятся, они автоматически начнут рассчитываться.
"""

import os
import sys
from functools import reduce
from typing import List

os.environ["SPARK_MAJOR_VERSION"] = "3.5.1"
os.environ["SPARK_HOME"] = "/usr/sdp/current/spark3.5.1-client/"
os.environ["PYSPARK_DRIVER"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

sys.path.insert(0, "/usr/sdp/current/spark3.5.1-client/python/")
sys.path.insert(0, "/usr/sdp/current/spark3.5.1-client/python/lib/py4j-0.10.9.7-src.zip")

from pyspark import SparkConf, StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window, functions as F


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
UUID_LIST_RE = rf"^{UUID}(\\s*,\\s*{UUID})*$"


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
            "valid_end_fact_dttm", "valid_dev_return_reason_txt", "valid_prblm_name",
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
# 6. СБОРКА, КОНТРОЛЬ КАЧЕСТВА И КОМПАКТНЫЙ АНАЛИЗ
# =============================================================================

summary_raw = reduce(lambda a, b: a.unionByName(b), summary_parts)
placeholder_window = Window.partitionBy("query_id")
final_df = (
    summary_raw
    .withColumn("_min_placeholder", F.min("_placeholder").over(placeholder_window))
    .filter(F.col("_placeholder") == F.col("_min_placeholder"))
    .drop("_placeholder", "_min_placeholder")
    .persist(StorageLevel.MEMORY_AND_DISK)
)

detail_df = (
    reduce(lambda a, b: a.unionByName(b), detail_parts)
    .dropDuplicates()
    .persist(StorageLevel.MEMORY_AND_DISK)
)

control_count = final_df.select("query_id").distinct().count()
if control_count != 31:
    raise RuntimeError(f"Ожидался 31 контроль, сформировано: {control_count}")

bad_types = final_df.filter(F.col("entity_type") != "MODEL").limit(1).count()
bad_ratios = final_df.filter(
    (F.col("numerator") < 0)
    | (F.col("denominator") < 0)
    | (F.col("numerator") > F.col("denominator"))
).limit(1).count()
bad_details = detail_df.filter(
    F.col("model_ver_sid").isNull()
    | F.lower(F.trim(F.col("model_ver_sid"))).isin("", "null", "none")
).limit(1).count()
if bad_types or bad_ratios or bad_details:
    raise RuntimeError(
        f"Нарушен контроль качества: entity={bad_types}, ratios={bad_ratios}, details={bad_details}"
    )

control_rollup = (
    final_df.groupBy(
        "query_id", "source_check_id", "query_name", "risk_domain",
        "severity", "control_status", "status_reason",
    )
    .agg(
        F.sum("numerator").cast("long").alias("affected_records"),
        F.sum("denominator").cast("long").alias("applicable_records"),
    )
    .withColumn(
        "risk_rate_pct",
        F.when(
            F.col("applicable_records") > 0,
            F.round(100.0 * F.col("affected_records") / F.col("applicable_records"), 2),
        ).cast("double"),
    )
)

version_rollup = (
    detail_df.groupBy("model_ver_sid")
    .agg(
        F.countDistinct("query_id").alias("failed_controls"),
        F.sum(F.when(F.col("severity") == "HIGH", 1).otherwise(0)).alias("high_issue_rows"),
        F.concat_ws(",", F.sort_array(F.collect_set(F.format_string("%02d", F.col("query_id"))))).alias("control_ids"),
    )
    .orderBy(F.desc("failed_controls"), F.desc("high_issue_rows"), "model_ver_sid")
)

issue_rollup = (
    detail_df.groupBy("query_id", "query_name", "issue_name")
    .agg(F.countDistinct("source_entity_sid").alias("affected_entities"))
    .orderBy(F.desc("affected_entities"), "query_id")
)

print("\n=== СТАТУС 31 КОНТРОЛЯ ===")
control_rollup.orderBy("query_id").show(SHOW_ROWS, truncate=False)

print("\n=== ТОП КОНТРОЛЕЙ ПО ЧИСЛУ ПРОБЛЕМ ===")
control_rollup.filter(F.col("control_status") == "ASSESSABLE").orderBy(
    F.desc("affected_records"), F.desc("risk_rate_pct"), "query_id"
).show(20, truncate=False)

print("\n=== ТОП КОНКРЕТНЫХ НЕДОСТАТКОВ ===")
issue_rollup.show(30, truncate=False)

print("\n=== ТОП ВЕРСИЙ ПО КОЛИЧЕСТВУ НАРУШЕННЫХ КОНТРОЛЕЙ ===")
version_rollup.show(30, truncate=False)

print("\n=== ПРИМЕР АГРЕГАТОВ ===")
final_df.orderBy("query_id", F.desc("metric_value")).show(SHOW_ROWS, truncate=False)

print("\n=== ПРИМЕР ДЕТАЛИЗАЦИИ ===")
detail_df.orderBy("query_id", "model_ver_sid").show(SHOW_ROWS, truncate=False)

(
    final_df.repartition(1)
    .write.mode("overwrite")
    .format("parquet")
    .saveAsTable(OUT_TABLE)
)

(
    detail_df.repartition(4, "query_id")
    .write.mode("overwrite")
    .format("parquet")
    .saveAsTable(OUT_DETAIL_TABLE)
)

final_df.unpersist()
detail_df.unpersist()
for frame in cached:
    frame.unpersist()

print(f"Готово: {OUT_TABLE}")
print(f"Готово: {OUT_DETAIL_TABLE}")
