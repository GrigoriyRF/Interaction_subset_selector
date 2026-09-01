"""50 экономных агрегатов по рискам моделей и GenAI-агентов.

Результат:
arnsdpsbx_t_team_ova_sva_2.pri_model_lib_analysis

Скрипт читает только текущие таблицы и только используемые столбцы,
агрегирует данные и не сохраняет персональные/текстовые детали.
"""

from functools import reduce
from typing import List

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F


SRC_DB = "prx_pri_custom_ris_l_library_risk_model_library"
OUT_TABLE = "arnsdpsbx_t_team_ova_sva_2.pri_model_lib_analysis"

OPEN_SLA_DAYS = 90
STALE_MONITORING_DAYS = 365
CRITICAL_SIGNIFICANCE = ("A", "B")
SHOW_ROWS = 300

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# Закрывающие статусы задаются regex, чтобы покрыть русские и английские коды.
CLOSED_STATUS_RE = r"DONE|COMPLETE|COMPLETED|CLOSED|CANCEL|CANCELED|REJECT|ARCHIV|ЗАВЕРШ|ЗАКРЫТ|ОТМЕН|ОТКЛОН"
RED_RESULT_RE = r"RED|КРАСН"
OOT_RE = r"OUT[-_ ]?OF[-_ ]?TIME|\bOOT\b|ВНЕВРЕМ"


def _ts(name: str):
    """Разбирает timestamp/date и распространённые строковые форматы."""
    raw = F.col(name)
    s = raw.cast("string")
    return F.coalesce(
        raw.cast("timestamp"),
        F.to_timestamp(s, "dd.MM.yyyy H:mm:ss"),
        F.to_timestamp(s, "dd.MM.yyyy HH:mm:ss"),
        F.to_timestamp(s, "yyyy-MM-dd HH:mm:ss.SSSSSS"),
        F.to_timestamp(s, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(s, "yyyy-MM-dd"),
    )


def load_current(table: str, columns: List[str]) -> DataFrame:
    """Читает актуальные строки и проецирует только нужные поля."""
    df = spark.table(f"{SRC_DB}.{table}")
    existing = set(df.columns)
    missing = sorted(set(columns) - existing)
    if missing:
        raise ValueError(f"{table}: отсутствуют ожидаемые поля: {missing}")

    if "CTL_ACTION" in existing:
        df = df.filter(
            F.col("CTL_ACTION").isNull()
            | (F.upper(F.col("CTL_ACTION").cast("string")) != "D")
        )
    if "START_DT" in existing:
        df = df.filter(_ts("START_DT").isNull() | (F.to_date(_ts("START_DT")) <= F.current_date()))
    if "END_DT" in existing:
        df = df.filter(_ts("END_DT").isNull() | (F.to_date(_ts("END_DT")) >= F.current_date()))

    exprs = []
    for c in columns:
        if c.endswith("_DTTM") or c in {"START_DT", "END_DT"}:
            exprs.append(_ts(c).alias(c))
        else:
            exprs.append(F.col(c))
    return df.select(*exprs)


VIEWS = {
    "model": ("T_MODEL", [
        "MODEL_SID", "MODEL_TYPE_NAME", "MODEL_ML_TASK_NAME", "MODEL_RSK_FLAG",
        "MODEL_RSK_TYPE_NAME", "MODEL_RSK_SGMNT_NAME", "MODEL_STTS_NAME",
        "MODEL_DEV_DPRTMT_NAME",
    ]),
    "mv": ("T_MODEL_VER", [
        "MODEL_VER_SID", "MODEL_SID", "MODEL_VER_STTS_NAME",
        "MODEL_VER_SIGNFCNT_CTGRY_CODE", "MODEL_VER_OWNER_DPRTMT_NAME",
        "MODEL_VER_DEV_DPRTMT_NAME", "MODEL_VER_METHOD_NAME", "MODEL_VER_TGT_TXT",
        "MODEL_VER_DATA_TYPE_NAME", "MODEL_VER_REPSTRY_LINK_TXT",
        "MODEL_VER_REPSTRY_COMMIT_SID", "MODEL_VER_DEV_REPORT_SID",
        "MODEL_VER_DATA_MART_LINK_TXT", "MODEL_VER_DEV_START_PLAN_DTTM",
        "MODEL_VER_DEV_START_FACT_DTTM", "MODEL_VER_DEV_END_PLAN_DTTM",
        "MODEL_VER_DEV_END_FACT_DTTM", "MODEL_VER_PREVALID_FLAG",
        "MODEL_VER_PREVALID_RSLT_NAME", "MODEL_VER_SOTA_FLAG",
        "MODEL_VER_PROJ_FEATURE_FLAG", "MODEL_VER_PROJ_FEATURE_SID",
        "MODEL_VER_DEV_SRC_NAME", "MODEL_VER_LLM_FLAG", "MODEL_VER_LLM_DESCR_TXT",
        "MODEL_VER_MONTRG_AUTO_FLAG", "MODEL_VER_MONTRG_AUTO_EXCLUDE_REASON_ARRAY",
        "MODEL_VER_CRTN_DTTM",
    ]),
    "sample": ("T_SAMPLE_DATA", [
        "SAMPLE_DATA_SID", "MODEL_VER_SID", "VALID_SID", "MONTRG_AUTO_RSLT_SID",
        "MONTRG_MANUAL_RSLT_SID", "SAMPLE_DATA_PROP_ARRAY", "SAMPLE_DATA_TYPE_NAME",
        "SAMPLE_DATA_STTS_NAME", "SAMPLE_DATA_NOT_METRIC_CALC_FLAG",
    ]),
    "metric": ("T_METRIC", [
        "METRIC_SID", "SAMPLE_DATA_SID", "METRIC_NAME", "METRIC_VAL", "METRIC_STTS_NAME",
    ]),
    "valid": ("T_VALID", [
        "VALID_SID", "MODEL_VER_SID", "VALID_CRTN_DTTM", "VALID_START_FACT_DTTM",
        "VALID_END_FACT_DTTM", "VALID_FREQ_START_PLAN_DTTM", "VALID_STTS_NAME",
        "VALID_RSLT_NAME", "VALID_DPRTMT_NAME", "VALID_TYPE_NAME",
        "VALID_DEV_RETURN_REASON_TXT", "VALID_NOT_SAMPLE_DATA_FLAG",
        "VALID_SRC_NOT_TABLE_FLAG", "VALID_REPORT_SID", "VALID_PRBLM_NAME",
        "VALID_ALT_FLAG", "VALID_ALT_METRIC_NAME", "VALID_ALT_METRIC_VAL",
        "VALID_RED_ZONE_OWNER_APRVL_RSK_FLAG", "VALID_RED_ZONE_COMT_RSK_DECSN_SID",
        "VALID_RED_ZONE_COMT_RSK_DECSN_LINK_TXT", "VALID_AGENT_USG_FLAG",
        "VALID_AGENT_TEST_QUALITY_CORR_PCT", "VALID_AGENT_TEST_QUALITY_REUSE_PCT",
        "VALID_AGENT_TEST_QUANTITY_REUSE_PCT", "VALID_AGENT_NON_USG_REASON_NAME",
        "VALID_PREVALID_REUSE_LVL_NAME", "VALID_PREVALID_REUSE_LVL_REASON_ARRAY",
    ]),
    "x_auto": ("T_MODEL_VER_X_MONTRG_AUTO", ["MODEL_VER_SID", "MONTRG_AUTO_SID"]),
    "auto": ("T_MONTRG_AUTO", [
        "MONTRG_AUTO_SID", "MONTRG_AUTO_STTS_NAME", "MONTRG_AUTO_NEXT_DTTM",
        "MONTRG_AUTO_PROJ_SCHED_STTS_NAME", "MONTRG_AUTO_PROJ_SCHED_CRON_TXT",
        "MONTRG_AUTO_FREQ_NAME", "MONTRG_AUTO_DPRTMT_NAME",
    ]),
    "auto_r": ("T_MONTRG_AUTO_RSLT", [
        "MONTRG_AUTO_RSLT_SID", "MONTRG_AUTO_SID", "MONTRG_AUTO_RSLT_NAME",
        "MONTRG_AUTO_RSLT_CRTN_DTTM", "MONTRG_AUTO_RSLT_END_DTTM",
        "MONTRG_AUTO_RSLT_STTS_NAME", "MONTRG_AUTO_RSLT_PRBLM_NAME",
        "MONTRG_AUTO_RSLT_METRIC_VAL", "MONTRG_AUTO_RSLT_NOT_METRIC_MAIN_FLAG",
        "MONTRG_AUTO_RSLT_VALID_NOT_SAMPLE_DATA_FLAG", "MONTRG_AUTO_RSLT_REPORT_SID",
    ]),
    "x_manual": ("T_MODEL_VER_X_MONTRG_MANUAL", ["MODEL_VER_SID", "MONTRG_MANUAL_SID"]),
    "manual": ("T_MONTRG_MANUAL", [
        "MONTRG_MANUAL_SID", "MONTRG_MANUAL_STTS_NAME", "MONTRG_MANUAL_NEXT_DTTM",
        "MONTRG_MANUAL_LAST_RSLT_NAME", "MONTRG_MANUAL_LAST_RSLT_DTTM",
        "MONTRG_MANUAL_DPRTMT_NAME",
    ]),
    "manual_r": ("T_MONTRG_MANUAL_RSLT", [
        "MONTRG_MANUAL_RSLT_SID", "MONTRG_MANUAL_SID", "MONTRG_MANUAL_RSLT_NAME",
        "MONTRG_MANUAL_RSLT_CRTN_DTTM", "MONTRG_MANUAL_RSLT_END_DTTM",
        "MONTRG_MANUAL_RSLT_STTS_NAME", "MONTRG_MANUAL_RSLT_PRBLM_NAME",
        "MONTRG_MANUAL_RSLT_METRIC_VAL", "MONTRG_MANUAL_RSLT_NOT_METRIC_MAIN_FLAG",
        "MONTRG_MANUAL_RSLT_VALID_NOT_SAMPLE_DATA_FLAG", "MONTRG_MANUAL_RSLT_REPORT_SID",
    ]),
    "mvp": ("T_MODEL_VER_PROM", [
        "MODEL_VER_PROM_SID", "MODEL_VER_SID", "MODEL_VER_PROM_STTS_NAME",
        "MODEL_VER_PROM_VALID_IT_RSLT_QGM_FLAG", "MODEL_VER_PROM_IMPLM_FLAG",
    ]),
    "valid_it": ("T_VALID_IT", [
        "VALID_IT_SID", "MODEL_VER_PROM_SID", "VALID_IT_STTS_NAME", "VALID_IT_RSLT_NAME",
        "VALID_IT_PRBLM_NAME", "VALID_IT_SRC_NOT_PROM_FLAG", "VALID_IT_REPORT_SID",
    ]),
    "x_pilot": ("T_MODEL_VER_PROM_X_PILOT_IMPLM", ["MODEL_VER_PROM_SID", "PILOT_IMPLM_SID"]),
    "pilot": ("T_PILOT_IMPLM", [
        "PILOT_IMPLM_SID", "PILOT_IMPLM_STTS_NAME", "PILOT_IMPLM_START_PLAN_DTTM",
        "PILOT_IMPLM_START_FACT_DTTM", "PILOT_IMPLM_END_FACT_DTTM",
    ]),
    "x_prom": ("T_MODEL_VER_PROM_X_PROM_IMPLM", ["MODEL_VER_PROM_SID", "PROM_IMPLM_SID"]),
    "prom": ("T_PROM_IMPLM", [
        "PROM_IMPLM_SID", "PROM_IMPLM_STTS_NAME", "PROM_IMPLM_START_PLAN_DTTM",
        "PROM_IMPLM_START_FACT_DTTM", "PROM_IMPLM_END_FACT_DTTM",
    ]),
    "genai": ("T_GENAI", [
        "GENAI_SID", "GENAI_STTS_NAME", "GENAI_AGENT_FLAG", "GENAI_PRIORITY_LVL_CODE",
        "GENAI_MATURITY_LVL_ORD", "GENAI_OWNER_DPRTMT_NAME",
    ]),
    "gv": ("T_GENAI_VER", [
        "GENAI_VER_SID", "GENAI_SID", "GENAI_VER_STTS_NAME",
        "GENAI_VER_SIGNFCNT_CTGRY_CODE", "GENAI_VER_REPSTRY_LINK_TXT",
        "GENAI_VER_REPSTRY_COMMIT_LINK_TXT", "GENAI_VER_RELEASE_LINK_TXT",
        "GENAI_VER_DEV_REPORT_SID", "GENAI_VER_SAMPLE_DATA_KNOWLEDGE_LVL_NAME",
        "GENAI_VER_MONTRG_AUTO_DATA_FLAG", "GENAI_VER_MONTRG_AUTO_MODEL_FLAG",
        "GENAI_VER_MONTRG_AUTO_EXCLUDE_REASON_DATA_ARRAY",
        "GENAI_VER_MONTRG_AUTO_EXCLUDE_REASON_MODEL_ARRAY",
        "GENAI_VER_MONTRG_LAST_RSLT_NAME", "GENAI_VER_MONTRG_LAST_RSLT_DTTM",
        "GENAI_VER_METRIC_KEY_CODE", "GENAI_VER_METRIC_KEY_VAL",
        "GENAI_VER_METRIC_ASSMNT_AUTO_CMNT_TXT", "GENAI_VER_USG_DAYS_QTY",
        "GENAI_VER_USG_SAME_TIME_CNT", "GENAI_VER_FIN_EFFECT_PLAN_MILLION_QTY",
        "GENAI_VER_FIN_EFFECT_FACT_MILLION_QTY",
    ]),
    "g_sample": ("T_GENAI_VER_SAMPLE_DATA", [
        "GENAI_VER_SAMPLE_DATA_SID", "GENAI_VER_SID", "GENAI_VER_SAMPLE_DATA_TYPE_NAME",
        "GENAI_VER_SAMPLE_DATA_KIND_NAME",
    ]),
    "g_valid": ("T_GENAI_VER_VALID", [
        "GENAI_VER_VALID_SID", "GENAI_VER_SID", "GENAI_VER_VALID_STTS_NAME",
        "GENAI_VER_VALID_RSLT_NAME", "GENAI_VER_VALID_CRTN_DTTM",
        "GENAI_VER_VALID_START_FACT_DTTM", "GENAI_VER_VALID_END_FACT_DTTM",
        "GENAI_VER_VALID_REPORT_SID", "GENAI_VER_VALID_ALT_FLAG",
        "GENAI_VER_VALID_ALT_METRIC_NAME", "GENAI_VER_VALID_ALT_METRIC_VAL",
        "GENAI_VER_VALID_METRIC_CHG_STAT_SIGNFCNT_FLAG",
    ]),
}


cached_frames: List[DataFrame] = []
for view_name, (table_name, columns) in VIEWS.items():
    frame = load_current(table_name, columns).persist(StorageLevel.MEMORY_AND_DISK)
    frame.createOrReplaceTempView(view_name)
    cached_frames.append(frame)

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW mv_scope AS
SELECT * FROM mv
WHERE UPPER(COALESCE(CAST(MODEL_VER_STTS_NAME AS STRING), ''))
      NOT RLIKE '{CLOSED_STATUS_RE}'
""")


results: List[DataFrame] = []


def add_query(
    query_id: int,
    query_name: str,
    risk_domain: str,
    entity_type: str,
    metric_name: str,
    dim_1_name: str,
    dim_2_name: str,
    source_tables: str,
    interpretation: str,
    sql_body: str,
) -> None:
    """sql_body возвращает d1, d2, metric_value, numerator, denominator."""
    raw = spark.sql(sql_body)
    out = raw.select(
        F.lit(query_id).cast("int").alias("query_id"),
        F.lit(query_name).alias("query_name"),
        F.lit(risk_domain).alias("risk_domain"),
        F.lit(entity_type).alias("entity_type"),
        F.lit(metric_name).alias("metric_name"),
        F.lit(dim_1_name).alias("dim_1_name"),
        F.coalesce(F.col("d1").cast("string"), F.lit("$NULL$")).alias("dim_1_value"),
        F.lit(dim_2_name).alias("dim_2_name"),
        F.coalesce(F.col("d2").cast("string"), F.lit("$ALL$")).alias("dim_2_value"),
        F.col("metric_value").cast("double").alias("metric_value"),
        F.col("numerator").cast("long").alias("numerator"),
        F.col("denominator").cast("long").alias("denominator"),
        F.current_date().alias("as_of_dt"),
        F.lit(source_tables).alias("source_tables"),
        F.lit(interpretation).alias("interpretation"),
    )
    results.append(out)


def dist(qid, name, domain, entity, metric, d1_name, d2_name, sources, note, from_sql, d1, d2="NULL"):
    add_query(qid, name, domain, entity, metric, d1_name, d2_name, sources, note, f"""
        SELECT CAST({d1} AS STRING) d1, CAST({d2} AS STRING) d2,
               CAST(COUNT(*) AS DOUBLE) metric_value,
               COUNT(*) numerator, CAST(NULL AS BIGINT) denominator
        FROM {from_sql}
        GROUP BY {d1}, {d2}
    """)


def gap(qid, name, domain, entity, d1_name, d2_name, sources, note, from_sql, condition, d1, d2="NULL"):
    add_query(qid, name, domain, entity, "risk_rate_pct", d1_name, d2_name, sources, note, f"""
        SELECT CAST({d1} AS STRING) d1, CAST({d2} AS STRING) d2,
               100.0 * SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) / COUNT(*) metric_value,
               SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) numerator,
               COUNT(*) denominator
        FROM {from_sql}
        GROUP BY {d1}, {d2}
    """)


sig = "COALESCE(MODEL_VER_SIGNFCNT_CTGRY_CODE, '$NULL$')"
gsig = "COALESCE(GENAI_VER_SIGNFCNT_CTGRY_CODE, '$NULL$')"
truth = lambda c: f"COALESCE(LOWER(CAST({c} AS STRING)) IN ('true','1','да','yes','y'), FALSE)"


# 1–15. Портфель, документация, данные и разработка
dist(1, "Портфель версий по значимости и статусу", "portfolio", "MODEL", "version_count",
     "significance", "status", "T_MODEL_VER", "Базовая структура активного портфеля.",
     "mv", sig, "COALESCE(MODEL_VER_STTS_NAME, '$NULL$')")

dist(2, "Модели по типу и ML-задаче", "portfolio", "MODEL", "version_count",
     "model_type", "ml_task", "T_MODEL,T_MODEL_VER", "Концентрация типов моделей и задач.",
     "mv_scope v JOIN model m ON v.MODEL_SID=m.MODEL_SID",
     "COALESCE(m.MODEL_TYPE_NAME, '$NULL$')", "COALESCE(m.MODEL_ML_TASK_NAME, '$NULL$')")

dist(3, "Риск-модели по виду и сегменту риска", "portfolio", "MODEL", "version_count",
     "risk_type", "risk_segment", "T_MODEL,T_MODEL_VER", "Профиль версий риск-моделей.",
     f"mv_scope v JOIN model m ON v.MODEL_SID=m.MODEL_SID WHERE {truth('m.MODEL_RSK_FLAG')}",
     "COALESCE(m.MODEL_RSK_TYPE_NAME, '$NULL$')", "COALESCE(m.MODEL_RSK_SGMNT_NAME, '$NULL$')")

dist(4, "Критичные версии по подразделению-владельцу", "portfolio", "MODEL", "version_count",
     "owner_department", "significance", "T_MODEL_VER", "Концентрация критичных A/B-версий по владельцам.",
     f"mv_scope WHERE MODEL_VER_SIGNFCNT_CTGRY_CODE IN {CRITICAL_SIGNIFICANCE}",
     "COALESCE(MODEL_VER_OWNER_DPRTMT_NAME, '$NULL$')", sig)

add_query(5, "Пробелы ключевых метаданных", "documentation", "MODEL", "risk_rate_pct",
          "missing_field", "significance", "T_MODEL_VER", "Доля активных версий с незаполненным ключевым полем.", f"""
    SELECT indicator d1, significance d2,
           100.0*SUM(is_gap)/COUNT(*) metric_value, SUM(is_gap) numerator, COUNT(*) denominator
    FROM (
      SELECT {sig} significance,
             STACK(4,
               'owner_department', CASE WHEN MODEL_VER_OWNER_DPRTMT_NAME IS NULL OR TRIM(CAST(MODEL_VER_OWNER_DPRTMT_NAME AS STRING))='' THEN 1 ELSE 0 END,
               'method', CASE WHEN MODEL_VER_METHOD_NAME IS NULL OR TRIM(CAST(MODEL_VER_METHOD_NAME AS STRING))='' THEN 1 ELSE 0 END,
               'target', CASE WHEN MODEL_VER_TGT_TXT IS NULL OR TRIM(CAST(MODEL_VER_TGT_TXT AS STRING))='' THEN 1 ELSE 0 END,
               'significance', CASE WHEN MODEL_VER_SIGNFCNT_CTGRY_CODE IS NULL OR TRIM(CAST(MODEL_VER_SIGNFCNT_CTGRY_CODE AS STRING))='' THEN 1 ELSE 0 END
             ) AS (indicator, is_gap)
      FROM mv_scope
    ) s GROUP BY indicator, significance
""")

gap(6, "Отсутствие ссылки на репозиторий", "reproducibility", "MODEL", "significance", "dev_source",
    "T_MODEL_VER", "Без ссылки нельзя воспроизвести код версии.", "mv_scope",
    "MODEL_VER_REPSTRY_LINK_TXT IS NULL OR TRIM(CAST(MODEL_VER_REPSTRY_LINK_TXT AS STRING))=''",
    sig, "COALESCE(MODEL_VER_DEV_SRC_NAME, '$NULL$')")

gap(7, "Отсутствие идентификатора коммита", "reproducibility", "MODEL", "significance", "dev_source",
    "T_MODEL_VER", "Без commit ID код версии не фиксирован однозначно.", "mv_scope",
    "MODEL_VER_REPSTRY_COMMIT_SID IS NULL OR TRIM(CAST(MODEL_VER_REPSTRY_COMMIT_SID AS STRING))=''",
    sig, "COALESCE(MODEL_VER_DEV_SRC_NAME, '$NULL$')")

gap(8, "Отсутствие отчёта о разработке", "documentation", "MODEL", "significance", "status",
    "T_MODEL_VER", "Проверяет наличие доказательной базы разработки.", "mv_scope",
    "MODEL_VER_DEV_REPORT_SID IS NULL OR TRIM(CAST(MODEL_VER_DEV_REPORT_SID AS STRING))=''",
    sig, "COALESCE(MODEL_VER_STTS_NAME, '$NULL$')")

add_query(9, "Отсутствие target или типа данных", "data_governance", "MODEL", "risk_rate_pct",
          "missing_field", "significance", "T_MODEL_VER", "Неполная постановка задачи или описание входных данных.", f"""
    SELECT indicator d1, significance d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator, COUNT(*) denominator
    FROM (
      SELECT {sig} significance,
             STACK(2,
               'target', CASE WHEN MODEL_VER_TGT_TXT IS NULL OR TRIM(CAST(MODEL_VER_TGT_TXT AS STRING))='' THEN 1 ELSE 0 END,
               'data_type', CASE WHEN MODEL_VER_DATA_TYPE_NAME IS NULL OR TRIM(CAST(MODEL_VER_DATA_TYPE_NAME AS STRING))='' THEN 1 ELSE 0 END
             ) AS (indicator, is_gap)
      FROM mv_scope
    ) s GROUP BY indicator, significance
""")

gap(10, "Версии без связанных выборок", "data_governance", "MODEL", "significance", "status",
    "T_MODEL_VER,T_SAMPLE_DATA", "Разрыв связи между версией модели и данными.",
    "mv_scope v LEFT JOIN (SELECT DISTINCT MODEL_VER_SID FROM sample WHERE MODEL_VER_SID IS NOT NULL) s ON v.MODEL_VER_SID=s.MODEL_VER_SID",
    "s.MODEL_VER_SID IS NULL", sig, "COALESCE(v.MODEL_VER_STTS_NAME, '$NULL$')")

gap(11, "Версии без OOT-выборки", "data_governance", "MODEL", "significance", "status",
    "T_MODEL_VER,T_SAMPLE_DATA", "Отсутствие OOT снижает доказанность временной устойчивости.", f"""
      mv_scope v LEFT JOIN (
        SELECT DISTINCT MODEL_VER_SID FROM sample
        WHERE UPPER(COALESCE(CAST(SAMPLE_DATA_PROP_ARRAY AS STRING),'')) RLIKE '{OOT_RE}'
      ) s ON v.MODEL_VER_SID=s.MODEL_VER_SID
    """, "s.MODEL_VER_SID IS NULL", sig, "COALESCE(v.MODEL_VER_STTS_NAME, '$NULL$')")

add_query(12, "Выборки без пригодных метрик", "data_governance", "MODEL", "risk_rate_pct",
          "sample_type", "gap_type", "T_SAMPLE_DATA,T_METRIC", "Выявляет выборки, для которых количественная оценка отсутствует или запрещена.", f"""
    WITH x AS (
      SELECT s.SAMPLE_DATA_SID, COALESCE(s.SAMPLE_DATA_TYPE_NAME,'$NULL$') sample_type,
             MAX(CASE WHEN m.METRIC_SID IS NOT NULL AND m.METRIC_VAL IS NOT NULL THEN 1 ELSE 0 END) has_metric,
             MAX(CASE WHEN {truth('s.SAMPLE_DATA_NOT_METRIC_CALC_FLAG')} THEN 1 ELSE 0 END) no_calc
      FROM sample s LEFT JOIN metric m ON s.SAMPLE_DATA_SID=m.SAMPLE_DATA_SID
      GROUP BY s.SAMPLE_DATA_SID, COALESCE(s.SAMPLE_DATA_TYPE_NAME,'$NULL$')
    ), y AS (
      SELECT sample_type, STACK(2, 'no_metric', 1-has_metric, 'metric_calc_disabled', no_calc) AS (gap_type,is_gap) FROM x
    )
    SELECT sample_type d1, gap_type d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator, COUNT(*) denominator FROM y GROUP BY sample_type,gap_type
""")

gap(13, "Просроченная разработка", "development", "MODEL", "department", "significance",
    "T_MODEL_VER", "Факт окончания позже плана либо открытая работа после плановой даты.", "mv_scope",
    "MODEL_VER_DEV_END_PLAN_DTTM IS NOT NULL AND ((MODEL_VER_DEV_END_FACT_DTTM IS NULL AND DATE(MODEL_VER_DEV_END_PLAN_DTTM)<CURRENT_DATE()) OR MODEL_VER_DEV_END_FACT_DTTM>MODEL_VER_DEV_END_PLAN_DTTM)",
    "COALESCE(MODEL_VER_DEV_DPRTMT_NAME,'$NULL$')", sig)

add_query(14, "Длительность разработки", "development", "MODEL", "duration_days",
          "department", "statistic", "T_MODEL_VER", "Медиана и p90 длительности завершённой разработки.", """
    WITH x AS (
      SELECT COALESCE(MODEL_VER_DEV_DPRTMT_NAME,'$NULL$') department,
             DATEDIFF(DATE(MODEL_VER_DEV_END_FACT_DTTM),DATE(MODEL_VER_DEV_START_FACT_DTTM)) days
      FROM mv WHERE MODEL_VER_DEV_START_FACT_DTTM IS NOT NULL AND MODEL_VER_DEV_END_FACT_DTTM IS NOT NULL
    ), stats AS (
      SELECT department,PERCENTILE_APPROX(days,0.50) p50,
             PERCENTILE_APPROX(days,0.90) p90,COUNT(*) n
      FROM x WHERE days>=0 GROUP BY department
    )
    SELECT department d1,'p50' d2,CAST(p50 AS DOUBLE) metric_value,
           CAST(NULL AS BIGINT) numerator,n denominator FROM stats
    UNION ALL
    SELECT department,'p90',CAST(p90 AS DOUBLE),CAST(NULL AS BIGINT),n FROM stats
""")

gap(15, "Неконсистентность FeatureStore", "data_governance", "MODEL", "significance", "status",
    "T_MODEL_VER", "Флаг FeatureStore установлен, но идентификатор проекта отсутствует.", "mv_scope",
    f"{truth('MODEL_VER_PROJ_FEATURE_FLAG')} AND MODEL_VER_PROJ_FEATURE_SID IS NULL",
    sig, "COALESCE(MODEL_VER_STTS_NAME,'$NULL$')")


# 16–32. Валидация
gap(16, "Покрытие валидацией", "validation", "MODEL", "significance", "status",
    "T_MODEL_VER,T_VALID", "Доля активных версий без единой валидации.",
    "mv_scope v LEFT JOIN (SELECT DISTINCT MODEL_VER_SID FROM valid) x ON v.MODEL_VER_SID=x.MODEL_VER_SID",
    "x.MODEL_VER_SID IS NULL", sig, "COALESCE(v.MODEL_VER_STTS_NAME,'$NULL$')")

dist(17, "Результаты валидации", "validation", "MODEL", "validation_count",
     "result", "significance", "T_VALID,T_MODEL_VER", "Распределение итогов валидации.",
     "valid v LEFT JOIN mv m ON v.MODEL_VER_SID=m.MODEL_VER_SID",
     "COALESCE(v.VALID_RSLT_NAME,'$NULL$')", "COALESCE(m.MODEL_VER_SIGNFCNT_CTGRY_CODE,'$NULL$')")

gap(18, "Возраст незавершённой валидации", "validation", "MODEL", "department", "status",
    "T_VALID", f"Доля открытых валидаций старше {OPEN_SLA_DAYS} дней.",
    f"valid WHERE UPPER(COALESCE(VALID_STTS_NAME,'')) NOT RLIKE '{CLOSED_STATUS_RE}'",
    f"DATEDIFF(CURRENT_DATE(),DATE(COALESCE(VALID_START_FACT_DTTM,VALID_CRTN_DTTM)))>{OPEN_SLA_DAYS}",
    "COALESCE(VALID_DPRTMT_NAME,'$NULL$')", "COALESCE(VALID_STTS_NAME,'$NULL$')")

gap(19, "Просроченная периодическая валидация", "validation", "MODEL", "department", "status",
    "T_VALID", "Плановая дата прошла, а валидация не закрыта.", "valid",
    f"VALID_FREQ_START_PLAN_DTTM IS NOT NULL AND DATE(VALID_FREQ_START_PLAN_DTTM)<CURRENT_DATE() AND UPPER(COALESCE(VALID_STTS_NAME,'')) NOT RLIKE '{CLOSED_STATUS_RE}'",
    "COALESCE(VALID_DPRTMT_NAME,'$NULL$')", "COALESCE(VALID_STTS_NAME,'$NULL$')")

add_query(20, "Длительность валидации", "validation", "MODEL", "duration_days",
          "department", "statistic", "T_VALID", "Медиана и p90 длительности завершённой валидации.", """
    WITH x AS (
      SELECT COALESCE(VALID_DPRTMT_NAME,'$NULL$') department,
             DATEDIFF(DATE(VALID_END_FACT_DTTM),DATE(VALID_START_FACT_DTTM)) days
      FROM valid WHERE VALID_START_FACT_DTTM IS NOT NULL AND VALID_END_FACT_DTTM IS NOT NULL
    ), stats AS (
      SELECT department,PERCENTILE_APPROX(days,0.50) p50,
             PERCENTILE_APPROX(days,0.90) p90,COUNT(*) n
      FROM x WHERE days>=0 GROUP BY department
    )
    SELECT department d1,'p50' d2,CAST(p50 AS DOUBLE) metric_value,
           CAST(NULL AS BIGINT) numerator,n denominator FROM stats
    UNION ALL
    SELECT department,'p90',CAST(p90 AS DOUBLE),CAST(NULL AS BIGINT),n FROM stats
""")

dist(21, "Возвраты разработчику", "validation", "MODEL", "return_count",
     "return_reason", "department", "T_VALID", "Частые причины возврата показывают системные дефекты разработки.",
     "valid WHERE VALID_DEV_RETURN_REASON_TXT IS NOT NULL AND TRIM(CAST(VALID_DEV_RETURN_REASON_TXT AS STRING))<>''",
     "VALID_DEV_RETURN_REASON_TXT", "COALESCE(VALID_DPRTMT_NAME,'$NULL$')")

gap(22, "Валидации без выборки", "validation", "MODEL", "result", "type",
    "T_VALID", "Доля валидаций, помеченных как проведённые без выборки.", "valid",
    truth("VALID_NOT_SAMPLE_DATA_FLAG"), "COALESCE(VALID_RSLT_NAME,'$NULL$')", "COALESCE(VALID_TYPE_NAME,'$NULL$')")

gap(23, "Нетабличные источники валидации", "validation", "MODEL", "result", "type",
    "T_VALID", "Нетабличные данные ограничивают воспроизводимость и автоматизацию.", "valid",
    truth("VALID_SRC_NOT_TABLE_FLAG"), "COALESCE(VALID_RSLT_NAME,'$NULL$')", "COALESCE(VALID_TYPE_NAME,'$NULL$')")

gap(24, "Валидации без отчёта", "validation", "MODEL", "result", "department",
    "T_VALID", "Результат не подкреплён ссылкой на отчёт.", "valid",
    "VALID_REPORT_SID IS NULL OR TRIM(CAST(VALID_REPORT_SID AS STRING))=''",
    "COALESCE(VALID_RSLT_NAME,'$NULL$')", "COALESCE(VALID_DPRTMT_NAME,'$NULL$')")

gap(25, "Красный результат без принятия риска владельцем", "validation", "MODEL", "department", "result",
    "T_VALID", "Красная модель требует явного принятия риска владельцем.",
    f"valid WHERE UPPER(COALESCE(VALID_RSLT_NAME,'')) RLIKE '{RED_RESULT_RE}'",
    f"NOT ({truth('VALID_RED_ZONE_OWNER_APRVL_RSK_FLAG')})",
    "COALESCE(VALID_DPRTMT_NAME,'$NULL$')", "COALESCE(VALID_RSLT_NAME,'$NULL$')")

gap(26, "Красный результат без решения КРГ", "validation", "MODEL", "department", "result",
    "T_VALID", "Проверяет наличие ID и ссылки решения КРГ.",
    f"valid WHERE UPPER(COALESCE(VALID_RSLT_NAME,'')) RLIKE '{RED_RESULT_RE}'",
    "VALID_RED_ZONE_COMT_RSK_DECSN_SID IS NULL OR TRIM(CAST(VALID_RED_ZONE_COMT_RSK_DECSN_SID AS STRING))='' OR VALID_RED_ZONE_COMT_RSK_DECSN_LINK_TXT IS NULL OR TRIM(CAST(VALID_RED_ZONE_COMT_RSK_DECSN_LINK_TXT AS STRING))=''",
    "COALESCE(VALID_DPRTMT_NAME,'$NULL$')", "COALESCE(VALID_RSLT_NAME,'$NULL$')")

dist(27, "Типы ошибок валидации", "validation", "MODEL", "problem_count",
     "problem", "result", "T_VALID", "Показывает наиболее частые классы выявленных рисков.",
     "valid WHERE VALID_PRBLM_NAME IS NOT NULL", "VALID_PRBLM_NAME", "COALESCE(VALID_RSLT_NAME,'$NULL$')")

add_query(28, "Альтернативное моделирование и метрики", "validation", "MODEL", "risk_rate_pct",
          "indicator", "significance", "T_VALID,T_MODEL_VER", "Покрытие независимой альтернативой и полнота её метрики.", f"""
    WITH x AS (
      SELECT COALESCE(m.MODEL_VER_SIGNFCNT_CTGRY_CODE,'$NULL$') significance,
             STACK(2,
               'alt_not_run', CASE WHEN NOT ({truth('v.VALID_ALT_FLAG')}) THEN 1 ELSE 0 END,
               'alt_metric_missing', CASE WHEN {truth('v.VALID_ALT_FLAG')} AND (v.VALID_ALT_METRIC_NAME IS NULL OR v.VALID_ALT_METRIC_VAL IS NULL) THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM valid v LEFT JOIN mv m ON v.MODEL_VER_SID=m.MODEL_VER_SID
    ) SELECT indicator d1, significance d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator, COUNT(*) denominator FROM x GROUP BY indicator,significance
""")

gap(29, "Использование агента-валидатора", "agent_validation", "MODEL", "type", "result",
    "T_VALID", "Доля валидаций, где агент не использовался.", "valid",
    f"NOT ({truth('VALID_AGENT_USG_FLAG')})", "COALESCE(VALID_TYPE_NAME,'$NULL$')", "COALESCE(VALID_RSLT_NAME,'$NULL$')")

add_query(30, "Качество и переиспользование тестов агента", "agent_validation", "MODEL", "percent",
          "agent_metric", "department", "T_VALID", "Средние проценты корректности и переиспользования тестов агента.", f"""
    SELECT metric d1, department d2, AVG(value) metric_value,
           CAST(NULL AS BIGINT) numerator, COUNT(value) denominator
    FROM (
      SELECT COALESCE(VALID_DPRTMT_NAME,'$NULL$') department,
             STACK(3,
               'quality_correct', CAST(VALID_AGENT_TEST_QUALITY_CORR_PCT AS DOUBLE),
               'quality_reuse', CAST(VALID_AGENT_TEST_QUALITY_REUSE_PCT AS DOUBLE),
               'quantity_reuse', CAST(VALID_AGENT_TEST_QUANTITY_REUSE_PCT AS DOUBLE)
             ) AS (metric,value)
      FROM valid WHERE {truth('VALID_AGENT_USG_FLAG')}
    ) x GROUP BY metric,department
""")

dist(31, "Причины неиспользования агента", "agent_validation", "MODEL", "validation_count",
     "reason", "type", "T_VALID", "Показывает ограничения применимости агента-валидатора.",
     f"valid WHERE NOT ({truth('VALID_AGENT_USG_FLAG')})",
     "COALESCE(VALID_AGENT_NON_USG_REASON_NAME,'$NULL$')", "COALESCE(VALID_TYPE_NAME,'$NULL$')")

dist(32, "Переиспользование превалидации", "validation", "MODEL", "validation_count",
     "reuse_level", "reason", "T_VALID", "Показывает степень и причины неполного переиспользования превалидации.",
     "valid", "COALESCE(VALID_PREVALID_REUSE_LVL_NAME,'$NULL$')", "COALESCE(CAST(VALID_PREVALID_REUSE_LVL_REASON_ARRAY AS STRING),'$NULL$')")


# 33–42. Мониторинг, внедрение и целостность
gap(33, "Покрытие автоматическим мониторингом", "monitoring", "MODEL", "significance", "status",
    "T_MODEL_VER,T_MODEL_VER_X_MONTRG_AUTO,T_MONTRG_AUTO", "Версия считается покрытой только при наличии связанной карточки мониторинга.",
    """mv_scope v LEFT JOIN (
      SELECT x.MODEL_VER_SID,MAX(CASE WHEN a.MONTRG_AUTO_SID IS NOT NULL THEN 1 ELSE 0 END) has_auto
      FROM x_auto x LEFT JOIN auto a ON x.MONTRG_AUTO_SID=a.MONTRG_AUTO_SID
      GROUP BY x.MODEL_VER_SID
    ) a ON v.MODEL_VER_SID=a.MODEL_VER_SID""",
    "COALESCE(a.has_auto,0)=0", sig, "COALESCE(v.MODEL_VER_STTS_NAME,'$NULL$')")

gap(34, "Просроченный автоматический мониторинг", "monitoring", "MODEL", "department", "status",
    "T_MONTRG_AUTO", "Дата следующего запуска прошла, карточка не закрыта.", "auto",
    f"MONTRG_AUTO_NEXT_DTTM IS NOT NULL AND DATE(MONTRG_AUTO_NEXT_DTTM)<CURRENT_DATE() AND UPPER(COALESCE(MONTRG_AUTO_STTS_NAME,'')) NOT RLIKE '{CLOSED_STATUS_RE}'",
    "COALESCE(MONTRG_AUTO_DPRTMT_NAME,'$NULL$')", "COALESCE(MONTRG_AUTO_STTS_NAME,'$NULL$')")

add_query(35, "Разрывы расписания автомониторинга", "monitoring", "MODEL", "risk_rate_pct",
          "schedule_gap", "frequency", "T_MONTRG_AUTO", "Выявляет неактивное расписание и отсутствующий cron.", f"""
    SELECT indicator d1, frequency d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator, COUNT(*) denominator
    FROM (
      SELECT COALESCE(MONTRG_AUTO_FREQ_NAME,'$NULL$') frequency,
             STACK(2,
               'inactive_schedule', CASE WHEN UPPER(COALESCE(MONTRG_AUTO_PROJ_SCHED_STTS_NAME,'')) RLIKE 'INACTIVE|DISABLED|ОТКЛ' THEN 1 ELSE 0 END,
               'cron_missing', CASE WHEN MONTRG_AUTO_PROJ_SCHED_CRON_TXT IS NULL OR TRIM(CAST(MONTRG_AUTO_PROJ_SCHED_CRON_TXT AS STRING))='' THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM auto
    ) x GROUP BY indicator,frequency
""")

add_query(36, "Результаты и пробелы автомониторинга", "monitoring", "MODEL", "result_count_or_gap_pct",
          "indicator_or_result", "result", "T_MONTRG_AUTO_RSLT,T_SAMPLE_DATA", "Цвет результата и отсутствие метрики, выборки или отчёта.", f"""
    WITH x AS (
      SELECT r.*, MAX(CASE WHEN s.SAMPLE_DATA_SID IS NOT NULL THEN 1 ELSE 0 END) OVER(PARTITION BY r.MONTRG_AUTO_RSLT_SID) has_sample
      FROM auto_r r LEFT JOIN sample s ON r.MONTRG_AUTO_RSLT_SID=s.MONTRG_AUTO_RSLT_SID
    ), one_row AS (SELECT DISTINCT * FROM x), gaps AS (
      SELECT COALESCE(MONTRG_AUTO_RSLT_NAME,'$NULL$') result,
             STACK(3,
               'metric_missing', CASE WHEN MONTRG_AUTO_RSLT_METRIC_VAL IS NULL OR {truth('MONTRG_AUTO_RSLT_NOT_METRIC_MAIN_FLAG')} THEN 1 ELSE 0 END,
               'sample_missing', CASE WHEN has_sample=0 OR {truth('MONTRG_AUTO_RSLT_VALID_NOT_SAMPLE_DATA_FLAG')} THEN 1 ELSE 0 END,
               'report_missing', CASE WHEN MONTRG_AUTO_RSLT_REPORT_SID IS NULL THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM one_row
    ) SELECT indicator d1,result d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM gaps GROUP BY indicator,result
""")

add_query(37, "Покрытие и просрочка ручного мониторинга", "monitoring", "MODEL", "risk_rate_pct",
          "indicator", "significance", "T_MODEL_VER,T_MODEL_VER_X_MONTRG_MANUAL,T_MONTRG_MANUAL", "Отсутствие ручного мониторинга и нарушение следующей даты.", f"""
    WITH x AS (
      SELECT {sig} significance,
             STACK(2,
               'manual_monitor_missing', CASE WHEN COALESCE(m.has_manual,0)=0 THEN 1 ELSE 0 END,
               'manual_monitor_overdue', CASE WHEN COALESCE(m.has_overdue,0)=1 THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM mv_scope v LEFT JOIN (
        SELECT xm.MODEL_VER_SID,
               MAX(CASE WHEN m.MONTRG_MANUAL_SID IS NOT NULL THEN 1 ELSE 0 END) has_manual,
               MAX(CASE WHEN m.MONTRG_MANUAL_NEXT_DTTM IS NOT NULL
                         AND DATE(m.MONTRG_MANUAL_NEXT_DTTM)<CURRENT_DATE()
                         AND UPPER(COALESCE(m.MONTRG_MANUAL_STTS_NAME,'')) NOT RLIKE '{CLOSED_STATUS_RE}'
                        THEN 1 ELSE 0 END) has_overdue
        FROM x_manual xm LEFT JOIN manual m ON xm.MONTRG_MANUAL_SID=m.MONTRG_MANUAL_SID
        GROUP BY xm.MODEL_VER_SID
      ) m ON v.MODEL_VER_SID=m.MODEL_VER_SID
    ) SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM x GROUP BY indicator,significance
""")

add_query(38, "Результаты и пробелы ручного мониторинга", "monitoring", "MODEL", "result_count_or_gap_pct",
          "indicator", "result", "T_MONTRG_MANUAL_RSLT,T_SAMPLE_DATA", "Цвет результата и отсутствие метрики, выборки или отчёта.", f"""
    WITH x AS (
      SELECT r.*, MAX(CASE WHEN s.SAMPLE_DATA_SID IS NOT NULL THEN 1 ELSE 0 END) OVER(PARTITION BY r.MONTRG_MANUAL_RSLT_SID) has_sample
      FROM manual_r r LEFT JOIN sample s ON r.MONTRG_MANUAL_RSLT_SID=s.MONTRG_MANUAL_RSLT_SID
    ), one_row AS (SELECT DISTINCT * FROM x), gaps AS (
      SELECT COALESCE(MONTRG_MANUAL_RSLT_NAME,'$NULL$') result,
             STACK(3,
               'metric_missing', CASE WHEN MONTRG_MANUAL_RSLT_METRIC_VAL IS NULL OR {truth('MONTRG_MANUAL_RSLT_NOT_METRIC_MAIN_FLAG')} THEN 1 ELSE 0 END,
               'sample_missing', CASE WHEN has_sample=0 OR {truth('MONTRG_MANUAL_RSLT_VALID_NOT_SAMPLE_DATA_FLAG')} THEN 1 ELSE 0 END,
               'report_missing', CASE WHEN MONTRG_MANUAL_RSLT_REPORT_SID IS NULL THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM one_row
    ) SELECT indicator d1,result d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM gaps GROUP BY indicator,result
""")

add_query(39, "Расхождение авто- и ручного мониторинга", "monitoring", "MODEL", "risk_rate_pct",
          "significance", "auto_result", "T_MODEL_VER,monitoring tables", "Сравнивает последние результаты двух независимых каналов мониторинга.", f"""
    WITH auto_latest AS (
      SELECT MODEL_VER_SID,MONTRG_AUTO_RSLT_NAME FROM (
        SELECT xa.MODEL_VER_SID,ar.MONTRG_AUTO_RSLT_NAME,
               ROW_NUMBER() OVER(PARTITION BY xa.MODEL_VER_SID
                 ORDER BY COALESCE(ar.MONTRG_AUTO_RSLT_END_DTTM,ar.MONTRG_AUTO_RSLT_CRTN_DTTM) DESC) rn
        FROM x_auto xa JOIN auto_r ar ON xa.MONTRG_AUTO_SID=ar.MONTRG_AUTO_SID
      ) x WHERE rn=1
    ), manual_latest AS (
      SELECT MODEL_VER_SID,MONTRG_MANUAL_RSLT_NAME FROM (
        SELECT xm.MODEL_VER_SID,mr.MONTRG_MANUAL_RSLT_NAME,
               ROW_NUMBER() OVER(PARTITION BY xm.MODEL_VER_SID
                 ORDER BY COALESCE(mr.MONTRG_MANUAL_RSLT_END_DTTM,mr.MONTRG_MANUAL_RSLT_CRTN_DTTM) DESC) rn
        FROM x_manual xm JOIN manual_r mr ON xm.MONTRG_MANUAL_SID=mr.MONTRG_MANUAL_SID
      ) x WHERE rn=1
    )
    SELECT {sig} d1,COALESCE(a.MONTRG_AUTO_RSLT_NAME,'$NULL$') d2,
           100.0*SUM(CASE WHEN COALESCE(UPPER(a.MONTRG_AUTO_RSLT_NAME),'')<>COALESCE(UPPER(m.MONTRG_MANUAL_RSLT_NAME),'') THEN 1 ELSE 0 END)/COUNT(*) metric_value,
           SUM(CASE WHEN COALESCE(UPPER(a.MONTRG_AUTO_RSLT_NAME),'')<>COALESCE(UPPER(m.MONTRG_MANUAL_RSLT_NAME),'') THEN 1 ELSE 0 END) numerator,
           COUNT(*) denominator
    FROM mv_scope v JOIN auto_latest a ON v.MODEL_VER_SID=a.MODEL_VER_SID
    JOIN manual_latest m ON v.MODEL_VER_SID=m.MODEL_VER_SID
    GROUP BY {sig},COALESCE(a.MONTRG_AUTO_RSLT_NAME,'$NULL$')
""")

add_query(40, "ИТ-валидация и QGM промышленной версии", "implementation", "MODEL", "risk_rate_pct",
          "indicator", "prom_status", "T_MODEL_VER_PROM,T_VALID_IT", "Контроль допуска промышленной версии.", f"""
    WITH x AS (
      SELECT COALESCE(p.MODEL_VER_PROM_STTS_NAME,'$NULL$') prom_status,
             STACK(2,
               'it_validation_missing', CASE WHEN i.MODEL_VER_PROM_SID IS NULL THEN 1 ELSE 0 END,
               'qgm_not_positive', CASE WHEN NOT ({truth('p.MODEL_VER_PROM_VALID_IT_RSLT_QGM_FLAG')}) THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM mvp p LEFT JOIN (SELECT DISTINCT MODEL_VER_PROM_SID FROM valid_it) i
        ON p.MODEL_VER_PROM_SID=i.MODEL_VER_PROM_SID
    ) SELECT indicator d1,prom_status d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM x GROUP BY indicator,prom_status
""")

add_query(41, "Просрочка пилота и промышленного внедрения", "implementation", "MODEL", "risk_rate_pct",
          "implementation_type", "status", "implementation tables", "Плановая дата старта прошла, но фактический старт отсутствует или опоздал.", """
    WITH impl AS (
      SELECT 'pilot' kind, p.PILOT_IMPLM_STTS_NAME status, p.PILOT_IMPLM_START_PLAN_DTTM plan_dt, p.PILOT_IMPLM_START_FACT_DTTM fact_dt
      FROM pilot p
      UNION ALL
      SELECT 'industrial', p.PROM_IMPLM_STTS_NAME, p.PROM_IMPLM_START_PLAN_DTTM, p.PROM_IMPLM_START_FACT_DTTM FROM prom p
    ) SELECT kind d1,COALESCE(status,'$NULL$') d2,
             100.0*SUM(CASE WHEN plan_dt IS NOT NULL AND ((fact_dt IS NULL AND DATE(plan_dt)<CURRENT_DATE()) OR fact_dt>plan_dt) THEN 1 ELSE 0 END)/COUNT(*) metric_value,
             SUM(CASE WHEN plan_dt IS NOT NULL AND ((fact_dt IS NULL AND DATE(plan_dt)<CURRENT_DATE()) OR fact_dt>plan_dt) THEN 1 ELSE 0 END) numerator,
             COUNT(*) denominator FROM impl GROUP BY kind,COALESCE(status,'$NULL$')
""")

add_query(42, "Нарушения ссылочной целостности", "data_quality", "ALL", "orphan_rate_pct",
          "relation", "", "key entity and bridge tables", "Доля дочерних записей, для которых отсутствует родитель.", """
    WITH checks AS (
      SELECT 'MODEL_VER->MODEL' relation, SUM(CASE WHEN m.MODEL_SID IS NULL THEN 1 ELSE 0 END) bad, COUNT(*) total
        FROM mv v LEFT JOIN model m ON v.MODEL_SID=m.MODEL_SID WHERE v.MODEL_SID IS NOT NULL
      UNION ALL SELECT 'VALID->MODEL_VER',SUM(CASE WHEN v.MODEL_VER_SID IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM valid x LEFT JOIN mv v ON x.MODEL_VER_SID=v.MODEL_VER_SID WHERE x.MODEL_VER_SID IS NOT NULL
      UNION ALL SELECT 'SAMPLE->MODEL_VER',SUM(CASE WHEN v.MODEL_VER_SID IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM sample x LEFT JOIN mv v ON x.MODEL_VER_SID=v.MODEL_VER_SID WHERE x.MODEL_VER_SID IS NOT NULL
      UNION ALL SELECT 'METRIC->SAMPLE',SUM(CASE WHEN s.SAMPLE_DATA_SID IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM metric x LEFT JOIN sample s ON x.SAMPLE_DATA_SID=s.SAMPLE_DATA_SID WHERE x.SAMPLE_DATA_SID IS NOT NULL
      UNION ALL SELECT 'X_AUTO->AUTO',SUM(CASE WHEN a.MONTRG_AUTO_SID IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM x_auto x LEFT JOIN auto a ON x.MONTRG_AUTO_SID=a.MONTRG_AUTO_SID
      UNION ALL SELECT 'X_MANUAL->MANUAL',SUM(CASE WHEN m.MONTRG_MANUAL_SID IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM x_manual x LEFT JOIN manual m ON x.MONTRG_MANUAL_SID=m.MONTRG_MANUAL_SID
      UNION ALL SELECT 'GENAI_VER->GENAI',SUM(CASE WHEN g.GENAI_SID IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM gv v LEFT JOIN genai g ON v.GENAI_SID=g.GENAI_SID WHERE v.GENAI_SID IS NOT NULL
    ) SELECT relation d1,CAST(NULL AS STRING) d2,
             CASE WHEN total=0 THEN CAST(NULL AS DOUBLE) ELSE 100.0*bad/total END metric_value,
             COALESCE(bad,0) numerator,total denominator FROM checks
""")


# 43–50. GenAI и агенты
add_query(43, "Портфель GenAI и агентов", "genai_portfolio", "GENAI", "version_count",
          "portfolio_attribute", "value", "T_GENAI,T_GENAI_VER", "Статус, значимость, признак агента, критичность и зрелость портфеля.", """
    SELECT attribute d1,value d2,CAST(COUNT(*) AS DOUBLE) metric_value,
           COUNT(*) numerator,CAST(NULL AS BIGINT) denominator
    FROM (
      SELECT STACK(5,
        'version_status',COALESCE(CAST(v.GENAI_VER_STTS_NAME AS STRING),'$NULL$'),
        'significance',COALESCE(CAST(v.GENAI_VER_SIGNFCNT_CTGRY_CODE AS STRING),'$NULL$'),
        'agent_flag',COALESCE(CAST(g.GENAI_AGENT_FLAG AS STRING),'$NULL$'),
        'priority',COALESCE(CAST(g.GENAI_PRIORITY_LVL_CODE AS STRING),'$NULL$'),
        'maturity',COALESCE(CAST(g.GENAI_MATURITY_LVL_ORD AS STRING),'$NULL$')
      ) AS (attribute,value)
      FROM gv v JOIN genai g ON v.GENAI_SID=g.GENAI_SID
    ) x GROUP BY attribute,value
""")

add_query(44, "Документирование и воспроизводимость GenAI", "genai_governance", "GENAI", "risk_rate_pct",
          "missing_field", "significance", "T_GENAI,T_GENAI_VER", "Пробелы владельца, репозитория, коммита, релиза и отчёта разработки.", f"""
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
    FROM (
      SELECT {gsig} significance,
             STACK(5,
               'owner',CASE WHEN g.GENAI_OWNER_DPRTMT_NAME IS NULL OR TRIM(CAST(g.GENAI_OWNER_DPRTMT_NAME AS STRING))='' THEN 1 ELSE 0 END,
               'repository',CASE WHEN v.GENAI_VER_REPSTRY_LINK_TXT IS NULL OR TRIM(CAST(v.GENAI_VER_REPSTRY_LINK_TXT AS STRING))='' THEN 1 ELSE 0 END,
               'commit',CASE WHEN v.GENAI_VER_REPSTRY_COMMIT_LINK_TXT IS NULL OR TRIM(CAST(v.GENAI_VER_REPSTRY_COMMIT_LINK_TXT AS STRING))='' THEN 1 ELSE 0 END,
               'release',CASE WHEN v.GENAI_VER_RELEASE_LINK_TXT IS NULL OR TRIM(CAST(v.GENAI_VER_RELEASE_LINK_TXT AS STRING))='' THEN 1 ELSE 0 END,
               'development_report',CASE WHEN v.GENAI_VER_DEV_REPORT_SID IS NULL THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM gv v JOIN genai g ON v.GENAI_SID=g.GENAI_SID
    ) x GROUP BY indicator,significance
""")

add_query(45, "Данные GenAI", "genai_data", "GENAI", "risk_rate_pct",
          "indicator", "significance", "T_GENAI_VER,T_GENAI_VER_SAMPLE_DATA", "Отсутствие выборки или уровня знания данных.", f"""
    WITH x AS (
      SELECT v.*,CASE WHEN s.GENAI_VER_SID IS NULL THEN 0 ELSE 1 END has_sample
      FROM gv v LEFT JOIN (SELECT DISTINCT GENAI_VER_SID FROM g_sample) s ON v.GENAI_VER_SID=s.GENAI_VER_SID
    ), y AS (
      SELECT {gsig} significance,STACK(2,
        'sample_missing',CASE WHEN has_sample=0 THEN 1 ELSE 0 END,
        'data_knowledge_level_missing',CASE WHEN GENAI_VER_SAMPLE_DATA_KNOWLEDGE_LVL_NAME IS NULL OR TRIM(CAST(GENAI_VER_SAMPLE_DATA_KNOWLEDGE_LVL_NAME AS STRING))='' THEN 1 ELSE 0 END
      ) AS (indicator,is_gap) FROM x
    ) SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
      FROM y GROUP BY indicator,significance
""")

add_query(46, "Покрытие и результаты валидации GenAI", "genai_validation", "GENAI", "version_count",
          "validation_result", "significance", "T_GENAI_VER,T_GENAI_VER_VALID", "Распределение результатов, включая версии без валидации.", f"""
    WITH latest AS (
      SELECT * FROM (
        SELECT x.*,ROW_NUMBER() OVER(PARTITION BY GENAI_VER_SID
          ORDER BY COALESCE(GENAI_VER_VALID_END_FACT_DTTM,GENAI_VER_VALID_START_FACT_DTTM,GENAI_VER_VALID_CRTN_DTTM) DESC) rn
        FROM g_valid x
      ) z WHERE rn=1
    )
    SELECT COALESCE(x.GENAI_VER_VALID_RSLT_NAME,'NO_VALIDATION') d1,{gsig} d2,
           CAST(COUNT(*) AS DOUBLE) metric_value,COUNT(*) numerator,
           CAST(NULL AS BIGINT) denominator
    FROM gv v LEFT JOIN latest x ON v.GENAI_VER_SID=x.GENAI_VER_SID
    GROUP BY COALESCE(x.GENAI_VER_VALID_RSLT_NAME,'NO_VALIDATION'),{gsig}
""")

add_query(47, "Контроли валидации GenAI", "genai_validation", "GENAI", "risk_rate_pct_or_days",
          "indicator", "result", "T_GENAI_VER_VALID", "Пробелы отчёта/альтернативной метрики и длительность процесса.", f"""
    WITH base AS (
      SELECT COALESCE(GENAI_VER_VALID_RSLT_NAME,'$NULL$') result,
             CASE WHEN GENAI_VER_VALID_REPORT_SID IS NULL THEN 1 ELSE 0 END report_gap,
             CASE WHEN {truth('GENAI_VER_VALID_ALT_FLAG')} AND (GENAI_VER_VALID_ALT_METRIC_NAME IS NULL OR GENAI_VER_VALID_ALT_METRIC_VAL IS NULL) THEN 1 ELSE 0 END alt_gap,
             DATEDIFF(DATE(GENAI_VER_VALID_END_FACT_DTTM),DATE(GENAI_VER_VALID_START_FACT_DTTM)) days
      FROM g_valid
    ), gaps AS (
      SELECT indicator d1,result d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
      FROM (SELECT result,STACK(2,'report_missing',report_gap,'alt_metric_missing',alt_gap) AS (indicator,is_gap) FROM base) x
      GROUP BY indicator,result
    ), duration AS (
      SELECT 'duration_p90_days' d1,result d2,PERCENTILE_APPROX(days,0.90) metric_value,
             CAST(NULL AS BIGINT) numerator,COUNT(days) denominator FROM base WHERE days>=0 GROUP BY result
    ) SELECT * FROM gaps UNION ALL SELECT * FROM duration
""")

add_query(48, "Мониторинг GenAI", "genai_monitoring", "GENAI", "risk_rate_pct",
          "indicator", "significance", "T_GENAI_VER", f"Покрытие модели/данных и результат старше {STALE_MONITORING_DAYS} дней.", f"""
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
    FROM (
      SELECT {gsig} significance,STACK(5,
        'model_monitoring_off',CASE WHEN NOT ({truth('GENAI_VER_MONTRG_AUTO_MODEL_FLAG')}) THEN 1 ELSE 0 END,
        'data_monitoring_off',CASE WHEN NOT ({truth('GENAI_VER_MONTRG_AUTO_DATA_FLAG')}) THEN 1 ELSE 0 END,
        'model_exclusion_reason_missing',CASE WHEN NOT ({truth('GENAI_VER_MONTRG_AUTO_MODEL_FLAG')}) AND (GENAI_VER_MONTRG_AUTO_EXCLUDE_REASON_MODEL_ARRAY IS NULL OR TRIM(CAST(GENAI_VER_MONTRG_AUTO_EXCLUDE_REASON_MODEL_ARRAY AS STRING))='') THEN 1 ELSE 0 END,
        'data_exclusion_reason_missing',CASE WHEN NOT ({truth('GENAI_VER_MONTRG_AUTO_DATA_FLAG')}) AND (GENAI_VER_MONTRG_AUTO_EXCLUDE_REASON_DATA_ARRAY IS NULL OR TRIM(CAST(GENAI_VER_MONTRG_AUTO_EXCLUDE_REASON_DATA_ARRAY AS STRING))='') THEN 1 ELSE 0 END,
        'last_result_missing_or_stale',CASE WHEN GENAI_VER_MONTRG_LAST_RSLT_DTTM IS NULL OR DATEDIFF(CURRENT_DATE(),DATE(GENAI_VER_MONTRG_LAST_RSLT_DTTM))>{STALE_MONITORING_DAYS} THEN 1 ELSE 0 END
      ) AS (indicator,is_gap) FROM gv
    ) x GROUP BY indicator,significance
""")

add_query(49, "Ключевые метрики GenAI", "genai_metrics", "GENAI", "risk_rate_pct",
          "indicator", "significance", "T_GENAI_VER", "Проверяет наличие названия, значения и автоматической оценки ключевой метрики.", f"""
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
    FROM (
      SELECT {gsig} significance,STACK(3,
        'metric_code_missing',CASE WHEN GENAI_VER_METRIC_KEY_CODE IS NULL OR TRIM(CAST(GENAI_VER_METRIC_KEY_CODE AS STRING))='' THEN 1 ELSE 0 END,
        'metric_value_missing',CASE WHEN GENAI_VER_METRIC_KEY_VAL IS NULL THEN 1 ELSE 0 END,
        'auto_assessment_missing',CASE WHEN GENAI_VER_METRIC_ASSMNT_AUTO_CMNT_TXT IS NULL OR TRIM(CAST(GENAI_VER_METRIC_ASSMNT_AUTO_CMNT_TXT AS STRING))='' THEN 1 ELSE 0 END
      ) AS (indicator,is_gap) FROM gv
    ) x GROUP BY indicator,significance
""")

add_query(50, "Экономика и лимиты использования GenAI", "genai_usage", "GENAI", "risk_rate_pct",
          "indicator", "significance", "T_GENAI_VER", "Недостижение планового эффекта и отсутствие лимитов использования.", f"""
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
    FROM (
      SELECT {gsig} significance,STACK(3,
        'financial_effect_below_plan',CASE WHEN CAST(GENAI_VER_FIN_EFFECT_PLAN_MILLION_QTY AS DOUBLE)>0 AND COALESCE(CAST(GENAI_VER_FIN_EFFECT_FACT_MILLION_QTY AS DOUBLE),0)<CAST(GENAI_VER_FIN_EFFECT_PLAN_MILLION_QTY AS DOUBLE) THEN 1 ELSE 0 END,
        'usage_days_limit_missing',CASE WHEN GENAI_VER_USG_DAYS_QTY IS NULL THEN 1 ELSE 0 END,
        'concurrent_usage_limit_missing',CASE WHEN GENAI_VER_USG_SAME_TIME_CNT IS NULL THEN 1 ELSE 0 END
      ) AS (indicator,is_gap) FROM gv
    ) x GROUP BY indicator,significance
""")


if len(results) != 50:
    raise RuntimeError(f"Ожидалось 50 запросов, сформировано: {len(results)}")

final_df = reduce(lambda left, right: left.unionByName(right), results).persist(StorageLevel.MEMORY_AND_DISK)

# Удобный предпросмотр перед записью. Это агрегаты, поэтому объём ограничен.
final_df.orderBy("query_id", F.desc("metric_value")).show(SHOW_ROWS, truncate=False)

(
    final_df
    .repartition(1)
    .write
    .mode("overwrite")
    .format("parquet")
    .saveAsTable(OUT_TABLE)
)

print(f"Готово: {OUT_TABLE}")

final_df.unpersist()
for frame in cached_frames:
    frame.unpersist()
