"""Единый аудит актуальных версий: 31 контроль + журнал изменений.

Spark 3.5.1 / Hive. Запуск: %run model_current_audit.py
Никакие исходные/Hive-таблицы не изменяются. Полные результаты -> Parquet.
CONFIG ниже. latest_created = одна последняя созданная версия на модель,
а не последняя технически обновлённая запись. При неоднозначности — ошибка.
current_all = все бизнес-актуальные SID версий (в т.ч. параллельные версии).
production_only независимо ограничивает выбранные версии признаком эксплуатации.
История изменений сохраняется только для выбранных SID; её строки не являются
дополнительными версиями. «Текущее значение» — последнее значение В ЖУРНАЛЕ.
31 правило сохранено из исходного .py, включая области применимости и p90.
NOT_ASSESSABLE — нет применимых записей, не признак отсутствия риска.
Исторические валидации доступны лишь в пределах текущих записей t_valid.
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from functools import reduce
from typing import List

# Работает и внутри уже запущенного ноутбука, и в окружении исходного скрипта.
try:
    import pyspark
except ImportError:
    spark_home = '/usr/sdp/current/spark3.5.1-client/'
    os.environ.setdefault('SPARK_HOME', spark_home)
    os.environ.setdefault('SPARK_MAJOR_VERSION', '3.5.1')
    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
    sys.path[:0] = [spark_home + 'python/',
                   spark_home + 'python/lib/py4j-0.10.9.7-src.zip']
from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame, Window, functions as F

CONFIG = {
    'source_db': 'prx_pri_custom_ris_l_library_custom_risk_model_library',
    'version_mode': 'latest_created',  # или current_all
    'production_only': False,         # True: только промышленная эксплуатация
    'output_root': 'model_audit_runs', # путь в Hadoop FS; для локального file:///...
    'local_log_dir': './model_audit_logs',
    'show_rows': 30,
    'export_excel': False,            # нужен openpyxl; полный журнал уже в Parquet
    'excel_max_rows': 1_048_575,
}
SRC_DB = CONFIG['source_db']
SHOW_ROWS = CONFIG['show_rows']
MIN_GROUP_FOR_P90 = 5
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
local_dir = Path(CONFIG['local_log_dir']) / RUN_ID
local_dir.mkdir(parents=True, exist_ok=False)
log = logging.getLogger('model_current_audit.' + RUN_ID)
log.setLevel(logging.INFO)
log.propagate = False
for handler in (logging.StreamHandler(), logging.FileHandler(
        local_dir / 'run.log', encoding='utf-8')):
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(handler)

spark = SparkSession.getActiveSession() or (
    SparkSession.builder.appName('model_current_audit').enableHiveSupport().getOrCreate())
spark.conf.set('spark.sql.session.timeZone', 'Europe/Moscow')
AS_OF_DT = str(spark.sql('SELECT current_date() d').first()['d'])
AS_OF_TS = str(spark.sql('SELECT current_timestamp() t').first()['t'])
audit = []
cached = []
outputs = {}

def record(stage, count, note=''):
    audit.append((stage, int(count), note))
    log.info('%s: %s %s', stage, count, note)

def require(df, fields, name):
    missing = sorted(set(fields) - set(df.columns))
    if missing:
        raise ValueError(f'{name}: отсутствуют поля {missing}')

def ts_col(name):
    return F.coalesce(F.expr(f'try_cast(`{name}` as timestamp)'),
        F.try_to_timestamp(F.col(name).cast('string'), F.lit('dd.MM.yyyy H:mm:ss')))

def parse_dates(df, names, label):
    # Не превращаем непарсящиеся непустые даты в «бесконечно актуальные» записи.
    for name in names:
        if name not in df.columns:
            continue
        text = F.lower(F.trim(F.col(name).cast('string')))
        present = F.col(name).isNotNull() & ~text.isin('', 'null', 'none')
        if df.filter(present & ts_col(name).isNull()).limit(1).count():
            raise ValueError(f'{label}.{name}: непарсящаяся дата; проверьте формат')
        df = df.withColumn(name, ts_col(name))
    return df

def current(table, key, fields):
    raw = spark.table(f'{SRC_DB}.{table}')
    require(raw, fields + [key], table)
    cols = list(dict.fromkeys(fields + [key] + [c for c in
        ('start_dt', 'end_dt', 'ctl_datechange', 'ctl_datecreate_dttm', 'ctl_action')
        if c in raw.columns]))
    d = raw.select(*cols)
    d = parse_dates(d, [c for c in cols if c.endswith('_dttm')] +
                    ['start_dt', 'end_dt', 'ctl_datechange'], table)
    if d.filter(F.col(key).isNull() | (F.trim(F.col(key).cast('string')) == '')).limit(1).count():
        raise ValueError(f'{table}: пустой ключ {key}')
    day = F.lit(AS_OF_DT).cast('date')
    # Включительные границы по календарной дате, как в исходном .py.
    if 'start_dt' in cols:
        d = d.filter(F.col('start_dt').isNull() | (F.to_date('start_dt') <= day))
    if 'end_dt' in cols:
        d = d.filter(F.col('end_dt').isNull() | (F.to_date('end_dt') >= day))
    order = [F.col(c).desc_nulls_last() for c in
             ('ctl_datechange', 'start_dt', 'ctl_datecreate_dttm') if c in cols]
    if not order:
        raise ValueError(f'{table}: нет технических дат для выбора текущей строки')
    d = d.withColumn('_rank', F.dense_rank().over(
        Window.partitionBy(key).orderBy(*order))).filter('_rank = 1').drop('_rank').distinct()
    if d.groupBy(key).count().filter('count > 1').limit(1).count():
        raise ValueError(f'{table}: конфликт актуальных строк по {key}, даты совпадают')
    # Удаление после ранжирования: не воскрешаем старую запись вместо tombstone.
    if 'ctl_action' in cols:
        d = d.filter(F.coalesce(F.upper(F.trim('ctl_action')) != 'D', F.lit(True)))
    record('current.' + table, d.count())
    return d.select(*list(dict.fromkeys(fields + [key])))

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



def prepare_scope():
    table, key, fields = SOURCES['mv']
    mv = current(table, key, fields)
    mode = CONFIG['version_mode']
    if mode == 'latest_created':
        if mv.filter(F.col('model_sid').isNull() |
                     F.col('model_ver_crtn_dttm').isNull()).limit(1).count():
            raise ValueError('Для latest_created нужны model_sid и model_ver_crtn_dttm '
                             'у всех версий. Заполните даты либо выберите current_all.')
        if mv.filter(F.col('model_ver_crtn_dttm') > F.lit(AS_OF_TS).cast('timestamp')).limit(1).count():
            raise ValueError('Дата создания версии в будущем: latest_created неоднозначен')
        mv = mv.withColumn('_r', F.dense_rank().over(Window.partitionBy('model_sid')
            .orderBy(F.col('model_ver_crtn_dttm').desc()))).filter('_r = 1').drop('_r')
        if mv.groupBy('model_sid').count().filter('count > 1').limit(1).count():
            raise ValueError('Несколько последних версий с одинаковой датой создания; '
                             'требуется бизнес-правило выбора или current_all')
    elif mode != 'current_all':
        raise ValueError('version_mode: latest_created или current_all')
    record('selected_before_status_filter', mv.count(), mode)
    # Не подменяем последнюю закрытую версию более старой открытой.
    mv = mv.filter(~F.upper(F.coalesce(F.col('model_ver_stts_name'), F.lit('')))
                   .rlike(CLOSED_MODEL_STATUS_RE))
    models = current('t_model', 'model_sid', ['model_sid', 'model_name', 'model_code'])
    anlt = current('t_model_ver_anlt_dtl', 'model_ver_sid', [
        'model_ver_sid', 'model_ver_stts_name', 'model_stts_name', 'model_ver_prom_expl_flag'])
    anlt = anlt.withColumnRenamed('model_ver_stts_name', 'analytics_version_status')
    context = mv.join(models, 'model_sid', 'left').join(anlt, 'model_ver_sid', 'left')
    if CONFIG['production_only']:
        context = context.filter(F.expr(truth('model_ver_prom_expl_flag')))
    context = context.persist(StorageLevel.MEMORY_AND_DISK)
    cached.append(context)
    record('scope_versions', context.count())
    record('scope_models', context.select('model_sid').distinct().count())
    record('status_disagreement', context.filter(
        F.col('analytics_version_status').isNotNull() &
        ~F.col('analytics_version_status').eqNullSafe(F.col('model_ver_stts_name'))).count(),
        'Контроли используют статус t_model_ver; второй статус сохранён в контексте')
    context.select(*fields).createOrReplaceTempView('mv')
    for view in ('sample', 'metric', 'valid'):
        t, k, f = SOURCES[view]
        d = current(t, k, f).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(d)
        d.createOrReplaceTempView(view)
    return context

def prepare_views():
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




summary_parts = []
detail_parts = []
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
        F.lit(AS_OF_DT).cast("date").alias("as_of_dt"),
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
        F.lit(AS_OF_DT).cast("date").alias("as_of_dt"),
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
            F.lit(AS_OF_DT).cast("date").alias("as_of_dt"),
        ).dropDuplicates(["query_id", "source_entity_type", "source_entity_sid", "issue_name"])
    )




def build_controls():
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
           CASE WHEN ({real_date('model_ver_dev_start_fact_dttm')} AND model_ver_dev_start_fact_dttm>TIMESTAMP '{AS_OF_TS}')
                      OR ({real_date('model_ver_dev_end_fact_dttm')} AND model_ver_dev_end_fact_dttm>TIMESTAMP '{AS_OF_TS}')
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




def assemble_controls(context):
    raw = reduce(lambda a, b: a.unionByName(b), summary_parts)
    w = Window.partitionBy('query_id')
    summary = (raw.withColumn('_min', F.min('_placeholder').over(w))
        .filter('_placeholder = _min').drop('_placeholder', '_min')
        .persist(StorageLevel.MEMORY_AND_DISK))
    detail = reduce(lambda a, b: a.unionByName(b), detail_parts).distinct().persist(
        StorageLevel.MEMORY_AND_DISK)
    cached.extend([summary, detail])
    keys = context.select('model_ver_sid').distinct()
    if detail.join(keys, 'model_ver_sid', 'left_anti').limit(1).count():
        raise RuntimeError('В детализацию попали версии вне выбранного периметра')
    if summary.select('query_id').distinct().count() != 31:
        raise RuntimeError('Ожидалось 31 правило')
    if summary.filter('(numerator < 0) OR (numerator > denominator)').limit(1).count():
        raise RuntimeError('Неверные числители/знаменатели')
    rollup = summary.groupBy('query_id', 'query_name', 'risk_domain', 'severity',
        'control_status', 'status_reason').agg(
        F.sum('numerator').alias('affected_records'),
        F.sum('denominator').alias('applicable_records'))
    rollup = rollup.withColumn('risk_rate_pct', F.when(F.col('applicable_records') > 0,
        F.round(100.0 * F.col('affected_records') / F.col('applicable_records'), 2)))
    version_stats = detail.groupBy('model_ver_sid').agg(
        F.countDistinct('query_id').alias('failed_controls'),
        F.countDistinct(F.when(F.col('severity') == 'HIGH', F.col('query_id')))
            .alias('failed_high_controls'),
        F.sort_array(F.collect_set('query_id')).alias('control_ids'))
    report = context.join(version_stats, 'model_ver_sid', 'left').fillna(
        {'failed_controls': 0, 'failed_high_controls': 0})
    # Ноль срабатываний не означает, что все 31 правило применимы к версии.
    enriched = detail.join(context.select('model_ver_sid', 'model_sid', 'model_code',
        'model_name', 'model_ver_stts_name', 'model_ver_prom_expl_flag'), 'model_ver_sid')
    return summary, enriched, rollup, report

def changes(context):
    table = 't_ent_param_chg'
    d = spark.table(f'{SRC_DB}.{table}')
    cols = ['ent_sid', 'ent_type_name', 'ent_param_chg_sid', 'ent_param_chg_name',
        'ent_param_chg_prev_val', 'ent_param_chg_val', 'ent_param_chg_type_code',
        'ent_param_chg_usr_sid', 'ent_param_chg_usr_name', 'start_dttm', 'end_dttm',
        'ctl_action', 'ctl_datechange']
    require(d, cols, table)
    d = parse_dates(d.select(*cols), ['start_dttm', 'end_dttm', 'ctl_datechange'], table)
    d = d.filter(F.coalesce(F.upper(F.trim('ctl_action')) != 'D', F.lit(True)))
    d = d.filter(F.col('start_dttm').isNull() |
                 (F.col('start_dttm') <= F.lit(AS_OF_TS).cast('timestamp')))
    # ent_param_chg_sid — идентификатор ПАРАМЕТРА, не события. Нельзя
    # дедуплицировать по нему: исчезнет история. Удаляем только точные повторы.
    d = d.distinct().filter(F.upper('ent_param_chg_sid').startswith('MODEL_VERSION_') |
                          F.upper('ent_type_name').isin('MODEL_VERSION', 'MODEL_VER'))
    ctx = context.select('model_ver_sid', 'model_sid', 'model_name', 'model_code',
        'model_ver_stts_name', 'model_ver_signfcnt_ctgry_code', 'model_ver_prom_expl_flag')
    history = ctx.alias('m').join(d.alias('c'),
        F.col('m.model_ver_sid') == F.col('c.ent_sid'), 'inner').select('m.*',
        F.col('c.ent_type_name').alias('entity_type'),
        F.col('c.ent_param_chg_sid').alias('parameter_sid'),
        F.col('c.ent_param_chg_name').alias('parameter_name'),
        F.col('c.ent_param_chg_type_code').alias('change_type_code'),
        F.col('c.start_dttm').alias('change_dttm'),
        F.col('c.end_dttm').alias('period_end_dttm'),
        F.col('c.ctl_datechange').alias('source_change_dttm'),
        F.col('c.ent_param_chg_prev_val').alias('previous_value'),
        F.col('c.ent_param_chg_val').alias('new_value'),
        F.col('c.ent_param_chg_usr_sid').alias('change_user_sid'),
        F.col('c.ent_param_chg_usr_name').alias('change_user_name'))
    part = Window.partitionBy('model_ver_sid', 'parameter_sid')
    desc = part.orderBy(F.col('change_dttm').desc_nulls_last(),
                        F.col('source_change_dttm').desc_nulls_last())
    asc = part.orderBy(F.col('change_dttm').asc_nulls_last(),
                       F.col('source_change_dttm').asc_nulls_last())
    h = history.withColumn('change_number', F.dense_rank().over(asc))
    h = h.withColumn('_latest', F.dense_rank().over(desc) == 1)
    h = h.withColumn('latest_candidate_count', F.sum(F.col('_latest').cast('int')).over(part))
    h = h.withColumn('undated_change_count', F.sum(F.col('change_dttm').isNull().cast('int')).over(part))
    certain = (F.col('latest_candidate_count') == 1) & (F.col('undated_change_count') == 0)
    h = h.withColumn('current_value_order_known', certain)
    h = h.withColumn('is_latest_change', F.when(certain, F.col('_latest')))
    h = h.withColumn('current_parameter_value', F.when(certain,
        F.first('new_value', ignorenulls=False).over(desc.rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing)))).drop('_latest')
    log.warning('Журнал: уникальный ID события в исходном ноутбуке отсутствует; '
                'удалены только точные повторы. current_parameter_value — последнее '
                'значение в журнале, не сверка с карточкой.')
    return h

def red_in_production(context):
    c = spark.table('valid_completed').join(context.select('model_ver_sid'), 'model_ver_sid')
    c = c.withColumn('is_red', F.upper(F.coalesce(F.col('valid_rslt_name'), F.lit(''))).rlike('КРАСН|RED'))
    c = c.withColumn('_dated', F.expr(real_date('valid_end_fact_dttm')) &
        (F.col('valid_end_fact_dttm') <= F.lit(AS_OF_TS).cast('timestamp')))
    p = Window.partitionBy('model_ver_sid')
    c = c.withColumn('_undated', F.sum((~F.col('_dated')).cast('int')).over(p))
    c = c.withColumn('_max', F.max(F.when(F.col('_dated'), F.col('valid_end_fact_dttm'))).over(p))
    c = c.withColumn('_latest', F.col('_dated') & (F.col('valid_end_fact_dttm') == F.col('_max')))
    c = c.withColumn('_ties', F.sum(F.coalesce(F.col('_latest').cast('int'), F.lit(0))).over(p))
    c = c.withColumn('is_latest_completed_validation', F.when(
        (F.col('_undated') == 0) & (F.col('_ties') == 1), F.col('_latest')))
    return c.filter('is_red').join(context.filter(F.expr(truth('model_ver_prom_expl_flag'))),
        'model_ver_sid').drop('_dated', '_undated', '_max', '_latest', '_ties')

def save_output(name, df):
    df = df.withColumn('run_id', F.lit(RUN_ID))
    path = CONFIG['output_root'].rstrip('/') + '/' + RUN_ID + '/' + name
    log.info('Сохранение %s', name)
    df.write.mode('errorifexists').parquet(path)
    outputs[name] = path
    return df

def main():
    global version_context, risk_summary, risk_details, control_rollup
    global version_report, model_change_log, red_production
    log.info('Старт %s; срез %s; конфигурация %s', RUN_ID, AS_OF_TS, CONFIG)
    version_context = prepare_scope()
    prepare_views()
    build_controls()
    risk_summary, risk_details, control_rollup, version_report = assemble_controls(version_context)
    model_change_log = changes(version_context).persist(StorageLevel.MEMORY_AND_DISK)
    cached.append(model_change_log)
    red_production = red_in_production(version_context)
    changes_by_version = model_change_log.groupBy('model_ver_sid').agg(
        F.count('*').alias('change_count'), F.countDistinct('parameter_sid').alias('changed_parameters'),
        F.min('change_dttm').alias('first_change_dttm'), F.max('change_dttm').alias('last_change_dttm'))
    version_report = version_report.join(changes_by_version, 'model_ver_sid', 'left').fillna(
        {'change_count': 0, 'changed_parameters': 0})
    change_summary = model_change_log.groupBy('parameter_sid', 'parameter_name').agg(
        F.count('*').alias('change_count'), F.countDistinct('model_ver_sid').alias('version_count'))
    frames = {'version_context': version_context, 'risk_summary': risk_summary,
        'risk_details': risk_details, 'control_rollup': control_rollup,
        'version_report': version_report, 'model_change_log': model_change_log,
        'change_summary': change_summary, 'red_production': red_production}
    for name, df in frames.items():
        save_output(name, df)
    save_output('run_diagnostics', spark.createDataFrame(audit, 'stage string, count long, note string'))
    control_rollup.orderBy('query_id').show(31, truncate=False)
    version_report.select('model_code', 'model_name', 'model_ver_sid', 'failed_controls',
        'failed_high_controls', 'change_count').orderBy(F.desc('failed_controls')).show(SHOW_ROWS, False)
    if CONFIG['export_excel']:
        # Потоковая запись без toPandas; превышение лимита — явная ошибка, не усечение.
        from openpyxl import Workbook
        excel = Workbook(write_only=True)
        sheet = excel.create_sheet('changes')
        sheet.append(model_change_log.columns)
        for i, row in enumerate(model_change_log.toLocalIterator(), 1):
            if i > CONFIG['excel_max_rows']:
                raise RuntimeError('Журнал превышает лимит Excel; полный Parquet уже сохранён')
            cells = []
            for x in row:
                s = None if x is None else str(x)
                if s is not None and len(s) > 32767:
                    raise RuntimeError('Значение длиннее лимита ячейки Excel; используйте Parquet')
                if s and s.startswith(('=', '+', '-', '@')):
                    s = "'" + s
                cells.append(s)
            sheet.append(cells)
        excel.save(local_dir / 'model_change_log.xlsx')
    log.info('Завершено. Результаты: %s', outputs)

if __name__ == '__main__':
    state = 'FAILED'
    try:
        main()
        state = 'SUCCESS'
    except Exception:
        log.exception('Аудит остановлен; частичные результаты перечислены в manifest.json')
        raise
    finally:
        (local_dir / 'manifest.json').write_text(json.dumps({
            'run_id': RUN_ID, 'status': state, 'as_of_dt': AS_OF_DT,
            'as_of_ts': AS_OF_TS, 'config': CONFIG, 'outputs': outputs,
            'diagnostics': audit}, ensure_ascii=False, indent=2), encoding='utf-8')
        for frame in cached:
            frame.unpersist()
        # Spark не останавливается: сессия ноутбука остаётся доступной.
