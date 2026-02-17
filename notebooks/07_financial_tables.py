# Databricks notebook source

# MAGIC %md
# MAGIC # 07 — Financial Tables
# MAGIC
# MAGIC **Purpose:** Build three gold-layer financial tables from FactSet Fundamentals (ff_v3)
# MAGIC and Estimates (fe_v4) data, filtered to companies in the config table.
# MAGIC
# MAGIC All tables are **full rebuild** (CREATE OR REPLACE) — no incremental mode needed
# MAGIC since financial data is relatively small.
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 0 | Parameters — optional ticker override |
# MAGIC | 1 | Verify source table schemas |
# MAGIC | 2 | Build `company_profile` from ff_entity_profiles |
# MAGIC | 3 | Build `company_financials` (quarterly + annual + LTM) |
# MAGIC | 4 | Build `consensus_estimates` (actuals + consensus + guidance) |
# MAGIC | 5 | Validation |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Parameters

# COMMAND ----------

dbutils.widgets.text("tickers", "", "Comma-separated ticker_regions (empty = all active)")

# COMMAND ----------

override = dbutils.widgets.get("tickers").strip()
if override:
    ticker_list = [t.strip() for t in override.split(",")]
    print(f"Override mode: processing {len(ticker_list)} tickers: {ticker_list}")
    company_filter = spark.sql(f"""
        SELECT * FROM ks_factset_research_v3.gold.demo_companies
        WHERE ticker_region IN ({','.join(f"'{t}'" for t in ticker_list)})
    """)
else:
    company_filter = spark.sql("SELECT * FROM ks_factset_research_v3.gold.v_active_companies")
    print(f"Config mode: processing {company_filter.count()} active companies")

company_filter.createOrReplaceTempView("target_companies")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Verify Source Table Schemas
# MAGIC
# MAGIC Print full schema and 5 sample rows for each FactSet source table.
# MAGIC Identify entity key columns and metric column names before building tables.

# COMMAND ----------

source_tables = [
    "delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_qf",
    "delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_af",
    "delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_ltm",
    "delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles",
    "delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_act_qf",
    "delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_conh_qf",
    "delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_guid_qf",
]

for table_name in source_tables:
    print("=" * 90)
    print(f"TABLE: {table_name}")
    print("=" * 90)
    df = spark.table(table_name)
    print(f"\n{'Column Name':<45} {'Data Type':<30} {'Nullable'}")
    print("-" * 90)
    for field in df.schema.fields:
        print(f"{field.name:<45} {str(field.dataType):<30} {field.nullable}")
    print(f"\nRow count: {df.count():,}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 — Sample Rows: ff_basic_qf (Quarterly Fundamentals)

# COMMAND ----------

display(spark.table("delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_qf").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 — Sample Rows: ff_basic_af (Annual Fundamentals)

# COMMAND ----------

display(spark.table("delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_af").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 — Sample Rows: ff_basic_ltm (Last Twelve Months)

# COMMAND ----------

display(spark.table("delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_ltm").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 — Sample Rows: ff_entity_profiles

# COMMAND ----------

display(spark.table("delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 — Sample Rows: fe_basic_act_qf (Estimates — Actuals)

# COMMAND ----------

display(spark.table("delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_act_qf").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.6 — Sample Rows: fe_basic_conh_qf (Estimates — Consensus)

# COMMAND ----------

display(spark.table("delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_conh_qf").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.7 — Sample Rows: fe_basic_guid_qf (Estimates — Guidance)

# COMMAND ----------

display(spark.table("delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_guid_qf").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: `company_profile`
# MAGIC
# MAGIC `ks_factset_research_v3.gold.company_profile` — one row per company, from
# MAGIC `ff_entity_profiles` joined to `target_companies` on `entity_id`.
# MAGIC
# MAGIC The `ff_entity_profiles` table stores profile data in a nested `ENTITY_PROFILE` column
# MAGIC (struct/map). We extract fields from it and pivot by `ENTITY_PROFILE_TYPE` if needed.
# MAGIC
# MAGIC **Output columns:** ticker, ticker_region, entity_id, company_name, country, sector,
# MAGIC industry, sub_industry, plus other useful profile fields.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.a — Inspect ENTITY_PROFILE structure
# MAGIC
# MAGIC Discover the nested field names inside the ENTITY_PROFILE column.

# COMMAND ----------

from pyspark.sql import functions as F

ep_df = spark.table("delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles")

# ENTITY_PROFILE is a STRING column — print raw samples to understand the format
print("=== ENTITY_PROFILE raw content (first 500 chars, 3 samples) ===\n")
samples = (
    ep_df
    .where(F.col("ENTITY_PROFILE").isNotNull())
    .where(F.length(F.col("ENTITY_PROFILE")) > 0)
    .select("FACTSET_ENTITY_ID", "ENTITY_PROFILE_TYPE", "ENTITY_PROFILE")
    .limit(3)
    .collect()
)
for i, row in enumerate(samples):
    print(f"--- Sample {i+1}: {row['FACTSET_ENTITY_ID']} (type={row['ENTITY_PROFILE_TYPE']}) ---")
    print(row["ENTITY_PROFILE"][:500])
    print()

# COMMAND ----------

# Show distinct ENTITY_PROFILE_TYPE values
display(ep_df.select("ENTITY_PROFILE_TYPE").distinct().orderBy("ENTITY_PROFILE_TYPE"))

# COMMAND ----------

# Sample: show ENTITY_PROFILE content for one of our target companies
display(spark.sql("""
    SELECT ep.*
    FROM delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles ep
    INNER JOIN target_companies tc ON tc.entity_id = ep.FACTSET_ENTITY_ID
    LIMIT 5
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.b — Build company_profile table
# MAGIC
# MAGIC The `ff_entity_profiles` table may not contain structured profile fields directly.
# MAGIC As a fallback, build the company profile from `sym_entity` (which has proper name,
# MAGIC country, entity type) joined with the config table's display_name.
# MAGIC
# MAGIC **Adjust this cell** after reviewing Step 2.a output — if `ENTITY_PROFILE` contains
# MAGIC parseable data (JSON, key-value pairs), extract relevant fields from it.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.company_profile
    USING DELTA
    AS
    SELECT
        tc.ticker,
        tc.ticker_region,
        tc.entity_id,
        se.ENTITY_PROPER_NAME                   AS company_name,
        se.ISO_COUNTRY                          AS country,
        se.ENTITY_TYPE                          AS entity_type,
        ep.ENTITY_PROFILE_TYPE                  AS profile_type,
        ep.ENTITY_PROFILE                       AS profile_raw,
        tc.fsym_id,
        tc.display_name
    FROM target_companies tc
    INNER JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON tc.entity_id = se.FACTSET_ENTITY_ID
    LEFT JOIN delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles ep
        ON tc.entity_id = ep.FACTSET_ENTITY_ID
""")

profile_count = spark.table("ks_factset_research_v3.gold.company_profile").count()
print(f"company_profile created with {profile_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: `company_financials`
# MAGIC
# MAGIC `ks_factset_research_v3.gold.company_financials` — UNION of quarterly, annual, and LTM
# MAGIC financial data from FactSet Fundamentals, filtered by `target_companies`.
# MAGIC
# MAGIC **Join key:** `FSYM_ID` from ff tables = `fsym_id` from target_companies.
# MAGIC
# MAGIC **Output columns:** ticker, company_name, entity_id, period_date, period_type,
# MAGIC fiscal_year, fiscal_quarter, revenue, ebitda, net_income, eps_diluted, total_debt,
# MAGIC interest_expense, operating_cash_flow, capex, free_cash_flow, total_assets,
# MAGIC shareholders_equity, currency, source_table.
# MAGIC
# MAGIC > **NOTE:** Column names below use standard FactSet FF naming conventions.
# MAGIC > Verify against the Step 1 schema output and adjust if any names differ.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.company_financials
    USING DELTA
    AS

    -- Quarterly fundamentals
    SELECT
        tc.ticker,
        tc.display_name                 AS company_name,
        tc.entity_id,
        ff.DATE                         AS period_date,
        'Q'                             AS period_type,
        ff.FF_FYR                       AS fiscal_year,
        ff.FF_FPNC                      AS fiscal_quarter,
        ff.FF_SALES                     AS revenue,
        ff.FF_OPER_INC                  AS operating_income,
        ff.FF_NET_INCOME                AS net_income,
        ff.FF_EPS_REPORTED              AS eps,
        ff.FF_DEBT                      AS total_debt,
        ff.FF_INT_EXP_TOT              AS interest_expense,
        ff.FF_FUNDS_OPER_GROSS          AS operating_cash_flow,
        ff.FF_INVEST_ACTIV_CF           AS investing_cash_flow,
        ff.FF_ASSETS                    AS total_assets,
        ff.FF_SHLDRS_EQ                 AS shareholders_equity,
        ff.CURRENCY                     AS currency,
        'ff_basic_qf'                   AS source_table
    FROM delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_qf ff
    INNER JOIN target_companies tc
        ON ff.FSYM_ID = tc.fsym_id

    UNION ALL

    -- Annual fundamentals
    SELECT
        tc.ticker,
        tc.display_name                 AS company_name,
        tc.entity_id,
        ff.DATE                         AS period_date,
        'A'                             AS period_type,
        ff.FF_FYR                       AS fiscal_year,
        NULL                            AS fiscal_quarter,
        ff.FF_SALES                     AS revenue,
        ff.FF_OPER_INC                  AS operating_income,
        ff.FF_NET_INCOME                AS net_income,
        ff.FF_EPS_REPORTED              AS eps,
        ff.FF_DEBT                      AS total_debt,
        ff.FF_INT_EXP_TOT              AS interest_expense,
        ff.FF_FUNDS_OPER_GROSS          AS operating_cash_flow,
        ff.FF_INVEST_ACTIV_CF           AS investing_cash_flow,
        ff.FF_ASSETS                    AS total_assets,
        ff.FF_SHLDRS_EQ                 AS shareholders_equity,
        ff.CURRENCY                     AS currency,
        'ff_basic_af'                   AS source_table
    FROM delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_af ff
    INNER JOIN target_companies tc
        ON ff.FSYM_ID = tc.fsym_id

    UNION ALL

    -- Last Twelve Months (no balance sheet or FF_FYR in this table)
    SELECT
        tc.ticker,
        tc.display_name                 AS company_name,
        tc.entity_id,
        ff.DATE                         AS period_date,
        'LTM'                           AS period_type,
        NULL                            AS fiscal_year,
        NULL                            AS fiscal_quarter,
        ff.FF_SALES                     AS revenue,
        ff.FF_OPER_INC                  AS operating_income,
        ff.FF_NET_INCOME                AS net_income,
        ff.FF_EPS_REPORTED              AS eps,
        NULL                            AS total_debt,
        ff.FF_INT_EXP_TOT              AS interest_expense,
        ff.FF_FUNDS_OPER_GROSS          AS operating_cash_flow,
        ff.FF_INVEST_ACTIV_CF           AS investing_cash_flow,
        NULL                            AS total_assets,
        NULL                            AS shareholders_equity,
        ff.CURRENCY                     AS currency,
        'ff_basic_ltm'                  AS source_table
    FROM delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_ltm ff
    INNER JOIN target_companies tc
        ON ff.FSYM_ID = tc.fsym_id
""")

fin_count = spark.table("ks_factset_research_v3.gold.company_financials").count()
print(f"company_financials created with {fin_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: `consensus_estimates`
# MAGIC
# MAGIC `ks_factset_research_v3.gold.consensus_estimates` — combines actuals, consensus, and
# MAGIC guidance from FactSet Estimates (fe_v4), filtered by `target_companies`.
# MAGIC
# MAGIC **Join key:** `FSYM_ID` from fe tables = `fsym_id` from target_companies.
# MAGIC
# MAGIC **Strategy:** Join actuals (fe_basic_act_qf) with consensus (fe_basic_conh_qf) and
# MAGIC guidance (fe_basic_guid_qf) on fsym_id + fiscal_year + fiscal_quarter, then unpivot
# MAGIC key metrics (EPS, Revenue, EBITDA) into a long-format table.
# MAGIC
# MAGIC **Output columns:** ticker, company_name, entity_id, period_date, fiscal_year,
# MAGIC fiscal_quarter, metric_name, actual_value, consensus_mean, consensus_high,
# MAGIC consensus_low, surprise, surprise_pct, beat_miss, guidance_low, guidance_high.
# MAGIC
# MAGIC > **NOTE:** Column names below use standard FactSet FE naming conventions.
# MAGIC > Verify against the Step 1 schema output and adjust if any names differ.

# COMMAND ----------

# Build temporary views for each estimates source, filtered to target companies
# Actuals: long format — one row per (FSYM_ID, FE_ITEM, FE_FP_END)
spark.sql("""
    CREATE OR REPLACE TEMP VIEW v_actuals AS
    SELECT
        tc.ticker,
        tc.display_name     AS company_name,
        tc.entity_id,
        tc.fsym_id,
        act.FE_FP_END       AS period_date,
        act.CURRENCY        AS currency,
        act.FE_ITEM         AS metric_name,
        act.ACTUAL_VALUE    AS actual_value
    FROM delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_act_qf act
    INNER JOIN target_companies tc
        ON act.FSYM_ID = tc.fsym_id
""")

# COMMAND ----------

# Consensus: long format — use most recent consensus snapshot per item/period
# (max CONS_START_DATE gives the latest consensus before the period ended)
# Materialize the max start date first to avoid a self-join on the Delta Share table
spark.sql("""
    CREATE OR REPLACE TEMP VIEW v_cons_latest AS
    SELECT FSYM_ID, FE_ITEM, FE_FP_END, MAX(CONS_START_DATE) AS max_start
    FROM delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_conh_qf
    GROUP BY FSYM_ID, FE_ITEM, FE_FP_END
""")
# Cache it so Spark doesn't fold it back into a self-join
spark.table("v_cons_latest").cache().count()

spark.sql("""
    CREATE OR REPLACE TEMP VIEW v_consensus AS
    SELECT
        con.FSYM_ID         AS fsym_id,
        con.FE_ITEM         AS metric_name,
        con.FE_FP_END       AS period_date,
        con.FE_MEAN         AS consensus_mean,
        con.FE_HIGH         AS consensus_high,
        con.FE_LOW          AS consensus_low,
        con.FE_NUM_EST      AS num_estimates
    FROM delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_conh_qf con
    INNER JOIN target_companies tc
        ON con.FSYM_ID = tc.fsym_id
    INNER JOIN v_cons_latest latest
        ON con.FSYM_ID = latest.FSYM_ID
        AND con.FE_ITEM = latest.FE_ITEM
        AND con.FE_FP_END = latest.FE_FP_END
        AND con.CONS_START_DATE = latest.max_start
""")

# COMMAND ----------

# Guidance: long format — pivot LOW/HIGH into columns per item/period
spark.sql("""
    CREATE OR REPLACE TEMP VIEW v_guidance AS
    SELECT
        g.FSYM_ID           AS fsym_id,
        g.FE_ITEM           AS metric_name,
        g.FE_FP_END         AS period_date,
        MAX(CASE WHEN g.GUIDANCE_TYPE = 'LOW'  THEN g.GUIDANCE_VALUE END) AS guidance_low,
        MAX(CASE WHEN g.GUIDANCE_TYPE = 'HIGH' THEN g.GUIDANCE_VALUE END) AS guidance_high
    FROM delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_guid_qf g
    INNER JOIN target_companies tc
        ON g.FSYM_ID = tc.fsym_id
    GROUP BY g.FSYM_ID, g.FE_ITEM, g.FE_FP_END
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 — Combine actuals + consensus + guidance
# MAGIC
# MAGIC All three source tables are already in long format keyed by
# MAGIC `(FSYM_ID, FE_ITEM, FE_FP_END)`. Join them and compute surprise metrics.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.consensus_estimates
    USING DELTA
    AS
    SELECT
        a.ticker,
        a.company_name,
        a.entity_id,
        a.period_date,
        a.metric_name,
        a.actual_value,
        c.consensus_mean,
        c.consensus_high,
        c.consensus_low,
        c.num_estimates,
        CASE WHEN a.actual_value IS NOT NULL AND c.consensus_mean IS NOT NULL
             THEN a.actual_value - c.consensus_mean END
                                        AS surprise,
        CASE WHEN a.actual_value IS NOT NULL AND c.consensus_mean IS NOT NULL
                  AND c.consensus_mean != 0
             THEN (a.actual_value - c.consensus_mean) / ABS(c.consensus_mean) * 100
             END                        AS surprise_pct,
        CASE WHEN a.actual_value IS NOT NULL AND c.consensus_mean IS NOT NULL THEN
             CASE WHEN a.actual_value > c.consensus_mean THEN 'BEAT'
                  WHEN a.actual_value < c.consensus_mean THEN 'MISS'
                  ELSE 'MEET' END
             END                        AS beat_miss,
        g.guidance_low,
        g.guidance_high
    FROM v_actuals a
    LEFT JOIN v_consensus c
        ON a.fsym_id = c.fsym_id
        AND a.metric_name = c.metric_name
        AND a.period_date = c.period_date
    LEFT JOIN v_guidance g
        ON a.fsym_id = g.fsym_id
        AND a.metric_name = g.metric_name
        AND a.period_date = g.period_date
""")

est_count = spark.table("ks_factset_research_v3.gold.consensus_estimates").count()
print(f"consensus_estimates created with {est_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.a — Row Counts

# COMMAND ----------

profile_count = spark.table("ks_factset_research_v3.gold.company_profile").count()
fin_count = spark.table("ks_factset_research_v3.gold.company_financials").count()
est_count = spark.table("ks_factset_research_v3.gold.consensus_estimates").count()

print(f"company_profile:      {profile_count:,} rows (expect ~20 companies)")
print(f"company_financials:   {fin_count:,} rows")
print(f"consensus_estimates:  {est_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.b — company_profile: All Rows

# COMMAND ----------

display(spark.sql("""
    SELECT
        ticker,
        ticker_region,
        entity_id,
        company_name,
        country,
        entity_type,
        display_name
    FROM ks_factset_research_v3.gold.company_profile
    ORDER BY ticker
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.c — NVDA: Last 8 Quarters Revenue / EBITDA / EPS

# COMMAND ----------

display(spark.sql("""
    SELECT
        ticker,
        period_date,
        fiscal_year,
        fiscal_quarter,
        revenue,
        operating_income,
        net_income,
        eps
    FROM ks_factset_research_v3.gold.company_financials
    WHERE ticker = 'NVDA'
      AND period_type = 'Q'
    ORDER BY period_date DESC
    LIMIT 8
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.d — NVDA: Last 4 Quarters Beat/Miss

# COMMAND ----------

display(spark.sql("""
    SELECT
        ticker,
        period_date,
        metric_name,
        actual_value,
        consensus_mean,
        surprise,
        surprise_pct,
        beat_miss
    FROM ks_factset_research_v3.gold.consensus_estimates
    WHERE ticker = 'NVDA'
    ORDER BY period_date DESC, metric_name
    LIMIT 12
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.e — Financials: Period Type Distribution

# COMMAND ----------

display(spark.sql("""
    SELECT
        period_type,
        COUNT(DISTINCT ticker) AS companies,
        COUNT(*) AS total_rows
    FROM ks_factset_research_v3.gold.company_financials
    GROUP BY period_type
    ORDER BY period_type
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.f — Estimates: Metric Distribution

# COMMAND ----------

display(spark.sql("""
    SELECT
        metric_name,
        COUNT(DISTINCT ticker) AS companies,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN beat_miss = 'BEAT' THEN 1 ELSE 0 END) AS beats,
        SUM(CASE WHEN beat_miss = 'MISS' THEN 1 ELSE 0 END) AS misses,
        SUM(CASE WHEN beat_miss = 'MEET' THEN 1 ELSE 0 END) AS meets
    FROM ks_factset_research_v3.gold.consensus_estimates
    GROUP BY metric_name
    ORDER BY metric_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.g — Final Summary

# COMMAND ----------

distinct_companies = spark.sql("""
    SELECT COUNT(DISTINCT ticker) FROM ks_factset_research_v3.gold.company_profile
""").collect()[0][0]

distinct_periods = spark.sql("""
    SELECT COUNT(DISTINCT CONCAT(ticker, '-', CAST(period_date AS STRING), '-', period_type))
    FROM ks_factset_research_v3.gold.company_financials
""").collect()[0][0]

distinct_estimates = spark.sql("""
    SELECT COUNT(*)
    FROM ks_factset_research_v3.gold.consensus_estimates
""").collect()[0][0]

print(f"Financial tables ready with {distinct_companies} companies, {distinct_periods:,} periods, {distinct_estimates:,} estimates")
