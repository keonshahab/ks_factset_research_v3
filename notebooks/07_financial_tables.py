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

ep_df = spark.table("delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles")

# Show the schema of the ENTITY_PROFILE column (struct fields)
profile_field = [f for f in ep_df.schema.fields if f.name == "ENTITY_PROFILE"][0]
print(f"ENTITY_PROFILE data type: {profile_field.dataType}")
print()

# If it's a struct, list its sub-fields
if hasattr(profile_field.dataType, 'fields'):
    print(f"{'Sub-field Name':<45} {'Data Type':<30}")
    print("-" * 75)
    for sub in profile_field.dataType.fields:
        print(f"{sub.name:<45} {str(sub.dataType):<30}")
else:
    print("ENTITY_PROFILE is not a struct — printing raw type for inspection:")
    print(profile_field.dataType)

# COMMAND ----------

# Show distinct ENTITY_PROFILE_TYPE values
display(ep_df.select("ENTITY_PROFILE_TYPE").distinct().orderBy("ENTITY_PROFILE_TYPE"))

# COMMAND ----------

# Sample: show ENTITY_PROFILE content for one of our target companies
display(spark.sql("""
    SELECT ep.*
    FROM delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles ep
    INNER JOIN target_companies tc ON tc.entity_id = ep.FACTSET_ENTITY_ID
    LIMIT 3
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.b — Build company_profile table
# MAGIC
# MAGIC Extract fields from the `ENTITY_PROFILE` struct. Adjust field names below
# MAGIC based on the sub-field discovery in Step 2.a.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.company_profile
    USING DELTA
    AS
    SELECT
        tc.ticker,
        tc.ticker_region,
        tc.entity_id,
        ep.ENTITY_PROFILE.ENTITY_PROPER_NAME    AS company_name,
        ep.ENTITY_PROFILE.ISO_COUNTRY            AS country,
        ep.ENTITY_PROFILE.SECTOR_CODE            AS sector,
        ep.ENTITY_PROFILE.INDUSTRY_CODE          AS industry,
        ep.ENTITY_PROFILE.SUB_INDUSTRY_CODE      AS sub_industry,
        ep.ENTITY_PROFILE.ENTITY_TYPE            AS entity_type,
        ep.ENTITY_PROFILE.YEAR_FOUNDED           AS year_founded,
        ep.ENTITY_PROFILE.ISO_COUNTRY_INCORP     AS country_incorporated,
        ep.ENTITY_PROFILE.ENTITY_SUB_TYPE        AS entity_sub_type,
        ep.ENTITY_PROFILE_TYPE,
        tc.fsym_id,
        tc.display_name
    FROM target_companies tc
    INNER JOIN delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles ep
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
        ff.FF_FY                        AS fiscal_year,
        ff.FF_FQN                       AS fiscal_quarter,
        ff.FF_SALES                     AS revenue,
        ff.FF_EBITDA                    AS ebitda,
        ff.FF_NET_INC                   AS net_income,
        ff.FF_EPS_DIL                   AS eps_diluted,
        ff.FF_DEBT_TOT                  AS total_debt,
        ff.FF_INT_EXP                   AS interest_expense,
        ff.FF_OPER_CF                   AS operating_cash_flow,
        ff.FF_CAPEX                     AS capex,
        ff.FF_FREE_CF                   AS free_cash_flow,
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
        ff.FF_FY                        AS fiscal_year,
        NULL                            AS fiscal_quarter,
        ff.FF_SALES                     AS revenue,
        ff.FF_EBITDA                    AS ebitda,
        ff.FF_NET_INC                   AS net_income,
        ff.FF_EPS_DIL                   AS eps_diluted,
        ff.FF_DEBT_TOT                  AS total_debt,
        ff.FF_INT_EXP                   AS interest_expense,
        ff.FF_OPER_CF                   AS operating_cash_flow,
        ff.FF_CAPEX                     AS capex,
        ff.FF_FREE_CF                   AS free_cash_flow,
        ff.FF_ASSETS                    AS total_assets,
        ff.FF_SHLDRS_EQ                 AS shareholders_equity,
        ff.CURRENCY                     AS currency,
        'ff_basic_af'                   AS source_table
    FROM delta_share_factset_do_not_delete_or_edit.ff_v3.ff_basic_af ff
    INNER JOIN target_companies tc
        ON ff.FSYM_ID = tc.fsym_id

    UNION ALL

    -- Last Twelve Months
    SELECT
        tc.ticker,
        tc.display_name                 AS company_name,
        tc.entity_id,
        ff.DATE                         AS period_date,
        'LTM'                           AS period_type,
        ff.FF_FY                        AS fiscal_year,
        NULL                            AS fiscal_quarter,
        ff.FF_SALES                     AS revenue,
        ff.FF_EBITDA                    AS ebitda,
        ff.FF_NET_INC                   AS net_income,
        ff.FF_EPS_DIL                   AS eps_diluted,
        ff.FF_DEBT_TOT                  AS total_debt,
        ff.FF_INT_EXP                   AS interest_expense,
        ff.FF_OPER_CF                   AS operating_cash_flow,
        ff.FF_CAPEX                     AS capex,
        ff.FF_FREE_CF                   AS free_cash_flow,
        ff.FF_ASSETS                    AS total_assets,
        ff.FF_SHLDRS_EQ                 AS shareholders_equity,
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
spark.sql("""
    CREATE OR REPLACE TEMP VIEW v_actuals AS
    SELECT
        tc.ticker,
        tc.display_name     AS company_name,
        tc.entity_id,
        tc.fsym_id,
        act.DATE            AS period_date,
        act.CURRENCY        AS currency,
        act.FE_FY           AS fiscal_year,
        act.FE_FQN          AS fiscal_quarter,
        act.FE_EPS          AS actual_eps,
        act.FE_SALES        AS actual_revenue,
        act.FE_EBITDA       AS actual_ebitda
    FROM delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_act_qf act
    INNER JOIN target_companies tc
        ON act.FSYM_ID = tc.fsym_id
""")

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TEMP VIEW v_consensus AS
    SELECT
        tc.fsym_id,
        con.FE_FY           AS fiscal_year,
        con.FE_FQN          AS fiscal_quarter,
        con.FE_MEAN_EPS     AS cons_mean_eps,
        con.FE_HIGH_EPS     AS cons_high_eps,
        con.FE_LOW_EPS      AS cons_low_eps,
        con.FE_MEAN_SALES   AS cons_mean_revenue,
        con.FE_HIGH_SALES   AS cons_high_revenue,
        con.FE_LOW_SALES    AS cons_low_revenue,
        con.FE_MEAN_EBITDA  AS cons_mean_ebitda,
        con.FE_HIGH_EBITDA  AS cons_high_ebitda,
        con.FE_LOW_EBITDA   AS cons_low_ebitda
    FROM delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_conh_qf con
    INNER JOIN target_companies tc
        ON con.FSYM_ID = tc.fsym_id
""")

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TEMP VIEW v_guidance AS
    SELECT
        tc.fsym_id,
        guid.FE_FY          AS fiscal_year,
        guid.FE_FQN         AS fiscal_quarter,
        guid.FE_GUID_EPS_LOW    AS guid_eps_low,
        guid.FE_GUID_EPS_HIGH   AS guid_eps_high,
        guid.FE_GUID_SALES_LOW  AS guid_revenue_low,
        guid.FE_GUID_SALES_HIGH AS guid_revenue_high,
        guid.FE_GUID_EBITDA_LOW AS guid_ebitda_low,
        guid.FE_GUID_EBITDA_HIGH AS guid_ebitda_high
    FROM delta_share_factset_do_not_delete_or_edit.fe_v4.fe_basic_guid_qf guid
    INNER JOIN target_companies tc
        ON guid.FSYM_ID = tc.fsym_id
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 — Combine and unpivot into long format

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.consensus_estimates
    USING DELTA
    AS

    -- EPS
    SELECT
        a.ticker,
        a.company_name,
        a.entity_id,
        a.period_date,
        a.fiscal_year,
        a.fiscal_quarter,
        'EPS'                           AS metric_name,
        a.actual_eps                    AS actual_value,
        c.cons_mean_eps                 AS consensus_mean,
        c.cons_high_eps                 AS consensus_high,
        c.cons_low_eps                  AS consensus_low,
        CASE WHEN a.actual_eps IS NOT NULL AND c.cons_mean_eps IS NOT NULL
             THEN a.actual_eps - c.cons_mean_eps END
                                        AS surprise,
        CASE WHEN a.actual_eps IS NOT NULL AND c.cons_mean_eps IS NOT NULL
                  AND c.cons_mean_eps != 0
             THEN (a.actual_eps - c.cons_mean_eps) / ABS(c.cons_mean_eps) * 100
             END                        AS surprise_pct,
        CASE WHEN a.actual_eps IS NOT NULL AND c.cons_mean_eps IS NOT NULL THEN
             CASE WHEN a.actual_eps > c.cons_mean_eps THEN 'BEAT'
                  WHEN a.actual_eps < c.cons_mean_eps THEN 'MISS'
                  ELSE 'MEET' END
             END                        AS beat_miss,
        g.guid_eps_low                  AS guidance_low,
        g.guid_eps_high                 AS guidance_high
    FROM v_actuals a
    LEFT JOIN v_consensus c
        ON a.fsym_id = c.fsym_id
        AND a.fiscal_year = c.fiscal_year
        AND a.fiscal_quarter = c.fiscal_quarter
    LEFT JOIN v_guidance g
        ON a.fsym_id = g.fsym_id
        AND a.fiscal_year = g.fiscal_year
        AND a.fiscal_quarter = g.fiscal_quarter

    UNION ALL

    -- Revenue
    SELECT
        a.ticker,
        a.company_name,
        a.entity_id,
        a.period_date,
        a.fiscal_year,
        a.fiscal_quarter,
        'Revenue'                       AS metric_name,
        a.actual_revenue                AS actual_value,
        c.cons_mean_revenue             AS consensus_mean,
        c.cons_high_revenue             AS consensus_high,
        c.cons_low_revenue              AS consensus_low,
        CASE WHEN a.actual_revenue IS NOT NULL AND c.cons_mean_revenue IS NOT NULL
             THEN a.actual_revenue - c.cons_mean_revenue END
                                        AS surprise,
        CASE WHEN a.actual_revenue IS NOT NULL AND c.cons_mean_revenue IS NOT NULL
                  AND c.cons_mean_revenue != 0
             THEN (a.actual_revenue - c.cons_mean_revenue) / ABS(c.cons_mean_revenue) * 100
             END                        AS surprise_pct,
        CASE WHEN a.actual_revenue IS NOT NULL AND c.cons_mean_revenue IS NOT NULL THEN
             CASE WHEN a.actual_revenue > c.cons_mean_revenue THEN 'BEAT'
                  WHEN a.actual_revenue < c.cons_mean_revenue THEN 'MISS'
                  ELSE 'MEET' END
             END                        AS beat_miss,
        g.guid_revenue_low              AS guidance_low,
        g.guid_revenue_high             AS guidance_high
    FROM v_actuals a
    LEFT JOIN v_consensus c
        ON a.fsym_id = c.fsym_id
        AND a.fiscal_year = c.fiscal_year
        AND a.fiscal_quarter = c.fiscal_quarter
    LEFT JOIN v_guidance g
        ON a.fsym_id = g.fsym_id
        AND a.fiscal_year = g.fiscal_year
        AND a.fiscal_quarter = g.fiscal_quarter

    UNION ALL

    -- EBITDA
    SELECT
        a.ticker,
        a.company_name,
        a.entity_id,
        a.period_date,
        a.fiscal_year,
        a.fiscal_quarter,
        'EBITDA'                        AS metric_name,
        a.actual_ebitda                 AS actual_value,
        c.cons_mean_ebitda              AS consensus_mean,
        c.cons_high_ebitda              AS consensus_high,
        c.cons_low_ebitda               AS consensus_low,
        CASE WHEN a.actual_ebitda IS NOT NULL AND c.cons_mean_ebitda IS NOT NULL
             THEN a.actual_ebitda - c.cons_mean_ebitda END
                                        AS surprise,
        CASE WHEN a.actual_ebitda IS NOT NULL AND c.cons_mean_ebitda IS NOT NULL
                  AND c.cons_mean_ebitda != 0
             THEN (a.actual_ebitda - c.cons_mean_ebitda) / ABS(c.cons_mean_ebitda) * 100
             END                        AS surprise_pct,
        CASE WHEN a.actual_ebitda IS NOT NULL AND c.cons_mean_ebitda IS NOT NULL THEN
             CASE WHEN a.actual_ebitda > c.cons_mean_ebitda THEN 'BEAT'
                  WHEN a.actual_ebitda < c.cons_mean_ebitda THEN 'MISS'
                  ELSE 'MEET' END
             END                        AS beat_miss,
        g.guid_ebitda_low               AS guidance_low,
        g.guid_ebitda_high              AS guidance_high
    FROM v_actuals a
    LEFT JOIN v_consensus c
        ON a.fsym_id = c.fsym_id
        AND a.fiscal_year = c.fiscal_year
        AND a.fiscal_quarter = c.fiscal_quarter
    LEFT JOIN v_guidance g
        ON a.fsym_id = g.fsym_id
        AND a.fiscal_year = g.fiscal_year
        AND a.fiscal_quarter = g.fiscal_quarter
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
        sector,
        industry,
        sub_industry
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
        ebitda,
        eps_diluted
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
        fiscal_year,
        fiscal_quarter,
        metric_name,
        actual_value,
        consensus_mean,
        surprise,
        surprise_pct,
        beat_miss
    FROM ks_factset_research_v3.gold.consensus_estimates
    WHERE ticker = 'NVDA'
    ORDER BY fiscal_year DESC, fiscal_quarter DESC, metric_name
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
