# Databricks notebook source

# MAGIC %md
# MAGIC # 13 — Genie Space Setup: Executive Portfolio Intelligence
# MAGIC
# MAGIC **Purpose:** Create metric views joining FactSet structured data (financials, estimates, profiles),
# MAGIC FactSet unstructured document summaries (filings, earnings transcripts, news), and sell-side
# MAGIC trading book holdings — then register them with a Databricks Genie Space for executive self-service.
# MAGIC
# MAGIC **Metric Views Created:**
# MAGIC | View | Description |
# MAGIC |------|-------------|
# MAGIC | `mv_portfolio_overview` | One row per company: exposure, financials, risk, research coverage |
# MAGIC | `mv_desk_exposure` | Granular breakdown by desk, asset class, book type |
# MAGIC | `mv_earnings_performance` | Beat/miss analysis for portfolio companies vs consensus |
# MAGIC | `mv_risk_compliance` | Positions with Volcker, restricted, MNPI, concentration flags |
# MAGIC | `mv_desk_pnl` | Daily P&L by desk with cumulative totals |
# MAGIC | `mv_financial_trends` | Quarterly financials with period-over-period changes |
# MAGIC | `mv_research_coverage` | Document coverage per company (filings, earnings, news) |
# MAGIC
# MAGIC **Genie Space ID:** `01f10bcf76ce1bdf98baf4ba025d26f2`
# MAGIC
# MAGIC **Prerequisites:** Notebooks 01–09 (FactSet structured + unstructured pipeline) and position data in `ks_position_sample`.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Parameters

# COMMAND ----------

CATALOG = "ks_factset_research_v3"
METRICS_SCHEMA = f"{CATALOG}.metrics"
GOLD_SCHEMA = f"{CATALOG}.gold"
DEMO_SCHEMA = f"{CATALOG}.demo"
POSITION_CATALOG = "ks_position_sample"
HOLDINGS_TABLE = f"{POSITION_CATALOG}.analytics.v_holdings_factset_enriched"
PNL_TABLE = f"{POSITION_CATALOG}.analytics.v_desk_pnl_daily"
RISK_TABLE = f"{POSITION_CATALOG}.analytics.v_risk_concentration_monitor"
GENIE_SPACE_ID = "01f10bcf76ce1bdf98baf4ba025d26f2"

print(f"Catalog:          {CATALOG}")
print(f"Metrics Schema:   {METRICS_SCHEMA}")
print(f"Holdings Table:   {HOLDINGS_TABLE}")
print(f"Genie Space ID:   {GENIE_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Create Metrics Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {METRICS_SCHEMA}")
print(f"✓ Schema {METRICS_SCHEMA} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Metric View 1 — Portfolio Overview
# MAGIC
# MAGIC Executive view: one row per company in the portfolio, joining latest holdings exposure
# MAGIC to company profile, latest LTM financials, risk flags, and research document coverage.
# MAGIC This is THE primary view for the Genie Space.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {METRICS_SCHEMA}.mv_portfolio_overview AS

WITH latest_date AS (
    SELECT MAX(as_of_date) AS max_date
    FROM {HOLDINGS_TABLE}
),

holdings_agg AS (
    SELECT
        h.ticker_region,
        SUM(h.notional_amount)          AS total_notional,
        SUM(h.market_value)             AS total_market_value,
        COUNT(*)                        AS position_count,
        COUNT(DISTINCT h.desk)          AS desk_count,
        CONCAT_WS(', ', COLLECT_SET(h.asset_class)) AS asset_class_list,
        SUM(h.`var`)                    AS total_var,
        SUM(h.dv01)                     AS total_dv01,
        SUM(h.cs01)                     AS total_cs01
    FROM {HOLDINGS_TABLE} h
    CROSS JOIN latest_date ld
    WHERE h.as_of_date = ld.max_date
    GROUP BY h.ticker_region
),

ticker_map AS (
    SELECT ticker_region, ticker
    FROM {GOLD_SCHEMA}.demo_companies
),

ltm_financials AS (
    SELECT f.*
    FROM {GOLD_SCHEMA}.company_financials f
    INNER JOIN (
        SELECT ticker, MAX(period_date) AS max_period_date
        FROM {GOLD_SCHEMA}.company_financials
        WHERE period_type = 'LTM'
        GROUP BY ticker
    ) latest ON f.ticker = latest.ticker AND f.period_date = latest.max_period_date
    WHERE f.period_type = 'LTM'
),

compliance AS (
    SELECT
        ticker_region,
        MAX(CASE WHEN volcker_flag THEN 1 ELSE 0 END)        AS has_volcker,
        MAX(CASE WHEN restricted_list_flag THEN 1 ELSE 0 END) AS has_restricted,
        MAX(CASE WHEN mnpi_flag THEN 1 ELSE 0 END)           AS has_mnpi,
        MAX(CASE WHEN concentration_flag THEN 1 ELSE 0 END)  AS has_concentration
    FROM {RISK_TABLE}
    GROUP BY ticker_region
),

filing_stats AS (
    SELECT
        ticker,
        COUNT(*)            AS filing_count,
        MAX(acceptance_date) AS latest_filing_date
    FROM {DEMO_SCHEMA}.filing_documents
    GROUP BY ticker
),

earnings_stats AS (
    SELECT
        ticker,
        COUNT(*)  AS earnings_count,
        MAX(LAST_DAY(MAKE_DATE(transcript_year, transcript_quarter * 3, 1))) AS latest_earnings_date
    FROM {DEMO_SCHEMA}.earnings_documents
    GROUP BY ticker
),

news_stats AS (
    SELECT
        ticker,
        COUNT(*)        AS news_count,
        MAX(story_date) AS latest_news_date
    FROM {DEMO_SCHEMA}.news_documents
    GROUP BY ticker
)

SELECT
    tm.ticker,
    ha.ticker_region,
    cp.company_name,
    cp.sector,
    cp.industry,
    cp.country,
    ha.total_notional,
    ha.total_market_value,
    ha.position_count,
    ha.desk_count,
    ha.asset_class_list,
    lf.revenue                                              AS ltm_revenue,
    lf.ebitda                                               AS ltm_ebitda,
    lf.net_income                                           AS ltm_net_income,
    lf.eps_diluted                                          AS ltm_eps,
    ROUND(lf.total_debt / NULLIF(lf.ebitda, 0), 2)         AS leverage_ratio,
    ha.total_var,
    ha.total_dv01,
    ha.total_cs01,
    CASE
        WHEN COALESCE(c.has_volcker, 0) + COALESCE(c.has_restricted, 0)
           + COALESCE(c.has_mnpi, 0) + COALESCE(c.has_concentration, 0) > 0
        THEN TRUE ELSE FALSE
    END                                                     AS has_compliance_flags,
    COALESCE(fs.filing_count, 0)                            AS filing_count,
    COALESCE(es.earnings_count, 0)                          AS earnings_count,
    COALESCE(ns.news_count, 0)                              AS news_count,
    fs.latest_filing_date,
    es.latest_earnings_date,
    ns.latest_news_date
FROM holdings_agg ha
JOIN ticker_map tm             ON ha.ticker_region = tm.ticker_region
LEFT JOIN {GOLD_SCHEMA}.company_profile cp ON tm.ticker = cp.ticker
LEFT JOIN ltm_financials lf    ON tm.ticker = lf.ticker
LEFT JOIN compliance c         ON ha.ticker_region = c.ticker_region
LEFT JOIN filing_stats fs      ON tm.ticker = fs.ticker
LEFT JOIN earnings_stats es    ON tm.ticker = es.ticker
LEFT JOIN news_stats ns        ON tm.ticker = ns.ticker
""")

print("✓ Created mv_portfolio_overview")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Metric View 2 — Desk Exposure Summary
# MAGIC
# MAGIC One row per (desk, ticker, asset_class, book_type) on the latest date.
# MAGIC Answers: "What does each desk hold, in what instruments, and at what risk?"

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {METRICS_SCHEMA}.mv_desk_exposure AS

WITH latest_date AS (
    SELECT MAX(as_of_date) AS max_date
    FROM {HOLDINGS_TABLE}
),

dc AS (
    SELECT ticker_region, ticker
    FROM {GOLD_SCHEMA}.demo_companies
)

SELECT
    h.as_of_date,
    h.desk,
    dc.ticker,
    h.ticker_region,
    cp.company_name,
    cp.sector,
    cp.industry,
    h.asset_class,
    h.book_type,
    SUM(h.notional_amount)      AS notional_amount,
    SUM(h.market_value)         AS market_value,
    SUM(h.`var`)                AS `var`,
    SUM(h.dv01)                 AS dv01,
    SUM(h.cs01)                 AS cs01,
    COUNT(*)                    AS position_count,
    COUNT(DISTINCT h.trader)    AS trader_count
FROM {HOLDINGS_TABLE} h
CROSS JOIN latest_date ld
JOIN dc ON h.ticker_region = dc.ticker_region
LEFT JOIN {GOLD_SCHEMA}.company_profile cp ON dc.ticker = cp.ticker
WHERE h.as_of_date = ld.max_date
GROUP BY
    h.as_of_date, h.desk, dc.ticker, h.ticker_region,
    cp.company_name, cp.sector, cp.industry,
    h.asset_class, h.book_type
""")

print("✓ Created mv_desk_exposure")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Metric View 3 — Earnings Performance
# MAGIC
# MAGIC Join consensus estimates to portfolio — shows beat/miss for companies we hold.
# MAGIC Answers: "Which of our holdings beat or missed earnings?"

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {METRICS_SCHEMA}.mv_earnings_performance AS

WITH latest_date AS (
    SELECT MAX(as_of_date) AS max_date
    FROM {HOLDINGS_TABLE}
),

portfolio AS (
    SELECT
        dc.ticker,
        SUM(h.notional_amount) AS firm_notional,
        COUNT(*)               AS firm_position_count
    FROM {HOLDINGS_TABLE} h
    CROSS JOIN latest_date ld
    JOIN {GOLD_SCHEMA}.demo_companies dc ON h.ticker_region = dc.ticker_region
    WHERE h.as_of_date = ld.max_date
    GROUP BY dc.ticker
)

SELECT
    ce.ticker,
    cp.company_name,
    cp.sector,
    YEAR(ce.period_date)    AS fiscal_year,
    QUARTER(ce.period_date) AS fiscal_quarter,
    ce.period_date,
    ce.metric_name,
    ce.actual_value,
    ce.consensus_mean,
    ce.surprise,
    ce.surprise_pct,
    ce.beat_miss,
    p.firm_notional,
    p.firm_position_count
FROM {GOLD_SCHEMA}.consensus_estimates ce
JOIN portfolio p ON ce.ticker = p.ticker
LEFT JOIN {GOLD_SCHEMA}.company_profile cp ON ce.ticker = cp.ticker
""")

print("✓ Created mv_earnings_performance")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Metric View 4 — Risk & Compliance
# MAGIC
# MAGIC Positions from the risk concentration monitor enriched with company profile.
# MAGIC Includes positions >$10M or with compliance flags.
# MAGIC Answers: "Which positions have regulatory or concentration risk?"

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {METRICS_SCHEMA}.mv_risk_compliance AS

SELECT
    r.as_of_date,
    dc.ticker,
    r.ticker_region,
    cp.company_name,
    cp.sector,
    cp.industry,
    r.desk,
    r.asset_class,
    r.book_type,
    r.notional_amount,
    r.market_value,
    r.volcker_flag,
    r.restricted_list_flag,
    r.mnpi_flag,
    r.concentration_flag,
    (CASE WHEN r.volcker_flag THEN 1 ELSE 0 END
   + CASE WHEN r.restricted_list_flag THEN 1 ELSE 0 END
   + CASE WHEN r.mnpi_flag THEN 1 ELSE 0 END
   + CASE WHEN r.concentration_flag THEN 1 ELSE 0 END)  AS flag_count,
    r.`var`,
    r.dv01,
    r.cs01,
    r.trader,
    r.counterparty
FROM {RISK_TABLE} r
JOIN {GOLD_SCHEMA}.demo_companies dc ON r.ticker_region = dc.ticker_region
LEFT JOIN {GOLD_SCHEMA}.company_profile cp ON dc.ticker = cp.ticker
""")

print("✓ Created mv_risk_compliance")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Metric View 5 — Desk P&L
# MAGIC
# MAGIC Daily P&L by desk with cumulative running totals and position context.
# MAGIC Answers: "How is each desk performing over time?"

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {METRICS_SCHEMA}.mv_desk_pnl AS

WITH daily_positions AS (
    SELECT
        as_of_date,
        desk,
        book_type,
        COUNT(*)                AS position_count,
        SUM(notional_amount)    AS total_notional
    FROM {HOLDINGS_TABLE}
    GROUP BY as_of_date, desk, book_type
)

SELECT
    pnl.as_of_date,
    pnl.desk,
    pnl.book_type,
    pnl.daily_pnl,
    SUM(pnl.daily_pnl) OVER (
        PARTITION BY pnl.desk, pnl.book_type
        ORDER BY pnl.as_of_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_pnl,
    COALESCE(dp.position_count, 0) AS position_count,
    COALESCE(dp.total_notional, 0) AS total_notional
FROM {PNL_TABLE} pnl
LEFT JOIN daily_positions dp
    ON  pnl.as_of_date = dp.as_of_date
    AND pnl.desk       = dp.desk
    AND pnl.book_type  = dp.book_type
""")

print("✓ Created mv_desk_pnl")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7: Metric View 6 — Financial Trends
# MAGIC
# MAGIC Quarterly financials for all tracked companies with period-over-period changes.
# MAGIC Flags which companies are currently in the portfolio.
# MAGIC Answers: "Show revenue/EBITDA/EPS trends for our holdings."

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {METRICS_SCHEMA}.mv_financial_trends AS

WITH latest_date AS (
    SELECT MAX(as_of_date) AS max_date
    FROM {HOLDINGS_TABLE}
),

portfolio AS (
    SELECT
        dc.ticker,
        SUM(h.notional_amount) AS firm_notional
    FROM {HOLDINGS_TABLE} h
    CROSS JOIN latest_date ld
    JOIN {GOLD_SCHEMA}.demo_companies dc ON h.ticker_region = dc.ticker_region
    WHERE h.as_of_date = ld.max_date
    GROUP BY dc.ticker
)

SELECT
    f.ticker,
    cp.company_name,
    cp.sector,
    cp.industry,
    f.period_date,
    YEAR(f.period_date)     AS fiscal_year,
    QUARTER(f.period_date)  AS fiscal_quarter,
    f.period_type,
    f.revenue,
    f.ebitda,
    f.net_income,
    f.eps_diluted,
    f.total_debt,
    f.operating_cash_flow,
    f.free_cash_flow,
    ROUND(f.total_debt / NULLIF(f.ebitda, 0), 2) AS leverage_ratio,
    ROUND(
        (f.revenue - LAG(f.revenue) OVER (PARTITION BY f.ticker ORDER BY f.period_date))
        / NULLIF(ABS(LAG(f.revenue) OVER (PARTITION BY f.ticker ORDER BY f.period_date)), 0) * 100,
    1) AS revenue_qoq_pct,
    ROUND(
        (f.ebitda - LAG(f.ebitda) OVER (PARTITION BY f.ticker ORDER BY f.period_date))
        / NULLIF(ABS(LAG(f.ebitda) OVER (PARTITION BY f.ticker ORDER BY f.period_date)), 0) * 100,
    1) AS ebitda_qoq_pct,
    ROUND(
        (f.eps_diluted - LAG(f.eps_diluted) OVER (PARTITION BY f.ticker ORDER BY f.period_date))
        / NULLIF(ABS(LAG(f.eps_diluted) OVER (PARTITION BY f.ticker ORDER BY f.period_date)), 0) * 100,
    1) AS eps_qoq_pct,
    COALESCE(p.firm_notional, 0)                                    AS firm_notional,
    CASE WHEN p.ticker IS NOT NULL THEN TRUE ELSE FALSE END         AS in_portfolio
FROM {GOLD_SCHEMA}.company_financials f
LEFT JOIN {GOLD_SCHEMA}.company_profile cp ON f.ticker = cp.ticker
LEFT JOIN portfolio p ON f.ticker = p.ticker
WHERE f.period_type = 'QF'
""")

print("✓ Created mv_financial_trends")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8: Metric View 7 — Research Coverage
# MAGIC
# MAGIC Document coverage per company: filings, earnings transcripts, and news.
# MAGIC Flags coverage gaps where portfolio companies lack adequate research documents.
# MAGIC Answers: "Which holdings have the most/least research coverage?"

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {METRICS_SCHEMA}.mv_research_coverage AS

WITH latest_date AS (
    SELECT MAX(as_of_date) AS max_date
    FROM {HOLDINGS_TABLE}
),

portfolio AS (
    SELECT
        dc.ticker,
        SUM(h.notional_amount) AS firm_notional
    FROM {HOLDINGS_TABLE} h
    CROSS JOIN latest_date ld
    JOIN {GOLD_SCHEMA}.demo_companies dc ON h.ticker_region = dc.ticker_region
    WHERE h.as_of_date = ld.max_date
    GROUP BY dc.ticker
),

filing_stats AS (
    SELECT
        ticker,
        COUNT(*)              AS filing_count,
        MAX(acceptance_date)  AS latest_filing_date,
        CONCAT_WS(', ', COLLECT_SET(doc_type_label)) AS filing_types
    FROM {DEMO_SCHEMA}.filing_documents
    GROUP BY ticker
),

earnings_stats AS (
    SELECT
        ticker,
        COUNT(*) AS earnings_count,
        MAX(LAST_DAY(MAKE_DATE(transcript_year, transcript_quarter * 3, 1))) AS latest_earnings_date
    FROM {DEMO_SCHEMA}.earnings_documents
    GROUP BY ticker
),

news_stats AS (
    SELECT
        ticker,
        COUNT(*)        AS news_count,
        MAX(story_date) AS latest_news_date,
        SUM(CASE WHEN has_analyst_commentary THEN 1 ELSE 0 END) AS analyst_commentary_count
    FROM {DEMO_SCHEMA}.news_documents
    GROUP BY ticker
),

all_tickers AS (
    SELECT DISTINCT ticker FROM {GOLD_SCHEMA}.company_profile
)

SELECT
    t.ticker,
    cp.company_name,
    cp.sector,
    CASE WHEN p.ticker IS NOT NULL THEN TRUE ELSE FALSE END     AS in_portfolio,
    COALESCE(p.firm_notional, 0)                                AS firm_notional,
    COALESCE(fs.filing_count, 0)                                AS filing_count,
    fs.latest_filing_date,
    fs.filing_types,
    COALESCE(es.earnings_count, 0)                              AS earnings_count,
    es.latest_earnings_date,
    COALESCE(ns.news_count, 0)                                  AS news_count,
    ns.latest_news_date,
    COALESCE(ns.analyst_commentary_count, 0)                    AS analyst_commentary_count,
    COALESCE(fs.filing_count, 0)
        + COALESCE(es.earnings_count, 0)
        + COALESCE(ns.news_count, 0)                            AS total_documents,
    CASE
        WHEN p.ticker IS NOT NULL AND (
            (CASE WHEN COALESCE(fs.filing_count, 0) > 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE(es.earnings_count, 0) > 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE(ns.news_count, 0) > 0 THEN 1 ELSE 0 END)
        ) < 2 THEN TRUE
        ELSE FALSE
    END                                                         AS has_coverage_gap
FROM all_tickers t
LEFT JOIN {GOLD_SCHEMA}.company_profile cp ON t.ticker = cp.ticker
LEFT JOIN portfolio p           ON t.ticker = p.ticker
LEFT JOIN filing_stats fs       ON t.ticker = fs.ticker
LEFT JOIN earnings_stats es     ON t.ticker = es.ticker
LEFT JOIN news_stats ns         ON t.ticker = ns.ticker
""")

print("✓ Created mv_research_coverage")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9: Add Comments & Descriptions to All Views
# MAGIC
# MAGIC Genie uses view and column comments to understand what the data means and how to query it.
# MAGIC Every column gets a business-context description with units, calculation logic, and usage guidance.

# COMMAND ----------

# -- View and column comments for all 7 metric views --
# Organized as: { view_name: { "__view__": "...", col: "...", ... } }

view_comments = {
    # -----------------------------------------------------------------------
    # mv_portfolio_overview
    # -----------------------------------------------------------------------
    "mv_portfolio_overview": {
        "__view__": (
            "Executive portfolio overview: one row per company showing total exposure, "
            "latest LTM financials, risk metrics, and research document coverage. "
            "Primary dimensions: sector, industry, country. "
            "Key measures: total_notional, leverage_ratio, total_var. "
            "Use this view for portfolio-level questions about exposure, financial health, and risk."
        ),
        "ticker": "Short ticker symbol (e.g. NVDA). Primary key for joining to FactSet structured and unstructured data.",
        "ticker_region": "FactSet-format ticker with region suffix (e.g. NVDA-US). Primary key for joining to holdings and position data.",
        "company_name": "Full legal company name from FactSet entity data.",
        "sector": "GICS sector classification (e.g. Information Technology, Health Care). Top level of sector > industry hierarchy.",
        "industry": "GICS industry classification (e.g. Semiconductors, Pharmaceuticals). Second level of sector > industry hierarchy.",
        "country": "Country of domicile for the company (e.g. United States, Germany).",
        "total_notional": "Total notional exposure in USD across all desks, asset classes, and book types for this company. Sum of all position notional amounts on the latest as_of_date.",
        "total_market_value": "Total mark-to-market value in USD across all positions for this company on the latest as_of_date.",
        "position_count": "Number of individual positions (rows) the firm holds in this company across all desks and asset classes.",
        "desk_count": "Number of distinct trading desks that hold positions in this company.",
        "asset_class_list": "Comma-separated list of distinct asset classes held for this company (e.g. Equity, Credit, Rates).",
        "ltm_revenue": "Last Twelve Months (LTM) total revenue in USD from FactSet Fundamentals. Trailing 12-month aggregate.",
        "ltm_ebitda": "Last Twelve Months (LTM) EBITDA in USD from FactSet Fundamentals. Earnings Before Interest, Taxes, Depreciation, and Amortization.",
        "ltm_net_income": "Last Twelve Months (LTM) net income in USD from FactSet Fundamentals.",
        "ltm_eps": "Last Twelve Months (LTM) diluted earnings per share in USD from FactSet Fundamentals.",
        "leverage_ratio": "Total Debt / EBITDA (LTM). Lower is better. Below 2x is conservative, 2-4x is moderate, above 4x is considered highly leveraged. NULL if EBITDA is zero or unavailable.",
        "total_var": "Total Value-at-Risk in USD across all positions for this company. Higher values indicate greater potential loss exposure.",
        "total_dv01": "Total DV01 (dollar value of a basis point) in USD. Measures interest rate sensitivity across all positions for this company.",
        "total_cs01": "Total CS01 (credit spread 01) in USD. Measures credit spread sensitivity across all positions for this company.",
        "has_compliance_flags": "TRUE if any position in this company has a Volcker, restricted list, MNPI, or concentration flag. FALSE otherwise.",
        "filing_count": "Number of SEC filing documents (10-K, 10-Q, 8-K, etc.) available for this company in the research document store.",
        "earnings_count": "Number of earnings call transcript documents available for this company in the research document store.",
        "news_count": "Number of StreetAccount news articles available for this company in the research document store.",
        "latest_filing_date": "Date of the most recent SEC filing available for this company.",
        "latest_earnings_date": "Approximate date of the most recent earnings transcript, derived from transcript year and quarter (last day of quarter-ending month).",
        "latest_news_date": "Date of the most recent StreetAccount news article for this company.",
    },

    # -----------------------------------------------------------------------
    # mv_desk_exposure
    # -----------------------------------------------------------------------
    "mv_desk_exposure": {
        "__view__": (
            "Desk-level exposure detail: one row per (desk, ticker, asset_class, book_type) combination "
            "on the latest as_of_date. Use this view for desk-level drill-downs, asset class breakdowns, "
            "and trader allocation analysis. For firm-level totals, use mv_portfolio_overview instead."
        ),
        "as_of_date": "Position date. All rows reflect the latest available snapshot date from the holdings system.",
        "desk": "Trading desk name (e.g. Equity Flow, Credit Trading). Represents the organizational unit managing positions.",
        "ticker": "Short ticker symbol (e.g. NVDA). Links to FactSet structured and unstructured data.",
        "ticker_region": "FactSet-format ticker with region suffix (e.g. NVDA-US). Links to holdings and position data.",
        "company_name": "Full legal company name from FactSet entity data.",
        "sector": "GICS sector classification. Top level of sector > industry hierarchy.",
        "industry": "GICS industry classification. Second level of sector > industry hierarchy.",
        "asset_class": "Asset class of the position (e.g. Equity, Credit, Rates, FX). Categorizes the instrument type.",
        "book_type": "Trading book classification (e.g. Trading, Banking, Hedge). Determines regulatory treatment.",
        "notional_amount": "Total notional exposure in USD for this desk-ticker-asset-book combination.",
        "market_value": "Total mark-to-market value in USD for this desk-ticker-asset-book combination.",
        "var": "Value-at-Risk in USD for this desk-ticker-asset-book combination. Measures potential loss.",
        "dv01": "DV01 (dollar value of a basis point) in USD. Measures interest rate sensitivity for this group.",
        "cs01": "CS01 (credit spread 01) in USD. Measures credit spread sensitivity for this group.",
        "position_count": "Number of individual positions in this desk-ticker-asset-book combination.",
        "trader_count": "Number of distinct traders managing positions in this desk-ticker-asset-book group.",
    },

    # -----------------------------------------------------------------------
    # mv_earnings_performance
    # -----------------------------------------------------------------------
    "mv_earnings_performance": {
        "__view__": (
            "Earnings beat/miss analysis for portfolio companies: each row is one metric (EPS, Revenue, etc.) "
            "for one company in one fiscal period, compared to consensus estimates. "
            "Only includes companies currently in the portfolio. "
            "Use this view to identify which holdings outperformed or underperformed expectations."
        ),
        "ticker": "Short ticker symbol (e.g. NVDA). Identifies the company.",
        "company_name": "Full legal company name from FactSet entity data.",
        "sector": "GICS sector classification for grouping and filtering.",
        "fiscal_year": "Fiscal year of the earnings period (e.g. 2024). Derived from the period end date.",
        "fiscal_quarter": "Fiscal quarter (1-4) of the earnings period. Derived from the period end date.",
        "period_date": "End date of the fiscal period for which the estimate applies.",
        "metric_name": "Name of the estimated metric (e.g. EPS, Revenue, EBITDA). Identifies what was forecasted.",
        "actual_value": "The actual reported value for this metric and period from FactSet Estimates actuals.",
        "consensus_mean": "The Wall Street consensus mean estimate for this metric and period. Average of analyst forecasts.",
        "surprise": "Absolute surprise: actual_value minus consensus_mean. Positive means the company exceeded expectations.",
        "surprise_pct": "Surprise as a percentage of consensus: (actual - consensus) / |consensus| * 100. Positive means beat.",
        "beat_miss": "Categorical label: Beat (actual > consensus), Miss (actual < consensus), or In-line (within threshold).",
        "firm_notional": "Total notional exposure in USD that the firm currently holds in this company. Provides exposure context for the earnings result.",
        "firm_position_count": "Number of positions the firm currently holds in this company.",
    },

    # -----------------------------------------------------------------------
    # mv_risk_compliance
    # -----------------------------------------------------------------------
    "mv_risk_compliance": {
        "__view__": (
            "Risk and compliance monitor: positions with notional above $10M or with regulatory/compliance flags. "
            "Sourced from the risk concentration monitor. Enriched with company profile for sector/industry context. "
            "Use this view to identify positions requiring compliance review or management attention."
        ),
        "as_of_date": "Date of the risk snapshot. Positions and flags are assessed as of this date.",
        "ticker": "Short ticker symbol (e.g. NVDA). Links to FactSet data.",
        "ticker_region": "FactSet-format ticker with region suffix (e.g. NVDA-US). Links to position systems.",
        "company_name": "Full legal company name from FactSet entity data.",
        "sector": "GICS sector classification for grouping flagged positions.",
        "industry": "GICS industry classification for detailed categorization of flagged positions.",
        "desk": "Trading desk holding the flagged position.",
        "asset_class": "Asset class of the flagged position (e.g. Equity, Credit).",
        "book_type": "Trading book classification (e.g. Trading, Banking). Affects regulatory treatment.",
        "notional_amount": "Notional exposure in USD for this flagged position.",
        "market_value": "Mark-to-market value in USD for this flagged position.",
        "volcker_flag": "TRUE if this position is flagged under the Volcker Rule (proprietary trading restrictions). Requires compliance review.",
        "restricted_list_flag": "TRUE if this company is on the firms restricted list (e.g. due to advisory relationship or pending deal). Trading may be prohibited.",
        "mnpi_flag": "TRUE if material non-public information (MNPI) has been identified for this company. Trading restrictions may apply.",
        "concentration_flag": "TRUE if this position exceeds concentration thresholds. May require risk committee approval.",
        "flag_count": "Total number of compliance flags on this position (0-4). Sum of volcker, restricted, MNPI, and concentration flags. Higher counts indicate more compliance concern.",
        "var": "Value-at-Risk in USD for this flagged position.",
        "dv01": "DV01 (dollar value of a basis point) in USD for this flagged position.",
        "cs01": "CS01 (credit spread 01) in USD for this flagged position.",
        "trader": "Name of the trader responsible for this position.",
        "counterparty": "Counterparty name for this position (relevant for OTC derivatives and credit positions).",
    },

    # -----------------------------------------------------------------------
    # mv_desk_pnl
    # -----------------------------------------------------------------------
    "mv_desk_pnl": {
        "__view__": (
            "Daily profit and loss by desk and book type, with cumulative running totals and position context. "
            "Use this view for P&L attribution, desk performance comparison, and trend analysis over time."
        ),
        "as_of_date": "Trading date for this P&L record.",
        "desk": "Trading desk name. Each desk has its own P&L tracked independently.",
        "book_type": "Trading book classification (e.g. Trading, Banking). P&L is tracked per book type.",
        "daily_pnl": "Profit or loss in USD realized on this specific trading day. Positive is profit, negative is loss.",
        "cumulative_pnl": "Running cumulative P&L in USD for this desk and book type, ordered by date from earliest to latest. Calculated as SUM(daily_pnl) over all prior dates.",
        "position_count": "Number of positions held on this desk and book type on this date. Provides context for P&L magnitude.",
        "total_notional": "Total notional exposure in USD for this desk and book type on this date. Provides context for P&L as a percentage of book size.",
    },

    # -----------------------------------------------------------------------
    # mv_financial_trends
    # -----------------------------------------------------------------------
    "mv_financial_trends": {
        "__view__": (
            "Quarterly financial trends for all tracked companies with period-over-period percentage changes. "
            "Includes revenue, EBITDA, EPS, debt, and cash flow metrics. Flags which companies are currently "
            "in the portfolio with firm_notional exposure. Use this view for financial trend analysis and screening."
        ),
        "ticker": "Short ticker symbol (e.g. NVDA). Identifies the company.",
        "company_name": "Full legal company name from FactSet entity data.",
        "sector": "GICS sector classification. Top level of sector > industry hierarchy.",
        "industry": "GICS industry classification. Second level of sector > industry hierarchy.",
        "period_date": "End date of the fiscal quarter (e.g. 2024-06-30 for Q2 2024).",
        "fiscal_year": "Calendar year of the period end date (e.g. 2024).",
        "fiscal_quarter": "Calendar quarter of the period end date (1-4).",
        "period_type": "Financial period type. Filtered to QF (quarterly fiscal) in this view.",
        "revenue": "Total quarterly revenue in USD from FactSet Fundamentals.",
        "ebitda": "Quarterly EBITDA in USD. Earnings Before Interest, Taxes, Depreciation, and Amortization.",
        "net_income": "Quarterly net income in USD from FactSet Fundamentals.",
        "eps_diluted": "Quarterly diluted earnings per share in USD.",
        "total_debt": "Total debt outstanding in USD at period end.",
        "operating_cash_flow": "Operating cash flow in USD for the quarter.",
        "free_cash_flow": "Free cash flow in USD for the quarter. Typically operating_cash_flow minus capex.",
        "leverage_ratio": "Total Debt / EBITDA for this quarter. Below 2x is conservative, 2-4x is moderate, above 4x is highly leveraged. NULL if EBITDA is zero.",
        "revenue_qoq_pct": "Revenue quarter-over-quarter change as a percentage. Calculated as (current - prior) / |prior| * 100. Positive means growth.",
        "ebitda_qoq_pct": "EBITDA quarter-over-quarter change as a percentage. Calculated as (current - prior) / |prior| * 100. Positive means improvement.",
        "eps_qoq_pct": "Diluted EPS quarter-over-quarter change as a percentage. Calculated as (current - prior) / |prior| * 100. Positive means growth.",
        "firm_notional": "Current total notional exposure in USD that the firm holds in this company. Zero if not in portfolio.",
        "in_portfolio": "TRUE if the firm currently holds positions in this company, FALSE otherwise.",
    },

    # -----------------------------------------------------------------------
    # mv_research_coverage
    # -----------------------------------------------------------------------
    "mv_research_coverage": {
        "__view__": (
            "Research document coverage per company: counts and latest dates for SEC filings, "
            "earnings transcripts, and StreetAccount news articles. Flags coverage gaps where "
            "portfolio companies have fewer than 2 document types available. "
            "Use this view to assess research readiness and identify gaps."
        ),
        "ticker": "Short ticker symbol (e.g. NVDA). Identifies the company.",
        "company_name": "Full legal company name from FactSet entity data.",
        "sector": "GICS sector classification for grouping.",
        "in_portfolio": "TRUE if the firm currently holds positions in this company, FALSE otherwise.",
        "firm_notional": "Current total notional exposure in USD. Zero if not in portfolio.",
        "filing_count": "Number of SEC filing documents available (10-K, 10-Q, 8-K, etc.).",
        "latest_filing_date": "Date of the most recent SEC filing available for this company.",
        "filing_types": "Comma-separated list of distinct SEC filing types available (e.g. 10-K, 10-Q, 8-K).",
        "earnings_count": "Number of earnings call transcript documents available.",
        "latest_earnings_date": "Approximate date of the most recent earnings transcript (last day of quarter-ending month).",
        "news_count": "Number of StreetAccount news articles available for this company.",
        "latest_news_date": "Date of the most recent StreetAccount news article.",
        "analyst_commentary_count": "Number of news articles that contain sell-side analyst commentary or ratings changes.",
        "total_documents": "Total document count across all types: filing_count + earnings_count + news_count.",
        "has_coverage_gap": "TRUE if company is in portfolio but has fewer than 2 document types (filings, earnings, news) available. Indicates inadequate research coverage for a held position.",
    },
}

# COMMAND ----------

# Apply all comments to views and columns

def _esc(text):
    """Escape single quotes for SQL string literals."""
    return text.replace("'", "\\'")

for view_name, columns in view_comments.items():
    fqn = f"{METRICS_SCHEMA}.{view_name}"
    view_desc = columns.pop("__view__")

    # Comment on the view itself
    spark.sql(f"COMMENT ON VIEW {fqn} IS '{_esc(view_desc)}'")

    # Comment on each column
    for col_name, col_desc in columns.items():
        spark.sql(f"ALTER VIEW {fqn} ALTER COLUMN `{col_name}` COMMENT '{_esc(col_desc)}'")

    col_count = len(columns)
    print(f"✓ {view_name}: view comment + {col_count} column comments applied")

print(f"\n✓ All view and column comments applied across {len(view_comments)} views")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 10: Register Metric Views with Genie Space
# MAGIC
# MAGIC Add all 7 metric views to the existing Genie Space so executives can query them
# MAGIC through natural language.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get existing Genie Space configuration
space = w.genie.get_space(GENIE_SPACE_ID)
print(f"Genie Space:     {space.title}")
print(f"Current tables:  {space.table_identifiers}")

# COMMAND ----------

# Define metric views to register
metric_views = [
    f"{METRICS_SCHEMA}.mv_portfolio_overview",
    f"{METRICS_SCHEMA}.mv_desk_exposure",
    f"{METRICS_SCHEMA}.mv_earnings_performance",
    f"{METRICS_SCHEMA}.mv_risk_compliance",
    f"{METRICS_SCHEMA}.mv_desk_pnl",
    f"{METRICS_SCHEMA}.mv_financial_trends",
    f"{METRICS_SCHEMA}.mv_research_coverage",
]

# Combine existing tables with new metric views (dedup)
existing_tables = list(space.table_identifiers or [])
existing_set = set(existing_tables)
new_views = [v for v in metric_views if v not in existing_set]
all_tables = existing_tables + new_views

print(f"Existing tables: {len(existing_tables)}")
print(f"New views:       {len(new_views)}")
print(f"Total:           {len(all_tables)}")

# Update the Genie Space with all tables
try:
    w.genie.update_space(
        space_id=GENIE_SPACE_ID,
        table_identifiers=all_tables,
    )
    print(f"\n✓ Registered {len(new_views)} metric views with Genie Space via SDK")
except Exception as sdk_err:
    print(f"SDK update_space failed: {sdk_err}")
    print("Falling back to REST API...")

    import requests

    host = w.config.host.rstrip("/")
    token = w.config.token

    response = requests.patch(
        f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}",
        headers={"Authorization": f"Bearer {token}"},
        json={"table_identifiers": all_tables},
    )

    if response.status_code == 200:
        print(f"✓ Registered {len(new_views)} metric views with Genie Space via REST API")
    else:
        print(f"✗ REST API failed ({response.status_code}): {response.text}")
        print("Manual action: add these views in the Genie Space UI:")
        for v in metric_views:
            print(f"  - {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 11: Add Sample Questions to Genie Space
# MAGIC
# MAGIC Register suggested questions so executives see helpful starting points when they open the space.

# COMMAND ----------

sample_questions = [
    # Portfolio + Financials
    "What is our total firm exposure by asset class?",
    "Which companies in our book have the highest leverage ratios?",
    "Show revenue growth trends for our top 10 holdings by notional",
    "Which holdings beat earnings estimates last quarter?",
    "What is our exposure to companies with declining EBITDA?",
    # Risk + Compliance
    "Which positions have compliance flags?",
    "Show our largest single-name concentrations",
    "What is our total Volcker-flagged exposure?",
    "Which desks have the most risk flags?",
    "Show P&L by desk for the last 5 days",
    # Research Coverage
    "Which holdings have the most recent SEC filings?",
    "How many earnings transcripts do we have per company?",
    "Which companies have analyst commentary in the news?",
    "Show document coverage gaps for holdings with no filings available",
    "What is the breakdown of filing types across our portfolio?",
]

# Attempt to add sample questions via SDK or REST API
try:
    host = w.config.host.rstrip("/")
    token = w.config.token

    import requests

    response = requests.patch(
        f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}",
        headers={"Authorization": f"Bearer {token}"},
        json={"sample_questions": sample_questions},
    )

    if response.status_code == 200:
        print(f"✓ Registered {len(sample_questions)} sample questions with Genie Space")
    else:
        print(f"REST API returned {response.status_code}: {response.text}")
        print("Sample questions may need to be added manually in the Genie Space UI.")
except Exception as e:
    print(f"Could not register sample questions programmatically: {e}")
    print("Add these manually in the Genie Space UI under 'Sample Questions':")

for i, q in enumerate(sample_questions, 1):
    print(f"  {i:2d}. {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 12: Set Genie Space Instructions
# MAGIC
# MAGIC Configure the system instruction that guides how the Genie Space answers executive questions.

# COMMAND ----------

genie_instruction = """You are an executive portfolio intelligence assistant for a sell-side trading desk.

DATA AVAILABLE:
- Portfolio Overview (mv_portfolio_overview): Total exposure, financial metrics, and risk for each company we hold
- Desk Exposure (mv_desk_exposure): Breakdown by desk, asset class, and book type
- Earnings Performance (mv_earnings_performance): Beat/miss analysis for portfolio companies vs consensus estimates
- Risk & Compliance (mv_risk_compliance): Positions with Volcker, restricted list, MNPI, or concentration flags
- Desk P&L (mv_desk_pnl): Daily profit and loss by desk with cumulative totals
- Financial Trends (mv_financial_trends): Quarterly revenue, EBITDA, EPS with period-over-period changes
- Research Coverage (mv_research_coverage): SEC filings, earnings transcripts, and news coverage per company

KEY IDENTIFIERS:
- ticker: Short symbol (e.g., NVDA)
- ticker_region: FactSet format (e.g., NVDA-US) - used in holdings data
- Companies are linked across all views by ticker

FORMATTING:
- Always format dollar amounts: $X.XB for billions, $X.XM for millions
- Format percentages to 1 decimal: 12.3%
- Leverage ratios to 1 decimal: 3.2x
- When showing tables, include sector for context
- For risk flags, highlight any TRUE flags prominently

PRIORITIZATION:
- For exposure questions, use mv_portfolio_overview (aggregated) not mv_desk_exposure (granular) unless desk-level detail is requested
- For earnings questions, default to the most recent quarter
- For financial trends, show last 4 quarters unless specified
- Always note if a company has compliance flags when discussing exposure"""

try:
    w.genie.update_space(
        space_id=GENIE_SPACE_ID,
        description=genie_instruction,
    )
    print("✓ Genie Space instructions updated via SDK")
except Exception as sdk_err:
    print(f"SDK update failed: {sdk_err}")
    print("Falling back to REST API...")

    try:
        response = requests.patch(
            f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}",
            headers={"Authorization": f"Bearer {token}"},
            json={"description": genie_instruction},
        )
        if response.status_code == 200:
            print("✓ Genie Space instructions updated via REST API")
        else:
            print(f"✗ REST API failed ({response.status_code}): {response.text}")
    except Exception as rest_err:
        print(f"✗ REST API also failed: {rest_err}")
        print("Set the instruction manually in the Genie Space UI under 'Instructions'.")

print(f"\nInstruction length: {len(genie_instruction)} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 13: Validation
# MAGIC
# MAGIC Verify all metric views, row/column counts, sample data, and Genie Space registration.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13a: Row and Column Counts for All Views

# COMMAND ----------

print("=" * 70)
print("METRIC VIEW SUMMARY")
print("=" * 70)

view_names = [
    "mv_portfolio_overview",
    "mv_desk_exposure",
    "mv_earnings_performance",
    "mv_risk_compliance",
    "mv_desk_pnl",
    "mv_financial_trends",
    "mv_research_coverage",
]

row_counts = {}
for vn in view_names:
    fqn = f"{METRICS_SCHEMA}.{vn}"
    df = spark.table(fqn)
    rc = df.count()
    cc = len(df.columns)
    row_counts[vn] = rc
    print(f"  {vn:<35s}  {rc:>6,} rows  x  {cc:>2} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13b: Portfolio Overview — Executive Summary

# COMMAND ----------

print("=" * 70)
print("PORTFOLIO OVERVIEW — sorted by total_notional DESC")
print("=" * 70)

df_overview = spark.table(f"{METRICS_SCHEMA}.mv_portfolio_overview").orderBy("total_notional", ascending=False)
display(df_overview)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13c: Earnings Performance — Last Quarter Beat/Miss for Top 5 Holdings

# COMMAND ----------

from pyspark.sql import functions as F

# Get latest quarter in the estimates data
latest_q = (
    spark.table(f"{METRICS_SCHEMA}.mv_earnings_performance")
    .select(F.max("period_date").alias("max_pd"))
    .collect()[0]["max_pd"]
)
print(f"Latest earnings quarter end date: {latest_q}")

# Top 5 holdings by firm_notional, latest quarter
df_earnings = (
    spark.table(f"{METRICS_SCHEMA}.mv_earnings_performance")
    .filter(F.col("period_date") == latest_q)
    .orderBy("firm_notional", ascending=False)
    .limit(25)  # multiple metrics per ticker, so get enough rows
)
display(df_earnings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13d: Risk & Compliance — Flagged Positions by Flag Type

# COMMAND ----------

print("=" * 70)
print("RISK & COMPLIANCE — Flagged Position Counts by Type")
print("=" * 70)

df_risk = spark.table(f"{METRICS_SCHEMA}.mv_risk_compliance")

volcker_count = df_risk.filter(F.col("volcker_flag") == True).count()
restricted_count = df_risk.filter(F.col("restricted_list_flag") == True).count()
mnpi_count = df_risk.filter(F.col("mnpi_flag") == True).count()
concentration_count = df_risk.filter(F.col("concentration_flag") == True).count()
total_flagged = df_risk.filter(F.col("flag_count") > 0).count()
total_positions = df_risk.count()

print(f"  Volcker flags:        {volcker_count:>5}")
print(f"  Restricted list:      {restricted_count:>5}")
print(f"  MNPI flags:           {mnpi_count:>5}")
print(f"  Concentration flags:  {concentration_count:>5}")
print(f"  ---")
print(f"  Total flagged:        {total_flagged:>5} / {total_positions} positions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13e: Verify Genie Space Registration

# COMMAND ----------

# Reload Genie Space config to verify
space_check = w.genie.get_space(GENIE_SPACE_ID)
print(f"Genie Space: {space_check.title}")
print(f"Space ID:    {GENIE_SPACE_ID}")
print(f"\nRegistered tables ({len(space_check.table_identifiers or [])}):")
for t in (space_check.table_identifiers or []):
    marker = " ← metric view" if ".metrics." in t else ""
    print(f"  - {t}{marker}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13f: Test Genie Queries (Programmatic)

# COMMAND ----------

# Attempt programmatic Genie query if the API supports it
test_questions = [
    "What is our total firm exposure by sector?",
    "Which companies have the highest leverage ratios?",
    "Show positions with compliance flags",
]

print("=" * 70)
print("GENIE SPACE — Test Query Attempt")
print("=" * 70)

try:
    for q in test_questions:
        print(f"\nQ: {q}")
        result = w.genie.start_conversation(space_id=GENIE_SPACE_ID, content=q)
        print(f"  → Conversation started: {result.conversation_id}")
        print(f"  → Message ID: {result.message_id}")
        print("  (Check the Genie Space UI for full results)")
except Exception as e:
    print(f"\nProgrammatic query not available: {e}")
    print("\nTo test manually, open the Genie Space in Databricks:")
    print(f"  1. Navigate to the Genie Space (ID: {GENIE_SPACE_ID})")
    print(f"  2. Try these questions:")
    for q in test_questions:
        print(f"     - {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13g: Final Summary

# COMMAND ----------

rc_summary = ", ".join([f"{vn} ({row_counts[vn]:,})" for vn in view_names])

print()
print("=" * 70)
print("GENIE SPACE SETUP COMPLETE")
print("=" * 70)
print(f"  Space ID:        {GENIE_SPACE_ID}")
print(f"  Metric Views:    7 registered")
print(f"  Total Rows:      {rc_summary}")
print(f"  Sample Questions: {len(sample_questions)} configured")
print(f"  Instructions:    {len(genie_instruction)} chars")
print("=" * 70)
