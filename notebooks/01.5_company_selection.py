# Databricks notebook source

# MAGIC %md
# MAGIC # 01.5 — Company Selection: Crosswalk + Capital Markets
# MAGIC
# MAGIC **Goal:** Select 20 companies for the Research & Deal Intelligence agent.
# MAGIC
# MAGIC **Logic:**
# MAGIC 1. Pull the 43 tickers from `ks_position_sample.vendor_data.factset_symbology_xref` (the crosswalk)
# MAGIC 2. For each, count chunks in `edg_metadata` / `fcst_metadata` / `sa_metadata` — matching on `ticker_region` vs `primary_symbols[0]`
# MAGIC 3. Search additional capital-markets names **not** in the crosswalk but present in `edg_metadata`
# MAGIC 4. Pick top 20: prioritise crosswalk companies with strong coverage, fill with capital-markets names
# MAGIC
# MAGIC **Note:** The crosswalk table has no `company_name` column — company names are resolved from `edg_metadata`.
# MAGIC
# MAGIC **Output:** `ticker_region, company_name, filing_chunks, earnings_chunks, news_chunks, total_chunks, source`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Row

VECTOR_SCHEMA = "factset_vectors_do_not_edit_or_delete.vector"

edg_df  = spark.table(f"{VECTOR_SCHEMA}.edg_metadata")
fcst_df = spark.table(f"{VECTOR_SCHEMA}.fcst_metadata")
sa_df   = spark.table(f"{VECTOR_SCHEMA}.sa_metadata")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Crosswalk: Tickers from the Position Book

# COMMAND ----------

xref_df = spark.sql("""
    SELECT ticker_region, factset_entity_id, isin
    FROM ks_position_sample.vendor_data.factset_symbology_xref
""")

xref_count = xref_df.count()
print(f"Crosswalk tickers: {xref_count}")
display(xref_df.orderBy("ticker_region"))

# COMMAND ----------

xref_rows    = xref_df.collect()
xref_tickers = [row["ticker_region"] for row in xref_rows]
print(f"Tickers ({len(xref_tickers)}): {sorted(xref_tickers)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Document Coverage for Crosswalk Tickers
# MAGIC
# MAGIC Match crosswalk `ticker_region` against `primary_symbols[0]` in each source table.
# MAGIC Company names are resolved from EDG.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a — EDG: Filings (+ resolve company_name)

# COMMAND ----------

edg_counts = (
    edg_df
    .withColumn("ticker", F.get("primary_symbols", 0))
    .where(F.col("ticker").isNotNull())
    .where(F.col("ticker").isin(xref_tickers))
    .groupBy("ticker")
    .agg(
        F.count("*").alias("filing_chunks"),
        F.first("company_name").alias("company_name"),
    )
    .withColumnRenamed("ticker", "ticker_region")
)

print(f"EDG tickers matched: {edg_counts.count()} / {len(xref_tickers)}")
display(edg_counts.orderBy(F.desc("filing_chunks")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b — FCST: Earnings Transcripts

# COMMAND ----------

fcst_counts = (
    fcst_df
    .withColumn("ticker", F.get("primary_symbols", 0))
    .where(F.col("ticker").isNotNull())
    .where(F.col("ticker").isin(xref_tickers))
    .groupBy("ticker")
    .agg(F.count("*").alias("earnings_chunks"))
    .withColumnRenamed("ticker", "ticker_region")
)

print(f"FCST tickers matched: {fcst_counts.count()} / {len(xref_tickers)}")
display(fcst_counts.orderBy(F.desc("earnings_chunks")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c — SA: News / StreetAccount

# COMMAND ----------

sa_counts = (
    sa_df
    .withColumn("ticker", F.get("primary_symbols", 0))
    .where(F.col("ticker").isNotNull())
    .where(F.col("ticker").isin(xref_tickers))
    .groupBy("ticker")
    .agg(F.count("*").alias("news_chunks"))
    .withColumnRenamed("ticker", "ticker_region")
)

print(f"SA tickers matched: {sa_counts.count()} / {len(xref_tickers)}")
display(sa_counts.orderBy(F.desc("news_chunks")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d — Combine: Full coverage view for crosswalk tickers

# COMMAND ----------

# Start from the full crosswalk ticker list so every ticker appears even with 0 chunks
xref_spark = spark.createDataFrame([Row(ticker_region=t) for t in xref_tickers])

crosswalk_coverage = (
    xref_spark
    .join(edg_counts,  on="ticker_region", how="left")
    .join(fcst_counts, on="ticker_region", how="left")
    .join(sa_counts,   on="ticker_region", how="left")
    .fillna(0, subset=["filing_chunks", "earnings_chunks", "news_chunks"])
    .withColumn(
        "total_chunks",
        F.col("filing_chunks") + F.col("earnings_chunks") + F.col("news_chunks"),
    )
    .orderBy(F.desc("total_chunks"))
)

print("=" * 110)
print("STEP 2 RESULT: Crosswalk tickers — document coverage")
print("=" * 110)
display(
    crosswalk_coverage.select(
        "ticker_region", "company_name",
        "filing_chunks", "earnings_chunks", "news_chunks", "total_chunks",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Additional Capital-Markets Names (Not in Crosswalk)
# MAGIC
# MAGIC Search `edg_metadata` for these names to fill out to 20 companies.

# COMMAND ----------

CAPITAL_MARKETS_NAMES = [
    "Citigroup", "Wells Fargo", "BlackRock", "State Street", "Charles Schwab",
    "Capital One", "S&P Global", "MSCI", "Moody's", "FactSet",
    "ICE", "CME Group", "Nasdaq", "KKR", "Apollo",
    "Blackstone", "Carlyle", "Ares", "Brookfield", "Lazard",
    "Evercore", "PJT Partners", "Jefferies", "Raymond James", "BNY Mellon",
    "Northern Trust", "PNC", "US Bancorp", "Truist",
]

# Remove any whose ticker already appears in the crosswalk
# (we don't know their tickers yet, so we search EDG first and exclude after)
xref_tickers_upper = {t.upper() for t in xref_tickers}

# COMMAND ----------

# Search edg_metadata for each name (case-insensitive contains)
# and get their ticker from primary_symbols[0]

edg_with_ticker = (
    edg_df
    .withColumn("ticker", F.get("primary_symbols", 0))
    .where(F.col("ticker").isNotNull())
)

cm_results = []

for search_name in CAPITAL_MARKETS_NAMES:
    matches = (
        edg_with_ticker
        .where(F.upper(F.col("company_name")).contains(search_name.upper()))
        .groupBy("ticker", "company_name")
        .agg(F.count("*").alias("filing_chunks"))
        .orderBy(F.desc("filing_chunks"))
        .limit(1)
        .collect()
    )
    if matches:
        row = matches[0]
        ticker = row["ticker"]
        # Skip if this ticker is already in the crosswalk
        if ticker.upper() in xref_tickers_upper:
            print(f"  SKIP:  '{search_name}' → {ticker} (already in crosswalk)")
            continue
        cm_results.append(Row(
            ticker_region=ticker,
            company_name=row["company_name"],
            filing_chunks=row["filing_chunks"],
            search_term=search_name,
        ))
        print(f"  FOUND: '{search_name}' → {ticker} / '{row['company_name']}' ({row['filing_chunks']:,} chunks)")
    else:
        print(f"  MISS:  '{search_name}' — not found in edg_metadata")

print(f"\nCapital-markets names found (net new): {len(cm_results)}")

# COMMAND ----------

# For the found CM companies, also count FCST + SA chunks

if cm_results:
    cm_df = spark.createDataFrame(cm_results)
    cm_ticker_list = [r.ticker_region for r in cm_results]

    cm_fcst = (
        fcst_df
        .withColumn("ticker", F.get("primary_symbols", 0))
        .where(F.col("ticker").isNotNull())
        .where(F.col("ticker").isin(cm_ticker_list))
        .groupBy("ticker")
        .agg(F.count("*").alias("earnings_chunks"))
        .withColumnRenamed("ticker", "ticker_region")
    )

    cm_sa = (
        sa_df
        .withColumn("ticker", F.get("primary_symbols", 0))
        .where(F.col("ticker").isNotNull())
        .where(F.col("ticker").isin(cm_ticker_list))
        .groupBy("ticker")
        .agg(F.count("*").alias("news_chunks"))
        .withColumnRenamed("ticker", "ticker_region")
    )

    cm_coverage = (
        cm_df.select("ticker_region", "company_name", "filing_chunks")
        .join(cm_fcst, on="ticker_region", how="left")
        .join(cm_sa,   on="ticker_region", how="left")
        .fillna(0, subset=["earnings_chunks", "news_chunks"])
        .withColumn(
            "total_chunks",
            F.col("filing_chunks") + F.col("earnings_chunks") + F.col("news_chunks"),
        )
        .orderBy(F.desc("total_chunks"))
    )

    print("=" * 110)
    print("STEP 3 RESULT: Capital-markets names — document coverage")
    print("=" * 110)
    display(
        cm_coverage.select(
            "ticker_region", "company_name",
            "filing_chunks", "earnings_chunks", "news_chunks", "total_chunks",
        )
    )
else:
    cm_coverage = None
    print("No capital-markets names found in EDG.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Final 20 Recommendation
# MAGIC
# MAGIC **Priority:**
# MAGIC 1. Crosswalk companies with strong document coverage (IN_POSITION_BOOK)
# MAGIC 2. Fill with capital-markets names for breadth (RESEARCH_ONLY)

# COMMAND ----------

FINAL_N = 20

# Rank crosswalk companies by total coverage (descending), exclude zero-coverage
crosswalk_ranked = (
    crosswalk_coverage
    .where(F.col("total_chunks") > 0)
    .withColumn("source", F.lit("IN_POSITION_BOOK"))
    .orderBy(F.desc("total_chunks"))
)

crosswalk_qualified = crosswalk_ranked.count()
print(f"Crosswalk companies with any document coverage: {crosswalk_qualified}")

# COMMAND ----------

if crosswalk_qualified >= FINAL_N:
    final_20 = crosswalk_ranked.limit(FINAL_N)
    fill_count = 0
else:
    fill_needed = FINAL_N - crosswalk_qualified
    print(f"Need {fill_needed} additional companies from capital-markets list")

    if cm_coverage is not None:
        cm_ranked = (
            cm_coverage
            .withColumn("source", F.lit("RESEARCH_ONLY"))
            .orderBy(F.desc("total_chunks"))
            .limit(fill_needed)
        )
        fill_count = min(cm_ranked.count(), fill_needed)

        select_cols = [
            "ticker_region", "company_name",
            "filing_chunks", "earnings_chunks", "news_chunks", "total_chunks",
            "source",
        ]
        final_20 = crosswalk_ranked.select(select_cols).unionByName(
            cm_ranked.select(select_cols)
        )
    else:
        final_20 = crosswalk_ranked
        fill_count = 0

print(f"IN_POSITION_BOOK: {min(crosswalk_qualified, FINAL_N)}")
print(f"RESEARCH_ONLY:    {fill_count}")
print(f"Total selected:   {min(crosswalk_qualified, FINAL_N) + fill_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Output

# COMMAND ----------

final_result = (
    final_20
    .select(
        "ticker_region", "company_name",
        "filing_chunks", "earnings_chunks", "news_chunks", "total_chunks",
        "source",
    )
    .orderBy(
        F.when(F.col("source") == "IN_POSITION_BOOK", 0).otherwise(1),
        F.desc("total_chunks"),
    )
)

print("=" * 120)
print("SELECTED 20 COMPANIES FOR RESEARCH & DEAL INTELLIGENCE AGENT")
print("=" * 120)
display(final_result)

# COMMAND ----------

rows = final_result.collect()

print(f"\n{'Ticker':<15} {'Company':<35} {'Filings':>10} {'Earnings':>10} {'News':>10} {'Total':>10}  {'Source'}")
print("-" * 120)
for row in rows:
    print(
        f"{row['ticker_region']:<15} "
        f"{(row['company_name'] or '')[:34]:<35} "
        f"{row['filing_chunks']:>10,} "
        f"{row['earnings_chunks']:>10,} "
        f"{row['news_chunks']:>10,} "
        f"{row['total_chunks']:>10,}  "
        f"{row['source']}"
    )
print("-" * 120)

in_book  = sum(1 for r in rows if r["source"] == "IN_POSITION_BOOK")
research = sum(1 for r in rows if r["source"] == "RESEARCH_ONLY")
print(f"\nIN_POSITION_BOOK: {in_book}   RESEARCH_ONLY: {research}   Total: {len(rows)}")
print(
    f"Chunks — Filings: {sum(r['filing_chunks'] for r in rows):>10,}   "
    f"Earnings: {sum(r['earnings_chunks'] for r in rows):>10,}   "
    f"News: {sum(r['news_chunks'] for r in rows):>10,}   "
    f"Grand Total: {sum(r['total_chunks'] for r in rows):>10,}"
)
