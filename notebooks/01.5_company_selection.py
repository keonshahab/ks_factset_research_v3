# Databricks notebook source

# MAGIC %md
# MAGIC # 01.5 — Company Selection: Top 20 Multi-Source Companies
# MAGIC
# MAGIC **Goal:** Select 20 companies for the Research & Deal Intelligence agent.
# MAGIC
# MAGIC **Logic:**
# MAGIC 1. Rank top 30 companies from `edg_metadata` by chunk count (filings anchor)
# MAGIC 2. Check whether each also appears in `fcst_metadata` (earnings) and `sa_metadata` (news)
# MAGIC 3. Take the top 20 that appear in **at least 2 of 3** tables
# MAGIC 4. If fewer than 20 qualify, backfill with top filing-only companies
# MAGIC
# MAGIC **Data notes:**
# MAGIC - Tickers are stored in `primary_symbols` (ArrayType) — we extract element `[0]`
# MAGIC - `company_name` exists in EDG and FCST but **not** in SA
# MAGIC - SA company name is resolved via the EDG anchor (left join by ticker)
# MAGIC
# MAGIC **Output:** `ticker, company_name, filing_chunks, earnings_chunks, news_chunks`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG_SCHEMA = "factset_vectors_do_not_edit_or_delete.vector"

edg_df = spark.table(f"{CATALOG_SCHEMA}.edg_metadata")
fcst_df = spark.table(f"{CATALOG_SCHEMA}.fcst_metadata")
sa_df = spark.table(f"{CATALOG_SCHEMA}.sa_metadata")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Top 30 companies from EDG by chunk count
# MAGIC
# MAGIC Extract ticker from `primary_symbols[0]`, group with `company_name`.

# COMMAND ----------

edg_top30 = (
    edg_df
    .withColumn("ticker", F.element_at("primary_symbols", 1))
    .where(F.col("ticker").isNotNull())
    .groupBy("ticker", "company_name")
    .agg(F.count("*").alias("filing_chunks"))
    .orderBy(F.desc("filing_chunks"))
    .limit(30)
)

print("=== Top 30 companies by EDG chunk count ===")
display(edg_top30)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Chunk counts in FCST and SA for the same tickers

# COMMAND ----------

fcst_counts = (
    fcst_df
    .withColumn("ticker", F.element_at("primary_symbols", 1))
    .where(F.col("ticker").isNotNull())
    .groupBy("ticker")
    .agg(F.count("*").alias("earnings_chunks"))
)

sa_counts = (
    sa_df
    .withColumn("ticker", F.element_at("primary_symbols", 1))
    .where(F.col("ticker").isNotNull())
    .groupBy("ticker")
    .agg(F.count("*").alias("news_chunks"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Left-join FCST and SA onto top-30 filing companies

# COMMAND ----------

combined = (
    edg_top30
    .join(fcst_counts, on="ticker", how="left")
    .join(sa_counts, on="ticker", how="left")
    .fillna(0, subset=["earnings_chunks", "news_chunks"])
    .withColumn(
        "source_count",
        F.lit(1)  # EDG always present
        + F.when(F.col("earnings_chunks") > 0, 1).otherwise(0)
        + F.when(F.col("news_chunks") > 0, 1).otherwise(0)
    )
)

print("=== Top 30 filings companies with cross-source overlap ===")
display(
    combined
    .select("ticker", "company_name", "filing_chunks", "earnings_chunks", "news_chunks", "source_count")
    .orderBy(F.desc("source_count"), F.desc("filing_chunks"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Final 20 — at least 2/3 sources, backfill if needed

# COMMAND ----------

multi_source = (
    combined
    .where(F.col("source_count") >= 2)
    .orderBy(F.desc("source_count"), F.desc("filing_chunks"))
)

multi_source_count = multi_source.count()
print(f"Companies in >= 2 sources: {multi_source_count}")

single_source = (
    combined
    .where(F.col("source_count") < 2)
    .orderBy(F.desc("filing_chunks"))
)

# COMMAND ----------

FINAL_N = 20

if multi_source_count >= FINAL_N:
    final_20 = multi_source.limit(FINAL_N)
    backfill_count = 0
else:
    backfill_needed = FINAL_N - multi_source_count
    backfill = single_source.limit(backfill_needed)
    final_20 = multi_source.unionByName(backfill)
    backfill_count = backfill_needed

print(f"Multi-source companies selected: {min(multi_source_count, FINAL_N)}")
print(f"Backfill (filing-only) companies: {backfill_count}")
print(f"Total: {FINAL_N}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Output

# COMMAND ----------

final_result = (
    final_20
    .select("ticker", "company_name", "filing_chunks", "earnings_chunks", "news_chunks")
    .orderBy(F.desc("filing_chunks"))
)

print("=" * 90)
print("SELECTED 20 COMPANIES FOR RESEARCH & DEAL INTELLIGENCE AGENT")
print("=" * 90)
display(final_result)

# COMMAND ----------

rows = final_result.collect()

print(f"\n{'Ticker':<20} {'Company':<35} {'Filings':>10} {'Earnings':>10} {'News':>10}")
print("-" * 90)
for row in rows:
    print(f"{row['ticker']:<20} {row['company_name']:<35} {row['filing_chunks']:>10,} {row['earnings_chunks']:>10,} {row['news_chunks']:>10,}")
print("-" * 90)
print(f"{'TOTAL':<20} {'':<35} {sum(r['filing_chunks'] for r in rows):>10,} {sum(r['earnings_chunks'] for r in rows):>10,} {sum(r['news_chunks'] for r in rows):>10,}")
