# Databricks notebook source

# MAGIC %md
# MAGIC # 02 — Company Selection: Top 20 Multi-Source Companies
# MAGIC
# MAGIC **Goal:** Select 20 companies for the Research & Deal Intelligence agent.
# MAGIC
# MAGIC **Logic:**
# MAGIC 1. Rank top 30 companies from `edg_metadata` by chunk count (filings anchor)
# MAGIC 2. Check whether each also appears in `fcst_metadata` (earnings) and `sa_metadata` (news)
# MAGIC 3. Take the top 20 that appear in **at least 2 of 3** tables
# MAGIC 4. If fewer than 20 qualify, backfill with top filing-only companies
# MAGIC
# MAGIC **Output:** `ticker, company_name, filing_chunks, earnings_chunks, news_chunks`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Phase 1: Discover Ticker / Company Columns
# MAGIC
# MAGIC We don't hardcode column names — this cell finds candidate columns in each table
# MAGIC so you can confirm the right ones before running the analysis.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

CATALOG_SCHEMA = "factset_vectors_do_not_edit_or_delete.vector"

edg_df = spark.table(f"{CATALOG_SCHEMA}.edg_metadata")
fcst_df = spark.table(f"{CATALOG_SCHEMA}.fcst_metadata")
sa_df = spark.table(f"{CATALOG_SCHEMA}.sa_metadata")

# COMMAND ----------

def discover_all_columns(df, table_name):
    """Print every column with type, distinct count, and samples for string cols."""
    print(f"{'='*100}")
    print(f"  ALL COLUMNS IN {table_name}  ({df.count():,} rows)")
    print(f"{'='*100}")
    print(f"  {'Column':<40} {'Type':<15} {'Distinct':>10}   {'Sample Values'}")
    print(f"  {'-'*95}")

    for field in df.schema.fields:
        col_name = field.name
        col_type = str(field.dataType).replace("Type", "")
        is_string = isinstance(field.dataType, StringType)

        if is_string:
            n_distinct = df.select(F.countDistinct(col_name)).collect()[0][0]
            samples = (
                df.select(col_name)
                .where(F.col(col_name).isNotNull())
                .distinct()
                .limit(5)
                .collect()
            )
            sample_vals = [row[0][:40] if row[0] and len(row[0]) > 40 else row[0] for row in samples]
            print(f"  {col_name:<40} {col_type:<15} {n_distinct:>10,}   {sample_vals}")
        else:
            print(f"  {col_name:<40} {col_type:<15}")

    print()

discover_all_columns(edg_df, "edg_metadata")

# COMMAND ----------

discover_all_columns(fcst_df, "fcst_metadata")

# COMMAND ----------

discover_all_columns(sa_df, "sa_metadata")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Phase 2: Configure Column Names
# MAGIC
# MAGIC **After running Phase 1**, set the correct column names below.
# MAGIC These are the columns used for grouping/joining across tables.

# COMMAND ----------

# ── CONFIGURE THESE after reviewing Phase 1 output ──────────────────────────
# Replace with the actual column names discovered above.
# Look for:
#   - Ticker: a string col with ~hundreds to low-thousands of distinct values,
#             samples look like stock tickers (e.g. "AAPL", "MSFT-US", "000001-CN")
#   - Company: a string col with similar cardinality, samples are company names
#
# If one table lacks a ticker column, you may need to join through a shared ID
# (e.g. factset_entity_id, document_id) or use chunk_id → all_docs → product.

TICKER_COL_EDG  = "CHANGE_ME"   # <-- set from Phase 1 output
COMPANY_COL_EDG = "CHANGE_ME"   # <-- set from Phase 1 output

TICKER_COL_FCST  = "CHANGE_ME"
COMPANY_COL_FCST = "CHANGE_ME"

TICKER_COL_SA  = "CHANGE_ME"
COMPANY_COL_SA = "CHANGE_ME"

# ─────────────────────────────────────────────────────────────────────────────

print("Column configuration:")
print(f"  EDG  → ticker: {TICKER_COL_EDG:<30} company: {COMPANY_COL_EDG}")
print(f"  FCST → ticker: {TICKER_COL_FCST:<30} company: {COMPANY_COL_FCST}")
print(f"  SA   → ticker: {TICKER_COL_SA:<30} company: {COMPANY_COL_SA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Phase 3: Top 30 Filing Companies + Cross-Source Overlap

# COMMAND ----------

# Step 1: Top 30 companies from edg_metadata by chunk count
edg_top30 = (
    edg_df
    .groupBy(F.col(TICKER_COL_EDG).alias("ticker"), F.col(COMPANY_COL_EDG).alias("company_name"))
    .agg(F.count("*").alias("filing_chunks"))
    .orderBy(F.desc("filing_chunks"))
    .limit(30)
)

print("=== Top 30 companies by EDG chunk count ===")
display(edg_top30)

# COMMAND ----------

# Step 2: Chunk counts for the same tickers in FCST and SA
fcst_counts = (
    fcst_df
    .groupBy(F.col(TICKER_COL_FCST).alias("ticker"))
    .agg(F.count("*").alias("earnings_chunks"))
)

sa_counts = (
    sa_df
    .groupBy(F.col(TICKER_COL_SA).alias("ticker"))
    .agg(F.count("*").alias("news_chunks"))
)

# COMMAND ----------

# Step 3: Left-join FCST and SA counts onto the top-30 filing companies
combined = (
    edg_top30
    .join(fcst_counts, on="ticker", how="left")
    .join(sa_counts, on="ticker", how="left")
    .fillna(0, subset=["earnings_chunks", "news_chunks"])
    .withColumn(
        "source_count",
        (F.when(F.col("filing_chunks") > 0, 1).otherwise(0))
        + (F.when(F.col("earnings_chunks") > 0, 1).otherwise(0))
        + (F.when(F.col("news_chunks") > 0, 1).otherwise(0))
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
# MAGIC ## Phase 4: Final 20 — At Least 2/3 Sources, Backfill If Needed

# COMMAND ----------

# Companies appearing in >= 2 sources, ordered by filing depth
multi_source = (
    combined
    .where(F.col("source_count") >= 2)
    .orderBy(F.desc("source_count"), F.desc("filing_chunks"))
)

multi_source_count = multi_source.count()
print(f"Companies in >= 2 sources: {multi_source_count}")

# Companies in only 1 source (filing-only), ordered by filing depth
single_source = (
    combined
    .where(F.col("source_count") < 2)
    .orderBy(F.desc("filing_chunks"))
)

# COMMAND ----------

# Build final list of exactly 20
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

# Also print as plain text for easy copy-paste
rows = final_result.collect()

print(f"\n{'Ticker':<20} {'Company':<35} {'Filings':>10} {'Earnings':>10} {'News':>10}")
print("-" * 90)
for row in rows:
    print(f"{row['ticker']:<20} {row['company_name']:<35} {row['filing_chunks']:>10,} {row['earnings_chunks']:>10,} {row['news_chunks']:>10,}")
print("-" * 90)
print(f"{'TOTAL':<20} {'':<35} {sum(r['filing_chunks'] for r in rows):>10,} {sum(r['earnings_chunks'] for r in rows):>10,} {sum(r['news_chunks'] for r in rows):>10,}")
