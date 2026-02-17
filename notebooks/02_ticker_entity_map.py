# Databricks notebook source

# MAGIC %md
# MAGIC # 02 — Ticker-Entity Mapping Table
# MAGIC
# MAGIC **Purpose:** Create a gold-layer mapping table that links ticker symbols to FactSet entity IDs,
# MAGIC company names, countries, and entity types. This table serves as the universal lookup for
# MAGIC resolving tickers to entities across all downstream notebooks.
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1 | Create catalog and schemas |
# MAGIC | 2 | Build `ticker_entity_map` from FactSet symbology + entity tables |
# MAGIC | 3 | Validate coverage, uniqueness, and demo company presence |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Create Catalog and Schemas

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS ks_factset_research_v3")
print("Catalog ks_factset_research_v3: OK")

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS ks_factset_research_v3.gold")
print("Schema ks_factset_research_v3.gold: OK")

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS ks_factset_research_v3.demo")
print("Schema ks_factset_research_v3.demo: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Create `ticker_entity_map`
# MAGIC
# MAGIC Join three FactSet Delta Share tables:
# MAGIC - **sym_ticker_region** — maps `FSYM_ID` (regional, `-R` suffix) → ticker + region
# MAGIC - **ent_scr_sec_entity** — maps `FSYM_ID` (security, `-S` suffix) → `FACTSET_ENTITY_ID`
# MAGIC - **sym_entity** — maps `FACTSET_ENTITY_ID` → company name, country, entity type
# MAGIC
# MAGIC The first two tables use different FSYM_ID levels (`-R` vs `-S`) that share the
# MAGIC same base ID, so we join on `regexp_extract(FSYM_ID, '^(.+)-', 1)`.
# MAGIC
# MAGIC Filter to `ENTITY_TYPE = 'PUB'` (public companies only).

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.ticker_entity_map AS
    SELECT
        tr.TICKER_REGION       AS ticker_region,
        ese.FACTSET_ENTITY_ID  AS entity_id,
        se.ENTITY_PROPER_NAME  AS company_name,
        se.ISO_COUNTRY         AS country,
        se.ENTITY_TYPE         AS entity_type,
        tr.FSYM_ID             AS fsym_id
    FROM delta_share_factset_do_not_delete_or_edit.sym_v1.sym_ticker_region tr
    JOIN delta_share_factset_do_not_delete_or_edit.ent_v1.ent_scr_sec_entity ese
        ON regexp_extract(tr.FSYM_ID, '^(.+)-', 1) = regexp_extract(ese.FSYM_ID, '^(.+)-', 1)
    JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON ese.FACTSET_ENTITY_ID = se.FACTSET_ENTITY_ID
    WHERE se.ENTITY_TYPE = 'PUB'
""")
print("Table ks_factset_research_v3.gold.ticker_entity_map: CREATED")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 — Row Count & Cardinality

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.table("ks_factset_research_v3.gold.ticker_entity_map")

total_rows = df.count()
distinct_tickers = df.select("ticker_region").distinct().count()
distinct_entities = df.select("entity_id").distinct().count()
distinct_countries = df.select("country").distinct().count()

print(f"{'Metric':<30} {'Value':>12}")
print("-" * 44)
print(f"{'Total rows':<30} {total_rows:>12,}")
print(f"{'Distinct ticker_region':<30} {distinct_tickers:>12,}")
print(f"{'Distinct entity_id':<30} {distinct_entities:>12,}")
print(f"{'Distinct countries':<30} {distinct_countries:>12,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 — Sample Rows

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 — Top 10 Countries by Ticker Count

# COMMAND ----------

country_counts = (
    df.groupBy("country")
    .agg(F.countDistinct("ticker_region").alias("ticker_count"))
    .orderBy(F.desc("ticker_count"))
    .limit(10)
)

display(country_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 — Verify Demo Companies
# MAGIC
# MAGIC Confirm that the demo tickers from Notebook 01 are present in the mapping.

# COMMAND ----------

demo_tickers = ["AAPL-US", "MSFT-US", "AMZN-US", "GOOGL-US", "JPM-US"]

demo_df = df.filter(F.col("ticker_region").isin(demo_tickers))

print(f"Demo tickers requested: {len(demo_tickers)}")
print(f"Demo tickers found:     {demo_df.count()}")
print()

display(demo_df)

missing = set(demo_tickers) - set(row["ticker_region"] for row in demo_df.collect())
if missing:
    print(f"\nWARNING: Missing demo tickers: {sorted(missing)}")
else:
    print("\nAll demo tickers present.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 — Check for Duplicate `ticker_region` Values

# COMMAND ----------

dupes = (
    df.groupBy("ticker_region")
    .count()
    .filter(F.col("count") > 1)
    .orderBy(F.desc("count"))
)

dupe_count = dupes.count()
print(f"Ticker_region values with duplicates: {dupe_count:,}")

if dupe_count > 0:
    print("\nTop 20 duplicated ticker_region values:")
    display(dupes.limit(20))
else:
    print("No duplicates — ticker_region is unique.")
