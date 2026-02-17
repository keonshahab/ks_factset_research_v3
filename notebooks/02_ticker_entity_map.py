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
# MAGIC | 2 | Source table diagnostics (row counts, schemas, join-key overlap) |
# MAGIC | 3 | Build `ticker_entity_map` from FactSet symbology + entity tables |
# MAGIC | 4 | Validate coverage, uniqueness, and demo company presence |

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
# MAGIC ## Step 2: Source Table Diagnostics
# MAGIC
# MAGIC Before building the map, verify row counts, schemas, and join-key overlap
# MAGIC for each source table.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 — Row Counts

# COMMAND ----------

tr_df  = spark.table("delta_share_factset_do_not_delete_or_edit.sym_v1.sym_ticker_region")
ese_df = spark.table("delta_share_factset_do_not_delete_or_edit.ent_v1.ent_scr_sec_entity")
se_df  = spark.table("delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity")

print(f"{'Source Table':<60} {'Rows':>12}")
print("-" * 74)
print(f"{'sym_ticker_region':<60} {tr_df.count():>12,}")
print(f"{'ent_scr_sec_entity':<60} {ese_df.count():>12,}")
print(f"{'sym_entity':<60} {se_df.count():>12,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 — Schemas

# COMMAND ----------

for name, tdf in [("sym_ticker_region", tr_df), ("ent_scr_sec_entity", ese_df), ("sym_entity", se_df)]:
    print(f"=== {name} ===")
    for f in tdf.schema.fields:
        print(f"  {f.name:<40} {str(f.dataType)}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 — Sample Rows from Each Source

# COMMAND ----------

print("=== sym_ticker_region (5 rows) ===")
display(tr_df.limit(5))

# COMMAND ----------

print("=== ent_scr_sec_entity (5 rows) ===")
display(ese_df.limit(5))

# COMMAND ----------

print("=== sym_entity (5 rows) ===")
display(se_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 — ENTITY_TYPE Distribution
# MAGIC
# MAGIC Check what values exist before filtering to `'PUB'`.

# COMMAND ----------

from pyspark.sql import functions as F

display(
    se_df.groupBy("ENTITY_TYPE")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 — Join-Key Overlap Check

# COMMAND ----------

# Show sample FSYM_ID from each table to compare formats
print("=== FSYM_ID samples ===\n")
print("sym_ticker_region (first 5):")
for row in tr_df.select("FSYM_ID").limit(5).collect():
    print(f"  {row['FSYM_ID']}")

print("\nent_scr_sec_entity (first 5):")
for row in ese_df.select("FSYM_ID").limit(5).collect():
    print(f"  {row['FSYM_ID']}")

# Check base-ID overlap (strip suffix after last hyphen)
tr_base  = tr_df.select(F.regexp_extract("FSYM_ID", r"^(.+)-", 1).alias("base_id")).distinct()
ese_base = ese_df.select(F.regexp_extract("FSYM_ID", r"^(.+)-", 1).alias("base_id")).distinct()

base_overlap = tr_base.join(ese_base, "base_id", "inner").count()
print(f"\nBase-ID overlap (strip suffix): {base_overlap:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 — Available Tables in sym_v1 and ent_v1
# MAGIC
# MAGIC Find the bridge table that links regional (`-R`) to security (`-S`) FSYM_IDs.

# COMMAND ----------

print("=== Tables in sym_v1 ===")
sym_tables = spark.sql("SHOW TABLES IN delta_share_factset_do_not_delete_or_edit.sym_v1").collect()
for row in sym_tables:
    print(f"  {row['tableName']}")

print("\n=== Tables in ent_v1 ===")
ent_tables = spark.sql("SHOW TABLES IN delta_share_factset_do_not_delete_or_edit.ent_v1").collect()
for row in ent_tables:
    print(f"  {row['tableName']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Create `ticker_entity_map`
# MAGIC
# MAGIC Join three FactSet Delta Share tables:
# MAGIC - **sym_ticker_region** — maps `FSYM_ID` → ticker + region
# MAGIC - **ent_scr_sec_entity** — maps `FSYM_ID` → `FACTSET_ENTITY_ID`
# MAGIC - **sym_entity** — maps `FACTSET_ENTITY_ID` → company name, country, entity type
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
        ON tr.FSYM_ID = ese.FSYM_ID
    JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON ese.FACTSET_ENTITY_ID = se.FACTSET_ENTITY_ID
    WHERE se.ENTITY_TYPE = 'PUB'
""")
print("Table ks_factset_research_v3.gold.ticker_entity_map: CREATED")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 — Row Count & Cardinality

# COMMAND ----------

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
# MAGIC ### 4.2 — Sample Rows

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 — Top 10 Countries by Ticker Count

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
# MAGIC ### 4.4 — Verify Demo Companies
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
# MAGIC ### 4.5 — Check for Duplicate `ticker_region` Values

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
