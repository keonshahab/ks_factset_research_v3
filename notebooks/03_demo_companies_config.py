# Databricks notebook source

# MAGIC %md
# MAGIC # 03 — Demo Companies Config
# MAGIC
# MAGIC **Purpose:** Create a config table that controls which companies flow through the entire
# MAGIC Research & Deal Intelligence pipeline. All downstream notebooks (chunks, financials,
# MAGIC vector indexes) read from this config table to know which companies to include.
# MAGIC
# MAGIC **Adding a new company = INSERT a row here and rerun the pipeline.**
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1 | Create catalog and schemas (if they don't exist) |
# MAGIC | 2 | Create the `demo_companies` config table |
# MAGIC | 3 | Populate with initial 20 companies (lookup from source tables) |
# MAGIC | 4 | Create helper view `v_active_companies` |
# MAGIC | 5 | Validation |
# MAGIC | 6 | Document the "add a company" workflow |

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
# MAGIC ## Step 2: Create the Config Table
# MAGIC
# MAGIC `demo_companies` holds one row per company. Every downstream notebook queries this
# MAGIC table (via the `v_active_companies` view) to decide which companies to process.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.demo_companies (
        ticker_region       STRING      COMMENT 'FactSet ticker_region (e.g., NVDA-US)',
        ticker              STRING      COMMENT 'Short ticker extracted from ticker_region (e.g., NVDA)',
        company_name_edg    STRING      COMMENT 'Company name as it appears in edg_metadata (for filing joins)',
        company_name_fcst   STRING      COMMENT 'Company name as it appears in fcst_metadata (for earnings joins)',
        sa_symbol           STRING      COMMENT 'Symbol used in sa_metadata primary_symbols array (e.g., NVDA-US)',
        entity_id           STRING      COMMENT 'FactSet entity ID from ticker_entity_map',
        fsym_id             STRING      COMMENT 'FactSet security ID from ticker_entity_map',
        display_name        STRING      COMMENT 'Clean display name for the UI (e.g., NVIDIA Corp)',
        is_active           BOOLEAN     COMMENT 'Set to false to exclude without deleting',
        added_date          DATE        COMMENT 'When this company was added to the config',
        notes               STRING      COMMENT 'Optional notes (e.g., why included, data quality issues)'
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    COMMENT 'Config table controlling which companies flow through the Research & Deal Intelligence pipeline. Add a row and rerun the pipeline to include a new company.'
""")
print("Table ks_factset_research_v3.gold.demo_companies: CREATED")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Populate with Initial 20 Companies
# MAGIC
# MAGIC For each company, look up:
# MAGIC - `entity_id`, `fsym_id` from `ks_factset_research_v3.gold.ticker_entity_map`
# MAGIC - `company_name` from `factset_vectors_do_not_edit_or_delete.vector.edg_metadata`
# MAGIC - `company_name` from `factset_vectors_do_not_edit_or_delete.vector.fcst_metadata`
# MAGIC - `primary_symbols` from `factset_vectors_do_not_edit_or_delete.vector.sa_metadata`

# COMMAND ----------

from pyspark.sql import functions as F

VECTOR_SCHEMA = "factset_vectors_do_not_edit_or_delete.vector"

# The 20 companies to seed
COMPANIES = [
    ("CRM-US",  "CRM",  "Salesforce"),
    ("LLY-US",  "LLY",  "Eli Lilly"),
    ("JNJ-US",  "JNJ",  "Johnson & Johnson"),
    ("MA-US",   "MA",   "Mastercard"),
    ("AMD-US",  "AMD",  "AMD"),
    ("META-US", "META", "Meta Platforms"),
    ("ABBV-US", "ABBV", "AbbVie"),
    ("BAC-US",  "BAC",  "Bank of America"),
    ("WMT-US",  "WMT",  "Walmart"),
    ("V-US",    "V",    "Visa"),
    ("NVDA-US", "NVDA", "NVIDIA"),
    ("MSFT-US", "MSFT", "Microsoft"),
    ("CAT-US",  "CAT",  "Caterpillar"),
    ("INTC-US", "INTC", "Intel"),
    ("JPM-US",  "JPM",  "JPMorgan Chase"),
    ("CVX-US",  "CVX",  "Chevron"),
    ("COST-US", "COST", "Costco"),
    ("XOM-US",  "XOM",  "Exxon Mobil"),
    ("PG-US",   "PG",   "Procter & Gamble"),
    ("TSLA-US", "TSLA", "Tesla"),
]

ticker_regions = [c[0] for c in COMPANIES]
display_names  = {c[0]: c[2] for c in COMPANIES}
tickers        = {c[0]: c[1] for c in COMPANIES}

print(f"Companies to seed: {len(COMPANIES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a — Lookup entity_id and fsym_id from ticker_entity_map

# COMMAND ----------

tem_df = spark.table("ks_factset_research_v3.gold.ticker_entity_map")

entity_lookup = (
    tem_df
    .where(F.col("ticker_region").isin(ticker_regions))
    .select("ticker_region", "entity_id", "fsym_id")
    .dropDuplicates(["ticker_region"])
)

entity_map = {row["ticker_region"]: row for row in entity_lookup.collect()}

print(f"Entity map matches: {len(entity_map)} / {len(ticker_regions)}")
for tr in ticker_regions:
    if tr not in entity_map:
        print(f"  WARNING: {tr} not found in ticker_entity_map")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b — Lookup company_name from edg_metadata

# COMMAND ----------

edg_df = spark.table(f"{VECTOR_SCHEMA}.edg_metadata")

edg_names = (
    edg_df
    .withColumn("ticker", F.get("primary_symbols", 0))
    .where(F.col("ticker").isin(ticker_regions))
    .groupBy("ticker")
    .agg(F.first("company_name").alias("company_name_edg"))
)

edg_map = {row["ticker"]: row["company_name_edg"] for row in edg_names.collect()}

print(f"EDG name matches: {len(edg_map)} / {len(ticker_regions)}")
for tr in ticker_regions:
    if tr not in edg_map:
        print(f"  WARNING: {tr} not found in edg_metadata")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c — Lookup company_name from fcst_metadata

# COMMAND ----------

fcst_df = spark.table(f"{VECTOR_SCHEMA}.fcst_metadata")

fcst_names = (
    fcst_df
    .withColumn("ticker", F.get("primary_symbols", 0))
    .where(F.col("ticker").isin(ticker_regions))
    .groupBy("ticker")
    .agg(F.first("company_name").alias("company_name_fcst"))
)

fcst_map = {row["ticker"]: row["company_name_fcst"] for row in fcst_names.collect()}

print(f"FCST name matches: {len(fcst_map)} / {len(ticker_regions)}")
for tr in ticker_regions:
    if tr not in fcst_map:
        print(f"  WARNING: {tr} not found in fcst_metadata")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3d — Lookup sa_symbol from sa_metadata

# COMMAND ----------

sa_df = spark.table(f"{VECTOR_SCHEMA}.sa_metadata")

sa_symbols = (
    sa_df
    .withColumn("ticker", F.get("primary_symbols", 0))
    .where(F.col("ticker").isin(ticker_regions))
    .select("ticker")
    .dropDuplicates()
)

sa_map = {row["ticker"]: row["ticker"] for row in sa_symbols.collect()}

print(f"SA symbol matches: {len(sa_map)} / {len(ticker_regions)}")
for tr in ticker_regions:
    if tr not in sa_map:
        print(f"  WARNING: {tr} not found in sa_metadata")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3e — Name mismatch warnings
# MAGIC
# MAGIC Flag companies where the name differs between edg and fcst (may need fuzzy matching
# MAGIC in the chunk notebooks).

# COMMAND ----------

print(f"{'Ticker':<12} {'EDG name':<40} {'FCST name':<40} {'Match?'}")
print("-" * 100)

mismatch_count = 0
for tr in ticker_regions:
    edg_name  = edg_map.get(tr)
    fcst_name = fcst_map.get(tr)
    match = "YES" if edg_name == fcst_name else "NO"
    if edg_name != fcst_name:
        mismatch_count += 1
    print(f"{tr:<12} {str(edg_name)[:39]:<40} {str(fcst_name)[:39]:<40} {match}")

print("-" * 100)
if mismatch_count > 0:
    print(f"\nWARNING: {mismatch_count} companies have different names in edg vs fcst.")
    print("These may need fuzzy matching in the chunk notebooks.")
else:
    print("\nAll company names match across edg and fcst.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3f — INSERT rows

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DateType
from datetime import date

rows = []
for tr in ticker_regions:
    entity_row = entity_map.get(tr)
    rows.append((
        tr,
        tickers[tr],
        edg_map.get(tr),
        fcst_map.get(tr),
        sa_map.get(tr),
        entity_row["entity_id"] if entity_row else None,
        entity_row["fsym_id"]   if entity_row else None,
        display_names[tr],
        True,
        date.today(),
        "Initial seed — Notebook 03",
    ))

schema = StructType([
    StructField("ticker_region",    StringType(),  True),
    StructField("ticker",           StringType(),  True),
    StructField("company_name_edg", StringType(),  True),
    StructField("company_name_fcst",StringType(),  True),
    StructField("sa_symbol",        StringType(),  True),
    StructField("entity_id",        StringType(),  True),
    StructField("fsym_id",          StringType(),  True),
    StructField("display_name",     StringType(),  True),
    StructField("is_active",        BooleanType(), True),
    StructField("added_date",       DateType(),    True),
    StructField("notes",            StringType(),  True),
])

insert_df = spark.createDataFrame(rows, schema)
insert_df.write.mode("append").saveAsTable("ks_factset_research_v3.gold.demo_companies")

print(f"Inserted {insert_df.count()} rows into ks_factset_research_v3.gold.demo_companies")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Create Helper View
# MAGIC
# MAGIC `v_active_companies` filters to `is_active = true`. All downstream notebooks query this view.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE VIEW ks_factset_research_v3.gold.v_active_companies AS
    SELECT * FROM ks_factset_research_v3.gold.demo_companies
    WHERE is_active = true
""")
print("View ks_factset_research_v3.gold.v_active_companies: CREATED")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 — All 20 rows

# COMMAND ----------

config_df = spark.table("ks_factset_research_v3.gold.demo_companies")

print(f"Total rows: {config_df.count()}")
display(config_df.orderBy("ticker_region"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 — Verify no null entity_ids or fsym_ids

# COMMAND ----------

null_entity = config_df.where(F.col("entity_id").isNull()).count()
null_fsym   = config_df.where(F.col("fsym_id").isNull()).count()

print(f"Rows with null entity_id: {null_entity}")
print(f"Rows with null fsym_id:   {null_fsym}")

if null_entity > 0:
    print("\nWARNING — Companies missing entity_id:")
    display(config_df.where(F.col("entity_id").isNull()).select("ticker_region", "display_name"))

if null_fsym > 0:
    print("\nWARNING — Companies missing fsym_id:")
    display(config_df.where(F.col("fsym_id").isNull()).select("ticker_region", "display_name"))

if null_entity == 0 and null_fsym == 0:
    print("All entity_id and fsym_id values are populated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 — Verify non-null company names and sa_symbol

# COMMAND ----------

null_edg  = config_df.where(F.col("company_name_edg").isNull()).count()
null_fcst = config_df.where(F.col("company_name_fcst").isNull()).count()
null_sa   = config_df.where(F.col("sa_symbol").isNull()).count()

print(f"Rows with null company_name_edg:  {null_edg}")
print(f"Rows with null company_name_fcst: {null_fcst}")
print(f"Rows with null sa_symbol:         {null_sa}")

if null_edg > 0:
    print("\nWARNING — Companies missing company_name_edg:")
    display(config_df.where(F.col("company_name_edg").isNull()).select("ticker_region", "display_name"))

if null_fcst > 0:
    print("\nWARNING — Companies missing company_name_fcst:")
    display(config_df.where(F.col("company_name_fcst").isNull()).select("ticker_region", "display_name"))

if null_sa > 0:
    print("\nWARNING — Companies missing sa_symbol:")
    display(config_df.where(F.col("sa_symbol").isNull()).select("ticker_region", "display_name"))

if null_edg == 0 and null_fcst == 0 and null_sa == 0:
    print("All company_name_edg, company_name_fcst, and sa_symbol values are populated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 — Summary

# COMMAND ----------

active_count = config_df.where(F.col("is_active") == True).count()

print("=" * 80)
print(f"Config ready: {active_count} active companies.")
print()
print("To add a new company:")
print("  INSERT INTO ks_factset_research_v3.gold.demo_companies VALUES (...)")
print("  Then rerun the pipeline.")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: "Add a Company" Workflow
# MAGIC
# MAGIC Reference instructions for adding a new company to the pipeline.

# COMMAND ----------

print("""
================================================================================
TO ADD A NEW COMPANY:
================================================================================

1. Find the ticker_region (e.g., 'AAPL-US')

2. Look up the company name in each source table:

   -- edg (filings):
   SELECT DISTINCT company_name
   FROM factset_vectors_do_not_edit_or_delete.vector.edg_metadata
   WHERE company_name LIKE '%Apple%'

   -- fcst (earnings transcripts):
   SELECT DISTINCT company_name
   FROM factset_vectors_do_not_edit_or_delete.vector.fcst_metadata
   WHERE company_name LIKE '%Apple%'

   -- sa (news): check primary_symbols for 'AAPL-US'
   SELECT DISTINCT primary_symbols[0]
   FROM factset_vectors_do_not_edit_or_delete.vector.sa_metadata
   WHERE array_contains(primary_symbols, 'AAPL-US')

   -- ticker_entity_map: get entity_id and fsym_id
   SELECT entity_id, fsym_id
   FROM ks_factset_research_v3.gold.ticker_entity_map
   WHERE ticker_region = 'AAPL-US'

3. INSERT INTO ks_factset_research_v3.gold.demo_companies VALUES (
     'AAPL-US',           -- ticker_region
     'AAPL',              -- ticker
     'Apple Inc.',         -- company_name_edg  (from edg query above)
     'Apple Inc.',         -- company_name_fcst (from fcst query above)
     'AAPL-US',           -- sa_symbol          (from sa query above)
     '<entity_id>',       -- entity_id          (from ticker_entity_map)
     '<fsym_id>',         -- fsym_id            (from ticker_entity_map)
     'Apple Inc.',         -- display_name
     true,                -- is_active
     current_date(),      -- added_date
     'Added for demo'     -- notes
   )

4. Rerun the Lakeflow workflow (or run notebooks 04-08 individually)

================================================================================
""")
