# Databricks notebook source

# MAGIC %md
# MAGIC # 05 — Earnings Chunks
# MAGIC
# MAGIC **Purpose:** Build the `earnings_chunks` table by joining FactSet FCST metadata with document
# MAGIC text from `all_docs`, filtered to companies in the config table. Also creates an `earnings_documents`
# MAGIC summary table (one row per document).
# MAGIC
# MAGIC Supports two run modes:
# MAGIC - **full** — `CREATE OR REPLACE` the entire table (default, for initial builds)
# MAGIC - **incremental** — `MERGE` new chunks only (for adding companies without rebuilding)
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 0 | Parameters — run mode, optional ticker override |
# MAGIC | 1 | Build `earnings_chunks` (full or incremental) |
# MAGIC | 2 | Build `earnings_documents` summary |
# MAGIC | 3 | Validation |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Parameters

# COMMAND ----------

dbutils.widgets.dropdown("mode", "full", ["full", "incremental"], "Run mode")

# COMMAND ----------

dbutils.widgets.text("tickers", "", "Comma-separated ticker_regions to process (empty = all active from config)")

# COMMAND ----------

# Get run mode
mode = dbutils.widgets.get("mode").strip().lower()
print(f"Run mode: {mode}")

# COMMAND ----------

# Get company list
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
# MAGIC ## Step 1: Build `earnings_chunks`
# MAGIC
# MAGIC **Sources:**
# MAGIC - `factset_vectors_do_not_edit_or_delete.vector.fcst_metadata` (earnings transcript metadata)
# MAGIC - `factset_vectors_do_not_edit_or_delete.vector.all_docs` (chunked text, filtered to product = 'FCST')
# MAGIC - `target_companies` temp view (from config)
# MAGIC
# MAGIC **Join:** fcst → all_docs on chunk_id, then inner join to target_companies on company_name.
# MAGIC
# MAGIC **Transforms:** rename content → chunk_text, add ticker/ticker_region from config,
# MAGIC derive transcript_year, transcript_quarter, source_table.

# COMMAND ----------

earnings_chunks_sql = """
    SELECT * FROM (
        SELECT
            fcst.*,
            docs.content AS chunk_text,
            tc.ticker,
            tc.ticker_region,
            YEAR(fcst.event_date) AS transcript_year,
            CONCAT('Q', QUARTER(fcst.event_date)) AS transcript_quarter,
            'fcst_metadata' AS source_table,
            ROW_NUMBER() OVER (PARTITION BY fcst.chunk_id ORDER BY tc.ticker_region) AS _rn
        FROM factset_vectors_do_not_edit_or_delete.vector.fcst_metadata fcst
        JOIN factset_vectors_do_not_edit_or_delete.vector.all_docs docs
            ON fcst.chunk_id = docs.chunk_id AND docs.product = 'FCST'
        INNER JOIN target_companies tc
            ON fcst.company_name = tc.company_name_fcst
        WHERE docs.content IS NOT NULL
          AND TRIM(docs.content) != ''
          AND fcst.token_count != 0
    ) WHERE _rn = 1
"""

# COMMAND ----------

if mode == "full":
    print("Running FULL build: CREATE OR REPLACE TABLE ...")
    spark.sql(f"""
        CREATE OR REPLACE TABLE ks_factset_research_v3.demo.earnings_chunks
        USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        AS {earnings_chunks_sql}
    """)
    row_count = spark.table("ks_factset_research_v3.demo.earnings_chunks").count()
    print(f"earnings_chunks created with {row_count:,} rows")

elif mode == "incremental":
    print("Running INCREMENTAL build: MERGE new chunks ...")

    # Pre-merge count
    pre_count = spark.table("ks_factset_research_v3.demo.earnings_chunks").count()

    spark.sql(f"""
        MERGE INTO ks_factset_research_v3.demo.earnings_chunks AS target
        USING (
            {earnings_chunks_sql}
        ) AS source
        ON target.chunk_id = source.chunk_id
        WHEN NOT MATCHED THEN INSERT *
    """)

    post_count = spark.table("ks_factset_research_v3.demo.earnings_chunks").count()
    new_chunks = post_count - pre_count
    print(f"MERGE complete: {new_chunks:,} new chunks added ({pre_count:,} → {post_count:,})")

else:
    raise ValueError(f"Unknown mode: '{mode}'. Expected 'full' or 'incremental'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: `earnings_documents` Summary
# MAGIC
# MAGIC One row per `document_id` — always rebuilt from `earnings_chunks`.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.demo.earnings_documents
    USING DELTA
    AS
    SELECT
        document_id,
        FIRST(company_name)         AS company_name,
        FIRST(ticker)               AS ticker,
        FIRST(ticker_region)        AS ticker_region,
        FIRST(speaker_name)         AS speaker_name,
        FIRST(speaker_type)         AS speaker_type,
        FIRST(event_type_id)        AS event_type_id,
        FIRST(transcript_type)      AS transcript_type,
        FIRST(event_date)           AS event_date,
        FIRST(transcript_year)      AS transcript_year,
        FIRST(transcript_quarter)   AS transcript_quarter,
        COUNT(*)                    AS total_chunks,
        SUM(token_count)            AS total_tokens,
        SUM(LENGTH(chunk_text))     AS total_text_length
    FROM ks_factset_research_v3.demo.earnings_chunks
    GROUP BY document_id
""")

doc_count = spark.table("ks_factset_research_v3.demo.earnings_documents").count()
print(f"earnings_documents created with {doc_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.a — Row Counts

# COMMAND ----------

chunk_count = spark.table("ks_factset_research_v3.demo.earnings_chunks").count()
doc_count = spark.table("ks_factset_research_v3.demo.earnings_documents").count()
print(f"earnings_chunks:    {chunk_count:,} rows")
print(f"earnings_documents: {doc_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.b — Counts by Company

# COMMAND ----------

by_company = spark.sql("""
    SELECT
        ticker,
        company_name,
        COUNT(DISTINCT document_id) AS doc_count,
        COUNT(*)                    AS chunk_count
    FROM ks_factset_research_v3.demo.earnings_chunks
    GROUP BY ticker, company_name
    ORDER BY chunk_count DESC
""")
display(by_company)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.c — Counts by Date

# COMMAND ----------

by_date = spark.sql("""
    SELECT
        transcript_year,
        transcript_quarter,
        COUNT(DISTINCT document_id) AS doc_count,
        COUNT(*)                    AS chunk_count
    FROM ks_factset_research_v3.demo.earnings_chunks
    GROUP BY transcript_year, transcript_quarter
    ORDER BY transcript_year DESC, transcript_quarter DESC
""")
display(by_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.d — Compare to Expected (~39,507 earnings chunks)

# COMMAND ----------

expected_chunks = 39507
diff = chunk_count - expected_chunks
pct = (diff / expected_chunks) * 100 if expected_chunks else 0
print(f"Expected: ~{expected_chunks:,}")
print(f"Actual:    {chunk_count:,}")
print(f"Difference: {diff:+,} ({pct:+.1f}%)")
if abs(pct) > 20:
    print("WARNING: chunk count differs from expected by more than 20%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.e — Sample Rows

# COMMAND ----------

sample_df = spark.sql("""
    SELECT
        chunk_id,
        ticker,
        company_name,
        speaker_name,
        speaker_type,
        event_date,
        transcript_year,
        transcript_quarter,
        LEFT(chunk_text, 200) AS chunk_text_preview
    FROM ks_factset_research_v3.demo.earnings_chunks
    ORDER BY RAND()
    LIMIT 5
""")
display(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.f — Data Quality Checks

# COMMAND ----------

# Null chunk_text
null_text = spark.sql("""
    SELECT COUNT(*) AS null_chunk_text
    FROM ks_factset_research_v3.demo.earnings_chunks
    WHERE chunk_text IS NULL
""").collect()[0]["null_chunk_text"]
print(f"Rows with NULL chunk_text: {null_text}")
assert null_text == 0, f"FAIL: found {null_text} rows with NULL chunk_text"

# Duplicate chunk_ids
dup_chunks = spark.sql("""
    SELECT COUNT(*) AS dup_count FROM (
        SELECT chunk_id
        FROM ks_factset_research_v3.demo.earnings_chunks
        GROUP BY chunk_id
        HAVING COUNT(*) > 1
    )
""").collect()[0]["dup_count"]
print(f"Duplicate chunk_ids: {dup_chunks}")
if dup_chunks > 0:
    print("Duplicate chunk_id details:")
    display(spark.sql("""
        SELECT chunk_id, COUNT(*) AS cnt, COLLECT_SET(ticker_region) AS ticker_regions
        FROM ks_factset_research_v3.demo.earnings_chunks
        GROUP BY chunk_id
        HAVING COUNT(*) > 1
    """))
assert dup_chunks == 0, f"FAIL: found {dup_chunks} duplicate chunk_ids"

print("Data quality checks PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.g — Confirm Change Data Feed

# COMMAND ----------

cdf_props = spark.sql("""
    DESCRIBE DETAIL ks_factset_research_v3.demo.earnings_chunks
""").select("properties").collect()[0]["properties"]

cdf_enabled = cdf_props.get("delta.enableChangeDataFeed", "false")
print(f"delta.enableChangeDataFeed = {cdf_enabled}")
assert cdf_enabled == "true", "FAIL: Change Data Feed is not enabled on earnings_chunks"
print("Change Data Feed: CONFIRMED")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.h — Incremental Summary (if applicable)

# COMMAND ----------

if mode == "incremental":
    post_count = spark.table("ks_factset_research_v3.demo.earnings_chunks").count()
    print(f"Incremental run complete.")
    print(f"Total earnings_chunks: {post_count:,}")
    # Show newly added companies
    new_company_chunks = spark.sql("""
        SELECT
            ticker,
            company_name,
            COUNT(*) AS chunk_count
        FROM ks_factset_research_v3.demo.earnings_chunks
        GROUP BY ticker, company_name
        ORDER BY chunk_count DESC
    """)
    display(new_company_chunks)
else:
    print("Full mode — incremental summary not applicable.")
