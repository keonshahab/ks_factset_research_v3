# Databricks notebook source

# MAGIC %md
# MAGIC # 04 — Filing Chunks
# MAGIC
# MAGIC **Purpose:** Build the `filing_chunks` table by joining FactSet EDG metadata with document
# MAGIC text, filtered to companies in the config table. Also creates a `filing_documents` summary
# MAGIC table (one row per document).
# MAGIC
# MAGIC Supports two run modes:
# MAGIC - **full** — `CREATE OR REPLACE` the entire table (default, for initial builds)
# MAGIC - **incremental** — `MERGE` new chunks only (for adding companies without rebuilding)
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 0 | Parameters — run mode, optional ticker override |
# MAGIC | 1 | Build `filing_chunks` (full or incremental) |
# MAGIC | 2 | Build `filing_documents` summary |
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
# MAGIC ## Step 1: Build `filing_chunks`
# MAGIC
# MAGIC **Sources:**
# MAGIC - `factset_vectors_do_not_edit_or_delete.vector.edg_metadata` (filing metadata)
# MAGIC - `factset_vectors_do_not_edit_or_delete.vector.all_docs` (chunked text, filtered to product = 'EDG')
# MAGIC - `target_companies` temp view (from config)
# MAGIC
# MAGIC **Join:** edg → all_docs on chunk_id, then inner join to target_companies on company_name.
# MAGIC
# MAGIC **Transforms:** rename content → chunk_text, add ticker/ticker_region from config,
# MAGIC derive doc_type_label, is_exhibit, filing_year, filing_quarter, source_table.

# COMMAND ----------

filing_chunks_sql = """
    SELECT
        edg.*,
        docs.content AS chunk_text,
        tc.ticker,
        tc.ticker_region,
        CASE edg.fds_filing_type
            WHEN '10-K'  THEN '10-K Annual Report'
            WHEN '10KSB' THEN '10-K Annual Report'
            WHEN '10-Q'  THEN '10-Q Quarterly Report'
            WHEN '8-K'   THEN '8-K Current Report'
            WHEN '20-F'  THEN '20-F Annual Report (Foreign)'
            WHEN '6-K'   THEN '6-K Report (Foreign)'
            ELSE edg.fds_filing_type
        END AS doc_type_label,
        CASE
            WHEN edg.exhibit_level > 0 OR edg.exhibit_title IS NOT NULL THEN true
            ELSE false
        END AS is_exhibit,
        YEAR(edg.acceptance_date) AS filing_year,
        CONCAT('Q', QUARTER(edg.acceptance_date)) AS filing_quarter,
        'edg_metadata' AS source_table
    FROM factset_vectors_do_not_edit_or_delete.vector.edg_metadata edg
    JOIN factset_vectors_do_not_edit_or_delete.vector.all_docs docs
        ON edg.chunk_id = docs.chunk_id AND docs.product = 'EDG'
    INNER JOIN target_companies tc
        ON edg.company_name = tc.company_name_edg
    WHERE docs.content IS NOT NULL
      AND TRIM(docs.content) != ''
      AND edg.token_count != 0
"""

# COMMAND ----------

if mode == "full":
    print("Running FULL build: CREATE OR REPLACE TABLE ...")
    spark.sql(f"""
        CREATE OR REPLACE TABLE ks_factset_research_v3.demo.filing_chunks
        USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        AS {filing_chunks_sql}
    """)
    row_count = spark.table("ks_factset_research_v3.demo.filing_chunks").count()
    print(f"filing_chunks created with {row_count:,} rows")

elif mode == "incremental":
    print("Running INCREMENTAL build: MERGE new chunks ...")

    # Pre-merge count
    pre_count = spark.table("ks_factset_research_v3.demo.filing_chunks").count()

    spark.sql(f"""
        MERGE INTO ks_factset_research_v3.demo.filing_chunks AS target
        USING (
            {filing_chunks_sql}
        ) AS source
        ON target.chunk_id = source.chunk_id
        WHEN NOT MATCHED THEN INSERT *
    """)

    post_count = spark.table("ks_factset_research_v3.demo.filing_chunks").count()
    new_chunks = post_count - pre_count
    print(f"MERGE complete: {new_chunks:,} new chunks added ({pre_count:,} → {post_count:,})")

else:
    raise ValueError(f"Unknown mode: '{mode}'. Expected 'full' or 'incremental'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: `filing_documents` Summary
# MAGIC
# MAGIC One row per `document_id` — always rebuilt from `filing_chunks`.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.demo.filing_documents
    USING DELTA
    AS
    SELECT
        document_id,
        FIRST(company_name)     AS company_name,
        FIRST(ticker)           AS ticker,
        FIRST(ticker_region)    AS ticker_region,
        FIRST(fds_filing_type)  AS fds_filing_type,
        FIRST(doc_type_label)   AS doc_type_label,
        FIRST(is_exhibit)       AS is_exhibit,
        FIRST(exhibit_title)    AS exhibit_title,
        FIRST(acceptance_date)  AS acceptance_date,
        FIRST(filing_year)      AS filing_year,
        FIRST(filing_quarter)   AS filing_quarter,
        COUNT(*)                AS total_chunks,
        SUM(token_count)        AS total_tokens,
        SUM(LENGTH(chunk_text)) AS total_text_length
    FROM ks_factset_research_v3.demo.filing_chunks
    GROUP BY document_id
""")

doc_count = spark.table("ks_factset_research_v3.demo.filing_documents").count()
print(f"filing_documents created with {doc_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.a — Row Counts

# COMMAND ----------

chunk_count = spark.table("ks_factset_research_v3.demo.filing_chunks").count()
doc_count = spark.table("ks_factset_research_v3.demo.filing_documents").count()
print(f"filing_chunks:    {chunk_count:,} rows")
print(f"filing_documents: {doc_count:,} rows")

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
    FROM ks_factset_research_v3.demo.filing_chunks
    GROUP BY ticker, company_name
    ORDER BY chunk_count DESC
""")
display(by_company)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.c — Compare to Expected (~12,163 filing chunks)

# COMMAND ----------

expected_chunks = 12163
diff = chunk_count - expected_chunks
pct = (diff / expected_chunks) * 100 if expected_chunks else 0
print(f"Expected: ~{expected_chunks:,}")
print(f"Actual:    {chunk_count:,}")
print(f"Difference: {diff:+,} ({pct:+.1f}%)")
if abs(pct) > 20:
    print("WARNING: chunk count differs from expected by more than 20%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.d — Sample Rows

# COMMAND ----------

sample_df = spark.sql("""
    SELECT
        chunk_id,
        ticker,
        fds_filing_type,
        acceptance_date,
        LEFT(chunk_text, 200) AS chunk_text_preview
    FROM ks_factset_research_v3.demo.filing_chunks
    ORDER BY RAND()
    LIMIT 5
""")
display(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.e — Data Quality Checks

# COMMAND ----------

# Null chunk_text
null_text = spark.sql("""
    SELECT COUNT(*) AS null_chunk_text
    FROM ks_factset_research_v3.demo.filing_chunks
    WHERE chunk_text IS NULL
""").collect()[0]["null_chunk_text"]
print(f"Rows with NULL chunk_text: {null_text}")
assert null_text == 0, f"FAIL: found {null_text} rows with NULL chunk_text"

# Duplicate chunk_ids
dup_chunks = spark.sql("""
    SELECT COUNT(*) AS dup_count FROM (
        SELECT chunk_id
        FROM ks_factset_research_v3.demo.filing_chunks
        GROUP BY chunk_id
        HAVING COUNT(*) > 1
    )
""").collect()[0]["dup_count"]
print(f"Duplicate chunk_ids: {dup_chunks}")
assert dup_chunks == 0, f"FAIL: found {dup_chunks} duplicate chunk_ids"

print("Data quality checks PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.f — Confirm Change Data Feed

# COMMAND ----------

cdf_props = spark.sql("""
    DESCRIBE DETAIL ks_factset_research_v3.demo.filing_chunks
""").select("properties").collect()[0]["properties"]

cdf_enabled = cdf_props.get("delta.enableChangeDataFeed", "false")
print(f"delta.enableChangeDataFeed = {cdf_enabled}")
assert cdf_enabled == "true", "FAIL: Change Data Feed is not enabled on filing_chunks"
print("Change Data Feed: CONFIRMED")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.g — Incremental Summary (if applicable)

# COMMAND ----------

if mode == "incremental":
    post_count = spark.table("ks_factset_research_v3.demo.filing_chunks").count()
    print(f"Incremental run complete.")
    print(f"Total filing_chunks: {post_count:,}")
    # Show newly added companies
    new_company_chunks = spark.sql("""
        SELECT
            ticker,
            company_name,
            COUNT(*) AS chunk_count
        FROM ks_factset_research_v3.demo.filing_chunks
        GROUP BY ticker, company_name
        ORDER BY chunk_count DESC
    """)
    display(new_company_chunks)
else:
    print("Full mode — incremental summary not applicable.")
