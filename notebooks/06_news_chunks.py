# Databricks notebook source

# MAGIC %md
# MAGIC # 06 — News Chunks
# MAGIC
# MAGIC **Purpose:** Build the `news_chunks` table by joining FactSet SA (StreetAccount) metadata with
# MAGIC document text from `all_docs`, filtered to companies in the config table. Also creates a
# MAGIC `news_documents` summary table (one row per document).
# MAGIC
# MAGIC Supports two run modes:
# MAGIC - **full** — `CREATE OR REPLACE` the entire table (default, for initial builds)
# MAGIC - **incremental** — `MERGE` new chunks only (for adding companies without rebuilding)
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 0 | Parameters — run mode, optional ticker override |
# MAGIC | 1 | Build `news_chunks` (full or incremental) |
# MAGIC | 2 | Build `news_documents` summary |
# MAGIC | 3 | Validation |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Parameters

# COMMAND ----------

dbutils.widgets.text("tickers", "", "Comma-separated ticker_regions (empty = all active)")

# COMMAND ----------

dbutils.widgets.dropdown("mode", "full", ["full", "incremental"], "Run mode")

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
# MAGIC ## Step 1: Build `news_chunks`
# MAGIC
# MAGIC **Sources:**
# MAGIC - `factset_vectors_do_not_edit_or_delete.vector.sa_metadata` (news / StreetAccount metadata)
# MAGIC - `factset_vectors_do_not_edit_or_delete.vector.all_docs` (chunked text, filtered to product = 'SA')
# MAGIC - `target_companies` temp view (from config)
# MAGIC
# MAGIC **Join:** sa → all_docs on chunk_id, then inner join to target_companies via `array_contains(sa.primary_symbols, tc.sa_symbol)`.
# MAGIC
# MAGIC **Transforms:** include all sa_metadata columns, rename content → chunk_text, add ticker/ticker_region from config,
# MAGIC derive primary_ticker_region, story_year, story_month, industry, region, is_analyst_commentary, source_table.

# COMMAND ----------

news_chunks_sql = """
    SELECT * FROM (
        SELECT
            sa.*,
            docs.content AS chunk_text,
            tc.ticker,
            tc.ticker_region,
            sa.primary_symbols[0] AS primary_ticker_region,
            YEAR(sa.story_date) AS story_year,
            MONTH(sa.story_date) AS story_month,
            sa.fds_industries[0] AS industry,
            sa.fds_regions[0] AS region,
            sa.has_analyst_commentary AS is_analyst_commentary,
            'sa_metadata' AS source_table,
            ROW_NUMBER() OVER (PARTITION BY sa.chunk_id ORDER BY tc.ticker_region) AS _rn
        FROM factset_vectors_do_not_edit_or_delete.vector.sa_metadata sa
        JOIN factset_vectors_do_not_edit_or_delete.vector.all_docs docs
            ON sa.chunk_id = docs.chunk_id AND docs.product = 'SA'
        INNER JOIN target_companies tc
            ON array_contains(sa.primary_symbols, tc.sa_symbol)
        WHERE docs.content IS NOT NULL
          AND TRIM(docs.content) != ''
          AND sa.token_count != 0
    ) WHERE _rn = 1
"""

# COMMAND ----------

if mode == "full":
    print("Running FULL build: CREATE OR REPLACE TABLE ...")
    spark.sql(f"""
        CREATE OR REPLACE TABLE ks_factset_research_v3.demo.news_chunks
        USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        AS {news_chunks_sql}
    """)
    row_count = spark.table("ks_factset_research_v3.demo.news_chunks").count()
    print(f"news_chunks created with {row_count:,} rows")

elif mode == "incremental":
    print("Running INCREMENTAL build: MERGE new chunks ...")

    # Pre-merge count
    pre_count = spark.table("ks_factset_research_v3.demo.news_chunks").count()

    spark.sql(f"""
        MERGE INTO ks_factset_research_v3.demo.news_chunks AS target
        USING (
            {news_chunks_sql}
        ) AS source
        ON target.chunk_id = source.chunk_id
        WHEN NOT MATCHED THEN INSERT *
    """)

    post_count = spark.table("ks_factset_research_v3.demo.news_chunks").count()
    new_chunks = post_count - pre_count
    print(f"MERGE complete: {new_chunks:,} new chunks added ({pre_count:,} → {post_count:,})")

else:
    raise ValueError(f"Unknown mode: '{mode}'. Expected 'full' or 'incremental'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: `news_documents` Summary
# MAGIC
# MAGIC One row per `document_id` — always rebuilt from `news_chunks`.

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.demo.news_documents
    USING DELTA
    AS
    SELECT
        document_id,
        FIRST(ticker)                   AS ticker,
        FIRST(ticker_region)            AS ticker_region,
        FIRST(primary_ticker_region)    AS primary_ticker_region,
        FIRST(story_date)               AS story_date,
        FIRST(story_year)               AS story_year,
        FIRST(story_month)              AS story_month,
        FIRST(industry)                 AS industry,
        FIRST(region)                   AS region,
        FIRST(is_analyst_commentary)    AS is_analyst_commentary,
        COUNT(*)                        AS total_chunks,
        SUM(token_count)                AS total_tokens,
        SUM(LENGTH(chunk_text))         AS total_text_length
    FROM ks_factset_research_v3.demo.news_chunks
    GROUP BY document_id
""")

doc_count = spark.table("ks_factset_research_v3.demo.news_documents").count()
print(f"news_documents created with {doc_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.a — Row Counts

# COMMAND ----------

chunk_count = spark.table("ks_factset_research_v3.demo.news_chunks").count()
doc_count = spark.table("ks_factset_research_v3.demo.news_documents").count()
print(f"news_chunks:    {chunk_count:,} rows")
print(f"news_documents: {doc_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.b — Counts by Ticker

# COMMAND ----------

by_ticker = spark.sql("""
    SELECT
        ticker,
        ticker_region,
        COUNT(DISTINCT document_id) AS doc_count,
        COUNT(*)                    AS chunk_count
    FROM ks_factset_research_v3.demo.news_chunks
    GROUP BY ticker, ticker_region
    ORDER BY chunk_count DESC
""")
display(by_ticker)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.c — Analyst Commentary Breakdown

# COMMAND ----------

by_commentary = spark.sql("""
    SELECT
        is_analyst_commentary,
        COUNT(DISTINCT document_id) AS doc_count,
        COUNT(*)                    AS chunk_count
    FROM ks_factset_research_v3.demo.news_chunks
    GROUP BY is_analyst_commentary
    ORDER BY is_analyst_commentary
""")
display(by_commentary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.d — Date Distribution

# COMMAND ----------

by_date = spark.sql("""
    SELECT
        story_year,
        story_month,
        COUNT(DISTINCT document_id) AS doc_count,
        COUNT(*)                    AS chunk_count
    FROM ks_factset_research_v3.demo.news_chunks
    GROUP BY story_year, story_month
    ORDER BY story_year DESC, story_month DESC
""")
display(by_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.e — Compare to Expected (~2,925 news chunks)

# COMMAND ----------

expected_chunks = 2925
diff = chunk_count - expected_chunks
pct = (diff / expected_chunks) * 100 if expected_chunks else 0
print(f"Expected: ~{expected_chunks:,}")
print(f"Actual:    {chunk_count:,}")
print(f"Difference: {diff:+,} ({pct:+.1f}%)")
if abs(pct) > 20:
    print("WARNING: chunk count differs from expected by more than 20%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.f — Sample Rows

# COMMAND ----------

sample_df = spark.sql("""
    SELECT
        chunk_id,
        ticker,
        primary_ticker_region,
        story_date,
        story_year,
        story_month,
        industry,
        is_analyst_commentary,
        LEFT(chunk_text, 200) AS chunk_text_preview
    FROM ks_factset_research_v3.demo.news_chunks
    ORDER BY RAND()
    LIMIT 5
""")
display(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.g — Data Quality Checks

# COMMAND ----------

# Null chunk_text
null_text = spark.sql("""
    SELECT COUNT(*) AS null_chunk_text
    FROM ks_factset_research_v3.demo.news_chunks
    WHERE chunk_text IS NULL
""").collect()[0]["null_chunk_text"]
print(f"Rows with NULL chunk_text: {null_text}")
assert null_text == 0, f"FAIL: found {null_text} rows with NULL chunk_text"

# Duplicate chunk_ids
dup_chunks = spark.sql("""
    SELECT COUNT(*) AS dup_count FROM (
        SELECT chunk_id
        FROM ks_factset_research_v3.demo.news_chunks
        GROUP BY chunk_id
        HAVING COUNT(*) > 1
    )
""").collect()[0]["dup_count"]
print(f"Duplicate chunk_ids: {dup_chunks}")
if dup_chunks > 0:
    print("Duplicate chunk_id details:")
    display(spark.sql("""
        SELECT chunk_id, COUNT(*) AS cnt, COLLECT_SET(ticker_region) AS ticker_regions
        FROM ks_factset_research_v3.demo.news_chunks
        GROUP BY chunk_id
        HAVING COUNT(*) > 1
    """))
assert dup_chunks == 0, f"FAIL: found {dup_chunks} duplicate chunk_ids"

print("Data quality checks PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.h — Confirm Change Data Feed

# COMMAND ----------

cdf_props = spark.sql("""
    DESCRIBE DETAIL ks_factset_research_v3.demo.news_chunks
""").select("properties").collect()[0]["properties"]

cdf_enabled = cdf_props.get("delta.enableChangeDataFeed", "false")
print(f"delta.enableChangeDataFeed = {cdf_enabled}")
assert cdf_enabled == "true", "FAIL: Change Data Feed is not enabled on news_chunks"
print("Change Data Feed: CONFIRMED")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.i — Incremental Summary (if applicable)

# COMMAND ----------

if mode == "incremental":
    post_count = spark.table("ks_factset_research_v3.demo.news_chunks").count()
    print(f"Incremental run complete.")
    print(f"Total news_chunks: {post_count:,}")
    # Show newly added tickers
    new_ticker_chunks = spark.sql("""
        SELECT
            ticker,
            ticker_region,
            COUNT(*) AS chunk_count
        FROM ks_factset_research_v3.demo.news_chunks
        GROUP BY ticker, ticker_region
        ORDER BY chunk_count DESC
    """)
    display(new_ticker_chunks)
else:
    print("Full mode — incremental summary not applicable.")
