# Databricks notebook source

# MAGIC %md
# MAGIC # 08 — Vector Search Indexes
# MAGIC
# MAGIC **Purpose:** Create three Databricks Vector Search indexes with **managed embeddings** on the
# MAGIC chunk tables (`filing_chunks`, `earnings_chunks`, `news_chunks`). Databricks handles embedding
# MAGIC automatically at sync time using the `databricks-gte-large-en` model.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Chunk tables exist with `delta.enableChangeDataFeed = true` (Notebooks 04–06)
# MAGIC - Unity Catalog enabled
# MAGIC
# MAGIC **Parameters:**
# MAGIC - `vs_endpoint` — Vector Search endpoint name
# MAGIC - `force_recreate` — Set to `yes` to delete and rebuild all indexes (use after `CREATE OR REPLACE TABLE` on chunk tables)
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1 | Parameters & verify Vector Search endpoint |
# MAGIC | 2 | Create 3 Delta Sync indexes (managed embeddings) |
# MAGIC | 3 | Trigger sync and poll until ONLINE |
# MAGIC | 4 | Test each index with `query_text` |
# MAGIC | 5 | Summary — status and test scores |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Parameters & Verify Endpoint

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("vs_endpoint", "one-env-shared-endpoint-11", "Vector Search endpoint name")
dbutils.widgets.dropdown("force_recreate", "no", ["no", "yes"], "Force delete & recreate indexes")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import time

vsc = VectorSearchClient()

# COMMAND ----------

ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint").strip()
CATALOG = "ks_factset_research_v3"
SCHEMA = "demo"
EMBEDDING_MODEL = "databricks-gte-large-en"
FORCE_RECREATE = dbutils.widgets.get("force_recreate").strip().lower() == "yes"

print(f"Endpoint: {ENDPOINT_NAME}")
if FORCE_RECREATE:
    print("FORCE RECREATE mode — existing indexes will be deleted and rebuilt")

# COMMAND ----------

# Verify shared endpoint is online
endpoint_info = vsc.get_endpoint(ENDPOINT_NAME)
ep_state = endpoint_info.get("endpoint_status", {}).get("state", "UNKNOWN")
print(f"Endpoint '{ENDPOINT_NAME}' — status: {ep_state}")
assert ep_state == "ONLINE", f"Shared endpoint is not ONLINE (state: {ep_state}). Contact your workspace admin."

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Create Delta Sync Indexes (Managed Embeddings)
# MAGIC
# MAGIC Three indexes, all using:
# MAGIC - `embedding_source_column = "chunk_text"` — Databricks embeds this column automatically
# MAGIC - `embedding_model_endpoint_name = "databricks-gte-large-en"` — the embedding model
# MAGIC - `pipeline_type = "TRIGGERED"` — sync on demand, not continuous
# MAGIC
# MAGIC No pre-computed embedding columns needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.a — Filing Search Index

# COMMAND ----------

FILING_INDEX = f"{CATALOG}.{SCHEMA}.filing_search_index"
FILING_SOURCE = f"{CATALOG}.{SCHEMA}.filing_chunks"

FILING_SYNC_COLS = [
    "chunk_id",
    "document_id",
    "company_name",
    "ticker",
    "ticker_region",
    "fds_filing_type",
    "doc_type_label",
    "is_exhibit",
    "exhibit_title",
    "chunk_text",
    "chunk_num",
    "chunk_order",
    "acceptance_date",
    "filing_year",
    "filing_quarter",
    "text_length",
    "token_count",
]

def _create_or_recreate_index(index_name, source_table, sync_cols):
    """Create index, optionally deleting first if FORCE_RECREATE is set."""
    exists = False
    try:
        vsc.get_index(ENDPOINT_NAME, index_name).describe()
        exists = True
    except Exception:
        pass

    if exists and FORCE_RECREATE:
        print(f"Deleting existing index '{index_name}' for recreation ...")
        vsc.delete_index(ENDPOINT_NAME, index_name)
        # Wait for deletion to propagate
        time.sleep(10)
        exists = False

    if exists:
        print(f"Index '{index_name}' already exists — skipping creation")
    else:
        print(f"Creating index '{index_name}' ...")
        vsc.create_delta_sync_index(
            endpoint_name=ENDPOINT_NAME,
            index_name=index_name,
            source_table_name=source_table,
            primary_key="chunk_id",
            pipeline_type="TRIGGERED",
            embedding_source_column="chunk_text",
            embedding_model_endpoint_name=EMBEDDING_MODEL,
            columns_to_sync=sync_cols,
        )
        print(f"Index '{index_name}': CREATE requested")

# COMMAND ----------

_create_or_recreate_index(FILING_INDEX, FILING_SOURCE, FILING_SYNC_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.b — Earnings Search Index

# COMMAND ----------

EARNINGS_INDEX = f"{CATALOG}.{SCHEMA}.earnings_search_index"
EARNINGS_SOURCE = f"{CATALOG}.{SCHEMA}.earnings_chunks"

EARNINGS_SYNC_COLS = [
    "chunk_id",
    "document_id",
    "company_name",
    "ticker",
    "ticker_region",
    "chunk_text",
    "chunk_num",
    "chunk_order",
    "transcript_year",
    "transcript_quarter",
    "text_length",
    "token_count",
]

# Add optional columns if they exist in the source table
optional_earnings_cols = ["speaker_name", "speaker_type", "event_type_id"]
earnings_table_cols = [c.name for c in spark.table(EARNINGS_SOURCE).schema]
for col in optional_earnings_cols:
    if col in earnings_table_cols:
        EARNINGS_SYNC_COLS.append(col)
        print(f"  + adding '{col}' to earnings sync columns")
    else:
        print(f"  - '{col}' not found in {EARNINGS_SOURCE}, skipping")

print(f"Earnings sync columns ({len(EARNINGS_SYNC_COLS)}): {EARNINGS_SYNC_COLS}")

# COMMAND ----------

_create_or_recreate_index(EARNINGS_INDEX, EARNINGS_SOURCE, EARNINGS_SYNC_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.c — News Search Index

# COMMAND ----------

NEWS_INDEX = f"{CATALOG}.{SCHEMA}.news_search_index"
NEWS_SOURCE = f"{CATALOG}.{SCHEMA}.news_chunks"

NEWS_SYNC_COLS = [
    "chunk_id",
    "document_id",
    "ticker",
    "ticker_region",
    "primary_ticker_region",
    "headline",
    "chunk_text",
    "chunk_num",
    "chunk_order",
    "story_date",
    "story_year",
    "story_month",
    "has_analyst_commentary",
    "element_type",
    "industry",
    "region",
    "text_length",
    "token_count",
]

_create_or_recreate_index(NEWS_INDEX, NEWS_SOURCE, NEWS_SYNC_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Trigger Sync and Poll Until ONLINE

# COMMAND ----------

INDEX_NAMES = [FILING_INDEX, EARNINGS_INDEX, NEWS_INDEX]

# Trigger initial sync on each index
for idx_name in INDEX_NAMES:
    try:
        vsc.get_index(ENDPOINT_NAME, idx_name).sync()
        print(f"Sync triggered: {idx_name}")
    except Exception as e:
        print(f"Sync trigger for {idx_name}: {e}")

# COMMAND ----------

# Poll until all indexes are ONLINE
MAX_WAIT_MINUTES = 90
POLL_INTERVAL = 60

start_time = time.time()
pending = set(INDEX_NAMES)

print(f"Waiting for {len(pending)} indexes to reach ONLINE status ...")
print(f"Max wait: {MAX_WAIT_MINUTES} minutes, polling every {POLL_INTERVAL}s\n")

while pending:
    elapsed = (time.time() - start_time) / 60
    if elapsed > MAX_WAIT_MINUTES:
        print(f"\nTIMEOUT after {elapsed:.1f} minutes. Still pending: {pending}")
        break

    for idx_name in list(pending):
        try:
            status = vsc.get_index(ENDPOINT_NAME, idx_name).describe()
            state = status.get("status", {}).get("ready", False)
            detail = status.get("status", {}).get("message", "")
            if state:
                print(f"  ONLINE: {idx_name}")
                pending.discard(idx_name)
            else:
                print(f"  [{elapsed:.0f}m] {idx_name}: not ready — {detail}")
        except Exception as e:
            print(f"  [{elapsed:.0f}m] {idx_name}: error checking status — {e}")

    if pending:
        time.sleep(POLL_INTERVAL)

if not pending:
    total_min = (time.time() - start_time) / 60
    print(f"\nAll {len(INDEX_NAMES)} indexes are ONLINE ({total_min:.1f} minutes)")
else:
    print(f"\nWARNING: {len(pending)} indexes still not ready: {pending}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Test Each Index with `query_text`
# MAGIC
# MAGIC Managed embeddings support `query_text` directly — no need to pre-compute a query vector.

# COMMAND ----------

TEST_QUERY = "What were the key risk factors disclosed in the most recent annual report?"
NUM_RESULTS = 3

print(f"Test query: \"{TEST_QUERY}\"")
print(f"Returning top {NUM_RESULTS} results per index\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.a — Filing Index Test

# COMMAND ----------

filing_results = vsc.get_index(ENDPOINT_NAME, FILING_INDEX).similarity_search(
    query_text=TEST_QUERY,
    columns=["chunk_id", "ticker", "company_name", "fds_filing_type", "filing_year", "chunk_text"],
    num_results=NUM_RESULTS,
)

print(f"Filing index — {len(filing_results.get('result', {}).get('data_array', []))} results:")
for i, row in enumerate(filing_results.get("result", {}).get("data_array", [])):
    print(f"\n--- Result {i+1} (score: {row[-1]:.4f}) ---")
    print(f"  chunk_id:  {row[0]}")
    print(f"  ticker:    {row[1]}")
    print(f"  company:   {row[2]}")
    print(f"  type:      {row[3]}")
    print(f"  year:      {row[4]}")
    print(f"  text:      {str(row[5])[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.b — Earnings Index Test

# COMMAND ----------

earnings_query = "What is management's outlook for revenue growth next quarter?"

earnings_results = vsc.get_index(ENDPOINT_NAME, EARNINGS_INDEX).similarity_search(
    query_text=earnings_query,
    columns=["chunk_id", "ticker", "company_name", "transcript_year", "transcript_quarter", "chunk_text"],
    num_results=NUM_RESULTS,
)

print(f"Earnings index — {len(earnings_results.get('result', {}).get('data_array', []))} results:")
for i, row in enumerate(earnings_results.get("result", {}).get("data_array", [])):
    print(f"\n--- Result {i+1} (score: {row[-1]:.4f}) ---")
    print(f"  chunk_id:  {row[0]}")
    print(f"  ticker:    {row[1]}")
    print(f"  company:   {row[2]}")
    print(f"  year:      {row[3]}")
    print(f"  quarter:   {row[4]}")
    print(f"  text:      {str(row[5])[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.c — News Index Test

# COMMAND ----------

news_query = "Recent analyst commentary on semiconductor industry trends"

news_results = vsc.get_index(ENDPOINT_NAME, NEWS_INDEX).similarity_search(
    query_text=news_query,
    columns=["chunk_id", "ticker", "headline", "story_date", "industry", "chunk_text"],
    num_results=NUM_RESULTS,
)

print(f"News index — {len(news_results.get('result', {}).get('data_array', []))} results:")
for i, row in enumerate(news_results.get("result", {}).get("data_array", [])):
    print(f"\n--- Result {i+1} (score: {row[-1]:.4f}) ---")
    print(f"  chunk_id:  {row[0]}")
    print(f"  ticker:    {row[1]}")
    print(f"  headline:  {row[2]}")
    print(f"  date:      {row[3]}")
    print(f"  industry:  {row[4]}")
    print(f"  text:      {str(row[5])[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Summary — Status and Test Scores

# COMMAND ----------

print("=" * 80)
print("VECTOR SEARCH INDEX SUMMARY")
print("=" * 80)
print(f"\nEndpoint: {ENDPOINT_NAME}")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"Embedding source column: chunk_text")
print(f"Sync mode: TRIGGERED\n")

for idx_name in INDEX_NAMES:
    try:
        desc = vsc.get_index(ENDPOINT_NAME, idx_name).describe()
        ready = desc.get("status", {}).get("ready", False)
        num_rows = desc.get("status", {}).get("indexed_row_count", "N/A")
        state_str = "ONLINE" if ready else "NOT READY"
        print(f"  {idx_name}")
        print(f"    Status:       {state_str}")
        print(f"    Indexed rows: {num_rows}")
    except Exception as e:
        print(f"  {idx_name}")
        print(f"    Status: ERROR — {e}")
    print()

# Test result scores summary
print("-" * 80)
print("TEST QUERY RESULTS\n")

test_cases = [
    ("Filing",   TEST_QUERY,     filing_results),
    ("Earnings", earnings_query, earnings_results),
    ("News",     news_query,     news_results),
]

for label, query, results in test_cases:
    rows = results.get("result", {}).get("data_array", [])
    if rows:
        scores = [r[-1] for r in rows]
        print(f"  {label} index:")
        print(f"    Query:      \"{query[:80]}...\"" if len(query) > 80 else f"    Query:      \"{query}\"")
        print(f"    Results:    {len(rows)}")
        print(f"    Top score:  {scores[0]:.4f}")
        print(f"    Avg score:  {sum(scores)/len(scores):.4f}")
    else:
        print(f"  {label} index: NO RESULTS")
    print()

print("=" * 80)
print("DONE")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Re-Sync Note
# MAGIC
# MAGIC After adding a new company and rerunning chunk notebooks (04–06), trigger a re-sync
# MAGIC on each index. The managed embeddings will automatically embed the new rows.
# MAGIC
# MAGIC ```python
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC vsc = VectorSearchClient()
# MAGIC
# MAGIC ENDPOINT = dbutils.widgets.get("vs_endpoint").strip()
# MAGIC index_names = [
# MAGIC     "ks_factset_research_v3.demo.filing_search_index",
# MAGIC     "ks_factset_research_v3.demo.earnings_search_index",
# MAGIC     "ks_factset_research_v3.demo.news_search_index",
# MAGIC ]
# MAGIC
# MAGIC for idx in index_names:
# MAGIC     vsc.get_index(ENDPOINT, idx).sync()
# MAGIC     print(f"Sync triggered: {idx}")
# MAGIC ```
