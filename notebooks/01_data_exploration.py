# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Data Exploration & Schema Discovery
# MAGIC
# MAGIC **Purpose:** Validate coverage and discover schemas across all FactSet source data.
# MAGIC
# MAGIC | Source | Table | Expected Text Column |
# MAGIC |---|---|---|
# MAGIC | Filings (EDG) | `factset_vectors_do_not_edit_or_delete.vector.edg_metadata` | `table_text` (confirmed) |
# MAGIC | Earnings Transcripts (FCST) | `factset_vectors_do_not_edit_or_delete.vector.fcst_metadata` | **unknown** |
# MAGIC | News / StreetAccount (SA) | `factset_vectors_do_not_edit_or_delete.vector.sa_metadata` | **unknown** |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1: Full Schema Discovery — Vector Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 — EDG Metadata (Filings)

# COMMAND ----------

edg_table = "factset_vectors_do_not_edit_or_delete.vector.edg_metadata"
edg_df = spark.table(edg_table)

print(f"=== SCHEMA: {edg_table} ===")
print(f"{'Column Name':<40} {'Data Type':<25} {'Nullable'}")
print("-" * 80)
for field in edg_df.schema.fields:
    print(f"{field.name:<40} {str(field.dataType):<25} {field.nullable}")

# COMMAND ----------

edg_count = edg_df.count()
print(f"Total row count: {edg_count:,}")

# COMMAND ----------

display(edg_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 — FCST Metadata (Earnings Transcripts)

# COMMAND ----------

fcst_table = "factset_vectors_do_not_edit_or_delete.vector.fcst_metadata"
fcst_df = spark.table(fcst_table)

print(f"=== SCHEMA: {fcst_table} ===")
print(f"{'Column Name':<40} {'Data Type':<25} {'Nullable'}")
print("-" * 80)
for field in fcst_df.schema.fields:
    print(f"{field.name:<40} {str(field.dataType):<25} {field.nullable}")

# COMMAND ----------

fcst_count = fcst_df.count()
print(f"Total row count: {fcst_count:,}")

# COMMAND ----------

display(fcst_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 — SA Metadata (News / StreetAccount)

# COMMAND ----------

sa_table = "factset_vectors_do_not_edit_or_delete.vector.sa_metadata"
sa_df = spark.table(sa_table)

print(f"=== SCHEMA: {sa_table} ===")
print(f"{'Column Name':<40} {'Data Type':<25} {'Nullable'}")
print("-" * 80)
for field in sa_df.schema.fields:
    print(f"{field.name:<40} {str(field.dataType):<25} {field.nullable}")

# COMMAND ----------

sa_count = sa_df.count()
print(f"Total row count: {sa_count:,}")

# COMMAND ----------

display(sa_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2: Text Column Identification
# MAGIC
# MAGIC For each table, profile every string column to find the body-text column.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 — EDG: Confirm `table_text` is the text body column

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

edg_string_cols = [
    field.name for field in edg_df.schema.fields
    if isinstance(field.dataType, StringType)
]

print(f"String columns in edg_metadata ({len(edg_string_cols)}):")
print(edg_string_cols)
print()

if edg_string_cols:
    edg_profile_exprs = []
    for col_name in edg_string_cols:
        edg_profile_exprs.extend([
            F.avg(F.length(F.col(col_name))).alias(f"{col_name}__avg_len"),
            F.max(F.length(F.col(col_name))).alias(f"{col_name}__max_len"),
        ])

    edg_profile = edg_df.select(edg_profile_exprs).collect()[0]

    print(f"{'Column':<40} {'Avg Length':>12} {'Max Length':>12}")
    print("-" * 66)
    for col_name in edg_string_cols:
        avg_len = edg_profile[f"{col_name}__avg_len"]
        max_len = edg_profile[f"{col_name}__max_len"]
        avg_display = f"{avg_len:,.1f}" if avg_len is not None else "NULL"
        max_display = f"{max_len:,}" if max_len is not None else "NULL"
        print(f"{col_name:<40} {avg_display:>12} {max_display:>12}")
else:
    print("No StringType columns found. Printing all column types for inspection:")
    for field in edg_df.schema.fields:
        print(f"  {field.name:<40} {str(field.dataType)}")

# COMMAND ----------

# Show a sample value (first 200 chars) for the confirmed text column
edg_sample = edg_df.select(
    F.substring(F.col("table_text"), 1, 200).alias("table_text_sample")
).where(F.col("table_text").isNotNull()).limit(1).collect()

print("=== EDG text body column: table_text (CONFIRMED) ===")
if edg_sample:
    print(f"Sample (first 200 chars): {edg_sample[0]['table_text_sample']}")
else:
    print("WARNING: No non-null values found in table_text!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 — FCST: Discover the text body column

# COMMAND ----------

fcst_string_cols = [
    field.name for field in fcst_df.schema.fields
    if isinstance(field.dataType, StringType)
]

print(f"String columns in fcst_metadata ({len(fcst_string_cols)}):")
print(fcst_string_cols)
print()

best_fcst_col = None
best_fcst_avg = 0

if fcst_string_cols:
    fcst_profile_exprs = []
    for col_name in fcst_string_cols:
        fcst_profile_exprs.extend([
            F.avg(F.length(F.col(col_name))).alias(f"{col_name}__avg_len"),
            F.max(F.length(F.col(col_name))).alias(f"{col_name}__max_len"),
        ])

    fcst_profile = fcst_df.select(fcst_profile_exprs).collect()[0]

    print(f"{'Column':<40} {'Avg Length':>12} {'Max Length':>12}")
    print("-" * 66)
    for col_name in fcst_string_cols:
        avg_len = fcst_profile[f"{col_name}__avg_len"]
        max_len = fcst_profile[f"{col_name}__max_len"]
        avg_display = f"{avg_len:,.1f}" if avg_len is not None else "NULL"
        max_display = f"{max_len:,}" if max_len is not None else "NULL"
        marker = ""
        if avg_len is not None and avg_len > best_fcst_avg:
            best_fcst_avg = avg_len
            best_fcst_col = col_name
            marker = " <-- candidate"
        print(f"{col_name:<40} {avg_display:>12} {max_display:>12}{marker}")

    print()
    print(f">>> Best candidate text body column: {best_fcst_col} (avg length {best_fcst_avg:,.1f})")
else:
    print("No StringType columns found. Printing all column types for inspection:")
    for field in fcst_df.schema.fields:
        print(f"  {field.name:<40} {str(field.dataType)}")

# COMMAND ----------

# Show sample value for each string column (first 200 chars) so we can visually confirm
print("=== FCST: Sample values for all string columns (first 200 chars) ===\n")

for col_name in fcst_string_cols:
    sample = fcst_df.select(
        F.substring(F.col(col_name), 1, 200).alias("sample")
    ).where(F.col(col_name).isNotNull()).limit(1).collect()
    sample_val = sample[0]["sample"] if sample else "<ALL NULL>"
    print(f"--- {col_name} ---")
    print(f"  {sample_val}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 — SA: Discover the text body column

# COMMAND ----------

sa_string_cols = [
    field.name for field in sa_df.schema.fields
    if isinstance(field.dataType, StringType)
]

print(f"String columns in sa_metadata ({len(sa_string_cols)}):")
print(sa_string_cols)
print()

best_sa_col = None
best_sa_avg = 0

if sa_string_cols:
    sa_profile_exprs = []
    for col_name in sa_string_cols:
        sa_profile_exprs.extend([
            F.avg(F.length(F.col(col_name))).alias(f"{col_name}__avg_len"),
            F.max(F.length(F.col(col_name))).alias(f"{col_name}__max_len"),
        ])

    sa_profile = sa_df.select(sa_profile_exprs).collect()[0]

    print(f"{'Column':<40} {'Avg Length':>12} {'Max Length':>12}")
    print("-" * 66)
    for col_name in sa_string_cols:
        avg_len = sa_profile[f"{col_name}__avg_len"]
        max_len = sa_profile[f"{col_name}__max_len"]
        avg_display = f"{avg_len:,.1f}" if avg_len is not None else "NULL"
        max_display = f"{max_len:,}" if max_len is not None else "NULL"
        marker = ""
        if avg_len is not None and avg_len > best_sa_avg:
            best_sa_avg = avg_len
            best_sa_col = col_name
            marker = " <-- candidate"
        print(f"{col_name:<40} {avg_display:>12} {max_display:>12}{marker}")

    print()
    if best_sa_col and best_sa_avg > 100:
        print(f">>> Best candidate text body column: {best_sa_col} (avg length {best_sa_avg:,.1f})")
    else:
        print(">>> WARNING: No string column with avg length > 100 found.")
        print(">>> The SA table may NOT contain body text — only metadata + headline.")
        print(f">>> Longest avg string column: {best_sa_col} (avg length {best_sa_avg:,.1f})")
else:
    print("No StringType columns found. Printing all column types for inspection:")
    for field in sa_df.schema.fields:
        print(f"  {field.name:<40} {str(field.dataType)}")

# COMMAND ----------

# Show sample value for each string column (first 200 chars) so we can visually confirm
print("=== SA: Sample values for all string columns (first 200 chars) ===\n")

for col_name in sa_string_cols:
    sample = sa_df.select(
        F.substring(F.col(col_name), 1, 200).alias("sample")
    ).where(F.col(col_name).isNotNull()).limit(1).collect()
    sample_val = sample[0]["sample"] if sample else "<ALL NULL>"
    print(f"--- {col_name} ---")
    print(f"  {sample_val}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 — Summary: Text Column Mapping

# COMMAND ----------

print("=" * 70)
print("TEXT COLUMN IDENTIFICATION — SUMMARY (metadata tables only)")
print("=" * 70)
print()
print(f"{'Table':<20} {'Text Body Column':<30} {'Status'}")
print("-" * 70)
print(f"{'edg_metadata':<20} {'table_text':<30} CONFIRMED")
print(f"{'fcst_metadata':<20} {str(best_fcst_col):<30} {'DISCOVERED' if best_fcst_avg > 100 else 'NO BODY TEXT — metadata only'}")
print(f"{'sa_metadata':<20} {str(best_sa_col):<30} {'DISCOVERED' if best_sa_avg > 100 else 'NO BODY TEXT — metadata only'}")
print()
print(">>> FCST and SA body text lives in the all_docs table (see Section 3).")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3: all_docs — The Text Content Table
# MAGIC
# MAGIC The metadata tables for FCST and SA do not contain body text.
# MAGIC The actual chunked text content is stored in `all_docs`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 — Schema & Row Count

# COMMAND ----------

all_docs_table = "factset_vectors_do_not_edit_or_delete.vector.all_docs"
all_docs_df = spark.table(all_docs_table)

print(f"=== SCHEMA: {all_docs_table} ===")
print(f"{'Column Name':<40} {'Data Type':<25} {'Nullable'}")
print("-" * 80)
for field in all_docs_df.schema.fields:
    print(f"{field.name:<40} {str(field.dataType):<25} {field.nullable}")

# COMMAND ----------

all_docs_count = all_docs_df.count()
print(f"Total row count: {all_docs_count:,}")

# COMMAND ----------

display(all_docs_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 — Identify Text Body Column in all_docs

# COMMAND ----------

all_docs_string_cols = [
    field.name for field in all_docs_df.schema.fields
    if isinstance(field.dataType, StringType)
]

print(f"String columns in all_docs ({len(all_docs_string_cols)}):")
print(all_docs_string_cols)
print()

if all_docs_string_cols:
    ad_profile_exprs = []
    for col_name in all_docs_string_cols:
        ad_profile_exprs.extend([
            F.avg(F.length(F.col(col_name))).alias(f"{col_name}__avg_len"),
            F.max(F.length(F.col(col_name))).alias(f"{col_name}__max_len"),
        ])

    ad_profile = all_docs_df.select(ad_profile_exprs).collect()[0]

    print(f"{'Column':<40} {'Avg Length':>12} {'Max Length':>12}")
    print("-" * 66)
    best_ad_col = None
    best_ad_avg = 0
    for col_name in all_docs_string_cols:
        avg_len = ad_profile[f"{col_name}__avg_len"]
        max_len = ad_profile[f"{col_name}__max_len"]
        avg_display = f"{avg_len:,.1f}" if avg_len is not None else "NULL"
        max_display = f"{max_len:,}" if max_len is not None else "NULL"
        marker = ""
        if avg_len is not None and avg_len > best_ad_avg:
            best_ad_avg = avg_len
            best_ad_col = col_name
            marker = " <-- candidate"
        print(f"{col_name:<40} {avg_display:>12} {max_display:>12}{marker}")

    print()
    print(f">>> Text body column: {best_ad_col} (avg length {best_ad_avg:,.1f})")
else:
    print("No StringType columns found. Printing all column types:")
    for field in all_docs_df.schema.fields:
        print(f"  {field.name:<40} {str(field.dataType)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 — Join Key Discovery: How does all_docs link to metadata tables?

# COMMAND ----------

# Find overlapping column names between all_docs and each metadata table
ad_cols = set(f.name for f in all_docs_df.schema.fields)
edg_cols = set(f.name for f in edg_df.schema.fields)
fcst_cols = set(f.name for f in fcst_df.schema.fields)
sa_cols = set(f.name for f in sa_df.schema.fields)

print("=== Shared columns (potential join keys) ===\n")

print(f"all_docs ∩ edg_metadata  ({len(ad_cols & edg_cols)}):")
for c in sorted(ad_cols & edg_cols):
    print(f"  {c}")
print()

print(f"all_docs ∩ fcst_metadata ({len(ad_cols & fcst_cols)}):")
for c in sorted(ad_cols & fcst_cols):
    print(f"  {c}")
print()

print(f"all_docs ∩ sa_metadata   ({len(ad_cols & sa_cols)}):")
for c in sorted(ad_cols & sa_cols):
    print(f"  {c}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 — Source Breakdown in all_docs

# COMMAND ----------

# Check if there's a product/source column that distinguishes EDG vs FCST vs SA
# Look for columns like 'product', 'source', 'doc_type', 'type'
candidate_source_cols = [c for c in all_docs_string_cols if c.lower() in (
    'product', 'source', 'doc_type', 'type', 'data_source', 'source_type'
)]

if candidate_source_cols:
    for col_name in candidate_source_cols:
        print(f"=== Distribution of '{col_name}' in all_docs ===")
        display(all_docs_df.groupBy(col_name).count().orderBy(F.desc("count")))
        print()
else:
    print("No obvious source-type column found. Checking all low-cardinality string columns...")
    for col_name in all_docs_string_cols:
        n_distinct = all_docs_df.select(F.countDistinct(col_name)).collect()[0][0]
        if n_distinct <= 20:
            print(f"\n=== '{col_name}' has {n_distinct} distinct values ===")
            display(all_docs_df.groupBy(col_name).count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 — Validate Join: Sample joined row for each source

# COMMAND ----------

# Test join with FCST metadata using chunk_id (most likely key)
# First check if chunk_id exists in all_docs
if "chunk_id" in ad_cols:
    print("=== Sample join: fcst_metadata → all_docs on chunk_id ===")
    fcst_joined = fcst_df.alias("m").join(
        all_docs_df.alias("d"),
        F.col("m.chunk_id") == F.col("d.chunk_id"),
        "inner"
    ).limit(3)
    print(f"Join result columns: {fcst_joined.columns}")
    display(fcst_joined)

    print("\n=== Sample join: sa_metadata → all_docs on chunk_id ===")
    sa_joined = sa_df.alias("m").join(
        all_docs_df.alias("d"),
        F.col("m.chunk_id") == F.col("d.chunk_id"),
        "inner"
    ).limit(3)
    display(sa_joined)
else:
    print("chunk_id not found in all_docs. Check shared columns above for the correct join key.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 — Final Summary

# COMMAND ----------

print("=" * 70)
print("COMPLETE DATA MAP — FINAL SUMMARY")
print("=" * 70)
print()
print("METADATA TABLES (for filtering):")
print(f"  edg_metadata   {edg_count:>12,} rows — has table_text inline")
print(f"  fcst_metadata  {fcst_count:>12,} rows — metadata only, join to all_docs for text")
print(f"  sa_metadata    {sa_count:>12,} rows — metadata only, join to all_docs for text")
print()
print("TEXT TABLE:")
print(f"  all_docs       {all_docs_count:>12,} rows — contains body text for all sources")
print()
print("JOIN KEY: check Section 3.3 output above")
print("SOURCE COLUMN: check Section 3.4 output above")
