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

edg_string_cols = [
    field.name for field in edg_df.schema.fields
    if str(field.dataType) == "StringType"
]

print(f"String columns in edg_metadata ({len(edg_string_cols)}):")
print(edg_string_cols)
print()

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
    if str(field.dataType) == "StringType"
]

print(f"String columns in fcst_metadata ({len(fcst_string_cols)}):")
print(fcst_string_cols)
print()

fcst_profile_exprs = []
for col_name in fcst_string_cols:
    fcst_profile_exprs.extend([
        F.avg(F.length(F.col(col_name))).alias(f"{col_name}__avg_len"),
        F.max(F.length(F.col(col_name))).alias(f"{col_name}__max_len"),
    ])

fcst_profile = fcst_df.select(fcst_profile_exprs).collect()[0]

print(f"{'Column':<40} {'Avg Length':>12} {'Max Length':>12}")
print("-" * 66)
best_fcst_col = None
best_fcst_avg = 0
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
    if str(field.dataType) == "StringType"
]

print(f"String columns in sa_metadata ({len(sa_string_cols)}):")
print(sa_string_cols)
print()

sa_profile_exprs = []
for col_name in sa_string_cols:
    sa_profile_exprs.extend([
        F.avg(F.length(F.col(col_name))).alias(f"{col_name}__avg_len"),
        F.max(F.length(F.col(col_name))).alias(f"{col_name}__max_len"),
    ])

sa_profile = sa_df.select(sa_profile_exprs).collect()[0]

print(f"{'Column':<40} {'Avg Length':>12} {'Max Length':>12}")
print("-" * 66)
best_sa_col = None
best_sa_avg = 0
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
print("TEXT COLUMN IDENTIFICATION — SUMMARY")
print("=" * 70)
print()
print(f"{'Table':<20} {'Text Body Column':<30} {'Status'}")
print("-" * 70)
print(f"{'edg_metadata':<20} {'table_text':<30} CONFIRMED")
print(f"{'fcst_metadata':<20} {str(best_fcst_col):<30} {'DISCOVERED' if best_fcst_avg > 100 else 'NEEDS REVIEW'}")
print(f"{'sa_metadata':<20} {str(best_sa_col):<30} {'DISCOVERED' if best_sa_avg > 100 else 'NEEDS REVIEW / MAY NOT EXIST'}")
print()
print(">>> Review the sample values above to confirm these identifications.")
print(">>> Update the config in src/ once confirmed.")
