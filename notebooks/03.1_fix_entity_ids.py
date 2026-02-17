# Databricks notebook source

# MAGIC %md
# MAGIC # 03.1 — Fix Entity ID Mappings
# MAGIC
# MAGIC **Purpose:** Detect and fix incorrect `entity_id` values in `demo_companies`.
# MAGIC Notebook 03 used `dropDuplicates` which picked arbitrary entity_ids from
# MAGIC multiple regex-join matches — some tickers got wrong mappings.
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1 | Detect mismatches: compare resolved name vs display_name |
# MAGIC | 2 | Find correct entity_ids for mismatched tickers |
# MAGIC | 3 | Update demo_companies config table |
# MAGIC | 4 | Rebuild company_profile |
# MAGIC | 5 | Validate — confirm all names match |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Detect Mismatches

# COMMAND ----------

from pyspark.sql import functions as F

mismatch_df = spark.sql("""
    SELECT
        dc.ticker,
        dc.ticker_region,
        dc.display_name,
        dc.entity_id           AS current_entity_id,
        se.ENTITY_PROPER_NAME  AS resolved_name,
        se.ENTITY_TYPE         AS resolved_type
    FROM ks_factset_research_v3.gold.demo_companies dc
    LEFT JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON dc.entity_id = se.FACTSET_ENTITY_ID
    WHERE dc.is_active = true
    ORDER BY dc.ticker
""")

rows = mismatch_df.collect()

print(f"{'Ticker':<8} {'Display Name':<25} {'Resolved Name':<35} {'Type':<6} {'Match?'}")
print("-" * 100)

mismatches = []
for row in rows:
    display = row["display_name"] or ""
    resolved = row["resolved_name"] or "(NULL)"
    # Check if display_name appears as a substring in the resolved name (case-insensitive)
    match = display.lower() in resolved.lower() or resolved.lower() in display.lower()
    flag = "OK" if match else "MISMATCH"
    print(f"{row['ticker']:<8} {display:<25} {resolved:<35} {row['resolved_type'] or 'N/A':<6} {flag}")
    if not match:
        mismatches.append(row)

print(f"\n{len(mismatches)} mismatch(es) found out of {len(rows)} companies")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Find Correct Entity IDs

# COMMAND ----------

if not mismatches:
    print("No mismatches — nothing to fix!")
else:
    fixes = {}  # ticker -> correct entity_id

    for mm in mismatches:
        ticker_region = mm["ticker_region"]
        display_name = (mm["display_name"] or "").lower()
        ticker = mm["ticker"]

        print(f"\n{'='*70}")
        print(f"Fixing: {ticker} (current entity_id={mm['current_entity_id']}, resolved='{mm['resolved_name']}')")
        print(f"Expected: '{mm['display_name']}'")

        # Find all candidate entity_ids for this ticker_region
        candidates = spark.sql(f"""
            SELECT DISTINCT
                ese.FACTSET_ENTITY_ID  AS entity_id,
                se.ENTITY_PROPER_NAME  AS entity_name,
                se.ENTITY_TYPE         AS entity_type
            FROM delta_share_factset_do_not_delete_or_edit.sym_v1.sym_ticker_region tr
            JOIN delta_share_factset_do_not_delete_or_edit.ent_v1.ent_scr_sec_entity ese
                ON regexp_extract(tr.FSYM_ID, '^(.+)-', 1)
                 = regexp_extract(ese.FSYM_ID, '^(.+)-', 1)
            JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
                ON ese.FACTSET_ENTITY_ID = se.FACTSET_ENTITY_ID
            WHERE tr.TICKER_REGION = '{ticker_region}'
        """).collect()

        print(f"  Found {len(candidates)} candidate(s):")
        best_match = None
        for c in candidates:
            ename = (c["entity_name"] or "").lower()
            etype = c["entity_type"] or ""
            is_pub = etype == "PUB"
            name_match = display_name in ename or ename in display_name
            marker = ""
            if is_pub and name_match:
                marker = " <<<< BEST MATCH"
                best_match = c
            elif is_pub:
                marker = " (PUB)"
            print(f"    {c['entity_id']:<15} {c['entity_name']:<40} type={etype}{marker}")

        # Fallback: if no name match among PUB, pick first PUB
        if best_match is None:
            pub_candidates = [c for c in candidates if c["entity_type"] == "PUB"]
            if pub_candidates:
                best_match = pub_candidates[0]
                print(f"  No name match — using first PUB entity: {best_match['entity_id']}")

        if best_match:
            fixes[ticker] = best_match["entity_id"]
            print(f"  FIX: {ticker} → entity_id = {best_match['entity_id']} ({best_match['entity_name']})")
        else:
            print(f"  WARNING: No suitable entity_id found for {ticker}")

    print(f"\n{'='*70}")
    print(f"Fixes to apply: {len(fixes)}")
    for t, eid in fixes.items():
        print(f"  {t} → {eid}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Update demo_companies

# COMMAND ----------

if not mismatches:
    print("No mismatches — skipping updates.")
else:
    update_count = 0
    for ticker, new_entity_id in fixes.items():
        spark.sql(f"""
            UPDATE ks_factset_research_v3.gold.demo_companies
            SET entity_id = '{new_entity_id}'
            WHERE ticker = '{ticker}'
        """)
        update_count += 1
        print(f"  Updated {ticker} → entity_id = {new_entity_id}")

    print(f"\n{update_count} row(s) updated in demo_companies")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Rebuild company_profile
# MAGIC
# MAGIC Re-create the `company_profile` table using the corrected entity_ids.
# MAGIC This is the same SQL as notebook 07, step 2.

# COMMAND ----------

# Reload target_companies view with updated config
spark.sql("""
    CREATE OR REPLACE TEMP VIEW target_companies AS
    SELECT * FROM ks_factset_research_v3.gold.v_active_companies
""")

spark.sql("""
    CREATE OR REPLACE TABLE ks_factset_research_v3.gold.company_profile
    USING DELTA
    AS
    SELECT
        tc.ticker,
        tc.ticker_region,
        tc.entity_id,
        COALESCE(se.ENTITY_PROPER_NAME, tc.display_name) AS company_name,
        se.ISO_COUNTRY                          AS country,
        se.ENTITY_TYPE                          AS entity_type,
        ep.ENTITY_PROFILE_TYPE                  AS profile_type,
        ep.ENTITY_PROFILE                       AS profile_raw,
        tc.fsym_id,
        tc.display_name
    FROM target_companies tc
    LEFT JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON tc.entity_id = se.FACTSET_ENTITY_ID
    LEFT JOIN delta_share_factset_do_not_delete_or_edit.ff_v3.ff_entity_profiles ep
        ON tc.entity_id = ep.FACTSET_ENTITY_ID
""")

profile_count = spark.table("ks_factset_research_v3.gold.company_profile").count()
print(f"company_profile rebuilt with {profile_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Validate

# COMMAND ----------

print("=" * 90)
print("VALIDATION: All companies after fix")
print("=" * 90)

validation_df = spark.sql("""
    SELECT
        dc.ticker,
        dc.display_name,
        dc.entity_id,
        cp.company_name        AS profile_company_name,
        cp.entity_type         AS profile_entity_type
    FROM ks_factset_research_v3.gold.demo_companies dc
    LEFT JOIN ks_factset_research_v3.gold.company_profile cp
        ON dc.ticker = cp.ticker
    WHERE dc.is_active = true
    ORDER BY dc.ticker
""")

val_rows = validation_df.collect()

print(f"\n{'Ticker':<8} {'Display Name':<25} {'Profile Name':<35} {'Type':<6} {'Match?'}")
print("-" * 90)

remaining_mismatches = 0
for row in val_rows:
    display = (row["display_name"] or "").lower()
    profile = (row["profile_company_name"] or "").lower()
    match = display in profile or profile in display
    flag = "OK" if match else "MISMATCH"
    if not match:
        remaining_mismatches += 1
    print(f"{row['ticker']:<8} {row['display_name'] or '':<25} {row['profile_company_name'] or '':<35} {row['profile_entity_type'] or '':<6} {flag}")

print(f"\nRemaining mismatches: {remaining_mismatches}")
assert remaining_mismatches == 0, f"FAIL: {remaining_mismatches} mismatches remain after fix"
print("\nAll entity_id mappings are correct.")
