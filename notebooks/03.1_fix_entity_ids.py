# Databricks notebook source

# MAGIC %md
# MAGIC # 03.1 — Fix Entity ID Mappings
# MAGIC
# MAGIC **Purpose:** Detect and fix incorrect `entity_id` values in `demo_companies`.
# MAGIC
# MAGIC Notebook 02's regex-join (`regexp_extract(FSYM_ID, '^(.+)-', 1)`) between
# MAGIC `sym_ticker_region` and `ent_scr_sec_entity` produces wrong matches because
# MAGIC the base FSYM_ID roots differ between listing-level (‑R) and security-level (‑S)
# MAGIC identifiers. `dropDuplicates` then picked arbitrary wrong rows.
# MAGIC
# MAGIC **Resolution strategy (in priority order):**
# MAGIC 1. **Crosswalk table** — `ks_position_sample.vendor_data.factset_symbology_xref`
# MAGIC    has curated `ticker_region → factset_entity_id` mappings
# MAGIC 2. **Name search** — search `sym_entity` by company name + `ENTITY_TYPE = 'PUB'`
# MAGIC 3. **fsym_id** — read directly from `sym_ticker_region` (no entity join needed)
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1 | Detect mismatches: compare resolved name vs display_name |
# MAGIC | 2 | Resolve correct entity_ids (crosswalk → name search) |
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
        dc.fsym_id             AS current_fsym_id,
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
    match = display.lower() in resolved.lower() or resolved.lower() in display.lower()
    flag = "OK" if match else "MISMATCH"
    print(f"{row['ticker']:<8} {display:<25} {resolved:<35} {row['resolved_type'] or 'N/A':<6} {flag}")
    if not match:
        mismatches.append(row)

print(f"\n{len(mismatches)} mismatch(es) found out of {len(rows)} companies")

if not mismatches:
    print("\nAll entity_id mappings are already correct — nothing to fix!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Resolve Correct Entity IDs
# MAGIC
# MAGIC **Strategy 1** — Crosswalk table (curated, most reliable)
# MAGIC **Strategy 2** — Name search in sym_entity (PUB entities only)

# COMMAND ----------

fixes = {}  # ticker -> {"entity_id": ..., "fsym_id": ..., "source": ...}

if mismatches:
    mismatch_tickers = {row["ticker"]: row for row in mismatches}
    mismatch_ticker_regions = [row["ticker_region"] for row in mismatches]
    mismatch_tr_sql = ", ".join(f"'{tr}'" for tr in mismatch_ticker_regions)

    # ── Strategy 1: Crosswalk table ──────────────────────────────────────
    print("Strategy 1: Crosswalk table (ks_position_sample.vendor_data.factset_symbology_xref)")
    print("-" * 80)

    try:
        xref_df = spark.sql(f"""
            SELECT
                xr.ticker_region,
                xr.factset_entity_id AS entity_id,
                se.ENTITY_PROPER_NAME AS entity_name,
                se.ENTITY_TYPE AS entity_type
            FROM ks_position_sample.vendor_data.factset_symbology_xref xr
            LEFT JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
                ON xr.factset_entity_id = se.FACTSET_ENTITY_ID
            WHERE xr.ticker_region IN ({mismatch_tr_sql})
        """)

        xref_rows = xref_df.collect()
        print(f"  Crosswalk returned {len(xref_rows)} row(s)")

        for xr in xref_rows:
            tr = xr["ticker_region"]
            ticker = tr.split("-")[0]
            mm = mismatch_tickers.get(ticker)
            if not mm:
                continue

            display = (mm["display_name"] or "").lower()
            ename = (xr["entity_name"] or "").lower()
            name_ok = display in ename or ename in display

            print(f"  {ticker:<8} → {xr['entity_id']:<15} {xr['entity_name'] or '(NULL)':<40} type={xr['entity_type'] or 'N/A':<5} name_match={name_ok}")

            if name_ok and ticker not in fixes:
                fixes[ticker] = {
                    "entity_id": xr["entity_id"],
                    "source": "crosswalk (name-matched)",
                }
            elif xr["entity_type"] == "PUB" and ticker not in fixes:
                fixes[ticker] = {
                    "entity_id": xr["entity_id"],
                    "source": "crosswalk (PUB)",
                }

        print(f"  Resolved via crosswalk: {len(fixes)}")

    except Exception as e:
        print(f"  Crosswalk table not available: {e}")

    # ── Strategy 2: Name search in sym_entity ────────────────────────────
    still_missing = [t for t in mismatch_tickers if t not in fixes]

    if still_missing:
        print(f"\nStrategy 2: Name search in sym_entity for {len(still_missing)} remaining tickers")
        print("-" * 80)

        for ticker in still_missing:
            mm = mismatch_tickers[ticker]
            display_name = mm["display_name"] or ""

            # Build search terms — try the full display_name first
            search_terms = [display_name]
            # Also try individual words > 3 chars (for multi-word names)
            words = [w for w in display_name.split() if len(w) > 3 and w not in ("Inc.", "Corp", "Corp.", "Ltd.", "Ltd", "Inc", "Co.", "Co", "Plc")]
            if words and words[0] != display_name:
                search_terms.append(words[0])

            found = False
            for term in search_terms:
                safe_term = term.replace("'", "''")
                candidates = spark.sql(f"""
                    SELECT
                        FACTSET_ENTITY_ID AS entity_id,
                        ENTITY_PROPER_NAME AS entity_name,
                        ENTITY_TYPE AS entity_type,
                        ISO_COUNTRY AS country
                    FROM delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity
                    WHERE ENTITY_PROPER_NAME LIKE '%{safe_term}%'
                      AND ENTITY_TYPE = 'PUB'
                    ORDER BY ENTITY_PROPER_NAME
                    LIMIT 20
                """).collect()

                if candidates:
                    print(f"\n  {ticker} — searching '{term}': {len(candidates)} PUB candidate(s)")
                    best = None
                    for c in candidates:
                        ename = (c["entity_name"] or "").lower()
                        # Prefer US entities and exact-ish name matches
                        is_us = c["country"] == "US"
                        name_close = display_name.lower() in ename
                        marker = ""
                        if name_close and is_us:
                            marker = " <<<< BEST"
                            if best is None:
                                best = c
                        elif name_close:
                            marker = " (name match)"
                            if best is None:
                                best = c
                        elif is_us:
                            marker = " (US)"
                        print(f"    {c['entity_id']:<15} {c['entity_name']:<45} {c['country'] or '??':<4} {marker}")

                    if best:
                        fixes[ticker] = {
                            "entity_id": best["entity_id"],
                            "source": f"name search ('{term}')",
                        }
                        print(f"    → SELECTED: {best['entity_id']} ({best['entity_name']})")
                        found = True
                        break

            if not found:
                print(f"\n  {ticker} — no suitable PUB entity found for '{display_name}'")

    # ── Also fix fsym_id from sym_ticker_region ──────────────────────────
    print(f"\nLooking up correct fsym_id values from sym_ticker_region...")

    fsym_df = spark.sql(f"""
        SELECT TICKER_REGION, FSYM_ID
        FROM delta_share_factset_do_not_delete_or_edit.sym_v1.sym_ticker_region
        WHERE TICKER_REGION IN ({mismatch_tr_sql})
    """)

    fsym_rows = fsym_df.dropDuplicates(["TICKER_REGION"]).collect()
    fsym_map = {row["TICKER_REGION"]: row["FSYM_ID"] for row in fsym_rows}
    print(f"  Found fsym_id for {len(fsym_map)} / {len(mismatch_ticker_regions)} ticker_regions")

    for ticker in fixes:
        mm = mismatch_tickers[ticker]
        tr = mm["ticker_region"]
        fixes[ticker]["fsym_id"] = fsym_map.get(tr)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(fixes)} fix(es) ready out of {len(mismatches)} mismatches")
    print(f"{'='*80}")
    for ticker, fix in sorted(fixes.items()):
        mm = mismatch_tickers[ticker]
        print(f"  {ticker:<8} {mm['current_entity_id']:<15} → {fix['entity_id']:<15}  fsym_id={fix.get('fsym_id', 'N/A')}  via {fix['source']}")

    unfixed = [t for t in mismatch_tickers if t not in fixes]
    if unfixed:
        print(f"\n  UNFIXED ({len(unfixed)}): {', '.join(sorted(unfixed))}")
        print("  These will need manual entity_id lookup.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Update demo_companies

# COMMAND ----------

if not fixes:
    print("No fixes to apply — skipping updates.")
else:
    update_count = 0
    for ticker, fix in fixes.items():
        new_entity_id = fix["entity_id"]
        new_fsym_id = fix.get("fsym_id")

        # Build SET clause
        set_parts = [f"entity_id = '{new_entity_id}'"]
        if new_fsym_id:
            set_parts.append(f"fsym_id = '{new_fsym_id}'")
        set_clause = ", ".join(set_parts)

        spark.sql(f"""
            UPDATE ks_factset_research_v3.gold.demo_companies
            SET {set_clause}
            WHERE ticker = '{ticker}'
        """)
        update_count += 1
        print(f"  Updated {ticker:<8} entity_id={new_entity_id}  fsym_id={new_fsym_id or '(unchanged)'}")

    print(f"\n{update_count} row(s) updated in demo_companies")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Rebuild company_profile

# COMMAND ----------

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
        se.ENTITY_PROPER_NAME  AS resolved_name,
        se.ENTITY_TYPE         AS resolved_type
    FROM ks_factset_research_v3.gold.demo_companies dc
    LEFT JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON dc.entity_id = se.FACTSET_ENTITY_ID
    WHERE dc.is_active = true
    ORDER BY dc.ticker
""")

val_rows = validation_df.collect()

print(f"\n{'Ticker':<8} {'Display Name':<25} {'Resolved Name':<35} {'Entity ID':<15} {'Type':<6} {'Match?'}")
print("-" * 100)

remaining_mismatches = 0
for row in val_rows:
    display = (row["display_name"] or "").lower()
    resolved = (row["resolved_name"] or "").lower()
    match = display in resolved or resolved in display
    flag = "OK" if match else "MISMATCH"
    if not match:
        remaining_mismatches += 1
    print(f"{row['ticker']:<8} {row['display_name'] or '':<25} {row['resolved_name'] or '(NULL)':<35} {row['entity_id'] or '':<15} {row['resolved_type'] or '':<6} {flag}")

print(f"\nRemaining mismatches: {remaining_mismatches}")

if remaining_mismatches > 0:
    print(f"\nWARNING: {remaining_mismatches} mismatches remain.")
    print("These may need manual entity_id lookup — see Step 2 output above for details.")
else:
    print("\nAll entity_id mappings are correct.")
