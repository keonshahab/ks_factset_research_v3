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
# MAGIC 1. **Name search** — search `sym_entity` by full proper name + `PUB` + `US`
# MAGIC 2. **Crosswalk table** — fallback, strict validation (non-NULL + PUB + name match)
# MAGIC 3. **fsym_id** — read directly from `sym_ticker_region` (no entity join needed)
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1 | Detect mismatches: compare resolved name vs display_name |
# MAGIC | 2 | Resolve correct entity_ids (name search → crosswalk fallback) |
# MAGIC | 3 | Update demo_companies config table |
# MAGIC | 4 | Rebuild company_profile |
# MAGIC | 5 | Validate — confirm all names match |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Detect Mismatches

# COMMAND ----------

from pyspark.sql import functions as F

# ── Base company names for search and validation ────────────────────────
# No legal suffixes (Inc., Corp., Co., Ltd.) — keeps LIKE searches broad
# enough to match regardless of punctuation ("Visa Inc." vs "Visa, Inc.")
# and makes substring validation work for abbreviations ("amd" wouldn't
# match "Advanced Micro Devices" but "advanced micro devices" does).
PROPER_NAMES = {
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "AMD":  "Advanced Micro Devices",
    "AMZN": "Amazon",
    "BAC":  "Bank of America",
    "CAT":  "Caterpillar",
    "COST": "Costco Wholesale",
    "CRM":  "Salesforce",
    "CVX":  "Chevron",
    "GOOGL": "Alphabet",
    "INTC": "Intel",
    "JNJ":  "Johnson & Johnson",
    "JPM":  "JPMorgan Chase",
    "LLY":  "Eli Lilly",
    "MA":   "Mastercard",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "PG":   "Procter & Gamble",
    "TSLA": "Tesla",
    "V":    "Visa",
    "WMT":  "Walmart",
    "XOM":  "Exxon Mobil",
}

mismatch_df = spark.sql("""
    SELECT
        dc.ticker,
        dc.ticker_region,
        dc.display_name,
        dc.entity_id           AS current_entity_id,
        dc.fsym_id             AS current_fsym_id,
        se.ENTITY_PROPER_NAME  AS resolved_name,
        se.ENTITY_TYPE         AS resolved_type,
        se.ISO_COUNTRY         AS resolved_country
    FROM ks_factset_research_v3.gold.demo_companies dc
    LEFT JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON dc.entity_id = se.FACTSET_ENTITY_ID
    WHERE dc.is_active = true
    ORDER BY dc.ticker
""")

rows = mismatch_df.collect()

print(f"{'Ticker':<8} {'Display Name':<25} {'Resolved Name':<35} {'Country':<4} {'Type':<6} {'Match?'}")
print("-" * 110)

mismatches = []
for row in rows:
    display = row["display_name"] or ""
    resolved = row["resolved_name"]  # keep as None if NULL
    resolved_str = resolved or "(NULL)"
    entity_type = row["resolved_type"] or "N/A"
    country = row["resolved_country"] or ""

    # A mapping is correct only if the entity is a non-NULL, US-based PUB
    # company whose name matches the display name or proper name.
    if resolved and entity_type == "PUB" and country == "US":
        proper = PROPER_NAMES.get(row["ticker"], display).lower()
        res_low = resolved.lower()
        disp_low = display.lower()
        match = (disp_low in res_low or res_low in disp_low
                 or proper in res_low or res_low in proper)
    else:
        match = False  # NULL, non-PUB, or non-US = always a mismatch

    flag = "OK" if match else "MISMATCH"
    print(f"{row['ticker']:<8} {display:<25} {resolved_str:<35} {country or '??':<4} {entity_type:<6} {flag}")
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
# MAGIC **Strategy 1** — Name search in `sym_entity` (PUB + US) — most reliable for known US companies
# MAGIC **Strategy 2** — Crosswalk table (fallback, strict name validation required)

# COMMAND ----------

fixes = {}  # ticker -> {"entity_id": ..., "fsym_id": ..., "source": ...}

if mismatches:
    mismatch_tickers = {row["ticker"]: row for row in mismatches}
    mismatch_ticker_regions = [row["ticker_region"] for row in mismatches]
    mismatch_tr_sql = ", ".join(f"'{tr}'" for tr in mismatch_ticker_regions)

    # ── Strategy 1: Name search in sym_entity (PUB + US) ─────────────
    # This is the most reliable strategy for well-known US companies.
    # We search by proper name, filter to PUB entities in the US, and
    # pick the shortest name match (= most specific / exact).
    print("Strategy 1: Name search in sym_entity (PUB + US)")
    print("-" * 80)

    for ticker in sorted(mismatch_tickers):
        mm = mismatch_tickers[ticker]
        display_name = mm["display_name"] or ""
        search_name = PROPER_NAMES.get(ticker, display_name)
        safe_name = search_name.replace("'", "''")

        candidates = spark.sql(f"""
            SELECT
                FACTSET_ENTITY_ID AS entity_id,
                ENTITY_PROPER_NAME AS entity_name,
                ISO_COUNTRY AS country
            FROM delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity
            WHERE LOWER(ENTITY_PROPER_NAME) LIKE LOWER('%{safe_name}%')
              AND ENTITY_TYPE = 'PUB'
              AND ISO_COUNTRY = 'US'
            ORDER BY LENGTH(ENTITY_PROPER_NAME)
            LIMIT 5
        """).collect()

        if candidates:
            best = candidates[0]  # shortest name = most specific match
            fixes[ticker] = {
                "entity_id": best["entity_id"],
                "source": f"name search ('{search_name}')",
            }
            print(f"  {ticker:<8} -> {best['entity_id']:<15} {best['entity_name']}")
            for c in candidates[1:]:
                print(f"           (also: {c['entity_id']:<15} {c['entity_name']})")
        else:
            print(f"  {ticker:<8} -- no PUB+US match for '{search_name}'")

    print(f"\n  Resolved via name search: {len(fixes)} / {len(mismatch_tickers)}")

    # ── Strategy 2: Crosswalk fallback ───────────────────────────────
    # Only used for tickers that Strategy 1 missed.  Strict validation:
    # the crosswalk entity must resolve to a non-NULL PUB entity whose
    # name matches the display_name.
    still_missing = [t for t in mismatch_tickers if t not in fixes]

    if still_missing:
        print(f"\nStrategy 2: Crosswalk table for {len(still_missing)} remaining tickers")
        print("-" * 80)

        still_missing_trs = [mismatch_tickers[t]["ticker_region"] for t in still_missing]
        still_missing_sql = ", ".join(f"'{tr}'" for tr in still_missing_trs)

        try:
            xref_df = spark.sql(f"""
                SELECT
                    xr.ticker_region,
                    xr.factset_entity_id AS entity_id,
                    se.ENTITY_PROPER_NAME AS entity_name,
                    se.ENTITY_TYPE AS entity_type,
                    se.ISO_COUNTRY AS country
                FROM ks_position_sample.vendor_data.factset_symbology_xref xr
                LEFT JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
                    ON xr.factset_entity_id = se.FACTSET_ENTITY_ID
                WHERE xr.ticker_region IN ({still_missing_sql})
            """)

            for xr in xref_df.collect():
                tr = xr["ticker_region"]
                ticker = tr.split("-")[0]
                mm = mismatch_tickers.get(ticker)
                if not mm or ticker in fixes:
                    continue

                ename = (xr["entity_name"] or "").strip()
                display = (mm["display_name"] or "").lower()

                # Strict: name must be non-empty, PUB type, and match
                if not ename:
                    print(f"  {ticker:<8} -> {xr['entity_id']:<15} SKIP (no name in sym_entity)")
                    continue

                name_ok = display in ename.lower() or ename.lower() in display
                if xr["entity_type"] == "PUB" and name_ok:
                    fixes[ticker] = {
                        "entity_id": xr["entity_id"],
                        "source": f"crosswalk (verified: {ename})",
                    }
                    print(f"  {ticker:<8} -> {xr['entity_id']:<15} {ename:<40} OK")
                else:
                    print(f"  {ticker:<8} -> {xr['entity_id']:<15} {ename:<40} SKIP (type={xr['entity_type']}, match={name_ok})")

        except Exception as e:
            print(f"  Crosswalk table not available: {e}")

    # ── Also fix fsym_id from sym_ticker_region ──────────────────────
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

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(fixes)} fix(es) ready out of {len(mismatches)} mismatches")
    print(f"{'='*80}")
    for ticker, fix in sorted(fixes.items()):
        mm = mismatch_tickers[ticker]
        print(f"  {ticker:<8} {mm['current_entity_id']:<15} -> {fix['entity_id']:<15}  fsym_id={fix.get('fsym_id', 'N/A')}  via {fix['source']}")

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
        se.ENTITY_TYPE         AS resolved_type,
        se.ISO_COUNTRY         AS resolved_country
    FROM ks_factset_research_v3.gold.demo_companies dc
    LEFT JOIN delta_share_factset_do_not_delete_or_edit.sym_v1.sym_entity se
        ON dc.entity_id = se.FACTSET_ENTITY_ID
    WHERE dc.is_active = true
    ORDER BY dc.ticker
""")

val_rows = validation_df.collect()

print(f"\n{'Ticker':<8} {'Display Name':<25} {'Resolved Name':<35} {'Entity ID':<15} {'Country':<4} {'Type':<6} {'Match?'}")
print("-" * 115)

remaining_mismatches = 0
for row in val_rows:
    display = (row["display_name"] or "").lower()
    resolved = row["resolved_name"]
    entity_type = row["resolved_type"] or ""
    country = row["resolved_country"] or ""

    # Correct = non-NULL, US-based PUB entity whose name matches display/proper name
    if resolved and entity_type == "PUB" and country == "US":
        proper = PROPER_NAMES.get(row["ticker"], display).lower()
        res_low = resolved.lower()
        match = (display in res_low or res_low in display
                 or proper in res_low or res_low in proper)
    else:
        match = False

    flag = "OK" if match else "MISMATCH"
    if not match:
        remaining_mismatches += 1
    print(f"{row['ticker']:<8} {row['display_name'] or '':<25} {row['resolved_name'] or '(NULL)':<35} {row['entity_id'] or '':<15} {country or '??':<4} {row['resolved_type'] or '':<6} {flag}")

print(f"\nRemaining mismatches: {remaining_mismatches}")

if remaining_mismatches > 0:
    print(f"\nWARNING: {remaining_mismatches} mismatches remain.")
    print("These may need manual entity_id lookup — see Step 2 output above for details.")
else:
    print("\nAll entity_id mappings are correct.")
