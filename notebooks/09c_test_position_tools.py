# Databricks notebook source

# MAGIC %md
# MAGIC # 09c — Test Position Tools
# MAGIC
# MAGIC **Purpose:** Validate every function in `src/position_tools.py` using **NVDA**
# MAGIC (in position book) and **ZM** (not in position book) to confirm graceful handling.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook `03_demo_companies_config` has been run (demo_companies populated).
# MAGIC - The `ks_position_sample` catalog is accessible.
# MAGIC
# MAGIC **Architecture:**
# MAGIC - Position-book membership is checked via `ks_position_sample.vendor_data.factset_symbology_xref`
# MAGIC - Position/PnL/risk data tables live in `ks_factset_research_v3.gold` (seeded below)
# MAGIC
# MAGIC **Step 0** discovers schemas in `ks_position_sample`, **Step 1** seeds sample data,
# MAGIC then **Steps 2–8** test each function.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Setup & Schema Discovery

# COMMAND ----------

import sys, os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "/Workspace/Repos"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0a — Discover schemas & tables in ks_position_sample

# COMMAND ----------

print("=" * 70)
print("SCHEMA DISCOVERY: ks_position_sample")
print("=" * 70)

schemas = [row.databaseName for row in spark.sql("SHOW SCHEMAS IN ks_position_sample").collect()]
print(f"\nSchemas ({len(schemas)}): {schemas}\n")

for schema in schemas:
    tables = spark.sql(f"SHOW TABLES IN ks_position_sample.{schema}").collect()
    table_names = [row.tableName for row in tables]
    print(f"  ks_position_sample.{schema}  ({len(table_names)} tables):")
    for t in table_names:
        print(f"    - {t}")
    print()

# COMMAND ----------

# Confirm the crosswalk table we depend on
xref_count = spark.table("ks_position_sample.vendor_data.factset_symbology_xref").count()
print(f"Crosswalk xref rows: {xref_count}")
display(spark.table("ks_position_sample.vendor_data.factset_symbology_xref").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Seed Sample Position Data
# MAGIC
# MAGIC Creates three tables in `ks_factset_research_v3.gold`:
# MAGIC - `position_exposures` — positions by desk / asset class / book type
# MAGIC - `position_pnl` — daily P&L by desk
# MAGIC - `position_risk_flags` — compliance flags
# MAGIC
# MAGIC Seed tickers are drawn from the crosswalk (`ks_position_sample.vendor_data.factset_symbology_xref`).

# COMMAND ----------

CATALOG = "ks_factset_research_v3"
SCHEMA = "gold"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a — position_exposures

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.position_exposures
USING DELTA
AS SELECT * FROM VALUES
  -- NVDA-US: multi-desk, multi-asset exposure
  ('NVDA-US', 'Equity Trading',      'Equity',       'Trading',    150000000.0, 148500000.0, 50000,  'USD', 'Momentum',    DATE '2024-12-31'),
  ('NVDA-US', 'Equity Trading',      'Equity',       'Trading',     25000000.0,  24800000.0, 8500,   'USD', 'Index Arb',   DATE '2024-12-31'),
  ('NVDA-US', 'Equity Derivatives',  'Options',      'Derivatives', 80000000.0,  12000000.0, 2000,   'USD', 'Vol Surface', DATE '2024-12-31'),
  ('NVDA-US', 'Equity Derivatives',  'Options',      'Derivatives',-35000000.0,  -5200000.0, 1000,   'USD', 'Hedge',       DATE '2024-12-31'),
  ('NVDA-US', 'Credit Trading',      'Convertible',  'Trading',     40000000.0,  41200000.0, 400,    'USD', 'CB Arb',      DATE '2024-12-31'),
  ('NVDA-US', 'Prime Brokerage',     'Equity',       'Financing',   60000000.0,  59500000.0, 20000,  'USD', 'Stock Loan',  DATE '2024-12-31'),
  -- MSFT-US
  ('MSFT-US', 'Equity Trading',      'Equity',       'Trading',    200000000.0, 198000000.0, 100000, 'USD', 'Core Long',   DATE '2024-12-31'),
  ('MSFT-US', 'Equity Derivatives',  'Options',      'Derivatives', 50000000.0,   7500000.0, 1500,   'USD', 'Collar',      DATE '2024-12-31'),
  -- JPM-US
  ('JPM-US',  'Equity Trading',      'Equity',       'Trading',     90000000.0,  88000000.0, 45000,  'USD', 'Value',       DATE '2024-12-31'),
  ('JPM-US',  'Credit Trading',      'Corporate Bond','Trading',    120000000.0, 118500000.0, 1200,  'USD', 'IG Credit',   DATE '2024-12-31'),
  -- TSLA-US
  ('TSLA-US', 'Equity Trading',      'Equity',       'Trading',     70000000.0,  68000000.0, 30000,  'USD', 'Momentum',    DATE '2024-12-31'),
  ('TSLA-US', 'Equity Derivatives',  'Options',      'Derivatives',-20000000.0,  -3000000.0, 500,    'USD', 'Put Spread',  DATE '2024-12-31')
AS t(ticker_region, desk, asset_class, book_type, notional, market_value, quantity, currency, strategy, position_date)
""")

count = spark.table(f"{CATALOG}.{SCHEMA}.position_exposures").count()
print(f"position_exposures created: {count} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.position_exposures"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b — position_pnl

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.position_pnl
USING DELTA
AS SELECT * FROM VALUES
  -- NVDA-US daily P&L (Equity Trading)
  ('NVDA-US', 'Equity Trading',     DATE '2024-12-31',  2500000.0,  8500000.0,  45000000.0),
  ('NVDA-US', 'Equity Trading',     DATE '2024-12-30', -1200000.0,  6000000.0,  42500000.0),
  ('NVDA-US', 'Equity Trading',     DATE '2024-12-27',  3100000.0,  7200000.0,  43700000.0),
  ('NVDA-US', 'Equity Trading',     DATE '2024-12-26',  1800000.0,  4100000.0,  40600000.0),
  ('NVDA-US', 'Equity Trading',     DATE '2024-12-24',   900000.0,  2300000.0,  38800000.0),
  -- NVDA-US daily P&L (Equity Derivatives)
  ('NVDA-US', 'Equity Derivatives', DATE '2024-12-31',  1500000.0,  4200000.0,  18000000.0),
  ('NVDA-US', 'Equity Derivatives', DATE '2024-12-30',  -800000.0,  2700000.0,  16500000.0),
  ('NVDA-US', 'Equity Derivatives', DATE '2024-12-27',  2200000.0,  3500000.0,  17300000.0),
  ('NVDA-US', 'Equity Derivatives', DATE '2024-12-26',   600000.0,  1300000.0,  15100000.0),
  ('NVDA-US', 'Equity Derivatives', DATE '2024-12-24',  -200000.0,   700000.0,  14500000.0),
  -- NVDA-US daily P&L (Credit Trading)
  ('NVDA-US', 'Credit Trading',     DATE '2024-12-31',   300000.0,   900000.0,   5200000.0),
  ('NVDA-US', 'Credit Trading',     DATE '2024-12-30',   150000.0,   600000.0,   4900000.0),
  ('NVDA-US', 'Credit Trading',     DATE '2024-12-27',  -100000.0,   450000.0,   4750000.0),
  -- MSFT-US
  ('MSFT-US', 'Equity Trading',     DATE '2024-12-31',  1800000.0,  6000000.0,  32000000.0),
  ('MSFT-US', 'Equity Trading',     DATE '2024-12-30',   500000.0,  4200000.0,  30200000.0),
  -- JPM-US
  ('JPM-US',  'Equity Trading',     DATE '2024-12-31',  -400000.0,  1200000.0,  12000000.0),
  ('JPM-US',  'Credit Trading',     DATE '2024-12-31',   800000.0,  2500000.0,   9800000.0)
AS t(ticker_region, desk, pnl_date, daily_pnl, mtd_pnl, ytd_pnl)
""")

count = spark.table(f"{CATALOG}.{SCHEMA}.position_pnl").count()
print(f"position_pnl created: {count} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.position_pnl"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1c — position_risk_flags

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.position_risk_flags
USING DELTA
AS SELECT * FROM VALUES
  ('NVDA-US', false, false, false, false, DATE '2024-12-31', 'No active flags'),
  ('MSFT-US', false, false, false, false, DATE '2024-12-31', 'No active flags'),
  ('JPM-US',  true,  false, false, true,  DATE '2024-12-31', 'Volcker: market-making; Concentration: >5pct of desk book'),
  ('TSLA-US', false, true,  true,  false, DATE '2024-12-31', 'Restricted: pending M&A; MNPI: board-level info')
AS t(ticker_region, volcker_flag, restricted_flag, mnpi_flag, concentration_flag, flag_date, notes)
""")

count = spark.table(f"{CATALOG}.{SCHEMA}.position_risk_flags").count()
print(f"position_risk_flags created: {count} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.position_risk_flags"))

# COMMAND ----------

print("=" * 70)
print("SAMPLE DATA SEEDED SUCCESSFULLY")
print("=" * 70)
print(f"  {CATALOG}.{SCHEMA}.position_exposures   — positions by desk/asset/book")
print(f"  {CATALOG}.{SCHEMA}.position_pnl          — daily P&L by desk")
print(f"  {CATALOG}.{SCHEMA}.position_risk_flags   — compliance flags")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Imports & Config

# COMMAND ----------

from src.position_tools import (
    get_firm_exposure,
    get_desk_pnl,
    get_risk_flags,
    get_position_summary,
    get_desk_positions,
    _resolve_ticker_region,
    _check_in_position_book,
    _fmt,
)

IN_BOOK_TICKER = "NVDA"
NOT_IN_BOOK_TICKER = "ZM"

print(f"In-book ticker:     {IN_BOOK_TICKER}")
print(f"Not-in-book ticker: {NOT_IN_BOOK_TICKER}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Formatting Helper

# COMMAND ----------

print("=" * 70)
print("TEST: _fmt formatting")
print("=" * 70)

assert _fmt(35_100_000_000) == "$35.1B", f"FAIL: got {_fmt(35_100_000_000)}"
assert _fmt(120_500_000) == "$120.5M", f"FAIL: got {_fmt(120_500_000)}"
assert _fmt(5_500) == "$5.5K", f"FAIL: got {_fmt(5_500)}"
assert _fmt(42.50) == "$42.50", f"FAIL: got {_fmt(42.50)}"
assert _fmt(-2_000_000_000) == "-$2.0B", f"FAIL: got {_fmt(-2_000_000_000)}"
assert _fmt(None) == "N/A", f"FAIL: got {_fmt(None)}"
assert _fmt(2.34, is_ratio=True) == "2.3x", f"FAIL: got {_fmt(2.34, is_ratio=True)}"

print("All formatting tests passed.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Ticker Resolution

# COMMAND ----------

print("=" * 70)
print("TEST: _resolve_ticker_region")
print("=" * 70)

# In-book ticker should resolve
nvda_region = _resolve_ticker_region(spark, IN_BOOK_TICKER)
assert nvda_region is not None, f"FAIL: could not resolve {IN_BOOK_TICKER}"
assert nvda_region == "NVDA-US", f"FAIL: expected NVDA-US, got {nvda_region}"
print(f"  {IN_BOOK_TICKER} → {nvda_region}")

# Not-in-book ticker should resolve to None (or a region not in xref)
zm_region = _resolve_ticker_region(spark, NOT_IN_BOOK_TICKER)
print(f"  {NOT_IN_BOOK_TICKER} → {zm_region}")

# Verify xref membership
nvda_in_book = _check_in_position_book(spark, "NVDA-US")
assert nvda_in_book is True, "FAIL: NVDA-US should be in position book"
print(f"  NVDA-US in position book: {nvda_in_book}")

if zm_region:
    zm_in_book = _check_in_position_book(spark, zm_region)
    print(f"  {zm_region} in position book: {zm_in_book}")
else:
    print(f"  {NOT_IN_BOOK_TICKER} not resolved — confirmed not in book")

print("\n_resolve_ticker_region: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: get_firm_exposure

# COMMAND ----------

print("=" * 70)
print("TEST: get_firm_exposure — IN-BOOK ticker")
print("=" * 70)

exposure_out = get_firm_exposure(spark, IN_BOOK_TICKER, "2024-12-31")

assert "result" in exposure_out, "FAIL: missing 'result' key"
assert "calculation_steps" in exposure_out, "FAIL: missing 'calculation_steps' key"
assert "sources" in exposure_out, "FAIL: missing 'sources' key"

exp = exposure_out["result"]
assert exp["ticker"] == IN_BOOK_TICKER, f"FAIL: expected ticker={IN_BOOK_TICKER}"
assert exp["in_position_book"] is True, "FAIL: expected in_position_book=True"
assert exp["total_notional_raw"] is not None, "FAIL: total_notional_raw is None"
assert exp["position_count"] > 0, "FAIL: expected positions > 0"

print(f"  Ticker:          {exp['ticker']}")
print(f"  Ticker Region:   {exp['ticker_region']}")
print(f"  As-of Date:      {exp['as_of_date']}")
print(f"  Total Notional:  {exp['total_notional']}")
print(f"  Position Count:  {exp['position_count']}")

print(f"  By Desk ({len(exp['by_desk'])}):")
for d in exp["by_desk"]:
    print(f"    {d['desk']:30s}  notional={d['notional']:>12s}  positions={d['positions']}")

print(f"  By Asset Class ({len(exp['by_asset_class'])}):")
for a in exp["by_asset_class"]:
    print(f"    {a['asset_class']:30s}  notional={a['notional']:>12s}  positions={a['positions']}")

print(f"  By Book Type ({len(exp['by_book_type'])}):")
for b in exp["by_book_type"]:
    print(f"    {b['book_type']:30s}  notional={b['notional']:>12s}  positions={b['positions']}")

print(f"  Steps: {exposure_out['calculation_steps']}")
print(f"  Sources: {exposure_out['sources']}")

print("\nget_firm_exposure (in-book): PASSED\n")

# COMMAND ----------

print("=" * 70)
print("TEST: get_firm_exposure — NOT-IN-BOOK ticker")
print("=" * 70)

exposure_nib = get_firm_exposure(spark, NOT_IN_BOOK_TICKER, "2024-12-31")

nib = exposure_nib["result"]
assert nib["in_position_book"] is False, "FAIL: expected in_position_book=False"
assert "message" in nib, "FAIL: expected a 'message' key for not-in-book"
assert NOT_IN_BOOK_TICKER in nib["message"], "FAIL: message should mention the ticker"

print(f"  Ticker:           {nib['ticker']}")
print(f"  In Position Book: {nib['in_position_book']}")
print(f"  Message:          {nib['message']}")
print(f"  Steps: {exposure_nib['calculation_steps']}")

print("\nget_firm_exposure (not-in-book): PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: get_desk_pnl

# COMMAND ----------

print("=" * 70)
print("TEST: get_desk_pnl — IN-BOOK ticker")
print("=" * 70)

pnl_out = get_desk_pnl(spark, IN_BOOK_TICKER, days=30)

assert "result" in pnl_out
assert "calculation_steps" in pnl_out
assert "sources" in pnl_out

pnl = pnl_out["result"]
assert pnl["ticker"] == IN_BOOK_TICKER
assert pnl["in_position_book"] is True
assert len(pnl["desk_summary"]) > 0, "FAIL: expected at least one desk in summary"
assert len(pnl["daily_detail"]) > 0, "FAIL: expected daily detail rows"

print(f"  Ticker:          {pnl['ticker']}")
print(f"  Ticker Region:   {pnl['ticker_region']}")
print(f"  Days Requested:  {pnl['days_requested']}")

print(f"  Desk Summary ({len(pnl['desk_summary'])} desks):")
for ds in pnl["desk_summary"]:
    print(
        f"    {ds['desk']:30s}  total={ds['total_pnl']:>12s}  "
        f"best={ds['best_day']:>12s}  worst={ds['worst_day']:>12s}  "
        f"days={ds['trading_days']}"
    )

print(f"  Daily Detail ({len(pnl['daily_detail'])} rows, first 5):")
for dd in pnl["daily_detail"][:5]:
    print(f"    {dd['pnl_date']}  {dd['desk']:25s}  daily={dd['daily_pnl']:>12s}")

print(f"  Steps: {pnl_out['calculation_steps']}")

print("\nget_desk_pnl (in-book): PASSED\n")

# COMMAND ----------

print("=" * 70)
print("TEST: get_desk_pnl — NOT-IN-BOOK ticker")
print("=" * 70)

pnl_nib = get_desk_pnl(spark, NOT_IN_BOOK_TICKER, days=30)

pnl_n = pnl_nib["result"]
assert pnl_n["in_position_book"] is False, "FAIL: expected in_position_book=False"
assert "message" in pnl_n

print(f"  Ticker:           {pnl_n['ticker']}")
print(f"  In Position Book: {pnl_n['in_position_book']}")
print(f"  Message:          {pnl_n['message']}")

print("\nget_desk_pnl (not-in-book): PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7: get_risk_flags

# COMMAND ----------

print("=" * 70)
print("TEST: get_risk_flags — IN-BOOK ticker (NVDA — all clear)")
print("=" * 70)

risk_out = get_risk_flags(spark, IN_BOOK_TICKER)

assert "result" in risk_out
assert "calculation_steps" in risk_out
assert "sources" in risk_out

risk = risk_out["result"]
assert risk["ticker"] == IN_BOOK_TICKER
assert risk["in_position_book"] is True
assert "flags" in risk, "FAIL: missing 'flags' key"

flags = risk["flags"]
assert "volcker" in flags, "FAIL: missing 'volcker' flag"
assert "restricted" in flags, "FAIL: missing 'restricted' flag"
assert "mnpi" in flags, "FAIL: missing 'mnpi' flag"
assert "concentration" in flags, "FAIL: missing 'concentration' flag"

# NVDA should have no active flags per our seed data
assert risk["any_active"] is False, "FAIL: NVDA should have no active risk flags"

print(f"  Ticker:          {risk['ticker']}")
print(f"  Ticker Region:   {risk['ticker_region']}")
print(f"  Flag Date:       {risk.get('flag_date', 'N/A')}")
print(f"  Any Active:      {risk['any_active']}")

for flag_name, flag_val in flags.items():
    status = "ACTIVE" if flag_val else "clear"
    print(f"    {flag_name:20s}: {status}")

print(f"  Notes: {risk.get('notes', '')}")
print(f"  Steps: {risk_out['calculation_steps']}")

print("\nget_risk_flags (NVDA — all clear): PASSED\n")

# COMMAND ----------

# Test a ticker with active flags (JPM has Volcker + concentration)
print("=" * 70)
print("TEST: get_risk_flags — JPM (Volcker + concentration active)")
print("=" * 70)

risk_jpm = get_risk_flags(spark, "JPM")
risk_j = risk_jpm["result"]
assert risk_j["in_position_book"] is True
assert risk_j["any_active"] is True, "FAIL: JPM should have active flags"
assert risk_j["flags"]["volcker"] is True, "FAIL: JPM Volcker should be active"
assert risk_j["flags"]["concentration"] is True, "FAIL: JPM concentration should be active"
assert risk_j["flags"]["restricted"] is False, "FAIL: JPM restricted should be clear"

print(f"  Ticker:     {risk_j['ticker']}")
print(f"  Any Active: {risk_j['any_active']}")
for fn, fv in risk_j["flags"].items():
    print(f"    {fn:20s}: {'ACTIVE' if fv else 'clear'}")
print(f"  Notes: {risk_j['notes']}")

print("\nget_risk_flags (JPM — active flags): PASSED\n")

# COMMAND ----------

print("=" * 70)
print("TEST: get_risk_flags — NOT-IN-BOOK ticker")
print("=" * 70)

risk_nib = get_risk_flags(spark, NOT_IN_BOOK_TICKER)

risk_n = risk_nib["result"]
assert risk_n["in_position_book"] is False, "FAIL: expected in_position_book=False"
assert "message" in risk_n

print(f"  Ticker:           {risk_n['ticker']}")
print(f"  In Position Book: {risk_n['in_position_book']}")
print(f"  Message:          {risk_n['message']}")

print("\nget_risk_flags (not-in-book): PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8: get_position_summary

# COMMAND ----------

print("=" * 70)
print("TEST: get_position_summary — IN-BOOK ticker")
print("=" * 70)

summary_out = get_position_summary(spark, IN_BOOK_TICKER)

assert "result" in summary_out
assert "calculation_steps" in summary_out
assert "sources" in summary_out

summary = summary_out["result"]
assert summary["ticker"] == IN_BOOK_TICKER
assert summary["in_position_book"] is True
assert "exposure" in summary, "FAIL: missing 'exposure' sub-result"
assert "risk" in summary, "FAIL: missing 'risk' sub-result"
assert "top_positions" in summary, "FAIL: missing 'top_positions'"

print(f"  Ticker:          {summary['ticker']}")
print(f"  Ticker Region:   {summary['ticker_region']}")
print(f"  As-of Date:      {summary.get('as_of_date', 'N/A')}")

# Exposure sub-result
exp_s = summary["exposure"]
print(f"  Exposure:")
print(f"    Total Notional: {exp_s.get('total_notional', 'N/A')}")
print(f"    Position Count: {exp_s.get('position_count', 'N/A')}")
print(f"    Desks:          {len(exp_s.get('by_desk', []))}")

# Risk sub-result
risk_s = summary["risk"]
print(f"  Risk Flags:")
print(f"    Any Active:     {risk_s.get('any_active')}")
for fn, fv in risk_s.get("flags", {}).items():
    print(f"      {fn}: {'ACTIVE' if fv else 'clear'}")

# Top positions
top_pos = summary["top_positions"]
assert len(top_pos) > 0, "FAIL: expected at least one top position"
print(f"  Top Positions ({len(top_pos)}):")
for i, pos in enumerate(top_pos, 1):
    print(
        f"    {i}. desk={str(pos.get('desk', 'N/A')):25s}  "
        f"asset={str(pos.get('asset_class', 'N/A')):15s}  "
        f"notional={pos.get('notional', 'N/A'):>12s}  "
        f"book={pos.get('book_type', 'N/A')}"
    )

print(f"\n  Steps ({len(summary_out['calculation_steps'])} total, first 5):")
for step in summary_out["calculation_steps"][:5]:
    print(f"    {step}")

print(f"  Sources: {summary_out['sources']}")

print("\nget_position_summary (in-book): PASSED\n")

# COMMAND ----------

print("=" * 70)
print("TEST: get_position_summary — NOT-IN-BOOK ticker")
print("=" * 70)

summary_nib = get_position_summary(spark, NOT_IN_BOOK_TICKER)

sum_n = summary_nib["result"]
assert sum_n["in_position_book"] is False, "FAIL: expected in_position_book=False"
assert "message" in sum_n

print(f"  Ticker:           {sum_n['ticker']}")
print(f"  In Position Book: {sum_n['in_position_book']}")
print(f"  Message:          {sum_n['message']}")

print("\nget_position_summary (not-in-book): PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9: get_desk_positions

# COMMAND ----------

print("=" * 70)
print("TEST: get_desk_positions — IN-BOOK ticker")
print("=" * 70)

# Use the first desk from the exposure breakdown
available_desks = [d["desk"] for d in exposure_out["result"].get("by_desk", [])]
test_desk = available_desks[0] if available_desks else "Equity Trading"
print(f"  Available desks: {available_desks}")
print(f"  Testing desk:    {test_desk}")

desk_out = get_desk_positions(spark, IN_BOOK_TICKER, test_desk)

assert "result" in desk_out
assert "calculation_steps" in desk_out
assert "sources" in desk_out

desk = desk_out["result"]
assert desk["ticker"] == IN_BOOK_TICKER
assert desk["in_position_book"] is True
assert desk["position_count"] > 0, "FAIL: expected positions > 0 for the desk"

print(f"  Ticker:          {desk['ticker']}")
print(f"  Ticker Region:   {desk['ticker_region']}")
print(f"  Desk:            {desk['desk']}")
print(f"  As-of Date:      {desk['as_of_date']}")
print(f"  Total Notional:  {desk['total_notional']}")
print(f"  Total MV:        {desk['total_market_value']}")
print(f"  Position Count:  {desk['position_count']}")

print(f"  Positions:")
for i, pos in enumerate(desk["positions"], 1):
    print(
        f"    {i}. asset={str(pos.get('asset_class', 'N/A')):15s}  "
        f"book={str(pos.get('book_type', 'N/A')):12s}  "
        f"notional={pos.get('notional', 'N/A'):>12s}  "
        f"mv={pos.get('market_value', 'N/A'):>12s}  "
        f"strategy={pos.get('strategy', '')}"
    )

print(f"  Steps: {desk_out['calculation_steps']}")

print("\nget_desk_positions (in-book): PASSED\n")

# COMMAND ----------

print("=" * 70)
print("TEST: get_desk_positions — NOT-IN-BOOK ticker")
print("=" * 70)

desk_nib = get_desk_positions(spark, NOT_IN_BOOK_TICKER, "Equity Trading")

desk_n = desk_nib["result"]
assert desk_n["in_position_book"] is False, "FAIL: expected in_position_book=False"
assert "message" in desk_n

print(f"  Ticker:           {desk_n['ticker']}")
print(f"  In Position Book: {desk_n['in_position_book']}")
print(f"  Message:          {desk_n['message']}")

print("\nget_desk_positions (not-in-book): PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary

# COMMAND ----------

print("=" * 70)
print("ALL POSITION TOOLS TESTS PASSED")
print("=" * 70)
print()
print(f"In-book ticker:     {IN_BOOK_TICKER}")
print(f"Not-in-book ticker: {NOT_IN_BOOK_TICKER}")
print()
print("Tests executed:")
print("  0. Schema discovery             — SHOW SCHEMAS/TABLES in ks_position_sample")
print("  1. Seed sample data             — position_exposures, position_pnl, position_risk_flags")
print("  2. _fmt                         — formatting helper ($X.XB, X.Xx)")
print("  3. _resolve_ticker_region       — ticker → ticker_region resolution")
print("  4. get_firm_exposure            — total notional + desk/asset/book breakdown")
print("  5. get_desk_pnl                 — daily P&L by desk (last 30 days)")
print("  6. get_risk_flags               — Volcker, restricted, MNPI, concentration")
print("  7. get_position_summary         — combined exposure + risk + top 10 positions")
print("  8. get_desk_positions           — full detail for one desk")
print()
print("Graceful not-in-book handling:")
print("  All 5 public functions return in_position_book=False with a clear message")
print("  when the ticker is not found in ks_position_sample.vendor_data.factset_symbology_xref")
print()
print("Cross-catalog architecture:")
print("  Position book membership → ks_position_sample.vendor_data.factset_symbology_xref")
print("  Position data tables     → ks_factset_research_v3.gold.position_*")
