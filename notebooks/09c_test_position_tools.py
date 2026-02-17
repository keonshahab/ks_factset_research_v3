# Databricks notebook source

# MAGIC %md
# MAGIC # 09c — Test Position Tools
# MAGIC
# MAGIC **Purpose:** Validate every function in `src/position_tools.py` against live
# MAGIC `ks_position_sample.*` tables using **NVDA** (in position book) and **ZM**
# MAGIC (not in position book) to confirm graceful handling.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook `01.5_company_selection` has been run (demo_companies populated).
# MAGIC - The `ks_position_sample` catalog is accessible.
# MAGIC
# MAGIC **Cross-catalog tables queried:**
# MAGIC | Table | Catalog |
# MAGIC |---|---|
# MAGIC | `vendor_data.factset_symbology_xref` | `ks_position_sample` |
# MAGIC | `positions.current_positions` | `ks_position_sample` |
# MAGIC | `positions.daily_pnl` | `ks_position_sample` |
# MAGIC | `risk.compliance_flags` | `ks_position_sample` |
# MAGIC | `gold.demo_companies` | `ks_factset_research_v3` |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Setup

# COMMAND ----------

import sys, os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "/Workspace/Repos"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
# MAGIC ## Step 1: Formatting Helper

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
# MAGIC ## Step 2: Ticker Resolution

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
# MAGIC ## Step 3: get_firm_exposure

# COMMAND ----------

print("=" * 70)
print("TEST: get_firm_exposure — IN-BOOK ticker")
print("=" * 70)

# Use a recent date; the function will query positions for that date
exposure_out = get_firm_exposure(spark, IN_BOOK_TICKER, "2024-12-31")

assert "result" in exposure_out, "FAIL: missing 'result' key"
assert "calculation_steps" in exposure_out, "FAIL: missing 'calculation_steps' key"
assert "sources" in exposure_out, "FAIL: missing 'sources' key"

exp = exposure_out["result"]
assert exp["ticker"] == IN_BOOK_TICKER, f"FAIL: expected ticker={IN_BOOK_TICKER}"
assert exp["in_position_book"] is True, "FAIL: expected in_position_book=True"

print(f"  Ticker:          {exp['ticker']}")
print(f"  Ticker Region:   {exp['ticker_region']}")
print(f"  As-of Date:      {exp.get('as_of_date')}")
print(f"  Total Notional:  {exp.get('total_notional')}")
print(f"  Position Count:  {exp.get('position_count')}")
print(f"  By Desk:         {len(exp.get('by_desk', []))} desks")
for d in exp.get("by_desk", []):
    print(f"    {d['desk']:30s}  notional={d['notional']:>12s}  positions={d['positions']}")
print(f"  By Asset Class:  {len(exp.get('by_asset_class', []))} classes")
for a in exp.get("by_asset_class", []):
    print(f"    {a['asset_class']:30s}  notional={a['notional']:>12s}  positions={a['positions']}")
print(f"  By Book Type:    {len(exp.get('by_book_type', []))} types")
for b in exp.get("by_book_type", []):
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
# MAGIC ## Step 4: get_desk_pnl

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

print(f"  Ticker:          {pnl['ticker']}")
print(f"  Ticker Region:   {pnl['ticker_region']}")
print(f"  Days Requested:  {pnl['days_requested']}")

if pnl.get("desk_summary"):
    print(f"  Desk Summary:    {len(pnl['desk_summary'])} desks")
    for ds in pnl["desk_summary"]:
        print(
            f"    {ds['desk']:30s}  total={ds['total_pnl']:>12s}  "
            f"best={ds['best_day']:>12s}  worst={ds['worst_day']:>12s}  "
            f"days={ds['trading_days']}"
        )
    print(f"  Daily Detail:    {len(pnl.get('daily_detail', []))} rows (first 5):")
    for dd in pnl.get("daily_detail", [])[:5]:
        print(f"    {dd['pnl_date']}  {dd['desk']:20s}  daily={dd['daily_pnl']:>12s}")
else:
    print(f"  Message: {pnl.get('message', 'No P&L data')}")

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
# MAGIC ## Step 5: get_risk_flags

# COMMAND ----------

print("=" * 70)
print("TEST: get_risk_flags — IN-BOOK ticker")
print("=" * 70)

risk_out = get_risk_flags(spark, IN_BOOK_TICKER)

assert "result" in risk_out
assert "calculation_steps" in risk_out
assert "sources" in risk_out

risk = risk_out["result"]
assert risk["ticker"] == IN_BOOK_TICKER
assert risk["in_position_book"] is True
assert "flags" in risk, "FAIL: missing 'flags' key"

print(f"  Ticker:          {risk['ticker']}")
print(f"  Ticker Region:   {risk['ticker_region']}")
print(f"  Flag Date:       {risk.get('flag_date', 'N/A')}")
print(f"  Any Active:      {risk.get('any_active')}")

flags = risk["flags"]
assert "volcker" in flags, "FAIL: missing 'volcker' flag"
assert "restricted" in flags, "FAIL: missing 'restricted' flag"
assert "mnpi" in flags, "FAIL: missing 'mnpi' flag"
assert "concentration" in flags, "FAIL: missing 'concentration' flag"

for flag_name, flag_val in flags.items():
    status = "ACTIVE" if flag_val else "clear"
    print(f"    {flag_name:20s}: {status}")

print(f"  Notes: {risk.get('notes', '')}")
print(f"  Steps: {risk_out['calculation_steps']}")

print("\nget_risk_flags (in-book): PASSED\n")

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
# MAGIC ## Step 6: get_position_summary

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

print(f"  Ticker:          {summary['ticker']}")
print(f"  Ticker Region:   {summary['ticker_region']}")
print(f"  As-of Date:      {summary.get('as_of_date', 'N/A')}")

# Exposure sub-result
if "exposure" in summary and summary["exposure"]:
    exp_s = summary["exposure"]
    print(f"  Exposure:")
    print(f"    Total Notional: {exp_s.get('total_notional', 'N/A')}")
    print(f"    Position Count: {exp_s.get('position_count', 'N/A')}")
    print(f"    Desks:          {len(exp_s.get('by_desk', []))}")

# Risk sub-result
if "risk" in summary and summary["risk"]:
    risk_s = summary["risk"]
    print(f"  Risk Flags:")
    print(f"    Any Active:     {risk_s.get('any_active')}")
    for fn, fv in risk_s.get("flags", {}).items():
        print(f"      {fn}: {'ACTIVE' if fv else 'clear'}")

# Top positions
top_pos = summary.get("top_positions", [])
print(f"  Top Positions:   {len(top_pos)} returned")
for i, pos in enumerate(top_pos[:5], 1):
    print(
        f"    {i}. desk={pos.get('desk', 'N/A'):20s}  "
        f"asset={pos.get('asset_class', 'N/A'):15s}  "
        f"notional={pos.get('notional', 'N/A'):>12s}  "
        f"book={pos.get('book_type', 'N/A')}"
    )
if len(top_pos) > 5:
    print(f"    ... and {len(top_pos) - 5} more")

print(f"  Steps ({len(summary_out['calculation_steps'])} total): showing first 5")
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
# MAGIC ## Step 7: get_desk_positions

# COMMAND ----------

print("=" * 70)
print("TEST: get_desk_positions — IN-BOOK ticker")
print("=" * 70)

# First, discover available desks from the exposure result
available_desks = [d["desk"] for d in exposure_out["result"].get("by_desk", [])]
if available_desks:
    test_desk = available_desks[0]
    print(f"  Available desks: {available_desks}")
    print(f"  Testing desk:    {test_desk}")
else:
    test_desk = "Equity Trading"
    print(f"  No desks discovered from exposure — using default: {test_desk}")

desk_out = get_desk_positions(spark, IN_BOOK_TICKER, test_desk)

assert "result" in desk_out
assert "calculation_steps" in desk_out
assert "sources" in desk_out

desk = desk_out["result"]
assert desk["ticker"] == IN_BOOK_TICKER
assert desk["in_position_book"] is True

print(f"  Ticker:          {desk['ticker']}")
print(f"  Ticker Region:   {desk['ticker_region']}")
print(f"  Desk:            {desk.get('desk')}")
print(f"  As-of Date:      {desk.get('as_of_date', 'N/A')}")
print(f"  Total Notional:  {desk.get('total_notional', 'N/A')}")
print(f"  Total MV:        {desk.get('total_market_value', 'N/A')}")
print(f"  Position Count:  {desk.get('position_count', 0)}")

for i, pos in enumerate(desk.get("positions", [])[:5], 1):
    print(
        f"    {i}. asset={pos.get('asset_class', 'N/A'):15s}  "
        f"book={pos.get('book_type', 'N/A'):10s}  "
        f"notional={pos.get('notional', 'N/A'):>12s}  "
        f"mv={pos.get('market_value', 'N/A'):>12s}"
    )

remaining = desk.get("position_count", 0) - 5
if remaining > 0:
    print(f"    ... and {remaining} more positions")

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
print("  1. _fmt                         — formatting helper ($X.XB, X.Xx)")
print("  2. _resolve_ticker_region       — ticker → ticker_region resolution")
print("  3. get_firm_exposure            — total notional + desk/asset/book breakdown")
print("  4. get_desk_pnl                 — daily P&L by desk (last 30 days)")
print("  5. get_risk_flags               — Volcker, restricted, MNPI, concentration")
print("  6. get_position_summary         — combined exposure + risk + top 10 positions")
print("  7. get_desk_positions           — full detail for one desk")
print()
print("Graceful not-in-book handling:")
print("  All 5 public functions return in_position_book=False with a clear message")
print("  when the ticker is not found in ks_position_sample.vendor_data.factset_symbology_xref")
