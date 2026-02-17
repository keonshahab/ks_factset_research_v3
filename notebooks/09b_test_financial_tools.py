# Databricks notebook source

# MAGIC %md
# MAGIC # 09b — Test Financial Tools
# MAGIC
# MAGIC **Purpose:** Validate every function in `src/financial_tools.py` against live
# MAGIC `ks_factset_research_v3.gold.*` tables using **NVDA** as the test ticker.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook `07_financial_tables` has been run (gold tables populated).

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

from src.financial_tools import (
    get_company_profile,
    get_financial_summary,
    compare_periods,
    calculate_leverage_ratio,
    calculate_debt_service_coverage,
    check_covenant_compliance,
    compare_to_estimates,
    calculate_pro_forma_leverage,
    _fmt,
    _pct,
)

TEST_TICKER = "NVDA"
print(f"Test ticker: {TEST_TICKER}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Formatting Helpers

# COMMAND ----------

print("=" * 70)
print("TEST: _fmt and _pct formatting")
print("=" * 70)

# Billions
assert _fmt(35_100_000_000) == "$35.1B", f"FAIL: got {_fmt(35_100_000_000)}"
# Millions
assert _fmt(120_500_000) == "$120.5M", f"FAIL: got {_fmt(120_500_000)}"
# Thousands
assert _fmt(5_500) == "$5.5K", f"FAIL: got {_fmt(5_500)}"
# Small
assert _fmt(42.50) == "$42.50", f"FAIL: got {_fmt(42.50)}"
# Negative
assert _fmt(-2_000_000_000) == "-$2.0B", f"FAIL: got {_fmt(-2_000_000_000)}"
# None
assert _fmt(None) == "N/A", f"FAIL: got {_fmt(None)}"
# Ratio
assert _fmt(2.34, is_ratio=True) == "2.3x", f"FAIL: got {_fmt(2.34, is_ratio=True)}"
# Percentage
assert _pct(4.6) == "+4.6%", f"FAIL: got {_pct(4.6)}"
assert _pct(-1.2) == "-1.2%", f"FAIL: got {_pct(-1.2)}"
assert _pct(None) == "N/A", f"FAIL: got {_pct(None)}"

print("All formatting tests passed.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: get_company_profile

# COMMAND ----------

print("=" * 70)
print("TEST: get_company_profile")
print("=" * 70)

profile_out = get_company_profile(spark, TEST_TICKER)

assert "result" in profile_out, "FAIL: missing 'result' key"
assert "calculation_steps" in profile_out, "FAIL: missing 'calculation_steps' key"
assert "sources" in profile_out, "FAIL: missing 'sources' key"

profile = profile_out["result"]
assert profile, f"FAIL: no profile found for {TEST_TICKER}"
assert profile.get("ticker") == TEST_TICKER, f"FAIL: expected ticker={TEST_TICKER}"

print(f"  ticker:       {profile.get('ticker')}")
print(f"  company_name: {profile.get('company_name')}")
print(f"  country:      {profile.get('country')}")
print(f"  entity_type:  {profile.get('entity_type')}")
print(f"  entity_id:    {profile.get('entity_id')}")
print(f"  Steps: {profile_out['calculation_steps']}")
print(f"  Sources: {profile_out['sources']}")

print("\nget_company_profile: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: get_financial_summary

# COMMAND ----------

print("=" * 70)
print("TEST: get_financial_summary (LTM)")
print("=" * 70)

summary_out = get_financial_summary(spark, TEST_TICKER, period_type="LTM")
assert summary_out["result"] is not None, f"FAIL: no LTM summary for {TEST_TICKER}"

summary = summary_out["result"]
assert summary["ticker"] == TEST_TICKER
assert summary["period_type"] == "LTM"

# Verify formatting — values should contain $ and B/M suffixes
assert "$" in summary["revenue"], f"FAIL: revenue not formatted: {summary['revenue']}"

print(f"  Period:             {summary['period_date']}")
print(f"  Revenue:            {summary['revenue']}")
print(f"  Operating income:   {summary['operating_income']}")
print(f"  Net income:         {summary['net_income']}")
print(f"  EPS:                {summary['eps']}")
print(f"  Interest expense:   {summary['interest_expense']}")
print(f"  Operating CF:       {summary['operating_cash_flow']}")
print(f"  Total debt:         {summary['total_debt']}")
print(f"  Total assets:       {summary['total_assets']}")
print(f"  Equity:             {summary['shareholders_equity']}")
print(f"  Currency:           {summary['currency']}")
print(f"  Steps: {summary_out['calculation_steps']}")

print("\nget_financial_summary (LTM): PASSED\n")

# COMMAND ----------

# Also test quarterly
print("TEST: get_financial_summary (Quarterly)")
q_out = get_financial_summary(spark, TEST_TICKER, period_type="Q")
assert q_out["result"] is not None, f"FAIL: no Q summary for {TEST_TICKER}"
q_summary = q_out["result"]
print(f"  Most recent Q: {q_summary['period_date']}  Revenue: {q_summary['revenue']}")
print("get_financial_summary (Q): PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: compare_periods

# COMMAND ----------

print("=" * 70)
print("TEST: compare_periods (Annual, 2 periods)")
print("=" * 70)

cmp_out = compare_periods(spark, TEST_TICKER, period_type="A", num_periods=2)
assert cmp_out["result"] is not None, f"FAIL: compare_periods returned None for {TEST_TICKER}"

cmp = cmp_out["result"]
assert cmp["ticker"] == TEST_TICKER
assert "changes" in cmp
assert "revenue" in cmp["changes"]

print(f"  Current period: {cmp['current_period']}")
print(f"  Prior period:   {cmp['prior_period']}")
for metric, vals in cmp["changes"].items():
    print(f"  {metric:20s}: {vals['current']:>12s} vs {vals['prior']:>12s}  ({vals['change_pct']})")
print(f"  Steps: {cmp_out['calculation_steps'][:3]}...")

print("\ncompare_periods: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: calculate_leverage_ratio

# COMMAND ----------

print("=" * 70)
print("TEST: calculate_leverage_ratio")
print("=" * 70)

lev_out = calculate_leverage_ratio(spark, TEST_TICKER)
assert lev_out["result"] is not None, f"FAIL: no leverage data for {TEST_TICKER}"

lev = lev_out["result"]
assert lev["ticker"] == TEST_TICKER
assert lev["debt_to_equity_raw"] is not None, "FAIL: debt_to_equity_raw is None"
assert lev["debt_to_assets_raw"] is not None, "FAIL: debt_to_assets_raw is None"
assert "x" in lev["debt_to_equity"], f"FAIL: ratio not formatted: {lev['debt_to_equity']}"

print(f"  Period:           {lev['period_date']}")
print(f"  Total debt:       {lev['total_debt']}")
print(f"  Equity:           {lev['shareholders_equity']}")
print(f"  Total assets:     {lev['total_assets']}")
print(f"  Debt-to-Equity:   {lev['debt_to_equity']}")
print(f"  Debt-to-Assets:   {lev['debt_to_assets']}")

for step in lev_out["calculation_steps"]:
    print(f"    {step}")

print("\ncalculate_leverage_ratio: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: calculate_debt_service_coverage

# COMMAND ----------

print("=" * 70)
print("TEST: calculate_debt_service_coverage")
print("=" * 70)

dscr_out = calculate_debt_service_coverage(spark, TEST_TICKER)
assert dscr_out["result"] is not None, f"FAIL: no DSCR data for {TEST_TICKER}"

dscr = dscr_out["result"]
assert dscr["ticker"] == TEST_TICKER

print(f"  Period:             {dscr['period_date']}")
print(f"  Operating CF:       {dscr['operating_cash_flow']}")
print(f"  Interest expense:   {dscr['interest_expense']}")
print(f"  DSCR:               {dscr['dscr']}")

if dscr["dscr_raw"] is None:
    print("  (interest expense is N/A — no debt service obligation)")
else:
    assert dscr["dscr_raw"] > 0, "FAIL: dscr_raw should be positive"

for step in dscr_out["calculation_steps"]:
    print(f"    {step}")

print("\ncalculate_debt_service_coverage: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7: check_covenant_compliance

# COMMAND ----------

print("=" * 70)
print("TEST: check_covenant_compliance")
print("=" * 70)

# Test leverage covenants (always available from balance sheet)
covenants = {
    "debt_to_equity": 5.0,
    "debt_to_assets": 0.9,
}

cov_out = check_covenant_compliance(spark, TEST_TICKER, covenants)
assert cov_out["result"] is not None, "FAIL: covenant check returned None"

cov = cov_out["result"]
assert cov["ticker"] == TEST_TICKER
assert "covenants" in cov

print(f"  All compliant: {cov['all_compliant']}")
for name, detail in cov["covenants"].items():
    print(f"  {name:20s}: actual={detail['actual']}, threshold={detail['threshold']}, status={detail['status']}")

for step in cov_out["calculation_steps"]:
    print(f"    {step}")

# Test with min_dscr — may be UNKNOWN if interest_expense is N/A
dscr_cov_out = check_covenant_compliance(spark, TEST_TICKER, {"min_dscr": 1.0})
dscr_status = dscr_cov_out["result"]["covenants"]["min_dscr"]["status"]
print(f"\n  min_dscr covenant: status={dscr_status}")
assert dscr_status in ("COMPLIANT", "UNKNOWN"), f"FAIL: unexpected status {dscr_status}"

# Now test with a tight threshold to force a breach
tight_covenants = {"debt_to_equity": 0.001}
tight_out = check_covenant_compliance(spark, TEST_TICKER, tight_covenants)
tight_cov = tight_out["result"]["covenants"]["debt_to_equity"]
assert tight_cov["status"] == "BREACH", f"FAIL: expected BREACH with threshold=0.001, got {tight_cov['status']}"
print(f"  Tight threshold test: D/E threshold=0.001 → {tight_cov['status']} (expected)")

print("\ncheck_covenant_compliance: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8: compare_to_estimates

# COMMAND ----------

print("=" * 70)
print("TEST: compare_to_estimates (EPS)")
print("=" * 70)

est_out = compare_to_estimates(spark, TEST_TICKER, metric_name="EPS", num_periods=4)
assert est_out["result"] is not None, f"FAIL: no EPS estimate data for {TEST_TICKER}"

est = est_out["result"]
assert est["ticker"] == TEST_TICKER
assert est["metric_name"] == "EPS"
assert len(est["periods"]) > 0, "FAIL: no periods returned"

print(f"  Metric: {est['metric_name']}")
print(f"  Periods: {est['total_periods']}")
print(f"  Beat count: {est['beat_count']}/{est['total_periods']}")

for p in est["periods"]:
    print(f"    {p['period_date']}: actual={p['actual']}, consensus={p['consensus_mean']}, "
          f"→ {p['beat_miss']} ({p['surprise_pct']})")

for step in est_out["calculation_steps"]:
    print(f"    {step}")

print("\ncompare_to_estimates: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9: calculate_pro_forma_leverage

# COMMAND ----------

print("=" * 70)
print("TEST: calculate_pro_forma_leverage")
print("=" * 70)

# Add $5B of hypothetical debt
additional = 5_000_000_000
pf_out = calculate_pro_forma_leverage(spark, TEST_TICKER, additional_debt=additional)
assert pf_out["result"] is not None, f"FAIL: pro-forma returned None for {TEST_TICKER}"

pf = pf_out["result"]
assert pf["ticker"] == TEST_TICKER
assert pf["pro_forma_debt_to_equity_raw"] is not None, "FAIL: pro-forma D/E is None"

# Pro-forma should be higher than current
lev_current = calculate_leverage_ratio(spark, TEST_TICKER)["result"]
assert pf["pro_forma_debt_to_equity_raw"] > lev_current["debt_to_equity_raw"], \
    "FAIL: pro-forma D/E should be higher than current after adding debt"

print(f"  Additional debt:       {pf['additional_debt']}")
print(f"  Current D/E:           {pf['current_debt_to_equity']}")
print(f"  Pro-forma D/E:         {pf['pro_forma_debt_to_equity']}")
print(f"  Current D/A:           {pf['current_debt_to_assets']}")
print(f"  Pro-forma D/A:         {pf['pro_forma_debt_to_assets']}")

for step in pf_out["calculation_steps"]:
    print(f"    {step}")

print("\ncalculate_pro_forma_leverage: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary

# COMMAND ----------

print("=" * 70)
print("ALL FINANCIAL TOOLS TESTS PASSED")
print("=" * 70)
print()
print(f"Test ticker: {TEST_TICKER}")
print()
print("Tests executed:")
print("  1. _fmt / _pct                  — formatting helpers ($X.XB, X.Xx, +X.X%)")
print("  2. get_company_profile          — profile metadata lookup")
print("  3. get_financial_summary        — LTM + quarterly snapshot")
print("  4. compare_periods              — annual YoY deltas")
print("  5. calculate_leverage_ratio     — D/E and D/A ratios")
print("  6. calculate_debt_service_coverage — DSCR from cash flow / interest")
print("  7. check_covenant_compliance    — pass + forced breach scenarios")
print("  8. compare_to_estimates         — EPS actual vs consensus beat/miss")
print("  9. calculate_pro_forma_leverage — what-if with $5B additional debt")
