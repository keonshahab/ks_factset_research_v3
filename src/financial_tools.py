"""
Financial tools — structured queries against ks_factset_research_v3.gold.* tables.

Every public function returns a dict with three keys:
    result            – the computed answer (number, dict, or list)
    calculation_steps – human-readable list of steps taken
    sources           – list of table names queried

Usage (in a Databricks notebook):
    from src.financial_tools import get_financial_summary
    out = get_financial_summary(spark, "NVDA")
    print(out["result"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG = "ks_factset_research_v3"
SCHEMA = "gold"

PROFILE_TABLE = f"{CATALOG}.{SCHEMA}.company_profile"
FINANCIALS_TABLE = f"{CATALOG}.{SCHEMA}.company_financials"
ESTIMATES_TABLE = f"{CATALOG}.{SCHEMA}.consensus_estimates"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], suffix: str = "", is_ratio: bool = False) -> str:
    """Format a number for display.

    Rules:
        >=1 B  → $X.XB   (e.g. $35.1B)
        >=1 M  → $X.XM   (e.g. $120.5M)
        <1 M   → $X.XK or raw
        ratio  → X.Xx    (e.g. 2.3x)
    """
    if value is None:
        return "N/A"

    if is_ratio:
        return f"{value:,.1f}x"

    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if abs_val >= 1_000_000_000:
        return f"{sign}${abs_val / 1_000_000_000:,.1f}B{suffix}"
    if abs_val >= 1_000_000:
        return f"{sign}${abs_val / 1_000_000:,.1f}M{suffix}"
    if abs_val >= 1_000:
        return f"{sign}${abs_val / 1_000:,.1f}K{suffix}"
    return f"{sign}${abs_val:,.2f}{suffix}"


def _pct(value: Optional[float]) -> str:
    """Format a percentage."""
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


# ---------------------------------------------------------------------------
# Internal query helper
# ---------------------------------------------------------------------------

def _query_financials(spark, sql: str) -> List[Dict[str, Any]]:
    """Run *sql* via Spark and return a list of row-dicts."""
    rows = spark.sql(sql).collect()
    return [row.asDict() for row in rows]


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide two nullable floats; return None on null or zero denominator."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def get_company_profile(spark, ticker: str) -> Dict[str, Any]:
    """Return company profile metadata.

    Returns dict with keys: result, calculation_steps, sources.
    """
    sql = f"""
        SELECT *
        FROM {PROFILE_TABLE}
        WHERE ticker = '{ticker}'
        LIMIT 1
    """
    rows = _query_financials(spark, sql)
    profile = rows[0] if rows else {}

    return {
        "result": profile,
        "calculation_steps": [
            f"Queried {PROFILE_TABLE} for ticker={ticker}",
        ],
        "sources": [PROFILE_TABLE],
    }


def get_financial_summary(
    spark,
    ticker: str,
    period_type: str = "LTM",
) -> Dict[str, Any]:
    """Return the most recent financial snapshot for *ticker*.

    Parameters
    ----------
    period_type : str
        'LTM' (default), 'Q' (quarterly), or 'A' (annual).
    """
    sql = f"""
        SELECT *
        FROM {FINANCIALS_TABLE}
        WHERE ticker = '{ticker}'
          AND period_type = '{period_type}'
        ORDER BY period_date DESC
        LIMIT 1
    """
    rows = _query_financials(spark, sql)
    if not rows:
        return {
            "result": None,
            "calculation_steps": [f"No {period_type} data found for {ticker}"],
            "sources": [FINANCIALS_TABLE],
        }

    row = rows[0]
    summary = {
        "ticker": ticker,
        "period_date": str(row.get("period_date", "")),
        "period_type": period_type,
        "revenue": _fmt(row.get("revenue")),
        "operating_income": _fmt(row.get("operating_income")),
        "net_income": _fmt(row.get("net_income")),
        "eps": f"${row['eps']:.2f}" if row.get("eps") is not None else "N/A",
        "total_debt": _fmt(row.get("total_debt")),
        "total_assets": _fmt(row.get("total_assets")),
        "interest_expense": _fmt(row.get("interest_expense")),
        "operating_cash_flow": _fmt(row.get("operating_cash_flow")),
        "shareholders_equity": _fmt(row.get("shareholders_equity")),
        "currency": row.get("currency", "USD"),
    }

    return {
        "result": summary,
        "calculation_steps": [
            f"Queried {FINANCIALS_TABLE} for {ticker} period_type={period_type}",
            f"Selected most recent period: {summary['period_date']}",
            "Formatted all monetary values",
        ],
        "sources": [FINANCIALS_TABLE],
    }


def compare_periods(
    spark,
    ticker: str,
    period_type: str = "A",
    num_periods: int = 2,
) -> Dict[str, Any]:
    """Compare the most recent *num_periods* periods for *ticker*.

    Returns year-over-year or quarter-over-quarter deltas for key metrics.
    """
    sql = f"""
        SELECT *
        FROM {FINANCIALS_TABLE}
        WHERE ticker = '{ticker}'
          AND period_type = '{period_type}'
        ORDER BY period_date DESC
        LIMIT {num_periods}
    """
    rows = _query_financials(spark, sql)

    if len(rows) < 2:
        return {
            "result": None,
            "calculation_steps": [
                f"Queried {FINANCIALS_TABLE} for {ticker} period_type={period_type}",
                f"Found {len(rows)} period(s) — need at least 2 to compare",
            ],
            "sources": [FINANCIALS_TABLE],
        }

    current, prior = rows[0], rows[1]
    metrics = ["revenue", "operating_income", "net_income", "eps"]
    comparison: Dict[str, Any] = {
        "ticker": ticker,
        "current_period": str(current.get("period_date", "")),
        "prior_period": str(prior.get("period_date", "")),
        "changes": {},
    }

    steps = [
        f"Queried {FINANCIALS_TABLE} for {ticker} ({period_type}), last {num_periods} periods",
        f"Current: {comparison['current_period']}, Prior: {comparison['prior_period']}",
    ]

    for m in metrics:
        cur_val = current.get(m)
        pri_val = prior.get(m)
        change_pct = (
            ((cur_val - pri_val) / abs(pri_val)) * 100
            if cur_val is not None and pri_val is not None and pri_val != 0
            else None
        )
        comparison["changes"][m] = {
            "current": _fmt(cur_val),
            "prior": _fmt(pri_val),
            "change_pct": _pct(change_pct),
        }
        steps.append(f"  {m}: {_fmt(cur_val)} vs {_fmt(pri_val)} → {_pct(change_pct)}")

    return {
        "result": comparison,
        "calculation_steps": steps,
        "sources": [FINANCIALS_TABLE],
    }


# ---------------------------------------------------------------------------
# Leverage & debt tools
# ---------------------------------------------------------------------------

def calculate_leverage_ratio(spark, ticker: str) -> Dict[str, Any]:
    """Compute leverage ratios from the most recent quarterly or annual data.

    Returns debt-to-equity and debt-to-assets.
    """
    sql = f"""
        SELECT *
        FROM {FINANCIALS_TABLE}
        WHERE ticker = '{ticker}'
          AND period_type IN ('Q', 'A')
          AND total_debt IS NOT NULL
        ORDER BY period_date DESC
        LIMIT 1
    """
    rows = _query_financials(spark, sql)
    if not rows:
        return {
            "result": None,
            "calculation_steps": [f"No balance-sheet data found for {ticker}"],
            "sources": [FINANCIALS_TABLE],
        }

    row = rows[0]
    debt = row.get("total_debt")
    equity = row.get("shareholders_equity")
    assets = row.get("total_assets")

    d_e = _safe_div(debt, equity)
    d_a = _safe_div(debt, assets)

    steps = [
        f"Queried {FINANCIALS_TABLE} for {ticker}, most recent balance-sheet period",
        f"Period: {row.get('period_date')} ({row.get('period_type')})",
        f"Total debt:          {_fmt(debt)}",
        f"Shareholders equity: {_fmt(equity)}",
        f"Total assets:        {_fmt(assets)}",
        f"Debt-to-Equity = {_fmt(debt)} / {_fmt(equity)} = {_fmt(d_e, is_ratio=True)}",
        f"Debt-to-Assets = {_fmt(debt)} / {_fmt(assets)} = {_fmt(d_a, is_ratio=True)}",
    ]

    return {
        "result": {
            "ticker": ticker,
            "period_date": str(row.get("period_date", "")),
            "total_debt": _fmt(debt),
            "shareholders_equity": _fmt(equity),
            "total_assets": _fmt(assets),
            "debt_to_equity": _fmt(d_e, is_ratio=True),
            "debt_to_assets": _fmt(d_a, is_ratio=True),
            "debt_to_equity_raw": d_e,
            "debt_to_assets_raw": d_a,
        },
        "calculation_steps": steps,
        "sources": [FINANCIALS_TABLE],
    }


def calculate_debt_service_coverage(spark, ticker: str) -> Dict[str, Any]:
    """Compute the debt-service coverage ratio (DSCR).

    DSCR = operating_cash_flow / interest_expense   (using LTM if available).
    """
    # Prefer LTM, fall back to most recent annual
    sql = f"""
        SELECT *
        FROM {FINANCIALS_TABLE}
        WHERE ticker = '{ticker}'
          AND interest_expense IS NOT NULL
          AND operating_cash_flow IS NOT NULL
        ORDER BY
            CASE period_type WHEN 'LTM' THEN 1 WHEN 'A' THEN 2 ELSE 3 END,
            period_date DESC
        LIMIT 1
    """
    rows = _query_financials(spark, sql)
    if not rows:
        return {
            "result": None,
            "calculation_steps": [f"No cash-flow / interest data found for {ticker}"],
            "sources": [FINANCIALS_TABLE],
        }

    row = rows[0]
    ocf = row.get("operating_cash_flow")
    int_exp = row.get("interest_expense")
    dscr = _safe_div(ocf, int_exp)

    steps = [
        f"Queried {FINANCIALS_TABLE} for {ticker}",
        f"Period: {row.get('period_date')} ({row.get('period_type')})",
        f"Operating cash flow: {_fmt(ocf)}",
        f"Interest expense:    {_fmt(int_exp)}",
        f"DSCR = {_fmt(ocf)} / {_fmt(int_exp)} = {_fmt(dscr, is_ratio=True)}",
    ]

    return {
        "result": {
            "ticker": ticker,
            "period_date": str(row.get("period_date", "")),
            "operating_cash_flow": _fmt(ocf),
            "interest_expense": _fmt(int_exp),
            "dscr": _fmt(dscr, is_ratio=True),
            "dscr_raw": dscr,
        },
        "calculation_steps": steps,
        "sources": [FINANCIALS_TABLE],
    }


def check_covenant_compliance(
    spark,
    ticker: str,
    covenants: Dict[str, float],
) -> Dict[str, Any]:
    """Check current financials against covenant thresholds.

    Parameters
    ----------
    covenants : dict
        Mapping of ratio name to maximum allowed value.
        Supported keys: ``debt_to_equity``, ``debt_to_assets``, ``min_dscr``.

    Example::

        check_covenant_compliance(spark, "NVDA", {
            "debt_to_equity": 2.0,
            "debt_to_assets": 0.6,
            "min_dscr": 1.5,
        })
    """
    leverage = calculate_leverage_ratio(spark, ticker)
    dscr_out = calculate_debt_service_coverage(spark, ticker)

    results: Dict[str, Any] = {}
    steps = ["Computed leverage ratios and DSCR", "Checking against covenants:"]

    lev = leverage["result"] or {}
    dscr_res = dscr_out["result"] or {}

    for covenant, threshold in covenants.items():
        if covenant == "debt_to_equity":
            actual = lev.get("debt_to_equity_raw")
            compliant = actual is not None and actual <= threshold
        elif covenant == "debt_to_assets":
            actual = lev.get("debt_to_assets_raw")
            compliant = actual is not None and actual <= threshold
        elif covenant == "min_dscr":
            actual = dscr_res.get("dscr_raw")
            compliant = actual is not None and actual >= threshold
        else:
            actual = None
            compliant = None

        actual_str = _fmt(actual, is_ratio=True) if actual is not None else "N/A"
        status = "COMPLIANT" if compliant else ("BREACH" if compliant is False else "UNKNOWN")
        results[covenant] = {
            "threshold": threshold,
            "actual": actual_str,
            "actual_raw": actual,
            "status": status,
        }
        steps.append(f"  {covenant}: actual={actual_str}, threshold={threshold}, → {status}")

    all_compliant = all(
        v["status"] == "COMPLIANT" for v in results.values()
    )

    return {
        "result": {
            "ticker": ticker,
            "all_compliant": all_compliant,
            "covenants": results,
        },
        "calculation_steps": steps,
        "sources": [FINANCIALS_TABLE],
    }


# ---------------------------------------------------------------------------
# Estimates comparison
# ---------------------------------------------------------------------------

def compare_to_estimates(
    spark,
    ticker: str,
    metric_name: str = "EPS",
    num_periods: int = 4,
) -> Dict[str, Any]:
    """Compare actuals to consensus estimates for the last *num_periods* quarters.

    Parameters
    ----------
    metric_name : str
        FactSet estimate item — e.g. 'EPS', 'REVENUE', 'EBITDA'.
    """
    sql = f"""
        SELECT *
        FROM {ESTIMATES_TABLE}
        WHERE ticker = '{ticker}'
          AND metric_name = '{metric_name}'
          AND actual_value IS NOT NULL
        ORDER BY period_date DESC
        LIMIT {num_periods}
    """
    rows = _query_financials(spark, sql)
    if not rows:
        return {
            "result": None,
            "calculation_steps": [f"No {metric_name} estimate data found for {ticker}"],
            "sources": [ESTIMATES_TABLE],
        }

    periods: List[Dict[str, Any]] = []
    steps = [
        f"Queried {ESTIMATES_TABLE} for {ticker}, metric={metric_name}, last {num_periods} periods",
    ]

    for row in rows:
        actual = row.get("actual_value")
        consensus = row.get("consensus_mean")
        surprise = row.get("surprise")
        surprise_pct = row.get("surprise_pct")
        beat_miss = row.get("beat_miss", "N/A")

        period = {
            "period_date": str(row.get("period_date", "")),
            "actual": actual,
            "consensus_mean": consensus,
            "surprise": surprise,
            "surprise_pct": _pct(surprise_pct),
            "beat_miss": beat_miss,
            "num_estimates": row.get("num_estimates"),
            "guidance_low": row.get("guidance_low"),
            "guidance_high": row.get("guidance_high"),
        }
        periods.append(period)
        steps.append(
            f"  {period['period_date']}: actual={actual}, consensus={consensus}, "
            f"→ {beat_miss} ({_pct(surprise_pct)})"
        )

    beat_count = sum(1 for p in periods if p["beat_miss"] == "BEAT")
    steps.append(f"Beat rate: {beat_count}/{len(periods)} periods")

    return {
        "result": {
            "ticker": ticker,
            "metric_name": metric_name,
            "periods": periods,
            "beat_count": beat_count,
            "total_periods": len(periods),
        },
        "calculation_steps": steps,
        "sources": [ESTIMATES_TABLE],
    }


# ---------------------------------------------------------------------------
# Pro-forma leverage
# ---------------------------------------------------------------------------

def calculate_pro_forma_leverage(
    spark,
    ticker: str,
    additional_debt: float,
) -> Dict[str, Any]:
    """What-if: recalculate leverage if *additional_debt* (in millions) is added.

    Useful for modeling acquisition-financing scenarios.
    """
    leverage = calculate_leverage_ratio(spark, ticker)
    if leverage["result"] is None:
        return {
            "result": None,
            "calculation_steps": leverage["calculation_steps"]
            + [f"Cannot compute pro-forma — no balance-sheet data"],
            "sources": leverage["sources"],
        }

    lev = leverage["result"]

    # Re-query raw values for arithmetic
    sql = f"""
        SELECT total_debt, shareholders_equity, total_assets
        FROM {FINANCIALS_TABLE}
        WHERE ticker = '{ticker}'
          AND period_type IN ('Q', 'A')
          AND total_debt IS NOT NULL
        ORDER BY period_date DESC
        LIMIT 1
    """
    rows = _query_financials(spark, sql)
    row = rows[0]

    debt = row["total_debt"] or 0
    equity = row["shareholders_equity"]
    assets = row["total_assets"]

    # additional_debt is in millions — raw data is in millions too (FactSet ff_v3)
    new_debt = debt + additional_debt
    new_assets = (assets or 0) + additional_debt

    pf_d_e = _safe_div(new_debt, equity)
    pf_d_a = _safe_div(new_debt, new_assets)

    steps = [
        f"Current leverage from {lev.get('period_date', 'N/A')}:",
        f"  Debt-to-Equity: {lev['debt_to_equity']}",
        f"  Debt-to-Assets: {lev['debt_to_assets']}",
        f"Adding {_fmt(additional_debt)} of new debt:",
        f"  New total debt:   {_fmt(new_debt)}",
        f"  New total assets: {_fmt(new_assets)}",
        f"Pro-forma Debt-to-Equity = {_fmt(new_debt)} / {_fmt(equity)} = {_fmt(pf_d_e, is_ratio=True)}",
        f"Pro-forma Debt-to-Assets = {_fmt(new_debt)} / {_fmt(new_assets)} = {_fmt(pf_d_a, is_ratio=True)}",
    ]

    return {
        "result": {
            "ticker": ticker,
            "additional_debt": _fmt(additional_debt),
            "current_debt_to_equity": lev["debt_to_equity"],
            "current_debt_to_assets": lev["debt_to_assets"],
            "pro_forma_debt_to_equity": _fmt(pf_d_e, is_ratio=True),
            "pro_forma_debt_to_assets": _fmt(pf_d_a, is_ratio=True),
            "pro_forma_debt_to_equity_raw": pf_d_e,
            "pro_forma_debt_to_assets_raw": pf_d_a,
        },
        "calculation_steps": steps,
        "sources": [FINANCIALS_TABLE],
    }
