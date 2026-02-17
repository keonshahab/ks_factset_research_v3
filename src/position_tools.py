"""
Position tools — cross-catalog queries for position book data.

Every public function returns a dict with three keys:
    result            – the computed answer (number, dict, or list)
    calculation_steps – human-readable list of steps taken
    sources           – list of table names queried

Ticker resolution: accepts a short ticker (e.g. "NVDA") and resolves it to
ticker_region (e.g. "NVDA-US") via the demo_companies config table, then
checks the position crosswalk (ks_position_sample) to confirm the ticker is
in the book.

Position data tables live in ks_factset_research_v3.gold and are seeded by
the test notebook (09c).  The crosswalk lives in ks_position_sample.

Usage (in a Databricks notebook):
    from src.position_tools import get_firm_exposure
    out = get_firm_exposure(spark, "NVDA", "2024-12-31")
    print(out["result"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_CATALOG = "ks_position_sample"
RESEARCH_CATALOG = "ks_factset_research_v3"
SCHEMA = "gold"

# Cross-catalog: position-book membership (ks_position_sample)
XREF_TABLE = f"{POSITION_CATALOG}.vendor_data.factset_symbology_xref"

# Position data tables (ks_factset_research_v3.gold)
POSITIONS_TABLE = f"{RESEARCH_CATALOG}.{SCHEMA}.position_exposures"
PNL_TABLE = f"{RESEARCH_CATALOG}.{SCHEMA}.position_pnl"
RISK_TABLE = f"{RESEARCH_CATALOG}.{SCHEMA}.position_risk_flags"

# Lookup
DEMO_COMPANIES = f"{RESEARCH_CATALOG}.{SCHEMA}.demo_companies"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query(spark, sql: str) -> List[Dict[str, Any]]:
    """Run *sql* via Spark and return a list of row-dicts."""
    rows = spark.sql(sql).collect()
    return [row.asDict() for row in rows]


def _safe_query(spark, sql: str) -> List[Dict[str, Any]]:
    """Like _query but returns [] if the table does not exist."""
    try:
        return _query(spark, sql)
    except Exception as exc:
        if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
            return []
        raise


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


def _resolve_ticker_region(spark, ticker: str) -> Optional[str]:
    """Resolve a short ticker (e.g. 'NVDA') to ticker_region (e.g. 'NVDA-US').

    Uses the demo_companies config table first, falls back to the crosswalk
    xref (matching on the ``TICKER-%`` pattern).
    Returns None if the ticker is not found anywhere.
    """
    # Try demo_companies first (fast, small table)
    rows = _safe_query(spark, f"""
        SELECT ticker_region
        FROM {DEMO_COMPANIES}
        WHERE ticker = '{ticker}'
        LIMIT 1
    """)
    if rows:
        return rows[0]["ticker_region"]

    # Fallback: check xref table (ticker_region format is "TICKER-REGION")
    rows = _safe_query(spark, f"""
        SELECT ticker_region
        FROM {XREF_TABLE}
        WHERE ticker_region LIKE '{ticker}-%'
        LIMIT 1
    """)
    if rows:
        return rows[0]["ticker_region"]

    return None


def _check_in_position_book(spark, ticker_region: str) -> bool:
    """Check if a ticker_region exists in the position crosswalk."""
    rows = _safe_query(spark, f"""
        SELECT 1
        FROM {XREF_TABLE}
        WHERE ticker_region = '{ticker_region}'
        LIMIT 1
    """)
    return len(rows) > 0


def _not_in_book(ticker: str, ticker_region: Optional[str], steps: List[str]) -> Dict[str, Any]:
    """Standard return payload when a ticker is not in the position book."""
    tr_display = ticker_region or "N/A"
    detail = (
        f"ticker_region '{ticker_region}' not found in {XREF_TABLE}"
        if ticker_region
        else f"Ticker '{ticker}' could not be resolved to a ticker_region"
    )
    return {
        "result": {
            "ticker": ticker,
            "ticker_region": tr_display,
            "in_position_book": False,
            "message": f"{ticker} is not in the position book",
        },
        "calculation_steps": steps + [detail],
        "sources": [DEMO_COMPANIES, XREF_TABLE],
    }


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def get_firm_exposure(
    spark,
    ticker: str,
    as_of_date: str,
) -> Dict[str, Any]:
    """Return total notional exposure and breakdown by desk, asset_class, and book_type.

    Parameters
    ----------
    ticker : str
        Short ticker (e.g. 'NVDA').
    as_of_date : str
        Position date in 'YYYY-MM-DD' format.

    Returns dict with keys: result, calculation_steps, sources.
    """
    ticker_region = _resolve_ticker_region(spark, ticker)
    steps = [f"Resolved ticker '{ticker}' → ticker_region '{ticker_region}'"]

    if ticker_region is None or not _check_in_position_book(spark, ticker_region):
        return _not_in_book(ticker, ticker_region, steps)

    steps.append(f"Confirmed {ticker_region} is in position book")

    # Total notional
    total_rows = _safe_query(spark, f"""
        SELECT SUM(notional) AS total_notional,
               COUNT(*) AS position_count
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND position_date = '{as_of_date}'
    """)
    total = total_rows[0] if total_rows else {}
    total_notional = total.get("total_notional")
    position_count = total.get("position_count", 0)

    steps.append(f"Queried {POSITIONS_TABLE} for {ticker_region} as of {as_of_date}")
    steps.append(f"Total notional: {_fmt(total_notional)}, positions: {position_count}")

    # Breakdown by desk
    desk_rows = _safe_query(spark, f"""
        SELECT desk,
               SUM(notional) AS notional,
               COUNT(*) AS positions
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND position_date = '{as_of_date}'
        GROUP BY desk
        ORDER BY 2 DESC
    """)

    # Breakdown by asset_class
    ac_rows = _safe_query(spark, f"""
        SELECT asset_class,
               SUM(notional) AS notional,
               COUNT(*) AS positions
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND position_date = '{as_of_date}'
        GROUP BY asset_class
        ORDER BY 2 DESC
    """)

    # Breakdown by book_type
    bt_rows = _safe_query(spark, f"""
        SELECT book_type,
               SUM(notional) AS notional,
               COUNT(*) AS positions
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND position_date = '{as_of_date}'
        GROUP BY book_type
        ORDER BY 2 DESC
    """)

    steps.append(
        f"Computed breakdowns: {len(desk_rows)} desks, "
        f"{len(ac_rows)} asset classes, {len(bt_rows)} book types"
    )

    return {
        "result": {
            "ticker": ticker,
            "ticker_region": ticker_region,
            "in_position_book": True,
            "as_of_date": as_of_date,
            "total_notional": _fmt(total_notional),
            "total_notional_raw": total_notional,
            "position_count": position_count,
            "by_desk": [
                {
                    "desk": r["desk"],
                    "notional": _fmt(r["notional"]),
                    "notional_raw": r["notional"],
                    "positions": r["positions"],
                }
                for r in desk_rows
            ],
            "by_asset_class": [
                {
                    "asset_class": r["asset_class"],
                    "notional": _fmt(r["notional"]),
                    "notional_raw": r["notional"],
                    "positions": r["positions"],
                }
                for r in ac_rows
            ],
            "by_book_type": [
                {
                    "book_type": r["book_type"],
                    "notional": _fmt(r["notional"]),
                    "notional_raw": r["notional"],
                    "positions": r["positions"],
                }
                for r in bt_rows
            ],
        },
        "calculation_steps": steps,
        "sources": [POSITIONS_TABLE, XREF_TABLE, DEMO_COMPANIES],
    }


def get_desk_pnl(
    spark,
    ticker: str,
    days: int = 30,
) -> Dict[str, Any]:
    """Return daily P&L by desk for the last *days* trading days.

    Parameters
    ----------
    ticker : str
        Short ticker (e.g. 'NVDA').
    days : int
        Number of most recent trading days to return (default 30).

    Returns dict with keys: result, calculation_steps, sources.
    """
    ticker_region = _resolve_ticker_region(spark, ticker)
    steps = [f"Resolved ticker '{ticker}' → ticker_region '{ticker_region}'"]

    if ticker_region is None or not _check_in_position_book(spark, ticker_region):
        return _not_in_book(ticker, ticker_region, steps)

    steps.append(f"Confirmed {ticker_region} is in position book")

    # Daily detail (last N days relative to most recent date in the table)
    rows = _safe_query(spark, f"""
        SELECT desk,
               pnl_date,
               daily_pnl,
               mtd_pnl,
               ytd_pnl
        FROM {PNL_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND pnl_date >= (
              SELECT DATE_SUB(MAX(pnl_date), {days})
              FROM {PNL_TABLE}
              WHERE ticker_region = '{ticker_region}'
          )
        ORDER BY pnl_date DESC, desk
    """)
    steps.append(f"Queried {PNL_TABLE} for last {days} days: {len(rows)} rows")

    if not rows:
        return {
            "result": {
                "ticker": ticker,
                "ticker_region": ticker_region,
                "in_position_book": True,
                "days_requested": days,
                "message": "No P&L data found for this period",
                "desk_summary": [],
                "daily_detail": [],
            },
            "calculation_steps": steps + ["No P&L rows returned"],
            "sources": [PNL_TABLE, XREF_TABLE, DEMO_COMPANIES],
        }

    # Aggregate by desk
    desk_summary = _safe_query(spark, f"""
        SELECT desk,
               SUM(daily_pnl) AS total_pnl,
               MIN(daily_pnl) AS worst_day,
               MAX(daily_pnl) AS best_day,
               COUNT(*) AS trading_days
        FROM {PNL_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND pnl_date >= (
              SELECT DATE_SUB(MAX(pnl_date), {days})
              FROM {PNL_TABLE}
              WHERE ticker_region = '{ticker_region}'
          )
        GROUP BY desk
        ORDER BY 2 DESC
    """)

    for d in desk_summary:
        steps.append(
            f"  {d['desk']}: total P&L {_fmt(d['total_pnl'])}, "
            f"best {_fmt(d['best_day'])}, worst {_fmt(d['worst_day'])}"
        )

    return {
        "result": {
            "ticker": ticker,
            "ticker_region": ticker_region,
            "in_position_book": True,
            "days_requested": days,
            "desk_summary": [
                {
                    "desk": d["desk"],
                    "total_pnl": _fmt(d["total_pnl"]),
                    "total_pnl_raw": d["total_pnl"],
                    "best_day": _fmt(d["best_day"]),
                    "worst_day": _fmt(d["worst_day"]),
                    "trading_days": d["trading_days"],
                }
                for d in desk_summary
            ],
            "daily_detail": [
                {
                    "desk": r["desk"],
                    "pnl_date": str(r["pnl_date"]),
                    "daily_pnl": _fmt(r["daily_pnl"]),
                    "daily_pnl_raw": r["daily_pnl"],
                    "mtd_pnl": _fmt(r.get("mtd_pnl")),
                    "ytd_pnl": _fmt(r.get("ytd_pnl")),
                }
                for r in rows
            ],
        },
        "calculation_steps": steps,
        "sources": [PNL_TABLE, XREF_TABLE, DEMO_COMPANIES],
    }


def get_risk_flags(spark, ticker: str) -> Dict[str, Any]:
    """Return compliance / risk flags for a given ticker.

    Checks for: Volcker, restricted list, MNPI, concentration risk.

    Returns dict with keys: result, calculation_steps, sources.
    """
    ticker_region = _resolve_ticker_region(spark, ticker)
    steps = [f"Resolved ticker '{ticker}' → ticker_region '{ticker_region}'"]

    if ticker_region is None or not _check_in_position_book(spark, ticker_region):
        return _not_in_book(ticker, ticker_region, steps)

    steps.append(f"Confirmed {ticker_region} is in position book")

    rows = _safe_query(spark, f"""
        SELECT *
        FROM {RISK_TABLE}
        WHERE ticker_region = '{ticker_region}'
        ORDER BY flag_date DESC
        LIMIT 1
    """)
    steps.append(f"Queried {RISK_TABLE} for {ticker_region}")

    if not rows:
        return {
            "result": {
                "ticker": ticker,
                "ticker_region": ticker_region,
                "in_position_book": True,
                "message": "No risk flags on file",
                "flags": {
                    "volcker": False,
                    "restricted": False,
                    "mnpi": False,
                    "concentration": False,
                },
                "any_active": False,
            },
            "calculation_steps": steps + ["No risk flag records found — all clear"],
            "sources": [RISK_TABLE, XREF_TABLE, DEMO_COMPANIES],
        }

    row = rows[0]
    flags = {
        "volcker": bool(row.get("volcker_flag", False)),
        "restricted": bool(row.get("restricted_flag", False)),
        "mnpi": bool(row.get("mnpi_flag", False)),
        "concentration": bool(row.get("concentration_flag", False)),
    }
    any_active = any(flags.values())

    for name, active in flags.items():
        status = "ACTIVE" if active else "clear"
        steps.append(f"  {name}: {status}")

    return {
        "result": {
            "ticker": ticker,
            "ticker_region": ticker_region,
            "in_position_book": True,
            "flag_date": str(row.get("flag_date", "")),
            "flags": flags,
            "any_active": any_active,
            "notes": row.get("notes", ""),
        },
        "calculation_steps": steps,
        "sources": [RISK_TABLE, XREF_TABLE, DEMO_COMPANIES],
    }


def get_position_summary(spark, ticker: str) -> Dict[str, Any]:
    """Return combined exposure, risk flags, and top 10 positions.

    This is a composite function that calls :func:`get_firm_exposure` (using
    the most recent position date), :func:`get_risk_flags`, and retrieves the
    top 10 individual positions by absolute notional.

    Returns dict with keys: result, calculation_steps, sources.
    """
    ticker_region = _resolve_ticker_region(spark, ticker)
    steps = [f"Resolved ticker '{ticker}' → ticker_region '{ticker_region}'"]

    if ticker_region is None or not _check_in_position_book(spark, ticker_region):
        return _not_in_book(ticker, ticker_region, steps)

    steps.append(f"Confirmed {ticker_region} is in position book")

    # Find the most recent position date
    date_rows = _safe_query(spark, f"""
        SELECT MAX(position_date) AS max_date
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
    """)
    max_date = date_rows[0].get("max_date") if date_rows else None

    if max_date is None:
        return {
            "result": {
                "ticker": ticker,
                "ticker_region": ticker_region,
                "in_position_book": True,
                "message": "No position data found",
            },
            "calculation_steps": steps + [f"No position rows found in {POSITIONS_TABLE}"],
            "sources": [POSITIONS_TABLE, XREF_TABLE, DEMO_COMPANIES],
        }

    as_of_date = str(max_date)
    steps.append(f"Most recent position date: {as_of_date}")

    # Get firm exposure (reuse the public function)
    exposure = get_firm_exposure(spark, ticker, as_of_date)
    steps.extend(exposure["calculation_steps"][1:])  # skip duplicate resolve step

    # Get risk flags (reuse the public function)
    risk = get_risk_flags(spark, ticker)
    steps.extend(risk["calculation_steps"][1:])  # skip duplicate resolve step

    # Top 10 positions by absolute notional
    top10_rows = _safe_query(spark, f"""
        SELECT desk, asset_class, book_type, notional, market_value,
               quantity, currency, strategy
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND position_date = '{as_of_date}'
        ORDER BY ABS(notional) DESC
        LIMIT 10
    """)
    steps.append(f"Retrieved top {len(top10_rows)} positions by |notional|")

    all_sources = sorted(set(
        exposure["sources"] + risk["sources"] + [POSITIONS_TABLE]
    ))

    return {
        "result": {
            "ticker": ticker,
            "ticker_region": ticker_region,
            "in_position_book": True,
            "as_of_date": as_of_date,
            "exposure": exposure["result"],
            "risk": risk["result"],
            "top_positions": [
                {
                    "desk": r.get("desk"),
                    "asset_class": r.get("asset_class"),
                    "book_type": r.get("book_type"),
                    "notional": _fmt(r.get("notional")),
                    "notional_raw": r.get("notional"),
                    "market_value": _fmt(r.get("market_value")),
                    "quantity": r.get("quantity"),
                    "currency": r.get("currency", "USD"),
                    "strategy": r.get("strategy", ""),
                }
                for r in top10_rows
            ],
        },
        "calculation_steps": steps,
        "sources": all_sources,
    }


def get_desk_positions(
    spark,
    ticker: str,
    desk: str,
) -> Dict[str, Any]:
    """Return full position detail for one desk.

    Parameters
    ----------
    ticker : str
        Short ticker (e.g. 'NVDA').
    desk : str
        Desk name (e.g. 'Equity Trading', 'Derivatives').

    Returns dict with keys: result, calculation_steps, sources.
    """
    ticker_region = _resolve_ticker_region(spark, ticker)
    steps = [f"Resolved ticker '{ticker}' → ticker_region '{ticker_region}'"]

    if ticker_region is None or not _check_in_position_book(spark, ticker_region):
        return _not_in_book(ticker, ticker_region, steps)

    steps.append(f"Confirmed {ticker_region} is in position book")

    # Find the most recent position date for this desk
    date_rows = _safe_query(spark, f"""
        SELECT MAX(position_date) AS max_date
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND desk = '{desk}'
    """)
    max_date = date_rows[0].get("max_date") if date_rows else None

    if max_date is None:
        return {
            "result": {
                "ticker": ticker,
                "ticker_region": ticker_region,
                "in_position_book": True,
                "desk": desk,
                "message": f"No positions found for desk '{desk}'",
                "positions": [],
            },
            "calculation_steps": steps + [f"No position rows found for desk '{desk}'"],
            "sources": [POSITIONS_TABLE, XREF_TABLE, DEMO_COMPANIES],
        }

    as_of_date = str(max_date)
    steps.append(f"Most recent position date for desk '{desk}': {as_of_date}")

    # Full detail for this desk
    rows = _safe_query(spark, f"""
        SELECT desk, asset_class, book_type, notional, market_value,
               quantity, currency, strategy, position_date
        FROM {POSITIONS_TABLE}
        WHERE ticker_region = '{ticker_region}'
          AND desk = '{desk}'
          AND position_date = '{as_of_date}'
        ORDER BY ABS(notional) DESC
    """)
    steps.append(f"Retrieved {len(rows)} positions for desk '{desk}'")

    total_notional = sum(r.get("notional", 0) or 0 for r in rows)
    total_mv = sum(r.get("market_value", 0) or 0 for r in rows)
    steps.append(f"Desk total notional: {_fmt(total_notional)}, market value: {_fmt(total_mv)}")

    return {
        "result": {
            "ticker": ticker,
            "ticker_region": ticker_region,
            "in_position_book": True,
            "desk": desk,
            "as_of_date": as_of_date,
            "total_notional": _fmt(total_notional),
            "total_notional_raw": total_notional,
            "total_market_value": _fmt(total_mv),
            "total_market_value_raw": total_mv,
            "position_count": len(rows),
            "positions": [
                {
                    "asset_class": r.get("asset_class"),
                    "book_type": r.get("book_type"),
                    "notional": _fmt(r.get("notional")),
                    "notional_raw": r.get("notional"),
                    "market_value": _fmt(r.get("market_value")),
                    "market_value_raw": r.get("market_value"),
                    "quantity": r.get("quantity"),
                    "currency": r.get("currency", "USD"),
                    "strategy": r.get("strategy", ""),
                }
                for r in rows
            ],
        },
        "calculation_steps": steps,
        "sources": [POSITIONS_TABLE, XREF_TABLE, DEMO_COMPANIES],
    }
