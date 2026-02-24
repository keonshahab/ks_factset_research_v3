"""
FactSet Research Agent — code-based model for MLflow / Mosaic AI Agent Framework.

This module is designed for **code-based logging** with MLflow:

    mlflow.pyfunc.log_model(
        name="agent",
        python_model="src/agent.py",
        ...
    )

All constants (system prompt, tool definitions, config) are defined here so the
model is fully self-contained and does not depend on notebook globals.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import mlflow.pyfunc
from mlflow.deployments import get_deploy_client
from mlflow.types.llm import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatChunkChoice,
    ChatChoiceDelta,
)

from src.citation_engine import CitationEngine
from src.financial_tools import (
    get_company_profile,
    get_financial_summary,
    compare_periods,
    calculate_leverage_ratio,
    calculate_debt_service_coverage,
    check_covenant_compliance,
    compare_to_estimates,
    calculate_pro_forma_leverage,
)
from src.position_tools import (
    get_firm_exposure,
    get_desk_pnl,
    get_risk_flags,
    get_position_summary,
    get_top_holdings,
    get_desk_positions,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_ENDPOINT = "databricks-claude-sonnet-4-6"   # system.ai.databricks-claude-sonnet-4-6
LLM_ENDPOINT_FAST = "databricks-claude-haiku-4-5"  # faster model for pre-fetched briefings
MAX_TOOL_ROUNDS = 8
WAREHOUSE_ID = "4b9b953939869799"


# ---------------------------------------------------------------------------
# SQL Warehouse Proxy (for Model Serving — no Spark / JVM available)
# ---------------------------------------------------------------------------

class _DictRow:
    """Mimics a PySpark Row with `.asDict()` support."""

    def __init__(self, data: dict):
        self._data = data

    def asDict(self):
        return dict(self._data)


class _PendingQuery:
    """Mimics the object returned by `spark.sql(query)` — supports `.collect()`.

    Each query opens its own connection (with a fresh auth token) and closes
    it when done.  This avoids stale-connection / expired-token failures in
    long-running Model Serving containers.
    """

    def __init__(self, connect_fn, sql, *, proxy_host="", proxy_warehouse_id="", proxy_user=""):
        self._connect_fn = connect_fn
        self._sql = sql
        self._proxy_host = proxy_host
        self._proxy_warehouse_id = proxy_warehouse_id
        self._proxy_user = proxy_user

    def collect(self):
        connection = self._connect_fn()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(self._sql)
                columns = [desc[0] for desc in cursor.description]
                return [_DictRow(dict(zip(columns, row))) for row in cursor.fetchall()]
            except Exception as e:
                raise RuntimeError(
                    f"SQL warehouse query failed. "
                    f"host={self._proxy_host!r}, "
                    f"warehouse_id={self._proxy_warehouse_id!r}, "
                    f"current_user={self._proxy_user!r}, "
                    f"sql={self._sql[:200]!r}, "
                    f"error={e}"
                ) from e
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass
        finally:
            try:
                connection.close()
            except Exception:
                pass


class _SQLWarehouseProxy:
    """Drop-in replacement for SparkSession that routes SQL through a
    Databricks SQL warehouse via ``databricks-sql-connector``.

    Provides the same ``proxy.sql(query).collect()`` →
    ``[row.asDict() for row in rows]`` interface used by financial_tools
    and position_tools.

    Each ``.sql(query).collect()`` call opens a **fresh** connection with
    a newly-minted auth token and closes it when done.  This avoids
    stale-connection / expired-token failures in long-running Model
    Serving containers.
    """

    def __init__(self, warehouse_id: str):
        import logging

        from databricks import sql as dbsql
        from databricks.sdk.core import Config

        logger = logging.getLogger("_SQLWarehouseProxy")

        # Store for per-query connection creation
        self._dbsql = dbsql
        self._cfg = Config()

        self._host = (
            (self._cfg.host or "")
            .removeprefix("https://")
            .removeprefix("http://")
            .rstrip("/")
        )
        if not self._host:
            raise RuntimeError(
                "_SQLWarehouseProxy: Databricks host not found. "
                f"DATABRICKS_HOST env var = {os.environ.get('DATABRICKS_HOST', '<not set>')!r}"
            )

        self._warehouse_id = warehouse_id
        self._http_path = f"/sql/1.0/warehouses/{warehouse_id}"

        logger.warning(
            "_SQLWarehouseProxy: host=%r, http_path=%r, auth_type=%s",
            self._host,
            self._http_path,
            self._cfg.auth_type,
        )

        # Verify connectivity and log the actual identity so we know
        # exactly who to grant UC permissions to.  Store the identity
        # so it can be included in query-time error messages.
        # NOTE: This is best-effort — if the warehouse is stopped and
        # needs to wake up, we log a warning and proceed.  The first
        # real query will trigger the wake-up instead.
        self._current_user = "UNKNOWN"
        self._session_user = "UNKNOWN"
        try:
            conn = self._new_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT current_user() AS current_user, session_user() AS session_user")
                row = cur.fetchone()
                cur.close()
                self._current_user = row[0] if row else "UNKNOWN"
                self._session_user = row[1] if row and len(row) > 1 else "UNKNOWN"
                logger.warning(
                    "_SQLWarehouseProxy: connection test passed. "
                    "current_user=%s, session_user=%s",
                    self._current_user,
                    self._session_user,
                )
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(
                "_SQLWarehouseProxy: connection test failed (non-fatal, "
                "will retry on first query). host=%r, http_path=%r, "
                "auth_type=%s, error=%s",
                self._host, self._http_path, self._cfg.auth_type, e,
            )

    def _new_connection(self):
        """Create a fresh connection with current credentials."""
        headers = self._cfg.authenticate()
        token = headers.get("Authorization", "").removeprefix("Bearer ")
        if not token:
            raise RuntimeError(
                "_SQLWarehouseProxy: No auth token found via SDK Config. "
                f"Auth type = {self._cfg.auth_type}, "
                f"DATABRICKS_TOKEN env var set = {bool(os.environ.get('DATABRICKS_TOKEN'))}"
            )
        return self._dbsql.connect(
            server_hostname=self._host,
            http_path=self._http_path,
            access_token=token,
        )

    def sql(self, query):
        """Return a _PendingQuery whose .collect() opens a fresh connection."""
        return _PendingQuery(
            self._new_connection, query,
            proxy_host=self._host,
            proxy_warehouse_id=self._warehouse_id,
            proxy_user=self._current_user,
        )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior research analyst assistant with access to FactSet data, \
financial analytics, and the firm's position book. Your job is to answer \
investment research questions with precision, transparency, and full source \
attribution.

## Available Data Sources

### Document Indexes (semantic search via vector indexes)
- **Filings** — SEC filings (10-K, 10-Q, 8-K, 20-F) chunked and embedded from FactSet EDG
- **Earnings** — Conference call transcripts from FactSet FCST
- **News** — StreetAccount headlines and stories

### Financial Tools
- **Company profile** — Sector, country, entity metadata
- **Financial summary** — Revenue, net income, EPS, debt, cash flow (LTM / quarterly / annual)
- **Period comparison** — YoY or QoQ deltas for key metrics
- **Leverage ratios** — Debt-to-Equity, Debt-to-Assets
- **Debt service coverage** — DSCR = operating cash flow / interest expense
- **Covenant compliance** — Check ratios against covenant thresholds
- **Estimates comparison** — Actuals vs. consensus (beat / miss tracking)
- **Pro-forma leverage** — What-if with additional debt

### Position Tools
- **Top holdings** — Top N holdings across the entire portfolio ranked by absolute \
notional. Does NOT require a ticker — use this for portfolio-wide questions like \
"what are my top 10 holdings", "biggest positions firm-wide", or "show me all holdings".
- **Firm exposure** — Total notional, breakdowns by desk / asset class / book type
- **Desk P&L** — Daily profit-and-loss by desk (last N trading days)
- **Risk flags** — Volcker, restricted list, MNPI, concentration
- **Position summary** — Combined exposure + risk + top 10 positions
- **Desk positions** — Full detail for a single desk

## Response Format

Structure every response as follows:

1. **Summary** — 2-3 sentence executive summary answering the question directly.
2. **Analysis** — Detailed findings with specific numbers, ratios, and comparisons.
3. **Position Context** — If the ticker is in our position book, include current \
exposure, P&L trends, and any active risk flags. If not in the book, state that clearly.
4. **Calculations** — Show the computation steps for any derived metrics \
(leverage, DSCR, covenant checks).
5. **Sources** — List every data source used with citation keys \
(e.g., [1] NVDA 10-K 2024).
6. **Confidence** — State HIGH / MEDIUM / LOW based on source relevance scores.
7. **Related Questions** — Suggest 2-3 follow-up questions the analyst might find useful.

## Performance Guidelines

- **For comprehensive / full briefing requests** (e.g., "full briefing", "complete \
analysis", "tell me everything about", "research report on"), ALWAYS use the \
`get_full_briefing` tool. It runs all financial, position, and document searches \
in parallel and returns everything in one call. This is much faster than calling \
individual tools one by one.
- **For any question, call ALL tools you need in a single round.** Do not spread \
tool calls across multiple rounds when you can anticipate what data you need upfront. \
For example, if you need both financial data and document search results, call all \
of them together in the same round.
- Prefer `get_position_summary` over calling `get_firm_exposure` + `get_risk_flags` \
separately — it returns both plus top positions in one call.

## Rules

- **For portfolio-wide questions** (e.g., "top 10 holdings", "biggest positions", \
"show me my portfolio", "what do we hold"), use `get_top_holdings`. This tool does \
NOT require a ticker — it returns the top holdings across the entire book.
- **Always proactively check positions** when answering research questions about a \
ticker. Even if the user only asks about fundamentals, include a brief position \
context section.
- Use the `get_position_summary` tool whenever a ticker is mentioned to check if it \
is in the book (or use `get_full_briefing` which includes it).
- Cite specific numbers — never say "revenue grew" without stating the amount and \
percentage.
- When comparing periods, always show both absolute values and percentage changes.
- For covenant analysis, use standard thresholds (Debt/Equity <= 3.0x, \
Debt/Assets <= 0.6x, DSCR >= 1.5x) unless the user specifies different thresholds.
- If a tool returns no data, say so explicitly rather than guessing.
- Format large numbers as $X.XB or $X.XM for readability.
"""


# ---------------------------------------------------------------------------
# Tool Definitions (14 tools — OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    # ── Citation Engine (1) ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search across SEC filings (10-K, 10-Q, 8-K), earnings "
                "transcripts, and StreetAccount news. Returns ranked text chunks "
                "with relevance scores. Use source_types to target specific indexes, "
                "or omit to search all three."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "source_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["filings", "earnings", "news"],
                        },
                        "description": "Which indexes to search. Omit to search all three.",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Filter results to a single ticker (e.g. 'NVDA').",
                    },
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter results to specific document IDs.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results per index (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # ── Financial Tools (8) ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_company_profile",
            "description": (
                "Return company metadata — sector, country, entity type, display name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. 'NVDA').",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_summary",
            "description": (
                "Return the most recent financial snapshot: revenue, operating income, "
                "net income, EPS, total debt, total assets, interest expense, "
                "cash flow, equity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. 'NVDA').",
                    },
                    "period_type": {
                        "type": "string",
                        "enum": ["LTM", "Q", "A"],
                        "description": "Period type: LTM (default), Q (quarterly), A (annual).",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": (
                "Compare the most recent periods for a ticker. Returns YoY or QoQ "
                "deltas for revenue, operating income, net income, EPS."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                    "period_type": {
                        "type": "string",
                        "enum": ["A", "Q"],
                        "description": "A for annual, Q for quarterly (default A).",
                    },
                    "num_periods": {
                        "type": "integer",
                        "description": "Number of periods to compare (default 2).",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_leverage_ratio",
            "description": (
                "Compute leverage ratios: Debt-to-Equity and Debt-to-Assets "
                "from the most recent balance sheet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_debt_service_coverage",
            "description": (
                "Compute the Debt Service Coverage Ratio "
                "(DSCR = operating cash flow / interest expense). Uses LTM if available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_covenant_compliance",
            "description": (
                "Check current financial ratios against covenant thresholds. "
                "Supported covenants: debt_to_equity (max), debt_to_assets (max), "
                "min_dscr (min). Returns COMPLIANT, BREACH, or UNKNOWN for each."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                    "covenants": {
                        "type": "object",
                        "description": (
                            "Covenant thresholds. Keys: 'debt_to_equity' (max), "
                            "'debt_to_assets' (max), 'min_dscr' (min)."
                        ),
                        "properties": {
                            "debt_to_equity": {"type": "number"},
                            "debt_to_assets": {"type": "number"},
                            "min_dscr": {"type": "number"},
                        },
                    },
                },
                "required": ["ticker", "covenants"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_estimates",
            "description": (
                "Compare actual results to consensus estimates for the last N quarters. "
                "Shows beat/miss for each period and overall beat rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                    "metric_name": {
                        "type": "string",
                        "description": "Estimate metric (default 'EPS'). E.g. 'EPS', 'REVENUE'.",
                    },
                    "num_periods": {
                        "type": "integer",
                        "description": "Number of recent quarters (default 4).",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_pro_forma_leverage",
            "description": (
                "What-if scenario: recalculate leverage after adding new debt "
                "(in millions). Useful for modeling acquisition financing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                    "additional_debt": {
                        "type": "number",
                        "description": "Additional debt to model, in millions.",
                    },
                },
                "required": ["ticker", "additional_debt"],
            },
        },
    },
    # ── Composite Tool (1) ───────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_full_briefing",
            "description": (
                "Comprehensive research briefing — runs ALL financial tools, position "
                "tools, and document searches in parallel for a single ticker. Returns "
                "company profile, financial summary (LTM), leverage ratios, DSCR, "
                "covenant compliance (standard thresholds), earnings vs estimates "
                "(EPS + Revenue, last 4 quarters), position summary, desk P&L (30 days), "
                "risk flags, plus semantic search results from earnings transcripts, "
                "SEC filings, and news. Use this for comprehensive / full briefing "
                "requests instead of calling many individual tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. 'NVDA').",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    # ── Position Tools (5) ───────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_firm_exposure",
            "description": (
                "Return total notional exposure for a ticker with breakdowns by "
                "desk, asset class, and book type. Shows if the ticker is in the "
                "position book."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                    "as_of_date": {
                        "type": "string",
                        "description": "Position date (YYYY-MM-DD).",
                    },
                },
                "required": ["ticker", "as_of_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_desk_pnl",
            "description": (
                "Return daily P&L by desk for the last N trading days. "
                "Includes desk summary (total, best day, worst day) and daily detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of recent trading days (default 30).",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_flags",
            "description": (
                "Return compliance / risk flags: Volcker, restricted list, MNPI, "
                "concentration. Shows any active flags and notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_position_summary",
            "description": (
                "Composite view: firm exposure + risk flags + top 10 positions "
                "for a ticker. Use this for a quick overview of our position in "
                "a name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_holdings",
            "description": (
                "Top N holdings across the entire portfolio ranked by absolute "
                "notional. Does NOT require a ticker — use this when the user asks "
                "about overall portfolio holdings, biggest positions, or top "
                "exposures firm-wide. Optionally filter by desk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of holdings to return (default 10).",
                    },
                    "desk": {
                        "type": "string",
                        "description": "Optional desk filter (e.g. 'Equity Trading'). Omit for all desks.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_desk_positions",
            "description": (
                "Full position detail for one desk: all positions with notional, "
                "market value, quantity, strategy. Use after get_firm_exposure to "
                "drill down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol.",
                    },
                    "desk": {
                        "type": "string",
                        "description": "Desk name (e.g. 'Equity Trading').",
                    },
                },
                "required": ["ticker", "desk"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pre-fetch helpers (detect "full briefing" requests before the first LLM call)
# ---------------------------------------------------------------------------

_BRIEFING_PATTERNS = [
    "full briefing", "full credit", "full equity", "full report",
    "full summary", "full analysis",
    "complete analysis", "comprehensive",
    "tell me everything", "research report", "research briefing",
    "credit briefing", "equity briefing",
    "all in one report", "one report",
    "credit and equity", "equity and credit",
]

# Common English words that look like tickers but aren't
_NON_TICKERS = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "LET", "MAY",
    "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM", "MAN",
    "SAY", "SHE", "TOO", "USE", "SEC", "EPS", "LTM", "YOY", "QOQ", "PNL",
    "API", "SQL", "USD", "ETF", "IPO", "CEO", "CFO", "COO", "CTO",
})


def _is_briefing_request(text: str) -> bool:
    """Return True if the user text looks like a comprehensive briefing request."""
    lower = text.lower()
    return any(p in lower for p in _BRIEFING_PATTERNS)


_TOP_HOLDINGS_PATTERNS = [
    "top holdings", "top positions", "biggest positions",
    "biggest holdings", "largest positions", "largest holdings",
    "my portfolio", "my holdings", "my positions",
    "portfolio overview", "portfolio summary",
    "what do we hold", "what do i hold", "what are we holding",
    "show me the portfolio", "show me the book",
    "across the portfolio", "across the book",
    "firm-wide", "firmwide", "all desks",
]


def _is_top_holdings_request(text: str) -> bool:
    """Return True if the user is asking for portfolio-wide holdings."""
    lower = text.lower()
    return any(p in lower for p in _TOP_HOLDINGS_PATTERNS)


def _extract_top_n(text: str) -> int:
    """Extract the number of holdings requested (e.g. 'top 10' → 10)."""
    match = re.search(r"\btop\s+(\d+)\b", text, re.IGNORECASE)
    if match:
        return min(int(match.group(1)), 50)
    return 10


def _extract_ticker_from_text(text: str) -> str | None:
    """Try to extract a ticker symbol from user text.

    Looks for patterns like 'on NVDA', 'for AAPL', 'about MSFT', or
    standalone uppercase words (2-5 chars) that aren't common English words.
    """
    # Try explicit patterns first: "on NVDA", "for AAPL", "about MSFT"
    match = re.search(
        r"\b(?:on|for|about|of|ticker[:\s]*)\s+([A-Z]{1,5})\b", text
    )
    if match and match.group(1) not in _NON_TICKERS:
        return match.group(1)

    # Fallback: any standalone uppercase 2-5 char word that looks like a ticker
    for word in re.findall(r"\b([A-Z]{2,5})\b", text):
        if word not in _NON_TICKERS:
            return word

    return None


def _slim_briefing(results: dict) -> dict:
    """Strip verbose metadata from briefing results to reduce token count.

    Removes ``calculation_steps``, ``sources``, and ``*_raw`` fields from
    every sub-result.  Keeps all user-facing data and citation text intact.
    """
    slimmed = {}
    for key, value in results.items():
        if isinstance(value, dict):
            slimmed[key] = {
                k: v
                for k, v in value.items()
                if k not in ("calculation_steps", "sources")
                and not k.endswith("_raw")
            }
            # Recurse one level into nested dicts (e.g., result.exposure)
            inner = slimmed[key].get("result")
            if isinstance(inner, dict):
                slimmed[key]["result"] = {
                    k: v
                    for k, v in inner.items()
                    if not k.endswith("_raw")
                }
        else:
            slimmed[key] = value
    return slimmed


def _format_briefing_markdown(data: dict) -> str:
    """Convert raw briefing tool results into pre-formatted markdown sections.

    The LLM only needs to write a Summary, Analysis narrative, and Related
    Questions — the structured data is already rendered.
    """
    sections: list[str] = []

    # ── Company Profile ──────────────────────────────────────────────
    prof = (data.get("company_profile") or {}).get("result") or {}
    if prof:
        sections.append(
            f"## Company Profile\n"
            f"- **Name:** {prof.get('company_name', 'N/A')}\n"
            f"- **Ticker:** {prof.get('ticker', 'N/A')}\n"
            f"- **Sector:** {prof.get('sector', 'N/A')}\n"
            f"- **Country:** {prof.get('country', 'N/A')}"
        )

    # ── Financial Summary (LTM) ──────────────────────────────────────
    fin = (data.get("financial_summary_ltm") or {}).get("result") or {}
    if fin:
        sections.append(
            f"## Financial Summary ({fin.get('period_type', 'LTM')} — {fin.get('period_date', 'N/A')})\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Revenue | {fin.get('revenue', 'N/A')} |\n"
            f"| Operating Income | {fin.get('operating_income', 'N/A')} |\n"
            f"| Net Income | {fin.get('net_income', 'N/A')} |\n"
            f"| EPS | {fin.get('eps', 'N/A')} |\n"
            f"| Total Debt | {fin.get('total_debt', 'N/A')} |\n"
            f"| Total Assets | {fin.get('total_assets', 'N/A')} |\n"
            f"| Shareholders' Equity | {fin.get('shareholders_equity', 'N/A')} |\n"
            f"| Operating Cash Flow | {fin.get('operating_cash_flow', 'N/A')} |\n"
            f"| Interest Expense | {fin.get('interest_expense', 'N/A')} |"
        )

    # ── Leverage Ratios ──────────────────────────────────────────────
    lev = (data.get("leverage_ratios") or {}).get("result") or {}
    if lev:
        sections.append(
            f"## Leverage Ratios (as of {lev.get('period_date', 'N/A')})\n"
            f"- **Debt-to-Equity:** {lev.get('debt_to_equity', 'N/A')}\n"
            f"- **Debt-to-Assets:** {lev.get('debt_to_assets', 'N/A')}"
        )

    # ── DSCR ─────────────────────────────────────────────────────────
    dscr = (data.get("dscr") or {}).get("result") or {}
    if dscr:
        sections.append(
            f"## Debt Service Coverage\n"
            f"- **DSCR:** {dscr.get('dscr', 'N/A')} "
            f"(OCF {dscr.get('operating_cash_flow', 'N/A')} / "
            f"Int Exp {dscr.get('interest_expense', 'N/A')})"
        )

    # ── Covenant Compliance ──────────────────────────────────────────
    cov = (data.get("covenant_compliance") or {}).get("result") or {}
    cov_items = cov.get("covenants") or {}
    if cov_items:
        rows = "\n".join(
            f"| {name} | {info.get('actual', 'N/A')} | {info.get('threshold', 'N/A')} | {info.get('status', 'N/A')} |"
            for name, info in cov_items.items()
        )
        sections.append(
            f"## Covenant Compliance\n"
            f"| Covenant | Actual | Threshold | Status |\n|---|---|---|---|\n{rows}"
        )

    # ── Earnings vs Estimates ────────────────────────────────────────
    for key, label in [("eps_vs_estimates", "EPS"), ("revenue_vs_estimates", "Revenue")]:
        est = (data.get(key) or {}).get("result") or {}
        periods = est.get("periods") or []
        if periods:
            rows = "\n".join(
                f"| {p.get('period_date', '')} | {p.get('actual', 'N/A')} | "
                f"{p.get('consensus_mean', 'N/A')} | {p.get('surprise_pct', 'N/A')} | "
                f"{p.get('beat_miss', 'N/A')} |"
                for p in periods
            )
            sections.append(
                f"## {label} vs Estimates (beat rate: {est.get('beat_count', '?')}/{est.get('total_periods', '?')})\n"
                f"| Period | Actual | Consensus | Surprise | Result |\n|---|---|---|---|---|\n{rows}"
            )

    # ── Position Summary ─────────────────────────────────────────────
    pos = (data.get("position_summary") or {}).get("result") or {}
    if pos.get("in_position_book"):
        exp = pos.get("exposure") or {}
        exp_result = exp.get("result") or exp
        risk = pos.get("risk") or {}
        risk_result = risk.get("result") or risk
        flags = risk_result.get("flags") or {}
        active_flags = [f for f, v in flags.items() if v]

        sections.append(
            f"## Position Summary (as of {pos.get('as_of_date', 'N/A')})\n"
            f"- **Total Notional:** {exp_result.get('total_notional', 'N/A')}\n"
            f"- **Position Count:** {exp_result.get('position_count', 'N/A')}\n"
            f"- **Risk Flags:** {', '.join(active_flags) if active_flags else 'None active'}"
        )

        # Desk breakdown
        desks = exp_result.get("by_desk") or []
        if desks:
            desk_rows = "\n".join(
                f"| {d.get('desk', '')} | {d.get('notional', 'N/A')} | {d.get('positions', '')} |"
                for d in desks
            )
            sections.append(
                f"### By Desk\n| Desk | Notional | Positions |\n|---|---|---|\n{desk_rows}"
            )
    elif pos:
        sections.append("## Position Summary\n- Not in the position book.")

    # ── Desk P&L ─────────────────────────────────────────────────────
    pnl = (data.get("desk_pnl_30d") or {}).get("result") or {}
    pnl_desks = pnl.get("desk_summary") or []
    if pnl_desks:
        pnl_rows = "\n".join(
            f"| {d.get('desk', '')} | {d.get('total_pnl', 'N/A')} | "
            f"{d.get('best_day', 'N/A')} | {d.get('worst_day', 'N/A')} |"
            for d in pnl_desks
        )
        sections.append(
            f"## Desk P&L (Last 30 Days)\n"
            f"| Desk | Total P&L | Best Day | Worst Day |\n|---|---|---|---|\n{pnl_rows}"
        )

    # ── Document Search Results (keep brief) ─────────────────────────
    for key, label in [
        ("earnings_commentary", "Earnings Commentary"),
        ("sec_filings", "SEC Filings"),
        ("recent_news", "Recent News"),
    ]:
        doc = data.get(key) or {}
        doc_results = doc.get("results") or []
        if doc_results:
            items = "\n".join(
                f"- **{r.get('doc_name', 'Unknown')}** (score {r.get('relevance_score', 0):.2f}): "
                f"{r.get('chunk_text', '')[:200]}"
                for r in doc_results
            )
            sections.append(f"## {label}\n{items}")

    return "\n\n".join(sections)


def _search_for_briefing(engine, query, source_types, ticker):
    """Helper: run a semantic search and return a serializable dict.

    Uses top_k=3 (not 5) and truncates chunk_text to 300 chars to keep
    the briefing payload compact for faster LLM synthesis.
    """
    results = engine.search(
        query=query,
        source_types=set(source_types),
        ticker=ticker,
        top_k=3,
    )
    return {
        "citations": engine.format_citations(results),
        "confidence": engine.get_confidence_level(results),
        "result_count": len(results),
        "results": [
            {
                "doc_name": r.doc_name,
                "chunk_text": r.chunk_text[:300] + ("..." if len(r.chunk_text) > 300 else ""),
                "relevance_score": round(r.relevance_score, 4),
                "source_type": r.source_type,
            }
            for r in results
        ],
    }


def _execute_full_briefing(ticker, engine, spark_session):
    """Run all tools in parallel for a comprehensive briefing."""
    tasks = {
        "company_profile": lambda: get_company_profile(spark_session, ticker),
        "financial_summary_ltm": lambda: get_financial_summary(spark_session, ticker, "LTM"),
        "leverage_ratios": lambda: calculate_leverage_ratio(spark_session, ticker),
        "dscr": lambda: calculate_debt_service_coverage(spark_session, ticker),
        "covenant_compliance": lambda: check_covenant_compliance(
            spark_session, ticker,
            {"debt_to_equity": 3.0, "debt_to_assets": 0.6, "min_dscr": 1.5},
        ),
        "eps_vs_estimates": lambda: compare_to_estimates(spark_session, ticker, "EPS", 4),
        "revenue_vs_estimates": lambda: compare_to_estimates(spark_session, ticker, "REVENUE", 4),
        "position_summary": lambda: get_position_summary(spark_session, ticker),
        "desk_pnl_30d": lambda: get_desk_pnl(spark_session, ticker, 30),
        "earnings_commentary": lambda: _search_for_briefing(
            engine,
            f"{ticker} management commentary guidance outlook forward-looking statements",
            ["earnings"],
            ticker,
        ),
        "sec_filings": lambda: _search_for_briefing(
            engine,
            f"{ticker} key risk factors business overview financial condition",
            ["filings"],
            ticker,
        ),
        "recent_news": lambda: _search_for_briefing(
            engine,
            f"{ticker} latest developments analyst ratings price target",
            ["news"],
            ticker,
        ),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {"error": str(e)}

    return _slim_briefing(results)


def execute_tool(tool_name, arguments, engine, spark_session):
    """Execute a tool by name and return the result as a JSON string."""
    # ── Composite Tool ───────────────────────────────────────────────
    if tool_name == "get_full_briefing":
        return json.dumps(
            _execute_full_briefing(arguments["ticker"], engine, spark_session),
            default=str,
        )

    # ── Citation Engine ──────────────────────────────────────────────
    elif tool_name == "search_documents":
        source_types = set(arguments.get("source_types", [])) or None
        results = engine.search(
            query=arguments["query"],
            source_types=source_types,
            ticker=arguments.get("ticker"),
            doc_ids=arguments.get("doc_ids"),
            top_k=arguments.get("top_k", 5),
        )
        citations = engine.format_citations(results)
        confidence = engine.get_confidence_level(results)
        return json.dumps(
            {
                "results": [
                    {
                        "citation_key": r.citation_key,
                        "doc_name": r.doc_name,
                        "source_type": r.source_type,
                        "relevance_score": round(r.relevance_score, 4),
                        "chunk_text": r.chunk_text,
                        "metadata": {k: str(v) for k, v in r.metadata.items()},
                    }
                    for r in results
                ],
                "citations": citations,
                "confidence": confidence,
                "result_count": len(results),
            },
            default=str,
        )

    # ── Financial Tools ──────────────────────────────────────────────
    elif tool_name == "get_company_profile":
        return json.dumps(
            get_company_profile(spark_session, arguments["ticker"]),
            default=str,
        )
    elif tool_name == "get_financial_summary":
        return json.dumps(
            get_financial_summary(
                spark_session,
                arguments["ticker"],
                period_type=arguments.get("period_type", "LTM"),
            ),
            default=str,
        )
    elif tool_name == "compare_periods":
        return json.dumps(
            compare_periods(
                spark_session,
                arguments["ticker"],
                period_type=arguments.get("period_type", "A"),
                num_periods=arguments.get("num_periods", 2),
            ),
            default=str,
        )
    elif tool_name == "calculate_leverage_ratio":
        return json.dumps(
            calculate_leverage_ratio(spark_session, arguments["ticker"]),
            default=str,
        )
    elif tool_name == "calculate_debt_service_coverage":
        return json.dumps(
            calculate_debt_service_coverage(spark_session, arguments["ticker"]),
            default=str,
        )
    elif tool_name == "check_covenant_compliance":
        return json.dumps(
            check_covenant_compliance(
                spark_session,
                arguments["ticker"],
                arguments["covenants"],
            ),
            default=str,
        )
    elif tool_name == "compare_to_estimates":
        return json.dumps(
            compare_to_estimates(
                spark_session,
                arguments["ticker"],
                metric_name=arguments.get("metric_name", "EPS"),
                num_periods=arguments.get("num_periods", 4),
            ),
            default=str,
        )
    elif tool_name == "calculate_pro_forma_leverage":
        return json.dumps(
            calculate_pro_forma_leverage(
                spark_session,
                arguments["ticker"],
                arguments["additional_debt"],
            ),
            default=str,
        )

    # ── Position Tools ───────────────────────────────────────────────
    elif tool_name == "get_firm_exposure":
        return json.dumps(
            get_firm_exposure(
                spark_session,
                arguments["ticker"],
                arguments["as_of_date"],
            ),
            default=str,
        )
    elif tool_name == "get_desk_pnl":
        return json.dumps(
            get_desk_pnl(
                spark_session,
                arguments["ticker"],
                days=arguments.get("days", 30),
            ),
            default=str,
        )
    elif tool_name == "get_risk_flags":
        return json.dumps(
            get_risk_flags(spark_session, arguments["ticker"]),
            default=str,
        )
    elif tool_name == "get_position_summary":
        return json.dumps(
            get_position_summary(spark_session, arguments["ticker"]),
            default=str,
        )
    elif tool_name == "get_top_holdings":
        return json.dumps(
            get_top_holdings(
                spark_session,
                top_n=arguments.get("top_n", 10),
                desk=arguments.get("desk"),
            ),
            default=str,
        )
    elif tool_name == "get_desk_positions":
        return json.dumps(
            get_desk_positions(
                spark_session,
                arguments["ticker"],
                arguments["desk"],
            ),
            default=str,
        )
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------

class FactSetResearchAgent(mlflow.pyfunc.ChatModel):
    """FactSet Research Agent — tool-calling agent backed by Claude Opus 4.6.

    Orchestrates 14 tools across document search, financial analytics, and
    position management.  Deployed via Mosaic AI Agent Framework.
    """

    def load_context(self, context):
        """Initialize LLM client, citation engine, and SQL backend.

        In a Databricks notebook the active SparkSession is used.  In Model
        Serving (no JVM) we fall back to a lightweight SQL warehouse proxy
        that uses ``databricks-sql-connector`` over HTTP.
        """
        import logging

        logger = logging.getLogger("FactSetResearchAgent")
        logger.warning("load_context: starting initialization ...")

        logger.warning("load_context: creating deploy client ...")
        self.client = get_deploy_client("databricks")

        logger.warning("load_context: creating CitationEngine ...")
        self.engine = CitationEngine()

        # Use the active SparkSession if running on a Databricks cluster
        # (notebook context).  In Model Serving there is no pre-existing
        # Spark session, so getActiveSession() returns None and we fall
        # back to a lightweight SQL warehouse proxy over HTTP.
        # NOTE: Do NOT use getOrCreate() — in Model Serving it can create
        # a local Spark session (if pyspark is installed) that cannot
        # access Unity Catalog tables.
        logger.warning("load_context: setting up SQL backend ...")
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark is None:
                raise RuntimeError("No active Spark session")
            self.spark = spark
            logger.warning("load_context: using active SparkSession")
        except Exception:
            self.spark = _SQLWarehouseProxy(WAREHOUSE_ID)
            logger.warning("load_context: using _SQLWarehouseProxy")

        # OpenAI-compatible client for streaming the final synthesis call.
        # Databricks model-serving endpoints speak the OpenAI wire protocol.
        logger.warning("load_context: creating OpenAI streaming client ...")
        try:
            from openai import OpenAI

            _host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
            _token = os.environ.get("DATABRICKS_TOKEN", "")
            self.openai_client = OpenAI(
                api_key=_token,
                base_url=f"{_host}/serving-endpoints",
            )
            logger.warning("load_context: OpenAI streaming client ready")
        except Exception as _oc_err:
            self.openai_client = None
            logger.warning(
                "load_context: OpenAI client unavailable (%s) — "
                "predict_stream will fall back to word-chunked output",
                _oc_err,
            )

        logger.warning("load_context: initialization complete.")

    @mlflow.trace(name="research_agent")
    def predict(self, context, messages, params=None):
        """Run the agent's tool-calling loop.

        NOTE: predict_stream is preferred for interactive use (AI Playground).

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
        messages : list[ChatMessage]
            Conversation history.
        params : ChatParams, optional
            Supports ``custom_inputs`` with keys:
            - ``ticker`` (str): default ticker for the conversation
            - ``active_doc_ids`` (list[str]): document IDs to scope searches

        Returns
        -------
        ChatCompletionResponse
        """
        # ── Extract custom inputs ────────────────────────────────────
        custom = {}
        if params:
            if hasattr(params, "custom_inputs") and params.custom_inputs:
                custom = params.custom_inputs
            elif isinstance(params, dict) and "custom_inputs" in params:
                custom = params["custom_inputs"]

        ticker = custom.get("ticker")
        active_doc_ids = custom.get("active_doc_ids")

        # ── Build conversation with system prompt ────────────────────
        system_content = SYSTEM_PROMPT
        if ticker:
            system_content += f"\n\n## Current Context\n- **Active ticker:** {ticker}"
        if active_doc_ids:
            system_content += (
                f"\n- **Active document IDs:** {', '.join(active_doc_ids)}"
            )

        conversation = [{"role": "system", "content": system_content}]

        for msg in messages:
            if hasattr(msg, "to_dict"):
                conversation.append(msg.to_dict())
            elif isinstance(msg, dict):
                conversation.append(msg)
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        # ── Extract last user message ────────────────────────────────
        last_user_text = ""
        for msg in reversed(messages):
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if role == "user" and content:
                last_user_text = content
                break

        # ── Diagnostic: return SQL identity for UC permission grants ──
        if last_user_text.strip() == "__DIAGNOSTIC_IDENTITY__":
            try:
                rows = self.spark.sql("SELECT current_user() AS cu").collect()
                identity = rows[0].asDict()["cu"] if rows else "UNKNOWN"
            except Exception as e:
                identity = f"ERROR: {e}"
            return _build_response(f"SERVING_IDENTITY={identity}")

        # ── Pre-fetch: detect "full briefing" and run tools before LLM ──

        prefetch_ticker = ticker or _extract_ticker_from_text(last_user_text)

        is_prefetched = False
        if prefetch_ticker and _is_briefing_request(last_user_text):
            with mlflow.start_span(name="prefetch_full_briefing") as pf_span:
                pf_span.set_inputs({"ticker": prefetch_ticker})
                prefetch_data = _execute_full_briefing(
                    prefetch_ticker, self.engine, self.spark,
                )
                pf_span.set_outputs({"sections": list(prefetch_data.keys())})

            # Format the data as markdown — the LLM only writes narrative
            briefing_md = _format_briefing_markdown(prefetch_data)

            # Inject synthetic tool-call with pre-formatted markdown
            synth_call_id = "prefetch_001"
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": synth_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_full_briefing",
                        "arguments": json.dumps({"ticker": prefetch_ticker}),
                    },
                }],
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": synth_call_id,
                "content": (
                    "All data has been retrieved and pre-formatted below. "
                    "Write a concise briefing report using ONLY this data. "
                    "Include the pre-formatted tables as-is and add a short "
                    "executive Summary at the top, brief Analysis narrative, "
                    "and 2-3 Related Questions at the bottom. "
                    "Do NOT call any more tools.\n\n"
                    + briefing_md
                ),
            })
            is_prefetched = True

        # ── Pre-fetch: detect "top holdings" (no ticker needed) ──────
        if not is_prefetched and _is_top_holdings_request(last_user_text):
            top_n = _extract_top_n(last_user_text)
            with mlflow.start_span(name="prefetch_top_holdings") as th_span:
                th_span.set_inputs({"top_n": top_n})
                holdings_result = get_top_holdings(self.spark, top_n=top_n)
                th_span.set_outputs({"count": len(holdings_result.get("result", {}).get("holdings", []))})

            synth_call_id = "prefetch_holdings_001"
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": synth_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_top_holdings",
                        "arguments": json.dumps({"top_n": top_n}),
                    },
                }],
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": synth_call_id,
                "content": (
                    "Portfolio holdings data retrieved. Present the results "
                    "in a clear table format. Do NOT call any more tools.\n\n"
                    + json.dumps(holdings_result, default=str)
                ),
            })
            is_prefetched = True

        # ── Tool-calling loop ────────────────────────────────────────
        for round_num in range(MAX_TOOL_ROUNDS):
            # Use the fast model for pre-fetched briefings (tables are
            # already formatted — the LLM just writes narrative).
            endpoint = LLM_ENDPOINT_FAST if is_prefetched else LLM_ENDPOINT

            with mlflow.start_span(name=f"llm_call_{round_num}") as span:
                span.set_inputs({
                    "round": round_num,
                    "message_count": len(conversation),
                    "endpoint": endpoint,
                })

                response = self.client.predict(
                    endpoint=endpoint,
                    inputs={
                        "messages": conversation,
                        "tools": TOOLS,
                        "max_tokens": 4096,
                    },
                )

                choice = response["choices"][0]
                message = choice["message"]
                span.set_outputs({
                    "finish_reason": choice.get("finish_reason"),
                })

            # If no tool calls, return the final text response
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return _build_response(message.get("content", ""))

            # Append assistant message (with tool_calls) to conversation
            conversation.append(message)

            # Execute tool calls in parallel
            def _run_tool(tc):
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])

                # Inject default ticker if not specified by the LLM
                if (
                    ticker
                    and "ticker" not in fn_args
                    and "ticker" in _tool_param_names(fn_name)
                ):
                    fn_args["ticker"] = ticker

                # Inject active_doc_ids for search_documents
                if (
                    active_doc_ids
                    and "doc_ids" not in fn_args
                    and fn_name == "search_documents"
                ):
                    fn_args["doc_ids"] = active_doc_ids

                with mlflow.start_span(name=f"tool_{fn_name}") as tspan:
                    tspan.set_inputs(fn_args)
                    try:
                        tool_result = execute_tool(
                            fn_name, fn_args, self.engine, self.spark,
                        )
                    except Exception as tool_err:
                        tool_result = json.dumps({
                            "error": str(tool_err),
                            "tool": fn_name,
                        })
                    tspan.set_outputs({"result_length": len(tool_result)})

                return tc["id"], tool_result

            with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                futures = {pool.submit(_run_tool, tc): tc for tc in tool_calls}
                results = {}
                for future in as_completed(futures):
                    call_id, result = future.result()
                    results[call_id] = result

            # Append tool results in original order (must match tool_calls order)
            for tc in tool_calls:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": results[tc["id"]],
                })

        # Safety: hit max rounds without a final text answer
        return _build_response(
            "I reached the maximum number of tool-calling rounds. "
            "Here is what I gathered so far — please refine your question "
            "if you need additional detail."
        )

    @mlflow.trace(name="research_agent_stream")
    def predict_stream(self, context, messages, params=None):
        """Stream the agent's response token-by-token.

        Tool-calling rounds run non-streaming (we need the complete response
        to extract tool calls).  The **final** text is chunked word-by-word
        so tokens arrive at the client promptly.

        Yields
        ------
        ChatCompletionChunk
        """
        logger = logging.getLogger("FactSetResearchAgent")
        logger.warning("predict_stream: incoming request with %d messages", len(messages))

        # ── Extract custom inputs ────────────────────────────────────
        custom = {}
        if params:
            if hasattr(params, "custom_inputs") and params.custom_inputs:
                custom = params.custom_inputs
            elif isinstance(params, dict) and "custom_inputs" in params:
                custom = params["custom_inputs"]

        ticker = custom.get("ticker")
        active_doc_ids = custom.get("active_doc_ids")

        # ── Build conversation with system prompt ────────────────────
        system_content = SYSTEM_PROMPT
        if ticker:
            system_content += f"\n\n## Current Context\n- **Active ticker:** {ticker}"
        if active_doc_ids:
            system_content += (
                f"\n- **Active document IDs:** {', '.join(active_doc_ids)}"
            )

        conversation = [{"role": "system", "content": system_content}]

        for msg in messages:
            if hasattr(msg, "to_dict"):
                conversation.append(msg.to_dict())
            elif isinstance(msg, dict):
                conversation.append(msg)
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        # ── Extract last user message ────────────────────────────────
        last_user_text = ""
        for msg in reversed(messages):
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if role == "user" and content:
                last_user_text = content
                break

        # ── Diagnostic: return SQL identity ──────────────────────────
        if last_user_text.strip() == "__DIAGNOSTIC_IDENTITY__":
            try:
                rows = self.spark.sql("SELECT current_user() AS cu").collect()
                identity = rows[0].asDict()["cu"] if rows else "UNKNOWN"
            except Exception as e:
                identity = f"ERROR: {e}"
            yield _make_chunk(f"SERVING_IDENTITY={identity}")
            return

        # ── Pre-fetch: detect "full briefing" ────────────────────────
        prefetch_ticker = ticker or _extract_ticker_from_text(last_user_text)
        is_prefetched = False

        if prefetch_ticker and _is_briefing_request(last_user_text):
            with mlflow.start_span(name="prefetch_full_briefing") as pf_span:
                pf_span.set_inputs({"ticker": prefetch_ticker})
                prefetch_data = _execute_full_briefing(
                    prefetch_ticker, self.engine, self.spark,
                )
                pf_span.set_outputs({"sections": list(prefetch_data.keys())})

            briefing_md = _format_briefing_markdown(prefetch_data)

            synth_call_id = "prefetch_001"
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": synth_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_full_briefing",
                        "arguments": json.dumps({"ticker": prefetch_ticker}),
                    },
                }],
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": synth_call_id,
                "content": (
                    "All data has been retrieved and pre-formatted below. "
                    "Write a concise briefing report using ONLY this data. "
                    "Include the pre-formatted tables as-is and add a short "
                    "executive Summary at the top, brief Analysis narrative, "
                    "and 2-3 Related Questions at the bottom. "
                    "Do NOT call any more tools.\n\n"
                    + briefing_md
                ),
            })
            is_prefetched = True

        # ── Pre-fetch: detect "top holdings" (no ticker needed) ──────
        if not is_prefetched and _is_top_holdings_request(last_user_text):
            top_n = _extract_top_n(last_user_text)
            logger.warning("predict_stream: pre-fetching top %d holdings", top_n)
            with mlflow.start_span(name="prefetch_top_holdings") as th_span:
                th_span.set_inputs({"top_n": top_n})
                holdings_result = get_top_holdings(self.spark, top_n=top_n)
                th_span.set_outputs({"count": len(holdings_result.get("result", {}).get("holdings", []))})

            synth_call_id = "prefetch_holdings_001"
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": synth_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_top_holdings",
                        "arguments": json.dumps({"top_n": top_n}),
                    },
                }],
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": synth_call_id,
                "content": (
                    "Portfolio holdings data retrieved. Present the results "
                    "in a clear table format. Do NOT call any more tools.\n\n"
                    + json.dumps(holdings_result, default=str)
                ),
            })
            is_prefetched = True

        # ── Tool-calling loop (streaming) ─────────────────────────────
        for round_num in range(MAX_TOOL_ROUNDS):
            endpoint = LLM_ENDPOINT_FAST if is_prefetched else LLM_ENDPOINT

            with mlflow.start_span(name=f"llm_call_{round_num}") as span:
                span.set_inputs({
                    "round": round_num,
                    "message_count": len(conversation),
                    "endpoint": endpoint,
                })

                # Use OpenAI streaming client — tokens arrive as they're
                # generated instead of waiting for the full response.
                streamed_text = ""
                tool_call_acc = {}  # index -> {id, type, function:{name, arguments}}
                try:
                    if self.openai_client is None:
                        raise RuntimeError("no openai client")
                    stream = self.openai_client.chat.completions.create(
                        model=endpoint,
                        messages=conversation,
                        tools=TOOLS,
                        max_tokens=4096,
                        stream=True,
                    )
                    for chunk in stream:
                        choice = chunk.choices[0] if chunk.choices else None
                        if not choice:
                            continue
                        delta = choice.delta

                        # Stream text tokens to client immediately
                        if delta.content:
                            streamed_text += delta.content
                            yield _make_chunk(delta.content)

                        # Accumulate tool-call deltas (sent incrementally)
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_call_acc:
                                    tool_call_acc[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc_delta.id:
                                    tool_call_acc[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        tool_call_acc[idx]["function"]["name"] = tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        tool_call_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

                    span.set_outputs({"finish_reason": "stop" if not tool_call_acc else "tool_calls", "streamed": True})

                except Exception as _stream_err:
                    # Fallback: non-streaming via deploy client
                    logger.warning("predict_stream: streaming failed (%s), falling back to deploy client", _stream_err)
                    response = self.client.predict(
                        endpoint=endpoint,
                        inputs={
                            "messages": conversation,
                            "tools": TOOLS,
                            "max_tokens": 4096,
                        },
                    )
                    choice = response["choices"][0]
                    message = choice["message"]
                    span.set_outputs({"finish_reason": choice.get("finish_reason"), "streamed": False})

                    fb_tool_calls = message.get("tool_calls")
                    if not fb_tool_calls:
                        text = message.get("content", "")
                        words = text.split(" ")
                        for i, word in enumerate(words):
                            suffix = " " if i < len(words) - 1 else ""
                            yield _make_chunk(word + suffix)
                        return

                    # Convert to same format as streaming accumulator
                    for i, tc in enumerate(fb_tool_calls):
                        tool_call_acc[i] = tc
                    streamed_text = message.get("content") or ""

            # If no tool calls → final text already streamed
            if not tool_call_acc:
                return

            # Build tool_calls list from accumulated deltas
            tool_calls = [tool_call_acc[i] for i in sorted(tool_call_acc.keys())]
            conversation.append({
                "role": "assistant",
                "content": streamed_text or None,
                "tool_calls": tool_calls,
            })

            tool_names = [tc["function"]["name"] for tc in tool_calls]
            logger.warning("predict_stream round %d: calling tools %s", round_num, tool_names)

            def _run_tool(tc):
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])

                if (
                    ticker
                    and "ticker" not in fn_args
                    and "ticker" in _tool_param_names(fn_name)
                ):
                    fn_args["ticker"] = ticker

                if (
                    active_doc_ids
                    and "doc_ids" not in fn_args
                    and fn_name == "search_documents"
                ):
                    fn_args["doc_ids"] = active_doc_ids

                with mlflow.start_span(name=f"tool_{fn_name}") as tspan:
                    tspan.set_inputs(fn_args)
                    try:
                        tool_result = execute_tool(
                            fn_name, fn_args, self.engine, self.spark,
                        )
                    except Exception as tool_err:
                        tool_result = json.dumps({
                            "error": str(tool_err),
                            "tool": fn_name,
                        })
                    tspan.set_outputs({"result_length": len(tool_result)})

                return tc["id"], tool_result

            with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                futures = {pool.submit(_run_tool, tc): tc for tc in tool_calls}
                results = {}
                for future in as_completed(futures):
                    call_id, result = future.result()
                    results[call_id] = result

            for tc in tool_calls:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": results[tc["id"]],
                })

        # Safety: hit max rounds
        yield _make_chunk(
            "I reached the maximum number of tool-calling rounds. "
            "Please refine your question for additional detail."
        )


# ---------------------------------------------------------------------------
# Helpers (module-level so they stay out of the pickled instance)
# ---------------------------------------------------------------------------

def _build_response(content):
    """Wrap text content in a ChatCompletionResponse."""
    return ChatCompletionResponse.from_dict(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    )


def _make_chunk(content):
    """Create a single ChatCompletionChunk for streaming."""
    return ChatCompletionChunk(
        choices=[
            ChatChunkChoice(
                delta=ChatChoiceDelta(role="assistant", content=content),
            )
        ]
    )


def _tool_param_names(tool_name):
    """Return the set of parameter names for a given tool."""
    for t in TOOLS:
        if t["function"]["name"] == tool_name:
            return set(
                t["function"]["parameters"].get("properties", {}).keys()
            )
    return set()


# ---------------------------------------------------------------------------
# Register the model for code-based logging
# ---------------------------------------------------------------------------

mlflow.models.set_model(FactSetResearchAgent())
