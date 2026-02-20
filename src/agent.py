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
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import mlflow.pyfunc
from mlflow.deployments import get_deploy_client
from mlflow.types.llm import ChatCompletionResponse

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
    get_desk_positions,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_ENDPOINT = "databricks-claude-sonnet-4-6"   # system.ai.databricks-claude-sonnet-4-6
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
        self._current_user = "UNKNOWN"
        self._session_user = "UNKNOWN"
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
        except Exception as e:
            raise RuntimeError(
                f"_SQLWarehouseProxy: connection test failed. "
                f"host={self._host!r}, http_path={self._http_path!r}, "
                f"auth_type={self._cfg.auth_type}, error={e}"
            ) from e
        finally:
            try:
                conn.close()
            except Exception:
                pass

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
    "complete analysis", "comprehensive",
    "tell me everything", "research report", "research briefing",
    "credit briefing", "equity briefing",
    "all in one report", "one report",
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


def _search_for_briefing(engine, query, source_types, ticker):
    """Helper: run a semantic search and return a serializable dict."""
    results = engine.search(
        query=query,
        source_types=set(source_types),
        ticker=ticker,
        top_k=5,
    )
    return {
        "citations": engine.format_citations(results),
        "confidence": engine.get_confidence_level(results),
        "result_count": len(results),
        "results": [
            {
                "doc_name": r.doc_name,
                "chunk_text": r.chunk_text,
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
        self.client = get_deploy_client("databricks")
        self.engine = CitationEngine()

        # Use the active SparkSession if running on a Databricks cluster
        # (notebook context).  In Model Serving there is no pre-existing
        # Spark session, so getActiveSession() returns None and we fall
        # back to a lightweight SQL warehouse proxy over HTTP.
        # NOTE: Do NOT use getOrCreate() — in Model Serving it can create
        # a local Spark session (if pyspark is installed) that cannot
        # access Unity Catalog tables.
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark is None:
                raise RuntimeError("No active Spark session")
            self.spark = spark
        except Exception:
            self.spark = _SQLWarehouseProxy(WAREHOUSE_ID)

    @mlflow.trace(name="research_agent")
    def predict(self, context, messages, params=None):
        """Run the agent's tool-calling loop.

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

        # ── Pre-fetch: detect "full briefing" and run tools before LLM ──
        last_user_text = ""
        for msg in reversed(messages):
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if role == "user" and content:
                last_user_text = content
                break

        prefetch_ticker = ticker or _extract_ticker_from_text(last_user_text)

        if prefetch_ticker and _is_briefing_request(last_user_text):
            with mlflow.start_span(name="prefetch_full_briefing") as pf_span:
                pf_span.set_inputs({"ticker": prefetch_ticker})
                prefetch_data = _execute_full_briefing(
                    prefetch_ticker, self.engine, self.spark,
                )
                pf_span.set_outputs({"sections": list(prefetch_data.keys())})

            # Inject a synthetic tool-call cycle so the LLM sees pre-fetched
            # data as if it had called get_full_briefing itself.
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
                "content": json.dumps(prefetch_data, default=str),
            })

        # ── Tool-calling loop ────────────────────────────────────────
        for round_num in range(MAX_TOOL_ROUNDS):
            with mlflow.start_span(name=f"llm_call_{round_num}") as span:
                span.set_inputs({
                    "round": round_num,
                    "message_count": len(conversation),
                })

                response = self.client.predict(
                    endpoint=LLM_ENDPOINT,
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
