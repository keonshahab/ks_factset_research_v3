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

LLM_ENDPOINT = "databricks-claude-opus-4-6"   # system.ai.databricks-claude-opus-4-6
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
    """Mimics the object returned by `spark.sql(query)` — supports `.collect()`."""

    def __init__(self, cursor, sql):
        self._cursor = cursor
        self._sql = sql

    def collect(self):
        self._cursor.execute(self._sql)
        columns = [desc[0] for desc in self._cursor.description]
        return [_DictRow(dict(zip(columns, row))) for row in self._cursor.fetchall()]


class _SQLWarehouseProxy:
    """Drop-in replacement for SparkSession that routes SQL through a
    Databricks SQL warehouse via ``databricks-sql-connector``.

    Provides the same ``proxy.sql(query).collect()`` →
    ``[row.asDict() for row in rows]`` interface used by financial_tools
    and position_tools.
    """

    def __init__(self, warehouse_id: str):
        from databricks import sql as dbsql
        from databricks.sdk.core import Config

        # Use the SDK auth chain — it automatically resolves credentials
        # in Model Serving, notebooks, and local dev environments.
        cfg = Config()

        host = (cfg.host or "").removeprefix("https://").removeprefix("http://")
        if not host:
            raise RuntimeError(
                "_SQLWarehouseProxy: Databricks host not found. "
                f"DATABRICKS_HOST env var = {os.environ.get('DATABRICKS_HOST', '<not set>')!r}"
            )

        # Extract Bearer token from SDK auth headers
        headers = cfg.authenticate()
        token = headers.get("Authorization", "").removeprefix("Bearer ")
        if not token:
            raise RuntimeError(
                "_SQLWarehouseProxy: No auth token found via SDK Config. "
                f"Auth type = {cfg.auth_type}, "
                f"DATABRICKS_TOKEN env var set = {bool(os.environ.get('DATABRICKS_TOKEN'))}"
            )

        self._connection = dbsql.connect(
            server_hostname=host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            access_token=token,
        )

    def sql(self, query):
        """Return a _PendingQuery whose .collect() executes the SQL."""
        cursor = self._connection.cursor()
        return _PendingQuery(cursor, query)


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

## Rules

- **Always proactively check positions** when answering research questions about a \
ticker. Even if the user only asks about fundamentals, include a brief position \
context section.
- Use the `get_position_summary` tool whenever a ticker is mentioned to check if it \
is in the book.
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

def execute_tool(tool_name, arguments, engine, spark_session):
    """Execute a tool by name and return the result as a JSON string."""
    # ── Citation Engine ──────────────────────────────────────────────
    if tool_name == "search_documents":
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

        # Try SparkSession first (works in notebook / cluster context)
        try:
            from pyspark.sql import SparkSession

            self.spark = SparkSession.builder.getOrCreate()
        except Exception:
            # Model Serving: no JVM — use SQL warehouse proxy instead
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

            # Execute each tool call
            for tc in tool_calls:
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
                    tool_result = execute_tool(
                        fn_name, fn_args, self.engine, self.spark,
                    )
                    tspan.set_outputs({"result_length": len(tool_result)})

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
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
