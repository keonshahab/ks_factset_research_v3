# Databricks notebook source

# MAGIC %md
# MAGIC # 09d — Agent Setup & Deployment
# MAGIC
# MAGIC **Purpose:** Build, register, and deploy the FactSet Research Agent using
# MAGIC Mosaic AI Agent Framework.  The agent orchestrates **14 tools** spanning
# MAGIC document search (`CitationEngine`), financial analysis (`financial_tools`),
# MAGIC and position / risk management (`position_tools`).
# MAGIC
# MAGIC **LLM:** `system.ai.databricks-claude-opus-4-6` (Databricks-hosted — no API key required)
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebooks `01`–`08` have been run (tables and vector indexes populated).
# MAGIC - Notebook `09c` has been run (position sample data seeded).
# MAGIC - `databricks-vectorsearch`, `databricks-agents`, `mlflow>=2.14` installable.
# MAGIC
# MAGIC **Architecture:**
# MAGIC ```
# MAGIC User question ──► Agent (Claude Opus 4.6) ──► Tool calls ──► Structured response
# MAGIC                        │                          │
# MAGIC                        ▼                          ▼
# MAGIC                   System prompt              14 tools:
# MAGIC                   (role, indexes,            • search_documents  (citation_engine)
# MAGIC                    format rules,             • 8 financial_tools
# MAGIC                    position check)           • 5 position_tools
# MAGIC ```
# MAGIC
# MAGIC **Input:** `question`, `ticker`, `active_doc_ids`, `conversation_history`
# MAGIC
# MAGIC **Deploy:** Mosaic AI Agent Framework → `ks_factset_research_v3_agent` endpoint.  MLflow tracing.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Setup

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch databricks-agents mlflow>=2.14 -q

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os, json

repo_root = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "__file__" in dir()
    else "/Workspace/Repos"
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Imports & Configuration

# COMMAND ----------

import importlib
import mlflow
import mlflow.pyfunc
from mlflow.deployments import get_deploy_client
from mlflow.types.llm import ChatCompletionResponse

# Hot-reload source modules so edits take effect without cluster restart
import src.citation_engine
import src.financial_tools
import src.position_tools
importlib.reload(src.citation_engine)
importlib.reload(src.financial_tools)
importlib.reload(src.position_tools)

from src.citation_engine import (
    CitationEngine,
    FILINGS,
    EARNINGS,
    NEWS,
    ALL_SOURCE_TYPES,
)
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

print("All modules loaded.")

# COMMAND ----------

# Configuration
CATALOG = "ks_factset_research_v3"
SCHEMA = "gold"
LLM_ENDPOINT = "databricks-claude-opus-4-6"          # system.ai.databricks-claude-opus-4-6
AGENT_ENDPOINT = "ks_factset_research_v3_agent"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.research_agent"
MAX_TOOL_ROUNDS = 8   # safety limit on tool-calling iterations

print(f"LLM endpoint:       {LLM_ENDPOINT}")
print(f"Agent endpoint:     {AGENT_ENDPOINT}")
print(f"Registered model:   {REGISTERED_MODEL_NAME}")
print(f"Max tool rounds:    {MAX_TOOL_ROUNDS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: System Prompt

# COMMAND ----------

SYSTEM_PROMPT = """You are a senior research analyst assistant with access to FactSet data, \
financial analytics, and the firm's position book. Your job is to answer investment research \
questions with precision, transparency, and full source attribution.

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
3. **Position Context** — If the ticker is in our position book, include current exposure, \
P&L trends, and any active risk flags. If not in the book, state that clearly.
4. **Calculations** — Show the computation steps for any derived metrics \
(leverage, DSCR, covenant checks).
5. **Sources** — List every data source used with citation keys \
(e.g., [1] NVDA 10-K 2024).
6. **Confidence** — State HIGH / MEDIUM / LOW based on source relevance scores.
7. **Related Questions** — Suggest 2-3 follow-up questions the analyst might find useful.

## Rules

- **Always proactively check positions** when answering research questions about a ticker. \
Even if the user only asks about fundamentals, include a brief position context section.
- Use the `get_position_summary` tool whenever a ticker is mentioned to check if it is \
in the book.
- Cite specific numbers — never say "revenue grew" without stating the amount and percentage.
- When comparing periods, always show both absolute values and percentage changes.
- For covenant analysis, use standard thresholds (Debt/Equity <= 3.0x, \
Debt/Assets <= 0.6x, DSCR >= 1.5x) unless the user specifies different thresholds.
- If a tool returns no data, say so explicitly rather than guessing.
- Format large numbers as $X.XB or $X.XM for readability.
"""

print(f"System prompt: {len(SYSTEM_PROMPT):,} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Tool Definitions
# MAGIC
# MAGIC 14 tools registered in OpenAI-compatible function-calling format:
# MAGIC - 1 from `citation_engine` (unified search)
# MAGIC - 8 from `financial_tools`
# MAGIC - 5 from `position_tools`

# COMMAND ----------

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

print(f"Registered {len(TOOLS)} tools:")
for t in TOOLS:
    print(f"  • {t['function']['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Tool Dispatcher

# COMMAND ----------

def execute_tool(tool_name, arguments, engine, spark_session):
    """Execute a tool by name and return the result as a JSON string.

    Parameters
    ----------
    tool_name : str
        The tool function name (must match one of the 14 registered tools).
    arguments : dict
        Parsed arguments from the LLM tool call.
    engine : CitationEngine
        Initialized citation engine instance.
    spark_session : SparkSession
        Active Spark session for SQL queries.

    Returns
    -------
    str
        JSON-encoded tool result.
    """
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


print("Tool dispatcher defined.")
print(f"Handles 14 tools: citation_engine (1), financial_tools (8), position_tools (5)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Agent Class
# MAGIC
# MAGIC `FactSetResearchAgent` is an `mlflow.pyfunc.ChatModel` that runs a
# MAGIC tool-calling loop: send messages + tools to the LLM, execute any tool
# MAGIC calls, append results, repeat until the LLM produces a final text answer.

# COMMAND ----------

class FactSetResearchAgent(mlflow.pyfunc.ChatModel):
    """FactSet Research Agent — tool-calling agent backed by Claude Opus 4.6.

    Orchestrates 14 tools across document search, financial analytics, and
    position management.  Deployed via Mosaic AI Agent Framework.
    """

    def __init__(self):
        # Store config as instance attributes so they survive pickling
        self.system_prompt = SYSTEM_PROMPT
        self.tools = TOOLS
        self.llm_endpoint = LLM_ENDPOINT
        self.max_tool_rounds = MAX_TOOL_ROUNDS
        # Initialized lazily in load_context
        self.client = None
        self.engine = None
        self.spark = None

    def load_context(self, context):
        """Initialize LLM client, citation engine, and Spark session."""
        from pyspark.sql import SparkSession

        self.client = get_deploy_client("databricks")
        self.engine = CitationEngine()
        self.spark = SparkSession.builder.getOrCreate()

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
        ChatResponse (dict)
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
        system_content = self.system_prompt
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
        for round_num in range(self.max_tool_rounds):
            with mlflow.start_span(name=f"llm_call_{round_num}") as span:
                span.set_inputs({
                    "round": round_num,
                    "message_count": len(conversation),
                })

                response = self.client.predict(
                    endpoint=self.llm_endpoint,
                    inputs={
                        "messages": conversation,
                        "tools": self.tools,
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
                return self._build_response(message.get("content", ""))

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
                    and "ticker" in self._tool_param_names(fn_name)
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
        return self._build_response(
            "I reached the maximum number of tool-calling rounds. "
            "Here is what I gathered so far — please refine your question "
            "if you need additional detail."
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _tool_param_names(self, tool_name):
        """Return the set of parameter names for a given tool."""
        for t in self.tools:
            if t["function"]["name"] == tool_name:
                return set(
                    t["function"]["parameters"].get("properties", {}).keys()
                )
        return set()

    @staticmethod
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


print("FactSetResearchAgent class defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Local Test — 7 Queries
# MAGIC
# MAGIC Test the agent in the notebook before deployment.

# COMMAND ----------

# Initialize agent for local testing (bypass load_context)
agent = FactSetResearchAgent()
agent.client = get_deploy_client("databricks")
agent.engine = CitationEngine()
agent.spark = spark   # Databricks notebook global

print("Agent initialized for local testing.\n")

# COMMAND ----------

TEST_TICKER = "NVDA"

TEST_QUERIES = [
    {
        "label": "1. Debt/EBITDA & covenant headroom",
        "question": "What is Total Debt/EBITDA and covenant headroom?",
    },
    {
        "label": "2. Estimates beat/miss",
        "question": "Did the company beat estimates?",
    },
    {
        "label": "3. 10-K risk factors",
        "question": "Key risks in the 10-K?",
    },
    {
        "label": "4. Recent news",
        "question": "Recent news?",
    },
    {
        "label": "5. Quarter-over-quarter comparison",
        "question": "Compare Q4 to Q3",
    },
    {
        "label": "6. Exposure & risk flags",
        "question": "What's our exposure and any risk flags?",
    },
    {
        "label": "7. Full summary",
        "question": "Full summary — research, financials, positions",
    },
]

print(f"Test ticker: {TEST_TICKER}")
print(f"Test queries: {len(TEST_QUERIES)}")
for tq in TEST_QUERIES:
    print(f"  {tq['label']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Test Queries

# COMMAND ----------

class _FakeParams:
    """Minimal stand-in for ChatParams to carry custom_inputs in local tests."""
    def __init__(self, custom_inputs):
        self.custom_inputs = custom_inputs


test_results = {}

for tq in TEST_QUERIES:
    label = tq["label"]
    question = tq["question"]

    print("=" * 70)
    print(f"TEST: {label}")
    print(f"  Q: {question}")
    print("=" * 70)

    messages = [{"role": "user", "content": question}]
    params = _FakeParams({"ticker": TEST_TICKER})

    try:
        response = agent.predict(
            context=None, messages=messages, params=params,
        )
        content = response.choices[0].message.content

        # Validate response
        assert content is not None, "FAIL: response content is None"
        assert len(content) > 50, f"FAIL: response too short ({len(content)} chars)"

        test_results[label] = "PASSED"

        # Print truncated preview
        preview = content[:500]
        if len(content) > 500:
            preview += f"\n... ({len(content):,} chars total)"
        print(f"\n{preview}\n")

    except Exception as exc:
        test_results[label] = f"FAILED: {exc}"
        print(f"\n  ERROR: {exc}\n")

# COMMAND ----------

# Test summary
print("=" * 70)
print("LOCAL TEST SUMMARY")
print("=" * 70)

passed = sum(1 for v in test_results.values() if v == "PASSED")
total = len(test_results)

for label, status in test_results.items():
    indicator = "PASS" if status == "PASSED" else "FAIL"
    print(f"  [{indicator}] {label}")

print(f"\n{passed}/{total} tests passed")
assert passed == total, f"{total - passed} test(s) failed"
print("All local agent tests passed.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7: Log & Register Model

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

input_example = {
    "messages": [
        {"role": "user", "content": "What is Total Debt/EBITDA for NVDA?"}
    ],
}

print(f"Logging model to: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

with mlflow.start_run(run_name="research_agent_v1") as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=FactSetResearchAgent(),
        registered_model_name=REGISTERED_MODEL_NAME,
        input_example=input_example,
        code_paths=[
            os.path.join(repo_root, "src"),
        ],
        pip_requirements=[
            "databricks-vectorsearch",
            "databricks-agents",
            "mlflow>=2.14",
            "pyspark",
        ],
    )

    # Tag the run with metadata
    mlflow.set_tags({
        "agent_type": "research_agent",
        "llm_endpoint": LLM_ENDPOINT,
        "tool_count": str(len(TOOLS)),
        "catalog": CATALOG,
    })

    print(f"Model logged:")
    print(f"  Run ID:     {run.info.run_id}")
    print(f"  Model URI:  {model_info.model_uri}")
    print(f"  Registered: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8: Deploy to Serving Endpoint

# COMMAND ----------

from databricks.agents import deploy

deployment = deploy(
    model_name=REGISTERED_MODEL_NAME,
    model_version=model_info.registered_model_version,
    endpoint_name=AGENT_ENDPOINT,
)

print(f"Deployment initiated:")
print(f"  Endpoint:      {AGENT_ENDPOINT}")
print(f"  Model:         {REGISTERED_MODEL_NAME}")
print(f"  Model Version: {model_info.registered_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9: Test Deployed Endpoint

# COMMAND ----------

import time

print(f"Waiting for endpoint '{AGENT_ENDPOINT}' to become ready...")

deploy_client = get_deploy_client("databricks")

for attempt in range(30):
    try:
        test_response = deploy_client.predict(
            endpoint=AGENT_ENDPOINT,
            inputs={
                "messages": [
                    {"role": "user", "content": "What is Total Debt/EBITDA for NVDA?"}
                ],
            },
        )
        print(f"Endpoint is ready (attempt {attempt + 1}).\n")
        break
    except Exception:
        time.sleep(10)
else:
    print("WARNING: Endpoint may not be ready yet. Deployed tests may fail.\n")

# COMMAND ----------

# Run 7 test queries against the deployed endpoint
deployed_results = {}

for tq in TEST_QUERIES:
    label = tq["label"]
    question = tq["question"]

    print("=" * 70)
    print(f"DEPLOYED TEST: {label}")
    print("=" * 70)

    try:
        resp = deploy_client.predict(
            endpoint=AGENT_ENDPOINT,
            inputs={
                "messages": [{"role": "user", "content": question}],
                "custom_inputs": {"ticker": TEST_TICKER},
            },
        )

        content = resp["choices"][0]["message"]["content"]
        assert content and len(content) > 50, "Response too short"

        deployed_results[label] = "PASSED"
        print(f"  Response length: {len(content):,} chars")
        print(f"  Preview: {content[:200]}...\n")

    except Exception as exc:
        deployed_results[label] = f"FAILED: {exc}"
        print(f"  ERROR: {exc}\n")

# COMMAND ----------

# Deployed test summary
print("=" * 70)
print("DEPLOYED ENDPOINT TEST SUMMARY")
print("=" * 70)

dep_passed = sum(1 for v in deployed_results.values() if v == "PASSED")
dep_total = len(deployed_results)

for label, status in deployed_results.items():
    indicator = "PASS" if status == "PASSED" else "FAIL"
    print(f"  [{indicator}] {label}")

print(f"\n{dep_passed}/{dep_total} deployed tests passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary

# COMMAND ----------

print("=" * 70)
print("AGENT SETUP & DEPLOYMENT COMPLETE")
print("=" * 70)
print()
print(f"LLM:                 {LLM_ENDPOINT} (system.ai.databricks-claude-opus-4-6)")
print(f"Registered model:    {REGISTERED_MODEL_NAME}")
print(f"Serving endpoint:    {AGENT_ENDPOINT}")
print(f"Tools registered:    {len(TOOLS)}")
print()
print("Tools:")
print("  Citation Engine (1):")
print("    • search_documents              — semantic search: filings, earnings, news")
print()
print("  Financial Tools (8):")
print("    • get_company_profile           — company metadata")
print("    • get_financial_summary         — latest financial snapshot")
print("    • compare_periods               — YoY / QoQ deltas")
print("    • calculate_leverage_ratio      — Debt/Equity, Debt/Assets")
print("    • calculate_debt_service_coverage — DSCR ratio")
print("    • check_covenant_compliance     — ratio vs threshold checks")
print("    • compare_to_estimates          — actuals vs consensus")
print("    • calculate_pro_forma_leverage  — what-if with additional debt")
print()
print("  Position Tools (5):")
print("    • get_firm_exposure             — total notional + desk/asset/book breakdown")
print("    • get_desk_pnl                  — daily P&L by desk")
print("    • get_risk_flags                — Volcker, restricted, MNPI, concentration")
print("    • get_position_summary          — combined exposure + risk + top positions")
print("    • get_desk_positions            — drill into a single desk")
print()
print("System prompt includes:")
print("  • Document index descriptions (filings, earnings, news)")
print("  • Financial tool descriptions (leverage, estimates, covenants)")
print("  • Position tool descriptions (exposure, P&L, risk flags)")
print("  • Response format: Summary → Analysis → Position Context →")
print("    Calculations → Sources → Confidence → Related Questions")
print("  • Rule: proactively check positions when answering research questions")
print()
print("Input schema:")
print("  • question            — natural-language research question")
print("  • ticker              — default ticker for the conversation (via custom_inputs)")
print("  • active_doc_ids      — document IDs to scope searches (via custom_inputs)")
print("  • conversation_history — prior messages (via messages array)")
print()
print("MLflow tracing: enabled")
print("  • Spans for each LLM call (llm_call_0, llm_call_1, ...)")
print("  • Spans for each tool execution (tool_search_documents, tool_get_risk_flags, ...)")
print()
print("Test queries validated:")
for tq in TEST_QUERIES:
    print(f"  {tq['label']}")
