# Databricks notebook source

# MAGIC %md
# MAGIC # 09d — Agent Setup & Deployment
# MAGIC
# MAGIC **Purpose:** Build, register, and deploy the FactSet Research Agent using
# MAGIC Mosaic AI Agent Framework.  The agent orchestrates **14 tools** spanning
# MAGIC document search (`CitationEngine`), financial analysis (`financial_tools`),
# MAGIC and position / risk management (`position_tools`).
# MAGIC
# MAGIC **LLM:** `system.ai.databricks-claude-sonnet-4-6` (Databricks-hosted — no API key required)
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebooks `01`–`08` have been run (tables and vector indexes populated).
# MAGIC - Notebook `09c` has been run (position sample data seeded).
# MAGIC - `databricks-vectorsearch`, `databricks-agents`, `mlflow>=2.14` installable.
# MAGIC
# MAGIC **Architecture:**
# MAGIC ```
# MAGIC User question ──► Agent (Claude Sonnet 4.6) ──► Tool calls ──► Structured response
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
# MAGIC
# MAGIC **Model code:** `src/agent.py` — self-contained module logged via
# MAGIC [code-based logging](https://mlflow.org/docs/latest/models.html#models-from-code)
# MAGIC (avoids cloudpickle serialization issues with Spark / VectorSearchClient).

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

if "__file__" in dir():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    # Databricks notebook: derive repo root from the notebook path
    _nb_path = (
        dbutils.notebook.entry_point  # noqa: F821
        .getDbutils().notebook().getContext()
        .notebookPath().get()
    )
    # _nb_path is e.g. "/Repos/<user>/ks_factset_research_v3/notebooks/09d_agent_setup"
    repo_root = "/Workspace" + str(_nb_path).rsplit("/notebooks/", 1)[0]

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

# Hot-reload source modules so edits take effect without cluster restart
import src.citation_engine
import src.financial_tools
import src.position_tools
import src.agent
importlib.reload(src.citation_engine)
importlib.reload(src.financial_tools)
importlib.reload(src.position_tools)
importlib.reload(src.agent)

from src.citation_engine import CitationEngine
from src.agent import (
    FactSetResearchAgent,
    SYSTEM_PROMPT,
    TOOLS,
    LLM_ENDPOINT,
    MAX_TOOL_ROUNDS,
)

print("All modules loaded.")

# COMMAND ----------

# Configuration
CATALOG = "ks_factset_research_v3"
SCHEMA = "gold"
AGENT_ENDPOINT = "ks_factset_research_v3_agent"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.research_agent"
AGENT_MODULE_PATH = os.path.join(repo_root, "src", "agent.py")

print(f"LLM endpoint:       {LLM_ENDPOINT}")
print(f"Agent endpoint:     {AGENT_ENDPOINT}")
print(f"Registered model:   {REGISTERED_MODEL_NAME}")
print(f"Max tool rounds:    {MAX_TOOL_ROUNDS}")
print(f"Agent module:       {AGENT_MODULE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Verify Agent Module
# MAGIC
# MAGIC The agent class, system prompt, 14 tool definitions, and tool dispatcher
# MAGIC all live in `src/agent.py`.  This cell confirms the module loaded correctly.

# COMMAND ----------

print(f"System prompt: {len(SYSTEM_PROMPT):,} characters")
print()
print(f"Registered {len(TOOLS)} tools:")
for t in TOOLS:
    print(f"  • {t['function']['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Local Test — 7 Queries
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
# MAGIC ## Step 4: Log & Register Model
# MAGIC
# MAGIC Uses **code-based logging** (`python_model=` path to `.py` file) instead of
# MAGIC pickle-based logging.  This avoids serialization errors with Spark /
# MAGIC VectorSearchClient references and is the MLflow-recommended approach.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Databricks resource dependencies — required so Model Serving can authenticate
# to the LLM endpoint, vector search indexes, and SQL warehouse.
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksSQLWarehouse,
)

resources = [
    DatabricksServingEndpoint(endpoint_name="databricks-claude-sonnet-4-6"),
    DatabricksVectorSearchIndex(index_name="ks_factset_research_v3.demo.filing_search_index"),
    DatabricksVectorSearchIndex(index_name="ks_factset_research_v3.demo.earnings_search_index"),
    DatabricksVectorSearchIndex(index_name="ks_factset_research_v3.demo.news_search_index"),
    DatabricksSQLWarehouse(warehouse_id="4b9b953939869799"),  # for financial & position queries
]

print(f"Logging model to: {REGISTERED_MODEL_NAME}")
print(f"Agent module:     {AGENT_MODULE_PATH}")
print(f"Resources:        {len(resources)} Databricks dependencies")
for r in resources:
    print(f"  • {r}")

# COMMAND ----------

with mlflow.start_run(run_name="research_agent_v1") as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AGENT_MODULE_PATH,
        registered_model_name=REGISTERED_MODEL_NAME,
        code_paths=[
            os.path.join(repo_root, "src"),
        ],
        pip_requirements=[
            "databricks-vectorsearch",
            "databricks-agents",
            "mlflow>=2.14",
            "databricks-sql-connector",
            "databricks-sdk",
        ],
        resources=resources,
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
# MAGIC ## Step 5: Deploy to Serving Endpoint

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
# MAGIC ## Step 5.5: Grant UC Permissions to Serving Endpoint
# MAGIC
# MAGIC The serving endpoint runs as a **system service principal** created by the
# MAGIC Agent Framework.  This principal can authenticate to the SQL warehouse
# MAGIC (because we declared `DatabricksSQLWarehouse` as a resource), but it still
# MAGIC needs explicit Unity Catalog GRANT statements for the schemas and tables
# MAGIC the agent queries:
# MAGIC
# MAGIC | Catalog | Schema | Tables |
# MAGIC |---------|--------|--------|
# MAGIC | `ks_factset_research_v3` | `gold` | company_profile, company_financials, consensus_estimates, position_exposures, position_pnl, position_risk_flags, demo_companies |
# MAGIC | `ks_position_sample` | `vendor_data` | factset_symbology_xref |
# MAGIC
# MAGIC This step discovers the service principal and runs the GRANTs automatically.

# COMMAND ----------

import re
import time as _time

deploy_client = get_deploy_client("databricks")

# ── Wait for the endpoint to accept requests ─────────────────────────
print(f"Waiting for endpoint '{AGENT_ENDPOINT}' to start...")
_endpoint_ready = False
for _attempt in range(30):
    try:
        deploy_client.predict(
            endpoint=AGENT_ENDPOINT,
            inputs={"messages": [{"role": "user", "content": "hello"}]},
        )
        _endpoint_ready = True
        break
    except Exception as _e:
        _err = str(_e)
        # If we get an actual application error (permissions, etc.) the
        # endpoint IS running — it just can't access the data yet.
        if any(k in _err for k in ("INSUFFICIENT_PERMISSIONS", "current_user=", "SQL warehouse query failed")):
            _endpoint_ready = True
            break
        _time.sleep(10)

if not _endpoint_ready:
    print("WARNING: endpoint may not be ready yet. Continuing with grants anyway.\n")

# ── Discover the serving principal ────────────────────────────────────
# The agent's _SQLWarehouseProxy embeds `current_user=` in error messages.
# Make a test call and parse the error to get the exact principal name.
serving_principal = None

# Use a query that forces a SQL tool call so permission errors surface.
_discovery_msg = "List the top 3 tickers in the financial data."
try:
    _test = deploy_client.predict(
        endpoint=AGENT_ENDPOINT,
        inputs={
            "messages": [{"role": "user", "content": _discovery_msg}],
            "custom_inputs": {"ticker": "AAPL"},
        },
    )
    # The endpoint may return 200 even when tools fail with permission errors.
    # Inspect the response content for INSUFFICIENT_PERMISSIONS.
    _resp_content = str(_test)
    if "INSUFFICIENT_PERMISSIONS" in _resp_content:
        print("Endpoint returned 200 but response contains INSUFFICIENT_PERMISSIONS.")
        # Try to extract current_user from the response text
        _match = re.search(r"current_user='([^']+)'", _resp_content)
        if _match:
            serving_principal = _match.group(1)
            print(f"Discovered serving principal from response: {serving_principal}")
        else:
            print("  Could not extract principal from response. Falling back to SDK lookup ...")
    else:
        print("Endpoint responded successfully — UC permissions are already granted.")
        serving_principal = "__ALREADY_GRANTED__"
except Exception as _e:
    _err_msg = str(_e)
    # Try to extract current_user from our enhanced error message
    _match = re.search(r"current_user='([^']+)'", _err_msg)
    if _match:
        serving_principal = _match.group(1)
        print(f"Discovered serving principal: {serving_principal}")
    else:
        print("Could not auto-discover the serving principal from the error message.")
        print(f"  Error snippet: {_err_msg[:300]}")
        print()
        print("Falling back to querying the serving endpoint permissions ...")

# ── Fallback: look up the service principal from the endpoint config ──
if serving_principal is None:
    try:
        from databricks.sdk import WorkspaceClient
        _w = WorkspaceClient()
        _ep = _w.serving_endpoints.get(name=AGENT_ENDPOINT)

        # Method 1: Check served_entities for environment_vars or owner info
        for _entity in (_ep.config.served_entities or []):
            _entity_str = str(_entity)
            # Some Agent Framework endpoints embed the SP name in env vars
            _sp_match = re.search(r"service.principal[._]?(?:name)?['\"]?\s*[:=]\s*['\"]?([^'\"}\s,]+)", _entity_str, re.IGNORECASE)
            if _sp_match:
                serving_principal = _sp_match.group(1)
                print(f"Discovered serving principal from served_entities: {serving_principal}")
                break

        # Method 2: Check endpoint ACL for a service principal
        if serving_principal is None:
            _perms = _w.serving_endpoints.get_permissions(serving_endpoint_id=_ep.id)
            for _acl in (_perms.access_control_list or []):
                if getattr(_acl, "service_principal_name", None):
                    serving_principal = _acl.service_principal_name
                    print(f"Discovered serving principal from endpoint ACL: {serving_principal}")
                    break

        # Method 3: Use the endpoint creator/owner as a last resort
        if serving_principal is None and getattr(_ep, "creator", None):
            serving_principal = _ep.creator
            print(f"Using endpoint creator as principal: {serving_principal}")

    except Exception as _sdk_err:
        print(f"  SDK lookup failed: {_sdk_err}")

# ── Run the GRANT statements ─────────────────────────────────────────
if serving_principal and serving_principal != "__ALREADY_GRANTED__":
    print(f"\nGranting UC permissions to: {serving_principal}\n")

    _grants = [
        # ── ks_factset_research_v3.gold (financial + position tables) ──
        f"GRANT USE CATALOG ON CATALOG ks_factset_research_v3 TO `{serving_principal}`",
        f"GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.gold TO `{serving_principal}`",
        f"GRANT SELECT ON SCHEMA ks_factset_research_v3.gold TO `{serving_principal}`",
        # ── ks_factset_research_v3.demo (vector search source tables) ──
        f"GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.demo TO `{serving_principal}`",
        f"GRANT SELECT ON SCHEMA ks_factset_research_v3.demo TO `{serving_principal}`",
        # ── ks_position_sample.vendor_data (ticker crosswalk) ──
        f"GRANT USE CATALOG ON CATALOG ks_position_sample TO `{serving_principal}`",
        f"GRANT USE SCHEMA ON SCHEMA ks_position_sample.vendor_data TO `{serving_principal}`",
        f"GRANT SELECT ON TABLE ks_position_sample.vendor_data.factset_symbology_xref TO `{serving_principal}`",
    ]

    for _stmt in _grants:
        try:
            spark.sql(_stmt)
            print(f"  OK  {_stmt}")
        except Exception as _ge:
            err_str = str(_ge)
            # Ignore "already granted" or "already has" style messages
            if "already" in err_str.lower():
                print(f"  SKIP {_stmt}  (already granted)")
            else:
                print(f"  WARN {_stmt}")
                print(f"       -> {err_str[:200]}")

    print("\nUC permissions granted. Testing endpoint again ...\n")

    # Quick verification
    try:
        _verify = deploy_client.predict(
            endpoint=AGENT_ENDPOINT,
            inputs={
                "messages": [{"role": "user", "content": "What is NVDA?"}],
                "custom_inputs": {"ticker": "NVDA"},
            },
        )
        _content = _verify.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"Verification PASSED — response length: {len(_content):,} chars")
        print(f"  Preview: {_content[:200]}...")
    except Exception as _ve:
        print(f"Verification failed: {_ve}")
        print("The grants may need a few seconds to propagate. Try the Playground again shortly.")

elif serving_principal != "__ALREADY_GRANTED__":
    print("\n" + "=" * 70)
    print("MANUAL STEP REQUIRED — Grant UC permissions")
    print("=" * 70)
    print()
    print("Run these SQL commands in a notebook or SQL editor, replacing")
    print("<PRINCIPAL> with the serving endpoint's identity.")
    print("(Check the endpoint Logs tab for 'current_user=' to find it.)")
    print()
    print("  GRANT USE CATALOG ON CATALOG ks_factset_research_v3 TO `<PRINCIPAL>`;")
    print("  GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.gold TO `<PRINCIPAL>`;")
    print("  GRANT SELECT ON SCHEMA ks_factset_research_v3.gold TO `<PRINCIPAL>`;")
    print("  GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.demo TO `<PRINCIPAL>`;")
    print("  GRANT SELECT ON SCHEMA ks_factset_research_v3.demo TO `<PRINCIPAL>`;")
    print("  GRANT USE CATALOG ON CATALOG ks_position_sample TO `<PRINCIPAL>`;")
    print("  GRANT USE SCHEMA ON SCHEMA ks_position_sample.vendor_data TO `<PRINCIPAL>`;")
    print("  GRANT SELECT ON TABLE ks_position_sample.vendor_data.factset_symbology_xref TO `<PRINCIPAL>`;")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Test Deployed Endpoint

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
print(f"LLM:                 {LLM_ENDPOINT} (system.ai.{LLM_ENDPOINT})")
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
