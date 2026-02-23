# Databricks notebook source

# MAGIC %md
# MAGIC # 09d — Agent Setup & Deployment
# MAGIC
# MAGIC **Purpose:** Build, register, and deploy the FactSet Research Agent using
# MAGIC Mosaic AI Agent Framework.  The agent orchestrates **15 tools** spanning
# MAGIC document search (`CitationEngine`), financial analysis (`financial_tools`),
# MAGIC position / risk management (`position_tools`), and a composite briefing tool.
# MAGIC
# MAGIC **LLMs:**
# MAGIC - `system.ai.databricks-claude-sonnet-4-6` — tool-calling & normal queries
# MAGIC - `system.ai.databricks-claude-haiku-4-5` — fast briefing synthesis (pre-fetched requests)
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebooks `01`–`08` have been run (tables and vector indexes populated).
# MAGIC - Notebook `09c` has been run (position sample data seeded).
# MAGIC - `databricks-vectorsearch`, `databricks-agents`, `mlflow>=2.14` installable.
# MAGIC
# MAGIC **Architecture:**
# MAGIC ```
# MAGIC User question ──► Pre-fetch detection ──► Briefing? ─── Yes ──► Run all 12 sub-tasks
# MAGIC                                               │                  in parallel, then
# MAGIC                                               No                 synthesize with Haiku
# MAGIC                                               │
# MAGIC                                               ▼
# MAGIC                                     Agent (Claude Sonnet 4.6)
# MAGIC                                          │
# MAGIC                                          ▼
# MAGIC                                     Tool calls ──► 15 tools:
# MAGIC                                                    • get_full_briefing   (composite)
# MAGIC                                                    • search_documents    (citation_engine)
# MAGIC                                                    • 8 financial_tools
# MAGIC                                                    • 5 position_tools
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
    LLM_ENDPOINT_FAST,
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
print(f"LLM endpoint (fast):{LLM_ENDPOINT_FAST}")
print(f"Agent endpoint:     {AGENT_ENDPOINT}")
print(f"Registered model:   {REGISTERED_MODEL_NAME}")
print(f"Max tool rounds:    {MAX_TOOL_ROUNDS}")
print(f"Agent module:       {AGENT_MODULE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Verify Agent Module
# MAGIC
# MAGIC The agent class, system prompt, 15 tool definitions, and tool dispatcher
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
    DatabricksServingEndpoint(endpoint_name="databricks-claude-haiku-4-5"),
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
            "openai>=1.0",
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
from databricks.sdk import WorkspaceClient as _WC

# ── Guard: abort if there is already an in-progress deployment ────────
# Calling deploy() while a previous config update is still pending
# replaces the entire endpoint config and restarts ALL served entities,
# putting every version back into "Creating".  Wait for the previous
# deployment to finish (or delete the endpoint) before re-deploying.

_guard_client = _WC()
try:
    _existing_ep = _guard_client.serving_endpoints.get(name=AGENT_ENDPOINT)
    _cfg_update = (
        str(getattr(_existing_ep.state, "config_update", ""))
        if _existing_ep.state else ""
    )
    if "IN_PROGRESS" in _cfg_update or "NOT_READY" in _cfg_update:
        raise RuntimeError(
            f"Endpoint '{AGENT_ENDPOINT}' already has a deployment in progress "
            f"(state.config_update={_cfg_update}). "
            f"Wait for it to finish or delete the endpoint before re-deploying."
        )
    print(f"Endpoint '{AGENT_ENDPOINT}' exists, no in-progress update — safe to deploy.")
except Exception as _guard_err:
    if "RESOURCE_DOES_NOT_EXIST" in str(_guard_err) or "not found" in str(_guard_err).lower():
        print(f"Endpoint '{AGENT_ENDPOINT}' does not exist yet — will create.")
    elif isinstance(_guard_err, RuntimeError):
        raise  # re-raise our own guard error
    else:
        print(f"Guard check warning: {_guard_err}")
        print("Proceeding with deploy anyway.\n")

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
# MAGIC ## Step 5.5: Grant UC Permissions & Wait for Endpoint Ready
# MAGIC
# MAGIC The serving endpoint runs as a **system service principal** created by the
# MAGIC Agent Framework.  This principal needs explicit Unity Catalog GRANT statements
# MAGIC for the schemas and tables the agent queries.
# MAGIC
# MAGIC **Critical:** Permissions must be granted *while the endpoint is still starting*,
# MAGIC not after it becomes Ready.  The platform's health-check calls `predict`,
# MAGIC which triggers tool calls that need UC access.  If permissions aren't in place,
# MAGIC the health-check fails and the endpoint stays stuck in "Creating" forever.
# MAGIC
# MAGIC **Strategy:** Poll the endpoint's container logs for the `current_user=<uuid>`
# MAGIC printed by `_SQLWarehouseProxy.__init__`, grant permissions immediately,
# MAGIC then continue waiting for the endpoint to reach READY.

# COMMAND ----------

import re
import time as _time
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()
deploy_client = get_deploy_client("databricks")

# ── Helpers ───────────────────────────────────────────────────────────

def _get_served_model_name():
    """Extract the served model/entity name from the endpoint config."""
    try:
        _ep = _w.serving_endpoints.get(name=AGENT_ENDPOINT)
        for _cfg in [getattr(_ep, "pending_config", None), _ep.config]:
            if _cfg and getattr(_cfg, "served_entities", None):
                return _cfg.served_entities[0].name
            if _cfg and getattr(_cfg, "served_models", None):
                return _cfg.served_models[0].model_name
    except Exception:
        pass
    return None


def _discover_principal_from_logs():
    """Scrape container logs for current_user=<uuid> printed by _SQLWarehouseProxy."""
    _sm_name = _get_served_model_name()
    if not _sm_name:
        return None
    try:
        _logs_resp = _w.serving_endpoints.logs(
            name=AGENT_ENDPOINT,
            served_model_name=_sm_name,
        )
        _logs_text = getattr(_logs_resp, "logs", "") or str(_logs_resp)
        _match = re.search(r"current_user=([0-9a-f-]{36})", _logs_text)
        if _match:
            return _match.group(1)
    except Exception:
        pass
    return None


def _discover_principal_from_diagnostic():
    """Send __DIAGNOSTIC_IDENTITY__ and parse SERVING_IDENTITY from the response."""
    try:
        _diag = deploy_client.predict(
            endpoint=AGENT_ENDPOINT,
            inputs={
                "messages": [{"role": "user", "content": "__DIAGNOSTIC_IDENTITY__"}],
            },
        )
        _diag_str = str(_diag)
        _match = re.search(r"SERVING_IDENTITY=(\S+)", _diag_str)
        if _match and _match.group(1) not in ("UNKNOWN", "ERROR:"):
            return _match.group(1)
    except Exception:
        pass
    return None


def _grant_uc_permissions(principal):
    """Run all required GRANT statements for the serving principal."""
    _grants = [
        f"GRANT USE CATALOG ON CATALOG ks_factset_research_v3 TO `{principal}`",
        f"GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.gold TO `{principal}`",
        f"GRANT SELECT ON SCHEMA ks_factset_research_v3.gold TO `{principal}`",
        f"GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.demo TO `{principal}`",
        f"GRANT SELECT ON SCHEMA ks_factset_research_v3.demo TO `{principal}`",
        f"GRANT USE CATALOG ON CATALOG ks_position_sample TO `{principal}`",
        f"GRANT USE SCHEMA ON SCHEMA ks_position_sample.vendor_data TO `{principal}`",
        f"GRANT SELECT ON TABLE ks_position_sample.vendor_data.factset_symbology_xref TO `{principal}`",
    ]
    for _stmt in _grants:
        try:
            spark.sql(_stmt)
            print(f"    OK  {_stmt}")
        except Exception as _ge:
            _err = str(_ge)
            if "already" in _err.lower():
                print(f"    SKIP {_stmt}  (already granted)")
            else:
                print(f"    WARN {_stmt}")
                print(f"         -> {_err[:200]}")

# ── Main polling loop ─────────────────────────────────────────────────
# Poll endpoint state via SDK.  As soon as we discover the service
# principal from the container logs, grant UC permissions so the
# health-check predict call can succeed and the endpoint can finish
# transitioning to READY.

serving_principal = None
_grants_applied = False

print(f"Polling endpoint '{AGENT_ENDPOINT}' ...")
print("  Will grant UC permissions as soon as the service principal is discovered.\n")

for _attempt in range(90):  # up to ~15 min
    # ── 1. Check endpoint state ──────────────────────────────────────
    try:
        _ep = _w.serving_endpoints.get(name=AGENT_ENDPOINT)
        _state = _ep.state
        _ready = str(getattr(_state, "ready", "")) if _state else ""
        _config_update = str(getattr(_state, "config_update", "")) if _state else ""
    except Exception as _e:
        print(f"  [{_attempt+1:02d}] SDK poll error: {_e}")
        _time.sleep(10)
        continue

    # ── 2. If READY, we're done ──────────────────────────────────────
    if "READY" in _ready:
        print(f"  [{_attempt+1:02d}] Endpoint is READY!")
        break

    # ── 3. Try to discover principal & grant permissions ─────────────
    if not _grants_applied:
        # Method A: container logs (available once the container starts)
        if not serving_principal:
            serving_principal = _discover_principal_from_logs()
            if serving_principal:
                print(f"  [{_attempt+1:02d}] Discovered principal from container logs: {serving_principal}")

        # Method B: diagnostic predict (only works if endpoint accepts requests)
        if not serving_principal and _attempt >= 12:
            serving_principal = _discover_principal_from_diagnostic()
            if serving_principal:
                print(f"  [{_attempt+1:02d}] Discovered principal from diagnostic call: {serving_principal}")

        # Grant as soon as we have the identity
        if serving_principal and not _grants_applied:
            print(f"\n  Granting UC permissions to {serving_principal} ...")
            _grant_uc_permissions(serving_principal)
            _grants_applied = True
            print("  Permissions granted — endpoint health-check should pass now.\n")

    print(f"  [{_attempt+1:02d}] ready={_ready}, config_update={_config_update}, grants={'applied' if _grants_applied else 'pending'}")
    _time.sleep(10)

else:
    print(f"\n  WARNING: Endpoint did not reach READY within 15 minutes.")
    print("  Check the Serving UI → Events and Logs tabs for errors.")

# ── Post-loop: handle cases where principal was never found ───────────
if not _grants_applied:
    if not serving_principal:
        print("\n" + "=" * 70)
        print("MANUAL STEP REQUIRED — Grant UC permissions")
        print("=" * 70)
        print()
        print("Could not auto-discover the serving principal.")
        print("Go to Serving endpoint → Logs tab, search for 'current_user='")
        print("to find the service principal UUID, then run:\n")
        _manual_grants = [
            "GRANT USE CATALOG ON CATALOG ks_factset_research_v3 TO `<PRINCIPAL>`;",
            "GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.gold TO `<PRINCIPAL>`;",
            "GRANT SELECT ON SCHEMA ks_factset_research_v3.gold TO `<PRINCIPAL>`;",
            "GRANT USE SCHEMA ON SCHEMA ks_factset_research_v3.demo TO `<PRINCIPAL>`;",
            "GRANT SELECT ON SCHEMA ks_factset_research_v3.demo TO `<PRINCIPAL>`;",
            "GRANT USE CATALOG ON CATALOG ks_position_sample TO `<PRINCIPAL>`;",
            "GRANT USE SCHEMA ON SCHEMA ks_position_sample.vendor_data TO `<PRINCIPAL>`;",
            "GRANT SELECT ON TABLE ks_position_sample.vendor_data.factset_symbology_xref TO `<PRINCIPAL>`;",
        ]
        for _g in _manual_grants:
            print(f"  {_g}")
        print()
    else:
        # We found the principal but grants somehow weren't applied
        print(f"\nApplying grants now for {serving_principal} ...")
        _grant_uc_permissions(serving_principal)
        _grants_applied = True

# ── Quick verification ────────────────────────────────────────────────
if _grants_applied:
    print("\nVerifying endpoint with a test query ...")
    _time.sleep(5)  # brief pause for grants to propagate
    try:
        _verify = deploy_client.predict(
            endpoint=AGENT_ENDPOINT,
            inputs={
                "messages": [{"role": "user", "content": "What is the Debt/Equity ratio for NVDA?"}],
                "custom_inputs": {"ticker": "NVDA"},
            },
        )
        _content = _verify.get("choices", [{}])[0].get("message", {}).get("content", "")
        if any(k in _content for k in ("INSUFFICIENT_PRIVILEGES", "INSUFFICIENT_PERMISSIONS")):
            print(f"  Verification FAILED — still seeing permissions errors")
            print(f"  Preview: {_content[:300]}")
        else:
            print(f"  Verification PASSED — response length: {len(_content):,} chars")
            print(f"  Preview: {_content[:200]}...")
    except Exception as _ve:
        print(f"  Verification call failed: {_ve}")
        print("  Grants may need a few seconds to propagate. Try the Playground shortly.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5.6: Grant Trace-Logging Permissions
# MAGIC
# MAGIC The serving endpoint's service principal also needs workspace-level
# MAGIC permissions to write MLflow traces.  Without this, traces silently
# MAGIC fail with `PERMISSION_DENIED` on the backing notebook/experiment.
# MAGIC
# MAGIC This cell reuses the `serving_principal` discovered in Step 5.5.
# MAGIC If it wasn't found there, it re-discovers it from the container logs.

# COMMAND ----------

import re as _re
from databricks.sdk import WorkspaceClient as _WSC
from databricks.sdk.service.iam import ObjectPermissions

_w2 = _WSC()

# ── Re-discover principal if Step 5.5 didn't find it ──────────────────
_trace_principal = globals().get("serving_principal")

if not _trace_principal:
    print("serving_principal not set from Step 5.5, re-discovering from logs ...")
    try:
        _ep2 = _w2.serving_endpoints.get(name=AGENT_ENDPOINT)
        _sm2 = None
        for _cfg2 in [getattr(_ep2, "pending_config", None), _ep2.config]:
            if _cfg2 and getattr(_cfg2, "served_entities", None):
                _sm2 = _cfg2.served_entities[0].name
                break
            if _cfg2 and getattr(_cfg2, "served_models", None):
                _sm2 = _cfg2.served_models[0].model_name
                break
        if _sm2:
            _logs2 = _w2.serving_endpoints.logs(name=AGENT_ENDPOINT, served_model_name=_sm2)
            _logs_text2 = getattr(_logs2, "logs", "") or str(_logs2)
            _match2 = _re.search(r"current_user=([0-9a-f-]{36})", _logs_text2)
            if _match2:
                _trace_principal = _match2.group(1)
    except Exception as _e2:
        print(f"  Log scrape failed: {_e2}")

if not _trace_principal:
    print("ERROR: Could not discover service principal.")
    print("Skipping trace permission grants. Traces will not be logged.")
    print("You can manually grant permissions later using the service principal UUID.")
else:
    print(f"Service principal: {_trace_principal}")

    # ── 1. Grant permission on the MLflow experiment ──────────────────
    #    The model's experiment is where traces are logged by default.
    _experiment_path = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}"
    _granted_items = []

    # Try to find and grant on the model's experiment
    try:
        _exp = mlflow.get_experiment_by_name(f"/{REGISTERED_MODEL_NAME}")
        if _exp is None:
            _exp = mlflow.get_experiment_by_name(_experiment_path)
        if _exp:
            _w2.permissions.update(
                "experiments",
                _exp.experiment_id,
                access_control_list=[
                    {
                        "service_principal_name": _trace_principal,
                        "permission_level": "CAN_MANAGE",
                    }
                ],
            )
            _granted_items.append(f"experiment {_exp.experiment_id}")
    except Exception as _exp_err:
        print(f"  WARN: Could not grant experiment permission: {_exp_err}")

    # ── 2. Grant CAN_RUN on the backing notebook if we know its ID ────
    #    The trace error referenced notebook ID 25844095406540
    try:
        # Try the known notebook ID from the error logs
        _notebook_ids = ["25844095406540"]

        # Also try to find the notebook backing the registered model
        try:
            _mv = mlflow.MlflowClient().get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
            if _mv:
                _run = mlflow.get_run(_mv[0].run_id)
                _nb_path = _run.data.tags.get("mlflow.databricks.notebookPath", "")
                if _nb_path:
                    _nb_id = _run.data.tags.get("mlflow.databricks.notebookID", "")
                    if _nb_id and _nb_id not in _notebook_ids:
                        _notebook_ids.append(_nb_id)
        except Exception:
            pass

        for _nb_id in _notebook_ids:
            try:
                _w2.permissions.update(
                    "notebooks",
                    _nb_id,
                    access_control_list=[
                        {
                            "service_principal_name": _trace_principal,
                            "permission_level": "CAN_RUN",
                        }
                    ],
                )
                _granted_items.append(f"notebook {_nb_id}")
            except Exception as _nb_err:
                print(f"  WARN: Could not grant notebook {_nb_id} permission: {_nb_err}")
    except Exception as _nb_outer_err:
        print(f"  WARN: Notebook permission grant failed: {_nb_outer_err}")

    # ── Summary ───────────────────────────────────────────────────────
    if _granted_items:
        print(f"\nTrace permissions granted on: {', '.join(_granted_items)}")
    else:
        print("\nWARN: No trace permissions were granted. Traces may not be logged.")
    print("Traces should now appear in the Serving UI → Traces tab.")

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
print(f"LLM (primary):       {LLM_ENDPOINT} (tool-calling & normal queries)")
print(f"LLM (fast):          {LLM_ENDPOINT_FAST} (briefing synthesis)")
print(f"Registered model:    {REGISTERED_MODEL_NAME}")
print(f"Serving endpoint:    {AGENT_ENDPOINT}")
print(f"Tools registered:    {len(TOOLS)}")
print()
print("Tools:")
print("  Composite (1):")
print("    • get_full_briefing             — runs ALL 12 sub-tools in parallel")
print()
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
print("Performance optimizations:")
print("  • Pre-fetch detection: 'full briefing' requests bypass first LLM call")
print("  • Dual-model routing: Haiku for briefing synthesis, Sonnet for tool-calling")
print("  • Pre-formatted markdown: structured data rendered as tables before LLM")
print("  • Parallel vector search: 3 indexes queried concurrently")
print()
print("System prompt includes:")
print("  • Document index descriptions (filings, earnings, news)")
print("  • Financial tool descriptions (leverage, estimates, covenants)")
print("  • Position tool descriptions (exposure, P&L, risk flags)")
print("  • Performance guidelines: use get_full_briefing for comprehensive queries")
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
