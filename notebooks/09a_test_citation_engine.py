# Databricks notebook source

# MAGIC %md
# MAGIC # 09a — Test Citation Engine
# MAGIC
# MAGIC **Purpose:** Validate `CitationEngine` — query routing, per-index search, combined search,
# MAGIC ticker filtering, and citation formatting.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook `08_vector_indexes` has been run (all three indexes are ONLINE).
# MAGIC - `src/citation_engine.py` is accessible on the repo root.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 0: Setup

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch -q

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os

# Ensure the repo root is on the Python path so `from src.citation_engine import …` works
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "/Workspace/Repos"
# In Databricks, the repo root is usually already on the path; add it just in case
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

from src.citation_engine import (
    CitationEngine,
    SearchResult,
    FILINGS,
    EARNINGS,
    NEWS,
    ALL_SOURCE_TYPES,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1: Test `route_query` — Keyword Heuristics

# COMMAND ----------

print("=" * 70)
print("TEST: route_query")
print("=" * 70)

test_cases = [
    # (query, expected_source_types)
    ("What are the risk factors in the 10-K filing?",     {FILINGS}),
    ("What did management say on the earnings call?",     {EARNINGS}),
    ("Recent analyst news on semiconductor companies",    {NEWS}),
    ("10-K annual report risk factors",                   {FILINGS}),
    ("Earnings call guidance and revenue growth outlook", {EARNINGS}),
    ("StreetAccount headline on acquisition deal",        {NEWS}),
    ("Tell me about NVIDIA",                              ALL_SOURCE_TYPES),   # no keywords → all
]

passed = 0
for query, expected in test_cases:
    result = CitationEngine.route_query(query)
    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        print(f"  {status}: route_query(\"{query}\")")
        print(f"         expected {expected}, got {result}")

print(f"\nroute_query: {passed}/{len(test_cases)} passed")
assert passed == len(test_cases), f"route_query: {len(test_cases) - passed} test(s) failed"
print("All route_query tests passed.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2: Connect to Indexes

# COMMAND ----------

print("Connecting to all three vector search indexes...")
engine = CitationEngine()
print(f"  Filing index:   {engine.filing_index}")
print(f"  Earnings index: {engine.earnings_index}")
print(f"  News index:     {engine.news_index}")
print("Connection successful.\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3: Per-Index Search

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.a — Filings Only

# COMMAND ----------

print("=" * 70)
print("TEST: search — filings only")
print("=" * 70)

filing_results = engine.search(
    query="What were the key risk factors disclosed in the most recent annual report?",
    source_types={FILINGS},
    top_k=3,
)

print(f"Returned {len(filing_results)} filing results")
assert len(filing_results) > 0, "FAIL: expected at least 1 filing result"
for r in filing_results:
    assert r.source_type == FILINGS, f"FAIL: expected source_type={FILINGS}, got {r.source_type}"
    assert r.chunk_text, "FAIL: chunk_text is empty"
    assert r.doc_name, "FAIL: doc_name is empty"
    assert r.relevance_score > 0, "FAIL: relevance_score should be positive"

for i, r in enumerate(filing_results):
    print(f"\n--- Filing Result {i+1} (score: {r.relevance_score:.4f}) ---")
    print(f"  doc_name:    {r.doc_name}")
    print(f"  source_type: {r.source_type}")
    print(f"  metadata:    ticker={r.metadata.get('ticker')}, type={r.metadata.get('fds_filing_type')}, year={r.metadata.get('filing_year')}")
    print(f"  text:        {r.chunk_text[:150]}...")

print("\nFilings-only search: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.b — Earnings Only

# COMMAND ----------

print("=" * 70)
print("TEST: search — earnings only")
print("=" * 70)

earnings_results = engine.search(
    query="What is management's outlook for revenue growth next quarter?",
    source_types={EARNINGS},
    top_k=3,
)

print(f"Returned {len(earnings_results)} earnings results")
assert len(earnings_results) > 0, "FAIL: expected at least 1 earnings result"
for r in earnings_results:
    assert r.source_type == EARNINGS, f"FAIL: expected source_type={EARNINGS}, got {r.source_type}"
    assert r.chunk_text, "FAIL: chunk_text is empty"

for i, r in enumerate(earnings_results):
    print(f"\n--- Earnings Result {i+1} (score: {r.relevance_score:.4f}) ---")
    print(f"  doc_name:    {r.doc_name}")
    print(f"  source_type: {r.source_type}")
    print(f"  metadata:    ticker={r.metadata.get('ticker')}, quarter={r.metadata.get('transcript_quarter')}, year={r.metadata.get('transcript_year')}")
    print(f"  text:        {r.chunk_text[:150]}...")

print("\nEarnings-only search: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.c — News Only

# COMMAND ----------

print("=" * 70)
print("TEST: search — news only")
print("=" * 70)

news_results = engine.search(
    query="Recent analyst commentary on semiconductor industry trends",
    source_types={NEWS},
    top_k=3,
)

print(f"Returned {len(news_results)} news results")
assert len(news_results) > 0, "FAIL: expected at least 1 news result"
for r in news_results:
    assert r.source_type == NEWS, f"FAIL: expected source_type={NEWS}, got {r.source_type}"
    assert r.chunk_text, "FAIL: chunk_text is empty"

for i, r in enumerate(news_results):
    print(f"\n--- News Result {i+1} (score: {r.relevance_score:.4f}) ---")
    print(f"  doc_name:    {r.doc_name}")
    print(f"  source_type: {r.source_type}")
    print(f"  metadata:    ticker={r.metadata.get('ticker')}, headline={r.metadata.get('headline', '')[:60]}")
    print(f"  text:        {r.chunk_text[:150]}...")

print("\nNews-only search: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4: Combined Search (All Indexes)

# COMMAND ----------

print("=" * 70)
print("TEST: search — combined (all indexes)")
print("=" * 70)

combined_results = engine.search(
    query="What is NVIDIA's competitive position in the AI chip market?",
    top_k=3,
)

print(f"Returned {len(combined_results)} combined results")
assert len(combined_results) > 0, "FAIL: expected results from combined search"

# Verify results are sorted by relevance descending
for i in range(1, len(combined_results)):
    assert combined_results[i].relevance_score <= combined_results[i - 1].relevance_score, \
        f"FAIL: results not sorted — score[{i-1}]={combined_results[i-1].relevance_score:.4f} < score[{i}]={combined_results[i].relevance_score:.4f}"

source_types_seen = {r.source_type for r in combined_results}
print(f"Source types in results: {source_types_seen}")

for i, r in enumerate(combined_results):
    print(f"\n--- Combined Result {i+1} (score: {r.relevance_score:.4f}) ---")
    print(f"  doc_name:    {r.doc_name}")
    print(f"  source_type: {r.source_type}")
    print(f"  text:        {r.chunk_text[:150]}...")

print("\nCombined search: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5: Ticker Filter

# COMMAND ----------

print("=" * 70)
print("TEST: search — ticker filter (NVDA)")
print("=" * 70)

nvda_results = engine.search(
    query="Revenue growth and profitability",
    ticker="NVDA",
    top_k=3,
)

print(f"Returned {len(nvda_results)} results for ticker=NVDA")
assert len(nvda_results) > 0, "FAIL: expected results for NVDA"

for r in nvda_results:
    actual_ticker = r.metadata.get("ticker", "")
    assert actual_ticker == "NVDA", f"FAIL: expected ticker=NVDA, got '{actual_ticker}'"

for i, r in enumerate(nvda_results):
    print(f"\n--- NVDA Result {i+1} (score: {r.relevance_score:.4f}) ---")
    print(f"  doc_name:    {r.doc_name}")
    print(f"  source_type: {r.source_type}")
    print(f"  ticker:      {r.metadata.get('ticker')}")
    print(f"  text:        {r.chunk_text[:150]}...")

print("\nTicker filter: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6: Citation Formatting

# COMMAND ----------

print("=" * 70)
print("TEST: format_citations")
print("=" * 70)

citation_text = engine.format_citations(combined_results)
print(citation_text)
print()

# Verify numbered markers
for i in range(1, len(combined_results) + 1):
    assert f"[{i}]" in citation_text, f"FAIL: missing citation marker [{i}]"

# Verify citation_key was set on each result
for i, r in enumerate(combined_results, start=1):
    assert r.citation_key == f"[{i}]", f"FAIL: citation_key not set for result {i}"

# Edge case: empty results
empty_text = engine.format_citations([])
assert empty_text == "(no results)", f"FAIL: empty format_citations returned '{empty_text}'"

print("format_citations: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7: Confidence Level

# COMMAND ----------

print("=" * 70)
print("TEST: get_confidence_level")
print("=" * 70)

# Build synthetic results to test threshold logic
def _make_result(score: float) -> SearchResult:
    return SearchResult(
        chunk_text="test",
        doc_name="test",
        citation_key="",
        relevance_score=score,
        source_type=FILINGS,
    )

high_results = [_make_result(0.85), _make_result(0.80), _make_result(0.75)]
med_results  = [_make_result(0.70), _make_result(0.60), _make_result(0.55)]
low_results  = [_make_result(0.40), _make_result(0.30), _make_result(0.20)]

assert CitationEngine.get_confidence_level(high_results) == "HIGH", "FAIL: expected HIGH"
assert CitationEngine.get_confidence_level(med_results) == "MEDIUM", "FAIL: expected MEDIUM"
assert CitationEngine.get_confidence_level(low_results) == "LOW", "FAIL: expected LOW"
assert CitationEngine.get_confidence_level([]) == "LOW", "FAIL: expected LOW for empty"

# Also check with the real combined results
real_conf = CitationEngine.get_confidence_level(combined_results)
print(f"Confidence for combined results: {real_conf}")
assert real_conf in ("HIGH", "MEDIUM", "LOW"), f"FAIL: unexpected confidence '{real_conf}'"

print("get_confidence_level: PASSED\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary

# COMMAND ----------

print("=" * 70)
print("ALL CITATION ENGINE TESTS PASSED")
print("=" * 70)
print()
print("Tests executed:")
print("  1. route_query       — keyword heuristics → correct index routing")
print("  2. Per-index search  — filings, earnings, news individually")
print("  3. Combined search   — all indexes, results sorted by score")
print("  4. Ticker filter     — results restricted to NVDA only")
print("  5. format_citations  — numbered markers, previews, edge cases")
print("  6. get_confidence_level — HIGH / MEDIUM / LOW thresholds")
