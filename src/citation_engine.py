"""
CitationEngine — unified semantic search across filing, earnings, and news vector indexes.

Connects to three Databricks Vector Search indexes with managed embeddings
(databricks-gte-large-en) and returns ranked, citation-ready results.

Usage (in a Databricks notebook):
    from src.citation_engine import CitationEngine
    engine = CitationEngine()
    results = engine.search("What are NVIDIA's key risk factors?", ticker="NVDA")
    print(engine.format_citations(results))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    from databricks.vector_search.client import VectorSearchClient
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "databricks-vectorsearch is not installed. "
        "Run `%pip install databricks-vectorsearch -q` then "
        "`dbutils.library.restartPython()` before importing CitationEngine."
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENDPOINT_NAME = "one-env-shared-endpoint-11"
CATALOG = "ks_factset_research_v3"
SCHEMA = "demo"

FILING_INDEX = f"{CATALOG}.{SCHEMA}.filing_search_index"
EARNINGS_INDEX = f"{CATALOG}.{SCHEMA}.earnings_search_index"
NEWS_INDEX = f"{CATALOG}.{SCHEMA}.news_search_index"

# Columns to retrieve per index (order matters — score is always appended last)
FILING_COLUMNS = [
    "chunk_id", "document_id", "ticker", "company_name",
    "fds_filing_type", "doc_type_label", "filing_year", "filing_quarter",
    "chunk_text",
]
EARNINGS_COLUMNS = [
    "chunk_id", "document_id", "ticker", "company_name",
    "transcript_year", "transcript_quarter",
    "chunk_text",
]
NEWS_COLUMNS = [
    "chunk_id", "document_id", "ticker", "headline",
    "story_date", "industry",
    "chunk_text",
]

# Source-type labels
FILINGS = "filings"
EARNINGS = "earnings"
NEWS = "news"

ALL_SOURCE_TYPES: Set[str] = {FILINGS, EARNINGS, NEWS}

# Keyword heuristics for route_query
_FILING_KEYWORDS = [
    "10-k", "10-q", "10k", "10q", "8-k", "8k", "20-f", "20f", "6-k", "6k",
    "annual report", "quarterly report", "filing", "sec filing",
    "risk factor", "exhibit", "prospectus",
]
_EARNINGS_KEYWORDS = [
    "earnings", "transcript", "conference call", "earnings call",
    "guidance", "outlook", "management commentary", "q&a",
    "revenue growth", "margin", "forward-looking",
]
_NEWS_KEYWORDS = [
    "news", "headline", "analyst", "streetaccount", "press release",
    "breaking", "report", "upgrade", "downgrade", "rating",
    "acquisition", "merger", "deal",
]


# ---------------------------------------------------------------------------
# Data class for a single search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """One chunk returned from a vector-search query."""

    chunk_text: str
    doc_name: str
    citation_key: str
    relevance_score: float
    source_type: str
    # Raw metadata carried along for downstream use
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CitationEngine
# ---------------------------------------------------------------------------

class CitationEngine:
    """Unified search and citation across filing, earnings, and news indexes."""

    def __init__(
        self,
        endpoint_name: str = ENDPOINT_NAME,
        filing_index: str = FILING_INDEX,
        earnings_index: str = EARNINGS_INDEX,
        news_index: str = NEWS_INDEX,
    ):
        self.endpoint_name = endpoint_name
        self.filing_index = filing_index
        self.earnings_index = earnings_index
        self.news_index = news_index

        self._vsc = VectorSearchClient()

        # Eagerly resolve index handles so connection errors surface early
        self._indexes = {
            FILINGS: self._vsc.get_index(self.endpoint_name, self.filing_index),
            EARNINGS: self._vsc.get_index(self.endpoint_name, self.earnings_index),
            NEWS: self._vsc.get_index(self.endpoint_name, self.news_index),
        }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        source_types: Optional[Set[str]] = None,
        ticker: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Run semantic search across one or more indexes.

        Parameters
        ----------
        query : str
            Natural-language query (embedded automatically by the index).
        source_types : set of {"filings", "earnings", "news"}, optional
            Which indexes to query.  ``None`` queries all three.
        ticker : str, optional
            Filter results to a single ticker (e.g. "NVDA").
        doc_ids : list of str, optional
            Filter results to specific document_id values.
        doc_types : list of str, optional
            Filter filing results to specific fds_filing_type values (e.g. ["10-K"]).
        top_k : int
            Maximum results **per index**.

        Returns
        -------
        list of SearchResult
            Merged results sorted by relevance_score descending.
        """
        if source_types is None:
            source_types = ALL_SOURCE_TYPES

        results: List[SearchResult] = []

        if FILINGS in source_types:
            results.extend(self._search_filings(query, ticker, doc_ids, doc_types, top_k))
        if EARNINGS in source_types:
            results.extend(self._search_earnings(query, ticker, doc_ids, top_k))
        if NEWS in source_types:
            results.extend(self._search_news(query, ticker, doc_ids, top_k))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def format_citations(self, results: List[SearchResult], preview_chars: int = 120) -> str:
        """Return a numbered citation map with text previews.

        Example output::

            [1] NVDA 10-K 2025 (score 0.82)
                "Risk factors include supply chain concentration…"
            [2] NVDA Earnings Q4 2025 (score 0.79)
                "Management expects revenue growth of 20% next…"
        """
        if not results:
            return "(no results)"

        lines: List[str] = []
        for i, r in enumerate(results, start=1):
            r.citation_key = f"[{i}]"
            preview = r.chunk_text[:preview_chars].replace("\n", " ").strip()
            if len(r.chunk_text) > preview_chars:
                preview += "…"
            lines.append(f"[{i}] {r.doc_name} (score {r.relevance_score:.2f})")
            lines.append(f'    "{preview}"')
        return "\n".join(lines)

    @staticmethod
    def get_confidence_level(results: List[SearchResult]) -> str:
        """Classify overall confidence based on top relevance scores.

        Returns
        -------
        "HIGH", "MEDIUM", or "LOW"
        """
        if not results:
            return "LOW"

        top_score = results[0].relevance_score
        top_3_avg = (
            sum(r.relevance_score for r in results[:3]) / min(len(results), 3)
        )

        if top_score >= 0.78 and top_3_avg >= 0.72:
            return "HIGH"
        if top_score >= 0.65 and top_3_avg >= 0.55:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def route_query(query: str) -> Set[str]:
        """Use keyword heuristics to decide which indexes to search.

        Returns a set of source-type strings.  If nothing matches, returns
        all three indexes (safe default).
        """
        q = query.lower()
        targets: Set[str] = set()

        for kw in _FILING_KEYWORDS:
            if kw in q:
                targets.add(FILINGS)
                break

        for kw in _EARNINGS_KEYWORDS:
            if kw in q:
                targets.add(EARNINGS)
                break

        for kw in _NEWS_KEYWORDS:
            if kw in q:
                targets.add(NEWS)
                break

        return targets if targets else ALL_SOURCE_TYPES

    # -----------------------------------------------------------------------
    # Per-index search helpers
    # -----------------------------------------------------------------------

    def _search_filings(
        self, query: str, ticker: Optional[str],
        doc_ids: Optional[List[str]], doc_types: Optional[List[str]],
        top_k: int,
    ) -> List[SearchResult]:
        filters = self._build_filters(ticker=ticker, doc_ids=doc_ids)
        if doc_types:
            filters["fds_filing_type"] = doc_types[0] if len(doc_types) == 1 else doc_types

        raw = self._run_query(FILINGS, FILING_COLUMNS, query, top_k, filters)

        results: List[SearchResult] = []
        for row in raw:
            meta = dict(zip(FILING_COLUMNS, row[: len(FILING_COLUMNS)]))
            score = row[-1]
            doc_name = self._format_doc_name(FILINGS, meta)
            results.append(SearchResult(
                chunk_text=meta["chunk_text"],
                doc_name=doc_name,
                citation_key="",
                relevance_score=score,
                source_type=FILINGS,
                metadata=meta,
            ))
        return results

    def _search_earnings(
        self, query: str, ticker: Optional[str],
        doc_ids: Optional[List[str]], top_k: int,
    ) -> List[SearchResult]:
        filters = self._build_filters(ticker=ticker, doc_ids=doc_ids)
        raw = self._run_query(EARNINGS, EARNINGS_COLUMNS, query, top_k, filters)

        results: List[SearchResult] = []
        for row in raw:
            meta = dict(zip(EARNINGS_COLUMNS, row[: len(EARNINGS_COLUMNS)]))
            score = row[-1]
            doc_name = self._format_doc_name(EARNINGS, meta)
            results.append(SearchResult(
                chunk_text=meta["chunk_text"],
                doc_name=doc_name,
                citation_key="",
                relevance_score=score,
                source_type=EARNINGS,
                metadata=meta,
            ))
        return results

    def _search_news(
        self, query: str, ticker: Optional[str],
        doc_ids: Optional[List[str]], top_k: int,
    ) -> List[SearchResult]:
        filters = self._build_filters(ticker=ticker, doc_ids=doc_ids)
        raw = self._run_query(NEWS, NEWS_COLUMNS, query, top_k, filters)

        results: List[SearchResult] = []
        for row in raw:
            meta = dict(zip(NEWS_COLUMNS, row[: len(NEWS_COLUMNS)]))
            score = row[-1]
            doc_name = self._format_doc_name(NEWS, meta)
            results.append(SearchResult(
                chunk_text=meta["chunk_text"],
                doc_name=doc_name,
                citation_key="",
                relevance_score=score,
                source_type=NEWS,
                metadata=meta,
            ))
        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _run_query(
        self,
        source_type: str,
        columns: List[str],
        query: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[list]:
        """Execute a similarity_search against one index and return raw rows."""
        idx = self._indexes[source_type]
        kwargs: Dict = dict(
            query_text=query,
            columns=columns,
            num_results=top_k,
        )
        if filters:
            kwargs["filters"] = filters

        response = idx.similarity_search(**kwargs)
        return response.get("result", {}).get("data_array", [])

    @staticmethod
    def _build_filters(
        ticker: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict:
        """Build a Databricks VS filter dict from common parameters."""
        filters: Dict = {}
        if ticker:
            filters["ticker"] = ticker
        if doc_ids:
            filters["document_id"] = doc_ids[0] if len(doc_ids) == 1 else doc_ids
        return filters

    @staticmethod
    def _format_doc_name(source_type: str, meta: Dict) -> str:
        """Produce a human-readable document name.

        Filings  → "NVDA 10-K 2025"
        Earnings → "NVDA Earnings Q4 2025"
        News     → headline text (truncated)
        """
        if source_type == FILINGS:
            ticker = meta.get("ticker", "???")
            filing_type = meta.get("fds_filing_type", "Filing")
            year = meta.get("filing_year", "")
            return f"{ticker} {filing_type} {year}".strip()

        if source_type == EARNINGS:
            ticker = meta.get("ticker", "???")
            quarter = meta.get("transcript_quarter", "")
            year = meta.get("transcript_year", "")
            return f"{ticker} Earnings {quarter} {year}".strip()

        if source_type == NEWS:
            headline = meta.get("headline", "")
            if headline and len(headline) > 80:
                return headline[:77] + "..."
            return headline or "(untitled)"

        return "(unknown source)"
