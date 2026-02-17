"""
Research & Deal Intelligence — Gradio application.

Two-panel dark-themed layout:

  LEFT  (~30 %) — company selector, position badge, document library, upload
  RIGHT (~70 %) — chat input, agent responses with structured sections,
                   citation panel, related-question chips, conversation history

Agent endpoint : ks_factset_research_v3_agent  (MLflow model serving)
Data           : Databricks SQL  (ks_factset_research_v3 catalog)

Env vars required:
    DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
"""

from __future__ import annotations

import os
import re
from html import escape
from typing import Any

import gradio as gr
import requests
from databricks import sql as dbsql

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

CATALOG = "ks_factset_research_v3"
DEMO = f"{CATALOG}.demo"
GOLD = f"{CATALOG}.gold"

AGENT_ENDPOINT = "ks_factset_research_v3_agent"
AGENT_URL = (
    f"https://{DATABRICKS_HOST}/serving-endpoints/{AGENT_ENDPOINT}/invocations"
)
AGENT_TIMEOUT = 120  # seconds

# ═══════════════════════════════════════════════════════════════════════════
# Example queries  (7 chips)
# ═══════════════════════════════════════════════════════════════════════════

EXAMPLE_QUERIES = [
    "What are the key risk factors in NVDA's latest 10-K?",
    "Compare NVDA revenue and EPS growth over the last 3 years",
    "What is our firm's total exposure to MSFT across desks?",
    "Calculate leverage ratios and covenant compliance for JPM",
    "Summarize the latest META earnings call highlights",
    "Show all active risk flags on our NVDA positions",
    "How did INTC's actual EPS compare to consensus estimates?",
]

# ═══════════════════════════════════════════════════════════════════════════
# Database helpers
# ═══════════════════════════════════════════════════════════════════════════

def _conn():
    """Open a fresh Databricks SQL connection."""
    return dbsql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


def _query(sql: str) -> list[dict[str, Any]]:
    """Execute *sql* and return a list of row-dicts."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _safe(ticker: str) -> str:
    """Strip anything that is not alphanumeric from a ticker value."""
    return "".join(ch for ch in ticker if ch.isalnum())

# ═══════════════════════════════════════════════════════════════════════════
# Data loaders  (left panel)
# ═══════════════════════════════════════════════════════════════════════════

def load_companies() -> list[tuple[str, str]]:
    """Return (label, ticker) pairs from v_active_companies."""
    rows = _query(
        f"SELECT ticker, display_name "
        f"FROM {GOLD}.v_active_companies "
        f"ORDER BY ticker"
    )
    return [(f"{r['ticker']} \u2014 {r['display_name']}", r["ticker"]) for r in rows]


def load_position_badge(ticker: str | None) -> str:
    """Build position-badge HTML for *ticker*."""
    if not ticker:
        return _badge_empty()

    tk = _safe(ticker)

    exp = _query(f"""
        SELECT
            COALESCE(SUM(ABS(notional)), 0) AS total_notional,
            COUNT(*)                        AS position_count
        FROM {GOLD}.position_exposures
        WHERE ticker_region = (
            SELECT ticker_region
            FROM {GOLD}.demo_companies
            WHERE ticker = '{tk}'
            LIMIT 1
        )
    """)

    flags = _query(f"""
        SELECT
            COALESCE(SUM(CAST(volcker_flag       AS INT)), 0)
          + COALESCE(SUM(CAST(restricted_flag    AS INT)), 0)
          + COALESCE(SUM(CAST(mnpi_flag          AS INT)), 0)
          + COALESCE(SUM(CAST(concentration_flag AS INT)), 0) AS flag_count
        FROM {GOLD}.position_risk_flags
        WHERE ticker_region = (
            SELECT ticker_region
            FROM {GOLD}.demo_companies
            WHERE ticker = '{tk}'
            LIMIT 1
        )
    """)

    notional = (exp[0]["total_notional"] or 0) if exp else 0
    count = (exp[0]["position_count"] or 0) if exp else 0
    fc = (flags[0]["flag_count"] or 0) if flags else 0

    n_str = _fmt_notional(notional)
    flag_span = f'<span class="flag-count">\u2691 {fc} flags</span>' if fc else ""

    return (
        '<div class="position-badge">'
        f'  <span class="notional">{n_str}</span>'
        f'  <span class="pos-meta">across {count} positions</span>'
        f"  {flag_span}"
        "</div>"
    )


def load_filings(ticker: str) -> list[dict]:
    tk = _safe(ticker)
    return _query(f"""
        SELECT document_id, fds_filing_type, doc_type_label,
               filing_year, filing_quarter, total_chunks
        FROM {DEMO}.filing_documents
        WHERE ticker = '{tk}'
        ORDER BY filing_year DESC, filing_quarter DESC
    """)


def load_earnings(ticker: str) -> list[dict]:
    tk = _safe(ticker)
    return _query(f"""
        SELECT document_id, transcript_year, transcript_quarter,
               event_date, total_chunks
        FROM {DEMO}.earnings_documents
        WHERE ticker = '{tk}'
        ORDER BY transcript_year DESC, transcript_quarter DESC
    """)


def load_news(ticker: str) -> list[dict]:
    """Query news_chunks directly (news_documents lacks headline)."""
    tk = _safe(ticker)
    return _query(f"""
        SELECT document_id,
               MAX(headline)   AS headline,
               MAX(story_date) AS story_date,
               MAX(industry)   AS industry,
               COUNT(*)        AS total_chunks
        FROM {DEMO}.news_chunks
        WHERE ticker = '{tk}'
        GROUP BY document_id
        ORDER BY MAX(story_date) DESC
        LIMIT 50
    """)

# ═══════════════════════════════════════════════════════════════════════════
# Formatting helpers  (left panel)
# ═══════════════════════════════════════════════════════════════════════════

_FILING_MARKERS: dict[str, str] = {
    "10-K": "\U0001f535",   # blue circle
    "10-Q": "\U0001f7e2",   # green circle
    "8-K":  "\U0001f7e0",   # orange circle
    "20-F": "\U0001f7e3",   # purple circle
}


def _fmt_notional(value: float) -> str:
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.0f}"


def _badge_empty() -> str:
    return '<div class="position-badge empty">Select a company to view positions</div>'


def _filing_label(r: dict) -> str:
    ft = r.get("fds_filing_type", "")
    marker = _FILING_MARKERS.get(ft, "\U0001f4c4")
    label = r.get("doc_type_label") or ft
    period = str(r["filing_year"])
    q = r.get("filing_quarter")
    if q:
        period += f" Q{q}"
    return f"{marker} {label} \u2014 {period}"


def _earnings_label(r: dict) -> str:
    period = f"Q{r['transcript_quarter']} {r['transcript_year']}"
    date = r.get("event_date") or ""
    if date:
        return f"\U0001f4ca Earnings {period} ({date})"
    return f"\U0001f4ca Earnings {period}"


def _news_label(r: dict) -> str:
    headline = r.get("headline") or ""
    if len(headline) > 60:
        headline = headline[:57] + "\u2026"
    date = r.get("story_date") or ""
    if headline:
        return f"\U0001f4f0 {headline} ({date})"
    industry = r.get("industry") or ""
    return f"\U0001f4f0 {date} \u2014 {industry}"

# ═══════════════════════════════════════════════════════════════════════════
# Agent endpoint
# ═══════════════════════════════════════════════════════════════════════════

def call_agent(
    messages: list[dict[str, str]],
    ticker: str | None = None,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    """POST to the MLflow model-serving endpoint.

    Returns ``{"content": str, "error": str|None}``.
    """
    payload: dict[str, Any] = {"messages": messages}

    custom_inputs: dict[str, Any] = {}
    if ticker:
        custom_inputs["ticker"] = ticker
    if doc_ids:
        custom_inputs["active_doc_ids"] = doc_ids
    if custom_inputs:
        payload["custom_inputs"] = custom_inputs

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            AGENT_URL,
            json=payload,
            headers=headers,
            timeout=AGENT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return {"content": content, "error": None}

    except requests.exceptions.Timeout:
        return {
            "content": "",
            "error": (
                "Request timed out after "
                f"{AGENT_TIMEOUT}s. The agent may be busy \u2014 please retry."
            ),
        }
    except requests.exceptions.ConnectionError:
        return {
            "content": "",
            "error": (
                "Cannot reach the agent endpoint. "
                "Verify DATABRICKS_HOST is set correctly."
            ),
        }
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        detail = {
            401: "Authentication failed \u2014 check DATABRICKS_TOKEN.",
            403: "Access denied to the agent endpoint.",
            404: f"Endpoint \u2018{AGENT_ENDPOINT}\u2019 not found. "
                 "Has the model been deployed?",
        }.get(int(status) if str(status).isdigit() else 0,
              f"Agent returned HTTP {status}.")
        return {"content": "", "error": detail}
    except Exception as exc:  # noqa: BLE001
        return {"content": "", "error": f"Unexpected error: {exc}"}

# ═══════════════════════════════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════════════════════════════

_SECTION_NAMES = [
    "Summary",
    "Analysis",
    "Position Context",
    "Calculations",
    "Sources",
    "Confidence",
    "Related Questions",
]

_SECTION_RE = re.compile(
    r"^#{1,3}\s*\**("
    + "|".join(re.escape(s) for s in _SECTION_NAMES)
    + r")\**\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_sections(text: str) -> dict[str, str]:
    """Split structured agent response into named sections."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {"Summary": text.strip()}

    sections: dict[str, str] = {}
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["_preamble"] = preamble

    for i, m in enumerate(matches):
        name = m.group(1).strip().title()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    return sections


def _extract_citations(sources_text: str) -> list[dict[str, str]]:
    """Parse the *Sources* section into citation dicts."""
    citations: list[dict[str, str]] = []
    for line in sources_text.splitlines():
        line = line.strip().lstrip("- ")
        m = re.match(r"\[(\d+)\]\s*(.*)", line)
        if not m:
            continue
        key = m.group(1)
        label = m.group(2).strip()
        citations.append({
            "key": key,
            "label": label,
            "source_type": _classify_citation(label),
        })
    return citations


def _classify_citation(label: str) -> str:
    ll = label.lower()
    if any(k in ll for k in ("10-k", "10-q", "8-k", "20-f", "filing", "annual")):
        return "filing"
    if any(k in ll for k in ("earnings", "transcript", "call", "guidance")):
        return "earnings"
    if any(k in ll for k in ("news", "streetaccount", "headline", "analyst",
                              "upgrade", "downgrade")):
        return "news"
    return "filing"


def _extract_confidence(text: str) -> str:
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in text.upper():
            return level
    return "MEDIUM"


def _extract_related_qs(text: str) -> list[str]:
    qs: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^[\d.\-*]+\s*", "", line.strip()).strip()
        if line and "?" in line:
            qs.append(line)
    return qs[:3]

# ═══════════════════════════════════════════════════════════════════════════
# HTML rendering  (right panel)
# ═══════════════════════════════════════════════════════════════════════════

_CITE_COLORS: dict[str, str] = {
    "filing":   "#4A90D9",
    "earnings": "#9B59B6",
    "news":     "#FF8C42",
}


def _md_to_html(text: str) -> str:
    """Minimal markdown \u2192 HTML (bold, italic, code, lists, paragraphs)."""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)

    blocks = re.split(r"\n{2,}", text)
    out: list[str] = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        lines = blk.split("\n")
        if all(re.match(r"^[\-\*\d.]+\s", ln.strip()) for ln in lines if ln.strip()):
            items = "".join(
                f"<li>{re.sub(r'^[-*0-9.]+s*', '', ln.strip())}</li>"
                for ln in lines if ln.strip()
            )
            out.append(f"<ul>{items}</ul>")
        else:
            out.append(f"<p>{blk.replace(chr(10), '<br>')}</p>")
    return "".join(out)


def _inject_cite_badges(html: str, citations: list[dict]) -> str:
    """Replace ``[N]`` with coloured inline badges."""
    cmap = {c["key"]: c for c in citations}

    def _repl(m: re.Match) -> str:
        key = m.group(1)
        c = cmap.get(key)
        if not c:
            return m.group(0)
        color = _CITE_COLORS.get(c["source_type"], "#888")
        return f'<span class="cite-badge" style="background:{color}">[{key}]</span>'

    return re.sub(r"\[(\d+)\]", _repl, html)


# -- full response renderer ------------------------------------------------

def render_agent_response(content: str) -> tuple[str, str, list[str]]:
    """Render an agent response into *(response_html, citation_html, related_qs)*."""
    sections = _parse_sections(content)
    citations = _extract_citations(sections.get("Sources", ""))
    confidence = _extract_confidence(sections.get("Confidence", ""))
    related = _extract_related_qs(sections.get("Related Questions", ""))

    parts: list[str] = []

    # confidence pill
    parts.append(
        '<div class="confidence-row">'
        f'<span class="conf-badge conf-{confidence.lower()}">{confidence}</span>'
        " confidence"
        "</div>"
    )

    # optional preamble
    if "_preamble" in sections:
        html = _inject_cite_badges(_md_to_html(sections["_preamble"]), citations)
        parts.append(f'<div class="resp-pre">{html}</div>')

    # Summary
    if "Summary" in sections:
        html = _inject_cite_badges(_md_to_html(sections["Summary"]), citations)
        parts.append(
            '<div class="resp-sec resp-summary">'
            '<div class="sec-label">Summary</div>'
            f'<div class="sec-body">{html}</div></div>'
        )

    # Analysis
    if "Analysis" in sections:
        html = _inject_cite_badges(_md_to_html(sections["Analysis"]), citations)
        parts.append(
            '<div class="resp-sec resp-analysis">'
            '<div class="sec-label">Analysis</div>'
            f'<div class="sec-body">{html}</div></div>'
        )

    # Position Context  (highlighted)
    if "Position Context" in sections:
        html = _inject_cite_badges(
            _md_to_html(sections["Position Context"]), citations,
        )
        parts.append(
            '<div class="resp-sec resp-position">'
            '<div class="sec-label">Position Context</div>'
            f'<div class="sec-body">{html}</div></div>'
        )

    # Calculations  (accordion)
    if "Calculations" in sections:
        html = _md_to_html(sections["Calculations"])
        parts.append(
            '<details class="resp-sec resp-calc">'
            '<summary class="sec-label">Calculations</summary>'
            f'<div class="sec-body">{html}</div></details>'
        )

    resp_html = "".join(parts)

    # citation panel
    cite_html = ""
    if citations:
        items = []
        for c in citations:
            color = _CITE_COLORS.get(c["source_type"], "#888")
            items.append(
                f'<li class="cite-row">'
                f'<span class="cite-key" style="background:{color}">'
                f'[{c["key"]}]</span> {escape(c["label"])}</li>'
            )
        cite_html = (
            '<details class="cite-panel" open>'
            f'<summary>Sources ({len(citations)})</summary>'
            f'<ol class="cite-list">{"".join(items)}</ol>'
            "</details>"
        )

    return resp_html, cite_html, related


# -- conversation renderer -------------------------------------------------

def render_conversation(history: list[dict]) -> str:
    """Render the full conversation as scrollable HTML."""
    if not history:
        return (
            '<div class="conv-empty">'
            "Ask a question about your selected company and documents."
            "</div>"
        )

    parts: list[str] = []
    for msg in history:
        role = msg["role"]
        if role == "user":
            parts.append(
                '<div class="conv-row conv-user">'
                f'<div class="bubble bub-user">{escape(msg["content"])}</div>'
                "</div>"
            )
        elif role == "assistant":
            parts.append(
                '<div class="conv-row conv-asst">'
                '<div class="bubble bub-asst">'
                f'{msg.get("html", escape(msg["content"]))}'
                "</div></div>"
            )
        elif role == "error":
            parts.append(
                '<div class="conv-row conv-err">'
                f'<div class="bubble bub-err">{escape(msg["content"])}</div>'
                "</div>"
            )

    return f'<div class="conv-wrap">{"".join(parts)}</div>'


def _loading_html(history: list[dict]) -> str:
    """Conversation HTML with a loading indicator appended."""
    base = render_conversation(history)
    loader = (
        '<div class="conv-row conv-asst">'
        '<div class="bubble bub-asst bub-loading">'
        '<span class="ld-dots"><span>.</span><span>.</span><span>.</span></span>'
        " Analyzing with 14 research tools\u2026"
        "</div></div>"
    )
    if "<div class=\"conv-wrap\">" in base:
        return base.replace("</div>\n", f"{loader}</div>", 1).rstrip() or base[:-6] + loader + "</div>"
    return base + loader

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════

DARK_CSS = """
/* ================================================================ */
/*  GLOBAL                                                          */
/* ================================================================ */
.gradio-container { background: #0f0f1a !important; }

/* ================================================================ */
/*  LEFT PANEL                                                      */
/* ================================================================ */
.panel-header { text-align:center; padding:16px 8px 8px; }
.panel-header h2 {
    margin:0; font-size:1.35rem; font-weight:700;
    color:#e0e0e0; letter-spacing:.02em;
}
.panel-header .subtitle {
    margin:4px 0 0; font-size:.78rem; color:#7b8794; font-weight:400;
}

.position-badge {
    display:flex; align-items:center; gap:10px;
    padding:10px 14px; border-radius:8px;
    background:#16213e; border:1px solid #1f3460;
    font-size:.92rem; flex-wrap:wrap;
}
.position-badge.empty { color:#6b7b8d; font-style:italic; }
.position-badge .notional { font-weight:700; font-size:1.05rem; color:#4a9eff; }
.position-badge .pos-meta { color:#a0aec0; }
.position-badge .flag-count {
    background:#e94560; color:#fff; padding:2px 8px;
    border-radius:4px; font-size:.8rem; font-weight:600;
}
.tab-nav button { font-size:.85rem !important; }
.upload-hint { text-align:center; font-size:.75rem; color:#5a6a7a; margin-top:2px; }

/* ================================================================ */
/*  RIGHT PANEL — conversation                                      */
/* ================================================================ */
.conv-scroll {
    max-height: 520px; overflow-y: auto;
    padding: 8px 4px; scroll-behavior: smooth;
}
.conv-empty {
    display:flex; align-items:center; justify-content:center;
    min-height:200px; color:#4a5568; font-size:.95rem;
}
.conv-wrap { display:flex; flex-direction:column; gap:14px; }
.conv-row  { display:flex; }
.conv-user { justify-content:flex-end; }
.conv-asst { justify-content:flex-start; }
.conv-err  { justify-content:center; }

.bubble {
    max-width:92%; padding:12px 16px; border-radius:12px;
    font-size:.9rem; line-height:1.55; word-break:break-word;
}
.bub-user {
    background:#1e3a5f; color:#d0e0f0; border-bottom-right-radius:4px;
}
.bub-asst {
    background:#1a1a2e; color:#dce1e8;
    border:1px solid #252547; border-bottom-left-radius:4px;
}
.bub-err {
    background:#3d1521; color:#f0a0a0;
    border:1px solid #5a2030; font-size:.85rem;
}

/* loading dots */
.bub-loading { color:#7b8794; font-style:italic; }
.ld-dots span {
    animation: blink 1.4s infinite both;
    font-size:1.2em; font-weight:700;
}
.ld-dots span:nth-child(2) { animation-delay: .2s; }
.ld-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes blink {
    0%,80%,100% { opacity:.2; }
    40% { opacity:1; }
}

/* ================================================================ */
/*  RIGHT PANEL — response sections                                 */
/* ================================================================ */
.resp-sec {
    margin:10px 0; padding:10px 14px; border-radius:8px;
    border:1px solid #252547;
}
.resp-summary  { background:#111827; }
.resp-analysis { background:#111827; }
.resp-position {
    background:#0c2a1e; border-color:#155e3b;
}
.resp-calc {
    background:#1a1528; border-color:#302050; cursor:pointer;
}
.resp-calc summary { list-style:none; }
.resp-calc summary::before { content:"\\25B6  "; font-size:.7em; }
.resp-calc[open] summary::before { content:"\\25BC  "; }
.sec-label {
    font-size:.75rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.06em; color:#7b8794; margin-bottom:4px;
}
.sec-body p  { margin:4px 0; }
.sec-body ul { margin:4px 0 4px 18px; padding:0; }
.sec-body li { margin:2px 0; }
.resp-pre { margin-bottom:6px; }

/* ================================================================ */
/*  RIGHT PANEL — citations                                         */
/* ================================================================ */
.cite-badge {
    display:inline-block; color:#fff; font-size:.72rem; font-weight:700;
    padding:1px 6px; border-radius:3px; vertical-align:middle;
    cursor:default; margin:0 1px;
}
.cite-panel {
    margin:8px 0; padding:8px 12px; border-radius:8px;
    background:#12121f; border:1px solid #252547;
}
.cite-panel summary {
    font-size:.85rem; font-weight:600; color:#a0aec0; cursor:pointer;
}
.cite-list { margin:8px 0 0; padding-left:18px; }
.cite-row  { margin:4px 0; font-size:.83rem; color:#c0c8d4; }
.cite-key  {
    display:inline-block; color:#fff; font-size:.7rem; font-weight:700;
    padding:1px 6px; border-radius:3px; margin-right:4px;
}

/* ================================================================ */
/*  RIGHT PANEL — confidence badge                                  */
/* ================================================================ */
.confidence-row {
    font-size:.8rem; color:#7b8794; margin-bottom:6px;
    display:flex; align-items:center; gap:6px;
}
.conf-badge {
    display:inline-block; padding:2px 10px; border-radius:4px;
    font-size:.75rem; font-weight:700; color:#fff;
}
.conf-high   { background:#22865a; }
.conf-medium { background:#b08a28; }
.conf-low    { background:#c0392b; }

/* ================================================================ */
/*  RIGHT PANEL — example & related-question chips                  */
/* ================================================================ */
.example-chip-row { margin-bottom:4px !important; }
.example-chip-row button,
.rq-chip-row button {
    font-size:.78rem !important;
    white-space:normal !important;
    text-align:left !important;
    line-height:1.3 !important;
}
.examples-hdr {
    font-size:.82rem; color:#5a6a7a; margin:4px 0 6px;
}
.rq-label {
    font-size:.8rem; color:#7b8794; margin-bottom:2px; display:block;
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# JS  (dark mode + auto-scroll conversation)
# ═══════════════════════════════════════════════════════════════════════════

DARK_JS = """
() => {
    document.documentElement.classList.add('dark');
    const obs = new MutationObserver(() => {
        document.querySelectorAll('.conv-scroll').forEach(el => {
            el.scrollTop = el.scrollHeight;
        });
    });
    setTimeout(() => {
        const app = document.querySelector('.gradio-container');
        if (app) obs.observe(app, {childList:true, subtree:true});
    }, 800);
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════════════

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    # pre-load companies (graceful fallback)
    try:
        company_data = load_companies()
    except Exception:
        company_data = []

    company_labels = [label for label, _ in company_data]
    company_map: dict[str, str] = {
        label: ticker for label, ticker in company_data
    }

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=DARK_CSS,
        js=DARK_JS,
        title="Research & Deal Intelligence",
    ) as app:

        # ── shared state ──────────────────────────────────────────────
        selected_ticker = gr.State(value=None)
        selected_doc_ids = gr.State(value=[])
        conv_history = gr.State(value=[])

        with gr.Row():
            # ==========================================================
            # LEFT PANEL  (~30 %)
            # ==========================================================
            with gr.Column(scale=3, min_width=320):

                gr.HTML(
                    '<div class="panel-header">'
                    "  <h2>Research &amp; Deal Intelligence</h2>"
                    '  <p class="subtitle">Powered by Databricks + Claude</p>'
                    "</div>"
                )

                company_dd = gr.Dropdown(
                    choices=company_labels,
                    label="Company",
                    info="Select a company to load documents",
                    filterable=True,
                )

                position_html = gr.HTML(value=_badge_empty())

                with gr.Tabs():
                    with gr.Tab("\U0001f4c1 Filings"):
                        filing_checks = gr.CheckboxGroup(
                            choices=[], label="Select filings", value=[],
                        )
                    with gr.Tab("\U0001f4ca Earnings"):
                        earnings_checks = gr.CheckboxGroup(
                            choices=[], label="Select transcripts", value=[],
                        )
                    with gr.Tab("\U0001f4f0 News"):
                        news_checks = gr.CheckboxGroup(
                            choices=[], label="Select articles", value=[],
                        )

                gr.Markdown("---")
                gr.File(
                    label="\U0001f4ce Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx", ".csv"],
                    interactive=True,
                    height=80,
                )
                gr.HTML(
                    '<p class="upload-hint">'
                    "Drop files here for analysis (coming soon)</p>"
                )

            # ==========================================================
            # RIGHT PANEL  (~70 %)
            # ==========================================================
            with gr.Column(scale=7, min_width=500):

                # -- conversation area ---------------------------------
                conv_html = gr.HTML(
                    value=render_conversation([]),
                    elem_classes=["conv-scroll"],
                )

                # -- citation panel ------------------------------------
                cite_html = gr.HTML(value="")

                # -- related-question chips (up to 3) ------------------
                rq_label = gr.HTML(
                    value="", visible=False,
                )
                with gr.Row(visible=False, elem_classes=["rq-chip-row"]) as rq_row:
                    rq_btn0 = gr.Button("", size="sm", variant="secondary",
                                        visible=False)
                    rq_btn1 = gr.Button("", size="sm", variant="secondary",
                                        visible=False)
                    rq_btn2 = gr.Button("", size="sm", variant="secondary",
                                        visible=False)

                # -- example-query chips -------------------------------
                examples_hdr = gr.HTML(
                    '<div class="examples-hdr">Try an example:</div>',
                    visible=True,
                )
                with gr.Row(visible=True,
                            elem_classes=["example-chip-row"]) as ex_row_a:
                    ex_btns_a = [
                        gr.Button(q, size="sm", variant="secondary",
                                  elem_classes=["example-chip"])
                        for q in EXAMPLE_QUERIES[:4]
                    ]
                with gr.Row(visible=True,
                            elem_classes=["example-chip-row"]) as ex_row_b:
                    ex_btns_b = [
                        gr.Button(q, size="sm", variant="secondary",
                                  elem_classes=["example-chip"])
                        for q in EXAMPLE_QUERIES[4:]
                    ]

                # -- chat input ----------------------------------------
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder=(
                            "Ask about financials, filings, positions\u2026"
                        ),
                        show_label=False,
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button(
                        "Send", variant="primary", scale=1, min_width=80,
                    )

        # ==============================================================
        # Event handlers — left panel
        # ==============================================================

        def on_company_select(choice: str | None):
            if not choice:
                empty = (
                    None,
                    _badge_empty(),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    [],
                    [],                              # clear conversation
                    render_conversation([]),
                    "",                              # clear citations
                    gr.update(visible=False),         # rq_label
                    gr.update(visible=False),         # rq_row
                    gr.update(visible=False),         # rq0
                    gr.update(visible=False),         # rq1
                    gr.update(visible=False),         # rq2
                    gr.update(visible=True),          # examples_hdr
                    gr.update(visible=True),          # ex_row_a
                    gr.update(visible=True),          # ex_row_b
                )
                return empty

            ticker = company_map.get(choice, "")

            badge = load_position_badge(ticker)

            f_rows = load_filings(ticker)
            f_ch = [(_filing_label(r), r["document_id"]) for r in f_rows]

            e_rows = load_earnings(ticker)
            e_ch = [(_earnings_label(r), r["document_id"]) for r in e_rows]

            n_rows = load_news(ticker)
            n_ch = [(_news_label(r), r["document_id"]) for r in n_rows]

            return (
                ticker,
                badge,
                gr.update(choices=f_ch, value=[]),
                gr.update(choices=e_ch, value=[]),
                gr.update(choices=n_ch, value=[]),
                [],
                [],                              # clear conversation
                render_conversation([]),
                "",                              # clear citations
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        company_dd.change(
            on_company_select,
            inputs=[company_dd],
            outputs=[
                selected_ticker, position_html,
                filing_checks, earnings_checks, news_checks,
                selected_doc_ids,
                conv_history, conv_html, cite_html,
                rq_label, rq_row, rq_btn0, rq_btn1, rq_btn2,
                examples_hdr, ex_row_a, ex_row_b,
            ],
        )

        def on_doc_select(f: list[str], e: list[str], n: list[str]) -> list[str]:
            return list(set((f or []) + (e or []) + (n or [])))

        for _cb in (filing_checks, earnings_checks, news_checks):
            _cb.change(
                on_doc_select,
                inputs=[filing_checks, earnings_checks, news_checks],
                outputs=[selected_doc_ids],
            )

        # ==============================================================
        # Event handlers — right panel (chat)
        # ==============================================================

        # Shared output list (everything the send generator touches)
        _send_outputs = [
            chat_input, conv_history, conv_html, cite_html,
            rq_label, rq_row, rq_btn0, rq_btn1, rq_btn2,
            examples_hdr, ex_row_a, ex_row_b,
        ]

        def _do_send(
            text: str,
            history: list[dict],
            ticker: str | None,
            doc_ids: list[str],
        ):
            """Generator: yield loading state, then final state."""
            text = (text or "").strip()
            if not text:
                yield (
                    "", history, render_conversation(history), "",
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                )
                return

            # --- yield 1: loading -----------------------------------
            history = history + [{"role": "user", "content": text}]
            loading_conv = render_conversation(history)
            # append a loading bubble
            loader = (
                '<div class="conv-row conv-asst">'
                '<div class="bubble bub-asst bub-loading">'
                '<span class="ld-dots">'
                "<span>.</span><span>.</span><span>.</span>"
                "</span> Analyzing with 14 research tools\u2026"
                "</div></div>"
            )
            if loading_conv.endswith("</div>"):
                loading_conv = loading_conv[:-6] + loader + "</div>"
            else:
                loading_conv += loader

            yield (
                gr.update(value="", interactive=False),   # chat_input
                history,                                  # conv_history
                loading_conv,                             # conv_html
                "",                                       # cite_html
                gr.update(visible=False),                 # rq_label
                gr.update(visible=False),                 # rq_row
                gr.update(visible=False),                 # rq0
                gr.update(visible=False),                 # rq1
                gr.update(visible=False),                 # rq2
                gr.update(visible=False),                 # examples_hdr
                gr.update(visible=False),                 # ex_row_a
                gr.update(visible=False),                 # ex_row_b
            )

            # --- call agent ----------------------------------------
            agent_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in history
                if m["role"] in ("user", "assistant")
            ]
            result = call_agent(agent_msgs, ticker, doc_ids or None)

            # --- yield 2: final ------------------------------------
            if result["error"]:
                history = history + [
                    {"role": "error", "content": result["error"]}
                ]
                yield (
                    gr.update(value="", interactive=True),
                    history,
                    render_conversation(history),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
                return

            content = result["content"]
            resp_html, cit_html, related = render_agent_response(content)

            history = history + [
                {"role": "assistant", "content": content, "html": resp_html}
            ]

            # related-question buttons
            rq_updates: list = []
            for i, btn_ref in enumerate([rq_btn0, rq_btn1, rq_btn2]):
                if i < len(related):
                    rq_updates.append(
                        gr.update(value=related[i], visible=True)
                    )
                else:
                    rq_updates.append(gr.update(visible=False))

            show_rq = len(related) > 0

            yield (
                gr.update(value="", interactive=True),    # chat_input
                history,                                  # conv_history
                render_conversation(history),             # conv_html
                cit_html,                                 # cite_html
                gr.update(                                # rq_label
                    value='<span class="rq-label">Related questions:</span>',
                    visible=show_rq,
                ),
                gr.update(visible=show_rq),               # rq_row
                rq_updates[0],                            # rq0
                rq_updates[1],                            # rq1
                rq_updates[2],                            # rq2
                gr.update(visible=False),                 # examples_hdr
                gr.update(visible=False),                 # ex_row_a
                gr.update(visible=False),                 # ex_row_b
            )

        # wire send button + enter key
        _send_inputs = [chat_input, conv_history,
                        selected_ticker, selected_doc_ids]

        send_btn.click(
            _do_send, inputs=_send_inputs, outputs=_send_outputs,
        )
        chat_input.submit(
            _do_send, inputs=_send_inputs, outputs=_send_outputs,
        )

        # wire example-query chips
        def _make_chip_handler(query_text: str):
            def _handler(history, ticker, doc_ids):
                yield from _do_send(query_text, history, ticker, doc_ids)
            return _handler

        for btn in ex_btns_a + ex_btns_b:
            btn.click(
                _make_chip_handler(btn.value),
                inputs=[conv_history, selected_ticker, selected_doc_ids],
                outputs=_send_outputs,
            )

        # wire related-question chips
        def _rq_handler(rq_text, history, ticker, doc_ids):
            yield from _do_send(rq_text, history, ticker, doc_ids)

        for rq_btn in (rq_btn0, rq_btn1, rq_btn2):
            rq_btn.click(
                _rq_handler,
                inputs=[rq_btn, conv_history,
                        selected_ticker, selected_doc_ids],
                outputs=_send_outputs,
            )

    return app


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
