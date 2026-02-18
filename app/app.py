"""
Research & Deal Intelligence — Gradio application.

Two-panel dark-themed layout:

  LEFT  (~30 %) — company selector, position badge, document library, upload
  RIGHT (~70 %) — chat input, agent responses with structured sections,
                   citation panel, related-question chips, conversation history,
                   export (Copy / PDF)

On load the app pre-selects **NVDA** and shows a sample Q&A exchange so the
user immediately sees a fully-rendered response.

Agent endpoint : ks_factset_research_v3_agent  (MLflow model serving)
Data           : Databricks SQL  (ks_factset_research_v3 catalog)

Env vars required:
    DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
"""

from __future__ import annotations

import datetime
import os
import re
import tempfile
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

DEFAULT_TICKER = "NVDA"

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
# Sample Q&A  (pre-populated for NVDA on first load)
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_QUERY = "What are the key risk factors in NVDA's latest 10-K?"

SAMPLE_RESPONSE = """\
## Summary

NVIDIA's 2024 10-K identifies supply chain concentration, geopolitical export \
controls, and rapid technology obsolescence as its top risk factors. The \
company's heavy reliance on TSMC for advanced chip fabrication and increasing \
US-China trade restrictions present material threats to revenue growth [1] [2].

## Analysis

**Supply Chain Concentration:** NVIDIA depends on TSMC for 100% of its \
leading-edge GPU production (5nm and below). Any disruption could materially \
impact revenue, as alternative foundry capacity at comparable nodes is \
extremely limited [1].

**Export Controls:** The US Bureau of Industry and Security expanded \
restrictions on AI chip exports to China in October 2023. NVIDIA disclosed \
that **$5.0B** in annual Data Center revenue (approximately **18.5%** of \
total $27.0B) is at risk from current and potential future export \
controls [1] [2].

**Technology Obsolescence:** The AI accelerator market is intensifying with \
competition from AMD MI300X, Intel Gaudi, Google TPU v5, and custom ASICs \
from hyperscalers. NVIDIA must maintain its CUDA software moat and \
generational performance leadership to sustain its >80% market share [2].

**Customer Concentration:** The top 5 customers (Microsoft, Meta, Amazon, \
Google, Oracle) account for approximately **46%** of Data Center \
revenue [1].

## Position Context

NVIDIA is in our position book. **Total notional: $320.0M** across 6 \
positions on 4 desks. Largest exposure is Equity Trading ($175.0M notional, \
Momentum + Index Arb strategies). Equity Derivatives holds $45.0M in options \
(net long vol). **1 active risk flag:** concentration flag due to >$250M \
single-name exposure.

## Calculations

- Export control revenue at risk: $5.0B / $27.0B total = **18.5%** of revenue
- Customer concentration (top 5): ~46% of Data Center revenue
- Position concentration: $320.0M total notional = flagged

## Sources

- [1] NVDA 10-K 2024 -- Item 1A Risk Factors (filed Feb 2024)
- [2] NVDA 10-K 2024 -- Business Overview, Competition section
- [3] NVDA Earnings Q4 2024 -- Management commentary on China exposure

## Confidence

HIGH

## Related Questions

- How do NVIDIA's leverage ratios compare to AMD and Intel?
- What is the P&L trend on our NVDA Equity Derivatives positions?
- Has NVIDIA's consensus EPS estimate been revised since the latest export control announcement?
"""

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
    """Minimal markdown -> HTML (bold, italic, code, lists, paragraphs)."""
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
        _bullet_re = re.compile(r'^[-*0-9.]+\s*')
        if all(re.match(r"^[\-\*\d.]+\s", ln.strip()) for ln in lines if ln.strip()):
            items = "".join(
                "<li>" + _bullet_re.sub("", ln.strip()) + "</li>"
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

# ═══════════════════════════════════════════════════════════════════════════
# PDF export
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_latin1(text: str) -> str:
    """Replace common Unicode chars for Latin-1 PDF compatibility."""
    subs = {
        "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2691": "[!]",
        "\u25b6": ">", "\u25bc": "v", "\u2192": "->",
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf(history: list[dict], ticker: str | None) -> str | None:
    """Create a PDF report from conversation history. Returns file path."""
    try:
        from fpdf import FPDF  # fpdf2
    except ImportError:
        return None

    if not history:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- title ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Research & Deal Intelligence Report", ln=True)
    if ticker:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, _sanitize_latin1(f"Company: {ticker}"), ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(
        0, 6,
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True,
    )
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)

    # --- conversation ---
    for msg in history:
        if msg["role"] == "user":
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 80, 160)
            pdf.cell(0, 7, "Question:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 5, _sanitize_latin1(msg["content"]))
            pdf.ln(3)

        elif msg["role"] == "assistant":
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 7, "Analysis:", ln=True)
            sections = _parse_sections(msg["content"])
            for sec_name in _SECTION_NAMES:
                key = sec_name.title()
                if key not in sections:
                    continue
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 6, _sanitize_latin1(sec_name), ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 5, _sanitize_latin1(sections[key]))
                pdf.ln(2)
            pdf.ln(5)

    path = os.path.join(tempfile.gettempdir(), "research_report.pdf")
    pdf.output(path)
    return path

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
.upload-hint { text-align:center; font-size:.75rem; color:#5a6a7a; margin-top:2px; }

/* ================================================================ */
/*  RIGHT PANEL                                                     */
/* ================================================================ */
.right-panel { border-left:1px solid #1f2937; padding-left:12px !important; }

/* -- conversation ------------------------------------------------ */
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

/* -- response sections ------------------------------------------- */
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

/* -- citations --------------------------------------------------- */
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

/* -- confidence badge -------------------------------------------- */
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

/* -- chips ------------------------------------------------------- */
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

/* -- export bar -------------------------------------------------- */
.export-bar { margin-top:4px !important; }
.export-bar button {
    font-size:.78rem !important;
    min-width:70px !important;
}
.copy-toast {
    font-size:.75rem; color:#50c878; display:inline-block;
    margin-left:6px; opacity:0; transition:opacity .3s;
}
.copy-toast.show { opacity:1; }
"""

# ═══════════════════════════════════════════════════════════════════════════
# JS  (dark mode + auto-scroll + clipboard helper)
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

COPY_JS = """
(history) => {
    if (!history || !history.length) return;
    const lines = history
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => (m.role === 'user' ? 'Q: ' : 'A: ') + m.content);
    const text = lines.join('\\n\\n');
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
    } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════════════

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    # ── pre-load companies ────────────────────────────────────────────
    try:
        company_data = load_companies()
    except Exception:
        company_data = []

    company_labels = [label for label, _ in company_data]
    company_map: dict[str, str] = {
        label: ticker for label, ticker in company_data
    }

    # ── find default ticker (NVDA) ────────────────────────────────────
    default_label: str | None = None
    default_ticker: str | None = None
    for label, ticker in company_data:
        if ticker == DEFAULT_TICKER:
            default_label = label
            default_ticker = ticker
            break

    # ── pre-load left-panel data for default ticker ───────────────────
    init_badge = _badge_empty()
    init_f_choices: list[tuple[str, str]] = []
    init_e_choices: list[tuple[str, str]] = []
    init_n_choices: list[tuple[str, str]] = []

    if default_ticker:
        try:
            init_badge = load_position_badge(default_ticker)
            init_f_choices = [
                (_filing_label(r), r["document_id"])
                for r in load_filings(default_ticker)
            ]
            init_e_choices = [
                (_earnings_label(r), r["document_id"])
                for r in load_earnings(default_ticker)
            ]
            init_n_choices = [
                (_news_label(r), r["document_id"])
                for r in load_news(default_ticker)
            ]
        except Exception:
            pass

    # ── pre-build sample Q&A ──────────────────────────────────────────
    sample_html, sample_cite, sample_rqs = render_agent_response(SAMPLE_RESPONSE)
    init_history: list[dict] = [
        {"role": "user", "content": SAMPLE_QUERY},
        {"role": "assistant", "content": SAMPLE_RESPONSE, "html": sample_html},
    ]
    init_conv = render_conversation(init_history)

    has_sample = True  # example chips hidden, rq chips shown

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=DARK_CSS,
        js=DARK_JS,
        title="Research & Deal Intelligence",
    ) as app:

        # ── shared state ──────────────────────────────────────────────
        selected_ticker = gr.State(value=default_ticker)
        selected_doc_ids = gr.State(value=[])
        conv_history = gr.State(value=init_history)

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
                    value=default_label,
                    label="Company",
                    info="Select a company to load documents",
                    filterable=True,
                )

                position_html = gr.HTML(value=init_badge)

                with gr.Tabs():
                    with gr.Tab("\U0001f4c1 Filings"):
                        filing_checks = gr.CheckboxGroup(
                            choices=init_f_choices,
                            label="Select filings", value=[],
                        )
                    with gr.Tab("\U0001f4ca Earnings"):
                        earnings_checks = gr.CheckboxGroup(
                            choices=init_e_choices,
                            label="Select transcripts", value=[],
                        )
                    with gr.Tab("\U0001f4f0 News"):
                        news_checks = gr.CheckboxGroup(
                            choices=init_n_choices,
                            label="Select articles", value=[],
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
            with gr.Column(scale=7, min_width=500,
                           elem_classes=["right-panel"]):

                # -- conversation area ---------------------------------
                conv_html = gr.HTML(
                    value=init_conv,
                    elem_classes=["conv-scroll"],
                )

                # -- export bar ----------------------------------------
                with gr.Row(elem_classes=["export-bar"]):
                    copy_btn = gr.Button(
                        "Copy", size="sm", variant="secondary",
                        scale=0, min_width=70,
                    )
                    pdf_btn = gr.Button(
                        "PDF", size="sm", variant="secondary",
                        scale=0, min_width=70,
                    )
                    pdf_download = gr.File(
                        visible=False, label="Download",
                        scale=1,
                    )

                # -- citation panel ------------------------------------
                cite_html = gr.HTML(value=sample_cite)

                # -- related-question chips (up to 3) ------------------
                rq_label_html = gr.HTML(
                    value='<span class="rq-label">Related questions:</span>',
                    visible=has_sample and len(sample_rqs) > 0,
                )
                with gr.Row(
                    visible=has_sample and len(sample_rqs) > 0,
                    elem_classes=["rq-chip-row"],
                ) as rq_row:
                    rq_btn0 = gr.Button(
                        sample_rqs[0] if len(sample_rqs) > 0 else "",
                        size="sm", variant="secondary",
                        visible=len(sample_rqs) > 0,
                    )
                    rq_btn1 = gr.Button(
                        sample_rqs[1] if len(sample_rqs) > 1 else "",
                        size="sm", variant="secondary",
                        visible=len(sample_rqs) > 1,
                    )
                    rq_btn2 = gr.Button(
                        sample_rqs[2] if len(sample_rqs) > 2 else "",
                        size="sm", variant="secondary",
                        visible=len(sample_rqs) > 2,
                    )

                # -- example-query chips (hidden when sample Q&A) ------
                examples_hdr = gr.HTML(
                    '<div class="examples-hdr">Try an example:</div>',
                    visible=not has_sample,
                )
                with gr.Row(
                    visible=not has_sample,
                    elem_classes=["example-chip-row"],
                ) as ex_row_a:
                    ex_btns_a = [
                        gr.Button(q, size="sm", variant="secondary")
                        for q in EXAMPLE_QUERIES[:4]
                    ]
                with gr.Row(
                    visible=not has_sample,
                    elem_classes=["example-chip-row"],
                ) as ex_row_b:
                    ex_btns_b = [
                        gr.Button(q, size="sm", variant="secondary")
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
            # 17 outputs
            if not choice:
                return (
                    None,                                     # selected_ticker
                    _badge_empty(),                           # position_html
                    gr.update(choices=[], value=[]),           # filing_checks
                    gr.update(choices=[], value=[]),           # earnings_checks
                    gr.update(choices=[], value=[]),           # news_checks
                    [],                                       # selected_doc_ids
                    [],                                       # conv_history
                    render_conversation([]),                   # conv_html
                    "",                                       # cite_html
                    gr.update(visible=False),                 # rq_label_html
                    gr.update(visible=False),                 # rq_row
                    gr.update(visible=False),                 # rq0
                    gr.update(visible=False),                 # rq1
                    gr.update(visible=False),                 # rq2
                    gr.update(visible=True),                  # examples_hdr
                    gr.update(visible=True),                  # ex_row_a
                    gr.update(visible=True),                  # ex_row_b
                )

            ticker = company_map.get(choice, "")
            badge = load_position_badge(ticker)
            f_ch = [(_filing_label(r), r["document_id"])
                    for r in load_filings(ticker)]
            e_ch = [(_earnings_label(r), r["document_id"])
                    for r in load_earnings(ticker)]
            n_ch = [(_news_label(r), r["document_id"])
                    for r in load_news(ticker)]

            return (
                ticker,                                   # selected_ticker
                badge,                                    # position_html
                gr.update(choices=f_ch, value=[]),        # filing_checks
                gr.update(choices=e_ch, value=[]),        # earnings_checks
                gr.update(choices=n_ch, value=[]),        # news_checks
                [],                                       # selected_doc_ids
                [],                                       # conv_history
                render_conversation([]),                   # conv_html
                "",                                       # cite_html
                gr.update(visible=False),                 # rq_label_html
                gr.update(visible=False),                 # rq_row
                gr.update(visible=False),                 # rq0
                gr.update(visible=False),                 # rq1
                gr.update(visible=False),                 # rq2
                gr.update(visible=True),                  # examples_hdr
                gr.update(visible=True),                  # ex_row_a
                gr.update(visible=True),                  # ex_row_b
            )

        company_dd.change(
            on_company_select,
            inputs=[company_dd],
            outputs=[
                selected_ticker, position_html,
                filing_checks, earnings_checks, news_checks,
                selected_doc_ids,
                conv_history, conv_html, cite_html,
                rq_label_html, rq_row, rq_btn0, rq_btn1, rq_btn2,
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

        _send_outputs = [                         # 12 items
            chat_input, conv_history, conv_html, cite_html,
            rq_label_html, rq_row, rq_btn0, rq_btn1, rq_btn2,
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
                # 12 no-op outputs
                yield (
                    "", history, render_conversation(history), "",
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                )
                return

            # --- yield 1: loading ---
            history = history + [{"role": "user", "content": text}]
            loading_conv = render_conversation(history)
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
                gr.update(visible=False),                 # rq_label_html
                gr.update(visible=False),                 # rq_row
                gr.update(visible=False),                 # rq0
                gr.update(visible=False),                 # rq1
                gr.update(visible=False),                 # rq2
                gr.update(visible=False),                 # examples_hdr
                gr.update(visible=False),                 # ex_row_a
                gr.update(visible=False),                 # ex_row_b
            )

            # --- call agent ---
            agent_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in history
                if m["role"] in ("user", "assistant")
            ]
            result = call_agent(agent_msgs, ticker, doc_ids or None)

            # --- yield 2: final ---
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

            rq_updates: list = []
            for i in range(3):
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
                gr.update(                                # rq_label_html
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

        # ==============================================================
        # Event handlers — export
        # ==============================================================

        # Copy (JS-only, no server round-trip)
        copy_btn.click(
            fn=None,
            inputs=[conv_history],
            outputs=[],
            js=COPY_JS,
        )

        # PDF
        def _on_pdf(history, ticker):
            path = generate_pdf(history, ticker)
            if path and os.path.exists(path):
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)

        pdf_btn.click(
            _on_pdf,
            inputs=[conv_history, selected_ticker],
            outputs=[pdf_download],
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
