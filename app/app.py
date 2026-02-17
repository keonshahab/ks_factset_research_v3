"""
Left panel for Research & Deal Intelligence.

~30 % width dark-themed sidebar built with Gradio Blocks and the
Databricks SQL connector.  Populates:

- Company selector          (gold.v_active_companies)
- Position badge            (gold.position_exposures + gold.position_risk_flags)
- Document library tabs     (demo.filing_documents, demo.earnings_documents,
                             demo.news_chunks grouped by document_id)
- Upload area               (cosmetic placeholder)

Exposes two pieces of shared state for downstream panels:
    selected_ticker   – str | None
    selected_doc_ids  – list[str]

Env vars required:
    DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
from databricks import sql as dbsql

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

CATALOG = "ks_factset_research_v3"
DEMO = f"{CATALOG}.demo"
GOLD = f"{CATALOG}.gold"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# CSS (dark theme)
# ---------------------------------------------------------------------------

DARK_CSS = """
/* ---- force dark background ---- */
.gradio-container {
    background: #0f0f1a !important;
}

/* ---- panel header ---- */
.panel-header {
    text-align: center;
    padding: 16px 8px 8px;
}
.panel-header h2 {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: #e0e0e0;
    letter-spacing: 0.02em;
}
.panel-header .subtitle {
    margin: 4px 0 0;
    font-size: 0.78rem;
    color: #7b8794;
    font-weight: 400;
}

/* ---- position badge ---- */
.position-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    background: #16213e;
    border: 1px solid #1f3460;
    font-size: 0.92rem;
    flex-wrap: wrap;
}
.position-badge.empty {
    color: #6b7b8d;
    font-style: italic;
}
.position-badge .notional {
    font-weight: 700;
    font-size: 1.05rem;
    color: #4a9eff;
}
.position-badge .pos-meta {
    color: #a0aec0;
}
.position-badge .flag-count {
    background: #e94560;
    color: #fff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ---- tabs ---- */
.tab-nav button {
    font-size: 0.85rem !important;
}

/* ---- upload hint ---- */
.upload-hint {
    text-align: center;
    font-size: 0.75rem;
    color: #5a6a7a;
    margin-top: 2px;
}

/* ---- right placeholder ---- */
.right-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 600px;
    color: #4a5568;
    font-size: 1rem;
    border-left: 1px solid #1f2937;
}
"""

# JS snippet: force Gradio into dark mode on load
DARK_JS = """
() => {
    document.documentElement.classList.add('dark');
}
"""

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    # Pre-load company list (graceful fallback if DB unavailable)
    try:
        company_data = load_companies()
    except Exception:
        company_data = []

    company_labels = [label for label, _ in company_data]
    company_map: dict[str, str] = {label: ticker for label, ticker in company_data}

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=DARK_CSS,
        js=DARK_JS,
        title="Research & Deal Intelligence",
    ) as app:

        # ---- shared state ----
        selected_ticker = gr.State(value=None)
        selected_doc_ids = gr.State(value=[])

        with gr.Row():
            # ============================================================
            # LEFT PANEL  (~30 %)
            # ============================================================
            with gr.Column(scale=3, min_width=320):

                # -- header --
                gr.HTML(
                    '<div class="panel-header">'
                    "  <h2>Research &amp; Deal Intelligence</h2>"
                    '  <p class="subtitle">Powered by Databricks + Claude</p>'
                    "</div>"
                )

                # -- company selector --
                company_dd = gr.Dropdown(
                    choices=company_labels,
                    label="Company",
                    info="Select a company to load documents",
                    filterable=True,
                )

                # -- position badge --
                position_html = gr.HTML(value=_badge_empty())

                # -- document library (3 tabs) --
                with gr.Tabs():
                    with gr.Tab("\U0001f4c1 Filings"):
                        filing_checks = gr.CheckboxGroup(
                            choices=[], label="Select filings", value=[]
                        )
                    with gr.Tab("\U0001f4ca Earnings"):
                        earnings_checks = gr.CheckboxGroup(
                            choices=[], label="Select transcripts", value=[]
                        )
                    with gr.Tab("\U0001f4f0 News"):
                        news_checks = gr.CheckboxGroup(
                            choices=[], label="Select articles", value=[]
                        )

                # -- upload area (cosmetic) --
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
                    "Drop files here for analysis (coming soon)"
                    "</p>"
                )

            # ============================================================
            # RIGHT PANEL placeholder (~70 %)
            # ============================================================
            with gr.Column(scale=7, min_width=500):
                gr.HTML(
                    '<div class="right-placeholder">'
                    "<p>Chat and analysis panel will appear here.</p>"
                    "</div>"
                )

        # ----------------------------------------------------------------
        # Event handlers
        # ----------------------------------------------------------------

        def on_company_select(choice: str | None):
            """Refresh all widgets when the user picks a company."""
            if not choice:
                return (
                    None,
                    _badge_empty(),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    [],
                )

            ticker = company_map.get(choice, "")

            badge = load_position_badge(ticker)

            f_rows = load_filings(ticker)
            f_choices = [(_filing_label(r), r["document_id"]) for r in f_rows]

            e_rows = load_earnings(ticker)
            e_choices = [(_earnings_label(r), r["document_id"]) for r in e_rows]

            n_rows = load_news(ticker)
            n_choices = [(_news_label(r), r["document_id"]) for r in n_rows]

            return (
                ticker,
                badge,
                gr.update(choices=f_choices, value=[]),
                gr.update(choices=e_choices, value=[]),
                gr.update(choices=n_choices, value=[]),
                [],
            )

        company_dd.change(
            on_company_select,
            inputs=[company_dd],
            outputs=[
                selected_ticker,
                position_html,
                filing_checks,
                earnings_checks,
                news_checks,
                selected_doc_ids,
            ],
        )

        def on_doc_select(
            filings_sel: list[str],
            earnings_sel: list[str],
            news_sel: list[str],
        ) -> list[str]:
            """Merge selections from all three tabs into one list."""
            return list(
                set((filings_sel or []) + (earnings_sel or []) + (news_sel or []))
            )

        for _cb in (filing_checks, earnings_checks, news_checks):
            _cb.change(
                on_doc_select,
                inputs=[filing_checks, earnings_checks, news_checks],
                outputs=[selected_doc_ids],
            )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
