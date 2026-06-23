import sys
import os
import time
import re

import streamlit as st
import plotly.graph_objects as go


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import download_reports
from ingest import ingest_documents

# Download PDFs automatically
download_reports()

# Create embeddings if missing
if not os.path.exists("chroma_db"):
    print("Creating embeddings...")
    ingest_documents()

st.set_page_config(
    page_title="FinSight AI — Financial Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:#0D0F14; --surface:#141720; --border:#1E2438;
    --accent:#4F8EF7; --accent2:#22D3A5; --warn:#F5A623;
    --text:#E4E8F4; --muted:#5A6480;
    --sans:'DM Sans',sans-serif; --serif:'DM Serif Display',Georgia,serif;
}
html,body,[class*="css"]{ font-family:var(--sans); background-color:var(--bg); color:var(--text); }
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{background:transparent!important;}

/* FIX 1 — kill keyboard_double text in collapse button */
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]{
    font-size:0!important; width:0!important; height:0!important;
    overflow:hidden!important; color:transparent!important;
}
[data-testid="stSidebarCollapseButton"] button::after{
    content:'❮'; font-size:16px; color:#5A6480; line-height:1;
}

/* FIX 2 — sidebar open button: far left, vertically centered in header */
[data-testid="stHeader"]{
    background:transparent!important;
    z-index:1000!important;
    top:0!important;
    height:62px!important;
    display:flex!important;
    align-items:center!important;
    padding:0 0 0 12px!important;
    pointer-events:none!important;
}
[data-testid="stHeader"] [data-testid="stIconMaterial"]{
    font-size:0!important; width:0!important; height:0!important;
    overflow:hidden!important; color:transparent!important;
}
[data-testid="stHeader"] button{
    pointer-events:all!important;
    background:#1A1E2E!important;
    border:1px solid #252A3A!important;
    border-radius:8px!important;
    width:32px!important;
    height:32px!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    cursor:pointer!important;
    transition:all 0.18s!important;
    padding:0!important;
}
[data-testid="stHeader"] button:hover{
    background:#252A3A!important;
    border-color:#4F8EF7!important;
}
[data-testid="stHeader"] button::after{
    content:'›';
    font-size:22px;
    color:#6B7699;
    line-height:1;
    margin-top:-2px;
}
[data-testid="stHeader"] button:hover::after{
    color:#4F8EF7;
}

/* FIX 3 — fixed header bar, flexible with sidebar */
.fixed-header{
    position:fixed; top:0; right:0; left:300px; z-index:998;
    background:#0D0F14; border-bottom:1px solid #1A1E2E;
    padding:0.9rem 2rem 0.75rem;
    padding-left:2rem;
    transition:left 0.3s ease;
}
body:has([data-testid="stSidebar"][aria-expanded="false"]) .fixed-header{
    left:0;
    padding-left:72px;
}
[data-testid="stMainBlockContainer"]{ padding-top:80px!important; }

/* Sidebar */
[data-testid="stSidebar"]{ background:#0F1118!important; border-right:1px solid #1A1E2E!important; }
[data-testid="stSidebar"]*{ font-family:var(--sans)!important; }

/* Chat input */
[data-testid="stChatInput"] textarea{
    background:#141720!important; border:1px solid #1E2438!important;
    border-radius:14px!important; color:var(--text)!important;
    font-family:var(--sans)!important; font-size:14px!important; padding:14px 16px!important;
}
[data-testid="stChatInput"] textarea:focus{
    border-color:var(--accent)!important; box-shadow:0 0 0 3px rgba(79,142,247,0.12)!important;
}
[data-testid="stChatInput"] textarea::placeholder{ color:#3A4060!important; }

/* Chat messages */
[data-testid="stChatMessage"]{ background:transparent!important; border:none!important; padding:0!important; }
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"]{ display:none!important; }

/* Buttons */
.stButton>button{
    background:#141720!important; border:1px solid #1E2438!important; color:#7A84A8!important;
    border-radius:8px!important; font-family:var(--sans)!important; font-size:12px!important;
    transition:all 0.18s!important; text-align:left!important; padding:6px 12px!important;
}
.stButton>button:hover{ border-color:var(--accent)!important; color:var(--accent)!important; background:rgba(79,142,247,0.06)!important; }
hr{ border-color:#1A1E2E!important; margin:0.75rem 0!important; }
::-webkit-scrollbar{width:5px;} ::-webkit-scrollbar-track{background:transparent;} ::-webkit-scrollbar-thumb{background:#1E2438;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAZY IMPORT
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2")
    from main import run, LATEST_QUARTER
    from src.utils import extract_revenue
    from src.embedder import search
    return run, LATEST_QUARTER, extract_revenue, search

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
QTYPE_META = {
    "simple_qa":     {"label": "Q&A",       "color": "#4F8EF7", "icon": "💬"},
    "comparison":    {"label": "Comparison", "color": "#22D3A5", "icon": "⚖️"},
    "reasoning":     {"label": "Reasoning",  "color": "#F5A623", "icon": "🔍"},
    "summarization": {"label": "Summary",    "color": "#A78BFA", "icon": "📋"},
    "trend":         {"label": "Trend",      "color": "#FB7185", "icon": "📈"},
    "invalid":       {"label": "Invalid",    "color": "#5A6480", "icon": "⚠️"},
}

QUICK_TOPICS = [
    ("Revenue",    "What is Infosys Q4 revenue?"),
    ("Margins",    "What is Infosys Q4 EBIT margin?"),
    ("Deal wins",  "What are Infosys Q4 large deal wins?"),
    ("Headcount",  "What is Infosys employee count in Q4?"),
    ("Guidance",   "What is Infosys FY26 revenue guidance?"),
    ("YoY growth", "Compare Infosys Q4 revenue year over year"),
    ("Segments",   "What are Infosys revenue segments in Q4?"),
    ("Apple Q2",   "Summarize Apple Q2 earnings"),
]

# Real Infosys quarterly revenue (USD Million) — drives the sidebar mini trend
INFOSYS_TREND = [
    ("Q1", 4941),
    ("Q2", 5076),
    ("Q3", 5099),
    ("Q4", 5040),
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def badge(qtype: str) -> str:
    m = QTYPE_META.get(qtype, QTYPE_META["invalid"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:20px;'
        f'font-size:10px;font-weight:600;letter-spacing:0.3px;'
        f'background:{m["color"]}20;color:{m["color"]};'
        f'border:1px solid {m["color"]}35;">'
        f'{m["icon"]} {m["label"]}</span>'
    )


def build_trend_chart(trend_text: str):
    pattern = r"(Q[1-4]_\d{4}):\s*\$([\d,]+)M"
    matches = re.findall(pattern, trend_text)
    unique = {}
    for q, v in matches:
        unique[q] = v
    matches = list(unique.items())
    if len(matches) < 2:
        return None
    quarters = [m[0].replace("_", " ") for m in matches]
    values   = [float(m[1].replace(",", "")) for m in matches]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=quarters, y=values,
        mode="lines+markers+text",
        line=dict(color="#4F8EF7", width=2.5),
        marker=dict(size=7, color="#4F8EF7", line=dict(color="#0D0F14", width=2)),
        text=[f"${v:,.0f}M" for v in values],
        textposition="top center",
        textfont=dict(size=11, color="#C0CAE8"),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.07)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#5A6480", size=11),
        margin=dict(l=10, r=10, t=24, b=10),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#5A6480")),
        yaxis=dict(showgrid=True, gridcolor="#1A1E2E", zeroline=False,
                   tickprefix="$", ticksuffix="M", tickfont=dict(color="#5A6480")),
        hoverlabel=dict(bgcolor="#141720", font_color="#E4E8F4", bordercolor="#1E2438"),
        height=240,
    )
    return fig


def build_comparison_chart(result: str):
    pattern = r"(Q[1-4][\w\s]*?)[:\s]+\$?([\d,]+)M"
    matches = re.findall(pattern, result, re.IGNORECASE)
    if len(matches) < 2:
        return None
    labels = [m[0].strip()[:22] for m in matches]
    values = [float(m[1].replace(",", "")) for m in matches]
    colors = ["#4F8EF7", "#22D3A5", "#A78BFA", "#FB7185"]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        text=[f"${v:,.0f}M" for v in values],
        textposition="outside",
        textfont=dict(color="#C0CAE8", size=11),
        marker_color=[colors[i % len(colors)] for i in range(len(values))],
        marker_line_width=0,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#5A6480", size=11),
        margin=dict(l=10, r=10, t=24, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(color="#5A6480")),
        yaxis=dict(showgrid=True, gridcolor="#1A1E2E",
                   tickprefix="$", ticksuffix="M", tickfont=dict(color="#5A6480")),
        height=230,
    )
    return fig


def render_sources(sources: list):
    if not sources:
        return
    seen, unique = set(), []
    for s in sources:
        m = s.get("metadata", s)
        key = f"{m.get('company','')}_{m.get('quarter','')}_{m.get('section','')}"
        if key not in seen:
            seen.add(key)
            unique.append(m)
    SECTION_LABELS = {"revenue": "Revenue", "mda": "MD&A", "general": "General", "summary": "Summary"}
    COMPANY_ICONS  = {"infosys": "💼", "apple": "🍎"}
    tags_html = ""
    for m in unique[:4]:
        company = m.get("company", "").title()
        quarter = m.get("quarter", "").replace("_", " ")
        section = SECTION_LABELS.get(m.get("section", ""), m.get("section", "").title())
        icon    = COMPANY_ICONS.get(m.get("company", "").lower(), "📄")
        tags_html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'background:#0D1018;border:1px solid #1A1E2E;border-radius:6px;'
            f'padding:3px 10px;font-size:11px;color:#5A6480;margin:2px 4px 2px 0;">'
            f'{icon} <span style="color:#8A94B8;font-weight:500;">{company} {quarter}</span>'
            f' · {section}</span>'
        )
    st.markdown(
        f'<div style="margin-top:12px;">'
        f'<div style="font-size:9px;color:#3A4060;letter-spacing:1.2px;'
        f'text-transform:uppercase;font-weight:500;margin-bottom:6px;">Sources</div>'
        f'<div>{tags_html}</div></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# RENDER ANSWER
# ─────────────────────────────────────────────
def render_answer(result: str, qtype: str, elapsed: float, sources: list = []):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
        f'<div style="width:28px;height:28px;border-radius:8px;background:#141720;'
        f'border:1px solid #1E2438;display:flex;align-items:center;'
        f'justify-content:center;font-size:13px;flex-shrink:0;">📊</div>'
        f'{badge(qtype)}'
        f'<span style="font-size:10px;color:#2A3050;margin-left:2px;">{elapsed:.1f}s</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if qtype == "trend":
        fig = build_trend_chart(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"trend_{elapsed}_{hash(result)}")
    elif qtype == "comparison":
        fig = build_comparison_chart(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"comp_{elapsed}_{hash(result)}")

    def md_to_html(text):
        lines_raw = text.strip().splitlines()
        clean_lines = []
        for ln in lines_raw:
            s = ln.strip()
            if re.match(r"^(Quarter|Sources?|Source File?|Section)\s*:", s, re.IGNORECASE):
                continue
            clean_lines.append(ln)
        text = "\n".join(clean_lines).strip()

        text = re.sub(r"\s*\([^)]*(?:Quarter|Source|Section)[^)]*\)", "", text)
        text = re.sub(r",?\s*Quarter:\s*[\w_]+", "", text)
        text = re.sub(r",?\s*Section:\s*[\w_]+", "", text)
        text = re.sub(r",?\s*Source(?:s)?:\s*[^\n,)]+", "", text)
        text = re.sub(r"\s*\|\s*Quarter:[^|)\n]+", "", text)
        text = re.sub(r"\s*\|\s*Section:[^|)\n]+", "", text)

        HEADING_STYLE = (
            "font-size:10px;font-weight:700;color:#4F8EF7;"
            "letter-spacing:1.2px;text-transform:uppercase;"
            "margin:16px 0 7px;padding-bottom:5px;"
            "border-bottom:1px solid #1E2438;"
        )

        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if re.match(r"^(Quarter|Source|Source File)\s*:", line, re.IGNORECASE):
                continue

            if re.match(r"^\*\*.+\*\*:?\s*$", line):
                heading = re.sub(r"\*\*(.+?)\*\*", r"\1", line).rstrip(":").strip()
                out.append("<div style=\"" + HEADING_STYLE + "\">" + heading + "</div>")
                continue

            if (re.match(r"^[A-Za-z][A-Za-z\s\-]+:$", line)
                    and len(line) < 45
                    and "$" not in line and "%" not in line):
                heading = line.rstrip(":")
                out.append("<div style=\"" + HEADING_STYLE + "\">" + heading + "</div>")
                continue

            line = re.sub(r"\*\*(.+?)\*\*", r"<strong style=\"color:#E4E8F4;\">\1</strong>", line)

            if re.match(r"^\d+\.\s*", line) and re.match(r"^\d+\.\s*\S", line):
                cl = re.sub(r"^\d+\.\s*", "", line)
                num = re.match(r"^(\d+)\.", line).group(1)
                out.append(
                    "<div style=\"display:flex;gap:9px;margin:5px 0;align-items:flex-start;\">"
                    "<span style=\"color:#4F8EF7;flex-shrink:0;font-weight:600;min-width:18px;\">"
                    + num + ".</span>"
                    "<span style=\"flex:1;\">" + cl + "</span></div>"
                )
                continue

            if re.match(r"^[-•]\s+", line):
                cl = re.sub(r"^[-•]\s+", "", line)
                out.append(
                    "<div style=\"display:flex;gap:9px;margin:5px 0;align-items:flex-start;\">"
                    "<span style=\"color:#4F8EF7;flex-shrink:0;font-size:15px;line-height:1.5;\">•</span>"
                    "<span style=\"flex:1;\">" + cl + "</span></div>"
                )
                continue

            out.append("<p style=\"margin:5px 0;color:#C0CAE8;\">" + line + "</p>")

        return "\n".join(out)

    formatted = md_to_html(result)
    st.markdown(
        "<div style=\"background:#141720;border:1px solid #1E2438;"
        "border-radius:4px 16px 16px 16px;padding:1.25rem 1.5rem;"
        "font-size:13.5px;line-height:1.8;color:#C0CAE8;\">"
        + formatted + "</div>",
        unsafe_allow_html=True
    )
    render_sources(sources)
    st.markdown(
        f'<div style="margin-top:8px;text-align:right;">'
        f'<span style="font-size:10px;color:#2A3050;">'
        f'{len(sources)} source{"s" if len(sources)!=1 else ""} used</span></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# STICKY HEADER
# ─────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="fixed-header" id="fh">
        <div style="display:flex;align-items:baseline;gap:14px;">
            <span style="font-family:'DM Serif Display',serif;font-size:26px;color:#E4E8F4;letter-spacing:-0.5px;">FinSight AI</span>
            <span style="font-size:11px;color:#3A4060;font-weight:500;letter-spacing:1.8px;text-transform:uppercase;">Financial Intelligence System</span>
        </div>
        <p style="margin:4px 0 0;color:#5A6480;font-size:13px;">Ask questions about Infosys and Apple financials &middot; Q1&ndash;Q4 FY25</p>
    </div>
    <script>
    !function(){
        function upd(){
            var h=document.getElementById('fh'); if(!h)return;
            var sb=document.querySelector('[data-testid="stSidebar"]');
            h.style.left = (sb && sb.getAttribute('aria-expanded')!=='false')
                ? sb.getBoundingClientRect().width+'px' : '0px';
        }
        var iv=setInterval(function(){
            if(!document.querySelector('[data-testid="stSidebar"]'))return;
            clearInterval(iv); upd();
            var sb=document.querySelector('[data-testid="stSidebar"]');
            new MutationObserver(upd).observe(sb,{attributes:true});
            new ResizeObserver(upd).observe(sb);
        },100);
        window.addEventListener('resize',upd);
    }();
    </script>
    """, unsafe_allow_html=True)


def render_infosys_trend_bar():
    """
    Modern mini trend chart for the sidebar.
    Uses real Infosys revenue and normalizes bar heights between
    35%-100% (instead of raw % of max) so even the lowest quarter
    is clearly visible — this fixes the "half cut off" look that
    happened when bars were too close in raw percentage terms.
    """
    values = [v for _, v in INFOSYS_TREND]
    min_v, max_v = min(values), max(values)
    span = max_v - min_v if max_v != min_v else 1

    def norm_height(v):
        pct = (v - min_v) / span
        return 35 + pct * 65  # 35%-100% range, never "cut off"

    overall_change = values[-1] - values[0]
    overall_pct = (overall_change / values[0]) * 100

    bars_html = ""
    for i, (label, val) in enumerate(INFOSYS_TREND):
        h = norm_height(val)
        is_last = (i == len(INFOSYS_TREND) - 1)
        is_max = (val == max_v)
        color = "#22D3A5" if is_max else "#4F8EF7"
        opacity = "1" if (is_max or is_last) else "0.55"
        bars_html += (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;">'
            f'<span style="font-size:9px;color:#5A6480;font-weight:500;white-space:nowrap;">${val:,.0f}M</span>'
            f'<div style="width:100%;max-width:22px;height:{h:.0f}%;'
            f'background:{color};opacity:{opacity};border-radius:4px 4px 1px 1px;'
            f'transition:opacity 0.2s;" title="{label} ${val:,.0f}M"></div>'
            f'<span style="font-size:9px;color:#3A4060;font-weight:600;letter-spacing:0.3px;">{label}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#141720;border:1px solid #1E2438;border-radius:10px;'
        f'padding:14px 12px 10px;">'
        f'<div style="display:flex;align-items:flex-end;justify-content:space-between;'
        f'gap:6px;height:64px;margin-bottom:2px;">'
        f'{bars_html}'
        f'</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-top:10px;padding-top:10px;border-top:1px solid #1A1E2E;">'
        f'<span style="font-size:10px;color:#5A6480;">Q1 → Q4 change</span>'
        f'<span style="font-size:13px;font-weight:600;color:#22D3A5;">'
        f'{"+" if overall_change >= 0 else ""}{overall_pct:.1f}%</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.1rem 0 0.75rem;">
            <div style="font-family:'DM Serif Display',serif;font-size:20px;
                        color:#E4E8F4;letter-spacing:-0.3px;">FinSight AI</div>
            <div style="font-size:10px;color:#3A4060;letter-spacing:1.4px;
                        text-transform:uppercase;margin-top:2px;">FY25 Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Data Coverage ──
        st.markdown('<p style="font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#3A4060;font-weight:500;margin:0 0 10px;">Data Coverage</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background:#141720;border:1px solid #1E2438;border-radius:10px;padding:0.85rem 0.75rem;text-align:center;">
                <div style="font-size:22px;margin-bottom:5px;">🍎</div>
                <div style="font-size:12px;font-weight:600;color:#E4E8F4;margin-bottom:7px;">Apple</div>
                <div style="display:flex;flex-wrap:wrap;gap:3px;justify-content:center;">
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#4F8EF720;color:#4F8EF7;border:1px solid #4F8EF730;">Q1</span>
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#4F8EF720;color:#4F8EF7;border:1px solid #4F8EF730;">Q2</span>
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#1A1E2E;color:#3A4060;">Q3</span>
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#1A1E2E;color:#3A4060;">Q4</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background:#141720;border:1px solid #1E2438;border-radius:10px;padding:0.85rem 0.75rem;text-align:center;">
                <div style="font-size:22px;margin-bottom:5px;">💼</div>
                <div style="font-size:12px;font-weight:600;color:#E4E8F4;margin-bottom:7px;">Infosys</div>
                <div style="display:flex;flex-wrap:wrap;gap:3px;justify-content:center;">
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#4F8EF720;color:#4F8EF7;border:1px solid #4F8EF730;">Q1</span>
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#4F8EF720;color:#4F8EF7;border:1px solid #4F8EF730;">Q2</span>
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#4F8EF720;color:#4F8EF7;border:1px solid #4F8EF730;">Q3</span>
                    <span style="font-size:9px;padding:2px 6px;border-radius:4px;background:#4F8EF720;color:#4F8EF7;border:1px solid #4F8EF730;">Q4</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Key Metrics ──
        st.markdown('<p style="font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#3A4060;font-weight:500;margin:0 0 10px;">Key Metrics · FY25</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:4px;">
            <div style="background:#141720;border:1px solid #1E2438;border-radius:8px;padding:9px 11px;">
                <div style="font-size:15px;font-weight:600;color:#E4E8F4;letter-spacing:-0.3px;">$5,040M</div>
                <div style="font-size:10px;color:#5A6480;margin-top:2px;">Infosys Q4</div>
                <div style="font-size:10px;color:#22D3A5;margin-top:2px;">↑ 4.1% YoY</div>
            </div>
            <div style="background:#141720;border:1px solid #1E2438;border-radius:8px;padding:9px 11px;">
                <div style="font-size:15px;font-weight:600;color:#E4E8F4;letter-spacing:-0.3px;">20.9%</div>
                <div style="font-size:10px;color:#5A6480;margin-top:2px;">EBIT Margin</div>
                <div style="font-size:10px;color:#22D3A5;margin-top:2px;">↑ 0.7% QoQ</div>
            </div>
            <div style="background:#141720;border:1px solid #1E2438;border-radius:8px;padding:9px 11px;">
                <div style="font-size:15px;font-weight:600;color:#E4E8F4;letter-spacing:-0.3px;">$95.4B</div>
                <div style="font-size:10px;color:#5A6480;margin-top:2px;">Apple Q2</div>
                <div style="font-size:10px;color:#22D3A5;margin-top:2px;">↑ 5.1% YoY</div>
            </div>
            <div style="background:#141720;border:1px solid #1E2438;border-radius:8px;padding:9px 11px;">
                <div style="font-size:15px;font-weight:600;color:#E4E8F4;letter-spacing:-0.3px;">3.1%</div>
                <div style="font-size:10px;color:#5A6480;margin-top:2px;">FY26 Guide</div>
                <div style="font-size:10px;color:#F5A623;margin-top:2px;">CC growth</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Revenue Trend mini bar ──
        st.markdown('<p style="font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#3A4060;font-weight:500;margin:0 0 8px;">Revenue Trend · Infosys</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#141720;border:1px solid #1E2438;border-radius:8px;
                    padding:10px 14px;display:flex;align-items:flex-end;justify-content:space-between;">
            <div style="display:flex;align-items:flex-end;gap:5px;height:36px;">
                <div style="width:16px;height:55%;background:#4F8EF7;opacity:0.55;border-radius:2px 2px 0 0;" title="Q1 $4,554M"></div>
                <div style="width:16px;height:68%;background:#4F8EF7;opacity:0.7;border-radius:2px 2px 0 0;" title="Q2 $4,658M"></div>
                <div style="width:16px;height:84%;background:#4F8EF7;opacity:0.85;border-radius:2px 2px 0 0;" title="Q3 $4,936M"></div>
                <div style="width:16px;height:100%;background:#22D3A5;border-radius:2px 2px 0 0;" title="Q4 $5,040M"></div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:14px;color:#22D3A5;font-weight:600;">+10.7%</div>
                <div style="font-size:9px;color:#3A4060;margin-top:1px;">Q1 → Q4</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Quick Topics ──
        st.markdown('<p style="font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#3A4060;font-weight:500;margin:0 0 8px;">Quick Topics</p>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (label, query) in enumerate(QUICK_TOPICS):
            with cols[i % 2]:
                if st.button(label, key=f"topic_{label}", use_container_width=True):
                    st.session_state.pending_query = query
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Session stats ──
        if st.session_state.get("chat_history"):
            total = len([h for h in st.session_state.chat_history if h["role"] == "assistant"])
            st.markdown(
                f'<div style="font-size:11px;color:#3A4060;margin-bottom:8px;">'
                f'<span style="color:#5A6480;">Queries this session:</span> '
                f'<span style="color:#E4E8F4;font-weight:600;">{total}</span></div>',
                unsafe_allow_html=True
            )

        if st.button("🗑  Clear conversation", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.session_state.pending_query = None
            st.rerun()

        st.markdown("""
        <div style="margin-top:1.5rem;padding-top:0.75rem;border-top:1px solid #141720;">
            <div style="font-size:9px;color:#252A3A;letter-spacing:0.5px;line-height:1.8;">
                ChromaDB · Groq · LLaMA 3.1 8B<br>FY2025 · Q1–Q4
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────
def render_empty_state():
    st.markdown("""
    <div style="text-align:center;padding:4rem 1rem 2rem;color:#5A6480;">
        <div style="width:52px;height:52px;background:#141720;border:1px solid #1E2438;
                    border-radius:14px;display:flex;align-items:center;justify-content:center;
                    font-size:24px;margin:0 auto 1.25rem;">📊</div>
        <div style="font-family:'DM Serif Display',serif;font-size:20px;
                    color:#C8D0E8;margin-bottom:6px;">
            Ask anything about Infosys &amp; Apple
        </div>
        <div style="font-size:13px;color:#3A4060;line-height:1.8;">
            Revenue · Margins · Deal wins · Segments · Trends · Guidance
        </div>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        ("Infosys Q4 revenue",    "What is Infosys Q4 revenue?"),
        ("Apple Q2 summary",      "Summarize Apple Q2 earnings"),
        ("Infosys revenue trend", "Show Infosys revenue trend Q1 to Q4"),
        ("Compare Apple Q1 vs Q2","Compare Apple Q1 and Q2 revenue"),
        ("Q3 growth drivers",     "Why did Infosys revenue grow in Q3?"),
        ("FY26 guidance",         "What is Infosys FY26 revenue guidance?"),
    ]
    cols1 = st.columns(3)
    cols2 = st.columns(3)
    for i, (label, query) in enumerate(suggestions):
        col = cols1[i] if i < 3 else cols2[i - 3]
        with col:
            if st.button(label, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# USER BUBBLE
# ─────────────────────────────────────────────
def render_user_bubble(content: str):
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin-bottom:4px;">'
        f'<div style="background:#1A2340;border:1px solid #263060;'
        f'border-radius:16px 16px 4px 16px;padding:10px 16px;'
        f'max-width:75%;font-size:14px;color:#C8D4F0;line-height:1.6;">'
        f'{content}</div></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    with st.spinner("Loading models…"):
        run_fn, LATEST_QUARTER, extract_revenue, search_fn = load_pipeline()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    render_sidebar()

    # Sticky header — renders at top, stays fixed on scroll via CSS
    render_header()

    # ── Chat history ──
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        else:
            with st.chat_message("assistant"):
                render_answer(
                    msg["content"],
                    msg.get("qtype", "simple_qa"),
                    msg.get("elapsed", 0),
                    msg.get("sources", [])
                )

    # ── Empty state ──
    if not st.session_state.chat_history:
        render_empty_state()

    # ── Handle topic / suggestion click ──
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        process_query(query, run_fn)

    # ── Chat input ──
    if prompt := st.chat_input("Ask about Infosys or Apple financials…"):
        process_query(prompt, run_fn)


def process_query(query: str, run_fn):
    render_user_bubble(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            start = time.time()
            try:
                response = run_fn(query)
                elapsed  = time.time() - start
                if response is None:
                    result, qtype, sources = "Something went wrong. Please try again.", "invalid", []
                elif len(response) == 3:
                    result, qtype, sources = response
                else:
                    result, qtype = response
                    sources = []
            except Exception as e:
                elapsed = time.time() - start
                result, qtype, sources = f"Error: {str(e)}", "invalid", []

        render_answer(result, qtype, elapsed, sources)

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": result,
        "qtype":   qtype,
        "elapsed": elapsed,
        "sources": sources,
    })


if __name__ == "__main__":
    main()