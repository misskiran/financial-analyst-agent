"""
Financial Analyst AI Agent — Streamlit UI
==========================================
Run with:  streamlit run app.py
"""

import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import (
    search_news,
    get_stock_price,
    place_order,
    analysis_node,
    research_node,
    STALE_MINUTES,
    TRADE_QTY,
    AgentState,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AI Financial Analyst",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #091020 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1929 0%, #0a1020 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* ── Cards ── */
.agent-card {
    background: rgba(15, 25, 50, 0.8);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease;
}
.agent-card:hover { border-color: rgba(99,179,237,0.45); }

.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.6rem;
}
.card-value {
    font-size: 2rem;
    font-weight: 800;
    color: #e2e8f0;
    line-height: 1.1;
}
.card-sub {
    font-size: 0.82rem;
    color: #718096;
    margin-top: 0.25rem;
}

/* ── Hero title ── */
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #63b3ed, #9f7aea, #ed64a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #718096;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* ── Sentiment badges ── */
.badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}
.badge-positive { background: rgba(72,187,120,0.2); color: #68d391; border: 1px solid rgba(72,187,120,0.4); }
.badge-negative { background: rgba(252,129,74,0.2); color: #fc8181; border: 1px solid rgba(252,129,74,0.4); }
.badge-neutral  { background: rgba(99,179,237,0.15); color: #90cdf4; border: 1px solid rgba(99,179,237,0.3); }

/* ── Step indicators ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.step-icon { font-size: 1.25rem; width: 2rem; text-align: center; }
.step-label { color: #a0aec0; font-size: 0.9rem; flex: 1; }
.step-status { font-size: 0.8rem; font-weight: 600; }
.step-done    { color: #68d391; }
.step-running { color: #f6e05e; }
.step-pending { color: #4a5568; }

/* ── Headline list ── */
.headline-item {
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e0;
    font-size: 0.88rem;
    line-height: 1.5;
}
.headline-num {
    color: #63b3ed;
    font-weight: 700;
    margin-right: 0.5rem;
}

/* ── Stale banner ── */
.stale-banner {
    background: rgba(237,137,54,0.15);
    border: 1px solid rgba(237,137,54,0.4);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #fbd38d;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* ── Order success ── */
.order-success {
    background: rgba(72,187,120,0.12);
    border: 1px solid rgba(72,187,120,0.35);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    color: #68d391;
    font-size: 0.9rem;
}
.order-id {
    font-family: monospace;
    font-size: 0.82rem;
    color: #9ae6b4;
    margin-top: 0.4rem;
    word-break: break-all;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(99,179,237,0.12);
    margin: 1.25rem 0;
}

/* ── Sidebar text & inputs ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] input {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cbd5e0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.2) !important;
}
/* ── Main area labels ── */
label, .stTextInput label, .stNumberInput label {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase":            "idle",   # idle | researching | analysing | awaiting | trading | done
        "agent_state":      None,
        "log":              [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log.append(f"[{ts}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    ticker   = st.text_input("Ticker Symbol", value="NVDA").upper()
    qty      = st.number_input("Trade Quantity (shares)", min_value=1, max_value=100, value=TRADE_QTY)
    stale_m  = st.number_input("Stale Data Threshold (minutes)", min_value=1, max_value=60, value=STALE_MINUTES)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Workflow Steps")

    steps = [
        ("📡", "Research",         "researching"),
        ("🧠", "Analysis",         "analysing"),
        ("⏸️", "Human Approval",   "awaiting"),
        ("🚀", "Execute Trade",    "trading"),
        ("✅", "Complete",         "done"),
    ]
    phase = st.session_state.phase

    for icon, label, step_id in steps:
        if phase == step_id:
            status_html = '<span class="step-status step-running">⟳ Running</span>'
        elif steps.index((icon, label, step_id)) < [s[2] for s in steps].index(phase) if phase != "idle" and phase in [s[2] for s in steps] else False:
            status_html = '<span class="step-status step-done">✓ Done</span>'
        else:
            status_html = '<span class="step-status step-pending">○ Pending</span>'

        st.markdown(f"""
        <div class="step-row">
            <span class="step-icon">{icon}</span>
            <span class="step-label">{label}</span>
            {status_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if st.button("🔄 Reset Agent", use_container_width=True):
        for k in ["phase", "agent_state", "log"]:
            del st.session_state[k]
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="hero-title">📈 AI Financial Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by LangGraph · Gemini 2.0 · Tavily · Alpaca Paper Trading</div>', unsafe_allow_html=True)

# ── Phase: IDLE ──────────────────────────────────────────────────────────────
if st.session_state.phase == "idle":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div class="agent-card">
            <div class="card-title">Ready to Analyse</div>
            <div class="card-value">{ticker}</div>
            <div class="card-sub">Click <strong>Run Analysis</strong> to start the agentic workflow</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ Run Analysis", type="primary", use_container_width=True):
            st.session_state.phase = "researching"
            st.session_state.agent_state = {
                "ticker": ticker, "news_headlines": [], "current_price": None,
                "pe_ratio": None, "fetched_at": None, "sentiment": None,
                "analysis_summary": None, "human_approved": None,
                "trade_action": None, "order_id": None, "error": None,
            }
            log(f"Agent started for {ticker}")
            st.rerun()

# ── Phase: RESEARCHING ────────────────────────────────────────────────────────
elif st.session_state.phase == "researching":
    st.markdown('<div class="agent-card"><div class="card-title">📡 Research Node</div>', unsafe_allow_html=True)
    with st.spinner("Fetching news from Tavily & price from yfinance …"):
        s = st.session_state.agent_state
        headlines  = search_news(s["ticker"])
        price_data = get_stock_price(s["ticker"])
        s.update({
            "news_headlines": headlines,
            "current_price":  price_data["price"],
            "pe_ratio":       price_data["pe_ratio"],
            "fetched_at":     time.time(),
        })
        log(f"Research done — price ${price_data['price']}, {len(headlines)} headlines")
    st.session_state.phase = "analysing"
    st.rerun()

# ── Phase: ANALYSING ──────────────────────────────────────────────────────────
elif st.session_state.phase == "analysing":
    with st.spinner("🧠 Asking Gemini 2.0 to analyse sentiment …"):
        result = analysis_node(st.session_state.agent_state)
        st.session_state.agent_state.update(result)
        log(f"Analysis done — sentiment: {result.get('sentiment')}, action: {result.get('trade_action')}")
    st.session_state.phase = "awaiting"
    st.rerun()

# ── Phase: AWAITING APPROVAL ──────────────────────────────────────────────────
elif st.session_state.phase in ("awaiting", "trading", "done"):
    s = st.session_state.agent_state

    # ── Stale data check ────────────────────────────────────────────────────
    elapsed_min = (time.time() - (s.get("fetched_at") or time.time())) / 60
    if elapsed_min > stale_m and st.session_state.phase == "awaiting":
        st.markdown(f"""
        <div class="stale-banner">
            ⚠️ Data is <strong>{elapsed_min:.1f} min</strong> old (threshold: {stale_m} min).
            Please refresh to get the latest data before approving a trade.
        </div>""", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data Now", type="secondary"):
            st.session_state.phase = "researching"
            st.rerun()

    # ── Metric cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="agent-card">
            <div class="card-title">Ticker</div>
            <div class="card-value">{s['ticker']}</div>
            <div class="card-sub">NYSE / NASDAQ</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        price_str = f"${s['current_price']:.2f}" if s.get("current_price") else "N/A"
        st.markdown(f"""
        <div class="agent-card">
            <div class="card-title">Current Price</div>
            <div class="card-value">{price_str}</div>
            <div class="card-sub">Live market price</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        pe_str = f"{s['pe_ratio']:.1f}x" if s.get("pe_ratio") else "N/A"
        st.markdown(f"""
        <div class="agent-card">
            <div class="card-title">Trailing P/E</div>
            <div class="card-value">{pe_str}</div>
            <div class="card-sub">Price-to-Earnings ratio</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        sent = (s.get("sentiment") or "NEUTRAL").upper()
        badge_cls = {"POSITIVE": "badge-positive", "NEGATIVE": "badge-negative"}.get(sent, "badge-neutral")
        sent_icon = {"POSITIVE": "📈", "NEGATIVE": "📉"}.get(sent, "➡️")
        st.markdown(f"""
        <div class="agent-card">
            <div class="card-title">AI Sentiment</div>
            <div style="margin-top:0.5rem"><span class="badge {badge_cls}">{sent_icon} {sent}</span></div>
            <div class="card-sub" style="margin-top:0.5rem">Gemini 2.0 analysis</div>
        </div>""", unsafe_allow_html=True)

    # ── Analysis summary ─────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### 🧠 Gemini Analysis")
        summary = s.get("analysis_summary") or "No summary available."
        st.markdown(f"""
        <div class="agent-card">
            <div class="card-title">Summary</div>
            <p style="color:#cbd5e0;font-size:0.92rem;line-height:1.7;margin:0">{summary}</p>
        </div>""", unsafe_allow_html=True)

        if s.get("error"):
            st.error(f"⚠️ Error: {s['error']}")

    with right:
        st.markdown("#### 📰 News Headlines")
        headlines = s.get("news_headlines", [])
        if headlines:
            items = "".join(
                f'<div class="headline-item"><span class="headline-num">{i+1}.</span>{h}</div>'
                for i, h in enumerate(headlines)
            )
            st.markdown(f'<div class="agent-card">{items}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="agent-card"><span style="color:#4a5568">No headlines found.</span></div>', unsafe_allow_html=True)

    # ── Human Approval ────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### ⏸️ Human-in-the-Loop Approval")

    trade_action = s.get("trade_action")

    if st.session_state.phase == "awaiting":
        if trade_action:
            action_label = trade_action.upper()
            action_color = "#68d391" if trade_action == "buy" else "#fc8181"
            st.markdown(f"""
            <div class="agent-card" style="border-color: {action_color}44">
                <div class="card-title">Proposed Trade</div>
                <p style="color:#e2e8f0;font-size:0.95rem;margin:0">
                    Gemini recommends <strong style="color:{action_color}">{action_label}</strong>
                    &nbsp;<strong>{qty}</strong> share(s) of <strong>{s['ticker']}</strong>
                    at <strong>{price_str}</strong>
                </p>
            </div>""", unsafe_allow_html=True)

            col_approve, col_reject, _ = st.columns([1, 1, 2])
            with col_approve:
                if st.button(f"✅ Approve {action_label}", type="primary", use_container_width=True):
                    s["human_approved"] = True
                    s["trade_action"]   = trade_action
                    log(f"Trade APPROVED by human: {action_label} {qty} {s['ticker']}")
                    st.session_state.phase = "trading"
                    st.rerun()
            with col_reject:
                if st.button("❌ Reject", use_container_width=True):
                    s["human_approved"] = False
                    log("Trade REJECTED by human")
                    st.session_state.phase = "done"
                    st.rerun()
        else:
            st.markdown("""
            <div class="agent-card">
                <div class="card-title">No Trade Recommended</div>
                <p style="color:#a0aec0;font-size:0.9rem;margin:0">
                    Gemini's sentiment is NEUTRAL — the agent recommends <strong>HOLD</strong>.
                    No trade will be placed.
                </p>
            </div>""", unsafe_allow_html=True)
            if st.button("✅ Acknowledge & Finish", type="primary"):
                s["human_approved"] = False
                st.session_state.phase = "done"
                st.rerun()

    # ── Phase: TRADING ────────────────────────────────────────────────────────
    elif st.session_state.phase == "trading":
        with st.spinner(f"🚀 Placing {s.get('trade_action','').upper()} order via Alpaca …"):
            order_id = place_order(s["ticker"], s["trade_action"], qty)
            s["order_id"] = order_id
            log(f"Order placed — ID: {order_id}")
        st.session_state.phase = "done"
        st.rerun()

    # ── Phase: DONE ───────────────────────────────────────────────────────────
    elif st.session_state.phase == "done":
        order_id     = s.get("order_id")
        was_approved = s.get("human_approved")
        action_done  = s.get("trade_action")

        if was_approved and order_id and not order_id.startswith("ERROR"):
            st.markdown(f"""
            <div class="order-success">
                🎉 <strong>Order Placed Successfully</strong><br>
                Action: <strong>{(action_done or '').upper()}</strong> {qty} share(s) of <strong>{s['ticker']}</strong>
                via Alpaca Paper Trading
                <div class="order-id">Order ID: {order_id}</div>
            </div>""", unsafe_allow_html=True)
        elif was_approved and order_id and order_id.startswith("ERROR"):
            st.error(f"Trade failed: {order_id}")
        else:
            st.info("ℹ️ Trade was not executed (rejected or HOLD recommendation).")

        if st.button("▶ Run Again", type="primary"):
            for k in ["phase", "agent_state", "log"]:
                del st.session_state[k]
            st.rerun()

# ── Activity Log ──────────────────────────────────────────────────────────────
if st.session_state.log:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    with st.expander("🪵 Activity Log", expanded=False):
        log_text = "\n".join(st.session_state.log)
        st.code(log_text, language=None)
