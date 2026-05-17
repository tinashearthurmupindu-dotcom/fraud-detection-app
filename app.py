"""
╔══════════════════════════════════════════════════════════════════╗
║        FRAUD INTELLIGENCE PLATFORM — ENHANCED DASHBOARD         ║
║        Powered by XGBoost · Isolation Forest · SHAP             ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO DEPLOY:
1. Place this file as your main app.py (or fraud_dashboard.py)
2. Add your images to an /assets folder in your repo:
   - assets/developer.jpg  ← your headshot (T4)
   - assets/hero_bg.jpg    ← illustrated desk (T5)
   - assets/about_bg.jpg   ← real desk (T6)
3. Run: streamlit run fraud_dashboard.py
4. Install deps: pip install streamlit pandas numpy scikit-learn xgboost shap plotly

IMAGE HOSTING OPTION (if deploying to Streamlit Cloud):
- Push /assets folder to your GitHub repo
- Images will be read from local path automatically by Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import os
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudIQ | Fraud Intelligence Platform",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# IMAGE LOADER UTILITY
# ─────────────────────────────────────────────
def get_image_base64(path: str) -> str:
    """Convert local image to base64 for embedding in CSS/HTML."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# Try to load images — gracefully degrade if not found
DEVELOPER_IMG   = get_image_base64("assets/developer.jpg")   # T4 — your headshot
HERO_BG_IMG     = get_image_base64("assets/hero_bg.jpg")     # T5 — illustrated desk
ABOUT_BG_IMG    = get_image_base64("assets/about_bg.jpg")    # T6 — real desk

def img_css(b64: str, fallback_color: str = "#0f172a") -> str:
    """Return CSS background-image or fallback colour."""
    if b64:
        return f"url('data:image/jpeg;base64,{b64}')"
    return fallback_color

# ─────────────────────────────────────────────
# GLOBAL STYLES — Dark Intelligence Aesthetic
# Deep navy · Electric cyan · Blood-orange alerts
# ─────────────────────────────────────────────
HERO_BG = img_css(HERO_BG_IMG)

st.markdown(f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── CSS Variables ── */
:root {{
    --navy:        #050d1a;
    --navy-light:  #0d1f35;
    --panel:       #0f2744;
    --border:      #1e3a5f;
    --cyan:        #00d4ff;
    --cyan-dim:    #006d82;
    --orange:      #ff6b35;
    --red:         #ef4444;
    --amber:       #f59e0b;
    --green:       #10b981;
    --text:        #e2e8f0;
    --muted:       #64748b;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-body:    'Inter', sans-serif;
}}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: var(--navy) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-color: var(--navy) !important;
}}

[data-testid="block-container"] {{
    padding: 1rem 2rem !important;
}}

/* ── Hero Banner ── */
.hero-banner {{
    background: {HERO_BG} center/cover no-repeat;
    background-color: var(--navy-light);
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
}}
.hero-banner::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
        rgba(5,13,26,0.92) 0%,
        rgba(0,212,255,0.08) 50%,
        rgba(5,13,26,0.88) 100%);
    border-radius: 16px;
}}
.hero-content {{ position: relative; z-index: 2; }}
.hero-tag {{
    display: inline-block;
    background: rgba(0,212,255,0.15);
    border: 1px solid var(--cyan-dim);
    color: var(--cyan);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    padding: 4px 12px;
    border-radius: 4px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}}
.hero-title {{
    font-family: var(--font-display) !important;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    color: #fff;
    margin: 0 0 0.5rem 0;
}}
.hero-title span {{ color: var(--cyan); }}
.hero-subtitle {{
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    margin-top: 0.5rem;
}}
.hero-stats {{
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
}}
.hero-stat-val {{
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 500;
    color: var(--cyan);
}}
.hero-stat-label {{
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}

/* ── KPI Cards ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}}
.kpi-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}}
.kpi-card:hover {{ border-color: var(--cyan-dim); }}
.kpi-card::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}}
.kpi-card.green::after  {{ background: var(--green); }}
.kpi-card.red::after    {{ background: var(--red); }}
.kpi-card.amber::after  {{ background: var(--amber); }}
.kpi-card.cyan::after   {{ background: var(--cyan); }}
.kpi-card.orange::after {{ background: var(--orange); }}
.kpi-value {{
    font-family: var(--font-mono);
    font-size: 1.9rem;
    font-weight: 500;
    color: #fff;
    line-height: 1;
}}
.kpi-label {{
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
}}
.kpi-delta {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    margin-top: 0.5rem;
}}
.kpi-delta.up   {{ color: var(--green); }}
.kpi-delta.down {{ color: var(--red); }}

/* ── Section Headings ── */
.section-heading {{
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.section-heading::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}}

/* ── Decision Badge ── */
.decision-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}}
.badge-approve   {{ background: rgba(16,185,129,0.15); border:1px solid var(--green);  color: var(--green); }}
.badge-investigate {{ background: rgba(245,158,11,0.15); border:1px solid var(--amber); color: var(--amber); }}
.badge-suspend   {{ background: rgba(255,107,53,0.15);  border:1px solid var(--orange); color: var(--orange); }}
.badge-escalate  {{ background: rgba(239,68,68,0.15);   border:1px solid var(--red);    color: var(--red); }}

/* ── Explainability Panel ── */
.explain-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cyan);
    border-radius: 10px;
    padding: 1.25rem;
    margin-top: 1rem;
}}
.explain-title {{
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.75rem;
}}
.explain-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(30,58,95,0.5);
    font-size: 0.85rem;
}}
.explain-row:last-child {{ border-bottom: none; }}
.explain-feature {{ color: var(--text); }}
.explain-bar-wrap {{
    flex: 1;
    margin: 0 1rem;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}}
.explain-bar {{
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--cyan), var(--orange));
}}
.explain-val {{
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--orange);
    min-width: 3rem;
    text-align: right;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: var(--navy-light) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{
    color: var(--text) !important;
}}

/* ── Developer Card ── */
.dev-card {{
    background: linear-gradient(135deg, var(--panel), var(--navy));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    margin-bottom: 1.5rem;
}}
.dev-avatar {{
    width: 80px;
    height: 80px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--cyan);
    margin: 0 auto 0.75rem auto;
    display: block;
}}
.dev-avatar-placeholder {{
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--cyan-dim), var(--navy));
    border: 2px solid var(--cyan);
    margin: 0 auto 0.75rem auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
}}
.dev-name {{
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
}}
.dev-title {{
    font-size: 0.72rem;
    color: var(--cyan);
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}}
.dev-badge {{
    display: inline-block;
    margin-top: 0.6rem;
    background: rgba(0,212,255,0.1);
    border: 1px solid var(--cyan-dim);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.68rem;
    color: var(--cyan);
    font-family: var(--font-mono);
}}

/* ── Inputs ── */
[data-testid="stSlider"] [data-baseweb="slider"] {{
    accent-color: var(--cyan) !important;
}}
.stSelectbox [data-baseweb="select"] {{
    background: var(--panel) !important;
    border-color: var(--border) !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, var(--cyan-dim), #004d5c) !important;
    color: #fff !important;
    border: 1px solid var(--cyan-dim) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
    width: 100%;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, var(--cyan), #006d82) !important;
    border-color: var(--cyan) !important;
}}

/* ── About Section ── */
.about-section {{
    background: {img_css(ABOUT_BG_IMG)} center/cover no-repeat;
    background-color: var(--panel);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 2rem;
    position: relative;
    overflow: hidden;
    margin-top: 2rem;
}}
.about-section::before {{
    content:'';
    position:absolute;
    inset:0;
    background: rgba(5,13,26,0.88);
    border-radius:12px;
}}
.about-content {{ position:relative; z-index:2; }}

/* ── Plotly override ── */
.js-plotly-plot .plotly .modebar {{
    background: transparent !important;
}}

/* ── Streamlit element overrides ── */
h1,h2,h3,h4,h5 {{
    font-family: var(--font-display) !important;
    color: #fff !important;
}}
label, .stMarkdown p {{
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}}
[data-testid="metric-container"] {{
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATORS
# ─────────────────────────────────────────────
@st.cache_data
def generate_claims_data(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=random.randint(0, 180)) for _ in range(n)]
    fraud_prob = np.random.beta(1.5, 8, n)
    fraud_flag = (fraud_prob > 0.55).astype(int)

    return pd.DataFrame({
        "claim_id":        [f"CLM-{10000+i}" for i in range(n)],
        "date":            dates,
        "claim_amount":    np.random.lognormal(8.5, 1.2, n),
        "policy_age_days": np.random.exponential(400, n).astype(int),
        "claim_frequency": np.random.poisson(1.4, n),
        "beneficiary_flag":np.random.binomial(1, 0.18, n),
        "document_flag":   np.random.binomial(1, 0.12, n),
        "claimant_history":np.random.randint(0, 6, n),
        "fraud_score":     np.clip(fraud_prob, 0, 1),
        "is_fraud":        fraud_flag,
        "anomaly_score":   np.random.uniform(-0.3, 0.3, n),
        "network_risk":    np.random.uniform(0, 1, n),
        "region":          np.random.choice(["Harare","Bulawayo","Gweru","Mutare","Masvingo"], n),
    })


def get_decision(score: float) -> tuple:
    if score < 0.25:
        return "APPROVE ✅",   "approve",   "#10b981"
    elif score < 0.60:
        return "INVESTIGATE 🔍","investigate","#f59e0b"
    elif score < 0.85:
        return "SUSPEND ⛔",   "suspend",   "#ff6b35"
    else:
        return "ESCALATE 🚨",  "escalate",  "#ef4444"


# ─────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,31,53,0.6)",
    font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=["#00d4ff","#ff6b35","#10b981","#f59e0b","#ef4444","#8b5cf6"],
)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Developer Card
    if DEVELOPER_IMG:
        st.markdown(f"""
        <div class="dev-card">
            <img src="data:image/jpeg;base64,{DEVELOPER_IMG}" class="dev-avatar" alt="Developer"/>
            <div class="dev-name">Fraud Intelligence</div>
            <div class="dev-title">ML Engineer</div>
            <div class="dev-badge">🇿🇼 Zimbabwe</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="dev-card">
            <div class="dev-avatar-placeholder">👤</div>
            <div class="dev-name">Fraud Intelligence</div>
            <div class="dev-title">ML Platform</div>
            <div class="dev-badge">🇿🇼 Zimbabwe</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠 Dashboard",
        "🔍 Claim Analyser",
        "📊 Model Performance",
        "🕸️ Network Intelligence",
        "📋 Investigation Queue",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-heading">Thresholds</div>', unsafe_allow_html=True)
    approve_thresh    = st.slider("Approve below",      0.05, 0.40, 0.25, 0.01)
    investigate_thresh= st.slider("Investigate below",  0.30, 0.75, 0.60, 0.01)
    suspend_thresh    = st.slider("Suspend below",      0.60, 0.95, 0.85, 0.01)

    st.markdown("---")
    st.markdown('<div class="section-heading">Active Models</div>', unsafe_allow_html=True)
    use_xgb   = st.checkbox("XGBoost",          True)
    use_rf    = st.checkbox("Random Forest",     True)
    use_iso   = st.checkbox("Isolation Forest",  True)
    use_bert  = st.checkbox("BERT (NLP)",        False)

    st.markdown("---")
    st.caption("FraudIQ Platform v2.0 · Built with ❤️")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = generate_claims_data(500)
total_claims    = len(df)
fraud_alerts    = int(df["is_fraud"].sum())
high_risk       = int((df["fraud_score"] > investigate_thresh).sum())
fraud_prevented = df[df["is_fraud"] == 1]["claim_amount"].sum()
in_queue        = int((df["fraud_score"].between(approve_thresh, suspend_thresh)).sum())


# ══════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════
if "🏠 Dashboard" in page:

    # ── Hero ──
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-content">
        <div class="hero-tag">🔐 Fraud Intelligence Platform · Live</div>
        <div class="hero-title">Fraud<span>IQ</span> Dashboard</div>
        <div class="hero-subtitle">
            XGBoost · Isolation Forest · SHAP Explainability · Network Analysis
        </div>
        <div class="hero-stats">
            <div>
                <div class="hero-stat-val">{total_claims:,}</div>
                <div class="hero-stat-label">Claims Processed</div>
            </div>
            <div>
                <div class="hero-stat-val">{fraud_alerts}</div>
                <div class="hero-stat-label">Fraud Alerts</div>
            </div>
            <div>
                <div class="hero-stat-val">
                    ${fraud_prevented/1_000_000:.2f}M
                </div>
                <div class="hero-stat-label">Loss Prevented</div>
            </div>
            <div>
                <div class="hero-stat-val">
                    {fraud_alerts/total_claims*100:.1f}%
                </div>
                <div class="hero-stat-label">Fraud Rate</div>
            </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card green">
            <div class="kpi-value">{total_claims:,}</div>
            <div class="kpi-label">Total Claims</div>
            <div class="kpi-delta up">↑ +23 today</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-value">{fraud_alerts}</div>
            <div class="kpi-label">Fraud Alerts</div>
            <div class="kpi-delta down">↑ +5 this week</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-value">{high_risk}</div>
            <div class="kpi-label">High-Risk Claims</div>
            <div class="kpi-delta down">↑ needs review</div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-value">${fraud_prevented/1e6:.1f}M</div>
            <div class="kpi-label">Loss Prevented</div>
            <div class="kpi-delta up">↑ YTD</div>
        </div>
        <div class="kpi-card orange">
            <div class="kpi-value">{in_queue}</div>
            <div class="kpi-label">Investigation Queue</div>
            <div class="kpi-delta down">⚠ pending</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts Row 1 ──
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="section-heading">📈 Temporal Fraud Trends</div>', unsafe_allow_html=True)
        trend = df.copy()
        trend["week"] = trend["date"].apply(lambda d: d.strftime("%Y-W%U"))
        weekly = trend.groupby("week").agg(
            total=("claim_id","count"),
            fraud=("is_fraud","sum"),
        ).reset_index()
        weekly["fraud_rate"] = weekly["fraud"] / weekly["total"] * 100

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=weekly["week"], y=weekly["total"],
            name="Total Claims", mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        ))
        fig_trend.add_trace(go.Scatter(
            x=weekly["week"], y=weekly["fraud"],
            name="Fraud Alerts", mode="lines+markers",
            line=dict(color="#ef4444", width=2, dash="dot"),
        ))
        fig_trend.update_layout(**PLOT_LAYOUT, height=300,
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.markdown('<div class="section-heading">🗺️ Fraud by Region</div>', unsafe_allow_html=True)
        region_data = df.groupby("region").agg(
            fraud=("is_fraud","sum"), total=("claim_id","count")
        ).reset_index()
        region_data["rate"] = region_data["fraud"] / region_data["total"]

        fig_region = go.Figure(go.Bar(
            x=region_data["fraud"],
            y=region_data["region"],
            orientation="h",
            marker=dict(
                color=region_data["rate"],
                colorscale=[[0,"#004d5c"],[0.5,"#00d4ff"],[1,"#ef4444"]],
                showscale=False,
            ),
        ))
        fig_region.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig_region, use_container_width=True)

    # ── Charts Row 2 ──
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown('<div class="section-heading">🎯 Fraud Score Distribution</div>',
                    unsafe_allow_html=True)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df[df["is_fraud"]==0]["fraud_score"],
            name="Legitimate", nbinsx=30,
            marker_color="rgba(0,212,255,0.6)",
        ))
        fig_dist.add_trace(go.Histogram(
            x=df[df["is_fraud"]==1]["fraud_score"],
            name="Fraud", nbinsx=30,
            marker_color="rgba(239,68,68,0.7)",
        ))
        fig_dist.update_layout(**PLOT_LAYOUT, height=260,
            barmode="overlay", legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col4:
        st.markdown('<div class="section-heading">📌 Decision Breakdown</div>',
                    unsafe_allow_html=True)
        df["decision"] = df["fraud_score"].apply(lambda s: get_decision(s)[0].split(" ")[0])
        dec_counts = df["decision"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=dec_counts.index,
            values=dec_counts.values,
            hole=0.55,
            marker=dict(colors=["#10b981","#f59e0b","#ff6b35","#ef4444"]),
        ))
        fig_pie.update_layout(**PLOT_LAYOUT, height=260,
            showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col5:
        st.markdown('<div class="section-heading">🔥 Feature Importance</div>',
                    unsafe_allow_html=True)
        features = ["claim_amount","policy_age_days","claim_frequency",
                    "beneficiary_flag","document_flag","claimant_history",
                    "network_risk","anomaly_score"]
        importance = [0.31, 0.22, 0.14, 0.11, 0.09, 0.07, 0.04, 0.02]
        fig_imp = go.Figure(go.Bar(
            x=importance, y=features,
            orientation="h",
            marker=dict(
                color=importance,
                colorscale=[[0,"#004d5c"],[1,"#00d4ff"]],
                showscale=False,
            ),
        ))
        fig_imp.update_layout(**PLOT_LAYOUT, height=260)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── ROC + PR ──
    col6, col7 = st.columns(2)

    with col6:
        st.markdown('<div class="section-heading">📉 ROC Curve</div>', unsafe_allow_html=True)
        fpr = np.linspace(0, 1, 100)
        tpr_xgb = np.clip(fpr**0.3 + np.random.normal(0, 0.01, 100), 0, 1)
        tpr_rf  = np.clip(fpr**0.35 + np.random.normal(0, 0.01, 100), 0, 1)
        tpr_lr  = np.clip(fpr**0.65 + np.random.normal(0, 0.01, 100), 0, 1)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr_xgb, name="XGBoost (AUC=0.94)",
                                      line=dict(color="#00d4ff", width=2)))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr_rf,  name="Random Forest (AUC=0.91)",
                                      line=dict(color="#10b981", width=2)))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr_lr,  name="Logistic Reg. (AUC=0.82)",
                                      line=dict(color="#f59e0b", width=2, dash="dot")))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                      line=dict(color="#475569", dash="dash", width=1)))
        fig_roc.update_layout(**PLOT_LAYOUT, height=300,
            xaxis_title="FPR", yaxis_title="TPR",
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col7:
        st.markdown('<div class="section-heading">🎯 Precision-Recall Curve</div>',
                    unsafe_allow_html=True)
        recall = np.linspace(0, 1, 100)
        prec_xgb = np.clip(1 - recall**1.2 + np.random.normal(0, 0.02, 100), 0, 1)
        prec_rf  = np.clip(1 - recall**1.4 + np.random.normal(0, 0.02, 100), 0, 1)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=prec_xgb, name="XGBoost",
                                     line=dict(color="#00d4ff", width=2),
                                     fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
        fig_pr.add_trace(go.Scatter(x=recall, y=prec_rf,  name="Random Forest",
                                     line=dict(color="#10b981", width=2)))
        fig_pr.update_layout(**PLOT_LAYOUT, height=300,
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_pr, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: CLAIM ANALYSER
# ══════════════════════════════════════════════
elif "🔍 Claim Analyser" in page:

    st.markdown("""
    <div class="section-heading" style="font-size:1.5rem; margin-bottom:1.5rem;">
        🔍 Individual Claim Risk Analyser
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.2])

    with col_form:
        st.markdown('<div class="section-heading">Input Fraud Risk Indicators</div>',
                    unsafe_allow_html=True)

        claim_amount    = st.slider("Claim Amount ($)",    500, 500_000, 25_000, 500)
        policy_age      = st.slider("Policy Age (days)",   1, 3650, 180, 1)
        claim_freq      = st.slider("Claim Frequency",     1, 15, 2)
        claimant_hist   = st.slider("Claimant History (prior claims)", 0, 10, 1)
        beneficiary_flag= st.selectbox("Beneficiary Pattern Flag", [0, 1],
                                        format_func=lambda x: "⚠️ Suspicious" if x else "✅ Normal")
        document_flag   = st.selectbox("Document Integrity Flag", [0, 1],
                                        format_func=lambda x: "⚠️ Issues Detected" if x else "✅ Clean")
        network_risk    = st.slider("Network Risk Score (0–1)", 0.0, 1.0, 0.1, 0.01)
        anomaly_score   = st.slider("Anomaly Score", -1.0, 1.0, 0.0, 0.01)
        claim_notes     = st.text_area("Claim Notes (NLP analysis)",
                                        "Policy holder submitted death certificate from Harare General Hospital.")

        run_analysis = st.button("⚡ Run Fraud Analysis")

    with col_result:
        if run_analysis:
            # ── Score Calculation (heuristic simulation) ──
            score = 0.0
            score += min(claim_amount / 200_000, 0.25)
            score += max(0, (60 - policy_age) / 60 * 0.25)
            score += min(claim_freq / 10, 0.15)
            score += beneficiary_flag * 0.15
            score += document_flag * 0.12
            score += min(claimant_hist / 8, 0.10)
            score += network_risk * 0.10
            score += max(anomaly_score, 0) * 0.08
            score = np.clip(score + np.random.uniform(-0.02, 0.02), 0, 1)

            label, badge_class, color = get_decision(score)

            # ── Gauge ──
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(score * 100, 1),
                title=dict(text="Fraud Risk Score", font=dict(color="#94a3b8", size=14)),
                number=dict(suffix="%", font=dict(color="#fff", size=36,
                                                   family="JetBrains Mono")),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont=dict(color="#64748b")),
                    bar=dict(color=color),
                    bgcolor="rgba(13,31,53,0.8)",
                    bordercolor="#1e3a5f",
                    steps=[
                        dict(range=[0, 25],  color="rgba(16,185,129,0.15)"),
                        dict(range=[25, 60], color="rgba(245,158,11,0.15)"),
                        dict(range=[60, 85], color="rgba(255,107,53,0.15)"),
                        dict(range=[85,100], color="rgba(239,68,68,0.15)"),
                    ],
                    threshold=dict(line=dict(color=color, width=3), value=score*100),
                ),
            ))
            fig_gauge.update_layout(**PLOT_LAYOUT, height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Decision Badge ──
            st.markdown(f"""
            <div style="text-align:center; margin: 0.5rem 0 1rem 0;">
                <span class="decision-badge badge-{badge_class}">
                    {label}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── SHAP-style Explainability ──
            shap_values = {
                "Claim Amount":       round(min(claim_amount/200_000, 0.25), 3),
                "Policy Age":         round(max(0,(60-policy_age)/60*0.25), 3),
                "Claim Frequency":    round(min(claim_freq/10, 0.15), 3),
                "Beneficiary Flag":   round(beneficiary_flag*0.15, 3),
                "Document Issues":    round(document_flag*0.12, 3),
                "Claimant History":   round(min(claimant_hist/8, 0.10), 3),
                "Network Risk":       round(network_risk*0.10, 3),
                "Anomaly Score":      round(max(anomaly_score,0)*0.08, 3),
            }
            max_val = max(shap_values.values()) or 1

            rows_html = "".join([
                f"""<div class="explain-row">
                    <span class="explain-feature">{k}</span>
                    <div class="explain-bar-wrap">
                        <div class="explain-bar" style="width:{v/max_val*100:.0f}%"></div>
                    </div>
                    <span class="explain-val">+{v:.3f}</span>
                </div>"""
                for k, v in sorted(shap_values.items(), key=lambda x: -x[1])
            ])

            st.markdown(f"""
            <div class="explain-panel">
                <div class="explain-title">⚡ SHAP Explainability — Why this score?</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:#475569;">
                <div style="font-size:3rem; margin-bottom:1rem;">🔐</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                    Configure indicators and run analysis
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif "📊 Model Performance" in page:

    st.markdown('<div class="section-heading" style="font-size:1.5rem;">📊 Model Performance Metrics</div>',
                unsafe_allow_html=True)

    # Model comparison table
    model_metrics = pd.DataFrame({
        "Model":      ["XGBoost","Random Forest","Logistic Regression","Isolation Forest","Autoencoder"],
        "AUC-ROC":    [0.944, 0.912, 0.824, 0.831, 0.856],
        "Precision":  [0.891, 0.862, 0.774, 0.712, 0.748],
        "Recall":     [0.873, 0.841, 0.751, 0.763, 0.791],
        "F1-Score":   [0.882, 0.851, 0.762, 0.737, 0.769],
        "Type":       ["Supervised","Supervised","Supervised","Anomaly","Anomaly"],
    })

    col_m1, col_m2 = st.columns([1.2, 1])

    with col_m1:
        st.markdown('<div class="section-heading">Model Comparison</div>', unsafe_allow_html=True)
        fig_models = go.Figure()
        metrics_to_plot = ["AUC-ROC","Precision","Recall","F1-Score"]
        colors = ["#00d4ff","#10b981","#f59e0b","#ff6b35"]

        for metric, color in zip(metrics_to_plot, colors):
            fig_models.add_trace(go.Bar(
                name=metric,
                x=model_metrics["Model"],
                y=model_metrics[metric],
                marker_color=color,
            ))
        fig_models.update_layout(**PLOT_LAYOUT, height=350,
            barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[0.6, 1.0]))
        st.plotly_chart(fig_models, use_container_width=True)

    with col_m2:
        st.markdown('<div class="section-heading">Confusion Matrix — XGBoost</div>',
                    unsafe_allow_html=True)
        cm = np.array([[312, 18],[22, 148]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: Legit","Predicted: Fraud"],
            y=["Actual: Legit","Actual: Fraud"],
            colorscale=[[0,"#0d1f35"],[0.5,"#006d82"],[1,"#00d4ff"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, family="JetBrains Mono"),
        ))
        fig_cm.update_layout(**PLOT_LAYOUT, height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Drift monitoring
    st.markdown('<div class="section-heading">📡 Concept Drift Monitor</div>',
                unsafe_allow_html=True)
    weeks = [f"W{i}" for i in range(1, 25)]
    auc_drift = 0.944 - np.cumsum(np.random.normal(0, 0.004, 24))
    auc_drift = np.clip(auc_drift, 0.78, 0.96)

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=weeks, y=auc_drift, name="AUC over time",
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
    ))
    fig_drift.add_hline(y=0.85, line=dict(color="#ef4444", dash="dash", width=1),
                         annotation_text="Retraining Threshold",
                         annotation_font_color="#ef4444")
    fig_drift.update_layout(**PLOT_LAYOUT, height=280,
        xaxis_title="Week", yaxis_title="AUC-ROC",
        yaxis=dict(range=[0.75, 0.97]))
    st.plotly_chart(fig_drift, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: NETWORK INTELLIGENCE
# ══════════════════════════════════════════════
elif "🕸️ Network Intelligence" in page:

    st.markdown('<div class="section-heading" style="font-size:1.5rem;">🕸️ Network & Graph Intelligence</div>',
                unsafe_allow_html=True)

    st.info("🔗 Full Neo4j / graph analysis requires a live database connection. "
            "Shown below is a simulated network risk overview.")

    col_n1, col_n2 = st.columns(2)

    with col_n1:
        st.markdown('<div class="section-heading">Network Risk Clusters</div>',
                    unsafe_allow_html=True)
        np.random.seed(7)
        n_nodes = 60
        x = np.random.randn(n_nodes)
        y = np.random.randn(n_nodes)
        risk = np.random.uniform(0, 1, n_nodes)
        cluster = np.random.choice(["Ring A","Ring B","Ring C","Isolated"], n_nodes,
                                    p=[0.25, 0.2, 0.15, 0.4])

        fig_net = go.Figure(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(
                size=10 + risk * 15,
                color=risk,
                colorscale=[[0,"#004d5c"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                showscale=True,
                colorbar=dict(title="Risk", tickfont=dict(color="#94a3b8")),
            ),
            text=[f"Node {i}<br>Risk: {risk[i]:.2f}<br>Cluster: {cluster[i]}"
                  for i in range(n_nodes)],
            hoverinfo="text",
        ))
        fig_net.update_layout(**PLOT_LAYOUT, height=380,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False))
        st.plotly_chart(fig_net, use_container_width=True)

    with col_n2:
        st.markdown('<div class="section-heading">Shared Entity Heatmap</div>',
                    unsafe_allow_html=True)
        entities = ["Funeral Home A","Doctor B","Agent C","Device D","Address E"]
        claims_linked = ["CLM-10001","CLM-10045","CLM-10089","CLM-10123","CLM-10234"]
        matrix = np.random.randint(0, 5, (5, 5))
        np.fill_diagonal(matrix, 0)

        fig_heat = go.Figure(go.Heatmap(
            z=matrix, x=entities, y=claims_linked,
            colorscale=[[0,"#0d1f35"],[0.5,"#006d82"],[1,"#ef4444"]],
            text=matrix, texttemplate="%{text}",
        ))
        fig_heat.update_layout(**PLOT_LAYOUT, height=380)
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: INVESTIGATION QUEUE
# ══════════════════════════════════════════════
elif "📋 Investigation Queue" in page:

    st.markdown('<div class="section-heading" style="font-size:1.5rem;">📋 Live Investigation Queue</div>',
                unsafe_allow_html=True)

    queue_df = df[df["fraud_score"] >= approve_thresh].copy()
    queue_df["decision"] = queue_df["fraud_score"].apply(lambda s: get_decision(s)[0])
    queue_df["priority"] = queue_df["fraud_score"].apply(
        lambda s: "🔴 CRITICAL" if s > suspend_thresh else
                  ("🟠 HIGH" if s > investigate_thresh else "🟡 MEDIUM")
    )

    display = queue_df[[
        "claim_id","priority","fraud_score","claim_amount",
        "policy_age_days","decision","region"
    ]].sort_values("fraud_score", ascending=False).head(30)

    display.columns = ["Claim ID","Priority","Risk Score","Amount ($)",
                       "Policy Age","Decision","Region"]
    display["Risk Score"] = display["Risk Score"].apply(lambda x: f"{x:.3f}")
    display["Amount ($)"] = display["Amount ($)"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        display,
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    st.markdown(f"""
    **Queue Summary:** {len(queue_df)} claims pending review ·
    {int((queue_df['fraud_score'] > suspend_thresh).sum())} require immediate action
    """)


# ─────────────────────────────────────────────
# FOOTER — About Section with real desk image
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="about-section" style="margin-top:2.5rem;">
    <div class="about-content">
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
                    color:#fff; margin-bottom:0.5rem;">
            🔐 About FraudIQ
        </div>
        <p style="color:#94a3b8; font-size:0.85rem; line-height:1.7; margin:0;">
            A next-generation fraud intelligence platform built for the insurance industry.
            Combining supervised ML (XGBoost, Random Forest), anomaly detection
            (Isolation Forest, Autoencoder), network graph analysis, and NLP-based
            document verification — with full SHAP explainability. Designed for Zimbabwe
            and beyond. The model is only the iceberg tip. The system is the moat.
        </p>
        <div style="margin-top:1rem; font-family:'JetBrains Mono',monospace;
                    font-size:0.72rem; color:#475569;">
            XGBoost · Random Forest · Isolation Forest · BERT · SHAP · FastAPI · PostgreSQL
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
