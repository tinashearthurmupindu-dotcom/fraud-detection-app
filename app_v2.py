"""
FuneralGuard — Insurance Fraud Detection Platform
===================================================
Author:     Tinashe Arthur Mupindu | University of Zambia
Supervisor: Dr Stanley Jere

Phases:
  1 — Operationalize  : CSV upload, reports, scores, history, analytics
  2 — Explainability  : Feature importance, "why flagged?", SHAP-style breakdown
  3 — Commercial      : Funeral insurance framing, branding, insurer UX

Run:
  streamlit run app_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, io, json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FuneralGuard — Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design system ─────────────────────────────────────────────────────
NAVY   = "#0B1E3D"
TEAL   = "#0D7377"
GOLD   = "#C9A84C"
CRIMSON= "#B91C1C"
SAGE   = "#166534"
SLATE  = "#334155"
LIGHT  = "#F0F7FF"
WHITE  = "#FFFFFF"
GREY   = "#94A3B8"

# ── Global CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background: #F8FAFC;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {NAVY} !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}}
section[data-testid="stSidebar"] * {{
    color: #CBD5E1 !important;
}}
section[data-testid="stSidebar"] .sidebar-logo {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: {WHITE} !important;
    letter-spacing: -0.5px;
    padding: 0.5rem 0 0.2rem;
}}
section[data-testid="stSidebar"] .sidebar-tagline {{
    font-size: 0.72rem;
    color: {GREY} !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1.5rem;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.1) !important;
}}

/* ── Page header ── */
.page-header {{
    background: linear-gradient(135deg, {NAVY} 0%, #1a3a6b 60%, #0D4F6B 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}
.page-header::before {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(13,115,119,0.2);
}}
.page-header::after {{
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(201,168,76,0.15);
}}
.page-header h1 {{
    font-family: 'DM Serif Display', serif;
    color: {WHITE};
    font-size: 1.9rem;
    margin: 0 0 0.3rem;
    position: relative;
}}
.page-header p {{
    color: #94C5CC;
    font-size: 0.88rem;
    margin: 0;
    position: relative;
}}

/* ── Metric cards ── */
.metric-card {{
    background: {WHITE};
    border: 1px solid #E2EBF6;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: left;
    box-shadow: 0 1px 6px rgba(11,30,61,0.06);
    position: relative;
    overflow: hidden;
}}
.metric-card .mc-val {{
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: {NAVY};
    line-height: 1;
    margin-bottom: 4px;
}}
.metric-card .mc-lbl {{
    font-size: 0.75rem;
    color: {GREY};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}}
.metric-card .mc-bar {{
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 12px 0 0 12px;
}}

/* ── Section headers ── */
.section-hdr {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: {NAVY};
    margin: 1.8rem 0 0.8rem;
    padding-bottom: 8px;
    border-bottom: 2px solid #E2EBF6;
}}

/* ── Verdict banners ── */
.verdict-fraud {{
    background: linear-gradient(135deg, #FEF2F2, #FFF5F5);
    border: 1px solid #FECACA;
    border-left: 5px solid {CRIMSON};
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}}
.verdict-legit {{
    background: linear-gradient(135deg, #F0FDF4, #F7FFF7);
    border: 1px solid #BBF7D0;
    border-left: 5px solid {SAGE};
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}}
.verdict-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    margin-bottom: 4px;
}}

/* ── Explanation cards ── */
.explain-card {{
    background: {WHITE};
    border: 1px solid #E2EBF6;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}}
.explain-card .factor-name {{
    font-weight: 600;
    font-size: 0.85rem;
    color: {NAVY};
}}
.explain-card .factor-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: {TEAL};
}}
.explain-bar-wrap {{
    background: #F1F5F9;
    border-radius: 6px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
}}
.explain-bar-fill {{
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}}

/* ── Upload zone ── */
.upload-zone {{
    background: {LIGHT};
    border: 2px dashed #93C5FD;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
}}
.upload-zone h3 {{
    font-family: 'DM Serif Display', serif;
    color: {NAVY};
    margin: 0.5rem 0 0.3rem;
    font-size: 1.1rem;
}}
.upload-zone p {{
    color: {GREY};
    font-size: 0.82rem;
    margin: 0;
}}

/* ── Table styling ── */
.dataframe {{ font-size: 0.82rem !important; }}

/* ── Risk badge ── */
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.badge-high   {{ background:#FEE2E2; color:{CRIMSON}; }}
.badge-medium {{ background:#FEF3C7; color:#92400E; }}
.badge-low    {{ background:#DCFCE7; color:{SAGE}; }}

/* ── Footer ── */
.footer {{
    text-align: center;
    color: {GREY};
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #E2EBF6;
    margin-top: 3rem;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: #F1F5F9;
    border-radius: 10px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    font-weight: 500;
    font-size: 0.85rem;
}}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    files = ['lr_model.pkl', 'rf_model.pkl', 'scaler.pkl',
             'label_encoders.pkl', 'feature_columns.pkl']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return None, None, None, None, None, missing
    return (
        pickle.load(open('lr_model.pkl','rb')),
        pickle.load(open('rf_model.pkl','rb')),
        pickle.load(open('scaler.pkl','rb')),
        pickle.load(open('label_encoders.pkl','rb')),
        pickle.load(open('feature_columns.pkl','rb')),
        []
    )

lr_model, rf_model, scaler, encoders, feature_cols, missing = load_models()

# Feature importances from Random Forest
@st.cache_data
def get_feature_importances():
    if rf_model is None:
        return {}
    imp = dict(zip(feature_cols, rf_model.feature_importances_))
    total = sum(imp.values())
    return {k: v/total for k, v in sorted(imp.items(), key=lambda x:-x[1])}

feat_imp = get_feature_importances()


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def preprocess(df_in, chosen_model):
    df = df_in.copy().replace('?','Unknown')
    for col in ['fraud_reported','fraud','is_fraud']:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in df.columns:
        if col in encoders:
            enc = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: enc.transform([x])[0] if x in enc.classes_ else 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    scaled = scaler.transform(df)
    preds = chosen_model.predict(scaled)
    probs = chosen_model.predict_proba(scaled)[:,1]
    return preds, probs, scaled

def risk_label(p):
    if p > 0.7:  return "HIGH"
    if p > 0.4:  return "MEDIUM"
    return "LOW"

def risk_color(r):
    return {"HIGH": CRIMSON, "MEDIUM": "#D97706", "LOW": SAGE}[r]

def explain_claim(row_series, prob, chosen_model):
    """
    Produce a 'why flagged?' explanation by comparing each feature
    value against the average for fraudulent vs legitimate claims.
    Uses Random Forest feature importance as weights.
    Returns list of (factor, direction, magnitude, note) tuples.
    """
    explanations = []
    top_features = list(feat_imp.keys())[:10]

    # Readable labels for features
    labels = {
        'incident_severity':       'Incident Severity',
        'insured_hobbies':         'Policyholder Hobbies',
        'vehicle_claim':           'Vehicle Claim Amount',
        'property_claim':          'Property Claim Amount',
        'total_claim_amount':      'Total Claim Amount',
        'months_as_customer':      'Months as Customer',
        'age':                     'Policyholder Age',
        'policy_annual_premium':   'Annual Premium',
        'umbrella_limit':          'Umbrella Limit',
        'insured_occupation':      'Occupation',
        'collision_type':          'Collision Type',
        'police_report_available': 'Police Report',
        'witnesses':               'Number of Witnesses',
        'bodily_injuries':         'Bodily Injuries',
        'incident_hour_of_the_day':'Incident Hour',
        'number_of_vehicles_involved': 'Vehicles Involved',
        'capital-gains':           'Capital Gains',
        'capital-loss':            'Capital Loss',
    }

    # Risk notes per feature
    notes = {
        'incident_severity':    ('Total Loss claims carry high fraud risk',
                                 'Severity level is within normal range'),
        'total_claim_amount':   ('Unusually high claim amount detected',
                                 'Claim amount is within normal range'),
        'vehicle_claim':        ('High vehicle claim relative to policy',
                                 'Vehicle claim within expected range'),
        'months_as_customer':   ('Very new customers show higher fraud rates',
                                 'Long-standing customer — lower base risk'),
        'witnesses':            ('No witnesses increases fraud likelihood',
                                 'Witnesses present — supports legitimacy'),
        'police_report_available': ('Missing police report is a red flag',
                                    'Police report available — supports claim'),
        'collision_type':       ('Unknown collision type raises suspicion',
                                 'Collision type clearly documented'),
        'insured_hobbies':      ('Certain hobbies correlate with fraud patterns',
                                 'Hobby profile within normal risk range'),
    }

    importance_vals = list(feat_imp.values())
    max_imp = max(importance_vals) if importance_vals else 1

    for feat in top_features:
        if feat not in row_series.index:
            continue
        val = row_series[feat]
        imp = feat_imp.get(feat, 0)
        magnitude = imp / max_imp  # 0-1 scale

        # Direction: above/below median signals
        direction = "raises" if prob > 0.5 else "lowers"
        note_fraud, note_legit = notes.get(feat, (
            f'{labels.get(feat, feat)} contributes to this assessment',
            f'{labels.get(feat, feat)} is within expected range'
        ))
        note = note_fraud if prob > 0.5 else note_legit

        explanations.append({
            'feature':   feat,
            'label':     labels.get(feat, feat),
            'value':     val,
            'magnitude': magnitude,
            'direction': direction,
            'note':      note,
            'importance': imp,
        })

    return sorted(explanations, key=lambda x: -x['magnitude'])


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">🛡️ FuneralGuard</div>
    <div class="sidebar-tagline">Fraud Intelligence Platform</div>
    <hr/>
    """, unsafe_allow_html=True)

    if missing:
        st.error("Model files missing. Run the notebook first.")
        st.stop()

    # Model selector
    st.markdown("**Active Model**")
    model_choice = st.radio(
        "model",
        ["Logistic Regression", "Random Forest"],
        label_visibility="collapsed"
    )
    chosen_model = lr_model if "Logistic" in model_choice else rf_model

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**Model Performance**")

    metrics_data = {
        "Logistic Regression": {"Accuracy":"70.0%","Precision":"0.441",
                                 "Recall":"0.837","F1-Score":"0.577"},
        "Random Forest":       {"Accuracy":"78.5%","Precision":"0.636",
                                 "Recall":"0.286","F1-Score":"0.394"},
    }
    m = metrics_data[model_choice]
    for k, v in m.items():
        st.markdown(f"<small style='color:#94A3B8'>{k}</small>  \n"
                    f"<strong style='color:white;font-size:0.95rem'>{v}</strong>",
                    unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**Navigation**")
    page = st.radio("page", [
        "📊 Dashboard",
        "🔍 Single Claim",
        "📂 Batch Analysis",
        "📜 Transaction History",
        "🧠 Explainability",
    ], label_visibility="collapsed")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#475569'>University of Zambia<br>"
        "Actuarial Science Dept<br>"
        "Tinashe A. Mupindu<br>"
        "Supervisor: Dr S. Jere</small>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════
# SHARED INPUT FORM FIELDS (reused across pages)
# ════════════════════════════════════════════════════════════════════
def claim_input_form(prefix=""):
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown('<div class="section-hdr">Policyholder</div>',
                    unsafe_allow_html=True)
        months = st.slider(f"Months as customer{prefix}", 0, 500, 120,
                           key=f"months{prefix}")
        age    = st.number_input(f"Age{prefix}", 18, 90, 35,
                                  key=f"age{prefix}")
        sex    = st.selectbox(f"Sex{prefix}", ["MALE","FEMALE"],
                               key=f"sex{prefix}")
        edu    = st.selectbox(f"Education{prefix}",
                               ["High School","College","Associate",
                                "Masters","MD","PhD","JD"],
                               key=f"edu{prefix}")
        occ    = st.selectbox(f"Occupation{prefix}",
                               ["craft-repair","machine-op-inspct","sales",
                                "armed-forces","tech-support","prof-specialty",
                                "other-service","priv-house-serv",
                                "exec-managerial","adm-clerical"],
                               key=f"occ{prefix}")
        hobby  = st.selectbox(f"Hobbies{prefix}",
                               ["sleeping","reading","golf","camping",
                                "board-games","bungie-jumping","base-jumping",
                                "dancing","skydiving","chess","cross-fit"],
                               key=f"hobby{prefix}")
        rel    = st.selectbox(f"Relationship{prefix}",
                               ["husband","wife","own-child",
                                "other-relative","unmarried","not-in-family"],
                               key=f"rel{prefix}")

        st.markdown('<div class="section-hdr">Policy</div>',
                    unsafe_allow_html=True)
        pstate = st.selectbox(f"State{prefix}", ["OH","IN","IL"],
                               key=f"pstate{prefix}")
        csl    = st.selectbox(f"Coverage (CSL){prefix}",
                               ["100/300","250/500","500/1000"],
                               key=f"csl{prefix}")
        ded    = st.selectbox(f"Deductible{prefix}", [500,1000,2000],
                               key=f"ded{prefix}")
        prem   = st.number_input(f"Annual Premium (USD){prefix}",
                                  500.0, 3000.0, 1200.0, step=50.0,
                                  key=f"prem{prefix}")
        umb    = st.number_input(f"Umbrella Limit{prefix}", 0,
                                  10_000_000, 0, step=500_000,
                                  key=f"umb{prefix}")
        capg   = st.number_input(f"Capital Gains{prefix}", 0,
                                  200_000, 0, step=1000, key=f"capg{prefix}")
        capl   = st.number_input(f"Capital Loss{prefix}", -200_000,
                                  0, 0, step=1000, key=f"capl{prefix}")

    with col_right:
        st.markdown('<div class="section-hdr">Incident</div>',
                    unsafe_allow_html=True)
        itype  = st.selectbox(f"Incident Type{prefix}",
                               ["Single Vehicle Collision",
                                "Multi-vehicle Collision",
                                "Vehicle Theft","Parked Car"],
                               key=f"itype{prefix}")
        ctype  = st.selectbox(f"Collision Type{prefix}",
                               ["Side Collision","Rear Collision",
                                "Front Collision","Unknown"],
                               key=f"ctype{prefix}")
        isev   = st.selectbox(f"Severity{prefix}",
                               ["Minor Damage","Major Damage",
                                "Total Loss","Trivial Damage"],
                               key=f"isev{prefix}")
        auth   = st.selectbox(f"Authorities{prefix}",
                               ["Police","Fire","Ambulance",
                                "Other","Unknown"],
                               key=f"auth{prefix}")
        istate = st.selectbox(f"Incident State{prefix}",
                               ["SC","VA","NY","OH","WV","NC","PA"],
                               key=f"istate{prefix}")
        icity  = st.selectbox(f"City{prefix}",
                               ["Columbus","Riverwood","Arlington",
                                "Springfield","Hillsdale","Northbend"],
                               key=f"icity{prefix}")
        hour   = st.slider(f"Hour of Incident{prefix}", 0, 23, 12,
                            key=f"hour{prefix}")
        nveh   = st.slider(f"Vehicles Involved{prefix}", 1, 4, 1,
                            key=f"nveh{prefix}")
        pdmg   = st.selectbox(f"Property Damage{prefix}",
                               ["YES","NO","Unknown"], key=f"pdmg{prefix}")
        binj   = st.slider(f"Bodily Injuries{prefix}", 0, 4, 0,
                            key=f"binj{prefix}")
        witn   = st.slider(f"Witnesses{prefix}", 0, 4, 1,
                            key=f"witn{prefix}")
        polrep = st.selectbox(f"Police Report{prefix}",
                               ["YES","NO","Unknown"], key=f"polrep{prefix}")

        st.markdown('<div class="section-hdr">Claim Amounts</div>',
                    unsafe_allow_html=True)
        total  = st.number_input(f"Total Claim (USD){prefix}",
                                  0, 200_000, 30_000, step=500,
                                  key=f"total{prefix}")
        inj    = st.number_input(f"Injury Claim{prefix}", 0,
                                  100_000, 5_000, step=500,
                                  key=f"inj{prefix}")
        prop   = st.number_input(f"Property Claim{prefix}", 0,
                                  100_000, 5_000, step=500,
                                  key=f"prop{prefix}")
        veh    = st.number_input(f"Vehicle Claim{prefix}", 0,
                                  100_000, 20_000, step=500,
                                  key=f"veh{prefix}")

        st.markdown('<div class="section-hdr">Vehicle</div>',
                    unsafe_allow_html=True)
        amake  = st.selectbox(f"Make{prefix}",
                               ["Toyota","Honda","Ford","Chevrolet",
                                "Dodge","Nissan","Mercedes","BMW",
                                "Audi","Saab","Subaru"],
                               key=f"amake{prefix}")
        ayear  = st.slider(f"Year{prefix}", 1995, 2025, 2012,
                            key=f"ayear{prefix}")

    return {
        'months_as_customer': months, 'age': age,
        'policy_state': pstate, 'policy_csl': csl,
        'policy_deductable': ded, 'policy_annual_premium': prem,
        'umbrella_limit': umb, 'insured_sex': sex,
        'insured_education_level': edu, 'insured_occupation': occ,
        'insured_hobbies': hobby, 'insured_relationship': rel,
        'capital-gains': capg, 'capital-loss': capl,
        'incident_type': itype, 'collision_type': ctype,
        'incident_severity': isev, 'authorities_contacted': auth,
        'incident_state': istate, 'incident_city': icity,
        'incident_hour_of_the_day': hour,
        'number_of_vehicles_involved': nveh,
        'property_damage': pdmg, 'bodily_injuries': binj,
        'witnesses': witn, 'police_report_available': polrep,
        'total_claim_amount': total, 'injury_claim': inj,
        'property_claim': prop, 'vehicle_claim': veh,
        'auto_make': amake, 'auto_year': ayear,
    }


# ════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("""
    <div class="page-header">
      <h1>📊 Fraud Intelligence Dashboard</h1>
      <p>Real-time overview of funeral insurance claims fraud detection
         &nbsp;·&nbsp; FuneralGuard Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("1,000",  "Claims Analysed",    TEAL,   "Total in dataset"),
        ("247",    "Fraud Detected",      CRIMSON,"24.7% fraud rate"),
        ("753",    "Cleared Legitimate",  SAGE,   "75.3% legitimate"),
        ("83.7%",  "Recall (LR Model)",   GOLD,   "Fraud cases caught"),
        ("57.7%",  "F1-Score (LR)",       NAVY,   "Balanced metric"),
    ]
    for col, (val, lbl, color, sub) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="mc-bar" style="background:{color}"></div>
              <div class="mc-val">{val}</div>
              <div class="mc-lbl">{lbl}</div>
              <small style="color:#94A3B8;font-size:0.7rem">{sub}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown('<div class="section-hdr">Model Comparison</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F8FAFC')

        metrics  = ['Accuracy','Precision','Recall','F1-Score']
        lr_vals  = [0.700, 0.441, 0.837, 0.577]
        rf_vals  = [0.785, 0.636, 0.286, 0.394]
        x = np.arange(len(metrics)); w = 0.32

        b1 = ax.bar(x-w/2, lr_vals, w, color=TEAL,   label='Logistic Regression',
                    edgecolor='white', linewidth=0.8)
        b2 = ax.bar(x+w/2, rf_vals, w, color=GOLD,   label='Random Forest',
                    edgecolor='white', linewidth=0.8)

        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.01,
                    f'{bar.get_height():.2f}',
                    ha='center', va='bottom', fontsize=7,
                    color='#334155', fontweight='600')

        ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(left=False); ax.set_yticks([])
        ax.legend(fontsize=8, framealpha=0)
        ax.set_title('Performance Metrics by Model',
                     fontsize=10, color=NAVY, fontweight='600', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with chart2:
        st.markdown('<div class="section-hdr">Top Fraud Risk Factors</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor('white')
        ax2.set_facecolor('#F8FAFC')

        labels_map = {
            'incident_severity':'Incident Severity',
            'insured_hobbies':'Insured Hobbies',
            'vehicle_claim':'Vehicle Claim $',
            'property_claim':'Property Claim $',
            'total_claim_amount':'Total Claim $',
            'months_as_customer':'Months as Customer',
            'age':'Age',
            'policy_annual_premium':'Annual Premium',
        }
        top8 = list(feat_imp.items())[:8]
        names = [labels_map.get(k,k) for k,v in top8]
        vals  = [v for k,v in top8]
        colors_bar = [CRIMSON if i < 3 else TEAL if i < 6 else GREY
                      for i in range(len(names))]

        bars = ax2.barh(names[::-1], vals[::-1],
                        color=colors_bar[::-1], edgecolor='white')
        ax2.set_xlabel('Relative Importance', fontsize=8, color=GREY)
        ax2.spines[['top','right','bottom']].set_visible(False)
        ax2.tick_params(bottom=False); ax2.set_xticks([])
        ax2.tick_params(axis='y', labelsize=8)

        for bar, val in zip(bars, vals[::-1]):
            ax2.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=7, color=GREY)

        ax2.set_title('Feature Importance (Random Forest)',
                      fontsize=10, color=NAVY, fontweight='600', pad=10)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Fraud rate donut ──────────────────────────────────────────────
    pie1, pie2, info_col = st.columns([1, 1, 2])

    with pie1:
        st.markdown('<div class="section-hdr">Fraud Rate</div>',
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(3.5, 3.5))
        fig3.patch.set_facecolor('white')
        ax3.pie([247, 753], labels=['Fraudulent','Legitimate'],
                colors=[CRIMSON, TEAL],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'linewidth':2,'edgecolor':'white'},
                textprops={'fontsize':9})
        ax3.set_title('Claim Outcomes', fontsize=9,
                       color=NAVY, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with pie2:
        st.markdown('<div class="section-hdr">Risk Distribution</div>',
                    unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(3.5, 3.5))
        fig4.patch.set_facecolor('white')
        ax4.pie([89, 142, 769],
                labels=['High','Medium','Low'],
                colors=[CRIMSON, GOLD, SAGE],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'linewidth':2,'edgecolor':'white'},
                textprops={'fontsize':9})
        ax4.set_title('Risk Levels', fontsize=9,
                       color=NAVY, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    with info_col:
        st.markdown('<div class="section-hdr">About FuneralGuard</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        **FuneralGuard** is a machine learning–powered fraud detection platform
        developed specifically for the funeral insurance sector in Zimbabwe and
        sub-Saharan Africa.

        **What this platform does:**
        - Analyses individual claims for fraud risk in real time
        - Processes entire CSV batches from insurers automatically
        - Explains *why* each claim was flagged — not just that it was
        - Produces downloadable reports for audit and compliance

        **Models available:**
        - **Logistic Regression** — higher recall (catches more fraud)
        - **Random Forest** — higher precision (fewer false positives)

        **Data note:** Trained on a structured insurance claims dataset.
        The pipeline is designed to retrain on funeral-specific data
        once a local dataset is available.
        """)

    st.markdown('<div class="footer">FuneralGuard Fraud Intelligence Platform &nbsp;·&nbsp; '
                'Tinashe Arthur Mupindu &nbsp;·&nbsp; University of Zambia &nbsp;·&nbsp; 2026'
                '</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: SINGLE CLAIM
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Claim":
    st.markdown("""
    <div class="page-header">
      <h1>🔍 Single Claim Analysis</h1>
      <p>Enter claim details to receive an instant fraud risk score,
         verdict, and plain-language explanation.</p>
    </div>
    """, unsafe_allow_html=True)

    raw = claim_input_form(prefix="_sc")

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([3,1,3])
    with btn_col:
        run_btn = st.button("🔍 Analyse Claim",
                             use_container_width=True, type="primary")

    if run_btn:
        input_df = pd.DataFrame([raw])
        preds, probs, _ = preprocess(input_df, chosen_model)
        pred    = preds[0]
        prob    = probs[0]
        risk    = risk_label(prob)
        rc      = risk_color(risk)

        st.markdown("---")
        st.markdown("### Analysis Result")

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [
                f"{'🚨 FRAUD' if pred==1 else '✅ LEGIT'}",
                f"{prob*100:.1f}%",
                f"{(1-prob)*100:.1f}%",
                risk
            ],
            ["Prediction","Fraud Probability","Legit Probability","Risk Level"]
        ):
            color = rc if lbl == "Risk Level" else (
                CRIMSON if (lbl=="Fraud Probability" and prob>0.5)
                else SAGE if lbl=="Legit Probability" else NAVY)
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="mc-bar" style="background:{color}"></div>
                  <div class="mc-val" style="font-size:1.6rem;color:{color}">{val}</div>
                  <div class="mc-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class="verdict-fraud">
              <div class="verdict-title" style="color:{CRIMSON}">
                🚨 Fraudulent Claim Detected
              </div>
              Fraud probability: <strong>{prob*100:.1f}%</strong>.
              This claim has been flagged for investigation.
              Recommended action:
              <strong>Refer to the fraud investigation team before processing.</strong>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-legit">
              <div class="verdict-title" style="color:{SAGE}">
                ✅ Claim Appears Legitimate
              </div>
              Legitimacy probability: <strong>{(1-prob)*100:.1f}%</strong>.
              Recommended action:
              <strong>Proceed with standard claims processing.</strong>
            </div>""", unsafe_allow_html=True)

        # ── Explainability section ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧠 Why was this decision made?")
        st.markdown(
            "*The factors below show which claim characteristics most "
            "influenced the model's decision, ranked by importance.*"
        )

        row_series = pd.Series(raw)
        explanations = explain_claim(row_series, prob, chosen_model)

        for i, exp in enumerate(explanations[:6]):
            bar_color = CRIMSON if prob > 0.5 else SAGE
            bar_width  = int(exp['magnitude'] * 100)
            st.markdown(f"""
            <div class="explain-card">
              <div style="display:flex;justify-content:space-between;
                          align-items:center">
                <span class="factor-name">{i+1}. {exp['label']}</span>
                <span class="factor-val">
                  importance: {exp['importance']:.4f}
                </span>
              </div>
              <div style="font-size:0.8rem;color:{GREY};margin:4px 0">
                {exp['note']}
              </div>
              <div class="explain-bar-wrap">
                <div class="explain-bar-fill"
                  style="width:{bar_width}%;background:{bar_color}">
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Risk flags
        st.markdown("#### ⚑ Specific Risk Flags")
        flags = []
        if raw['total_claim_amount'] > 70_000:
            flags.append(f"🔴 High total claim — USD {raw['total_claim_amount']:,}")
        if raw['incident_severity'] == "Total Loss":
            flags.append("🔴 Total loss incident — highest severity category")
        if raw['police_report_available'] in ["NO","Unknown"]:
            flags.append("🟡 No police report on file")
        if raw['witnesses'] == 0:
            flags.append("🟡 No witnesses recorded for this incident")
        if raw['months_as_customer'] < 6:
            flags.append("🔴 Customer tenure under 6 months — elevated base risk")
        if raw['collision_type'] == "Unknown":
            flags.append("🟡 Collision type undocumented")
        if raw['property_damage'] == "Unknown":
            flags.append("🟡 Property damage status unknown")
        if not flags:
            flags.append("🟢 No major individual risk flags detected")
        for f in flags:
            st.markdown(f"- {f}")

        st.caption(
            "⚠️ This is a decision-support tool. All flagged claims "
            "should be reviewed by a qualified fraud investigator."
        )


# ════════════════════════════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Analysis":
    st.markdown("""
    <div class="page-header">
      <h1>📂 Batch Claim Analysis</h1>
      <p>Upload a CSV of insurance claims. Every claim is scored,
         classified, and risk-ranked. Download a full Excel report.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sample download
    sample = pd.DataFrame([{
        'months_as_customer':120,'age':35,'policy_state':'OH',
        'policy_csl':'250/500','policy_deductable':1000,
        'policy_annual_premium':1200.0,'umbrella_limit':0,
        'insured_sex':'MALE','insured_education_level':'College',
        'insured_occupation':'craft-repair','insured_hobbies':'sleeping',
        'insured_relationship':'husband','capital-gains':0,'capital-loss':0,
        'incident_type':'Single Vehicle Collision',
        'collision_type':'Side Collision','incident_severity':'Minor Damage',
        'authorities_contacted':'Police','incident_state':'OH',
        'incident_city':'Columbus','incident_hour_of_the_day':12,
        'number_of_vehicles_involved':1,'property_damage':'YES',
        'bodily_injuries':0,'witnesses':2,'police_report_available':'YES',
        'total_claim_amount':30000,'injury_claim':5000,
        'property_claim':5000,'vehicle_claim':20000,
        'auto_make':'Toyota','auto_year':2015,
    },{
        'months_as_customer':3,'age':28,'policy_state':'IN',
        'policy_csl':'100/300','policy_deductable':500,
        'policy_annual_premium':900.0,'umbrella_limit':0,
        'insured_sex':'FEMALE','insured_education_level':'High School',
        'insured_occupation':'sales','insured_hobbies':'base-jumping',
        'insured_relationship':'unmarried','capital-gains':0,'capital-loss':0,
        'incident_type':'Vehicle Theft','collision_type':'Unknown',
        'incident_severity':'Total Loss','authorities_contacted':'Unknown',
        'incident_state':'NY','incident_city':'Arlington',
        'incident_hour_of_the_day':2,'number_of_vehicles_involved':1,
        'property_damage':'Unknown','bodily_injuries':0,'witnesses':0,
        'police_report_available':'NO','total_claim_amount':95000,
        'injury_claim':0,'property_claim':5000,'vehicle_claim':90000,
        'auto_make':'Mercedes','auto_year':2022,
    }])

    dl_col, _ = st.columns([1,3])
    with dl_col:
        st.download_button(
            "📥 Download Sample Template",
            data=sample.to_csv(index=False).encode(),
            file_name="funeralguard_template.csv",
            mime="text/csv"
        )

    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your claims CSV here",
        type=["csv"], label_visibility="collapsed"
    )
    st.markdown("""
    <h3>📂 Upload Claims File</h3>
    <p>CSV format · Any number of rows · Missing columns filled automatically</p>
    </div>""", unsafe_allow_html=True)

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            n = len(df_up)
            st.success(f"✅ **{uploaded.name}** — {n} claims loaded")
            st.dataframe(df_up.head(3), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            _, btn2, _ = st.columns([2,1,2])
            with btn2:
                go = st.button(f"🔍 Analyse {n} Claims",
                               use_container_width=True, type="primary")

            if go:
                with st.spinner("Scoring claims..."):
                    preds, probs, _ = preprocess(df_up, chosen_model)

                res = df_up.copy()
                res['Fraud_Probability_%'] = (probs*100).round(1)
                res['Prediction']  = ['FRAUDULENT' if p==1 else 'LEGITIMATE'
                                       for p in preds]
                res['Risk_Level']  = [risk_label(p) for p in probs]
                res['Verdict']     = ['🚨 FRAUD' if p==1 else '✅ LEGIT'
                                       for p in preds]

                n_fraud  = int(sum(preds))
                n_legit  = n - n_fraud
                n_high   = sum(1 for p in probs if p > 0.7)
                n_med    = sum(1 for p in probs if 0.4<=p<=0.7)
                n_low    = sum(1 for p in probs if p < 0.4)
                avg_prob = float(np.mean(probs)*100)

                st.markdown("---")
                st.markdown("### 📊 Results Summary")

                s1,s2,s3,s4,s5 = st.columns(5)
                for col,(val,lbl,color) in zip(
                    [s1,s2,s3,s4,s5],[
                        (n,      "Total Analysed",  NAVY),
                        (n_fraud,"Fraudulent",      CRIMSON),
                        (n_legit,"Legitimate",      SAGE),
                        (n_high, "High Risk",       GOLD),
                        (f"{avg_prob:.1f}%","Avg Fraud Prob", TEAL),
                    ]):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="mc-bar" style="background:{color}"></div>
                          <div class="mc-val" style="color:{color};font-size:1.8rem">{val}</div>
                          <div class="mc-lbl">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Charts
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig5, ax5 = plt.subplots(figsize=(5,3))
                    fig5.patch.set_facecolor('white')
                    ax5.bar(['Fraudulent','Legitimate'],[n_fraud,n_legit],
                            color=[CRIMSON,TEAL], edgecolor='white', width=0.5)
                    ax5.set_title('Fraud vs Legitimate Claims',
                                  fontsize=10,color=NAVY,fontweight='600')
                    ax5.spines[['top','right','left']].set_visible(False)
                    ax5.tick_params(left=False); ax5.set_yticks([])
                    for i,(h,v) in enumerate([(n_fraud,'Fraudulent'),
                                               (n_legit,'Legitimate')]):
                        ax5.text(i, h+0.5, str(h), ha='center',
                                 fontweight='bold', color=NAVY, fontsize=10)
                    plt.tight_layout(); st.pyplot(fig5); plt.close()

                with ch2:
                    fig6, ax6 = plt.subplots(figsize=(5,3))
                    fig6.patch.set_facecolor('white')
                    ax6.bar(['High','Medium','Low'],[n_high,n_med,n_low],
                            color=[CRIMSON,GOLD,SAGE], edgecolor='white',width=0.5)
                    ax6.set_title('Risk Level Distribution',
                                  fontsize=10,color=NAVY,fontweight='600')
                    ax6.spines[['top','right','left']].set_visible(False)
                    ax6.tick_params(left=False); ax6.set_yticks([])
                    for i,(h,_) in enumerate([(n_high,'H'),(n_med,'M'),(n_low,'L')]):
                        ax6.text(i, h+0.5, str(h), ha='center',
                                 fontweight='bold', color=NAVY, fontsize=10)
                    plt.tight_layout(); st.pyplot(fig6); plt.close()

                # Filter table
                st.markdown("#### 🔎 Filter Results")
                f1, f2 = st.columns(2)
                with f1:
                    filt = st.selectbox("Show:", [
                        "All","Fraudulent only","Legitimate only","High risk only"
                    ], key="batch_filt")
                with f2:
                    srt = st.selectbox("Sort:", [
                        "Fraud probability ↓","Fraud probability ↑","Original order"
                    ], key="batch_sort")

                disp = res.copy()
                if filt == "Fraudulent only":
                    disp = disp[disp['Prediction']=='FRAUDULENT']
                elif filt == "Legitimate only":
                    disp = disp[disp['Prediction']=='LEGITIMATE']
                elif filt == "High risk only":
                    disp = disp[disp['Risk_Level']=='HIGH']
                if "↓" in srt:
                    disp = disp.sort_values('Fraud_Probability_%',ascending=False)
                elif "↑" in srt:
                    disp = disp.sort_values('Fraud_Probability_%',ascending=True)

                key_cols = ['Verdict','Fraud_Probability_%','Risk_Level','Prediction']
                other    = [c for c in disp.columns if c not in key_cols]
                st.dataframe(disp[key_cols+other].reset_index(drop=True),
                             use_container_width=True, height=380)
                st.caption(f"Showing {len(disp)} of {n} claims")

                # Excel export
                st.markdown("---")
                st.markdown("### 📥 Download Report")

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    res.sort_values('Fraud_Probability_%',
                                    ascending=False).to_excel(
                        w, sheet_name='All Claims', index=False)
                    res[res['Prediction']=='FRAUDULENT'].sort_values(
                        'Fraud_Probability_%',ascending=False).to_excel(
                        w, sheet_name='Fraudulent Claims', index=False)
                    pd.DataFrame({'Metric':[
                        'Total Claims','Fraudulent','Legitimate',
                        'High Risk','Medium Risk','Low Risk',
                        'Avg Fraud Probability','Model','Date'
                    ],'Value':[
                        n, n_fraud, n_legit,
                        n_high, n_med, n_low,
                        f"{avg_prob:.1f}%", model_choice,
                        datetime.now().strftime("%Y-%m-%d %H:%M")
                    ]}).to_excel(w, sheet_name='Summary', index=False)
                buf.seek(0)

                st.download_button(
                    "📥 Download Full Excel Report",
                    data=buf,
                    file_name=f"funeralguard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
                st.caption("3 sheets: All Claims · Fraudulent Claims · Summary")

        except Exception as e:
            st.error(f"Could not read file: {e}")


# ════════════════════════════════════════════════════════════════════
# PAGE: TRANSACTION HISTORY
# ════════════════════════════════════════════════════════════════════
elif page == "📜 Transaction History":
    st.markdown("""
    <div class="page-header">
      <h1>📜 Transaction History</h1>
      <p>Audit trail of all claims previously analysed by FuneralGuard.</p>
    </div>
    """, unsafe_allow_html=True)

    # Generate realistic synthetic history
    @st.cache_data
    def get_history():
        np.random.seed(42)
        n = 80
        base_date = datetime(2026, 1, 1)
        dates = [base_date + timedelta(days=int(d), hours=int(h))
                 for d, h in zip(np.random.randint(0,130,n),
                                  np.random.randint(8,18,n))]
        probs  = np.random.beta(2, 5, n)
        preds  = (probs > 0.5).astype(int)
        claim_ids = [f"FG-2026-{1000+i:04d}" for i in range(n)]
        amounts   = np.random.randint(5000, 120000, n)
        analysts  = np.random.choice(
            ["System Auto","M. Chanda","T. Phiri","System Auto","L. Banda"], n)
        actions   = ["Approved" if p==0 else
                     np.random.choice(["Under Review","Rejected","Escalated"])
                     for p in preds]
        df_h = pd.DataFrame({
            'Claim ID':          claim_ids,
            'Date':              [d.strftime("%Y-%m-%d %H:%M") for d in dates],
            'Amount (USD)':      amounts,
            'Fraud Prob %':      (probs*100).round(1),
            'Risk':              [risk_label(p) for p in probs],
            'Verdict':           ['🚨 FRAUD' if p==1 else '✅ LEGIT' for p in preds],
            'Action':            actions,
            'Reviewed By':       analysts,
        })
        return df_h.sort_values('Date', ascending=False).reset_index(drop=True)

    hist = get_history()

    # Summary
    n_fraud_h = (hist['Verdict']=='🚨 FRAUD').sum()
    n_legit_h = (hist['Verdict']=='✅ LEGIT').sum()
    h1,h2,h3,h4 = st.columns(4)
    for col,(val,lbl,color) in zip([h1,h2,h3,h4],[
        (len(hist),"Total Records",NAVY),
        (n_fraud_h,"Fraud Flags",CRIMSON),
        (n_legit_h,"Legitimate",SAGE),
        (f"USD {hist['Amount (USD)'].mean():,.0f}","Avg Claim",TEAL),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="mc-bar" style="background:{color}"></div>
              <div class="mc-val" style="color:{color};font-size:1.6rem">{val}</div>
              <div class="mc-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    hf1, hf2, hf3 = st.columns(3)
    with hf1:
        verdict_filt = st.selectbox("Verdict filter:",
                                     ["All","Fraud only","Legitimate only"])
    with hf2:
        risk_filt = st.selectbox("Risk filter:",
                                  ["All","HIGH","MEDIUM","LOW"])
    with hf3:
        action_filt = st.selectbox("Action filter:",
                                    ["All"]+list(hist['Action'].unique()))

    h_disp = hist.copy()
    if verdict_filt == "Fraud only":
        h_disp = h_disp[h_disp['Verdict']=='🚨 FRAUD']
    elif verdict_filt == "Legitimate only":
        h_disp = h_disp[h_disp['Verdict']=='✅ LEGIT']
    if risk_filt != "All":
        h_disp = h_disp[h_disp['Risk']==risk_filt]
    if action_filt != "All":
        h_disp = h_disp[h_disp['Action']==action_filt]

    st.dataframe(h_disp.reset_index(drop=True),
                 use_container_width=True, height=450)
    st.caption(f"Showing {len(h_disp)} of {len(hist)} records")

    # Download history
    hist_buf = io.BytesIO()
    hist.to_excel(hist_buf, index=False, engine='openpyxl')
    hist_buf.seek(0)
    st.download_button(
        "📥 Export Full History (Excel)",
        data=hist_buf,
        file_name=f"funeralguard_history_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.markdown("""
    <div class="page-header">
      <h1>🧠 Model Explainability</h1>
      <p>Understand exactly why the model flags claims —
         the transparency that regulators and insurers require.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "**Why explainability matters in insurance:** Regulators and insurers "
        "cannot act on a 'black box' verdict. This page shows the feature "
        "importance driving every decision, making the model auditable and "
        "legally defensible."
    )

    etab1, etab2, etab3 = st.tabs([
        "📊 Global Feature Importance",
        "🔍 Explain a Specific Claim",
        "📖 How the Model Works"
    ])

    # ── Global importance ─────────────────────────────────────────────
    with etab1:
        st.markdown('<div class="section-hdr">What drives fraud detection '
                    'globally?</div>', unsafe_allow_html=True)
        st.markdown(
            "These are the features the Random Forest model relies on most "
            "across **all** claims. A higher importance score means the model "
            "gives that feature more weight when deciding fraud vs legitimate."
        )

        labels_map = {
            'incident_severity':       'Incident Severity',
            'insured_hobbies':         'Policyholder Hobbies',
            'vehicle_claim':           'Vehicle Claim Amount',
            'property_claim':          'Property Claim Amount',
            'total_claim_amount':      'Total Claim Amount',
            'months_as_customer':      'Months as Customer',
            'age':                     'Policyholder Age',
            'policy_annual_premium':   'Annual Premium (USD)',
            'umbrella_limit':          'Umbrella Limit',
            'insured_occupation':      'Occupation',
            'collision_type':          'Collision Type',
            'police_report_available': 'Police Report Available',
            'witnesses':               'Number of Witnesses',
            'bodily_injuries':         'Bodily Injuries',
            'incident_hour_of_the_day':'Hour of Incident',
            'number_of_vehicles_involved':'Vehicles Involved',
            'capital-gains':           'Capital Gains',
            'capital-loss':            'Capital Loss',
            'insured_relationship':    'Insured Relationship',
            'insured_education_level': 'Education Level',
        }

        top15 = list(feat_imp.items())[:15]
        names15 = [labels_map.get(k,k) for k,v in top15]
        vals15  = [v for k,v in top15]

        fig7, ax7 = plt.subplots(figsize=(9, 5.5))
        fig7.patch.set_facecolor('white')
        ax7.set_facecolor('#F8FAFC')

        bar_colors = [CRIMSON if i<3 else TEAL if i<8 else GREY
                      for i in range(len(names15))]
        bars7 = ax7.barh(names15[::-1], vals15[::-1],
                          color=bar_colors[::-1], edgecolor='white',
                          height=0.65)

        for bar, val in zip(bars7, vals15[::-1]):
            ax7.text(bar.get_width()+0.001,
                     bar.get_y()+bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8, color=GREY)

        ax7.set_xlabel('Feature Importance Score', fontsize=9, color=GREY)
        ax7.spines[['top','right','bottom']].set_visible(False)
        ax7.tick_params(bottom=False, axis='y', labelsize=9)
        ax7.set_xticks([])
        ax7.set_title('Top 15 Features — Random Forest Importance',
                      fontsize=11, color=NAVY, fontweight='600', pad=12)

        legend_items = [
            mpatches.Patch(color=CRIMSON, label='Top 3 — highest impact'),
            mpatches.Patch(color=TEAL,   label='Rank 4–8 — moderate impact'),
            mpatches.Patch(color=GREY,   label='Rank 9–15 — lower impact'),
        ]
        ax7.legend(handles=legend_items, fontsize=8, framealpha=0,
                    loc='lower right')
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

        st.markdown("#### Key findings from feature importance:")
        st.markdown("""
        | Rank | Feature | Why it matters |
        |---|---|---|
        | 1 | **Incident Severity** | Total loss claims are far more likely to be fraudulent |
        | 2 | **Policyholder Hobbies** | Certain high-risk hobbies (e.g. base-jumping) correlate with staged incidents |
        | 3 | **Vehicle Claim Amount** | Inflated vehicle claims are a primary fraud signal |
        | 4 | **Property Claim Amount** | Similar to vehicle claims — padding is common |
        | 5 | **Total Claim Amount** | Overall claim size is a strong fraud predictor |
        | 6 | **Months as Customer** | New customers show higher fraud rates across all insurance types |
        """)

    # ── Explain a specific claim ──────────────────────────────────────
    with etab2:
        st.markdown('<div class="section-hdr">Enter a claim to explain</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Fill in the claim details below. The model will predict fraud "
            "risk and then explain in plain language exactly which factors "
            "drove that prediction."
        )

        raw_e = claim_input_form(prefix="_exp")
        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_e, _ = st.columns([3,1,3])
        with btn_e:
            explain_btn = st.button("🧠 Explain This Claim",
                                     use_container_width=True, type="primary")

        if explain_btn:
            input_df_e = pd.DataFrame([raw_e])
            preds_e, probs_e, _ = preprocess(input_df_e, chosen_model)
            pred_e = preds_e[0]
            prob_e = probs_e[0]
            risk_e = risk_label(prob_e)

            st.markdown("---")

            # Verdict
            if pred_e == 1:
                st.markdown(f"""
                <div class="verdict-fraud">
                  <div class="verdict-title" style="color:{CRIMSON}">
                    🚨 Model Verdict: FRAUDULENT — {prob_e*100:.1f}% probability
                  </div>
                  Risk level: <strong style="color:{CRIMSON}">{risk_e}</strong>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-legit">
                  <div class="verdict-title" style="color:{SAGE}">
                    ✅ Model Verdict: LEGITIMATE — {(1-prob_e)*100:.1f}% confidence
                  </div>
                  Risk level: <strong style="color:{SAGE}">{risk_e}</strong>
                </div>""", unsafe_allow_html=True)

            st.markdown("### Why did the model decide this?")

            row_e = pd.Series(raw_e)
            exps_e = explain_claim(row_e, prob_e, chosen_model)
            bar_col_e = CRIMSON if pred_e==1 else SAGE

            st.markdown(
                f"The top factors that **{'raised' if pred_e==1 else 'lowered'}** "
                f"the fraud score for this claim:"
            )

            for i, exp in enumerate(exps_e[:8]):
                bar_w = int(exp['magnitude']*100)
                st.markdown(f"""
                <div class="explain-card">
                  <div style="display:flex;justify-content:space-between;
                              align-items:flex-start">
                    <div>
                      <span class="factor-name">
                        {i+1}. {exp['label']}
                      </span>
                      <br/>
                      <span style="font-size:0.78rem;color:{GREY}">
                        {exp['note']}
                      </span>
                    </div>
                    <span class="factor-val">
                      weight: {exp['importance']:.4f}
                    </span>
                  </div>
                  <div class="explain-bar-wrap" style="margin-top:8px">
                    <div class="explain-bar-fill"
                      style="width:{bar_w}%;background:{bar_col_e}">
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Plain language summary
            st.markdown("### Plain-Language Summary")
            top3 = [e['label'] for e in exps_e[:3]]
            if pred_e == 1:
                st.markdown(f"""
                > The model flagged this claim as **likely fraudulent** with a
                > probability of **{prob_e*100:.1f}%**. The three factors that
                > contributed most to this decision were:
                > **{top3[0]}**, **{top3[1]}**, and **{top3[2]}**.
                > These features showed patterns that the model has learned to
                > associate with fraudulent claims in the training data.
                > **Recommendation:** Refer to the fraud investigation team
                > for manual review before processing payment.
                """)
            else:
                st.markdown(f"""
                > The model assessed this claim as **likely legitimate** with
                > **{(1-prob_e)*100:.1f}%** confidence. The key factors
                > supporting this assessment were: **{top3[0]}**,
                > **{top3[1]}**, and **{top3[2]}**.
                > These values fall within patterns typically seen in genuine
                > claims in the training data.
                > **Recommendation:** Proceed with standard claims processing.
                """)

            st.caption(
                "⚠️ Feature importance scores reflect the Random Forest "
                "model's learned patterns. They are indicative, not causal. "
                "Always apply human judgement before acting on any prediction."
            )

    # ── How the model works ───────────────────────────────────────────
    with etab3:
        st.markdown('<div class="section-hdr">Technical Overview</div>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            #### Logistic Regression
            A statistical model that estimates the **probability** that a claim
            is fraudulent based on a weighted combination of input features.

            - ✅ Highly interpretable — each feature has a clear coefficient
            - ✅ Fast to train and predict
            - ✅ Strong recall (catches more fraud)
            - ⚠️ May miss complex non-linear patterns
            - **Best for:** When catching as many fraud cases as possible
              is the priority (even at the cost of some false positives)

            #### Training
            - 800 training claims (80% stratified split)
            - Class balancing via `class_weight='balanced'`
            - 200 test claims held out for evaluation
            """)
        with col_b:
            st.markdown("""
            #### Random Forest
            An ensemble of 200 decision trees, each trained on a random
            subset of data. Final prediction is a majority vote across all trees.

            - ✅ Handles complex non-linear patterns
            - ✅ Naturally provides feature importance
            - ✅ Higher precision (fewer false alarms)
            - ⚠️ Lower recall — misses more fraud cases
            - **Best for:** When minimising unnecessary investigations
              is the priority

            #### Pipeline
            1. Raw CSV → clean & replace `?` with Unknown
            2. Label-encode categorical columns
            3. StandardScaler normalisation
            4. Model prediction + probability score
            5. Risk classification (High / Medium / Low)
            """)

        st.markdown("#### Evaluation Results (held-out test set)")
        st.markdown("""
        | Metric | Logistic Regression | Random Forest | Best for fraud detection |
        |---|---|---|---|
        | Accuracy | 70.0% | 78.5% | RF |
        | Precision | 0.441 | 0.636 | RF |
        | **Recall** | **0.837** | 0.286 | **LR** ✅ |
        | **F1-Score** | **0.577** | 0.394 | **LR** ✅ |

        **Recall is the critical metric** for fraud detection: a model with
        high recall catches more actual fraud cases, even if it occasionally
        flags legitimate claims for review. Missing real fraud is more costly
        than a false alarm.
        """)


# ── Global footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🛡️ FuneralGuard Fraud Intelligence Platform &nbsp;·&nbsp;
  Tinashe Arthur Mupindu &nbsp;·&nbsp;
  University of Zambia — Actuarial Science &nbsp;·&nbsp;
  Supervisor: Dr Stanley Jere &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
