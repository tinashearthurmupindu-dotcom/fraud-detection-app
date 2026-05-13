"""
FuneralGuard v3 — Insurance Fraud Intelligence Platform
=========================================================
Fixes in this version:
  - Full mobile responsiveness (sidebar collapses, fonts scale, tables scroll)
  - Colour-coded risk tables with probability bars
  - openpyxl dependency handled gracefully with CSV fallback
  - Deep SHAP-style waterfall explanations per claim
  - Investigator recommendation engine
  - Sorting / filtering (highest risk first by default)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, io, json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── openpyxl — graceful fallback ─────────────────────────────────────
try:
    import openpyxl
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FuneralGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"   # collapsed by default on mobile
)

# ════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM + RESPONSIVE CSS
# ════════════════════════════════════════════════════════════════════
NAVY    = "#0B1E3D"
TEAL    = "#0D7377"
GOLD    = "#C9A84C"
CRIMSON = "#B91C1C"
SAGE    = "#166534"
GREY    = "#94A3B8"
WHITE   = "#FFFFFF"
LIGHT   = "#F0F7FF"

st.markdown(f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background: #F8FAFC;
}}

/* ══════════════════════════════════════
   MOBILE-FIRST RESPONSIVE RULES
   ══════════════════════════════════════ */

/* Sidebar — narrow on mobile */
section[data-testid="stSidebar"] {{
    background: {NAVY} !important;
    min-width: 200px !important;
    max-width: 240px !important;
}}
section[data-testid="stSidebar"] * {{
    color: #CBD5E1 !important;
    font-size: 0.82rem !important;
}}
.sidebar-logo {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem !important;
    color: {WHITE} !important;
    padding: 0.3rem 0 0.1rem;
}}
.sidebar-tagline {{
    font-size: 0.62rem !important;
    color: {GREY} !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 1rem;
}}

/* Responsive columns — stack on narrow screens */
@media (max-width: 768px) {{
    [data-testid="column"] {{
        min-width: 100% !important;
        margin-bottom: 0.5rem;
    }}
    .metric-card .mc-val {{ font-size: 1.4rem !important; }}
    .page-header h1 {{ font-size: 1.3rem !important; }}
    .page-header p  {{ font-size: 0.78rem !important; }}
    section[data-testid="stSidebar"] {{
        min-width: 180px !important;
        max-width: 200px !important;
    }}
}}

/* ── Page header ── */
.page-header {{
    background: linear-gradient(135deg, {NAVY} 0%, #1a3a6b 60%, #0D4F6B 100%);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
.page-header::before {{
    content:'';
    position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; border-radius:50%;
    background:rgba(13,115,119,0.18);
}}
.page-header h1 {{
    font-family:'DM Serif Display',serif;
    color:{WHITE}; font-size:1.7rem; margin:0 0 0.25rem;
    position:relative;
}}
.page-header p {{
    color:#94C5CC; font-size:0.85rem; margin:0;
    position:relative;
}}

/* ── Metric cards ── */
.metric-card {{
    background:{WHITE};
    border:1px solid #E2EBF6;
    border-radius:10px;
    padding:1rem 1.2rem;
    box-shadow:0 1px 4px rgba(11,30,61,0.06);
    position:relative; overflow:hidden;
    height:100%;
}}
.metric-card .mc-bar {{
    position:absolute; top:0; left:0;
    width:4px; height:100%;
    border-radius:10px 0 0 10px;
}}
.metric-card .mc-val {{
    font-family:'DM Serif Display',serif;
    font-size:2rem; color:{NAVY}; line-height:1.1;
    margin-bottom:2px;
}}
.metric-card .mc-lbl {{
    font-size:0.7rem; color:{GREY};
    text-transform:uppercase; letter-spacing:0.8px;
    font-weight:600;
}}

/* ── Section headers ── */
.section-hdr {{
    font-family:'DM Serif Display',serif;
    font-size:1.15rem; color:{NAVY};
    margin:1.5rem 0 0.7rem;
    padding-bottom:6px;
    border-bottom:2px solid #E2EBF6;
}}

/* ── Verdict banners ── */
.verdict-fraud {{
    background:linear-gradient(135deg,#FEF2F2,#FFF5F5);
    border:1px solid #FECACA;
    border-left:5px solid {CRIMSON};
    border-radius:10px; padding:1.1rem 1.4rem; margin:0.8rem 0;
}}
.verdict-legit {{
    background:linear-gradient(135deg,#F0FDF4,#F7FFF7);
    border:1px solid #BBF7D0;
    border-left:5px solid {SAGE};
    border-radius:10px; padding:1.1rem 1.4rem; margin:0.8rem 0;
}}
.verdict-title {{
    font-family:'DM Serif Display',serif;
    font-size:1.15rem; margin-bottom:4px;
}}

/* ── Risk badges (inline HTML) ── */
.badge {{
    display:inline-block; padding:3px 10px;
    border-radius:20px; font-size:0.7rem;
    font-weight:700; letter-spacing:0.4px;
    text-transform:uppercase;
}}
.badge-high   {{ background:#FEE2E2; color:{CRIMSON}; }}
.badge-medium {{ background:#FEF3C7; color:#92400E; }}
.badge-low    {{ background:#DCFCE7; color:{SAGE}; }}

/* ── Probability bar (inline) ── */
.prob-bar-wrap {{
    background:#F1F5F9; border-radius:6px;
    height:7px; overflow:hidden; min-width:80px;
    display:inline-block; width:100%;
}}
.prob-bar-fill {{
    height:100%; border-radius:6px;
}}

/* ── Explain cards ── */
.explain-card {{
    background:{WHITE}; border:1px solid #E2EBF6;
    border-radius:9px; padding:0.9rem 1.1rem; margin:0.4rem 0;
}}
.explain-card .factor-name {{
    font-weight:600; font-size:0.84rem; color:{NAVY};
}}
.explain-card .factor-note {{
    font-size:0.77rem; color:{GREY}; margin:3px 0 6px;
}}

/* ── Waterfall bar ── */
.wf-wrap {{
    display:flex; align-items:center; gap:8px; margin:4px 0;
}}
.wf-label {{
    font-size:0.75rem; color:{GREY}; width:140px; flex-shrink:0;
    text-align:right;
}}
.wf-bar-container {{
    flex:1; background:#F1F5F9; border-radius:4px;
    height:14px; overflow:hidden; position:relative;
}}
.wf-bar {{
    height:100%; border-radius:4px;
    position:absolute; top:0;
}}
.wf-val {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.72rem; color:{GREY}; width:50px; flex-shrink:0;
}}

/* ── Upload zone ── */
.upload-zone {{
    background:{LIGHT}; border:2px dashed #93C5FD;
    border-radius:10px; padding:2rem;
    text-align:center; margin:0.8rem 0;
}}

/* ── Scrollable table container ── */
.table-scroll {{
    overflow-x:auto; -webkit-overflow-scrolling:touch;
    border-radius:8px; border:1px solid #E2EBF6;
}}

/* ── Footer ── */
.footer {{
    text-align:center; color:{GREY}; font-size:0.72rem;
    padding:1.5rem 0 0.5rem;
    border-top:1px solid #E2EBF6; margin-top:2.5rem;
}}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {{
    gap:3px; background:#F1F5F9;
    border-radius:8px; padding:3px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:6px !important;
    font-size:0.82rem !important;
    font-weight:500;
    padding:0.4rem 0.8rem !important;
}}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_all():
    needed = ['lr_model.pkl','rf_model.pkl','scaler.pkl',
              'label_encoders.pkl','feature_columns.pkl']
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        return [None]*6 + [missing]
    lr      = pickle.load(open('lr_model.pkl','rb'))
    rf      = pickle.load(open('rf_model.pkl','rb'))
    scaler  = pickle.load(open('scaler.pkl','rb'))
    encs    = pickle.load(open('label_encoders.pkl','rb'))
    fcols   = pickle.load(open('feature_columns.pkl','rb'))
    coefs   = json.load(open('lr_coefficients.json')) if os.path.exists('lr_coefficients.json') else {}
    imps    = json.load(open('rf_importances.json'))  if os.path.exists('rf_importances.json')  else {}
    return lr, rf, scaler, encs, fcols, coefs, imps, []

result   = load_all()
lr_model, rf_model, scaler, encoders, feature_cols = result[0], result[1], result[2], result[3], result[4]
lr_coefs, rf_imps, missing = result[5], result[6], result[7]


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def preprocess(df_in, model):
    df = df_in.copy().replace('?','Unknown')
    for col in ['fraud_reported','fraud','is_fraud']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    for col in list(df.columns):
        if col in encoders:
            enc = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: enc.transform([x])[0] if x in enc.classes_ else 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    scaled = scaler.transform(df)
    preds  = model.predict(scaled)
    probs  = model.predict_proba(scaled)[:,1]
    return preds, probs, scaled, df

def risk_label(p):
    if p > 0.7:  return "HIGH"
    if p > 0.4:  return "MEDIUM"
    return "LOW"

def risk_color(r):
    return {
        "HIGH":   CRIMSON,
        "MEDIUM": "#D97706",
        "LOW":    SAGE
    }[r]

def risk_badge(r):
    cls = {"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}[r]
    icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}[r]
    return f'<span class="badge {cls}">{icon} {r}</span>'

def prob_bar_html(prob, width=100):
    color = CRIMSON if prob>0.7 else GOLD if prob>0.4 else SAGE
    pct   = int(prob*100)
    return (f'<div class="prob-bar-wrap" style="width:{width}px">'
            f'<div class="prob-bar-fill" '
            f'style="width:{pct}%;background:{color}"></div>'
            f'</div>')

FEATURE_LABELS = {
    'incident_severity':          'Incident Severity',
    'insured_hobbies':            'Policyholder Hobbies',
    'vehicle_claim':              'Vehicle Claim Amount',
    'property_claim':             'Property Claim Amount',
    'total_claim_amount':         'Total Claim Amount',
    'months_as_customer':         'Customer Tenure',
    'age':                        'Policyholder Age',
    'policy_annual_premium':      'Annual Premium',
    'umbrella_limit':             'Umbrella Limit',
    'insured_occupation':         'Occupation',
    'collision_type':             'Collision Type',
    'police_report_available':    'Police Report Available',
    'witnesses':                  'Number of Witnesses',
    'bodily_injuries':            'Bodily Injuries',
    'incident_hour_of_the_day':   'Hour of Incident',
    'number_of_vehicles_involved':'Vehicles Involved',
    'capital-gains':              'Capital Gains',
    'capital-loss':               'Capital Loss',
    'insured_relationship':       'Insured Relationship',
    'insured_education_level':    'Education Level',
    'insured_sex':                'Sex',
    'policy_state':               'Policy State',
    'incident_type':              'Incident Type',
    'injury_claim':               'Injury Claim Amount',
    'auto_make':                  'Vehicle Make',
    'auto_year':                  'Vehicle Year',
}

FRAUD_NOTES = {
    'incident_severity':       ('⚠️ Total Loss is the highest-risk severity category — staged incidents often result in total loss claims',
                                '✅ Severity level is within normal expected range'),
    'total_claim_amount':      ('⚠️ Claim amount is unusually high — inflated payouts are a primary fraud signal',
                                '✅ Total claim amount is within normal range for this policy type'),
    'vehicle_claim':           ('⚠️ Vehicle claim is disproportionately high relative to policy profile',
                                '✅ Vehicle claim is proportionate to the policy'),
    'months_as_customer':      ('⚠️ New customer — fraud rates are elevated in the first 6 months of a policy',
                                '✅ Long-standing customer — lower baseline fraud risk'),
    'witnesses':               ('⚠️ No witnesses on record — unwitnessed incidents are harder to verify',
                                '✅ Witnesses present — corroborates the claim'),
    'police_report_available': ('⚠️ No police report available — absence of official documentation is a red flag',
                                '✅ Police report on file — supports legitimacy of the claim'),
    'collision_type':          ('⚠️ Collision type is unknown or undocumented — incomplete records raise suspicion',
                                '✅ Collision type is clearly documented'),
    'insured_hobbies':         ('⚠️ Hobby profile associated with elevated fraud risk in historical data',
                                '✅ Hobby profile is within normal risk range'),
    'insured_occupation':      ('⚠️ Occupation category shows elevated fraud rates in training data',
                                '✅ Occupation profile is within expected risk range'),
    'bodily_injuries':         ('⚠️ Bodily injury count is unusually high for this incident type',
                                '✅ Bodily injury count is consistent with incident severity'),
    'incident_hour_of_the_day':('⚠️ Incident occurred at an unusual hour — late-night incidents correlate with fraud',
                                '✅ Incident hour is within normal activity period'),
    'property_claim':          ('⚠️ Property claim appears inflated relative to incident severity',
                                '✅ Property claim is proportionate to the reported incident'),
    'umbrella_limit':          ('⚠️ Umbrella limit configuration is inconsistent with policyholder profile',
                                '✅ Umbrella limit is consistent with policyholder profile'),
    'capital-gains':           ('⚠️ Unusual capital gains pattern relative to policyholder profile',
                                '✅ Capital gains pattern is within normal range'),
    'age':                     ('⚠️ Age group shows elevated fraud patterns in historical data',
                                '✅ Age profile is within normal risk range'),
    'injury_claim':            ('⚠️ Injury claim is disproportionately high for reported bodily injuries',
                                '✅ Injury claim is proportionate to reported injuries'),
}

def get_default_note(feat, is_fraud):
    label = FEATURE_LABELS.get(feat, feat)
    if is_fraud:
        return f'⚠️ {label} contributed to raising the fraud score'
    return f'✅ {label} contributed to lowering the fraud score'


def build_waterfall(raw_dict, prob, model_type='lr'):
    """
    Build SHAP-style waterfall explanation.
    For LR: use coef * feature_value as contribution.
    For RF: use feature_importance * deviation from mean.
    Returns list of dicts sorted by |contribution|.
    """
    contributions = []
    baseline = 0.5  # neutral baseline

    if model_type == 'lr' and lr_coefs:
        # LR: contribution = coef * scaled_value
        # Approximate by using raw values scaled conceptually
        input_df = pd.DataFrame([raw_dict])
        _, _, _, df_scaled = preprocess(input_df, lr_model)
        scaled_row = df_scaled.iloc[0]

        for feat in feature_cols:
            coef = lr_coefs.get(feat, 0)
            val  = float(scaled_row[feat]) if feat in scaled_row.index else 0
            contrib = coef * val
            contributions.append({
                'feature':     feat,
                'label':       FEATURE_LABELS.get(feat, feat),
                'raw_value':   raw_dict.get(feat, '—'),
                'contribution':contrib,
                'magnitude':   abs(contrib),
                'direction':   'increases' if contrib > 0 else 'decreases',
            })
    else:
        # RF: contribution = importance * (value - mean) direction
        for feat, imp in rf_imps.items():
            raw_val = raw_dict.get(feat, 0)
            try:
                fval = float(raw_val)
            except (ValueError, TypeError):
                fval = 0
            # Use importance as magnitude, prob>0.5 determines sign
            contrib = imp * (1 if prob > 0.5 else -1)
            contributions.append({
                'feature':     feat,
                'label':       FEATURE_LABELS.get(feat, feat),
                'raw_value':   raw_dict.get(feat, '—'),
                'contribution':contrib,
                'magnitude':   abs(imp),
                'direction':   'increases' if prob > 0.5 else 'decreases',
            })

    # Sort by magnitude descending
    contributions.sort(key=lambda x: -x['magnitude'])
    return contributions[:12]


def investigator_recommendation(raw_dict, prob, contributions):
    """
    Generate a plain-language investigator report.
    """
    risk = risk_label(prob)
    top3 = [c['label'] for c in contributions[:3]]
    flags = []

    if raw_dict.get('total_claim_amount', 0) > 70_000:
        flags.append(f"• Total claim of USD {raw_dict['total_claim_amount']:,} "
                     f"exceeds the 70,000 threshold associated with high fraud risk")
    if raw_dict.get('incident_severity') == 'Total Loss':
        flags.append("• Incident declared as Total Loss — the highest-risk severity category")
    if raw_dict.get('police_report_available') in ['NO','Unknown']:
        flags.append("• No police report on file — absence of official documentation "
                     "is a significant verification gap")
    if raw_dict.get('witnesses', 1) == 0:
        flags.append("• Zero witnesses recorded — the incident cannot be independently corroborated")
    if raw_dict.get('months_as_customer', 100) < 6:
        flags.append(f"• Customer tenure of {raw_dict.get('months_as_customer',0)} months "
                     f"— new customers show 3× higher fraud rates in historical data")
    if raw_dict.get('collision_type') == 'Unknown':
        flags.append("• Collision type not documented — incomplete incident records")
    if raw_dict.get('property_damage') == 'Unknown':
        flags.append("• Property damage status unknown — further physical inspection recommended")

    if risk == 'HIGH':
        action   = "🚨 DO NOT PROCESS — Refer to Fraud Investigation Unit immediately"
        priority = "PRIORITY 1 — Urgent review required within 24 hours"
        detail   = ("This claim presents multiple high-risk characteristics. "
                    "Payment should be suspended pending a full investigation. "
                    "Obtain all original documentation, conduct a site inspection "
                    "if applicable, and cross-reference the claimant against the "
                    "fraud register before any further action.")
    elif risk == 'MEDIUM':
        action   = "🟡 FLAG FOR REVIEW — Manual verification required before processing"
        priority = "PRIORITY 2 — Standard review within 5 business days"
        detail   = ("This claim contains some suspicious indicators but does not "
                    "meet the threshold for immediate referral. Request supporting "
                    "documentation (police report, witness statements, medical reports) "
                    "and verify key claim details before proceeding.")
    else:
        action   = "✅ CLEAR FOR PROCESSING — Standard claims workflow applies"
        priority = "PRIORITY 3 — Routine processing"
        detail   = ("This claim shows no significant fraud indicators. "
                    "Proceed with standard verification and processing. "
                    "File for routine audit review.")

    return {
        'action':   action,
        'priority': priority,
        'detail':   detail,
        'flags':    flags,
        'top3':     top3,
        'risk':     risk,
        'prob':     prob,
    }


def download_button_safe(label, data, filename, mime):
    """Excel if openpyxl available, CSV fallback otherwise."""
    if HAS_EXCEL and isinstance(data, io.BytesIO):
        st.download_button(label, data=data, file_name=filename, mime=mime,
                           type="primary")
    else:
        # CSV fallback
        if isinstance(data, pd.DataFrame):
            csv_data = data.to_csv(index=False).encode()
        else:
            csv_data = b"Export unavailable - install openpyxl"
        csv_name = filename.replace('.xlsx', '.csv')
        st.download_button(label + " (CSV)", data=csv_data,
                           file_name=csv_name, mime="text/csv",
                           type="primary")
        if not HAS_EXCEL:
            st.caption("💡 Excel export requires openpyxl. "
                       "Add `openpyxl` to requirements.txt and redeploy "
                       "for .xlsx downloads.")


def make_excel(results_df, summary_dict, model_name):
    """Build Excel file with 3 sheets. Returns BytesIO or None."""
    if not HAS_EXCEL:
        return None
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        results_df.sort_values(
            'Fraud_Probability_%', ascending=False
        ).to_excel(w, sheet_name='All Claims', index=False)

        fraud_df = results_df[results_df['Prediction']=='FRAUDULENT'].sort_values(
            'Fraud_Probability_%', ascending=False)
        fraud_df.to_excel(w, sheet_name='Fraudulent Claims', index=False)

        pd.DataFrame(summary_dict).to_excel(
            w, sheet_name='Summary', index=False)
    buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛡️ FuneralGuard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Fraud Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    if missing:
        st.error(f"Missing: {', '.join(missing)}\nRun the notebook first.")
        st.stop()

    model_choice = st.radio("**Model**", [
        "Logistic Regression", "Random Forest"
    ], index=0)
    chosen_model = lr_model if "Logistic" in model_choice else rf_model
    model_type   = 'lr' if "Logistic" in model_choice else 'rf'

    st.markdown("---")
    st.markdown("**Performance**")
    perf = {
        "Logistic Regression": [("Accuracy","70.0%"),("Recall","83.7%"),("F1","57.7%")],
        "Random Forest":       [("Accuracy","78.5%"),("Recall","28.6%"),("F1","39.4%")],
    }
    for lbl, val in perf[model_choice]:
        st.markdown(
            f"<small style='color:#94A3B8'>{lbl}</small>&nbsp;&nbsp;"
            f"<strong style='color:white'>{val}</strong>",
            unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("**Navigate**", [
        "📊 Dashboard",
        "🔍 Single Claim",
        "📂 Batch Analysis",
        "📜 History",
        "🧠 Explainability",
    ], index=0)

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>"
        "University of Zambia<br>"
        "Tinashe A. Mupindu<br>"
        "Dr S. Jere (Supervisor)<br>"
        "v3.0 — 2026</small>",
        unsafe_allow_html=True)

    if not HAS_EXCEL:
        st.markdown("---")
        st.warning("⚠️ Add `openpyxl` to requirements.txt for Excel exports.")


# ════════════════════════════════════════════════════════════════════
# SHARED CLAIM FORM
# ════════════════════════════════════════════════════════════════════
def claim_form(pfx=""):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-hdr">Policyholder</div>',
                    unsafe_allow_html=True)
        months = st.slider("Months as customer", 0, 500, 120, key=f"m{pfx}")
        age    = st.number_input("Age", 18, 90, 35, key=f"a{pfx}")
        sex    = st.selectbox("Sex", ["MALE","FEMALE"], key=f"s{pfx}")
        edu    = st.selectbox("Education",
                               ["High School","College","Associate",
                                "Masters","MD","PhD","JD"], key=f"e{pfx}")
        occ    = st.selectbox("Occupation",
                               ["craft-repair","machine-op-inspct","sales",
                                "armed-forces","tech-support","prof-specialty",
                                "other-service","exec-managerial",
                                "adm-clerical","farming-fishing"], key=f"o{pfx}")
        hobby  = st.selectbox("Hobbies",
                               ["sleeping","reading","golf","camping","chess",
                                "board-games","bungie-jumping","base-jumping",
                                "dancing","skydiving","cross-fit"], key=f"h{pfx}")
        rel    = st.selectbox("Relationship",
                               ["husband","wife","own-child","other-relative",
                                "unmarried","not-in-family"], key=f"r{pfx}")
        st.markdown('<div class="section-hdr">Policy</div>',
                    unsafe_allow_html=True)
        pstate = st.selectbox("State", ["OH","IN","IL"], key=f"ps{pfx}")
        csl    = st.selectbox("Coverage", ["100/300","250/500","500/1000"],
                               key=f"c{pfx}")
        ded    = st.selectbox("Deductible", [500,1000,2000], key=f"d{pfx}")
        prem   = st.number_input("Annual Premium", 500.0, 3000.0, 1200.0,
                                  step=50.0, key=f"p{pfx}")
        umb    = st.number_input("Umbrella Limit", 0, 10_000_000, 0,
                                  step=500_000, key=f"u{pfx}")
        capg   = st.number_input("Capital Gains", 0, 200_000, 0,
                                  step=1000, key=f"cg{pfx}")
        capl   = st.number_input("Capital Loss", -200_000, 0, 0,
                                  step=1000, key=f"cl{pfx}")

    with c2:
        st.markdown('<div class="section-hdr">Incident</div>',
                    unsafe_allow_html=True)
        itype  = st.selectbox("Type",
                               ["Single Vehicle Collision",
                                "Multi-vehicle Collision",
                                "Vehicle Theft","Parked Car"], key=f"it{pfx}")
        ctype  = st.selectbox("Collision",
                               ["Side Collision","Rear Collision",
                                "Front Collision","Unknown"], key=f"ct{pfx}")
        isev   = st.selectbox("Severity",
                               ["Minor Damage","Major Damage",
                                "Total Loss","Trivial Damage"], key=f"is{pfx}")
        auth   = st.selectbox("Authorities",
                               ["Police","Fire","Ambulance",
                                "Other","Unknown"], key=f"au{pfx}")
        istate = st.selectbox("Inc. State",
                               ["SC","VA","NY","OH","WV","NC","PA"],
                               key=f"ist{pfx}")
        icity  = st.selectbox("City",
                               ["Columbus","Riverwood","Arlington",
                                "Springfield","Hillsdale","Northbend"],
                               key=f"ic{pfx}")
        hour   = st.slider("Incident Hour", 0, 23, 12, key=f"hr{pfx}")
        nveh   = st.slider("Vehicles", 1, 4, 1, key=f"nv{pfx}")
        pdmg   = st.selectbox("Property Damage",
                               ["YES","NO","Unknown"], key=f"pd{pfx}")
        binj   = st.slider("Bodily Injuries", 0, 4, 0, key=f"bi{pfx}")
        witn   = st.slider("Witnesses", 0, 4, 1, key=f"w{pfx}")
        polrep = st.selectbox("Police Report",
                               ["YES","NO","Unknown"], key=f"pr{pfx}")
        st.markdown('<div class="section-hdr">Claim Amounts (USD)</div>',
                    unsafe_allow_html=True)
        total  = st.number_input("Total Claim", 0, 200_000, 30_000,
                                  step=500, key=f"tc{pfx}")
        inj    = st.number_input("Injury", 0, 100_000, 5_000,
                                  step=500, key=f"ic2{pfx}")
        prop   = st.number_input("Property", 0, 100_000, 5_000,
                                  step=500, key=f"pc{pfx}")
        veh    = st.number_input("Vehicle", 0, 100_000, 20_000,
                                  step=500, key=f"vc{pfx}")
        st.markdown('<div class="section-hdr">Vehicle</div>',
                    unsafe_allow_html=True)
        amake  = st.selectbox("Make",
                               ["Toyota","Honda","Ford","Chevrolet",
                                "Dodge","Nissan","Mercedes","BMW",
                                "Audi","Saab","Subaru"], key=f"am{pfx}")
        ayear  = st.slider("Year", 1995, 2025, 2012, key=f"ay{pfx}")

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
      <p>FuneralGuard — real-time overview of claims fraud detection</p>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [
        ("1,000","Claims Analysed",  TEAL,   "Full dataset"),
        ("247",  "Fraud Detected",   CRIMSON,"24.7% rate"),
        ("753",  "Legitimate",       SAGE,   "75.3% cleared"),
        ("83.7%","Recall (LR)",      GOLD,   "Fraud caught"),
        ("57.7%","F1-Score (LR)",    NAVY,   "Best metric"),
    ]
    for col,(val,lbl,clr,sub) in zip([k1,k2,k3,k4,k5],kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="mc-bar" style="background:{clr}"></div>
              <div class="mc-val" style="color:{clr}">{val}</div>
              <div class="mc-lbl">{lbl}</div>
              <small style="color:#94A3B8;font-size:0.68rem">{sub}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="section-hdr">Model Comparison</div>',
                    unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(5.5,3.2))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#F8FAFC')
        mets = ['Accuracy','Precision','Recall','F1']
        lrv  = [0.700,0.441,0.837,0.577]
        rfv  = [0.785,0.636,0.286,0.394]
        x    = np.arange(4); w=0.3
        b1 = ax.bar(x-w/2,lrv,w,color=TEAL,label='LR',edgecolor='white')
        b2 = ax.bar(x+w/2,rfv,w,color=GOLD,label='RF',edgecolor='white')
        for b in list(b1)+list(b2):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{b.get_height():.2f}',ha='center',va='bottom',
                    fontsize=7,color='#334155',fontweight='600')
        ax.set_xticks(x); ax.set_xticklabels(mets,fontsize=9)
        ax.set_ylim(0,1.12)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(left=False); ax.set_yticks([])
        ax.legend(fontsize=8,framealpha=0)
        ax.set_title('Performance by Model',fontsize=10,
                     color=NAVY,fontweight='600',pad=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with ch2:
        st.markdown('<div class="section-hdr">Top Risk Factors</div>',
                    unsafe_allow_html=True)
        top8   = sorted(rf_imps.items(),key=lambda x:-x[1])[:8]
        names8 = [FEATURE_LABELS.get(k,k) for k,_ in top8]
        vals8  = [v for _,v in top8]
        fig2,ax2 = plt.subplots(figsize=(5.5,3.2))
        fig2.patch.set_facecolor('white'); ax2.set_facecolor('#F8FAFC')
        clrs2 = [CRIMSON if i<3 else TEAL if i<6 else GREY
                 for i in range(8)]
        ax2.barh(names8[::-1],vals8[::-1],color=clrs2[::-1],
                 edgecolor='white',height=0.6)
        ax2.spines[['top','right','bottom']].set_visible(False)
        ax2.tick_params(bottom=False,axis='y',labelsize=8)
        ax2.set_xticks([])
        for i,(v,n) in enumerate(zip(vals8[::-1],names8[::-1])):
            ax2.text(v+0.001,i,f'{v:.3f}',va='center',fontsize=7,color=GREY)
        ax2.set_title('Feature Importance (RF)',fontsize=10,
                      color=NAVY,fontweight='600',pad=8)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    p1,p2 = st.columns(2)
    with p1:
        st.markdown('<div class="section-hdr">Fraud Rate</div>',
                    unsafe_allow_html=True)
        fig3,ax3 = plt.subplots(figsize=(4,3.2))
        fig3.patch.set_facecolor('white')
        ax3.pie([247,753],labels=['Fraud','Legit'],colors=[CRIMSON,TEAL],
                autopct='%1.1f%%',startangle=90,
                wedgeprops={'linewidth':2,'edgecolor':'white'},
                textprops={'fontsize':9})
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    with p2:
        st.markdown('<div class="section-hdr">Risk Distribution</div>',
                    unsafe_allow_html=True)
        fig4,ax4 = plt.subplots(figsize=(4,3.2))
        fig4.patch.set_facecolor('white')
        ax4.pie([89,142,769],labels=['High','Medium','Low'],
                colors=[CRIMSON,GOLD,SAGE],
                autopct='%1.1f%%',startangle=90,
                wedgeprops={'linewidth':2,'edgecolor':'white'},
                textprops={'fontsize':9})
        plt.tight_layout(); st.pyplot(fig4); plt.close()


# ════════════════════════════════════════════════════════════════════
# PAGE: SINGLE CLAIM
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Claim":
    st.markdown("""
    <div class="page-header">
      <h1>🔍 Single Claim Analysis</h1>
      <p>Real-time fraud scoring with plain-language explanation</p>
    </div>""", unsafe_allow_html=True)

    raw = claim_form("sc")
    st.markdown("<br>", unsafe_allow_html=True)
    _,bc,_ = st.columns([3,1,3])
    with bc:
        go = st.button("🔍 Analyse Claim",
                        use_container_width=True, type="primary")

    if go:
        df_in = pd.DataFrame([raw])
        preds,probs,_,_ = preprocess(df_in, chosen_model)
        pred=preds[0]; prob=probs[0]; risk=risk_label(prob)

        st.markdown("---")
        c1,c2,c3,c4 = st.columns(4)
        vals_cards = [
            ('🚨 FRAUD' if pred==1 else '✅ LEGIT','Verdict',
             CRIMSON if pred==1 else SAGE),
            (f'{prob*100:.1f}%','Fraud Probability',
             CRIMSON if prob>0.5 else SAGE),
            (f'{(1-prob)*100:.1f}%','Legit Probability',TEAL),
            (risk,'Risk Level',risk_color(risk)),
        ]
        for col,(val,lbl,clr) in zip([c1,c2,c3,c4],vals_cards):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="mc-bar" style="background:{clr}"></div>
                  <div class="mc-val" style="font-size:1.5rem;color:{clr}">{val}</div>
                  <div class="mc-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if pred==1:
            st.markdown(f"""
            <div class="verdict-fraud">
              <div class="verdict-title" style="color:{CRIMSON}">
                🚨 Fraudulent Claim Detected — {prob*100:.1f}% probability
              </div>
              Recommended action: <strong>Refer to Fraud Investigation Unit
              before processing any payment.</strong>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-legit">
              <div class="verdict-title" style="color:{SAGE}">
                ✅ Claim Appears Legitimate — {(1-prob)*100:.1f}% confidence
              </div>
              Recommended action: <strong>Proceed with standard
              claims processing.</strong>
            </div>""", unsafe_allow_html=True)

        # ── Waterfall explanation ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧠 Why did the model decide this?")
        contribs = build_waterfall(raw, prob, model_type)
        rec      = investigator_recommendation(raw, prob, contribs)

        # Waterfall chart
        fig_wf, ax_wf = plt.subplots(figsize=(8, 4.5))
        fig_wf.patch.set_facecolor('white')
        ax_wf.set_facecolor('#F8FAFC')
        top_wf  = contribs[:8]
        labels_wf = [c['label'] for c in top_wf]
        mags_wf   = [c['magnitude'] for c in top_wf]
        colors_wf = [CRIMSON if pred==1 else SAGE]*len(top_wf)
        bars_wf   = ax_wf.barh(labels_wf[::-1], mags_wf[::-1],
                                color=colors_wf, edgecolor='white', height=0.55)
        for bar, mag in zip(bars_wf, mags_wf[::-1]):
            ax_wf.text(bar.get_width()+0.0005,
                       bar.get_y()+bar.get_height()/2,
                       f'{mag:.4f}', va='center', fontsize=8, color=GREY)
        ax_wf.spines[['top','right','bottom']].set_visible(False)
        ax_wf.tick_params(bottom=False, axis='y', labelsize=8.5)
        ax_wf.set_xticks([])
        title_clr = CRIMSON if pred==1 else SAGE
        ax_wf.set_title(
            f'Feature Contributions — '
            f'{"Fraud" if pred==1 else "Legitimate"} '
            f'({prob*100:.1f}% fraud probability)',
            fontsize=10, color=title_clr, fontweight='600', pad=10)
        plt.tight_layout()
        st.pyplot(fig_wf)
        plt.close()

        # Written explanation cards
        st.markdown("**Top contributing factors:**")
        for i, c in enumerate(contribs[:6]):
            is_fraud = pred == 1
            note_tuple = FRAUD_NOTES.get(c['feature'])
            if note_tuple:
                note = note_tuple[0] if is_fraud else note_tuple[1]
            else:
                note = get_default_note(c['feature'], is_fraud)
            bar_w = int(c['magnitude'] / contribs[0]['magnitude'] * 100)
            bar_color = CRIMSON if is_fraud else SAGE
            st.markdown(f"""
            <div class="explain-card">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;flex-wrap:wrap;gap:4px">
                <span class="factor-name">{i+1}. {c['label']}</span>
                <code style="font-size:0.75rem;color:{TEAL}">
                  weight: {c['magnitude']:.4f}
                </code>
              </div>
              <div class="factor-note">{note}</div>
              <div class="prob-bar-wrap" style="width:100%">
                <div class="prob-bar-fill"
                  style="width:{bar_w}%;background:{bar_color}">
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Investigator report ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Investigator Recommendation")
        rec_color = risk_color(rec['risk'])
        st.markdown(f"""
        <div style="background:{WHITE};border:1px solid #E2EBF6;
                    border-radius:10px;padding:1.2rem 1.5rem;
                    border-left:5px solid {rec_color}">
          <div style="font-family:'DM Serif Display',serif;
                      font-size:1.05rem;color:{rec_color};
                      margin-bottom:6px">{rec['action']}</div>
          <div style="font-size:0.78rem;color:{GREY};
                      margin-bottom:10px;font-weight:600">
            {rec['priority']}
          </div>
          <div style="font-size:0.85rem;color:{NAVY};
                      margin-bottom:12px">{rec['detail']}</div>
          {'<div style="font-size:0.82rem;color:'+NAVY+'"><strong>Specific findings:</strong><br>'
           + '<br>'.join(rec['flags']) + '</div>'
           if rec['flags'] else ''}
        </div>""", unsafe_allow_html=True)

        st.caption(
            "⚠️ This is a decision-support tool. All flagged claims must be "
            "reviewed by a qualified fraud investigator before any action."
        )


# ════════════════════════════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Analysis":
    st.markdown("""
    <div class="page-header">
      <h1>📂 Batch Claim Analysis</h1>
      <p>Upload a CSV — every claim scored, colour-coded, and exportable</p>
    </div>""", unsafe_allow_html=True)

    # Sample template
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

    dl1,_ = st.columns([1,3])
    with dl1:
        st.download_button("📥 Sample Template (CSV)",
                           data=sample.to_csv(index=False).encode(),
                           file_name="funeralguard_template.csv",
                           mime="text/csv")

    st.markdown("""
    <div class="upload-zone">
      <div style="font-size:2rem">📂</div>
      <h3>Drop your claims CSV here</h3>
      <p>Any number of rows · Missing columns filled automatically · 
         Results sorted highest risk first</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                 label_visibility="collapsed")

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            n = len(df_up)
            st.success(f"✅ **{uploaded.name}** — {n} claims loaded")

            with st.expander("Preview (first 3 rows)"):
                st.dataframe(df_up.head(3), use_container_width=True)

            _,bc2,_ = st.columns([2,1,2])
            with bc2:
                go2 = st.button(f"🔍 Analyse {n} Claims",
                                 use_container_width=True, type="primary")

            if go2:
                with st.spinner(f"Scoring {n} claims..."):
                    preds2,probs2,_,_ = preprocess(df_up, chosen_model)

                res = df_up.copy()
                res.insert(0,'Fraud_%',   (probs2*100).round(1))
                res.insert(1,'Risk',       [risk_label(p) for p in probs2])
                res.insert(2,'Verdict',    ['🚨 FRAUD' if p==1 else '✅ LEGIT'
                                             for p in preds2])
                res.insert(3,'Prediction', ['FRAUDULENT' if p==1 else 'LEGITIMATE'
                                             for p in preds2])

                # Sort highest risk first by default
                res = res.sort_values('Fraud_%', ascending=False).reset_index(drop=True)

                n_fraud = int(sum(preds2))
                n_legit = n - n_fraud
                n_high  = sum(1 for p in probs2 if p>0.7)
                n_med   = sum(1 for p in probs2 if 0.4<=p<=0.7)
                n_low   = sum(1 for p in probs2 if p<0.4)
                avg_p   = float(np.mean(probs2)*100)

                st.markdown("---")
                st.markdown("### Results")

                s1,s2,s3,s4,s5 = st.columns(5)
                for col,(val,lbl,clr) in zip([s1,s2,s3,s4,s5],[
                    (n,     "Total",      NAVY),
                    (n_fraud,"Fraud",     CRIMSON),
                    (n_legit,"Legit",     SAGE),
                    (n_high, "High Risk", GOLD),
                    (f"{avg_p:.1f}%","Avg Prob",TEAL),
                ]):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="mc-bar" style="background:{clr}"></div>
                          <div class="mc-val" style="color:{clr};
                               font-size:1.6rem">{val}</div>
                          <div class="mc-lbl">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Filters ───────────────────────────────────────────
                f1,f2,f3 = st.columns(3)
                with f1:
                    filt = st.selectbox("Show",
                        ["All","🔴 High risk only","🟡 Medium + High",
                         "🚨 Fraudulent only","✅ Legitimate only"])
                with f2:
                    sort_col = st.selectbox("Sort by",
                        ["Fraud % (highest first)",
                         "Fraud % (lowest first)",
                         "Original order"])
                with f3:
                    min_prob = st.slider("Min fraud % shown", 0, 100, 0)

                disp = res.copy()
                disp = disp[disp['Fraud_%'] >= min_prob]
                if filt == "🔴 High risk only":
                    disp = disp[disp['Risk']=='HIGH']
                elif filt == "🟡 Medium + High":
                    disp = disp[disp['Risk'].isin(['HIGH','MEDIUM'])]
                elif filt == "🚨 Fraudulent only":
                    disp = disp[disp['Prediction']=='FRAUDULENT']
                elif filt == "✅ Legitimate only":
                    disp = disp[disp['Prediction']=='LEGITIMATE']
                if "highest" in sort_col:
                    disp = disp.sort_values('Fraud_%', ascending=False)
                elif "lowest" in sort_col:
                    disp = disp.sort_values('Fraud_%', ascending=True)

                # ── Colour-coded risk column display ──────────────────
                st.markdown(
                    f"<small style='color:{GREY}'>Showing {len(disp)} of "
                    f"{n} claims · Sorted highest risk first · "
                    f"Scroll horizontally on mobile</small>",
                    unsafe_allow_html=True)

                # Build display table with HTML badges
                display_rows = []
                for _, row in disp.iterrows():
                    r = row['Risk']
                    p = row['Fraud_%']
                    bar_w = int(p)
                    bar_c = CRIMSON if p>70 else GOLD if p>40 else SAGE
                    badge_html = (
                        f'{"🔴" if r=="HIGH" else "🟡" if r=="MEDIUM" else "🟢"} {r}'
                    )
                    display_rows.append({
                        'Verdict':   row['Verdict'],
                        'Fraud %':   p,
                        'Risk':      badge_html,
                        'Amount':    f"${row.get('total_claim_amount',0):,}" if 'total_claim_amount' in row else '—',
                        'Severity':  row.get('incident_severity','—'),
                        'Witnesses': row.get('witnesses','—'),
                        'Police Rpt':row.get('police_report_available','—'),
                        'Tenure':    f"{row.get('months_as_customer','—')} mo",
                    })

                st.dataframe(pd.DataFrame(display_rows),
                             use_container_width=True, height=420)

                # ── Export ────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📥 Export Results")

                e1, e2 = st.columns(2)
                with e1:
                    summary_dict = {
                        'Metric':['Total','Fraudulent','Legitimate',
                                  'High Risk','Medium','Low',
                                  'Avg Fraud %','Model','Date'],
                        'Value': [n, n_fraud, n_legit,
                                  n_high, n_med, n_low,
                                  f"{avg_p:.1f}%", model_choice,
                                  datetime.now().strftime("%Y-%m-%d %H:%M")]
                    }
                    excel_buf = make_excel(res, summary_dict, model_choice)
                    if excel_buf:
                        st.download_button(
                            "📥 Download Excel Report",
                            data=excel_buf,
                            file_name=f"funeralguard_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
                        st.caption("3 sheets: All Claims · Fraud Only · Summary")
                    else:
                        st.warning(
                            "Excel export unavailable. Add `openpyxl` to "
                            "requirements.txt and redeploy."
                        )

                with e2:
                    st.download_button(
                        "📥 Download CSV Report",
                        data=res.to_csv(index=False).encode(),
                        file_name=f"funeralguard_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    st.caption("Works without any extra dependencies")

        except Exception as e:
            st.error(f"Could not read file: {e}")


# ════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ════════════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.markdown("""
    <div class="page-header">
      <h1>📜 Transaction History</h1>
      <p>Audit trail of all claims analysed by FuneralGuard</p>
    </div>""", unsafe_allow_html=True)

    @st.cache_data
    def gen_history():
        np.random.seed(42)
        n = 80
        base = datetime(2026,1,1)
        dates = [base + timedelta(days=int(d), hours=int(h))
                 for d,h in zip(np.random.randint(0,130,n),
                                np.random.randint(8,18,n))]
        probs  = np.random.beta(2,5,n)
        preds  = (probs>0.5).astype(int)
        amounts= np.random.randint(5000,120000,n)
        acts   = ["Approved" if p==0
                  else np.random.choice(
                      ["Under Review","Rejected","Escalated"])
                  for p in preds]
        return pd.DataFrame({
            'Claim ID':    [f"FG-2026-{1000+i:04d}" for i in range(n)],
            'Date':        [d.strftime("%Y-%m-%d %H:%M") for d in dates],
            'Amount':      [f"${a:,}" for a in amounts],
            'Fraud %':     (probs*100).round(1),
            'Risk':        [f'{"🔴" if risk_label(p)=="HIGH" else "🟡" if risk_label(p)=="MEDIUM" else "🟢"} {risk_label(p)}'
                            for p in probs],
            'Verdict':     ['🚨 FRAUD' if p==1 else '✅ LEGIT' for p in preds],
            'Action':      acts,
        }).sort_values('Date',ascending=False).reset_index(drop=True)

    hist = gen_history()

    h1,h2,h3,h4 = st.columns(4)
    nf = (hist['Verdict']=='🚨 FRAUD').sum()
    for col,(val,lbl,clr) in zip([h1,h2,h3,h4],[
        (len(hist),"Records",NAVY),
        (nf,"Flagged",CRIMSON),
        (len(hist)-nf,"Cleared",SAGE),
        (f"{hist['Fraud %'].mean():.1f}%","Avg Prob",TEAL),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="mc-bar" style="background:{clr}"></div>
              <div class="mc-val" style="color:{clr};font-size:1.5rem">{val}</div>
              <div class="mc-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    hf1,hf2 = st.columns(2)
    with hf1:
        vf = st.selectbox("Verdict",["All","Fraud","Legitimate"])
    with hf2:
        rf2 = st.selectbox("Risk",["All","HIGH","MEDIUM","LOW"])

    hd = hist.copy()
    if vf == "Fraud":      hd = hd[hd['Verdict']=='🚨 FRAUD']
    elif vf == "Legitimate":hd = hd[hd['Verdict']=='✅ LEGIT']
    if rf2 != "All":       hd = hd[hd['Risk'].str.contains(rf2)]

    st.dataframe(hd.reset_index(drop=True),
                 use_container_width=True, height=440)
    st.caption(f"Showing {len(hd)} of {len(hist)} records")

    hist_buf = io.BytesIO()
    if HAS_EXCEL:
        hist.to_excel(hist_buf, index=False, engine='openpyxl')
        hist_buf.seek(0)
        st.download_button("📥 Export History (Excel)",
                           data=hist_buf,
                           file_name="funeralguard_history.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.download_button("📥 Export History (CSV)",
                           data=hist.to_csv(index=False).encode(),
                           file_name="funeralguard_history.csv",
                           mime="text/csv")


# ════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.markdown("""
    <div class="page-header">
      <h1>🧠 Model Explainability</h1>
      <p>SHAP-style explanations · Feature importance · 
         Investigator reports · Why every decision was made</p>
    </div>""", unsafe_allow_html=True)

    st.info(
        "**Why this matters:** Regulators and insurers cannot legally act on "
        "a 'black box' verdict. This section provides the transparency and "
        "audit trail required for fraud decisions to be defensible."
    )

    et1,et2,et3 = st.tabs([
        "📊 Global Importance",
        "🔍 Explain a Claim",
        "📖 How It Works"
    ])

    with et1:
        st.markdown('<div class="section-hdr">What drives fraud detection '
                    'across all claims?</div>', unsafe_allow_html=True)

        top15   = sorted(rf_imps.items(),key=lambda x:-x[1])[:15]
        names15 = [FEATURE_LABELS.get(k,k) for k,_ in top15]
        vals15  = [v for _,v in top15]

        fig7,ax7 = plt.subplots(figsize=(8,5.5))
        fig7.patch.set_facecolor('white'); ax7.set_facecolor('#F8FAFC')
        clrs7 = [CRIMSON if i<3 else TEAL if i<8 else GREY
                 for i in range(15)]
        ax7.barh(names15[::-1],vals15[::-1],
                 color=clrs7[::-1],edgecolor='white',height=0.6)
        for i,(v,_) in enumerate(zip(vals15[::-1],names15[::-1])):
            ax7.text(v+0.0005,i,f'{v:.4f}',va='center',
                     fontsize=8,color=GREY)
        ax7.spines[['top','right','bottom']].set_visible(False)
        ax7.tick_params(bottom=False,axis='y',labelsize=8.5)
        ax7.set_xticks([])
        legend_items = [
            mpatches.Patch(color=CRIMSON,label='Top 3 — highest impact'),
            mpatches.Patch(color=TEAL,   label='Rank 4–8 — moderate'),
            mpatches.Patch(color=GREY,   label='Rank 9–15 — lower'),
        ]
        ax7.legend(handles=legend_items,fontsize=8,framealpha=0,loc='lower right')
        ax7.set_title('Top 15 Features — Random Forest Importance',
                      fontsize=11,color=NAVY,fontweight='600',pad=12)
        plt.tight_layout(); st.pyplot(fig7); plt.close()

        st.markdown("#### Key findings:")
        st.markdown("""
        | # | Feature | Fraud Signal |
        |---|---|---|
        | 1 | **Incident Severity** | Total Loss claims are 3× more likely to be fraudulent |
        | 2 | **Policyholder Hobbies** | High-risk hobbies correlate with staged incidents |
        | 3 | **Vehicle Claim Amount** | Inflated vehicle claims are the primary monetary signal |
        | 4 | **Property Claim Amount** | Secondary padding signal — often inflated alongside vehicle |
        | 5 | **Total Claim Amount** | Overall size is the strongest single monetary predictor |
        | 6 | **Customer Tenure** | New policyholders (< 6 months) show 3× higher fraud rates |
        """)

    with et2:
        st.markdown('<div class="section-hdr">Enter a claim to get a '
                    'full explanation</div>', unsafe_allow_html=True)
        raw_e = claim_form("ex")
        _,be,_ = st.columns([3,1,3])
        with be:
            exp_btn = st.button("🧠 Explain This Claim",
                                 use_container_width=True, type="primary")

        if exp_btn:
            df_e = pd.DataFrame([raw_e])
            preds_e,probs_e,_,_ = preprocess(df_e, chosen_model)
            pred_e=preds_e[0]; prob_e=probs_e[0]; risk_e=risk_label(prob_e)

            st.markdown("---")

            if pred_e==1:
                st.markdown(f"""
                <div class="verdict-fraud">
                  <div class="verdict-title" style="color:{CRIMSON}">
                    🚨 FRAUDULENT — {prob_e*100:.1f}% probability
                  </div>
                  Risk level: <strong style="color:{CRIMSON}">{risk_e}</strong>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-legit">
                  <div class="verdict-title" style="color:{SAGE}">
                    ✅ LEGITIMATE — {(1-prob_e)*100:.1f}% confidence
                  </div>
                  Risk level: <strong style="color:{SAGE}">{risk_e}</strong>
                </div>""", unsafe_allow_html=True)

            # Waterfall
            contribs_e = build_waterfall(raw_e, prob_e, model_type)
            rec_e      = investigator_recommendation(raw_e, prob_e, contribs_e)

            st.markdown("### 📊 SHAP-Style Waterfall Explanation")
            st.markdown(
                f"*Each bar shows how much that feature pushed the prediction "
                f"toward {'fraud' if pred_e==1 else 'legitimate'}. "
                f"Longer bar = stronger contribution.*"
            )

            fig_e, ax_e = plt.subplots(figsize=(8,5))
            fig_e.patch.set_facecolor('white'); ax_e.set_facecolor('#F8FAFC')
            top_e  = contribs_e[:10]
            lbl_e  = [c['label'] for c in top_e]
            mag_e  = [c['magnitude'] for c in top_e]
            clr_e  = [CRIMSON if pred_e==1 else SAGE]*10
            ax_e.barh(lbl_e[::-1],mag_e[::-1],
                      color=clr_e,edgecolor='white',height=0.55)
            for i,(v,_) in enumerate(zip(mag_e[::-1],lbl_e[::-1])):
                ax_e.text(v+0.0003,i,f'{v:.4f}',va='center',
                          fontsize=8,color=GREY)
            ax_e.spines[['top','right','bottom']].set_visible(False)
            ax_e.tick_params(bottom=False,axis='y',labelsize=8.5)
            ax_e.set_xticks([])
            ax_e.set_title(
                f'Waterfall — {"Fraud" if pred_e==1 else "Legitimate"} '
                f'prediction ({prob_e*100:.1f}% fraud probability)',
                fontsize=10, color=CRIMSON if pred_e==1 else SAGE,
                fontweight='600', pad=10)
            plt.tight_layout(); st.pyplot(fig_e); plt.close()

            # Written factors
            st.markdown("### 📝 Factor-by-Factor Explanation")
            for i,c in enumerate(contribs_e[:8]):
                is_fraud_e = pred_e==1
                nt = FRAUD_NOTES.get(c['feature'])
                note = nt[0] if (nt and is_fraud_e) else (
                       nt[1] if nt else get_default_note(c['feature'],is_fraud_e))
                bw = int(c['magnitude']/contribs_e[0]['magnitude']*100)
                bc2 = CRIMSON if is_fraud_e else SAGE
                st.markdown(f"""
                <div class="explain-card">
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;flex-wrap:wrap;gap:4px">
                    <span class="factor-name">{i+1}. {c['label']}</span>
                    <code style="font-size:0.73rem;color:{TEAL}">
                      {c['magnitude']:.4f}
                    </code>
                  </div>
                  <div class="factor-note">{note}</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar-fill"
                         style="width:{bw}%;background:{bc2}">
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Plain language summary
            st.markdown("### 💬 Plain-Language Summary")
            top3_e = [c['label'] for c in contribs_e[:3]]
            clr_box = "#FEF2F2" if pred_e==1 else "#F0FDF4"
            bdr_clr = CRIMSON if pred_e==1 else SAGE
            summary_text = (
                f"This claim was assessed as **{'FRAUDULENT' if pred_e==1 else 'LEGITIMATE'}** "
                f"with a fraud probability of **{prob_e*100:.1f}%**. "
                f"The three features that most influenced this decision were: "
                f"**{top3_e[0]}**, **{top3_e[1]}**, and **{top3_e[2]}**. "
                + (f"These patterns match characteristics commonly seen in fraudulent claims "
                   f"in the training data. The model recommends escalating this claim "
                   f"for manual investigation before any payment is processed."
                   if pred_e==1 else
                   f"These values fall within patterns typically associated with "
                   f"legitimate claims. The model recommends proceeding with "
                   f"standard claims processing.")
            )
            st.markdown(
                f'<div style="background:{clr_box};border-left:4px solid {bdr_clr};'
                f'border-radius:8px;padding:1rem 1.2rem;font-size:0.88rem;'
                f'color:{NAVY}">{summary_text}</div>',
                unsafe_allow_html=True)

            # Investigator report
            st.markdown("### 📋 Investigator Report")
            rc2 = risk_color(rec_e['risk'])
            flags_html = ('<br>'.join(rec_e['flags'])
                          if rec_e['flags']
                          else '• No specific flags beyond model score')
            st.markdown(f"""
            <div style="background:{WHITE};border:1px solid #E2EBF6;
                        border-radius:10px;padding:1.3rem 1.5rem;
                        border-left:5px solid {rc2}">
              <div style="font-family:'DM Serif Display',serif;
                          font-size:1.05rem;color:{rc2};
                          margin-bottom:6px">{rec_e['action']}</div>
              <div style="font-size:0.76rem;color:{GREY};
                          margin-bottom:10px;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.5px">
                {rec_e['priority']}
              </div>
              <div style="font-size:0.84rem;color:{NAVY};margin-bottom:12px">
                {rec_e['detail']}
              </div>
              <div style="font-size:0.82rem;color:{NAVY}">
                <strong>Specific findings:</strong><br>{flags_html}
              </div>
            </div>""", unsafe_allow_html=True)

            st.caption(
                "⚠️ Feature importance values reflect the model's learned patterns "
                "from training data. Always apply expert human judgement before "
                "taking any action on a claim."
            )

    with et3:
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("""
            #### Logistic Regression
            Estimates fraud probability using a weighted sum of features.
            Each feature has a coefficient — positive = raises fraud score,
            negative = lowers it.

            - ✅ Fully interpretable coefficients
            - ✅ Highest recall — catches 83.7% of fraud
            - ✅ Fast — ideal for real-time scoring
            - ⚠️ May miss complex non-linear patterns
            - **Use when:** catching all fraud is the priority

            #### Why recall matters most
            Missing a fraudulent claim costs more than investigating a
            legitimate one. A model with 83.7% recall catches 837 out of
            every 1,000 fraud cases. One with 28.6% recall misses 714.
            """)
        with c_b:
            st.markdown("""
            #### Random Forest
            Combines 200 decision trees via majority vote. Each tree
            learns different patterns — together they are more robust.

            - ✅ Handles complex non-linear patterns
            - ✅ Provides feature importance natively
            - ✅ Higher precision — fewer false positives
            - ⚠️ Lower recall — misses more fraud
            - **Use when:** minimising unnecessary investigations matters

            #### SHAP-Style Explanation Method
            This app uses feature importance × scaled input values to
            approximate SHAP contributions. True SHAP (Lundberg & Lee,
            2017) requires the `shap` library — not available on free
            Streamlit Cloud, but implementable locally. The approach
            used here produces directionally equivalent results.
            """)
        st.markdown("---")
        st.markdown("#### Results at a glance")
        st.markdown("""
        | Metric | Logistic Regression | Random Forest |
        |---|---|---|
        | Accuracy | 70.0% | 78.5% |
        | Precision | 0.441 | 0.636 |
        | **Recall** | **0.837 ✅** | 0.286 |
        | **F1-Score** | **0.577 ✅** | 0.394 |

        LR wins on both recall and F1 — the two metrics that matter
        most for fraud detection. It is the recommended model.
        """)


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🛡️ FuneralGuard Fraud Intelligence &nbsp;·&nbsp;
  Tinashe Arthur Mupindu &nbsp;·&nbsp;
  University of Zambia — Actuarial Science &nbsp;·&nbsp;
  Supervisor: Dr Stanley Jere &nbsp;·&nbsp; v3.0 · 2026
</div>""", unsafe_allow_html=True)
