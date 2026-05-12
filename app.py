"""
Funeral Insurance Fraud Detection — Streamlit Prototype
========================================================
Author:     Tinashe Arthur Mupindu
University: University of Zambia
Supervisor: Dr Stanley Jere

HOW TO RUN:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Funeral Insurance Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# ── CSS Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #BDD7EE; margin: 0.3rem 0 0; font-size: 0.95rem; }

    .tab-header {
        background: #EBF3FB;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #2E75B6;
        color: #1F3864;
        font-weight: 600;
    }
    .result-fraud {
        background: #FFF0F0;
        border-left: 6px solid #C00000;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-legit {
        background: #F0FFF4;
        border-left: 6px solid #375623;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .metric-box {
        background: #EBF3FB;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-box .metric-val { font-size: 1.6rem; font-weight: bold; color: #1F3864; }
    .metric-box .metric-lbl { font-size: 0.75rem; color: #555; }
    .section-title {
        color: #1F3864;
        font-weight: 700;
        font-size: 1.05rem;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 4px;
        margin: 1.2rem 0 0.8rem;
    }
    .upload-box {
        background: #F8FBFF;
        border: 2px dashed #2E75B6;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stat-card {
        background: white;
        border: 1px solid #D6E4F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stat-card .stat-val { font-size: 2rem; font-weight: bold; color: #1F3864; }
    .stat-card .stat-lbl { font-size: 0.8rem; color: #666; margin-top: 4px; }
    .footer {
        color: #888;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model files ──────────────────────────────────────────────────
@st.cache_resource
def load_model_files():
    required = ['best_fraud_model.pkl', 'scaler.pkl',
                 'label_encoders.pkl', 'feature_columns.pkl']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    with open('best_fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, encoders, feature_cols, []


model, scaler, encoders, feature_cols, missing_files = load_model_files()


# ── Helper: preprocess and predict ───────────────────────────────────
def preprocess_and_predict(input_df):
    df = input_df.copy()
    df = df.replace('?', 'Unknown')

    # Drop target column if accidentally included
    for col in ['fraud_reported', 'fraud', 'is_fraud']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categoricals
    for col in df.columns:
        if col in encoders:
            enc = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: enc.transform([x])[0] if x in enc.classes_ else 0
            )

    # Add missing columns as 0, align to training order
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    scaled = scaler.transform(df)
    predictions = model.predict(scaled)
    probabilities = model.predict_proba(scaled)[:, 1]
    return predictions, probabilities


# ── Risk level helper ─────────────────────────────────────────────────
def get_risk(prob):
    if prob > 0.7:
        return "HIGH"
    elif prob > 0.4:
        return "MEDIUM"
    return "LOW"


# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔍 Funeral Insurance Fraud Detection Prototype</h1>
  <p>Application of Machine Learning and Predictive Analytics &nbsp;|&nbsp;
     University of Zambia &nbsp;|&nbsp; Tinashe Arthur Mupindu &nbsp;|&nbsp;
     Supervisor: Dr Stanley Jere</p>
</div>
""", unsafe_allow_html=True)

if missing_files:
    st.error(
        f"**Missing model files:** {', '.join(missing_files)}\n\n"
        "Please run `fraud_detection_pipeline.ipynb` first to generate "
        "these files, then restart this app."
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Information")
    model_name = type(model).__name__
    display_name = "Random Forest" if "Forest" in model_name else "Logistic Regression"
    st.info(f"**Active model:** {display_name}")
    st.markdown("---")
    st.markdown("**Pipeline steps:**")
    st.markdown("""
    1. ✅ Data loaded & cleaned
    2. ✅ Features encoded & scaled
    3. ✅ Model trained (80% data)
    4. ✅ Model evaluated (20% data)
    5. ✅ Prototype deployed
    """)
    st.markdown("---")
    st.markdown("**Test results (held-out test set):**")
    st.markdown("""
    | Metric | LR | RF |
    |---|---|---|
    | Accuracy | 70.0% | 78.5% |
    | Precision | 0.441 | 0.636 |
    | Recall | **0.837** | 0.286 |
    | F1-Score | **0.577** | 0.394 |
    """)
    st.caption("LR = Logistic Regression (active) | RF = Random Forest")
    st.markdown("---")
    st.caption(
        "⚠️ Proof-of-concept prototype. Not for live claim decisions "
        "without further validation."
    )

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "📋 Single Claim Analysis",
    "📂 Batch Upload (CSV)"
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE CLAIM
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div class="tab-header">Enter the details of one claim '
        'and get an instant fraud risk prediction.</div>',
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">👤 Policyholder Details</div>',
                    unsafe_allow_html=True)
        months_as_customer = st.slider("Months as customer", 0, 500, 120)
        age = st.number_input("Policyholder age", 18, 90, 35)
        insured_sex = st.selectbox("Sex", ["MALE", "FEMALE"])
        insured_education_level = st.selectbox(
            "Education level",
            ["High School", "College", "Associate", "Masters", "MD", "PhD", "JD"]
        )
        insured_occupation = st.selectbox("Occupation", [
            "craft-repair", "machine-op-inspct", "sales", "armed-forces",
            "tech-support", "prof-specialty", "other-service", "priv-house-serv",
            "transport-moving", "handlers-cleaners", "adm-clerical",
            "farming-fishing", "protective-serv", "exec-managerial"
        ])
        insured_hobbies = st.selectbox("Hobbies", [
            "sleeping", "reading", "board-games", "bungie-jumping",
            "base-jumping", "golf", "camping", "dancing", "skydiving",
            "chess", "cross-fit", "paintball", "polo", "kayaking", "video-games"
        ])
        insured_relationship = st.selectbox("Relationship to insured", [
            "husband", "wife", "own-child", "other-relative",
            "unmarried", "not-in-family"
        ])

        st.markdown('<div class="section-title">📋 Policy Details</div>',
                    unsafe_allow_html=True)
        policy_state = st.selectbox("Policy state", ["OH", "IN", "IL"])
        policy_csl = st.selectbox("Coverage limit (CSL)",
                                   ["100/300", "250/500", "500/1000"])
        policy_deductable = st.selectbox("Deductible (USD)", [500, 1000, 2000])
        policy_annual_premium = st.number_input(
            "Annual premium (USD)", 500.0, 3000.0, 1200.0, step=50.0)
        umbrella_limit = st.number_input(
            "Umbrella limit (USD)", 0, 10_000_000, 0, step=500_000)
        capital_gains = st.number_input(
            "Capital gains (USD)", 0, 200_000, 0, step=1000)
        capital_loss = st.number_input(
            "Capital loss (USD)", -200_000, 0, 0, step=1000)

    with col_right:
        st.markdown('<div class="section-title">🚨 Incident Details</div>',
                    unsafe_allow_html=True)
        incident_type = st.selectbox("Incident type", [
            "Single Vehicle Collision", "Multi-vehicle Collision",
            "Vehicle Theft", "Parked Car"
        ])
        collision_type = st.selectbox("Collision type", [
            "Side Collision", "Rear Collision", "Front Collision", "Unknown"
        ])
        incident_severity = st.selectbox("Incident severity", [
            "Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"
        ])
        authorities_contacted = st.selectbox("Authorities contacted", [
            "Police", "Fire", "Ambulance", "Other", "Unknown"
        ])
        incident_state = st.selectbox("Incident state",
                                       ["SC", "VA", "NY", "OH", "WV", "NC", "PA"])
        incident_city = st.selectbox("Incident city", [
            "Columbus", "Riverwood", "Arlington", "Springfield",
            "Hillsdale", "Northbend", "Northbrook"
        ])
        incident_hour = st.slider("Hour of incident (0-23)", 0, 23, 12)
        num_vehicles = st.slider("Number of vehicles involved", 1, 4, 1)
        property_damage = st.selectbox("Property damage?",
                                        ["YES", "NO", "Unknown"])
        bodily_injuries = st.slider("Bodily injuries", 0, 4, 0)
        witnesses = st.slider("Number of witnesses", 0, 4, 1)
        police_report = st.selectbox("Police report available?",
                                      ["YES", "NO", "Unknown"])

        st.markdown('<div class="section-title">💰 Claim Amounts</div>',
                    unsafe_allow_html=True)
        total_claim = st.number_input(
            "Total claim amount (USD)", 0, 200_000, 30_000, step=500)
        injury_claim = st.number_input(
            "Injury claim (USD)", 0, 100_000, 5_000, step=500)
        property_claim = st.number_input(
            "Property claim (USD)", 0, 100_000, 5_000, step=500)
        vehicle_claim = st.number_input(
            "Vehicle claim (USD)", 0, 100_000, 20_000, step=500)

        st.markdown('<div class="section-title">🚗 Vehicle Details</div>',
                    unsafe_allow_html=True)
        auto_make = st.selectbox("Vehicle make", [
            "Toyota", "Honda", "Ford", "Chevrolet", "Dodge", "Nissan",
            "Mercedes", "BMW", "Audi", "Saab", "Subaru", "Accura", "Jeep"
        ])
        auto_year = st.slider("Vehicle year", 1995, 2025, 2010)

    st.markdown("---")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_btn = st.button("🔍 Analyse Claim",
                                 use_container_width=True, type="primary")

    if predict_btn:
        raw_input = {
            'months_as_customer':          months_as_customer,
            'age':                         age,
            'policy_state':                policy_state,
            'policy_csl':                  policy_csl,
            'policy_deductable':           policy_deductable,
            'policy_annual_premium':       policy_annual_premium,
            'umbrella_limit':              umbrella_limit,
            'insured_sex':                 insured_sex,
            'insured_education_level':     insured_education_level,
            'insured_occupation':          insured_occupation,
            'insured_hobbies':             insured_hobbies,
            'insured_relationship':        insured_relationship,
            'capital-gains':               capital_gains,
            'capital-loss':                capital_loss,
            'incident_type':               incident_type,
            'collision_type':              collision_type,
            'incident_severity':           incident_severity,
            'authorities_contacted':       authorities_contacted,
            'incident_state':              incident_state,
            'incident_city':               incident_city,
            'incident_hour_of_the_day':    incident_hour,
            'number_of_vehicles_involved': num_vehicles,
            'property_damage':             property_damage,
            'bodily_injuries':             bodily_injuries,
            'witnesses':                   witnesses,
            'police_report_available':     police_report,
            'total_claim_amount':          total_claim,
            'injury_claim':                injury_claim,
            'property_claim':              property_claim,
            'vehicle_claim':               vehicle_claim,
            'auto_make':                   auto_make,
            'auto_year':                   auto_year,
        }

        input_df = pd.DataFrame([raw_input])
        preds, probs = preprocess_and_predict(input_df)
        prediction = preds[0]
        fraud_prob = probs[0]
        legit_prob = 1 - fraud_prob

        st.markdown("---")
        st.markdown("### 📋 Analysis Result")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val">
                {'🚨 FRAUD' if prediction == 1 else '✅ LEGIT'}
              </div>
              <div class="metric-lbl">Prediction</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val"
                style="color:{'#C00000' if fraud_prob > 0.5 else '#375623'}">
                {fraud_prob*100:.1f}%
              </div>
              <div class="metric-lbl">Fraud Probability</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val">{legit_prob*100:.1f}%</div>
              <div class="metric-lbl">Legitimate Probability</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            risk = get_risk(fraud_prob)
            rc = {"HIGH": "#C00000", "MEDIUM": "#C9A84C", "LOW": "#375623"}[risk]
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val" style="color:{rc}">{risk}</div>
              <div class="metric-lbl">Risk Level</div>
            </div>""", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
              <strong>🚨 FRAUDULENT CLAIM DETECTED</strong><br>
              Fraud probability: <strong>{fraud_prob*100:.1f}%</strong>.
              Recommended action:
              <strong>Refer to fraud investigation team for manual review.</strong>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
              <strong>✅ CLAIM APPEARS LEGITIMATE</strong><br>
              Legitimacy probability: <strong>{legit_prob*100:.1f}%</strong>.
              Recommended action:
              <strong>Proceed with standard claims processing.</strong>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### ⚑ Key Risk Indicators")
        flags = []
        if total_claim > 70_000:
            flags.append(f"🔴 High total claim amount (USD {total_claim:,})")
        if incident_severity == "Total Loss":
            flags.append("🔴 Claim involves total loss of vehicle")
        if police_report in ["NO", "Unknown"]:
            flags.append("🟡 No police report available")
        if witnesses == 0:
            flags.append("🟡 No witnesses recorded")
        if property_damage == "Unknown":
            flags.append("🟡 Property damage status unknown")
        if collision_type == "Unknown":
            flags.append("🟡 Collision type unknown")
        if months_as_customer < 6:
            flags.append("🔴 Very new customer (less than 6 months)")
        if not flags:
            flags.append("🟢 No major individual risk flags identified")
        for flag in flags:
            st.markdown(f"- {flag}")

        st.caption(
            "⚠️ This prediction is generated by a machine learning model. "
            "It is a decision-support tool only and should not replace "
            "expert human judgement."
        )


# ════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="tab-header">Upload a CSV file of insurance claims. '
        'The model analyses every claim automatically and returns a fraud '
        'prediction and risk score for each one. Download the full results '
        'as Excel.</div>',
        unsafe_allow_html=True
    )

    with st.expander("📖 How to use batch upload — click to read"):
        st.markdown("""
        **Steps:**
        1. Download the sample template below so you know what columns are needed
        2. Fill it with your real claims data (one row per claim)
        3. Save as CSV from Excel: File → Save As → CSV
        4. Upload the file using the uploader below
        5. Click Analyse — results appear instantly
        6. Download the Excel results file with 3 sheets:
           All Claims, Fraudulent Claims only, and a Summary

        **Tips:**
        - Missing columns are filled with 0 automatically
        - The fraud_reported column is ignored if present
        - Values marked as ? are treated as Unknown
        - There is no limit on number of rows
        """)

    # Sample template download
    st.markdown("**Step 1 — Download the sample template to see the format:**")
    sample = pd.DataFrame([
        {
            'months_as_customer': 120, 'age': 35, 'policy_state': 'OH',
            'policy_csl': '250/500', 'policy_deductable': 1000,
            'policy_annual_premium': 1200.0, 'umbrella_limit': 0,
            'insured_sex': 'MALE', 'insured_education_level': 'College',
            'insured_occupation': 'craft-repair', 'insured_hobbies': 'sleeping',
            'insured_relationship': 'husband', 'capital-gains': 0,
            'capital-loss': 0, 'incident_type': 'Single Vehicle Collision',
            'collision_type': 'Side Collision', 'incident_severity': 'Minor Damage',
            'authorities_contacted': 'Police', 'incident_state': 'OH',
            'incident_city': 'Columbus', 'incident_hour_of_the_day': 12,
            'number_of_vehicles_involved': 1, 'property_damage': 'YES',
            'bodily_injuries': 0, 'witnesses': 2,
            'police_report_available': 'YES', 'total_claim_amount': 30000,
            'injury_claim': 5000, 'property_claim': 5000,
            'vehicle_claim': 20000, 'auto_make': 'Toyota', 'auto_year': 2015,
        },
        {
            'months_as_customer': 3, 'age': 28, 'policy_state': 'IN',
            'policy_csl': '100/300', 'policy_deductable': 500,
            'policy_annual_premium': 900.0, 'umbrella_limit': 0,
            'insured_sex': 'FEMALE', 'insured_education_level': 'High School',
            'insured_occupation': 'sales', 'insured_hobbies': 'base-jumping',
            'insured_relationship': 'unmarried', 'capital-gains': 0,
            'capital-loss': 0, 'incident_type': 'Vehicle Theft',
            'collision_type': 'Unknown', 'incident_severity': 'Total Loss',
            'authorities_contacted': 'Unknown', 'incident_state': 'NY',
            'incident_city': 'Arlington', 'incident_hour_of_the_day': 2,
            'number_of_vehicles_involved': 1, 'property_damage': 'Unknown',
            'bodily_injuries': 0, 'witnesses': 0,
            'police_report_available': 'NO', 'total_claim_amount': 95000,
            'injury_claim': 0, 'property_claim': 5000, 'vehicle_claim': 90000,
            'auto_make': 'Mercedes', 'auto_year': 2022,
        }
    ])
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample.to_csv(index=False).encode('utf-8'),
        file_name="sample_claims_template.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("**Step 2 — Upload your claims file:**")

    uploaded_file = st.file_uploader(
        "Drop your CSV file here or click Browse",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            total_claims = len(df_upload)

            st.success(
                f"✅ **{uploaded_file.name}** uploaded successfully "
                f"— {total_claims} claims ready to analyse"
            )
            st.markdown("**Preview (first 3 rows):**")
            st.dataframe(df_upload.head(3), use_container_width=True)

            st.markdown("---")
            _, btn_col, _ = st.columns([2, 1, 2])
            with btn_col:
                analyse_btn = st.button(
                    f"🔍 Analyse All {total_claims} Claims",
                    use_container_width=True,
                    type="primary"
                )

            if analyse_btn:
                with st.spinner(f"Analysing {total_claims} claims — please wait..."):
                    preds, probs = preprocess_and_predict(df_upload)

                # Build results table
                results_df = df_upload.copy()
                results_df['Verdict']             = ['🚨 FRAUD' if p == 1 else '✅ LEGIT' for p in preds]
                results_df['Prediction']          = ['FRAUDULENT' if p == 1 else 'LEGITIMATE' for p in preds]
                results_df['Fraud_Probability_%'] = (probs * 100).round(1)
                results_df['Risk_Level']          = [get_risk(p) for p in probs]

                # Summary numbers
                n_fraud = int(sum(preds))
                n_legit = total_claims - n_fraud
                n_high  = sum(1 for p in probs if p > 0.7)
                n_med   = sum(1 for p in probs if 0.4 <= p <= 0.7)
                n_low   = sum(1 for p in probs if p < 0.4)
                avg_prob = float(np.mean(probs) * 100)

                st.markdown("---")
                st.markdown("### 📊 Results")

                # Stat cards
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown(f"""
                    <div class="stat-card">
                      <div class="stat-val">{total_claims}</div>
                      <div class="stat-lbl">Total Claims Analysed</div>
                    </div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="stat-card">
                      <div class="stat-val" style="color:#C00000">{n_fraud}</div>
                      <div class="stat-lbl">Flagged as Fraudulent</div>
                    </div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""
                    <div class="stat-card">
                      <div class="stat-val" style="color:#375623">{n_legit}</div>
                      <div class="stat-lbl">Cleared as Legitimate</div>
                    </div>""", unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""
                    <div class="stat-card">
                      <div class="stat-val" style="color:#C9A84C">{n_high}</div>
                      <div class="stat-lbl">High Risk Claims</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(
                    f"> **{n_fraud} of {total_claims} claims "
                    f"({n_fraud/total_claims*100:.1f}%) flagged as potentially "
                    f"fraudulent.** Average fraud probability: {avg_prob:.1f}%"
                )

                # Filter and sort
                st.markdown("#### 🔎 Filter & Sort")
                fc1, fc2 = st.columns(2)
                with fc1:
                    show_filter = st.selectbox("Show:", [
                        "All claims", "Fraudulent only",
                        "Legitimate only", "High risk only"
                    ])
                with fc2:
                    sort_by = st.selectbox("Sort by:", [
                        "Fraud probability (highest first)",
                        "Fraud probability (lowest first)",
                        "Original order"
                    ])

                display_df = results_df.copy()
                if show_filter == "Fraudulent only":
                    display_df = display_df[display_df['Prediction'] == 'FRAUDULENT']
                elif show_filter == "Legitimate only":
                    display_df = display_df[display_df['Prediction'] == 'LEGITIMATE']
                elif show_filter == "High risk only":
                    display_df = display_df[display_df['Risk_Level'] == 'HIGH']

                if sort_by == "Fraud probability (highest first)":
                    display_df = display_df.sort_values(
                        'Fraud_Probability_%', ascending=False)
                elif sort_by == "Fraud probability (lowest first)":
                    display_df = display_df.sort_values(
                        'Fraud_Probability_%', ascending=True)

                # Put result columns first
                key_cols  = ['Verdict', 'Fraud_Probability_%',
                             'Risk_Level', 'Prediction']
                other_cols = [c for c in display_df.columns
                              if c not in key_cols]
                st.dataframe(
                    display_df[key_cols + other_cols].reset_index(drop=True),
                    use_container_width=True,
                    height=400
                )
                st.caption(
                    f"Showing {len(display_df)} of {total_claims} claims"
                )

                # ── Excel download ────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📥 Download Results")

                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                    results_df.sort_values(
                        'Fraud_Probability_%', ascending=False
                    ).to_excel(writer, sheet_name='All Claims', index=False)

                    results_df[results_df['Prediction'] == 'FRAUDULENT'].sort_values(
                        'Fraud_Probability_%', ascending=False
                    ).to_excel(writer, sheet_name='Fraudulent Claims', index=False)

                    pd.DataFrame({
                        'Metric': [
                            'Total Claims Analysed',
                            'Flagged as Fraudulent',
                            'Cleared as Legitimate',
                            'High Risk (>70%)',
                            'Medium Risk (40-70%)',
                            'Low Risk (<40%)',
                            'Average Fraud Probability',
                            'Model Used',
                            'Analysis Date & Time'
                        ],
                        'Value': [
                            total_claims, n_fraud, n_legit,
                            n_high, n_med, n_low,
                            f"{avg_prob:.1f}%",
                            display_name,
                            datetime.now().strftime("%Y-%m-%d %H:%M")
                        ]
                    }).to_excel(writer, sheet_name='Summary', index=False)

                excel_buf.seek(0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")

                st.download_button(
                    label="📥 Download Full Results as Excel",
                    data=excel_buf,
                    file_name=f"fraud_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument"
                          ".spreadsheetml.sheet",
                    type="primary"
                )
                st.caption(
                    "Excel file has 3 sheets: All Claims | "
                    "Fraudulent Claims | Summary"
                )
                st.caption(
                    "⚠️ These predictions are for decision-support only. "
                    "All flagged claims should be reviewed by a qualified "
                    "fraud investigator before any action is taken."
                )

        except Exception as e:
            st.error(
                f"**Could not read the file:** {str(e)}\n\n"
                "Make sure it is saved as a CSV file (not .xlsx) and try again."
            )

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Funeral Insurance Fraud Detection Prototype &nbsp;|&nbsp;
  Tinashe Arthur Mupindu &nbsp;|&nbsp; University of Zambia &nbsp;|&nbsp;
  Supervisor: Dr Stanley Jere &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)
