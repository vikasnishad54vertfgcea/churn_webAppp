# ============================================================
#   ChurnIQ — Customer Churn Intelligence Platform v2.0
#   Author: Data Scientist
#   Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CURRENCY — 1 USD ≈ 84 INR
# ─────────────────────────────────────────────
USD_TO_INR = 84

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Fixed metric visibility + all boxes
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── METRIC CARDS — Force visible in BOTH light & dark mode ── */
    [data-testid="metric-container"] {
        background: #1e293b !important;
        padding: 18px 20px !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        border-left: 5px solid #818cf8 !important;
        border-top: 1px solid #334155 !important;
    }
    [data-testid="metric-container"] label,
    [data-testid="metric-container"] [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 30px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 20px rgba(129,140,248,0.4) !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    /* ── Risk boxes ── */
    .risk-high   { background:linear-gradient(135deg,#ff416c,#ff4b2b);
                   color:white; padding:22px; border-radius:14px; text-align:center; margin:8px 0; }
    .risk-medium { background:linear-gradient(135deg,#f7971e,#ffd200);
                   color:#1a1a1a; padding:22px; border-radius:14px; text-align:center; margin:8px 0; }
    .risk-low    { background:linear-gradient(135deg,#11998e,#38ef7d);
                   color:white; padding:22px; border-radius:14px; text-align:center; margin:8px 0; }

    /* ── Revenue simulator box ── */
    .revenue-box {
        background:linear-gradient(135deg,#1a1a2e,#16213e);
        color:white; padding:24px; border-radius:16px;
        margin:10px 0; border:1px solid #667eea;
    }
    .rev-saved { color:#38ef7d; font-size:26px; font-weight:800; }
    .rev-loss  { color:#ff6b6b; font-size:20px; font-weight:700; }

    /* ── Explain cards ── */
    .explain-card {
        background:#f0f4ff; border-left:5px solid #667eea;
        padding:14px 18px; border-radius:10px; margin:5px 0;
    }
    .explain-card span { color:#1a1a2e; font-weight:600; }
    .explain-green {
        background:#f0faf4; border-left:5px solid #2ecc71;
        padding:14px 18px; border-radius:10px; margin:5px 0;
    }
    .explain-green span { color:#1a6b3c; font-weight:600; }

    /* ── Cost decision ── */
    .cost-yes { background:#e8faf0; border-left:5px solid #2ecc71;
                padding:16px; border-radius:10px; margin:6px 0; }
    .cost-no  { background:#fdf0f0; border-left:5px solid #e74c3c;
                padding:16px; border-radius:10px; margin:6px 0; }

    h1 { color:#1a1a2e !important; }
    h2 { color:#2c3e50 !important; }
    .stTabs [data-baseweb="tab"] { font-size:14px; font-weight:600; }
    .main { background-color:#f5f7fa; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_risk_level(prob):
    if prob >= 0.70:   return "🔴 HIGH RISK",   "risk-high",   "#ff416c"
    elif prob >= 0.30: return "🟡 MEDIUM RISK", "risk-medium", "#f7971e"
    else:              return "🟢 LOW RISK",    "risk-low",    "#11998e"


def explain_prediction(inp, prob):
    reasons   = []
    positives = []
    tenure   = inp.get('tenure', 0)
    monthly  = inp.get('MonthlyCharges', 0)
    contract = inp.get('Contract', '')
    internet = inp.get('InternetService', '')
    tech     = inp.get('TechSupport', '')
    senior   = inp.get('SeniorCitizen', '')
    payment  = inp.get('PaymentMethod', '')
    security = inp.get('OnlineSecurity', '')

    if contract == "Month-to-month":
        reasons.append(("📋 Month-to-month contract",
                        "No commitment = easy to leave. Churn risk +43%"))
    if tenure < 12:
        reasons.append(("⏳ New customer (tenure < 12 months)",
                        "First year is most vulnerable. Loyalty not yet built."))
    if monthly > 6300:
        reasons.append((f"💸 High monthly charges (₹{int(monthly):,})",
                        "Premium payer expects premium experience — unmet = churn."))
    if internet == "Fiber optic":
        reasons.append(("🌐 Fiber Optic service",
                        "Fiber users churn at ~42% — likely unmet expectations."))
    if tech == "No":
        reasons.append(("🛠️ No Tech Support",
                        "Customers without support feel abandoned during issues."))
    if security == "No":
        reasons.append(("🔒 No Online Security add-on",
                        "Unprotected customers feel less value in the package."))
    if senior == "Yes":
        reasons.append(("👴 Senior Citizen",
                        "Seniors churn at 42% — may struggle with billing complexity."))
    if payment == "Electronic check":
        reasons.append(("💳 Electronic Check payment",
                        "Manual payers are less sticky. AutoPay users churn far less."))

    if contract in ["One year", "Two year"]:
        positives.append("✅ Long-term contract — strong retention signal")
    if tenure > 24:
        positives.append("✅ Long-tenure customer — proven loyalty")
    if tech == "Yes":
        positives.append("✅ Has Tech Support — reduces frustration churn")
    if monthly < 3000:
        positives.append("✅ Low monthly charges — good value perception")
    if payment in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        positives.append("✅ AutoPay active — stickier billing relationship")

    return reasons, positives


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    import os
    local_files = ['Telco-Customer-Churn.csv', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']
    df = None
    for fname in local_files:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            break
    if df is None:
        for url in [
            'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
            'https://raw.githubusercontent.com/carrie-czx/TELCO/main/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        ]:
            try:
                df = pd.read_csv(url); break
            except Exception:
                continue
    if df is None:
        st.error("❌ Dataset not found! Download from Kaggle → rename to `Telco-Customer-Churn.csv` → place in app folder.")
        st.stop()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    df['SeniorCitizen']  = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df['MonthlyCharges'] = (df['MonthlyCharges'] * USD_TO_INR).round(0)
    df['TotalCharges']   = (df['TotalCharges']   * USD_TO_INR).round(0)
    return df


@st.cache_resource
def train_models(df):
    df_m = df.copy()
    df_m['Churn'] = LabelEncoder().fit_transform(df_m['Churn'])
    cat_cols = df_m.select_dtypes(include='object').columns.tolist()
    df_enc   = pd.get_dummies(df_m, columns=cat_cols, drop_first=True)
    X, y     = df_enc.drop(columns=['Churn']), df_enc['Churn']
    scaler   = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return lr, rf, X_te, y_te, X.columns.tolist(), scaler


def predict_customer(model, scaler, fcols, inp):
    idf = pd.DataFrame([inp])
    idf['TotalCharges'] = pd.to_numeric(idf['TotalCharges'], errors='coerce').fillna(0)
    cat_cols = idf.select_dtypes(include='object').columns.tolist()
    ienc = pd.get_dummies(idf, columns=cat_cols, drop_first=True)
    for c in fcols:
        if c not in ienc.columns: ienc[c] = 0
    ienc = ienc[fcols]
    ienc[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
        ienc[['tenure','MonthlyCharges','TotalCharges']])
    prob = model.predict_proba(ienc)[0][1]
    return int(prob >= 0.5), prob


# ─────────────────────────────────────────────
# LOAD + TRAIN
# ─────────────────────────────────────────────
with st.spinner('🔄 Loading data & training models... (first time ~30s)'):
    df = load_and_prepare_data()
    lr_model, rf_model, X_test, y_test, fcols, scaler = train_models(df)

lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
lr_acc   = accuracy_score(y_test, lr_preds)
rf_acc   = accuracy_score(y_test, rf_preds)
lr_auc   = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1])
rf_auc   = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
churned  = (df['Churn'] == 'Yes').sum()
total_c  = df.shape[0]
churn_rt = churned / total_c * 100


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ChurnIQ")
    st.markdown("*Customer Intelligence Platform v2.0*")
    st.markdown("---")
    page = st.radio("📍 Navigate", [
        "🏠 Home & Overview",
        "🔮 Predict Customer",
        "💰 Revenue Impact Simulator",
        "📈 Model Performance",
        "📊 Customer Behavior Analysis",
        "💡 Business Insights"
    ])
    st.markdown("---")
    st.markdown(f"**👥 Total Customers:** `{total_c:,}`")
    st.markdown(f"**📉 Churn Rate:** `{churn_rt:.1f}%`")
    st.markdown(f"**💰 Avg Monthly:** `₹{df['MonthlyCharges'].mean():,.0f}`")
    st.markdown(f"**🎯 RF Accuracy:** `{rf_acc*100:.1f}%`")
    st.markdown(f"**📊 ROC-AUC:** `{rf_auc:.3f}`")
    st.markdown(f"**💱 Currency:** `₹ INR (1$=₹{USD_TO_INR})`")
    st.markdown("---")
    st.caption("Built with ❤️ Streamlit + Scikit-learn")


# ════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════
if page == "🏠 Home & Overview":

    st.title("🧠 ChurnIQ — Customer Intelligence Dashboard")
    st.markdown("##### Predict · Explain · Retain — ML-Powered Churn Solution")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers",  f"{total_c:,}")
    c2.metric("🚪 Churned",          f"{churned:,}",         delta=f"-{churn_rt:.1f}%",       delta_color="inverse")
    c3.metric("💚 Retained",         f"{total_c-churned:,}", delta=f"+{100-churn_rt:.1f}%")
    c4.metric("🎯 RF Accuracy",      f"{rf_acc*100:.1f}%",   delta="Best Model ✅")
    c5.metric("📊 ROC-AUC",          f"{rf_auc:.3f}",        delta="Excellent 🏆")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🥧 Churn Distribution")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        counts = df['Churn'].value_counts()
        ax.pie(counts, labels=['No Churn','Churn'], autopct='%1.1f%%',
               colors=['#2ecc71','#e74c3c'], startangle=90,
               wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('Overall Churn Rate', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📋 Contract vs Churn")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cd = df.groupby(['Contract','Churn']).size().unstack()
        (cd.div(cd.sum(axis=1),axis=0)*100).plot(
            kind='bar', ax=ax, color=['#2ecc71','#e74c3c'], edgecolor='white', rot=15)
        ax.set_ylabel('%'); ax.legend(['No Churn','Churn'], fontsize=8)
        ax.set_title('Contract vs Churn', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col3:
        st.subheader("⏳ Tenure vs Churn")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        for label, color in [('No','#2ecc71'),('Yes','#e74c3c')]:
            ax.hist(df[df['Churn']==label]['tenure'], bins=20,
                    alpha=0.7, color=color, label=f'Churn:{label}', edgecolor='white')
        ax.set_xlabel('Tenure (months)'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.set_title('Tenure Distribution', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📌 Platform Modules")
    st.markdown("""
    | Module | What You Get |
    |--------|-------------|
    | 🔮 **Predict Customer** | Churn probability + Risk Level + Explanation + Cost Decision + Download Report |
    | 💰 **Revenue Simulator** | Calculate revenue loss, savings, ROI from retention campaigns |
    | 📈 **Model Performance** | Accuracy, Confusion Matrix, ROC Curve, Feature Importance |
    | 📊 **Customer Behavior** | Interactive explorer — filter any feature, download segments |
    | 💡 **Business Insights** | Strategies + Action Priority Matrix + Revenue impact estimate |
    """)


# ════════════════════════════════════════════════════
# PAGE 2 — PREDICT CUSTOMER
# ════════════════════════════════════════════════════
elif page == "🔮 Predict Customer":

    st.title("🔮 Customer Churn Predictor")
    st.markdown("Fill in customer profile → **Risk Level · Probability · Explanation · Cost Decision · Download**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender       = st.selectbox("Gender",           ["Male","Female"])
        senior       = st.selectbox("Senior Citizen",   ["No","Yes"])
        partner      = st.selectbox("Has Partner",      ["Yes","No"])
        dependents   = st.selectbox("Has Dependents",   ["Yes","No"])

    with col2:
        st.subheader("📱 Services")
        phone_service  = st.selectbox("Phone Service",    ["Yes","No"])
        multiple_lines = st.selectbox("Multiple Lines",   ["No","Yes","No phone service"])
        internet       = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        online_sec     = st.selectbox("Online Security",  ["No","Yes","No internet service"])
        online_backup  = st.selectbox("Online Backup",    ["No","Yes","No internet service"])
        device_protect = st.selectbox("Device Protection",["No","Yes","No internet service"])
        tech_support   = st.selectbox("Tech Support",     ["No","Yes","No internet service"])
        streaming_tv   = st.selectbox("Streaming TV",     ["No","Yes","No internet service"])
        streaming_mov  = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    with col3:
        st.subheader("💳 Billing")
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        monthly        = st.number_input("Monthly Charges (₹)", 840, 10080, 5460, step=50)
        total          = st.number_input("Total Charges (₹)", 0, 756000, int(tenure*monthly), step=100)
        contract       = st.selectbox("Contract Type",    ["Month-to-month","One year","Two year"])
        paperless      = st.selectbox("Paperless Billing",["Yes","No"])
        payment_method = st.selectbox("Payment Method",   [
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"])
        model_choice   = st.radio("🤖 Model", ["Random Forest","Logistic Regression"])

    st.markdown("---")

    if st.button("🚀 Predict Churn Now", use_container_width=True, type="primary"):

        inp = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_backup,
            "DeviceProtection": device_protect, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly, "TotalCharges": total
        }

        model = rf_model if model_choice == "Random Forest" else lr_model
        prediction, prob = predict_customer(model, scaler, fcols, inp)
        risk_label, risk_class, risk_color = get_risk_level(prob)
        reasons, positives = explain_prediction(inp, prob)

        st.info(f"📋 **Tenure:** {tenure}mo  |  **Monthly:** ₹{monthly:,}  |  "
                f"**Total Paid:** ₹{total:,}  |  **Contract:** {contract}  |  **Internet:** {internet}")

        # ── RESULT ──────────────────────────────────────────
        r1, r2, r3 = st.columns([1.2, 1, 1])

        with r1:
            st.markdown(f"""
            <div class="{risk_class}">
                <h2>{risk_label}</h2>
                <h1 style="font-size:54px;margin:10px 0">{prob*100:.1f}%</h1>
                <p style="font-size:16px;margin:4px">Churn Probability</p>
                <p style="font-size:12px;opacity:0.8">via {model_choice}</p>
            </div>""", unsafe_allow_html=True)

        with r2:
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            bars = ax.barh(['Stay','Churn'], [1-prob, prob],
                           color=['#2ecc71', risk_color], edgecolor='white', height=0.45)
            for bar, val in zip(bars, [1-prob, prob]):
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                        f'{val*100:.1f}%', va='center', fontweight='bold', fontsize=13)
            ax.set_xlim(0, 1.25)
            ax.set_title('Probability Breakdown', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with r3:
            # Risk Segmentation — which zone is customer in
            st.markdown("#### 📊 Risk Zone")
            fig, ax = plt.subplots(figsize=(4, 2.8))
            seg_vals = [0.30, 0.40, 0.30]
            seg_colors = ['#2ecc71','#f7971e','#e74c3c']
            seg_labels = ['Low\n0–30%','Medium\n30–70%','High\n70%+']
            brs = ax.bar(seg_labels, seg_vals, color=seg_colors, edgecolor='white', width=0.55)
            zone = 0 if prob < 0.30 else (1 if prob < 0.70 else 2)
            brs[zone].set_edgecolor('black'); brs[zone].set_linewidth(3)
            ax.set_ylim(0, 0.55); ax.set_ylabel('Segment')
            ax.set_title(f'Customer is in: {seg_labels[zone].split(chr(10))[0]} Zone',
                         fontweight='bold', fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")

        # ── IMPROVEMENT 4: EXPLAINABILITY ───────────────────
        e1, e2 = st.columns(2)
        with e1:
            st.subheader("🔍 Why Will This Customer Churn?")
            if reasons:
                for title, detail in reasons:
                    st.markdown(f"""
                    <div class="explain-card">
                        <span>⚠️ {title}</span><br>
                        <small style="color:#666">{detail}</small>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ No major churn signals detected.")

        with e2:
            st.subheader("💪 Why They Might Stay")
            if positives:
                for p in positives:
                    st.markdown(f"""
                    <div class="explain-green">
                        <span>{p}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No strong retention signals found.")

        # ── IMPROVEMENT 3: COST-SENSITIVE DECISION ──────────
        st.markdown("---")
        st.subheader("💡 Cost-Sensitive Retention Decision")
        cs1, cs2 = st.columns([1,1.2])
        with cs1:
            discount_amt   = st.number_input("Retention Offer / Discount (₹/month)", 100, 3000, 300, step=50)
            retention_rate = st.slider("Expected Retention Success (%)", 10, 90, 40,
                                        help="What % chance this customer stays after offer?")
        with cs2:
            net_gain      = (monthly - discount_amt) * (retention_rate / 100)
            annual_loss   = monthly * 12
            total_camp    = discount_amt
            profitable    = net_gain > 0

            if profitable:
                st.markdown(f"""
                <div class="cost-yes">
                    <b style="font-size:16px;color:#1a6b3c">✅ RETENTION IS PROFITABLE — Make the offer!</b><br><br>
                    Monthly Revenue &nbsp;&nbsp;&nbsp;: <b>₹{monthly:,.0f}</b><br>
                    Discount Offered &nbsp;&nbsp;&nbsp;: <b>₹{discount_amt:,.0f}</b><br>
                    Net After Discount : <b>₹{monthly-discount_amt:,.0f}</b><br>
                    Success Probability: <b>{retention_rate}%</b><br>
                    <b>Expected Net Gain : ₹{net_gain:,.0f}/month 💚</b><br>
                    Annual Loss if Lost : <span style="color:#c0392b">₹{annual_loss:,.0f}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="cost-no">
                    <b style="font-size:16px;color:#c0392b">❌ NOT PROFITABLE at this discount level</b><br><br>
                    Net Gain would be: <b>₹{net_gain:,.0f}</b> (negative)<br>
                    Annual Loss if Churns: <b>₹{annual_loss:,.0f}</b><br><br>
                    👉 <b>Try non-monetary retention:</b> free trial upgrade,
                    dedicated support agent, or loyalty points instead.
                </div>""", unsafe_allow_html=True)

        # ── RECOMMENDATIONS ──────────────────────────────────
        st.markdown("---")
        st.subheader("🎯 Recommended Retention Actions")
        recs = []
        if contract == "Month-to-month":
            recs.append("📋 **1-Year Contract** offer karo — 20% discount ke saath")
        if monthly > 6300:
            recs.append("💰 **Bundle Package** — Internet + TV + Phone at 25% off")
        if tech_support == "No":
            recs.append("🛠️ **Free TechSupport** 3-month trial — proven retention booster")
        if tenure < 12:
            recs.append("🎁 **New Customer Loyalty Program** — first-year cashback ya rewards")
        if senior == "Yes":
            recs.append("👴 **Senior Care Plan** — dedicated agent + simplified billing")
        if payment_method == "Electronic check":
            recs.append("💳 **AutoPay Setup** — ₹100/month discount incentive do")
        if online_sec == "No":
            recs.append("🔒 **Online Security** free trial — safety = retention value")
        if not recs:
            recs.append("📞 **Proactive satisfaction call** + loyalty reward offer karo")

        for r in recs:
            st.warning(r)

        # ── IMPROVEMENT 5: DOWNLOAD REPORT ──────────────────
        st.markdown("---")
        st.subheader("📥 Download Prediction Report")

        now_str = datetime.datetime.now().strftime('%d %b %Y %I:%M %p')
        report  = [
            "=" * 55,
            "     CHURNIQ — CUSTOMER CHURN PREDICTION REPORT",
            "=" * 55,
            f"  Generated : {now_str}",
            f"  Model     : {model_choice}",
            f"  Churn Prob: {prob*100:.1f}%",
            f"  Risk Level: {risk_label}",
            "-" * 55,
            "  CUSTOMER PROFILE",
            f"  Tenure          : {tenure} months",
            f"  Monthly Charges : ₹{monthly:,}",
            f"  Total Charges   : ₹{total:,}",
            f"  Contract        : {contract}",
            f"  Internet        : {internet}",
            f"  Senior Citizen  : {senior}",
            f"  Tech Support    : {tech_support}",
            f"  Payment Method  : {payment_method}",
            "-" * 55,
            "  CHURN REASONS",
        ]

        # Add reasons safely (fixes SyntaxError)
        if reasons:
            for t, d in reasons:
                report.append(f"  ⚠ {t}")
                report.append(f"    {d}")
        else:
            report.append("  No major risk factors detected.")

        report += [
            "-" * 55,
            "  COST DECISION",
            f"  Discount Offered: ₹{discount_amt:,}/month",
            f"  Success Rate    : {retention_rate}%",
            f"  Expected Net    : ₹{net_gain:,.0f}/month",
            f"  Decision        : {'✅ OFFER DISCOUNT' if profitable else '❌ USE NON-MONETARY RETENTION'}",
            "-" * 55,
            "  ACTIONS RECOMMENDED",
        ]
        for r in recs:
            report.append(f"  • {r.replace('**', '')}")
        report.append("=" * 55)

        csv_data = pd.DataFrame([{
            'Date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'Model': model_choice,
            'Churn_%': round(prob*100,1),
            'Risk_Level': risk_label,
            'Tenure_months': tenure,
            'Monthly_INR': monthly,
            'Total_INR': total,
            'Contract': contract,
            'Internet': internet,
            'Senior': senior,
            'TechSupport': tech_support,
            'Payment': payment_method,
            'Discount_INR': discount_amt,
            'Retention_%': retention_rate,
            'Net_Gain_INR': round(net_gain,0),
            'Profitable': 'Yes' if profitable else 'No'
        }]).to_csv(index=False)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("📄 Download TXT Report", report,
                file_name=f"churn_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True)
        with dl2:
            st.download_button("📊 Download CSV Report", csv_data,
                file_name=f"churn_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)


# ════════════════════════════════════════════════════
# PAGE 3 — REVENUE IMPACT SIMULATOR
# ════════════════════════════════════════════════════
elif page == "💰 Revenue Impact Simulator":

    st.title("💰 Revenue Impact Simulator")
    st.markdown("##### Quantify churn loss — and prove the ROI of your retention campaigns")
    st.markdown("---")

    s1, s2, s3 = st.columns(3)
    with s1:
        avg_rev       = st.number_input("Avg Monthly Revenue / Customer (₹)", 500, 20000, 5460, step=100)
        high_risk_n   = st.number_input("High-Risk Customers Identified",       10, 200000, 500, step=10)
    with s2:
        ret_pct       = st.slider("Expected Retention Rate (%)", 5, 80, 30,
                                   help="% of high-risk customers you expect to retain")
        camp_cost     = st.number_input("Campaign Cost per Customer (₹)", 0, 5000, 300, step=50,
                                         help="Discount + calling cost per customer")
    with s3:
        months_proj   = st.slider("Projection Period (months)", 1, 24, 12)

    st.markdown("---")

    # Calculations
    total_risk_rev  = avg_rev * high_risk_n * months_proj
    retained_n      = int(high_risk_n * ret_pct / 100)
    lost_n          = high_risk_n - retained_n
    rev_loss        = avg_rev * lost_n * months_proj
    rev_saved_gross = avg_rev * retained_n * months_proj
    total_camp_cost = camp_cost * high_risk_n
    net_saved       = rev_saved_gross - total_camp_cost
    roi             = ((net_saved - total_camp_cost) / max(total_camp_cost, 1)) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💸 Total Revenue at Risk",    f"₹{total_risk_rev:,.0f}")
    m2.metric("🔴 Revenue Lost (No Action)", f"₹{rev_loss:,.0f}",
              delta=f"{lost_n} customers", delta_color="inverse")
    m3.metric("🟢 Net Revenue Saved",        f"₹{net_saved:,.0f}",
              delta=f"{retained_n} retained ✅")
    m4.metric("🚀 Campaign ROI",             f"{roi:.0f}%")

    st.markdown("---")
    st.markdown(f"""
    <div class="revenue-box">
        <h2 style="color:#a78bfa">📈 {months_proj}-Month Revenue Simulation</h2>
        <table width="100%" style="color:white;font-size:15px;border-collapse:collapse">
            <tr><td style="padding:7px 0">👥 High-Risk Customers</td>
                <td style="text-align:right;font-weight:700">{high_risk_n:,}</td></tr>
            <tr style="border-top:1px solid #333">
                <td style="padding:7px 0">💸 Total Revenue at Risk</td>
                <td style="text-align:right" class="rev-loss">₹{total_risk_rev:,.0f}</td></tr>
            <tr style="border-top:1px solid #333">
                <td style="padding:7px 0">📉 Customers Lost (no campaign)</td>
                <td style="text-align:right;color:#ff6b6b;font-weight:700">{lost_n:,}</td></tr>
            <tr style="border-top:1px solid #333">
                <td style="padding:7px 0">✅ Customers Retained</td>
                <td style="text-align:right;color:#38ef7d;font-weight:700">{retained_n:,}</td></tr>
            <tr style="border-top:1px solid #333">
                <td style="padding:7px 0">💰 Gross Revenue Saved</td>
                <td style="text-align:right;color:#38ef7d;font-weight:700">₹{rev_saved_gross:,.0f}</td></tr>
            <tr style="border-top:1px solid #333">
                <td style="padding:7px 0">📦 Campaign Cost (total)</td>
                <td style="text-align:right;color:#ffd200;font-weight:700">₹{total_camp_cost:,.0f}</td></tr>
            <tr style="border-top:2px solid #667eea">
                <td style="padding:10px 0;font-size:18px;font-weight:800">🏆 Net Revenue Saved</td>
                <td style="text-align:right" class="rev-saved">₹{net_saved:,.0f}</td></tr>
            <tr><td style="padding:5px 0">📊 Campaign ROI</td>
                <td style="text-align:right;color:#a78bfa;font-weight:700">{roi:.0f}%</td></tr>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(6,4))
        cats   = ['At Risk','Rev Lost','Rev Saved\n(Gross)','Camp\nCost','Net\nSaved']
        vals   = [total_risk_rev, rev_loss, rev_saved_gross, total_camp_cost, net_saved]
        bcolors= ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
        brs    = ax.bar(cats, [v/100000 for v in vals], color=bcolors, edgecolor='white', width=0.55)
        for bar, val in zip(brs, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f'₹{val/100000:.1f}L', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('Amount (₹ Lakhs)')
        ax.set_title('Revenue Breakdown', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with v2:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.pie([retained_n, lost_n],
               labels=[f'Retained ({retained_n:,})',f'Lost ({lost_n:,})'],
               autopct='%1.0f%%', colors=['#2ecc71','#e74c3c'],
               startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('High-Risk Customer Outcome', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    sim_csv = pd.DataFrame([{
        'Avg_Monthly_Rev':avg_rev,'High_Risk_Count':high_risk_n,
        'Retention_%':ret_pct,'Camp_Cost_Each':camp_cost,'Months':months_proj,
        'Total_At_Risk':total_risk_rev,'Rev_Lost':rev_loss,
        'Retained_Count':retained_n,'Rev_Saved_Gross':rev_saved_gross,
        'Total_Camp_Cost':total_camp_cost,'Net_Saved':net_saved,'ROI_%':round(roi,1)
    }]).to_csv(index=False)
    st.download_button("📥 Download Simulation (CSV)", sim_csv,
                        f"simulation_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


# ════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.title("📈 Model Performance & Evaluation")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("LR Accuracy", f"{lr_acc*100:.2f}%")
    c2.metric("RF Accuracy", f"{rf_acc*100:.2f}%", delta=f"+{(rf_acc-lr_acc)*100:.2f}%")
    c3.metric("LR ROC-AUC", f"{lr_auc:.4f}")
    c4.metric("RF ROC-AUC", f"{rf_auc:.4f}", delta=f"+{(rf_auc-lr_auc):.4f}")

    st.markdown("---")
    cl, cr = st.columns(2)

    with cl:
        st.subheader("🔢 Confusion Matrices")
        fig, axes = plt.subplots(1,2,figsize=(9,4))
        for ax,preds,name in zip(axes,[lr_preds,rf_preds],['Logistic Reg','Random Forest']):
            cm = confusion_matrix(y_test,preds)
            sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
                        xticklabels=['No','Yes'],yticklabels=['No','Yes'],linewidths=0.5)
            ax.set_title(name,fontweight='bold')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cr:
        st.subheader("📉 ROC Curves")
        fig, ax = plt.subplots(figsize=(6,4))
        for m,name,color in [(lr_model,'Logistic Reg','#3498db'),(rf_model,'Random Forest','#e74c3c')]:
            fpr,tpr,_ = roc_curve(y_test, m.predict_proba(X_test)[:,1])
            ax.plot(fpr,tpr,label=f'{name} (AUC={roc_auc_score(y_test,m.predict_proba(X_test)[:,1]):.3f})',
                    color=color,lw=2)
        ax.plot([0,1],[0,1],'k--',lw=1,label='Random (0.500)')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC Curve Comparison',fontweight='bold'); ax.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🏆 Top 15 Feature Importances — Random Forest")
    fi = pd.DataFrame({'Feature':fcols,'Importance':rf_model.feature_importances_}
                       ).sort_values('Importance',ascending=False).head(15)
    fig,ax = plt.subplots(figsize=(10,6))
    colors = sns.color_palette('RdYlGn_r',n_colors=15)
    ax.barh(fi['Feature'][::-1], fi['Importance'][::-1], color=colors, edgecolor='white')
    for i,(val,_) in enumerate(zip(fi['Importance'][::-1],fi['Feature'][::-1])):
        ax.text(val+0.0005,i,f'{val:.4f}',va='center',fontsize=8)
    ax.set_title('Feature Importance',fontweight='bold'); ax.set_xlabel('Importance Score')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Model Comparison")
    st.dataframe(pd.DataFrame({
        'Model':          ['Logistic Regression','Random Forest'],
        'Accuracy':       [f'{lr_acc*100:.2f}%', f'{rf_acc*100:.2f}%'],
        'ROC-AUC':        [f'{lr_auc:.4f}', f'{rf_auc:.4f}'],
        'Explainability': ['High ✅','Medium ⚠️'],
        'Speed':          ['Fast ⚡','Moderate 🔄'],
        'Best For':       ['Presentations & Reporting','Production & Accuracy ✅']
    }), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER BEHAVIOR ANALYSIS
# ════════════════════════════════════════════════════
elif page == "📊 Customer Behavior Analysis":

    st.title("📊 Customer Behavior Analysis")
    st.markdown("##### Deep-dive into patterns — filter, explore, and export segments")
    st.markdown("---")

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("📋 Dataset Sample")
        n = st.slider("Rows to display", 5, 100, 10)
        st.dataframe(df.head(n), use_container_width=True)
    with t2:
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe().round(0), use_container_width=True)

    st.markdown("---")
    b1, b2 = st.columns(2)

    with b1:
        st.subheader("📈 Feature vs Churn Rate")
        cat_opts = ['Contract','InternetService','PaymentMethod','gender',
                    'SeniorCitizen','Partner','Dependents','TechSupport',
                    'OnlineSecurity','StreamingTV','PaperlessBilling']
        sel = st.selectbox("Select feature:", cat_opts)
        fig,ax = plt.subplots(figsize=(6,4))
        grp = df.groupby([sel,'Churn']).size().unstack()
        (grp.div(grp.sum(axis=1),axis=0)*100).plot(
            kind='bar',ax=ax,color=['#2ecc71','#e74c3c'],edgecolor='white',rot=20)
        ax.set_title(f'{sel} vs Churn Rate',fontweight='bold')
        ax.set_ylabel('%'); ax.legend(['No Churn','Churn'])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with b2:
        st.subheader("💰 Monthly Charges Distribution")
        fig,ax = plt.subplots(figsize=(6,4))
        for label,color in [('No','#2ecc71'),('Yes','#e74c3c')]:
            ax.hist(df[df['Churn']==label]['MonthlyCharges'],bins=25,
                    alpha=0.7,color=color,label=f'Churn:{label}',edgecolor='white')
        ax.set_xlabel('Monthly Charges (₹)'); ax.set_ylabel('Count')
        ax.set_title('Monthly Charges vs Churn',fontweight='bold'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🔍 Filter Customers")
    f1,f2,f3 = st.columns(3)
    with f1:
        fc = st.multiselect("Contract",       df['Contract'].unique().tolist(),       default=df['Contract'].unique().tolist())
    with f2:
        fi2= st.multiselect("Internet Service",df['InternetService'].unique().tolist(),default=df['InternetService'].unique().tolist())
    with f3:
        fch= st.multiselect("Churn Status",   ['Yes','No'],                           default=['Yes','No'])

    filtered = df[df['Contract'].isin(fc) & df['InternetService'].isin(fi2) & df['Churn'].isin(fch)]
    st.markdown(f"**Filtered: {len(filtered):,} customers**")
    st.dataframe(filtered.head(50), use_container_width=True)
    st.download_button("📥 Export Filtered Customers (CSV)", filtered.to_csv(index=False),
                        "filtered_customers.csv", "text/csv")


# ════════════════════════════════════════════════════
# PAGE 6 — BUSINESS INSIGHTS
# ════════════════════════════════════════════════════
elif page == "💡 Business Insights":

    st.title("💡 Business Insights & Strategy")
    st.markdown("---")

    r1, r2 = st.columns(2)
    with r1:
        st.error("🔴 **Month-to-Month customers** churn at ~43%")
        st.error("🔴 **New customers (0–12 months)** — highest risk window")
        st.error("🔴 **Fiber Optic users** churn at ~42%")
        st.error("🔴 **Senior citizens** churn at ~42% vs 23% others")
    with r2:
        st.success("🟢 **Two-year contracts** — only 3% churn")
        st.success("🟢 **Long-tenure (2+ years)** — proven loyalty")
        st.success("🟢 **TechSupport subscribers** churn significantly less")
        st.success("🟢 **AutoPay customers** are far stickier than manual payers")

    st.markdown("---")

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["📋 Contracts","💰 Pricing","🎯 Retention","👴 Seniors","💡 Why It Matters"])

    with tab1:
        st.markdown("""### 📋 Contract Strategy
- **15–20% discount** — monthly to annual upgrade incentive
- **"First-Year Loyalty Bonus"** — points, free month, or data upgrade
- **Month 10 nudge** — automatic renewal reminder + offer
- Month-to-month users > 6 months → **priority outreach list**""")

    with tab2:
        st.markdown("""### 💰 Pricing Strategy
- **Fiber Optic pricing review** — high churn = value gap
- **Price Lock Guarantee** — 2-year plan pe fixed rates
- **Bundle discount** — Internet + TV + Phone at 25% off
- Monthly > ₹6,700 → **automatic retention offer trigger**""")

    with tab3:
        st.markdown("""### 🎯 Proactive Retention
- **Monthly ML scoring** — identify high-risk before they call to cancel
- **60%+ probability** → dedicated retention team call
- **TechSupport + Security** push — both reduce churn significantly
- **AutoPay migration** — ₹100/month incentive to switch from electronic check
- High-value + high-risk → **dedicated Customer Success Manager**""")

    with tab4:
        st.markdown("""### 👴 Senior Citizen Program
- **"Senior Care Plan"** — simple billing, lower price, priority support
- **Dedicated senior helpline** — no IVR, direct human agent
- **Monthly welfare call** during first year
- **In-home tech assistance** for internet/streaming setup
- **Family bundle** — senior + family member discount""")

    with tab5:
        st.subheader("💡 Why It Matters — The Real Numbers")
        saved_c  = int(total_c * 0.065)
        avg_m    = df['MonthlyCharges'].mean()
        annual_s = saved_c * avg_m * 12
        acq_s    = saved_c * 5000
        st.markdown(f"""
**If churn drops just 6.5% (from 26.5% → 20%):**

| Metric | Value |
|--------|-------|
| Extra Customers Saved | **{saved_c:,}** |
| Avg Monthly Revenue | **₹{avg_m:,.0f}** |
| Annual Revenue Recovered | **₹{annual_s:,.0f}** |
| Acquisition Cost Saved (@ ₹5k) | **₹{acq_s:,.0f}** |
| **TOTAL ANNUAL GAIN** | **₹{annual_s+acq_s:,.0f}** |
        """)
        st.success(f"🏆 **Total Estimated Annual Gain: ₹{(annual_s+acq_s):,.0f}**  "
                   f"— just from a 6.5% churn reduction!")

    st.markdown("---")
    st.subheader("📊 Action Priority Matrix")
    st.dataframe(pd.DataFrame({
        'Strategy':            ['Contract Upgrade','Fiber Pricing Review','ML Outreach',
                                'Senior Care Plan','Tech Support Push','AutoPay Migration'],
        'Churn Reduction':     ['8–12%','5–8%','6–10%','4–6%','3–5%','2–4%'],
        'Revenue Impact':      ['Very High','High','Very High','Medium','Medium','Low'],
        'Priority':            ['🔴 P1 NOW','🔴 P1 NOW','🔴 P1 NOW',
                                '🟡 P2 SOON','🟡 P2 SOON','🟢 P3 LATER'],
        'Effort':              ['Low','Medium','Low','Medium','Low','Low'],
        'Owner':               ['Marketing','Product','Data Science','CX Team','Sales','Tech']
    }), use_container_width=True, hide_index=True)