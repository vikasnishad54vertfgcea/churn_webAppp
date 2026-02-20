# ================================================================
#   ChurnIQ — Customer Intelligence Platform v3.0
#   Features:
#     - Auto Data Cleaning Pipeline
#     - Auto EDA Engine
#     - Single + Batch Prediction
#     - 4 ML Model Comparison (auto selects best)
#     - Revenue Impact Simulator
#     - PDF Report Generator
#     - Model Explainability
#     - Prediction History
#   Run: streamlit run app.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import io
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              roc_auc_score, roc_curve, classification_report)

from reportlab.lib.pagesizes import A4
from reportlab.lib           import colors
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units     import inch
from reportlab.platypus      import (SimpleDocTemplate, Paragraph, Spacer,
                                      Table, TableStyle, HRFlowable)
from reportlab.lib.enums     import TA_CENTER, TA_LEFT

# ── Constants ─────────────────────────────────────────────────
USD_TO_INR = 84

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #1e293b !important;
    padding: 16px 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    border-left: 5px solid #818cf8 !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #f1f5f9 !important;
    font-size: 28px !important;
    font-weight: 900 !important;
}
.risk-high   { background:linear-gradient(135deg,#ff416c,#ff4b2b);
               color:white;padding:22px;border-radius:14px;text-align:center;margin:8px 0; }
.risk-medium { background:linear-gradient(135deg,#f7971e,#ffd200);
               color:#111;padding:22px;border-radius:14px;text-align:center;margin:8px 0; }
.risk-low    { background:linear-gradient(135deg,#11998e,#38ef7d);
               color:white;padding:22px;border-radius:14px;text-align:center;margin:8px 0; }
.card-info   { background:#1e293b;border-left:4px solid #818cf8;
               padding:12px 16px;border-radius:10px;margin:4px 0;color:#e2e8f0; }
.card-good   { background:#0d2b1f;border-left:4px solid #2ecc71;
               padding:12px 16px;border-radius:10px;margin:4px 0;color:#6ee7b7; }
.card-warn   { background:#2d1f00;border-left:4px solid #f7971e;
               padding:12px 16px;border-radius:10px;margin:4px 0;color:#fde68a; }
.card-bad    { background:#2b0d0d;border-left:4px solid #e74c3c;
               padding:12px 16px;border-radius:10px;margin:4px 0;color:#fca5a5; }
.rev-box     { background:linear-gradient(135deg,#1a1a2e,#16213e);
               color:white;padding:22px;border-radius:16px;
               margin:10px 0;border:1px solid #818cf8; }
.step-ok     { background:#0d2b1f;border:1px solid #2ecc71;
               padding:9px 13px;border-radius:7px;margin:3px 0;color:#6ee7b7;font-size:14px; }
.step-action { background:#2d1f00;border:1px solid #f7971e;
               padding:9px 13px;border-radius:7px;margin:3px 0;color:#fde68a;font-size:14px; }
h1,h2,h3 { color:#f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_risk(prob):
    if prob >= 0.70:   return "🔴 HIGH RISK",   "risk-high",   "#ff416c"
    elif prob >= 0.30: return "🟡 MEDIUM RISK", "risk-medium", "#f7971e"
    else:              return "🟢 LOW RISK",    "risk-low",    "#2ecc71"


def explain_churn(inp):
    reasons, stays = [], []
    t  = inp.get('tenure', 0)
    m  = inp.get('MonthlyCharges', 0)
    c  = inp.get('Contract', '')
    i  = inp.get('InternetService', '')
    ts = inp.get('TechSupport', '')
    sr = inp.get('SeniorCitizen', '')
    pm = inp.get('PaymentMethod', '')
    sc = inp.get('OnlineSecurity', '')

    if c  == "Month-to-month":   reasons.append(("📋 Month-to-month contract",   "No commitment = easy to leave. Churn risk +43%"))
    if t  <  12:                 reasons.append(("⏳ New customer (< 12 months)", "First year = highest vulnerability window"))
    if m  >  6300:               reasons.append((f"💸 High charges ₹{m:,.0f}",    "Premium payer expects premium — gap = churn"))
    if i  == "Fiber optic":      reasons.append(("🌐 Fiber Optic service",         "Fiber users churn ~42% — unmet expectations"))
    if ts == "No":               reasons.append(("🛠 No Tech Support",             "No help available = frustration = churn"))
    if sc == "No":               reasons.append(("🔒 No Online Security",          "Lower perceived value of service"))
    if sr == "Yes":              reasons.append(("👴 Senior Citizen",              "Seniors churn 2× more — billing complexity"))
    if pm == "Electronic check": reasons.append(("💳 Electronic Check",            "Manual payers are less sticky than AutoPay"))

    if c in ["One year","Two year"]:  stays.append("✅ Long-term contract — strong loyalty signal")
    if t  >  24:                      stays.append("✅ 2+ year tenure — proven loyal customer")
    if ts == "Yes":                   stays.append("✅ Has Tech Support — reduces frustration churn")
    if m  <  3000:                    stays.append("✅ Low monthly charges — good value perception")
    if pm in ["Bank transfer (automatic)","Credit card (automatic)"]:
                                      stays.append("✅ AutoPay active — stickier billing relationship")
    return reasons, stays


def dark_fig():
    """Apply dark background to current matplotlib figure."""
    fig = plt.gcf()
    fig.patch.set_facecolor('#0f172a')
    for ax in fig.get_axes():
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')


# ══════════════════════════════════════════════════════════════
# AUTO CLEANING ENGINE
# ══════════════════════════════════════════════════════════════

def auto_clean(df_raw):
    df  = df_raw.copy()
    log = []
    r0, c0 = df.shape

    # 1. Drop pure ID columns (unique string per row)
    id_cols = [c for c in df.columns
               if df[c].dtype == 'object' and df[c].nunique() == len(df)]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        log.append(("action", f"🗑 Dropped ID columns: {id_cols}"))

    # 2. Fix numeric-looking string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            conv = pd.to_numeric(df[col], errors='coerce')
            if conv.notna().sum() > len(df) * 0.7:
                df[col] = conv
                log.append(("action", f"🔧 Converted '{col}' string → numeric"))

    # 3. Remove duplicates
    dups = df.duplicated().sum()
    if dups:
        df.drop_duplicates(inplace=True)
        log.append(("action", f"🗑 Removed {dups} duplicate rows"))
    else:
        log.append(("ok", "✅ No duplicate rows found"))

    # 4. Handle missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        log.append(("ok", "✅ No missing values found"))
    else:
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            if pct > 50:
                df.drop(columns=[col], inplace=True)
                log.append(("action", f"🗑 Dropped '{col}' — {pct:.0f}% missing (too many)"))
            elif df[col].dtype in ['float64', 'int64']:
                med = df[col].median()
                df[col].fillna(med, inplace=True)
                log.append(("action", f"📊 Filled '{col}' with median ({med:.2f}) — {cnt} rows"))
            else:
                mode = df[col].mode()[0]
                df[col].fillna(mode, inplace=True)
                log.append(("action", f"📝 Filled '{col}' with mode ('{mode}') — {cnt} rows"))

    # 5. Outlier capping (IQR method)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    out_fixed = []
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        if cnt > 0:
            df[col] = df[col].clip(lower, upper)
            out_fixed.append(f"{col}({cnt})")
    if out_fixed:
        log.append(("action", f"📐 Capped outliers in: {', '.join(out_fixed)}"))
    else:
        log.append(("ok", "✅ No significant outliers detected"))

    # 6. SeniorCitizen 0/1 → No/Yes
    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype in ['int64', 'float64']:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
            log.append(("action", "🔧 SeniorCitizen: 0→No, 1→Yes"))

    # 7. Convert charges from USD to INR if values are small
    for col in ['MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            if df[col].max() < 1000:
                df[col] = (df[col] * USD_TO_INR).round(0)
                log.append(("action", f"💱 '{col}' converted USD→INR (×{USD_TO_INR})"))

    r1, c1 = df.shape
    log.append(("ok", f"✅ Done — {r1:,} rows × {c1} cols  (removed {r0-r1} rows, {c0-c1} cols)"))
    return df, log


# ══════════════════════════════════════════════════════════════
# AUTO EDA ENGINE
# ══════════════════════════════════════════════════════════════

def run_eda(df):
    num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Outliers per column
    outliers = {}
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        cnt = int(((df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)).sum())
        if cnt > 0: outliers[col] = cnt

    # Churn column
    churn_col = next((c for c in df.columns if 'churn' in c.lower()), None)

    return {
        'shape':      df.shape,
        'num_cols':   num_cols,
        'cat_cols':   cat_cols,
        'missing':    df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        'dups':       int(df.duplicated().sum()),
        'outliers':   outliers,
        'num_stats':  df[num_cols].describe().round(2) if num_cols else None,
        'corr':       df[num_cols].corr().round(2) if len(num_cols) >= 2 else None,
        'churn_col':  churn_col,
        'churn_dist': df[churn_col].value_counts().to_dict() if churn_col else {},
    }


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_default_data():
    df = None
    for fname in ['Telco-Customer-Churn.csv', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']:
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
            except: continue
    if df is None:
        st.error("❌ Dataset not found! Place `Telco-Customer-Churn.csv` in app folder.")
        st.stop()
    df, _ = auto_clean(df)
    return df


# ══════════════════════════════════════════════════════════════
# MODEL TRAINING — 4 MODELS
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def train_all_models(df):
    df_m      = df.copy()
    churn_col = next((c for c in df_m.columns if 'churn' in c.lower()), None)
    if not churn_col:
        st.error("❌ No 'Churn' column in dataset!"); st.stop()

    df_m[churn_col] = LabelEncoder().fit_transform(df_m[churn_col].astype(str))
    cat_cols  = df_m.select_dtypes(include='object').columns.tolist()
    df_enc    = pd.get_dummies(df_m, columns=cat_cols, drop_first=True)

    X, y      = df_enc.drop(columns=[churn_col]), df_enc[churn_col]
    num_cols  = X.select_dtypes(include=['float64','int64']).columns.tolist()
    scaler    = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                       class_weight='balanced',
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100,
                                                            max_depth=5, random_state=42),
    }

    trained = {}
    for name, m in model_defs.items():
        m.fit(X_tr, y_tr)
        preds  = m.predict(X_te)
        proba  = m.predict_proba(X_te)[:, 1]
        report = classification_report(y_te, preds, output_dict=True)
        trained[name] = {
            "model":    m,
            "preds":    preds,
            "proba":    proba,
            "accuracy": accuracy_score(y_te, preds),
            "auc":      roc_auc_score(y_te, proba),
            "report":   report,
        }

    best = max(trained, key=lambda k: trained[k]['auc'])
    return trained, X_te, y_te, X.columns.tolist(), scaler, num_cols, best, churn_col


def predict_one(model, scaler, fcols, num_cols, inp):
    idf  = pd.DataFrame([inp])
    cats = idf.select_dtypes(include='object').columns.tolist()
    ienc = pd.get_dummies(idf, columns=cats, drop_first=True)
    for c in fcols:
        if c not in ienc.columns: ienc[c] = 0
    ienc = ienc[fcols]
    avail_num = [c for c in num_cols if c in ienc.columns]
    if avail_num:
        ienc[avail_num] = scaler.transform(ienc[avail_num])
    prob = model.predict_proba(ienc)[0][1]
    return int(prob >= 0.5), prob


def batch_predict(model, scaler, fcols, num_cols, df_in, churn_col):
    df = df_in.copy()
    if churn_col in df.columns:
        df.drop(columns=[churn_col], inplace=True)
    cats  = df.select_dtypes(include='object').columns.tolist()
    denc  = pd.get_dummies(df, columns=cats, drop_first=True)
    for c in fcols:
        if c not in denc.columns: denc[c] = 0
    denc  = denc[fcols]
    avail = [c for c in num_cols if c in denc.columns]
    if avail:
        denc[avail] = scaler.transform(denc[avail])
    probs = model.predict_proba(denc)[:, 1]
    out   = df_in.copy()
    out['Churn_Probability_%'] = (probs * 100).round(1)
    out['Churn_Prediction']    = np.where(probs >= 0.5, 'Yes', 'No')
    out['Risk_Level']          = pd.cut(probs, bins=[0,.30,.70,1.0],
                                         labels=['🟢 Low','🟡 Medium','🔴 High'])
    return out.sort_values('Churn_Probability_%', ascending=False), probs


# ══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════

def build_pdf(data: dict) -> bytes:
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                                topMargin=0.6*inch, bottomMargin=0.6*inch,
                                leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    S_title = ParagraphStyle('T', parent=styles['Title'],
                              fontSize=22, textColor=colors.HexColor('#4f46e5'),
                              spaceAfter=4, alignment=TA_CENTER)
    S_sub   = ParagraphStyle('S', parent=styles['Normal'],
                              fontSize=10, textColor=colors.HexColor('#64748b'),
                              spaceAfter=2, alignment=TA_CENTER)
    S_h2    = ParagraphStyle('H2', parent=styles['Heading2'],
                              fontSize=13, textColor=colors.HexColor('#1e293b'),
                              spaceBefore=10, spaceAfter=5)
    S_body  = ParagraphStyle('B', parent=styles['Normal'],
                              fontSize=10, textColor=colors.HexColor('#334155'),
                              spaceAfter=3, leading=14)
    S_kpi   = ParagraphStyle('K', parent=styles['Normal'],
                              fontSize=14, textColor=colors.HexColor('#4f46e5'),
                              fontName='Helvetica-Bold', alignment=TA_CENTER)
    S_small = ParagraphStyle('SM', parent=styles['Normal'],
                              fontSize=8, textColor=colors.HexColor('#64748b'),
                              alignment=TA_CENTER)

    def hr(): return HRFlowable(width="100%", thickness=1,
                                 color=colors.HexColor('#4f46e5'), spaceAfter=4)
    def section(t):
        story.append(Spacer(1, 8))
        story.append(hr())
        story.append(Paragraph(t, S_h2))

    # Header
    story.append(Paragraph("🧠 ChurnIQ", S_title))
    story.append(Paragraph("Customer Churn Intelligence Report", S_sub))
    story.append(Paragraph(
        f"Generated: {data['date']}  |  Analyst: {data['analyst']}  |  Model: {data['model']}",
        S_small))
    story.append(Spacer(1, 10))

    # KPI Table
    kpi_rows = [
        ['📊 Customers', '📉 Churn Rate', '🎯 Accuracy', '📈 ROC-AUC'],
        [Paragraph(f"{data['total_c']:,}", S_kpi),
         Paragraph(f"{data['churn_rt']:.1f}%", S_kpi),
         Paragraph(f"{data['accuracy']*100:.1f}%", S_kpi),
         Paragraph(f"{data['auc']:.3f}", S_kpi)],
    ]
    tbl = Table(kpi_rows, colWidths=[1.7*inch]*4)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,0), colors.HexColor('#4f46e5')),
        ('TEXTCOLOR',    (0,0),(-1,0), colors.white),
        ('FONTNAME',     (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0),(-1,0), 9),
        ('BACKGROUND',   (0,1),(-1,1), colors.HexColor('#f0f4ff')),
        ('ALIGN',        (0,0),(-1,-1),'CENTER'),
        ('VALIGN',       (0,0),(-1,-1),'MIDDLE'),
        ('GRID',         (0,0),(-1,-1), 0.5, colors.HexColor('#c7d2fe')),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    story.append(tbl)

    # Revenue Impact
    if data.get('show_rev'):
        section("💰 Revenue Impact (12-Month Projection)")
        rev_rows = [
            ['Metric', 'Value'],
            ['Total Revenue at Risk',  f"₹{data['at_risk']:,.0f}"],
            ['Customers Retained',     f"{data['retained']:,}"],
            ['Gross Revenue Saved',    f"₹{data['rev_saved']:,.0f}"],
            ['Campaign Cost',          f"₹{data['camp_cost']:,.0f}"],
            ['Net Revenue Saved',      f"₹{data['net_saved']:,.0f}"],
            ['Campaign ROI',           f"{data['roi']:.0f}%"],
        ]
        rt = Table(rev_rows, colWidths=[3.5*inch, 3*inch])
        rt.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
            ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0),(-1,-1), 10),
            ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#f8faff'), colors.HexColor('#eef2ff')]),
            ('ALIGN',      (1,0),(1,-1), 'RIGHT'),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ]))
        story.append(rt)

    # Model Comparison
    if data.get('metrics'):
        section("🤖 Model Performance Comparison")
        best_auc = max(r[2] for r in data['metrics'])
        mrows    = [['Model','Accuracy','ROC-AUC','Status']]
        for row in data['metrics']:
            mrows.append([row[0], f"{row[1]*100:.2f}%", f"{row[2]:.4f}",
                          "🏆 Best" if row[2] == best_auc else ""])
        mt = Table(mrows, colWidths=[2.5*inch,1.4*inch,1.4*inch,1.1*inch])
        mt.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#4f46e5')),
            ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
            ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0),(-1,-1), 10),
            ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#f8faff'), colors.HexColor('#eef2ff')]),
            ('ALIGN',      (1,0),(-1,-1),'CENTER'),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ]))
        story.append(mt)

    # Top Features
    if data.get('features'):
        section("🔍 Top Churn Drivers (Feature Importance)")
        for i,(fname,fimp) in enumerate(data['features'][:8],1):
            bar = "█" * int(fimp*200)
            story.append(Paragraph(
                f"<b>{i}. {fname}</b> — {fimp*100:.2f}%  "
                f"<font color='#4f46e5'>{bar}</font>", S_body))

    # Recommendations
    section("💡 Business Recommendations")
    recs = [
        "Offer 1-year contract with 20% discount to month-to-month users",
        "Review Fiber Optic pricing — high churn = value mismatch",
        "Deploy ML scoring monthly — call customers with 60%+ churn probability",
        "Launch Senior Care Plan — dedicated support + simplified billing",
        "Push TechSupport & OnlineSecurity add-ons — proven churn reducers",
        "AutoPay migration campaign — offer ₹100/month discount incentive",
        "New Customer Loyalty Program — cashback or rewards in first 12 months",
    ]
    for r in recs:
        story.append(Paragraph(f"• {r}", S_body))

    # Key Findings
    section("📌 Key Findings")
    findings = [
        "Month-to-month contract customers churn at ~43% — highest risk segment",
        "New customers (0–12 months tenure) are most vulnerable to churn",
        "Fiber Optic users churn at ~42% — premium pricing vs perceived value gap",
        "Senior citizens churn at nearly 2× the rate of non-seniors",
        "TechSupport and OnlineSecurity subscribers churn significantly less",
        "AutoPay customers are far stickier than Electronic Check payers",
        "Two-year contract customers have only ~3% churn — loyalty anchor",
    ]
    for f in findings:
        story.append(Paragraph(f"• {f}", S_body))

    # Footer
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#94a3b8')))
    story.append(Spacer(1,4))
    story.append(Paragraph(
        "ChurnIQ v3.0 | Built with Streamlit + Scikit-learn + ReportLab | Confidential",
        ParagraphStyle('F', parent=styles['Normal'],
                        fontSize=8, textColor=colors.HexColor('#94a3b8'),
                        alignment=TA_CENTER)))

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
for k,v in [('uploaded_df',None),('use_uploaded',False),
             ('uploaded_name',None),('clean_log',[]),
             ('pred_history',[]),('file_type','CSV')]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 ChurnIQ v3.0")
    st.markdown("*Customer Intelligence Platform*")
    st.markdown("---")

    page = st.radio("📍 Navigate", [
        "🏠 Dashboard",
        "🧹 Data Cleaning & EDA",
        "🔮 Predict Single Customer",
        "📦 Batch Prediction",
        "🤖 Multi-Model Comparison",
        "💰 Revenue Simulator",
        "📄 PDF Report Generator",
        "💡 Business Insights",
    ])

    st.markdown("---")
    st.markdown("### 📂 Upload Your Data")

    up_file = st.file_uploader(
        "CSV, Excel ya JSON — sab supported!",
        type=["csv", "xlsx", "xls", "json"],
        help="Supported: .csv  .xlsx  .xls  .json"
    )

    if up_file:
        try:
            fname = up_file.name.lower()

            # ── Read based on file type ────────────────────────
            if fname.endswith(".csv"):
                raw = pd.read_csv(up_file)
                ftype = "CSV"

            elif fname.endswith((".xlsx", ".xls")):
                # Show sheet selector if multiple sheets
                import openpyxl
                xl  = pd.ExcelFile(up_file)
                sheets = xl.sheet_names
                if len(sheets) > 1:
                    sel_sheet = st.selectbox("📋 Select Sheet:", sheets)
                else:
                    sel_sheet = sheets[0]
                raw   = pd.read_excel(up_file, sheet_name=sel_sheet)
                ftype = f"Excel (sheet: {sel_sheet})"

            elif fname.endswith(".json"):
                raw = pd.read_json(up_file)
                # If nested JSON (records format), normalize it
                if isinstance(raw.iloc[0,0], dict):
                    raw = pd.json_normalize(raw.to_dict(orient='records'))
                ftype = "JSON"

            else:
                st.error("❌ Unsupported format!")
                raw = None

            if raw is not None:
                cleaned, clog = auto_clean(raw)
                st.session_state.uploaded_df   = cleaned
                st.session_state.use_uploaded  = True
                st.session_state.uploaded_name = up_file.name
                st.session_state.clean_log     = clog
                st.session_state.file_type     = ftype
                st.success(f"✅ {up_file.name}")
                st.caption(f"📁 Type: {ftype}")
                st.caption(f"📊 {cleaned.shape[0]:,} rows × {cleaned.shape[1]} cols")

        except ImportError:
            st.error("❌ Excel support ke liye run karo: pip install openpyxl")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    if st.session_state.use_uploaded:
        if st.button("🔄 Use Default Telco Data"):
            st.session_state.use_uploaded  = False
            st.session_state.uploaded_df   = None
            st.session_state.uploaded_name = None
            st.rerun()
    st.markdown("---")


# ── Load Data ─────────────────────────────────────────────────
with st.spinner("🔄 Loading data & training 4 models... (~30s first time)"):
    if st.session_state.use_uploaded and st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
    else:
        df = load_default_data()

    (trained_models, X_test, y_test,
     fcols, scaler, num_cols, best_name, churn_col) = train_all_models(df)

best_model = trained_models[best_name]["model"]
best_acc   = trained_models[best_name]["accuracy"]
best_auc   = trained_models[best_name]["auc"]

churn_bool = df[churn_col].astype(str).str.lower().isin(['yes','1','true'])
churned    = churn_bool.sum()
total_c    = len(df)
churn_rt   = churned / total_c * 100

# Sidebar stats
with st.sidebar:
    src = st.session_state.uploaded_name or "Telco Default"
    st.caption(f"📊 **Source:** {src}")
    st.caption(f"👥 **Customers:** {total_c:,}")
    st.caption(f"📉 **Churn Rate:** {churn_rt:.1f}%")
    st.caption(f"🏆 **Best Model:** {best_name}")
    st.caption(f"🎯 **Accuracy:** {best_acc*100:.1f}%")
    st.caption(f"📈 **AUC:** {best_auc:.3f}")
    st.caption(f"💱 1 USD = ₹{USD_TO_INR}")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.title("🧠 ChurnIQ — Customer Intelligence Dashboard")
    st.markdown("##### Predict · Clean · Explain · Retain — Complete ML Platform v3.0")
    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("👥 Customers",    f"{total_c:,}")
    c2.metric("🚪 Churned",      f"{churned:,}",
              delta=f"-{churn_rt:.1f}%", delta_color="inverse")
    c3.metric("💚 Retained",     f"{total_c-churned:,}",
              delta=f"+{100-churn_rt:.1f}%")
    c4.metric("🏆 Best Accuracy",f"{best_acc*100:.1f}%", delta=best_name)
    c5.metric("📈 Best AUC",     f"{best_auc:.3f}",
              delta="Excellent 🏆" if best_auc > 0.80 else "Good")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🥧 Churn Split")
        fig, ax = plt.subplots(figsize=(4,3.5))
        counts  = pd.Series({'No Churn': total_c-churned, 'Churn': churned})
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
               colors=['#2ecc71','#e74c3c'], startangle=90,
               wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('Overall Churn Rate', fontweight='bold')
        dark_fig()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🤖 Model Leaderboard")
        for name, mdata in sorted(trained_models.items(),
                                    key=lambda x: x[1]['auc'], reverse=True):
            badge = "🏆 " if name == best_name else "   "
            bc    = "#818cf8" if name == best_name else "#64748b"
            st.markdown(
                f'<div class="card-info" style="border-color:{bc}">'
                f'{badge}<b>{name}</b><br>'
                f'Acc: <b>{mdata["accuracy"]*100:.1f}%</b> &nbsp;|&nbsp; '
                f'AUC: <b>{mdata["auc"]:.3f}</b></div>',
                unsafe_allow_html=True)

    with col3:
        st.subheader("📌 What This App Does")
        features = [
            ("🧹", "Auto Data Cleaning",    "Missing, outliers, types fixed"),
            ("📊", "Auto EDA",              "Stats, correlations, charts"),
            ("🔮", "Single Prediction",     "Risk + explainability + actions"),
            ("📦", "Batch Prediction",      "1000s at once → CSV download"),
            ("🤖", "4 ML Models",           "Auto-selects the best one"),
            ("📄", "PDF Report",            "1-click professional report"),
        ]
        for icon, name, desc in features:
            st.markdown(
                f'<div class="card-info"><b>{icon} {name}</b>'
                f'<br><small style="color:#94a3b8">{desc}</small></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — DATA CLEANING & EDA
# ══════════════════════════════════════════════════════════════
elif page == "🧹 Data Cleaning & EDA":

    st.title("🧹 Auto Data Cleaning & Exploratory Analysis")
    st.markdown("Upload any CSV → App automatically cleans + generates full EDA report")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🧹 Cleaning Report", "📊 EDA Insights", "📈 Visual EDA"])

    # ── CLEANING ─────────────────────────────────────────────
    with tab1:
        st.subheader("🧹 Auto Cleaning Pipeline")

        if st.session_state.use_uploaded and st.session_state.clean_log:
            clog = st.session_state.clean_log
        else:
            _, clog = auto_clean(load_default_data())

        st.markdown(f"**{len(clog)} steps performed automatically:**")
        for typ, msg in clog:
            css = "step-ok" if typ == "ok" else "step-action"
            st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 Data Sample (after cleaning):**")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.markdown("**📊 Summary Statistics:**")
            num_df = df.select_dtypes(include=['float64','int64'])
            st.dataframe(num_df.describe().round(1), use_container_width=True)

        st.markdown("---")
        st.download_button(
            "📥 Download Cleaned CSV",
            df.to_csv(index=False),
            f"cleaned_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv", use_container_width=True
        )

    # ── EDA INSIGHTS ─────────────────────────────────────────
    with tab2:
        eda = run_eda(df)
        st.subheader("📊 Dataset Overview")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("📏 Rows",         f"{eda['shape'][0]:,}")
        m2.metric("📐 Columns",      f"{eda['shape'][1]}")
        m3.metric("🔢 Numeric Cols", f"{len(eda['num_cols'])}")
        m4.metric("📝 Cat Cols",     f"{len(eda['cat_cols'])}")

        st.markdown("---")
        e1, e2 = st.columns(2)

        with e1:
            st.subheader("❓ Missing Values")
            if eda['missing']:
                mdf = pd.DataFrame(eda['missing'].items(),
                                    columns=['Column','Missing'])
                mdf['%'] = (mdf['Missing']/len(df)*100).round(1)
                st.dataframe(mdf, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="card-good">✅ No missing values!</div>',
                            unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🔁 Duplicates")
            if eda['dups'] > 0:
                st.markdown(f'<div class="card-warn">⚠️ {eda["dups"]} duplicate rows found</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="card-good">✅ No duplicates!</div>',
                            unsafe_allow_html=True)

        with e2:
            st.subheader("📐 Outliers (IQR Method)")
            if eda['outliers']:
                odf = pd.DataFrame(eda['outliers'].items(),
                                    columns=['Column','Outlier Count'])
                st.dataframe(odf, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="card-good">✅ No major outliers!</div>',
                            unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🎯 Target Column Distribution")
            if eda['churn_dist']:
                for k,v in eda['churn_dist'].items():
                    pct = v/len(df)*100
                    st.markdown(
                        f'<div class="card-info"><b>{k}</b>: {v:,} ({pct:.1f}%)</div>',
                        unsafe_allow_html=True)
            else:
                st.info("No 'Churn' column detected")

        if eda['num_stats'] is not None:
            st.markdown("---")
            st.subheader("📊 Numeric Statistics")
            st.dataframe(eda['num_stats'], use_container_width=True)

    # ── VISUAL EDA ───────────────────────────────────────────
    with tab3:
        eda = run_eda(df)
        st.subheader("📈 Interactive Visual EDA")

        v1, v2 = st.columns(2)
        with v1:
            if eda['num_cols']:
                sel_num = st.selectbox("📊 Distribution of numeric feature:", eda['num_cols'])
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.hist(df[~churn_bool][sel_num].dropna(), bins=25,
                        alpha=0.7, color='#2ecc71', label='No Churn', edgecolor='white')
                ax.hist(df[churn_bool][sel_num].dropna(), bins=25,
                        alpha=0.7, color='#e74c3c', label='Churn', edgecolor='white')
                ax.set_xlabel(sel_num); ax.set_ylabel('Count')
                ax.set_title(f'{sel_num} vs Churn', fontweight='bold')
                ax.legend()
                dark_fig()
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with v2:
            if eda['cat_cols']:
                sel_cat = st.selectbox("📋 Category vs Churn:", eda['cat_cols'][:12])
                fig, ax = plt.subplots(figsize=(5.5, 4))
                grp = df.groupby([sel_cat, churn_col]).size().unstack(fill_value=0)
                pct = grp.div(grp.sum(axis=1), axis=0) * 100
                pct.plot(kind='bar', ax=ax, color=['#2ecc71','#e74c3c'],
                         edgecolor='white', rot=20)
                ax.set_ylabel('%')
                ax.set_title(f'{sel_cat} vs Churn %', fontweight='bold')
                ax.legend(fontsize=8)
                dark_fig()
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # Correlation Heatmap
        if eda['corr'] is not None and len(eda['num_cols']) >= 2:
            st.subheader("🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 5))
            mask = np.triu(np.ones_like(eda['corr'], dtype=bool))
            sns.heatmap(eda['corr'], annot=True, fmt='.2f', cmap='coolwarm',
                        ax=ax, mask=mask, linewidths=0.5, vmin=-1, vmax=1,
                        annot_kws={'size': 8})
            ax.set_title('Feature Correlation Matrix', fontweight='bold')
            dark_fig()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Boxplots for numeric cols vs churn
        if eda['num_cols']:
            st.subheader("📦 Boxplots — Numeric Features vs Churn")
            box_col = st.selectbox("Select feature for boxplot:", eda['num_cols'], key='box')
            fig, ax = plt.subplots(figsize=(6, 4))
            data_no  = df[~churn_bool][box_col].dropna()
            data_yes = df[churn_bool][box_col].dropna()
            ax.boxplot([data_no, data_yes], labels=['No Churn','Churn'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#1e293b', color='white'),
                       medianprops=dict(color='#818cf8', linewidth=2),
                       whiskerprops=dict(color='white'),
                       capprops=dict(color='white'),
                       flierprops=dict(markerfacecolor='#f7971e', markersize=4))
            ax.set_ylabel(box_col)
            ax.set_title(f'{box_col} Distribution by Churn', fontweight='bold')
            dark_fig()
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Single Customer":

    st.title("🔮 Single Customer Churn Predictor")
    st.markdown("Fill profile → **Risk Level · Probability · Why · Cost Decision · Download**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender     = st.selectbox("Gender",          ["Male","Female"])
        senior     = st.selectbox("Senior Citizen",  ["No","Yes"])
        partner    = st.selectbox("Has Partner",     ["Yes","No"])
        dependents = st.selectbox("Has Dependents",  ["Yes","No"])

    with col2:
        st.subheader("📱 Services")
        phone_s    = st.selectbox("Phone Service",   ["Yes","No"])
        multi_l    = st.selectbox("Multiple Lines",  ["No","Yes","No phone service"])
        internet   = st.selectbox("Internet Service",["Fiber optic","DSL","No"])
        online_sec = st.selectbox("Online Security", ["No","Yes","No internet service"])
        online_bk  = st.selectbox("Online Backup",   ["No","Yes","No internet service"])
        dev_prot   = st.selectbox("Device Protection",["No","Yes","No internet service"])
        tech_sup   = st.selectbox("Tech Support",    ["No","Yes","No internet service"])
        stream_tv  = st.selectbox("Streaming TV",    ["No","Yes","No internet service"])
        stream_mv  = st.selectbox("Streaming Movies",["No","Yes","No internet service"])

    with col3:
        st.subheader("💳 Billing")
        tenure   = st.slider("Tenure (months)", 0, 72, 12)
        monthly  = st.number_input("Monthly Charges (₹)", 840, 10080, 5460, step=50)
        total_ch = st.number_input("Total Charges (₹)", 0, 756000, tenure*monthly, step=100)
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paperless= st.selectbox("Paperless Billing", ["Yes","No"])
        payment  = st.selectbox("Payment Method", [
                      "Electronic check","Mailed check",
                      "Bank transfer (automatic)","Credit card (automatic)"])
        model_sel= st.selectbox("🤖 Model", list(trained_models.keys()),
                                 index=list(trained_models.keys()).index(best_name))

    st.markdown("---")

    if st.button("🚀 Predict Now", use_container_width=True, type="primary"):

        inp = {"gender":gender,"SeniorCitizen":senior,"Partner":partner,
               "Dependents":dependents,"tenure":tenure,"PhoneService":phone_s,
               "MultipleLines":multi_l,"InternetService":internet,
               "OnlineSecurity":online_sec,"OnlineBackup":online_bk,
               "DeviceProtection":dev_prot,"TechSupport":tech_sup,
               "StreamingTV":stream_tv,"StreamingMovies":stream_mv,
               "Contract":contract,"PaperlessBilling":paperless,
               "PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total_ch}

        model_obj        = trained_models[model_sel]["model"]
        pred, prob       = predict_one(model_obj, scaler, fcols, num_cols, inp)
        risk_lbl,risk_cls,risk_col = get_risk(prob)
        reasons, stays   = explain_churn(inp)

        # Save history
        st.session_state.pred_history.append({
            "Time":     datetime.datetime.now().strftime("%H:%M %d-%b"),
            "Model":    model_sel,
            "Tenure":   tenure,
            "Monthly":  f"₹{monthly:,}",
            "Contract": contract,
            "Churn_%":  f"{prob*100:.1f}%",
            "Risk":     risk_lbl,
        })

        st.info(f"📋 Tenure: **{tenure}mo** | Monthly: **₹{monthly:,}** | "
                f"Contract: **{contract}** | Model: **{model_sel}**")

        # Result row
        r1, r2, r3 = st.columns([1.2, 1, 1])

        with r1:
            st.markdown(f"""
            <div class="{risk_cls}">
              <h2>{risk_lbl}</h2>
              <h1 style="font-size:54px;margin:8px 0">{prob*100:.1f}%</h1>
              <p style="font-size:15px">Churn Probability</p>
              <small>via {model_sel}</small>
            </div>""", unsafe_allow_html=True)

        with r2:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.barh(['Stay','Churn'], [1-prob, prob],
                    color=['#2ecc71', risk_col], edgecolor='white', height=0.45)
            for i, v in enumerate([1-prob, prob]):
                ax.text(v+0.01, i, f'{v*100:.1f}%', va='center', fontweight='bold', color='white')
            ax.set_xlim(0, 1.25)
            ax.set_title('Probability Split', fontweight='bold')
            dark_fig()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with r3:
            zone = 0 if prob < 0.30 else (1 if prob < 0.70 else 2)
            fig, ax = plt.subplots(figsize=(4.5, 3))
            clrs = ['#2ecc71','#f7971e','#e74c3c']
            brs  = ax.bar(['Low\n<30%','Medium\n30-70%','High\n>70%'],
                          [0.30, 0.40, 0.30], color=clrs, edgecolor='white', width=0.55)
            brs[zone].set_edgecolor('white')
            brs[zone].set_linewidth(3)
            ax.set_ylim(0, 0.55)
            ax.set_title('Risk Zone', fontweight='bold')
            dark_fig()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Explainability
        st.markdown("---")
        ex1, ex2 = st.columns(2)

        with ex1:
            st.subheader("🔍 Why Will This Customer Churn?")
            if reasons:
                for t,d in reasons:
                    st.markdown(
                        f'<div class="card-bad"><b>⚠️ {t}</b>'
                        f'<br><small>{d}</small></div>',
                        unsafe_allow_html=True)
            else:
                st.markdown('<div class="card-good">✅ No major churn signals!</div>',
                            unsafe_allow_html=True)

        with ex2:
            st.subheader("💪 Why Might They Stay?")
            if stays:
                for s in stays:
                    st.markdown(f'<div class="card-good">{s}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="card-warn">⚠️ No strong retention signals</div>',
                            unsafe_allow_html=True)

        # Cost Decision
        st.markdown("---")
        st.subheader("💡 Retention Cost-Benefit Decision")
        cs1, cs2 = st.columns(2)

        with cs1:
            disc   = st.number_input("Retention Discount Offer (₹/month)", 100, 3000, 300, step=50)
            ret_rt = st.slider("Expected Retention Success (%)", 10, 90, 40)

        with cs2:
            net   = (monthly - disc) * (ret_rt / 100)
            loss  = monthly * 12
            prof  = net > 0
            cls   = "card-good" if prof else "card-bad"
            icon  = "✅ PROFITABLE — Make the offer!" if prof else "❌ NOT PROFITABLE at this discount"
            st.markdown(f"""
            <div class="{cls}">
              <b style="font-size:15px">{icon}</b><br><br>
              Monthly Revenue &nbsp;&nbsp;: <b>₹{monthly:,}</b><br>
              Discount Offered &nbsp;&nbsp;: <b>₹{disc:,}</b><br>
              Net After Discount : <b>₹{monthly-disc:,}</b><br>
              Success Rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>{ret_rt}%</b><br>
              <b>Expected Net Gain  : ₹{net:,.0f}/month</b><br>
              Annual Loss if Churns: ₹{loss:,.0f}
            </div>""", unsafe_allow_html=True)

        # Recommendations
        st.markdown("---")
        st.subheader("🎯 Recommended Retention Actions")
        recs = []
        if contract == "Month-to-month": recs.append("📋 **1-Year Contract** — 20% discount ke saath offer karo")
        if monthly  > 6300:             recs.append("💰 **Bundle Plan** — Internet+TV+Phone at 25% off")
        if tech_sup == "No":            recs.append("🛠️ **Free TechSupport** 3-month trial do")
        if tenure   < 12:               recs.append("🎁 **Loyalty Program** — cashback ya rewards enroll karo")
        if senior   == "Yes":           recs.append("👴 **Senior Care Plan** + dedicated agent assign karo")
        if payment  == "Electronic check": recs.append("💳 **AutoPay Setup** — ₹100/month discount incentive do")
        if online_sec == "No":          recs.append("🔒 **Online Security** free trial offer karo")
        if not recs:                    recs.append("📞 **Proactive satisfaction call** + loyalty reward do")
        for r in recs:
            st.warning(r)

        # Download
        st.markdown("---")
        pred_csv = pd.DataFrame([{
            "Date":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Model":        model_sel,
            "Churn_%":      round(prob*100, 1),
            "Risk":         risk_lbl,
            "Tenure":       tenure,
            "Monthly_INR":  monthly,
            "Contract":     contract,
            "Internet":     internet,
            "Senior":       senior,
            "TechSupport":  tech_sup,
            "Payment":      payment,
            "Discount_INR": disc,
            "Retention_%":  ret_rt,
            "Net_Gain_INR": round(net, 0),
            "Profitable":   "Yes" if prof else "No",
        }]).to_csv(index=False)

        st.download_button("📥 Download Prediction CSV", pred_csv,
            f"prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "📦 Batch Prediction":

    st.title("📦 Batch Prediction — Predict 1000s at Once")
    st.markdown("Upload CSV → All customers predicted → Download results by risk level")
    st.markdown("---")

    st.info("📋 **Steps:** Upload customer CSV → Select model → Click Predict → Download")

    b1, b2 = st.columns([2, 1])
    with b1:
        batch_file = st.file_uploader(
            "📂 Upload Customer Data (CSV / Excel / JSON)",
            type=["csv","xlsx","xls","json"],
            key="batch"
        )
    with b2:
        batch_model_name = st.selectbox("🤖 Model", list(trained_models.keys()),
                                         index=list(trained_models.keys()).index(best_name))
        threshold = st.slider("High Risk Threshold (%)", 30, 80, 50)

    if batch_file:
        try:
            bfname = batch_file.name.lower()
            if bfname.endswith(".csv"):
                batch_raw = pd.read_csv(batch_file)
            elif bfname.endswith((".xlsx",".xls")):
                batch_raw = pd.read_excel(batch_file)
            elif bfname.endswith(".json"):
                batch_raw = pd.read_json(batch_file)
            else:
                st.error("❌ Unsupported format"); batch_raw = None

            if batch_raw is None: st.stop()
            st.success(f"✅ Loaded: **{batch_file.name}** — {batch_raw.shape[0]:,} rows")

            with st.expander("👀 Preview uploaded data"):
                st.dataframe(batch_raw.head(5), use_container_width=True)

            if st.button("🚀 Predict All Customers", type="primary", use_container_width=True):
                with st.spinner(f"🔄 Predicting {batch_raw.shape[0]:,} customers..."):
                    batch_clean, _ = auto_clean(batch_raw)
                    bmodel         = trained_models[batch_model_name]["model"]
                    results_df, probs = batch_predict(
                        bmodel, scaler, fcols, num_cols, batch_clean, churn_col)

                st.markdown("---")
                high  = (probs >= threshold/100).sum()
                med   = ((probs >= 0.30) & (probs < threshold/100)).sum()
                low   = (probs < 0.30).sum()

                s1,s2,s3,s4 = st.columns(4)
                s1.metric("📋 Total",     f"{len(probs):,}")
                s2.metric("🔴 High Risk", f"{high:,}",
                          delta=f"{high/len(probs)*100:.1f}%", delta_color="inverse")
                s3.metric("🟡 Medium",    f"{med:,}")
                s4.metric("🟢 Low Risk",  f"{low:,}")

                bc1, bc2 = st.columns(2)
                with bc1:
                    fig, ax = plt.subplots(figsize=(5,4))
                    ax.pie([high, med, low],
                           labels=[f'High({high})',f'Medium({med})',f'Low({low})'],
                           autopct='%1.0f%%',
                           colors=['#e74c3c','#f7971e','#2ecc71'],
                           startangle=90,
                           wedgeprops={'edgecolor':'white','linewidth':2})
                    ax.set_title('Risk Distribution', fontweight='bold')
                    dark_fig()
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                with bc2:
                    fig, ax = plt.subplots(figsize=(5,4))
                    ax.hist(probs*100, bins=30, color='#818cf8',
                            edgecolor='white', alpha=0.85)
                    ax.axvline(threshold, color='#e74c3c', lw=2,
                               linestyle='--', label=f'Threshold {threshold}%')
                    ax.set_xlabel('Churn Probability (%)')
                    ax.set_ylabel('Count')
                    ax.set_title('Probability Distribution', fontweight='bold')
                    ax.legend()
                    dark_fig()
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                st.subheader(f"🔴 Top High-Risk Customers (>{threshold}%)")
                high_df = results_df[results_df['Churn_Probability_%'] >= threshold]
                st.dataframe(high_df.head(50), use_container_width=True, hide_index=True)

                st.markdown("---")
                d1, d2, d3 = st.columns(3)
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
                with d1:
                    st.download_button("📥 All Results",
                        results_df.to_csv(index=False),
                        f"batch_all_{ts}.csv", "text/csv",
                        use_container_width=True)
                with d2:
                    st.download_button("🔴 High Risk Only",
                        high_df.to_csv(index=False),
                        f"batch_highrisk_{ts}.csv", "text/csv",
                        use_container_width=True)
                with d3:
                    action_df = high_df.copy()
                    action_df['Recommended_Action'] = 'Proactive call + retention offer'
                    action_df['Priority']           = 'P1 — Immediate'
                    st.download_button("📋 Action Plan",
                        action_df.to_csv(index=False),
                        f"action_plan_{ts}.csv", "text/csv",
                        use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    else:
        st.markdown("---")
        st.markdown("#### 💡 No file? Demo batch on current loaded dataset:")
        if st.button("▶️ Run Demo Batch (500 customers)", use_container_width=True):
            with st.spinner("🔄 Running batch..."):
                bmodel = trained_models[best_name]["model"]
                demo_r, demo_p = batch_predict(
                    bmodel, scaler, fcols, num_cols, df.head(500), churn_col)

            st.success(f"✅ Predicted {len(demo_r)} customers!")
            st.dataframe(demo_r.head(20), use_container_width=True, hide_index=True)
            st.download_button("📥 Download Demo Results",
                demo_r.to_csv(index=False),
                "demo_batch.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — MULTI-MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Multi-Model Comparison":

    st.title("🤖 Multi-Model Performance Comparison")
    st.markdown("4 models trained simultaneously — automatic best model selection")
    st.markdown("---")

    # Leaderboard table
    rows = []
    for name, mdata in trained_models.items():
        r = mdata['report']
        rows.append({
            "Model":            name,
            "Accuracy":         f"{mdata['accuracy']*100:.2f}%",
            "ROC-AUC":          f"{mdata['auc']:.4f}",
            "Precision(Churn)": f"{r['1']['precision']:.3f}",
            "Recall(Churn)":    f"{r['1']['recall']:.3f}",
            "F1(Churn)":        f"{r['1']['f1-score']:.3f}",
            "Status":           "🏆 BEST" if name == best_name else "",
        })
    st.subheader("📊 Model Leaderboard")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📉 ROC Curves")
        fig, ax = plt.subplots(figsize=(6,5))
        clrs = ['#818cf8','#2ecc71','#e74c3c','#f7971e']
        for (name, mdata), clr in zip(trained_models.items(), clrs):
            fpr, tpr, _ = roc_curve(y_test, mdata['proba'])
            ax.plot(fpr, tpr, label=f"{name} ({mdata['auc']:.3f})", color=clr, lw=2)
        ax.plot([0,1],[0,1],'k--',lw=1,label='Random (0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — All Models', fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        dark_fig()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.subheader("📊 Accuracy vs AUC")
        fig, ax = plt.subplots(figsize=(6,5))
        names = list(trained_models.keys())
        accs  = [trained_models[n]['accuracy']*100 for n in names]
        aucs  = [trained_models[n]['auc']*100       for n in names]
        x     = np.arange(len(names))
        w     = 0.35
        ax.bar(x-w/2, accs, w, label='Accuracy %', color='#818cf8', edgecolor='white')
        ax.bar(x+w/2, aucs, w, label='AUC × 100',  color='#2ecc71', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylim(60,100)
        ax.set_ylabel('Score')
        ax.set_title('Model Score Comparison', fontweight='bold')
        ax.legend()
        dark_fig()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Confusion Matrices
    st.markdown("---")
    st.subheader("🔢 Confusion Matrices — All Models")
    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    for ax, (name, mdata) in zip(axes, trained_models.items()):
        cm = confusion_matrix(y_test, mdata['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No','Yes'], yticklabels=['No','Yes'],
                    linewidths=0.5)
        title = f"{name}" + ("\n🏆 BEST" if name == best_name else "")
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature Importance
    st.markdown("---")
    st.subheader(f"🔍 Feature Importance — {best_name}")
    imp_model = trained_models[best_name]["model"]
    if hasattr(imp_model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': fcols,
                            'Importance': imp_model.feature_importances_}
                           ).sort_values('Importance', ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10,6))
        clrs = sns.color_palette('RdYlGn_r', n_colors=15)
        ax.barh(fi['Feature'][::-1], fi['Importance'][::-1],
                color=clrs, edgecolor='white')
        for i, v in enumerate(fi['Importance'][::-1]):
            ax.text(v+0.001, i, f'{v:.4f}', va='center', fontsize=8, color='white')
        ax.set_title('Feature Importances', fontweight='bold')
        ax.set_xlabel('Importance Score')
        dark_fig()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Prediction History
    if st.session_state.pred_history:
        st.markdown("---")
        st.subheader("🕐 Prediction History (This Session)")
        hdf = pd.DataFrame(st.session_state.pred_history)
        st.dataframe(hdf, use_container_width=True, hide_index=True)
        st.download_button("📥 Download History", hdf.to_csv(index=False),
                            "prediction_history.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# PAGE 6 — REVENUE SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "💰 Revenue Simulator":

    st.title("💰 Revenue Impact Simulator")
    st.markdown("Quantify churn cost — Prove ROI of retention campaigns to management")
    st.markdown("---")

    s1, s2, s3 = st.columns(3)
    with s1:
        avg_rev   = st.number_input("Avg Monthly Revenue / Customer (₹)", 500, 20000, 5460, step=100)
        n_risk    = st.number_input("High-Risk Customers Identified", 10, 500000, 500, step=10)
    with s2:
        ret_pct   = st.slider("Expected Retention Rate (%)", 5, 80, 30,
                               help="% of high-risk customers you can retain")
        camp_cost = st.number_input("Campaign Cost / Customer (₹)", 0, 5000, 300, step=50)
    with s3:
        months    = st.slider("Projection Period (months)", 1, 24, 12)

    at_risk    = avg_rev * n_risk * months
    retained_n = int(n_risk * ret_pct / 100)
    lost_n     = n_risk - retained_n
    rev_loss   = avg_rev * lost_n * months
    rev_saved  = avg_rev * retained_n * months
    camp_total = camp_cost * n_risk
    net_saved  = rev_saved - camp_total
    roi        = (net_saved / max(camp_total, 1)) * 100

    st.markdown("---")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("💸 Revenue at Risk", f"₹{at_risk:,.0f}")
    m2.metric("🔴 Revenue Lost",    f"₹{rev_loss:,.0f}",
              delta=f"{lost_n} customers", delta_color="inverse")
    m3.metric("🟢 Net Revenue Saved",f"₹{net_saved:,.0f}",
              delta=f"{retained_n} retained")
    m4.metric("🚀 Campaign ROI",    f"{roi:.0f}%")

    st.markdown(f"""
    <div class="rev-box">
      <h2 style="color:#a78bfa">📈 {months}-Month Revenue Simulation</h2>
      <table width="100%" style="color:white;font-size:15px;border-collapse:collapse">
        <tr><td style="padding:7px 0">👥 High-Risk Customers</td>
            <td style="text-align:right;font-weight:700">{n_risk:,}</td></tr>
        <tr style="border-top:1px solid #334155">
          <td style="padding:7px 0">💸 Total Revenue at Risk</td>
          <td style="text-align:right;color:#ff6b6b;font-weight:700">₹{at_risk:,.0f}</td></tr>
        <tr style="border-top:1px solid #334155">
          <td style="padding:7px 0">✅ Customers Retained</td>
          <td style="text-align:right;color:#38ef7d;font-weight:700">{retained_n:,}</td></tr>
        <tr style="border-top:1px solid #334155">
          <td style="padding:7px 0">💰 Gross Revenue Saved</td>
          <td style="text-align:right;color:#38ef7d;font-weight:700">₹{rev_saved:,.0f}</td></tr>
        <tr style="border-top:1px solid #334155">
          <td style="padding:7px 0">📦 Campaign Cost</td>
          <td style="text-align:right;color:#ffd200;font-weight:700">₹{camp_total:,.0f}</td></tr>
        <tr style="border-top:2px solid #818cf8">
          <td style="padding:10px 0;font-size:18px;font-weight:800">🏆 Net Revenue Saved</td>
          <td style="text-align:right;color:#38ef7d;font-size:24px;font-weight:800">₹{net_saved:,.0f}</td></tr>
        <tr><td>📊 Campaign ROI</td>
            <td style="text-align:right;color:#a78bfa;font-weight:700">{roi:.0f}%</td></tr>
      </table>
    </div>""", unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(6,4))
        cats  = ['At Risk','Rev Lost','Rev Saved','Camp Cost','Net Saved']
        vals  = [at_risk, rev_loss, rev_saved, camp_total, net_saved]
        bclrs = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
        bars  = ax.bar(cats, [v/100000 for v in vals],
                       color=bclrs, edgecolor='white', width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f'₹{val/100000:.1f}L', ha='center', fontsize=9,
                    fontweight='bold', color='white')
        ax.set_ylabel('₹ Lakhs')
        ax.set_title('Revenue Breakdown', fontweight='bold')
        dark_fig()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with v2:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.pie([retained_n, lost_n],
               labels=[f'Retained\n{retained_n:,}', f'Lost\n{lost_n:,}'],
               autopct='%1.0f%%', colors=['#2ecc71','#e74c3c'],
               startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('Customer Outcome', fontweight='bold')
        dark_fig()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    sim_csv = pd.DataFrame([{
        'Avg_Revenue_INR': avg_rev, 'High_Risk_Count': n_risk,
        'Retention_%': ret_pct, 'Camp_Cost_INR': camp_cost,
        'Months': months, 'At_Risk_INR': at_risk,
        'Revenue_Lost_INR': rev_loss, 'Retained_Count': retained_n,
        'Net_Saved_INR': net_saved, 'ROI_%': round(roi,1)
    }]).to_csv(index=False)
    st.download_button("📥 Download Simulation CSV", sim_csv,
        f"simulation_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# PAGE 7 — PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════
elif page == "📄 PDF Report Generator":

    st.title("📄 Professional PDF Report Generator")
    st.markdown("One click → Executive-ready PDF report — share with management directly")
    st.markdown("---")

    st.subheader("⚙️ Configure Report")
    r1, r2 = st.columns(2)
    with r1:
        report_title  = st.text_input("Report Title", "ChurnIQ — Customer Churn Analysis Report")
        analyst_name  = st.text_input("Analyst Name", "Data Science Team")
        show_rev      = st.checkbox("Include Revenue Impact Section", value=True)
    with r2:
        avg_rev_pdf   = st.number_input("Avg Monthly Revenue (₹)", 500, 20000, 5460, step=100)
        n_risk_pdf    = st.number_input("High-Risk Customers", 10, 100000, 500, step=10)
        ret_pct_pdf   = st.slider("Expected Retention Rate (%)", 5, 80, 30)
        camp_cost_pdf = st.number_input("Campaign Cost / Customer (₹)", 0, 5000, 300, step=50)

    # Calcs for PDF
    months_pdf    = 12
    retained_pdf  = int(n_risk_pdf * ret_pct_pdf/100)
    rev_saved_pdf = avg_rev_pdf * retained_pdf * months_pdf
    camp_tot_pdf  = camp_cost_pdf * n_risk_pdf
    net_saved_pdf = rev_saved_pdf - camp_tot_pdf
    roi_pdf       = (net_saved_pdf / max(camp_tot_pdf, 1)) * 100

    # Feature importances
    imp_model = trained_models[best_name]["model"]
    top_feats = []
    if hasattr(imp_model, 'feature_importances_'):
        top_feats = sorted(zip(fcols, imp_model.feature_importances_),
                           key=lambda x: x[1], reverse=True)[:8]

    st.markdown("---")
    st.subheader("📋 Report Preview")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""<div class="card-info"><b>📋 Sections Included</b><br>
        • Executive KPI Summary<br>
        • Revenue Impact (12-month)<br>
        • 4-Model Comparison Table<br>
        • Top 8 Churn Drivers<br>
        • Business Recommendations<br>
        • Key Findings & Takeaways</div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class="card-good"><b>📊 Key Numbers</b><br>
        Customers: <b>{total_c:,}</b><br>
        Churn Rate: <b>{churn_rt:.1f}%</b><br>
        Best Model: <b>{best_name}</b><br>
        Accuracy: <b>{best_acc*100:.1f}%</b><br>
        AUC: <b>{best_auc:.3f}</b><br>
        Net Saved: <b>₹{net_saved_pdf:,.0f}</b></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""<div class="card-info"><b>📝 Details</b><br>
        Format: PDF (A4 Portrait)<br>
        Analyst: {analyst_name}<br>
        Date: {datetime.datetime.now().strftime("%d %b %Y")}<br>
        Dataset: {total_c:,} customers<br>
        Models Compared: {len(trained_models)}<br>
        Language: English</div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("📄 Generate PDF Now", type="primary", use_container_width=True):
        with st.spinner("🔄 Building professional PDF..."):
            pdf_data = {
                'title':     report_title,
                'date':      datetime.datetime.now().strftime("%d %b %Y, %I:%M %p"),
                'model':     best_name,
                'analyst':   analyst_name,
                'total_c':   total_c,
                'churn_rt':  churn_rt,
                'accuracy':  best_acc,
                'auc':       best_auc,
                'show_rev':  show_rev,
                'at_risk':   avg_rev_pdf * n_risk_pdf * months_pdf,
                'retained':  retained_pdf,
                'rev_saved': rev_saved_pdf,
                'camp_cost': camp_tot_pdf,
                'net_saved': net_saved_pdf,
                'roi':       roi_pdf,
                'features':  top_feats,
                'metrics':   [(n, d['accuracy'], d['auc'])
                               for n,d in trained_models.items()],
            }
            try:
                pdf_bytes = build_pdf(pdf_data)
                st.success("✅ PDF Generated Successfully!")
                st.download_button(
                    "📥 Download PDF Report",
                    data      = pdf_bytes,
                    file_name = f"ChurnIQ_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime      = "application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ PDF Error: {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════
# PAGE 8 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":

    st.title("💡 Business Insights & Strategy")
    st.markdown("---")

    r1, r2 = st.columns(2)
    with r1:
        for t in ["🔴 **Month-to-Month** customers — 43% churn",
                   "🔴 **New customers (0–12mo)** — highest risk window",
                   "🔴 **Fiber Optic users** — 42% churn rate",
                   "🔴 **Senior Citizens** — churn 2× average"]:
            st.error(t)
    with r2:
        for t in ["🟢 **Two-year contracts** — only 3% churn",
                   "🟢 **Tenure > 24 months** — proven loyalty",
                   "🟢 **TechSupport users** — churn significantly less",
                   "🟢 **AutoPay customers** — far stickier"]:
            st.success(t)

    st.markdown("---")
    tab1,tab2,tab3,tab4,tab5 = st.tabs(
        ["📋 Contracts","💰 Pricing","🎯 Retention","👴 Seniors","💡 Why It Matters"])

    with tab1:
        st.markdown("""### 📋 Contract Strategy
- **15–20% discount** for monthly → annual upgrade
- **Month 10 nudge** — auto renewal offer before decision
- Month-to-month > 6 months → **priority outreach list**
- **"First-Year Bonus"** — free month or data upgrade""")

    with tab2:
        st.markdown("""### 💰 Pricing Strategy
- **Fiber Optic pricing review** — high churn = value gap
- **Price Lock Guarantee** — 2-year plan pe fixed rates
- **Bundle 25% off** — Internet + TV + Phone combo
- Monthly > ₹6,700 → **automatic retention offer trigger**""")

    with tab3:
        st.markdown("""### 🎯 Proactive Retention
- **Monthly ML scoring** — flag risk before they cancel
- **60%+ probability** → dedicated retention team call
- **TechSupport + Security** push — both reduce churn
- **AutoPay migration** — ₹100/month cashback incentive
- High-value + high-risk → **Customer Success Manager**""")

    with tab4:
        st.markdown("""### 👴 Senior Citizen Program
- **"Senior Care Plan"** — simple billing, priority support
- **Dedicated hotline** — no IVR, direct human agent
- **Monthly welfare call** in first year
- **Family bundle** — senior + family member discount""")

    with tab5:
        saved_c  = int(total_c * 0.065)
        avg_m    = df['MonthlyCharges'].mean() if 'MonthlyCharges' in df.columns else 5460
        annual_s = saved_c * avg_m * 12
        acq_s    = saved_c * 5000
        st.markdown(f"""### 💡 Real Financial Impact
**If churn drops 6.5% (26.5% → 20%):**

| Metric | Value |
|--------|-------|
| Extra customers saved | **{saved_c:,}** |
| Avg monthly revenue | **₹{avg_m:,.0f}** |
| Annual revenue recovered | **₹{annual_s:,.0f}** |
| Acquisition cost saved (@ ₹5k) | **₹{acq_s:,.0f}** |
| **TOTAL ANNUAL GAIN** | **₹{annual_s+acq_s:,.0f}** |
""")
        st.success(f"🏆 Total: **₹{annual_s+acq_s:,.0f}/year** from just 6.5% churn reduction!")

    st.markdown("---")
    st.subheader("📊 Action Priority Matrix")
    st.dataframe(pd.DataFrame({
        'Strategy':        ['Contract Upgrade','Fiber Pricing','ML Outreach',
                            'Senior Plan','Tech Support Push','AutoPay Migration'],
        'Churn Reduction': ['8–12%','5–8%','6–10%','4–6%','3–5%','2–4%'],
        'Revenue Impact':  ['Very High','High','Very High','Medium','Medium','Low'],
        'Priority':        ['🔴 P1 NOW','🔴 P1 NOW','🔴 P1 NOW',
                            '🟡 P2 SOON','🟡 P2 SOON','🟢 P3 LATER'],
        'Effort':          ['Low','Medium','Low','Medium','Low','Low'],
        'Owner':           ['Marketing','Product','Data Science','CX Team','Sales','Tech'],
    }), use_container_width=True, hide_index=True)