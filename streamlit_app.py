import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from src.config import (
    MLFLOW_TRACKING_URI, MODEL_NAME,
    MODEL_FILE_PATH, MODEL_METADATA_PATH,
)


# ── Load model (cached) ──────────────────────────────────────
@st.cache_resource
def load_model():
    # ── Try standalone .cbm file (Docker / Streamlit Cloud) ──
    if os.path.exists(MODEL_FILE_PATH):
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE_PATH)
        threshold = 0.5
        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH) as f:
                threshold = json.load(f).get("threshold", 0.5)
    else:
        # ── Fallback: MLflow Registry (local dev) ──
        import mlflow
        import mlflow.catboost

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                threshold = client.get_run(latest.run_id).data.metrics.get("threshold", 0.5)
            else:
                threshold = 0.5
        except Exception:
            threshold = 0.5
        model = mlflow.catboost.load_model(f"models:/{MODEL_NAME}/None")

    explainer = shap.TreeExplainer(model)
    return model, explainer, threshold


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 5px solid #22c55e;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 Stroke Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CatBoost · MLflow · SHAP Explainability</p>', unsafe_allow_html=True)

model, explainer, optimal_threshold = load_model()

# ── Sidebar — Patient Information ─────────────────────────────
with st.sidebar:
    st.header("👤 Patient Information")
    st.markdown("---")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 0.0, 120.0, 45.0, 0.5)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    glucose = st.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 106.0, 0.1)
    bmi = st.slider("BMI", 10.0, 60.0, 28.9, 0.1)
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────
if predict_btn:
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking,
    }

    df = pd.DataFrame([input_data])
    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = 1 if probability >= optimal_threshold else 0

    # ── Result section ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
                <h2 style="color:#dc2626; margin:0;">⚠️ HIGH RISK</h2>
                <p style="font-size:1.1rem; margin:0.5rem 0 0 0;">
                    Stroke probability: <strong>{probability:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2 style="color:#16a34a; margin:0;">✅ LOW RISK</h2>
                <p style="font-size:1.1rem; margin:0.5rem 0 0 0;">
                    Stroke probability: <strong>{probability:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Metrics row ──
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability", f"{probability:.2%}")
        m2.metric("Threshold", f"{optimal_threshold:.2%}")
        m3.metric("Decision", "STROKE" if prediction else "NO STROKE")

    with col2:
        st.subheader("🔬 SHAP Explanation")

        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Waterfall data
        contributions = {col: float(val) for col, val in zip(df.columns, shap_values[0])}
        sorted_contrib = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 4.5))
        features = list(sorted_contrib.keys())
        values = list(sorted_contrib.values())
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]
        bars = ax.barh(features[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("SHAP Value (impact on stroke risk)", fontsize=10)
        ax.set_title("Feature Contributions", fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Detailed SHAP table ──
    st.markdown("---")
    st.subheader("📋 Feature Impact Details")

    detail_df = pd.DataFrame({
        "Feature": features,
        "Value": [str(input_data.get(f, "")) for f in features],
        "SHAP Impact": [round(v, 4) for v in values],
        "Direction": ["↑ Increases Risk" if v > 0 else "↓ Decreases Risk" for v in values],
    })
    st.dataframe(detail_df, use_container_width=False, hide_index=True)

else:
    # ── Welcome state ──
    st.info("👈 Fill in the patient information on the sidebar and click **Predict Risk** to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "CatBoost")
    with col2:
        st.metric("ROC-AUC", "0.85")
    with col3:
        st.metric("Threshold", f"{optimal_threshold:.2%}")

    st.markdown("""
    ### How It Works
    1. **Enter patient data** in the sidebar
    2. **Click Predict** to see stroke risk assessment
    3. **Review SHAP values** to understand which factors contribute most
    """)
