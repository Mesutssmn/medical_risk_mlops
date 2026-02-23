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

    # ── Load Scaler ──
    import joblib
    scaler = None
    # We expect scaler in same dir as model logic or from config path. 
    # Since existing logic uses explicit paths for model, let's use explicit path for scaler too.
    from src.config import SCALER_PATH
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    explainer = shap.TreeExplainer(model)
    return model, explainer, threshold, scaler


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

model, explainer, optimal_threshold, scaler = load_model()

# ── Sidebar Inputs ──
with st.sidebar:
    st.header("Patient Data")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    age = st.slider("Age", 0.0, 120.0, 25.0)
    
    ses = st.selectbox("Socioeconomic Status (SES)", ["Low", "Medium", "High"])
    
    # Move Numeric inputs UP so we can use them for logic
    avg_glucose = st.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
    
    # Glucose logic
    if avg_glucose >= 126:
        st.caption("⚠️ **> 126 mg/dL indicates Diabetes**")
        glucose_locked = True
    else:
        glucose_locked = False

    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    
    # BMI Category Calculation
    if bmi < 18.5:
        bmi_cat = "Underweight"
        bmi_color = "#3b82f6" # Blue
    elif 18.5 <= bmi < 25:
        bmi_cat = "Healthy Weight"
        bmi_color = "#22c55e" # Green
    elif 25 <= bmi < 30:
        bmi_cat = "Overweight"
        bmi_color = "#eab308" # Yellow/Orange
    elif 30 <= bmi < 35:
        bmi_cat = "Obesity (Class 1)"
        bmi_color = "#f97316" # Orange
    elif 35 <= bmi < 40:
        bmi_cat = "Obesity (Class 2)"
        bmi_color = "#ea580c" # Dark Orange
    else:
        bmi_cat = "Obesity (Class 3 - Severe)"
        bmi_color = "#dc2626" # Red
        
    st.markdown(f"**Category:** <span style='color:{bmi_color}'>{bmi_cat}</span>", unsafe_allow_html=True)
    
    # BMI > 35 constraint (User Request)
    if bmi > 35:
        st.caption("⚠️ **BMI > 35 implies high risk (Diabetes locked to Yes)**")
        bmi_locked = True
    else:
        bmi_locked = False
        
    # Enforce constraints
    # If either condition is met, lock Diabetes to "Yes"
    if glucose_locked or bmi_locked:
        diabetes_index = 1 # Yes
        disabled_diabetes = True
    else:
        diabetes_index = 0 # Default No, but user can change
        disabled_diabetes = False

    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    hypertension_val = 1 if hypertension == "Yes" else 0
    
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    
    # Diabetes Input (Conditional)
    diabetes_status = st.selectbox(
        "Diabetes", 
        ["No", "Yes"], 
        index=diabetes_index,
        disabled=disabled_diabetes
    )
    diabetes_val = 1 if diabetes_status == "Yes" else 0
    
    smoking_status = st.selectbox(
        "Smoking Status", 
        ["Never", "Former", "Current"] 
    )

# ── Main Panel ──

# Create input dataframe matching NEW schema
# Columns: Age, Gender, SES, Hypertension, Heart_Disease, BMI, Avg_Glucose, Diabetes, Smoking_Status
input_data = {
    "Age": [age],
    "Gender": [gender],
    "SES": [ses],
    "Hypertension": [hypertension_val],
    "Heart_Disease": [heart_disease_val],
    "BMI": [bmi],
    "Avg_Glucose": [avg_glucose],
    "Diabetes": [diabetes_val],
    "Smoking_Status": [smoking_status]
}

df = pd.DataFrame(input_data)

# ── Display Input Summary ──
# with st.expander("Patient Profile", expanded=True):
#     st.dataframe(df)

if st.button("Predict Risk", type="primary"):
    # ── Preprocessing ──
    try:
        # 1. Feature Engineering
        from src.data.preprocess import create_features
        df_processed = create_features(df)
        
        # 2. Scaling
        if scaler:
            from src.config import SCALING_FEATURES
            cols_to_scale = [c for c in SCALING_FEATURES if c in df_processed.columns]
            try:
                df_processed[cols_to_scale] = scaler.transform(df_processed[cols_to_scale])
            except Exception as e:
                st.warning(f"Scaling warning: {e}")
                pass
        
        # 3. Feature Alignment
        if hasattr(model, "feature_names_"):
            for feature in model.feature_names_:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0
            df_processed = df_processed[model.feature_names_]
            
        probability = float(model.predict_proba(df_processed)[:, 1][0])
        prediction = 1 if probability >= optimal_threshold else 0
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Did you retrain the model with the new dataset?")
        st.stop()

    # ── Result section ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Prediction Result")
        
        low_risk_limit = optimal_threshold * 0.85
        
        if probability >= optimal_threshold:
            st.markdown(f"""
            <div class="risk-high">
                <h2 style="color:#dc2626; margin:0;">⚠️ HIGH RISK</h2>
                <p style="font-size:1.1rem; margin:0.5rem 0 0 0;">
                    Stroke probability: <strong>{probability:.1%}</strong>
                </p>
                <p style="font-size:0.9rem; color:#4b5563; margin-top:0.5rem;">
                    Exceeds threshold of {optimal_threshold:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif probability >= low_risk_limit:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); border-left: 5px solid #f97316; padding: 1.5rem; border-radius: 0.75rem; margin: 1rem 0;">
                <h2 style="color:#c2410c; margin:0;">⚠️ MODERATE RISK</h2>
                <p style="font-size:1.1rem; margin:0.5rem 0 0 0;">
                    Stroke probability: <strong>{probability:.1%}</strong>
                </p>
                <p style="font-size:0.9rem; color:#4b5563; margin-top:0.5rem;">
                    Near threshold ({optimal_threshold:.1%})
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
                <p style="font-size:0.9rem; color:#4b5563; margin-top:0.5rem;">
                    Well below threshold ({optimal_threshold:.1%}).
                </p>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Probability", f"{probability:.2%}")
        m2.metric("Threshold", f"{optimal_threshold:.2%}")
        m3.metric("Decision", "STROKE" if prediction else "NO STROKE")

    with col2:
        st.subheader("🔬 SHAP Explanation")

        try:
            # Fix: Pass PROCESSED dataframe to SHAP, not raw input
            shap_values = explainer.shap_values(df_processed)
            if isinstance(shap_values, list): # Multiclass check
                shap_values = shap_values[1] # Class 1 (Stroke)

            # Waterfall data
            # Fix: Zip with df_processed.columns
            contributions = {col: float(val) for col, val in zip(df_processed.columns, shap_values[0])}
            
            # Show top 10 features
            sorted_contrib = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

            # Bar chart
            fig, ax = plt.subplots(figsize=(7, 4.5))
            features = list(sorted_contrib.keys())
            values = list(sorted_contrib.values())
            colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]
            bars = ax.barh(features[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
            ax.set_xlabel("SHAP Value (impact on stroke risk)", fontsize=10)
            ax.set_title("Top Feature Contributions", fontsize=12, fontweight="bold")
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
                "Value": [str(df_processed.iloc[0].get(f, "")) for f in features], # Use processed values
                "SHAP Impact": [round(v, 4) for v in values],
                "Direction": ["↑ Increases Risk" if v > 0 else "↓ Decreases Risk" for v in values],
            })
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"SHAP Error: {e}")

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
