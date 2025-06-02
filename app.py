import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import joblib
import base64
import io

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(layout="wide")
st.title("🧠 CKD Risk Predictor with SHAP & LIME")
st.markdown("This app predicts Chronic Kidney Disease (CKD) risk and explains the prediction using SHAP and LIME.")

# Input form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        bp = st.number_input("Blood Pressure", 50, 200, 80)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
        al = st.slider("Albumin", 0, 5, 1)
        su = st.slider("Sugar", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"], index=0)
    with col2:
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"], index=0)
        pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"], index=1)
        ba = st.selectbox("Bacteria", ["present", "notpresent"], index=1)
        bu = st.number_input("Blood Urea", 1.0, 400.0, 50.0)
        sc = st.number_input("Serum Creatinine", 0.1, 50.0, 1.5)
        hemo = st.number_input("Hemoglobin", 3.0, 17.0, 12.0)
    with col3:
        htn = st.selectbox("Hypertension", ["yes", "no"], index=1)
        dm = st.selectbox("Diabetes Mellitus", ["yes", "no"], index=1)
        appet = st.selectbox("Appetite", ["good", "poor"], index=0)
        ane = st.selectbox("Anemia", ["yes", "no"], index=1)
        submit = st.form_submit_button("Predict")

# Map categorical features
mapper = {
    "normal": 0,
    "abnormal": 1,
    "present": 1,
    "notpresent": 0,
    "yes": 1,
    "no": 0,
    "good": 0,
    "poor": 1
}

if submit:
    X_input = pd.DataFrame({
        "age": [age],
        "bp": [bp],
        "sg": [sg],
        "al": [al],
        "su": [su],
        "rbc": [mapper[rbc]],
        "pc": [mapper[pc]],
        "pcc": [mapper[pcc]],
        "ba": [mapper[ba]],
        "bu": [bu],
        "sc": [sc],
        "hemo": [hemo],
        "htn": [mapper[htn]],
        "dm": [mapper[dm]],
        "appet": [mapper[appet]],
        "ane": [mapper[ane]]
    })

    # Add derived features
    X_input["high_creatinine"] = (X_input["sc"] > 1.2).astype(int)
    X_input["bun_sc_ratio"] = X_input["bu"] / X_input["sc"]
    X_input["hemo_bu"] = X_input["hemo"] / (X_input["bu"] + 1)

    # Reorder columns
    final_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bu', 'sc', 'hemo',
                      'htn', 'dm', 'appet', 'ane', 'high_creatinine', 'bun_sc_ratio', 'hemo_bu']

    X_scaled = scaler.transform(X_input[final_features])
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("🔍 Prediction Result")
    st.success(f"CKD Risk: {'Positive' if prediction == 1 else 'Negative'}")
    st.info(f"Probability of CKD: {prob * 100:.2f}%")

    # SHAP Explanation
    st.subheader("📈 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 2))
    shap.force_plot(explainer.expected_value[1], shap_values[1], X_input[final_features], matplotlib=True, show=False)
    st.pyplot(fig)

    st.subheader("📊 SHAP Summary Plot")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_input[final_features], plot_type="bar", show=False)
    st.pyplot(fig2)

    # LIME Explanation
    st.subheader("🟢 LIME Explanation")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_scaled),
        feature_names=final_features,
        mode='classification'
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=X_scaled[0],
        predict_fn=model.predict_proba
    )

    lime_html = lime_exp.as_html()
    st.components.v1.html(lime_html, height=500)

    # PDP
    st.subheader("📐 Partial Dependence Plot (PDP)")
    fig_pdp, ax_pdp = plt.subplots(figsize=(8, 4))
    display = PartialDependenceDisplay.from_estimator(
        model, X_scaled, features=[final_features.index("sc")], ax=ax_pdp
    )
    st.pyplot(fig_pdp)
