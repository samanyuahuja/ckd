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
import streamlit.components.v1 as components
from streamlit.components.v1 import html
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay


# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(layout="wide")
st.title("CKD Risk Predictor with SHAP & LIME")
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
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"], index=0)
        pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"], index=1)
    with col2:
        
        ba = st.selectbox("Bacteria", ["present", "notpresent"], index=1)
        bu = st.number_input("Blood Urea", 1.0, 400.0, 50.0)
        sc = st.number_input("Serum Creatinine", 0.1, 50.0, 1.5)
        hemo = st.number_input("Hemoglobin", 3.0, 17.0, 12.0)
        bgr = st.number_input("Blood Glucose Random (bgr)", 70, 500, 100)
        sod = st.number_input("Sodium (sod)", 100, 150, 140)
        pot = st.number_input("Potassium (pot)", 1.0, 10.0, 4.0)
        htn = st.selectbox("Hypertension", ["yes", "no"], index=1)
    with col3:
        
        dm = st.selectbox("Diabetes Mellitus", ["yes", "no"], index=1)
        appet = st.selectbox("Appetite", ["good", "poor"], index=0)
        ane = st.selectbox("Anemia", ["yes", "no"], index=1)
        wbcc = st.number_input("White Blood Cell Count (wbcc)", 3000, 15000, 7000)
        rbcc = st.number_input("Red Blood Cell Count (rbcc)", 2.0, 6.0, 4.5)
        pe = st.selectbox("Pedal Edema (pe)", ["yes", "no"], index=1)
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
    "bgr": [bgr],
    "bu": [bu],
    "sc": [sc],
    "sod": [sod],
    "pot": [pot],
    "hemo": [hemo],
    "wbcc": [wbcc],
    "rbcc": [rbcc],
    "htn": [mapper[htn]],
    "dm": [mapper[dm]],
    "appet": [mapper[appet]],
    "pe": [mapper[pe]]
})


    # Add derived features
    X_input["bun_sc_ratio"] = X_input["bu"] / X_input["sc"]
    X_input["high_creatinine"] = (X_input["sc"] > 1.2).astype(int)
    X_input["hemo_bu"] = X_input["hemo"] / (X_input["bu"] + 1)


    # Reorder columns
    final_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'bgr', 'bu', 'sc', 'sod', 'pot',
                      'hemo', 'wbcc', 'rbcc', 'htn', 'dm', 'appet', 'pe', 'bun_sc_ratio',
                      'high_creatinine', 'hemo_bu']



    X_input_ordered = X_input[scaler.feature_names_in_]

    # Scale input
    X_scaled = scaler.transform(X_input_ordered)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]


    st.subheader("🔍 Prediction Result")
    st.success(f"CKD Risk: {'Positive' if prediction == 1 else 'Negative'}")
    st.info(f"Probability of CKD: {prob * 100:.2f}%")

    st.subheader("SHAP Explanation")

    # Get single input
    X_single = X_scaled[0:1]
    features_single = X_input[final_features].iloc[0]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)
    
    # Log shape
    st.write(f"shap_values shape: {np.array(shap_values).shape}")
    
    # Extract class 1 SHAP values
    shap_vals_class1 = shap_values[0, :, 1]  # (23,)
    
    # Expected value
    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    # Create force plot
    force_plot = shap.force_plot(
        base_value=expected_val,
        shap_values=shap_vals_class1,
        features=features_single,
        matplotlib=False,
        show=False
    )
    
    # Display in Streamlit
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.subheader("SHAP Force Plot")
    html(shap_html, height=300)
    
    # Summary Plot
    st.subheader("SHAP Summary Plot")
    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals_class1.reshape(1, -1), features_single.to_frame().T, plot_type="bar", show=False)
    st.pyplot(fig_summary)

    st.subheader("🔍 Input Data Check for LIME and PDP")
    st.write("X_input sample:")
    st.write(X_input[final_features].iloc[0])
    st.write("Prediction for input row:", model.predict(X_input[final_features].iloc[0:1]))
    st.write("Prediction probabilities:", model.predict_proba(X_input[final_features].iloc[0:1]))


    # LIME Explanation
    

    
    st.subheader("🔍 LIME Explanation")
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_input[final_features]),
        feature_names=final_features,
        class_names=['No CKD', 'CKD'],
        mode='classification',
        discretize_continuous=True
    )
    
    lime_exp = lime_explainer.explain_instance(
        data_row=X_input[final_features].iloc[0].values,
        predict_fn=lambda x: model.predict_proba(x).astype(float)
    )
    
    fig_lime = lime_exp.as_pyplot_figure()
    st.pyplot(fig_lime)

    
    
    
    st.subheader("Partial Dependence Plot (PDP)")
    
    fig_pdp, ax_pdp = plt.subplots(figsize=(8, 4))
    PartialDependenceDisplay.from_estimator(
        model,
        X_input[final_features],  # Unscaled features
        features=["sc"],          # Replace with a valid feature
        feature_names=final_features,
        ax=ax_pdp
    )
    st.pyplot(fig_pdp)
