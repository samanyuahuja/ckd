import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import PartialDependenceDisplay

@st.cache_resource
def load_resources():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    try:
        X_train_scaled = joblib.load("X_train_scaled.pkl")
    except:
        X_train_scaled = None
    return model, scaler, X_train_scaled

model, scaler, X_train_scaled = load_resources()

final_features = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'wbcc', 'rbcc', 'htn', 'dm', 'appet', 'pe',
    'bun_sc_ratio', 'high_creatinine', 'hemo_bu'
]

def preprocess_input(df):
    mapper = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0,
              "yes": 1, "no": 0, "good": 0, "poor": 1}
    for col in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'appet', 'pe']:
        if col in df.columns:
            df[col] = df[col].map(mapper).fillna(0)
    for col in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
                'sod', 'pot', 'hemo', 'wbcc', 'rbcc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df["bun_sc_ratio"] = np.where(df["sc"] == 0, 0, df["bu"] / df["sc"])
    df["high_creatinine"] = (df["sc"] > 1.2).astype(int)
    df["hemo_bu"] = df["hemo"] * df["bu"]
    for col in ['sc', 'bu', 'bgr', 'wbcc', 'rbcc']:
        df[col] = np.log1p(df[col])
    for col in final_features:
        if col not in df.columns:
            df[col] = 0
    return df[final_features]

st.title("CKD Prediction App with Explainability")
uploaded_file = st.file_uploader("Upload CSV file with required features", type=["csv"])
X_input_df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        X_input_df = preprocess_input(df)
        st.write("Uploaded Data Preview:", df.head())
    except Exception as e:
        st.error(f"File read error: {e}")
        st.stop()
else:
    st.subheader("Manual Input")
    st.write("Add manual input logic here (or copy from your original code)")

if X_input_df is not None:
    X_scaled = scaler.transform(X_input_df)
    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]
    st.subheader("ð® Prediction Result")
    st.write("CKD Likely:" if prediction[0] else "CKD Unlikely")
    st.write("Probability:", round(proba[0], 3))
    st.subheader("ð SHAP Explanation")
    explainer = shap.Explainer(model, X_input_df)
    shap_values = explainer(X_input_df)
    try:
        shap.plots.waterfall(shap_values[0])
    except Exception as e:
        st.error(f"Waterfall plot failed: {e}")
    try:
        shap.plots.bar(shap_values)
    except Exception as e:
        st.error(f"Bar plot failed: {e}")
    st.subheader("ð¢ LIME Explanation")
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled if X_train_scaled is not None else X_scaled,
            feature_names=final_features,
            class_names=["No CKD", "CKD"],
            mode="classification"
        )
        lime_exp = lime_explainer.explain_instance(X_scaled[0], model.predict_proba, num_features=10)
        st.pyplot(lime_exp.as_pyplot_figure())
    except Exception as e:
        st.error(f"LIME Error: {e}")
    st.subheader("Partial Dependence Plot (PDP)")
    try:
        feature = st.selectbox("Select feature for PDP", final_features)
        pdp_data = X_train_scaled if X_train_scaled is not None else X_scaled
        fig_pdp, ax_pdp = plt.subplots()
        PartialDependenceDisplay.from_estimator(model, pdp_data, [final_features.index(feature)], ax=ax_pdp, feature_names=final_features)
        st.pyplot(fig_pdp)
    except Exception as e:
        st.error(f"PDP Error: {e}")
