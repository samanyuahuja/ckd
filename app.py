import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
from io import StringIO

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Final features used in model
final_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'bgr', 'bu', 'sc',
                  'sod', 'pot', 'hemo', 'wbcc', 'rbcc', 'htn', 'dm', 'appet', 'pe',
                  'bun_sc_ratio', 'high_creatinine', 'hemo_bu']

st.title("CKD Prediction App with Explainability")
st.write("Upload your data below or manually enter values for prediction.")

# Upload CSV or manual entry
uploaded_file = st.file_uploader("Upload a CSV file with required features", type=["csv"])

if uploaded_file:
    X_input = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", X_input.head())
else:
    X_input = pd.DataFrame({feature: [0] for feature in final_features})
    for col in final_features:
        val = st.text_input(f"Enter value for {col}", value="0")
        X_input[col] = [float(val)]

# Ensure input has the same columns
X_input = X_input[final_features]
X_scaled = scaler.transform(X_input)

# Make prediction
prediction = model.predict(X_scaled)
proba = model.predict_proba(X_scaled)[:, 1]

st.subheader("Prediction")
st.write("CKD Likelihood (1 = CKD likely, 0 = CKD unlikely):", int(prediction[0]))
st.write("Probability of CKD:", round(proba[0], 3))

# SHAP Explanation
st.subheader("SHAP Explanation")
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)
expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
shap_vals_class1 = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

st.set_option('deprecation.showPyplotGlobalUse', False)
fig_summary, ax = plt.subplots()
shap.summary_plot([shap_vals_class1], X_input, plot_type="bar", show=False)
st.pyplot(fig_summary)

# Force plot
st.subheader("SHAP Force Plot")
shap_html = shap.force_plot(expected_value, shap_vals_class1, X_input.iloc[0], matplotlib=False)
from streamlit.components.v1 import html
html(shap_html.html(), height=300)

# LIME Explanation
st.subheader("LIME Explanation")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=scaler.transform(X_input),
    feature_names=final_features,
    class_names=['No CKD', 'CKD'],
    mode='classification'
)

lime_exp = lime_explainer.explain_instance(X_scaled[0], model.predict_proba, num_features=10)
fig_lime = lime_exp.as_pyplot_figure()
st.pyplot(fig_lime)

# Partial Dependence Plot
st.subheader("Partial Dependence Plot (PDP)")
feature_to_plot = st.selectbox("Select feature for PDP", final_features)
fig_pdp, ax_pdp = plt.subplots()
PartialDependenceDisplay.from_estimator(model, X_scaled, [final_features.index(feature_to_plot)], ax=ax_pdp, feature_names=final_features)
st.pyplot(fig_pdp)
