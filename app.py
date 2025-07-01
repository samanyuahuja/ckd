import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import PartialDependenceDisplay

# ---------------- Load model, scaler, and optional training data ----------------

@st.cache_resource
def load_resources():
    model = joblib.load("model (6).pkl")
    scaler = joblib.load("scaler (5).pkl")
    try:
        X_train_scaled = joblib.load("X_train_scaled.pkl")
    except Exception as e:
        st.warning(f"X_train_scaled failed to load: {e}")
        X_train_scaled = None
    return model, scaler, X_train_scaled

model, scaler, X_train_scaled = load_resources()

if X_train_scaled is None:
    st.warning("⚠ X_train_scaled is missing. LIME and PDP may fail without it.")
else:
    st.write("✅ X_train_scaled loaded. Shape:", X_train_scaled.shape)

st.write("✅ Model type:", type(model))
st.write("✅ Scaler feature_names_in_:", getattr(scaler, "feature_names_in_", "Not available"))

# ---------------- Define final features ----------------

final_features = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'wbcc', 'rbcc', 'htn', 'dm', 'appet', 'pe',
    'bun_sc_ratio', 'high_creatinine', 'hemo_bu'
]

# ---------------- Preprocessing function ----------------

def preprocess_input(df):
    mapper = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0,
              "yes": 1, "no": 0, "good": 0, "poor": 1}
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'appet', 'pe']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].map(mapper).fillna(0)

    numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
                    'sod', 'pot', 'hemo', 'wbcc', 'rbcc']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Derived features
    df["bun_sc_ratio"] = np.where(df["sc"] == 0, 0, df["bu"] / df["sc"])
    df["high_creatinine"] = (df["sc"] > 1.2).astype(int)
    df["hemo_bu"] = df["hemo"] * df["bu"]

    # Apply log1p to skewed features
    for col in ['sc', 'bu', 'bgr', 'wbcc', 'rbcc']:
        df[col] = np.log1p(df[col])

    for col in final_features:
        if col not in df.columns:
            df[col] = 0

    return df[final_features]

# ---------------- Streamlit App UI ----------------

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

    default = {
        'age': 45, 'bp': 80, 'sg': 1.015, 'al': 1, 'su': 0, 'rbc': 'normal',
        'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent', 'bgr': 150,
        'bu': 50, 'sc': 1.5, 'sod': 140, 'pot': 4.5, 'hemo': 12.0, 'wbcc': 7000,
        'rbcc': 4.5, 'htn': 'no', 'dm': 'no', 'appet': 'good', 'pe': 'no'
    }

    with st.form("manual_input"):
        cols = st.columns(3)
        inputs = {}
        with cols[0]:
            inputs['age'] = st.number_input("Age", value=default['age'])
            inputs['bp'] = st.number_input("Blood Pressure", value=default['bp'])
            inputs['sg'] = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
            inputs['al'] = st.slider("Albumin", 0, 5, value=default['al'])
            inputs['su'] = st.slider("Sugar", 0, 5, value=default['su'])
            inputs['rbc'] = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
            inputs['pc'] = st.selectbox("Pus Cell", ["normal", "abnormal"])

        with cols[1]:
            inputs['pcc'] = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
            inputs['ba'] = st.selectbox("Bacteria", ["present", "notpresent"])
            inputs['bgr'] = st.number_input("Blood Glucose Random", value=default['bgr'])
            inputs['bu'] = st.number_input("Blood Urea", value=default['bu'])
            inputs['sc'] = st.number_input("Serum Creatinine", value=default['sc'])
            inputs['sod'] = st.number_input("Sodium", value=default['sod'])
            inputs['pot'] = st.number_input("Potassium", value=default['pot'])

        with cols[2]:
            inputs['hemo'] = st.number_input("Hemoglobin", value=default['hemo'])
            inputs['wbcc'] = st.number_input("WBC Count", value=default['wbcc'])
            inputs['rbcc'] = st.number_input("RBC Count", value=default['rbcc'])
            inputs['htn'] = st.selectbox("Hypertension", ["yes", "no"])
            inputs['dm'] = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            inputs['appet'] = st.selectbox("Appetite", ["good", "poor"])
            inputs['pe'] = st.selectbox("Pedal Edema", ["yes", "no"])

        submit = st.form_submit_button("Predict")

    if submit:
        user_df = pd.DataFrame([inputs])
        X_input_df = preprocess_input(user_df)
        st.write("Input Summary:", user_df)

# ---------------- Prediction + Explainability ----------------

if X_input_df is not None:
    X_scaled = scaler.transform(X_input_df)

    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]

    st.subheader("🔮 Prediction Result")
    if len(prediction) == 1:
        st.write("CKD Likely:" if prediction[0] else "CKD Unlikely")
        st.write("Probability:", round(proba[0], 3))
    else:
        st.write("Predictions:", prediction.tolist())
        st.write("Probabilities:", proba.tolist())

    # ---------------- SHAP ----------------
    st.subheader("📊 SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    try:
        if isinstance(shap_values, list):
            # Multi-class model, use class 1
            shap_exp = shap.Explanation(
                values=shap_values[1],
                base_values=explainer.expected_value[1],
                data=X_scaled,
                feature_names=X_input_df.columns
            )
        else:
            # Binary model or Explanation returned
            shap_exp = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X_scaled,
                feature_names=X_input_df.columns
            )
    
        

        # Waterfall plot
        try:
            st.subheader("SHAP Waterfall Plot (Instance 0)")
            fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_exp[0], show=False)
            st.pyplot(fig_wf)
        except Exception as e:
            st.error(f"Waterfall plot failed: {e}")
            
            # Summary bar plot
        try:
            st.subheader("SHAP Summary Bar Plot")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            shap.plots.bar(shap_exp, show=False)
            st.pyplot(fig_bar)
    except Exception as e:
        st.error(f"Bar plot failed: {e}")


    # ---------------- LIME ----------------
    st.subheader("🟢 LIME Explanation")
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled if X_train_scaled is not None else X_scaled,
            feature_names=final_features,
            class_names=["No CKD", "CKD"],
            mode="classification"
        )
        lime_exp = lime_explainer.explain_instance(X_scaled[0], model.predict_proba, num_features=10)
        fig_lime = lime_exp.as_pyplot_figure()
        st.pyplot(fig_lime)
    except Exception as e:
        st.error(f"LIME Error: {e}")

    # ---------------- PDP ----------------
    st.subheader("📐 Partial Dependence Plot (PDP)")
    try:
        feature = st.selectbox("Select feature for PDP", final_features, index=final_features.index("hemo"))
        pdp_data = X_train_scaled if X_train_scaled is not None else X_scaled
        fig_pdp, ax_pdp = plt.subplots()
        PartialDependenceDisplay.from_estimator(
            model,
            pdp_data,
            features=[feature],
            ax=ax_pdp,
            feature_names=final_features
        )
        st.pyplot(fig_pdp)
    except Exception as e:
        st.error(f"PDP Error: {e}")
