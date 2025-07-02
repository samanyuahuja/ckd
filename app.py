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
def load_resources(model_choice="rf"):
    # Load the model
    if model_choice == "rf":
        model = joblib.load("Rf_model.pkl")
    elif model_choice == "logistic":
        model = joblib.load("Logistic_model.pkl")
    else:
        st.error("❌ Unknown model choice!")
        st.stop()
    
    # Load the scaler
    scaler = joblib.load("scaler_18.pkl")  # clean filename
    
    # Load X_train_res
    try:
        X_train_res = joblib.load("X_train_res.pkl")
    except Exception as e:
        st.warning(f"⚠ X_train_res failed to load: {e}")
        X_train_res = None
    
    return model, scaler, X_train_res

# Example: load Random Forest
model, scaler, X_train_res = load_resources(model_choice="rf")

# Feedback for user
if X_train_res is None:
    st.warning("⚠ X_train_res is missing. LIME and PDP may fail without it.")
else:
    st.write("✅ X_train_res loaded. Shape:", X_train_res.shape)

st.write("✅ Scaler mean_:", scaler.mean_.tolist())
st.write("✅ Scaler var_:", scaler.var_.tolist())





# ---------------- Define final features ----------------

final_features = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'wbcc', 'rbcc', 'htn', 'dm', 'appet', 'pe',
    'ane', 'bun_sc_ratio', 'high_creatinine', 'hemo_bu'
]

# ---------------- Preprocessing function ----------------

def preprocess_input(df):
    mapper = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0,
              "yes": 1, "no": 0, "good": 0, "poor": 1}
    categorical_cols = ['rbc', 'pc', 'ba', 'htn', 'dm', 'appet', 'pe', 'ane']

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
        'pc': 'normal', 'ba': 'notpresent', 'bgr': 150,
        'bu': 50, 'sc': 1.5, 'sod': 140, 'pot': 4.5, 'hemo': 12.0, 'wbcc': 7000,
        'rbcc': 4.5, 'htn': 'no', 'dm': 'no', 'appet': 'good', 'pe': 'no',
        'ane': 'no'   # ✅ Add ane (defaulting to 'no' or your preferred default)
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
            
            inputs['ba'] = st.selectbox("Bacteria", ["present", "notpresent"])
            inputs['bgr'] = st.number_input("Blood Glucose Random", value=default['bgr'])
            inputs['bu'] = st.number_input("Blood Urea", value=default['bu'])
            inputs['sc'] = st.number_input("Serum Creatinine", value=default['sc'])
            inputs['sod'] = st.number_input("Sodium", value=default['sod'])
            inputs['pot'] = st.number_input("Potassium", value=default['pot'])
            inputs['hemo'] = st.number_input("Hemoglobin", value=default['hemo'])
        with cols[2]:
            
            inputs['wbcc'] = st.number_input("WBC Count", value=default['wbcc'])
            inputs['rbcc'] = st.number_input("RBC Count", value=default['rbcc'])
            inputs['htn'] = st.selectbox("Hypertension", ["yes", "no"])
            inputs['dm'] = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            inputs['appet'] = st.selectbox("Appetite", ["good", "poor"])
            inputs['pe'] = st.selectbox("Pedal Edema", ["yes", "no"])
            inputs['ane'] = st.selectbox("Anemia", ["yes", "no"])


        submit = st.form_submit_button("Predict")

    if submit:
        user_df = pd.DataFrame([inputs])
        X_input_df = preprocess_input(user_df)
        st.write("Input Summary:", user_df)

# ---------------- Prediction + Explainability ----------------

if X_input_df is not None:
    # 🔍 Show initial features
    # 🔍 Debug before scaling
    debug_info = []
    debug_info.append("==== SCALER & INPUT DEBUG ====")
    debug_info.append(f"⚡ Scaler expects features: {scaler.feature_names_in_.tolist()}")
    debug_info.append(f"⚡ Input provided features: {X_input_df.columns.tolist()}")
    debug_info.append(f"✅ Input values before scaling: {X_input_df.iloc[0].to_dict()}")
    debug_info.append(f"✅ Scaler mean_: {scaler.mean_.tolist()}")
    debug_info.append(f"✅ Scaler var_: {scaler.var_.tolist()}")
    
    # Show in Streamlit
    for line in debug_info:
        st.write(line)
        
    # Align input features to match scaler
    for col in scaler.feature_names_in_:
        if col not in X_input_df.columns:
            st.warning(f"⚠ Adding missing column: {col}")
            X_input_df[col] = 0
    extra_cols = [col for col in X_input_df.columns if col not in scaler.feature_names_in_]
    if extra_cols:
        st.warning(f"⚠ Dropping extra columns: {extra_cols}")
        X_input_df = X_input_df.drop(columns=extra_cols)

    # Reorder to match scaler
    X_input_df = X_input_df[scaler.feature_names_in_]

    # ✅ Now safe to transform
    X_scaled = scaler.transform(X_input_df)

    # 🔍 Debug
    st.write("✅ Columns aligned. First scaled row:", X_scaled[0].tolist())

    # Predict
    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]

    st.subheader("🔮 Prediction Result")
    if len(prediction) == 1:
        st.write("CKD Likely:" if prediction[0] else "CKD Unlikely")
        st.write("Probability:", round(proba[0], 3))
    else:
        st.write("Predictions:", prediction.tolist())
        st.write("Probabilities:", [round(p, 3) for p in proba.tolist()])


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
    except Exception as e:
        st.error(f"SHAP plot failed: {e}")

    # ---------------- LIME ----------------
    # ---------------- LIME ----------------
    st.subheader("🟢 LIME Explanation")
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_res if X_train_res is not None else X_scaled,
            feature_names=final_features,
            class_names=["No CKD", "CKD"],
            mode="classification"
        )
        
        lime_exp = lime_explainer.explain_instance(
            X_scaled[0],
            lambda x: model.predict_proba(x).astype(float),  # ensure float dtype, proper slicing
            num_features=10
        )
        
        fig_lime = lime_exp.as_pyplot_figure()
        st.pyplot(fig_lime)
    
    except Exception as e:
        st.error(f"LIME Error: {e}")

    # ---------------- PDP ----------------
    st.subheader("📐 Partial Dependence Plot (PDP)")
    try:
        feature = st.selectbox("Select feature for PDP", final_features, index=final_features.index("hemo"))
        pdp_data = X_train_res if X_train_res is not None else X_scaled
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
