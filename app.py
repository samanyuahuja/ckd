import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import PartialDependenceDisplay
st.set_page_config(
    page_title="CKD Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# Force light theme + custom styling
st.markdown(
    """
    <style>
    /* Make all text labels, headers, and markdown black */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stSubheader, label {
        color: #000000 !important;
    }

    /* Background white for main app */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar light grey */
    section[data-testid="stSidebar"] {
        background-color: #f0f0f0;
    }

    /* Button style: black text on light grey */
    .stButton>button {
        background-color: #f0f0f0;
        color: #000000;
        border: 1px solid #000000;
    }
    .stButton>button:hover {
        background-color: #e0e0e0;
    }
    /* Force UI labels to look like small black text */
    label, .stFileUploader label, .stNumberInput label, .stSelectbox label, .stSlider label, .stTextInput label {
        font-size: 14px !important;
        font-weight: normal !important;
        color: #000000 !important;
    }

    /* Ensure markdown headers and form text is also black and small if needed */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #000000 !important;
    }

    .stSubheader, .stHeader {
        color: #000000 !important;
    }
    div.stAlert {
        background-color: black;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------- Load model, scaler, and optional training data ----------------

@st.cache_resource
def load_resources(model_choice="rf"):
    if model_choice == "rf":
        model = joblib.load("rf_model_final (1).pkl")
    elif model_choice == "logistic":
        model = joblib.load("logistic_model_final (1).pkl")
    else:
        st.error("❌ Unknown model choice!")
        st.stop()
    
    scaler = joblib.load("scaler_final (1).pkl")
    
    try:
        X_train_res = joblib.load("X_train_res_scaled_final.pkl")
    except Exception as e:
        st.warning(f"⚠ X_train_res failed to load: {e}")
        X_train_res = None

    return model, scaler, X_train_res

# Example: load Random Forest
model, scaler, X_train_res = load_resources(model_choice="rf")








# ---------------- Define final features ----------------

final_features = [
    'age', 'bp', 'al', 'su', 'rbc', 'pc', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'wbcc', 'htn', 'dm', 'appet', 'pe',
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

    numeric_cols = ['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc',
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

st.markdown("""
<div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
    <h4 style='color: black; margin: 0;'>🩺 CKD Risk Assessment</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
    <h4 style='color: black; margin: 0;'>🔹 Upload Your Medical Report (CSV Format)</h4>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Upload a CSV file containing your medical data.",
    type=["csv"],
    help="The file should include the required features like age, bp, sc, etc."
)

X_input_df = None

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    try:
        df = pd.read_csv(uploaded_file)
        X_input_df = preprocess_input(df)
        st.markdown("### 📋 Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"❌ File read error: {e}")
        st.stop()
else:
        st.markdown("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
        <h4 style='color: black; margin: 0;'>🔹 Or Manually Enter Your Medical Information</h4>
    </div>
    """, unsafe_allow_html=True)
   
    st.info("Please fill out the form below if you do not have a CSV file.")

    default = {
        'age': 45, 'bp': 80, 'al': 1, 'su': 0, 'rbc': 'normal',
        'pc': 'normal', 'ba': 'notpresent', 'bgr': 150,
        'bu': 50, 'sc': 1.5, 'sod': 140, 'pot': 4.5, 'hemo': 12.0, 'wbcc': 7000,
        'rbcc': 4.5, 'htn': 'no', 'dm': 'no', 'appet': 'good', 'pe': 'no',
        'ane': 'no'   # ✅ Add ane (defaulting to 'no' or your preferred default)
    }


    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 6px; border: 1px solid #ccc;'>
    """, unsafe_allow_html=True)
    
    with st.form("manual_input"):
        cols = st.columns(3)
        inputs = {}
        with cols[0]:
            inputs['age'] = st.number_input("Age", value=default['age'])
            inputs['bp'] = st.number_input("Blood Pressure", value=default['bp'])
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
    
        submit = st.form_submit_button("🔹 Predict CKD Risk")
    
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        user_df = pd.DataFrame([inputs])
        X_input_df = preprocess_input(user_df)
        st.write("Input Summary:", user_df)

# ---------------- Prediction + Explainability ----------------

if X_input_df is not None:
    # 🔍 Show initial features
    # 🔍 Debug before scaling
    

        
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

    # 🚨 Outlier z-score check
    for col, val in X_input_df.iloc[0].items():
        idx = scaler.feature_names_in_.tolist().index(col)
        mean = scaler.mean_[idx]
        std = np.sqrt(scaler.var_[idx])
        z = (val - mean) / std if std != 0 else 0
        if abs(z) > 5:
            st.warning(f"⚠ {col} is {round(z,2)} std devs from mean — possible outlier!")

 
    if np.isnan(X_scaled).any():
        st.error("❌ X_scaled contains NaN values! Please check your input data or preprocessing.")
        st.stop()
    
    # Check for column count mismatch
    if X_scaled.shape[1] != len(scaler.feature_names_in_):
        st.error(f"❌ X_scaled has {X_scaled.shape[1]} columns but scaler expects {len(scaler.feature_names_in_)}.")
        st.error(f"Scaler features: {scaler.feature_names_in_}")
        st.stop()
    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]

    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
        <h4 style='color: black; margin: 0;'>🔹 Prediction Result</h4>
    </div>
    """, unsafe_allow_html=True)
    if len(prediction) == 1:
        st.write("CKD Likely:" if prediction[0] else "CKD Unlikely")
        st.write("Probability:", round(proba[0], 3))
    else:
        st.write("Predictions:", prediction.tolist())
        st.write("Probabilities:", [round(p, 3) for p in proba.tolist()])
    # === Top Feature Importance (Beautiful Section) ===
    if hasattr(model, "feature_importances_"):
        feat_imp = dict(zip(scaler.feature_names_in_, model.feature_importances_))
        top_feats = sorted(feat_imp.items(), key=lambda x: -x[1])[:5]
    
        st.markdown("""
        <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; border: 1px solid #ccc;'>
            <h4 style='color: black; margin-bottom: 10px;'>🔹 Top Model Feature Importances</h4>
        </div>
        """, unsafe_allow_html=True)
    
        for feat, val in top_feats:
            st.markdown(f"<div style='color:black; font-size: 16px;'>• <b>{feat}</b>: {val:.3f}</div>", unsafe_allow_html=True)
    
    elif hasattr(model, "coef_"):
        coefs = dict(zip(scaler.feature_names_in_, model.coef_[0]))
        top_feats = sorted(coefs.items(), key=lambda x: -abs(x[1]))[:5]
    
        st.markdown("""
        <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; border: 1px solid #ccc;'>
            <h4 style='color: black; margin-bottom: 10px;'>🔹 Top Model Coefficients</h4>
        </div>
        """, unsafe_allow_html=True)
    
        for feat, val in top_feats:
            st.markdown(f"<div style='color:black; font-size: 16px;'>• <b>{feat}</b>: {val:.3f}</div>", unsafe_allow_html=True)
    # 📝 Professional CKD Report
    # Compute no CKD probability
    no_ckd_proba = 1 - proba[0]
    
    # 📝 Professional CKD Report
    if prediction[0] == 0 and no_ckd_proba >= 0.75:
        st.markdown(f"""
        ### 📝 CKD Risk Report  
        ✅ **Prediction:** {"No CKD" if prediction[0] == 0 else "Possible CKD"}  
        ✅ **Confidence in no CKD:** {round(no_ckd_proba, 3)}  
        ✅ **Confidence in CKD:** {round(proba[0], 3)}  
        """)
        
        if prediction[0] == 0 and no_ckd_proba >= 0.75:
            st.success("🟢 High confidence: No CKD likely.")
        elif prediction[0] == 0:
            st.info("ℹ No CKD predicted, but confidence is moderate.")
        else:
            st.warning("⚠ Possible CKD detected. Please consult a doctor.")
        
        # List top SHAP features that push toward no CKD
        try:
            if isinstance(shap_values, list):
                shap_contribs = shap_values[1][0]
            else:
                shap_contribs = shap_values[0]
            
            top_negative = sorted(
                zip(X_input_df.columns, shap_contribs),
                key=lambda x: x[1]
            )[:3]
            
            for feat, val in top_negative:
                st.write(f"• {feat}: contributed toward no CKD")
        except Exception as e:
            st.warning(f"Could not generate SHAP feature reasons: {e}")
        
        st.info("⚠ *Note: This is an AI model's prediction and should not replace medical advice. If you have any symptoms or concerns, please consult a healthcare provider.*")
    else:
        st.info(f"ℹ The model does not have high confidence in no CKD (no CKD probability = {round(no_ckd_proba, 3)}).")
    # ---------------- SHAP ----------------
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
        <h4 style='color: black; margin: 0;'>🔹 SHAP Explanation</h4>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown("""
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
                <h4 style='color: black; margin: 0;'>🔹 SHAP Waterfall Plot</h4>
            </div>
            """, unsafe_allow_html=True)
            fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_exp[0], show=False)
            st.pyplot(fig_wf)
        except Exception as e:
            st.error(f"Waterfall plot failed: {e}")
            
            # Summary bar plot
        try:
            st.markdown("""
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
                <h4 style='color: black; margin: 0;'>🔹 SHAP Summary Bar Plot</h4>
            </div>
            """, unsafe_allow_html=True)
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            shap.plots.bar(shap_exp, show=False)
            st.pyplot(fig_bar)
        except Exception as e:
            st.error(f"Bar plot failed: {e}")
    except Exception as e:
        st.error(f"SHAP plot failed: {e}")

    # ---------------- LIME ----------------
    # ---------------- LIME ----------------
    # ---------------- LIME ----------------
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
        <h4 style='color: black; margin: 0;'>🔹 LIME Explanation</h4>
    </div>
    """, unsafe_allow_html=True)
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_res,
            feature_names=scaler.feature_names_in_,
            class_names=["No CKD", "CKD"],
            mode="classification"
        )
        
        lime_exp = lime_explainer.explain_instance(
            X_scaled[0],
            model.predict_proba,
            num_features=10
        )
        
        fig_lime = lime_exp.as_pyplot_figure()
        st.pyplot(fig_lime)
    
    except Exception as e:
        st.error(f"LIME Error: {e}")
    
    # ---------------- PDP ----------------
    

    # Move selectbox outside plot logic so it just stores state
        # PDP
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 6px; border: 1px solid #ccc;'>
        <h4 style='color: black; margin: 0;'>🔹 Partial Dependence Plot (PDP)</h4>
    </div>
    """, unsafe_allow_html=True)

    try:
        fig, axs = plt.subplots(
            nrows=(len(final_features) + 2) // 3, 
            ncols=3, 
            figsize=(15, 5 * ((len(final_features) + 2) // 3))
        )
    
        axs = axs.flatten()
    
        for i, feature in enumerate(final_features):
            PartialDependenceDisplay.from_estimator(
                model,
                X_train_res,
                features=[feature],
                ax=axs[i],
                feature_names=final_features
            )
            axs[i].set_title(f"PDP: {feature}")
    
        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
    
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"PDP generation failed: {e}")
