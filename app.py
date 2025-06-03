# Step 1: Install necessary libraries

import streamlit as st
import pandas as pd
import numpy as np

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import streamlit.components.v1 as components
from io import StringIO
import os # Import the os module
import shap
from streamlit_shap import st_shap
from streamlit.components.v1 import html
import warnings
warnings.filterwarnings("ignore")
@st.cache_resource
def load_model():
    return joblib.load("my_model.pkl")

model = load_model()

# Optional: cache training data
@st.cache_data
def load_training_data():
    return pd.read_csv("X_train_scaled.csv")

# Load model and scaler
try:
    # Attempt to load the files from the current directory
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Model and Scaler loaded successfully.")

    # Define SHAP explainer globally after model is loaded
    explainer = shap.TreeExplainer(model)
except FileNotFoundError:
    st.error("Error: Model or scaler files not found.")
    st.info("Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory as this script.")
    st.stop() # Stop the app execution if files are missing


# Final features used in model
# NOTE: Ensure this list matches the features the model and scaler were trained on.
# The previous output from ipython-input-212 and 213 will show the feature names.
# Adjust this list if they don't match what's defined below.
# Example: scaler features_in_ might be ['age', 'bp', ..., 'hemo_bu']
# Example: model features_in_ might also be the same list.
# This list should match the columns provided to the model.
final_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'bgr', 'bu', 'sc', 'sod', 'pot',
                  'hemo', 'wbcc', 'rbcc', 'htn', 'dm', 'appet', 'pe', 'bun_sc_ratio',
                  'high_creatinine', 'hemo_bu']# Example list - verify this from your training code

st.title("CKD Prediction App with Explainability")
st.write("Upload your data below or manually enter values for prediction.")

# Input form
input_data = {}

# Use manual entry fields as the default unless a file is uploaded
use_manual_entry = True

uploaded_file = st.file_uploader("Upload a CSV file with required features", type=["csv"])

if uploaded_file:
    try:
        X_input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", X_input_df.head())
        use_manual_entry = False
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.info("Falling back to manual entry.")
        use_manual_entry = True


if use_manual_entry:
    st.subheader("Manual Input")
    # Define default values for manual entry form (optional, but good UX)
    # These should ideally be representative or based on feature distributions
    default_values = {
        'age': 45, 'bp': 80, 'sg': 1.015, 'al': 1, 'su': 0,
        'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'bgr': 150, 'bu': 50,
        'sc': 1.5, 'sod': 140, 'pot': 4.5, 'hemo': 12.0, 'wbcc': 7000,
        'rbcc': 4.5, 'htn': 'no', 'dm': 'no', 'appet': 'good', 'pe': 'no' # Add 'pe'
    }
    # Add placeholders for derived features; they will be calculated later
    default_values['ba'] = 'notpresent'
    default_values['bun_sc_ratio'] = default_values['bu'] / default_values['sc'] if default_values['sc'] != 0 else 0
    default_values['high_creatinine'] = 1 if default_values['sc'] > 1.2 else 0
    default_values['hemo_bu'] = default_values['hemo'] / (default_values['bu'] + 1) if default_values['bu'] >= 0 else 0


    # Input form with improved structure and default values
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data['age'] = st.number_input("Age", value=float(default_values.get('age', 0)))
            input_data['bp'] = st.number_input("Blood Pressure", value=float(default_values.get('bp', 0)))
            input_data['sg'] = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=[1.005, 1.010, 1.015, 1.020, 1.025].index(default_values.get('sg', 1.015)))
            input_data['al'] = st.slider("Albumin", 0, 5, value=default_values.get('al', 0))
            input_data['su'] = st.slider("Sugar", 0, 5, value=default_values.get('su', 0))
            input_data['rbc'] = st.selectbox("Red Blood Cells", ["normal", "abnormal"], index=["normal", "abnormal"].index(default_values.get('rbc', 'normal')))
            input_data['pc'] = st.selectbox("Pus Cell", ["normal", "abnormal"], index=["normal", "abnormal"].index(default_values.get('pc', 'normal')))
        with col2:
            input_data['pcc'] = st.selectbox("Pus Cell Clumps", ["present", "notpresent"], index=["present", "notpresent"].index(default_values.get('pcc', 'notpresent')))
            input_data['ba'] = st.selectbox("Bacteria", ["present", "notpresent"], index=["present", "notpresent"].index(default_values.get('ba', 'notpresent')))
            input_data['bgr'] = st.number_input("Blood Glucose Random", value=float(default_values.get('bgr', 0.0)))
            input_data['bu'] = st.number_input("Blood Urea", value=float(default_values.get('bu', 0.0)))
            input_data['sc'] = st.number_input("Serum Creatinine", value=float(default_values.get('sc', 0.0)))
            input_data['sod'] = st.number_input("Sodium", value=float(default_values.get('sod', 0.0)))
            input_data['pot'] = st.number_input("Potassium", value=float(default_values.get('pot', 0.0)))
        with col3:
            input_data['hemo'] = st.number_input("Hemoglobin", value=float(default_values.get('hemo', 0.0)))
            input_data['wbcc'] = st.number_input("White Blood Cell Count", value=float(default_values.get('wbcc', 0.0)))
            input_data['rbcc'] = st.number_input("Red Blood Cell Count", value=float(default_values.get('rbcc', 0.0)))
            input_data['htn'] = st.selectbox("Hypertension", ["yes", "no"], index=["yes", "no"].index(default_values.get('htn', 'no')))
            input_data['dm'] = st.selectbox("Diabetes Mellitus", ["yes", "no"], index=["yes", "no"].index(default_values.get('dm', 'no')))
            input_data['appet'] = st.selectbox("Appetite", ["good", "poor"], index=["good", "poor"].index(default_values.get('appet', 'good')))
            input_data['pe'] = st.selectbox("Pedal Edema", ["yes", "no"], index=["yes", "no"].index(default_values.get('pe', 'no'))) # Added 'pe'
        submit_manual = st.form_submit_button("Predict from Manual Input")

    if submit_manual:
        X_input_df = pd.DataFrame([input_data]) # Convert dictionary to DataFrame

        # Map categorical features for manual input
        mapper = {
            "normal": 0, "abnormal": 1,
            "present": 1, "notpresent": 0,
            "yes": 1, "no": 0,
            "good": 0, "poor": 1
        }
        categorical_cols_to_map = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'appet', 'pe'] # Include 'pe'
        for col in categorical_cols_to_map:
             if col in X_input_df.columns:
                X_input_df[col] = X_input_df[col].map(mapper).fillna(0) # Map and handle potential missing keys with 0

        # Ensure numerical columns are float/int
        for col in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'wbcc', 'rbcc']:
             if col in X_input_df.columns:
                X_input_df[col] = pd.to_numeric(X_input_df[col], errors='coerce').fillna(0) # Convert and fill errors/NaNs

        # Add derived features
        if 'bu' in X_input_df.columns and 'sc' in X_input_df.columns:
            X_input_df["bun_sc_ratio"] = X_input_df["bu"] / (X_input_df["sc"] + 1e-6) # Add small constant to avoid division by zero
        else:
             X_input_df["bun_sc_ratio"] = 0 # Default if base features missing

        if 'sc' in X_input_df.columns:
            X_input_df["high_creatinine"] = (X_input_df["sc"] > 1.2).astype(int)
        else:
            X_input_df["high_creatinine"] = 0

        if 'hemo' in X_input_df.columns and 'bu' in X_input_df.columns:
            # Note: Your previous code had hemo / (bu + 1), the original CKD notebook had hemo * bu.
            # Let's assume you meant the interaction hemo * bu as in the notebook before feature engineering.
            # If hemo / (bu+1) was intentional, keep that. Sticking to notebook's final feature list logic.
            X_input_df["hemo_bu"] = X_input_df["hemo"] * X_input_df["bu"]
        else:
            X_input_df["hemo_bu"] = 0


        # Apply log transformation to potentially skewed features
        # NOTE: These features were log-transformed *before* scaling and model training in your notebook.
        # You need to apply the *same* transformation *before* scaling the input data.
        skewed_features = ['sc', 'bu', 'bgr', 'wbcc', 'rbcc']
        for feature in skewed_features:
            if feature in X_input_df.columns:
                X_input_df[feature] = np.log1p(X_input_df[feature]) # Apply log1p

        # Ensure all final features are present, add missing ones with 0 or median/mean if appropriate
        # based on your imputation strategy, but 0 is simpler for a demo
        for feature in final_features:
            if feature not in X_input_df.columns:
                X_input_df[feature] = 0 # Add missing feature columns with default value

        # Reorder columns to match the training order
        X_input = X_input_df[final_features]


elif uploaded_file:
    # If a file was uploaded and read successfully, process it
    # Assume the uploaded CSV already contains the necessary features, including derived ones,
    # or that you need to add the derivation logic here for the file input as well.
    # For simplicity in this fix, let's assume the uploaded CSV is preprocessed
    # or contains the raw features that you need to process here.
    # Let's re-add the processing steps for the uploaded data too for consistency.

    # Assuming the uploaded file contains the raw features that the manual form takes
    # Re-process the uploaded dataframe
    try:
        # Map categorical features for uploaded file
        mapper = {
            "normal": 0, "abnormal": 1,
            "present": 1, "notpresent": 0,
            "yes": 1, "no": 0,
            "good": 0, "poor": 1
        }
        categorical_cols_to_map = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'appet', 'pe'] # Include 'pe'
        for col in categorical_cols_to_map:
             if col in X_input_df.columns:
                X_input_df[col] = X_input_df[col].map(mapper).fillna(0) # Map and handle potential missing keys with 0

        # Ensure numerical columns are float/int
        for col in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'wbcc', 'rbcc']:
             if col in X_input_df.columns:
                X_input_df[col] = pd.to_numeric(X_input_df[col], errors='coerce').fillna(0) # Convert and fill errors/NaNs

        # Add derived features
        if 'bu' in X_input_df.columns and 'sc' in X_input_df.columns:
            X_input_df["bun_sc_ratio"] = X_input_df["bu"] / (X_input_df["sc"] + 1e-6)
        else:
             X_input_df["bun_sc_ratio"] = 0

        if 'sc' in X_input_df.columns:
            X_input_df["high_creatinine"] = (X_input_df["sc"] > 1.2).astype(int)
        else:
            X_input_df["high_creatinine"] = 0

        if 'hemo' in X_input_df.columns and 'bu' in X_input_df.columns:
            X_input_df["hemo_bu"] = X_input_df["hemo"] * X_input_df["bu"]
        else:
            X_input_df["hemo_bu"] = 0

        # Apply log transformation
        skewed_features = ['sc', 'bu', 'bgr', 'wbcc', 'rbcc']
        for feature in skewed_features:
            if feature in X_input_df.columns:
                 X_input_df[feature] = np.log1p(X_input_df[feature])

        # Ensure all final features are present
        for feature in final_features:
            if feature not in X_input_df.columns:
                X_input_df[feature] = 0

        # Reorder columns to match the training order
        X_input = X_input_df[final_features]


    except Exception as e:
        st.error(f"Error processing uploaded data for prediction: {e}")
        st.stop() # Stop if data processing fails


# Check if X_input DataFrame is ready for prediction
# It will be ready if either manual input was submitted or a file was successfully uploaded and processed
if 'X_input' in locals() and not X_input.empty:

    # Check if the number of columns matches the scaler's expectations
    if X_input.shape[1] != len(scaler.feature_names_in_):
        st.error(f"Input data has {X_input.shape[1]} features, but the scaler expects {len(scaler.feature_names_in_)} features.")
        st.write("Scaler expected features:", scaler.feature_names_in_.tolist())
        st.write("Input features received:", X_input.columns.tolist())
        st.stop()


    # Scale the input data
    try:
        X_scaled = scaler.transform(X_input)
        st.write("Scaled Data Preview:", X_scaled[:5]) # Show preview of scaled data
    except Exception as e:
        st.error(f"Error during data scaling: {e}")
        st.stop()


    # Make prediction
    try:
        prediction = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[:, 1]
        shap_values = explainer.shap_values(X_scaled)

        st.subheader("Prediction")
        # Handle multiple predictions if a CSV was uploaded
        if X_input.shape[0] > 1:
            st.write("Predictions:", prediction.tolist())
            st.write("Probabilities of CKD:", [round(p, 3) for p in proba.tolist()])
            # For explanations, might want to pick one row or add a selector
            # For simplicity, explain the first row for now
            instance_to_explain_idx = 0
            st.write(f"\nShowing explanations for the first instance (Row {instance_to_explain_idx})")
            X_scaled_single = X_scaled[instance_to_explain_idx].reshape(1, -1)
            X_input_single_df = X_input.iloc[[instance_to_explain_idx]]
            prediction_single = prediction[instance_to_explain_idx]
            expected_value_single = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            shap_vals_class1_single = shap_values[1][instance_to_explain_idx] if isinstance(shap_values, list) else shap_values[instance_to_explain_idx]


        else:
            # Single prediction from manual input
            st.write("CKD Likelihood (1 = CKD likely, 0 = CKD unlikely):", int(prediction[0]))
            st.write("Probability of CKD:", round(proba[0], 3))
            instance_to_explain_idx = 0 # Only one instance
            X_scaled_single = X_scaled
            X_input_single_df = X_input
            prediction_single = prediction[0]
            expected_value_single = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            shap_vals_class1_single = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]


    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()


    # SHAP Explanation (for the single instance selected or the first instance)
      # Install via requirements.txt
    
    # SHAP Explanation
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
        
  


    # LIME
    
    
    
   
    
    # Assume these are defined in your app somewhere:
    # model: your trained sklearn-compatible model
    # X_scaled: numpy array of current input data, scaled (shape: n_samples x n_features)
    # final_features: list of feature names (length = n_features)
    # instance_to_explain_idx: int, index of instance to explain (e.g., 0 for first row)
    # X_scaled_single: numpy array containing single instance scaled (shape: 1 x n_features)
    # X_input_single_df: pandas DataFrame for the single instance (shape: 1 x n_features)
    
   

   
    
    # Assume model, X_scaled, final_features, X_scaled_single, instance_to_explain_idx are pre-defined
    
    # --------------------- LIME ---------------------
    st.subheader(f"🟢 LIME Explanation (Instance {instance_to_explain_idx})")
    
    try:
        if 'X_train_scaled' in locals():
            background_data_for_lime = X_train_scaled
            st.write(f"✅ Using X_train_scaled with shape: {background_data_for_lime.shape} for LIME.")
        else:
            background_data_for_lime = np.tile(X_scaled_single, (100, 1))
            st.write("⚠️ No X_train_scaled found; duplicated X_scaled_single 100 times for LIME background.")
            st.write(f"LIME background data shape: {background_data_for_lime.shape}")
    
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_data_for_lime,
            feature_names=final_features,
            class_names=['No CKD', 'CKD'],
            mode='classification'
        )
    
        lime_exp = lime_explainer.explain_instance(
            X_scaled_single[0],
            model.predict_proba,
            num_features=10
        )
        fig_lime = lime_exp.as_pyplot_figure()
        st.pyplot(fig_lime)
    
    except Exception as e:
        st.error(f"❌ Error generating LIME plot: {e}")
    
    # --------------------- PDP ---------------------
    st.subheader("📐 Partial Dependence Plot (PDP)")
    
    try:
        feature_to_plot = st.selectbox("Select feature for PDP", final_features, key="pdp_feature")
        
        if feature_to_plot:
            if 'X_train_scaled' in locals():
                pdp_data = X_train_scaled
                st.write(f"✅ Using X_train_scaled with shape: {pdp_data.shape} for PDP.")
            else:
                pdp_data = np.tile(X_scaled_single, (200, 1))
                st.write("⚠️ No X_train_scaled found; duplicated X_scaled_single 200 times for PDP.")
                st.write(f"PDP data shape: {pdp_data.shape}")
    
            pdp_data_df = pd.DataFrame(pdp_data, columns=final_features)
            feature_index = final_features.index(feature_to_plot)
            st.write(f"Generating PDP for feature '{feature_to_plot}' at index {feature_index}")
    
            fig_pdp, ax_pdp = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                model,
                pdp_data_df,
                features=[feature_index],
                ax=ax_pdp,
                feature_names=final_features
            )
            st.pyplot(fig_pdp)
    
    except Exception as e:
        st.error(f"❌ Error generating PDP plot: {e}")
