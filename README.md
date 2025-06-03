# CKD Predictor App 

A Streamlit-powered machine learning application that predicts Chronic Kidney Disease using patient medical data. Includes explainable AI add-ons like SHAP, LIME, and PDP to make predictions transparent and understandable.

## Features

- Random Forest Classifier for CKD prediction
- SHAP plots for global & local explanation
- LIME for real-time patient-specific insight
- Partial Dependence Plots (PDPs)
- Streamlit web interface
- Upload custom data for prediction
- Clean UI with expandable explanations

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- SHAP
- LIME
- Pandas, NumPy, Matplotlib 

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
