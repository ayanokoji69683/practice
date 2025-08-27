import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="AI Diabetes Predictor", page_icon="🧑‍⚕️", layout="centered")
st.title("🧑‍⚕️ AI-Powered Diabetes Risk Prediction")
st.write("Enter patient details to check risk of Diabetes using Machine Learning.")

# Inputs
preg = st.number_input("Number of Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 700, 100)
bp = st.number_input("Blood Pressure", 0, 250, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("🔍 Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][prediction] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes! (Confidence: {prob:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes (Confidence: {prob:.2f}%)")
