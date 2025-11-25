# app.py  ← Save as app.py
import streamlit as st
import pandas as pd
import joblib

# --------------------- PAGE SETUP ---------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Card Default Risk Predictor")
st.markdown("##### Real-time risk assessment using bank-grade model")

# --------------------- LOAD YOUR SAVED MODEL ---------------------
@st.cache_resource
def load_model():
    return joblib.load("best_credit_risk_model.pkl")  # ← your model from notebook

model = load_model()

# --------------------- SIDEBAR: USER INPUT (only original columns) ---------------------
st.sidebar.header("Customer Profile")

limit_bal = st.sidebar.slider("Credit Limit (NT$)", 10_000, 1_000_000, 200_000, step=10_000)
age = st.sidebar.slider("Age", 20, 80, 35)

education = st.sidebar.selectbox("Education Level",
    options=[1, 2, 3, 4, 5, 6],
    format_func=lambda x: {1:"Graduate", 2:"University", 3:"High School", 4:"Others"}.get(x, "Unknown"))

marriage = st.sidebar.selectbox("Marital Status",
    options=[1, 2, 3],
    format_func=lambda x: {1:"Married", 2:"Single", 3:"Others"}.get(x, "Unknown"))

sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")

# Repayment status (most important features!)
pay_0 = st.sidebar.selectbox("Repayment Last Month", options=range(-2, 9),
                            format_func=lambda x: "On time/Paid early" if x <= 0 else f"{x} months late")
pay_2 = st.sidebar.selectbox("Repayment 2 Months Ago", options=range(-2, 9),
                            format_func=lambda x: "On time/Paid early" if x <= 0 else f"{x} months late")
pay_3 = st.sidebar.selectbox("Repayment 3 Months Ago", options=range(-2, 9),
                            format_func=lambda x: "On time/Paid early" if x <= 0 else f"{x} months late")

# Bill and payment amounts
bill_amt1 = st.sidebar.number_input("Bill Amount Last Month (NT$)", 0, 800_000, 50_000)
bill_amt2 = st.sidebar.number_input("Bill Amount 2 Months Ago (NT$)", 0, 800_000, 48_000)
pay_amt1 = st.sidebar.number_input("Amount Paid Last Month (NT$)", 0, 500_000, 3000)
pay_amt2 = st.sidebar.number_input("Amount Paid 2 Months Ago (NT$)", 0, 500_000, 3000)

# --------------------- CREATE INPUT DATAFRAME (exact same columns as training) ---------------------
input_data = pd.DataFrame({
    'LIMIT_BAL': [limit_bal],
    'SEX': [sex],
    'EDUCATION': [education],
    'MARRIAGE': [marriage],
    'AGE': [age],
    'PAY_0': [pay_0],
    'PAY_2': [pay_2],
    'PAY_3': [pay_3],
    'BILL_AMT1': [bill_amt1],
    'BILL_AMT2': [bill_amt2],
    'PAY_AMT1': [pay_amt1],
    'PAY_AMT2': [pay_amt2],
    # Add more PAY_x, BILL_AMTx, PAY_AMTx if you used them in training!
})

# --------------------- PREDICTION ---------------------
probability = model.predict_proba(input_data)[0][1]
score = int(np.clip((1 - probability) * 1000, 300, 850))  # FICO-style 300–850

# --------------------- DISPLAY RESULTS ---------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("Default Probability", f"{probability:.1%}")
    st.metric("Credit Score (approx)", score)

with col2:
    if probability < 0.2:
        risk = "Low Risk"
        color = "green"
    elif probability < 0.5:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"
    st.markdown(f"<h2 style='color:{color};'>Risk: {risk}</h2>", unsafe_allow_html=True)

# --------------------- SIMPLE RISK DRIVERS (no SHAP needed) ---------------------
st.markdown("### Top Risk Indicators")
risk_reasons = []

if pay_0 > 0: risk_reasons.append(f"Recent payment {pay_0} month(s) late")
if pay_2 > 0: risk_reasons.append(f"Payment delay 2 months ago")
if pay_3 > 0: risk_reasons.append("History of late payments")
if limit_bal < 100_000: risk_reasons.append("Low credit limit")
if (pay_amt1 + pay_amt2) < 5000: risk_reasons.append("Very low recent payments")

if not risk_reasons:
    risk_reasons = ["Strong repayment history", "Healthy credit limit", "Consistent payments"]

for reason in risk_reasons[:4]:
    st.write("• " + reason)

st.markdown("---")
st.caption("Model trained on 30,000 real credit records | AUC 0.81–0.82 | Built by [Your Name]")