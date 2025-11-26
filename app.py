# app.py  ← Save as app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------- PAGE SETUP ---------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Card Default Risk Predictor")
st.markdown("##### Real-time risk assessment using bank-grade model")

# --------------------- LOAD YOUR SAVED MODEL ---------------------
@st.cache_resource
def load_model():
    return joblib.load("best_credit_risk_model.pkl")

model = load_model()

# --------------------- SIDEBAR: USER INPUT ---------------------
st.sidebar.header("Customer Profile")

credit_limit = st.sidebar.slider("Credit Limit (NT$)", 10_000, 1_000_000, 200_000, step=10_000)
age = st.sidebar.slider("Age", 20, 80, 35)

education = st.sidebar.selectbox("Education Level",
    options=[1, 2, 3, 4, 5, 6],
    format_func=lambda x: {1:"Graduate", 2:"University", 3:"High School", 4:"Others"}.get(x, "Unknown"))

marital_status = st.sidebar.selectbox("Marital Status",
    options=[1, 2, 3],
    format_func=lambda x: {1:"Married", 2:"Single", 3:"Others"}.get(x, "Unknown"))

gender = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")

# Repayment status (most important features!)
repay_status_sep = st.sidebar.selectbox("Repayment Last Month (Sep)", options=range(-2, 9),
                            format_func=lambda x: "On time/Paid early" if x <= 0 else f"{x} months late")
repay_status_aug = st.sidebar.selectbox("Repayment 2 Months Ago (Aug)", options=range(-2, 9),
                            format_func=lambda x: "On time/Paid early" if x <= 0 else f"{x} months late")
repay_status_jul = st.sidebar.selectbox("Repayment 3 Months Ago (Jul)", options=range(-2, 9),
                            format_func=lambda x: "On time/Paid early" if x <= 0 else f"{x} months late")

# Bill and payment amounts
bill_amt_sep = st.sidebar.number_input("Bill Amount Last Month (NT$)", 0, 800_000, 50_000)
pay_amt_sep = st.sidebar.number_input("Amount Paid Last Month (NT$)", 0, 500_000, 3000)

# --------------------- CREATE INPUT DATAFRAME (MUST HAVE EXACT SAME 25 COLUMNS) ---------------------
input_data = pd.DataFrame({
    'credit_limit': [credit_limit],
    'gender': [gender],
    'education': [education],
    'marital_status': [marital_status],
    'age': [age],
    
    # Repayment status — fill missing months with 0 (on-time)
    'repay_status_sep': [repay_status_sep],
    'repay_status_aug': [repay_status_aug],
    'repay_status_jul': [repay_status_jul],
    'repay_status_jun': [0],
    'repay_status_may': [0],
    'repay_status_apr': [0],
    
    # Bill amounts — fill older months with reasonable defaults
    'bill_amt_sep': [bill_amt_sep],
    'bill_amt_aug': [bill_amt_sep * 0.95],
    'bill_amt_jul': [bill_amt_sep * 0.90],
    'bill_amt_jun': [bill_amt_sep * 0.85],
    'bill_amt_may': [bill_amt_sep * 0.80],
    'bill_amt_apr': [bill_amt_sep * 0.75],
    
    # Payment amounts — fill with realistic defaults
    'pay_amt_sep': [pay_amt_sep],
    'pay_amt_aug': [max(pay_amt_sep * 0.9, 2000)],
    'pay_amt_jul': [max(pay_amt_sep * 0.8, 2000)],
    'pay_amt_jun': [max(pay_amt_sep * 0.7, 2000)],
    'pay_amt_may': [max(pay_amt_sep * 0.6, 2000)],
    'pay_amt_apr': [max(pay_amt_sep * 0.5, 2000)],
})

# --------------------- FEATURE ENGINEERING (Same as notebook) ---------------------
# Credit Utilization
input_data['utilization_sep'] = (input_data['bill_amt_sep'] / input_data['credit_limit']) * 100
input_data['utilization_aug'] = (input_data['bill_amt_aug'] / input_data['credit_limit']) * 100
input_data['utilization_jul'] = (input_data['bill_amt_jul'] / input_data['credit_limit']) * 100
input_data['utilization_jun'] = (input_data['bill_amt_jun'] / input_data['credit_limit']) * 100
input_data['utilization_may'] = (input_data['bill_amt_may'] / input_data['credit_limit']) * 100
input_data['utilization_apr'] = (input_data['bill_amt_apr'] / input_data['credit_limit']) * 100

input_data['avg_utilization'] = (input_data['utilization_sep'] + input_data['utilization_aug'] + 
                                  input_data['utilization_jul'] + input_data['utilization_jun'] + 
                                  input_data['utilization_may'] + input_data['utilization_apr']) / 6

input_data['max_utilization'] = input_data[['utilization_sep', 'utilization_aug', 'utilization_jul', 
                                             'utilization_jun', 'utilization_may', 'utilization_apr']].max(axis=1)

# Payment Ratio
for month in ['sep', 'aug', 'jul', 'jun', 'may', 'apr']:
    input_data[f'payment_ratio_{month}'] = input_data[f'pay_amt_{month}'] / input_data[f'bill_amt_{month}']
    input_data.loc[input_data[f'bill_amt_{month}'] == 0, f'payment_ratio_{month}'] = 0

input_data['avg_payment_ratio'] = (input_data['payment_ratio_sep'] + input_data['payment_ratio_aug'] + 
                                    input_data['payment_ratio_jul'] + input_data['payment_ratio_jun'] + 
                                    input_data['payment_ratio_may'] + input_data['payment_ratio_apr']) / 6

input_data['min_payment_ratio'] = input_data[['payment_ratio_sep', 'payment_ratio_aug', 'payment_ratio_jul',
                                               'payment_ratio_jun', 'payment_ratio_may', 'payment_ratio_apr']].min(axis=1)

# Drop intermediate columns (same as notebook)
columns_to_drop = ['utilization_sep', 'utilization_aug', 'utilization_jul', 
                   'utilization_jun', 'utilization_may', 'utilization_apr',
                   'payment_ratio_sep', 'payment_ratio_aug', 'payment_ratio_jul',
                   'payment_ratio_jun', 'payment_ratio_may', 'payment_ratio_apr']
input_data = input_data.drop(columns=columns_to_drop)

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

# --------------------- SIMPLE RISK DRIVERS ---------------------
st.markdown("### Top Risk Indicators")
risk_reasons = []

if repay_status_sep > 0: 
    risk_reasons.append(f"Recent payment {repay_status_sep} month(s) late")
if repay_status_aug > 0: 
    risk_reasons.append(f"Payment delay 2 months ago")
if repay_status_jul > 0: 
    risk_reasons.append("History of late payments")
if credit_limit < 100_000: 
    risk_reasons.append("Low credit limit")
if pay_amt_sep < 2500: 
    risk_reasons.append("Very low recent payments")

# Add insights from engineered features
avg_util = input_data['avg_utilization'].values[0]
max_util = input_data['max_utilization'].values[0]
avg_pay_ratio = input_data['avg_payment_ratio'].values[0]

if max_util > 80:
    risk_reasons.append(f"High credit utilization ({max_util:.1f}%)")
elif max_util > 50:
    risk_reasons.append(f"Moderate credit utilization ({max_util:.1f}%)")

if avg_pay_ratio < 0.1:
    risk_reasons.append("Minimal payment history")

if not risk_reasons:
    risk_reasons = ["Strong repayment history", "Healthy credit limit", "Consistent payments"]

for reason in risk_reasons[:5]:
    st.write("• " + reason)

st.markdown("---")
st.caption("Model trained on 30,000 real credit records | AUC 0.7643 | Random Forest Classifier")