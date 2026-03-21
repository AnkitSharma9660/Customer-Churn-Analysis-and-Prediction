import streamlit as st
import numpy as np
import pickle

# Load models
rf = pickle.load(open("rf_churn_model.pkl","rb"))
gb = pickle.load(open("gb_churn_model.pkl","rb"))

st.title("📞 Telecom Customer Churn Prediction")

st.subheader("Enter Customer Details")

# Numerical Inputs
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

# Auto calculate Total Charges
total = tenure * monthly
st.write(f"Estimated Total Charges: **{total:.2f}**")

# Binary Inputs
partner = st.selectbox("Partner", [0,1])
dependents = st.selectbox("Dependents", [0,1])
phoneservice = st.selectbox("Phone Service", [0,1])
paperless = st.selectbox("Paperless Billing", [0,1])
senior = st.selectbox("Senior Citizen", [0,1])

# Contract Encoding
contract = st.selectbox("Contract Type",
                        ["Month-to-month","One year","Two year"])

contract_map = {
    "Month-to-month":[1,0],
    "One year":[0,1],
    "Two year":[0,0]
}

# Internet Service Encoding
internet = st.selectbox("Internet Service",
                        ["Fiber optic","DSL","No"])

internet_map = {
    "Fiber optic":[1,0],
    "DSL":[0,1],
    "No":[0,0]
}

# Prediction Button
if st.button("Predict Churn"):

    features = np.array([[tenure, monthly, total,
                          partner, dependents,
                          phoneservice, paperless, senior,
                          contract_map[contract][0],
                          contract_map[contract][1],
                          internet_map[internet][0],
                          internet_map[internet][1]]])

    rf_pred = rf.predict(features)[0]
    gb_pred = gb.predict(features)[0]

    st.subheader("Prediction Result")

    st.write("Random Forest:",
             "Customer may churn" if rf_pred==1 else "Customer likely to stay")

    st.write("Gradient Boosting:",
             "Customer may churn" if gb_pred==1 else "Customer likely to stay")