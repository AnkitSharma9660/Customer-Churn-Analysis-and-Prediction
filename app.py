import streamlit as st
import numpy as np
import pickle

# ===== Load Models =====
rf = pickle.load(open("rf_churn_model.pkl","rb"))
gb = pickle.load(open("gb_churn_model.pkl","rb"))

st.title("📞 Telecom Customer Churn Prediction System")

st.subheader("Customer Demographic Information")

gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0,1])
senior = st.selectbox("Senior Citizen", [0,1])
partner = st.selectbox("Partner", [0,1])
dependents = st.selectbox("Dependents", [0,1])

st.subheader("Customer Subscription Details")

tenure = st.number_input("Tenure (Months)", 0, 72, 1)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

total = tenure * monthly
st.write(f"Estimated Total Charges: {total:.2f}")

paperless = st.selectbox("Paperless Billing", [0,1])

# ===== Phone Dependency =====
phone = st.selectbox("Phone Service", [0,1])

if phone == 1:
    multiple = st.selectbox("Multiple Lines", [0,1])
else:
    multiple = 0
    st.write("Multiple Lines auto set to 0")

# ===== Internet Dependency =====
internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

if internet == "No":
    dsl = 0
    fiber = 0
    no_internet = 1

    security = 0
    backup = 0
    device = 0
    support = 0
    tv = 0
    movies = 0

    st.write("All internet based services auto set to 0")

else:
    dsl = 1 if internet=="DSL" else 0
    fiber = 1 if internet=="Fiber optic" else 0
    no_internet = 0

    security = st.selectbox("Online Security", [0,1])
    backup = st.selectbox("Online Backup", [0,1])
    device = st.selectbox("Device Protection", [0,1])
    support = st.selectbox("Tech Support", [0,1])
    tv = st.selectbox("Streaming TV", [0,1])
    movies = st.selectbox("Streaming Movies", [0,1])

# ===== Contract Dummy =====
contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])

c_month = 1 if contract=="Month-to-month" else 0
c_year = 1 if contract=="One year" else 0
c_two = 1 if contract=="Two year" else 0

# ===== Payment Dummy =====
payment = st.selectbox("Payment Method",
                       ["Bank transfer (automatic)",
                        "Credit card (automatic)",
                        "Electronic check",
                        "Mailed check"])

p_bank = 1 if payment=="Bank transfer (automatic)" else 0
p_card = 1 if payment=="Credit card (automatic)" else 0
p_elec = 1 if payment=="Electronic check" else 0
p_mail = 1 if payment=="Mailed check" else 0

# ===== Prediction =====
if st.button("Predict Churn"):

    features = np.array([[gender, senior, partner, dependents,
                          tenure, phone, multiple,
                          security, backup, device, support,
                          tv, movies, paperless,
                          monthly, total,
                          dsl, fiber, no_internet,
                          c_month, c_year, c_two,
                          p_bank, p_card, p_elec, p_mail]])

    rf_pred = rf.predict(features)[0]
    gb_pred = gb.predict(features)[0]

    st.subheader("Prediction Result")

    st.write("🌲 Random Forest:",
             "⚠️ Customer may churn" if rf_pred==1 else "✅ Customer likely to stay")

    st.write("🚀 Gradient Boosting:",
             "⚠️ Customer may churn" if gb_pred==1 else "✅ Customer likely to stay")
