import streamlit as st
import pandas as pd
import joblib


# Load model and scaler
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="💰 Salary Predictor", layout="centered")
st.title("💼 AI-Powered Salary Predictor")

st.markdown("Enter employee details (numeric only). Encoded fields like workclass, marital status, and occupation have been removed to avoid errors.")

# --- Numeric Inputs Only ---
age = st.sidebar.slider("Age", 18, 65, 30)
education_num = st.sidebar.slider("Education Level (1-16)", 1, 16, 10)
gender = st.sidebar.number_input("Gender (0 = Female, 1 = Male)", 0, 1, 1)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Experience (years)", 0, 40, 5)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 10000, 0)
attendance_percent = st.sidebar.slider("Attendance %", 0, 100, 90)
monthly_attendance = st.sidebar.slider("Monthly Attendance", 0, 31, 25)

# --- Derived Features ---
net_capital = capital_gain - capital_loss
overwork_flag = 1 if hours_per_week > 45 else 0

# --- Final Input Data ---
input_data = pd.DataFrame([[
    age, education_num, gender, capital_gain, capital_loss,
    hours_per_week, experience, net_capital, overwork_flag,
    attendance_percent, monthly_attendance
]], columns=[
    'age', 'educational-num', 'gender', 'capital-gain', 'capital-loss',
    'hours-per-week', 'experience_est', 'net_capital', 'overwork_flag',
    'attendance_percent', 'monthly_attendance'
])

# Scale features
scaled_input = scaler.transform(input_data)

# Display
st.write("### 🔍 Input Summary")
st.dataframe(input_data)

# Predict
if st.button("🔮 Predict Salary"):
    prediction = model.predict(scaled_input)
    st.success(f"🤑 Predicted Monthly Salary: ₹{prediction[0]:,.2f}")


   
