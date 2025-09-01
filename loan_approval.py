import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load models, encoders, and scalers
models_path = "Models"
LightGBM_model = joblib.load(os.path.join(models_path, "lightgbm_model.pkl"))
scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(models_path, "label_encoders.pkl"))

def main():
    st.title("Loan Approval Prediction")
    st.write("Enter the applicant details to predict loan approval status.")

    # Input fields in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
        person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, value=5.0, step=0.5)
        loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    
    with col2:
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E"])
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=100)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=7.5, step=0.1)
        loan_percent_income = st.number_input("Loan/Income Ratio", min_value=0.0, value=0.2, step=0.01)
        cb_person_default_on_file = st.selectbox("Past Default", ["N", "Y"], help="Has the borrower defaulted before?")
        cb_person_cred_hist_length = st.number_input("Credit History (Years)", min_value=0, value=5, step=1)
        age_group = st.selectbox("Age Group", ["Young-Adult", "Adult"])

    # Predict button
    if st.button("Predict Loan Approval"):
        # Encode categorical features
        encoded_person_home_ownership = label_encoders['person_home_ownership'].transform([person_home_ownership])[0]
        encoded_loan_intent = label_encoders['loan_intent'].transform([loan_intent])[0]
        encoded_loan_grade = label_encoders['loan_grade'].transform([loan_grade])[0]
        encoded_cb_person_default_on_file = label_encoders['cb_person_default_on_file'].transform([cb_person_default_on_file])[0]
        encoded_age_group = label_encoders['age_group'].transform([age_group])[0]

        # Construct the input data
        input_data = np.array([
            person_age,
            person_income,
            encoded_person_home_ownership,
            person_emp_length,
            encoded_loan_intent,
            encoded_loan_grade,
            loan_amnt,
            loan_int_rate,
            loan_percent_income,
            encoded_cb_person_default_on_file,
            cb_person_cred_hist_length,
            encoded_age_group
        ]).reshape(1, -1)  # Reshape to 2D array

        # Scale the input data
        scaled_input_data = scaler.transform(input_data)

        # Make prediction and get probabilities
        probabilities = LightGBM_model.predict_proba(scaled_input_data)[0]  # Get probabilities for the first sample
        prob_reject = probabilities[0] * 100  # Probability of loan rejection
        prob_approve = probabilities[1] * 100  # Probability of loan approval

        # Display result
        st.write(f"Probability of Loan Rejection: {prob_reject:.2f}%")
        st.write(f"Probability of Loan Approval: {prob_approve:.2f}%")

        if prob_approve > prob_reject:
            st.balloons()
            st.success(f"✅ Loan Approved! (Confidence: {prob_approve:.2f}%)")
        else:
            st.error(f"❌ Loan Rejected (Confidence: {prob_reject:.2f}%)")

if __name__ == "__main__":
    main()