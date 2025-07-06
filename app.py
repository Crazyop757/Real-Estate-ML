import streamlit as st
import pandas as pd
import joblib

#  Loading regression model and features 
reg_model = joblib.load("models/best_random_forest_model.pkl")
reg_features = joblib.load("models/model_features.pkl")

st.set_page_config(page_title="🏠 Real Estate Price Insight", layout="centered")
st.title("🏠 Real Estate Price Insight")
st.markdown("Predict the estimated property price and check if your price is **fair or unfair** (optional).")

#Input Form 
with st.form("price_form"):
    total_sqft = st.number_input("📐 Total Sqft", min_value=100, step=50, value=1200)
    bath = st.number_input("🚿 Number of Bathrooms", min_value=1, step=1, value=2)
    bhk = st.number_input("🛏️ Number of BHK", min_value=1, step=1, value=2)

    actual_price = st.text_input("💰 Actual Price (Lakh) [optional]")

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_reg = pd.DataFrame([0] * len(reg_features), index=reg_features).T
    input_reg.at[0, 'total_sqft'] = total_sqft
    input_reg.at[0, 'bath'] = bath
    input_reg.at[0, 'bhk'] = bhk

    predicted_price = reg_model.predict(input_reg)[0]
    st.success(f"📊 Predicted Price: ₹ {round(predicted_price, 2)} Lakh")

    if actual_price.strip() != "":
        try:
            actual_price_val = float(actual_price)
            diff = (predicted_price - actual_price_val) / actual_price_val
            st.info(f"🧾 Actual Price: ₹ {actual_price_val} Lakh")
            st.info(f"📉 Price Difference: {round(diff * 100, 2)}%")

            if abs(diff) <= 0.15:
                st.success("Classified as: **FAIR Deal**")
            else:
                st.error("Classified as: **UNFAIR Deal**")

        except ValueError:
            st.error("Invalid number entered for Actual Price. Please enter a numeric value.")
    else:
        st.info("No actual price provided, so fairness classification skipped.")
