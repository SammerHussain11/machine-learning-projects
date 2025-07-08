import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page configuration
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Enter car details below to predict the **price** of a car.")

# Integer inputs (manual entry possible)
enginesize = st.number_input("Engine Size (cc)", min_value=1, max_value=1000, step=1, format="%d")
curbweight = st.number_input("Curb Weight (kg)", min_value=500, max_value=6000, step=1, format="%d")
highwaympg = st.number_input("Highway MPG", min_value=5, max_value=100, step=1, format="%d")
horsepower = st.number_input("Horsepower", min_value=20, max_value=600, step=1, format="%d")
carwidth = st.number_input("Car Width (inches)", min_value=50, max_value=80, step=1, format="%d")

# Predict button
if st.button("Predict Price"):
    # Prepare input for model
    input_data = np.array([[enginesize, curbweight, highwaympg, horsepower, carwidth]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    
    # Show result
    st.success(f"💰 Estimated Car Price: **${prediction:,.2f}**")
