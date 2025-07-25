# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# === Paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "car_price_model.pkl")

# === Load trained model ===
model = joblib.load(model_path)

# === Title ===
st.title("🚗 Car Price Predictor")

# === Input fields ===
present_price = st.number_input("Showroom Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
car_age = st.slider("Car Age (Years)", min_value=0, max_value=30)

# === Manual Encoding ===
fuel_map = {'Petrol': 2, 'Diesel': 1, 'CNG': 0}
seller_map = {'Dealer': 0, 'Individual': 1}
trans_map = {'Manual': 1, 'Automatic': 0}

fuel_encoded = fuel_map[fuel_type]
seller_encoded = seller_map[seller_type]
trans_encoded = trans_map[transmission]

# === Predict button ===
if st.button("Predict Price"):
    input_data = np.array([[present_price, kms_driven, owner,
                            fuel_encoded, seller_encoded,
                            trans_encoded, car_age]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ {round(prediction, 2)} lakhs")
