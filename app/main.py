import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import os
import gdown

def main():

    st.set_page_config(
        page_title= "Car Price predictor"
    )


    # model = pk.load(open("../model/model.pkl", "rb"))

    MODEL_PATH = "model.pkl"

    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1Iaw2JXeCLj9jzisi3PSTb2DWunr1RFuY"
        gdown.download(url, MODEL_PATH, quiet=False)

    with open(MODEL_PATH, "rb") as f:
        model = pk.load(f)

    st.title("Car Price Predictor")

    st.write("Enter car details to predict price")

    year = st.number_input("Production year", 1990,2025)
    mileage = st.number_input("Mileage (km)", 0, 500000)
    levy = st.number_input("Levy", 0, 20000)
    engine_volume = st.number_input("Engine Volume", 0.5, 6.0)
    leather = st.selectbox("Leather Interior", ["Yes", "No"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])

    # Convert inputs to dataframe
    input_data = pd.DataFrame({
        'Prod. year': [year],
        'Mileage': [mileage],
        'Levy': [levy],
        'Engine volume': [engine_volume],
        'Leather interior': [1 if leather == "Yes" else 0],
        'Fuel type': [fuel_type]
    })

    # Handle categorical variables
    input_data = pd.get_dummies(input_data)

    # Align with training columns
    model_columns = pk.load(open("../model/model_columns.pkl", "rb"))

    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    input_data = input_data[model_columns]

    # Prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Price: ${prediction:,.2f}")

if __name__ == "__main__":
    main()