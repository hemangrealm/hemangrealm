import streamlit as st 
import pandas as pd
import numpy as np
import joblib
model = joblib.load('house_model.pk1')
all_features = joblib.load('all_features.pk1')
st.title("House Prediction App")
st.write("Enter the details of the house to predict its price.")

lot_area= st.number_input("Lot Area (in square feet):",min_value=100, max_value=100000, value=5000)
overall_qual=st.slider_input("Overall Quality (1-10):", min_value=1, max_value=10, value=5)
year_built=st.number_input("Year Built:", min_value=1800,max_value=2024, value=2000)
total_bsmt_sf =st.number_input("Total Basement Area (in square feet):", min_value=0, value=1000)

# Create a DataFrame with the input features
gr_liv_area = st.number_input("Above Grade Living Area (in square feet):", min_value=0, value=1500)
user_input = {
    'LotArea': lot_area,
    'Overall Qual': overall_qual,
    'Year Built': year_built,
    'Total Bsmt SF': total_bsmt_sf,
    'GrLivArea': gr_liv_area
}

# Convert user input to 301 feature df
input_df = pd.DataFrame(np.zeros((1,len(all_features))), columns=all_features)

for key, value in user_input.items():
    if key in input_df.columns:
        input_df.at[0, key] = value

# Prediction
if st.button("Predict price"):
    try:
        predicted_price = model.predict(input_df)[0]
        st.success(f"The predicted house price is: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occured during prediction: {e}")