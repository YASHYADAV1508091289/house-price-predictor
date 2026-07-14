"""
Streamlit demo - run AFTER train.py has created model.pkl/scaler.pkl
Run: streamlit run app.py
This is the screen you record for your LinkedIn video.
"""
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("🏠 California House Price Predictor")
st.caption("Gradient Boosting model trained on the California Housing dataset")

st.sidebar.header("Input Features")
med_inc = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 5.0)
house_age = st.sidebar.slider("House Age (years)", 1, 52, 20)
ave_rooms = st.sidebar.slider("Avg Rooms per Household", 1.0, 15.0, 5.0)
ave_bedrms = st.sidebar.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.0)
population = st.sidebar.slider("Block Population", 3, 35000, 1500)
ave_occup = st.sidebar.slider("Avg Occupants per Household", 0.5, 10.0, 3.0)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 34.0)
longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -118.0)

rooms_per_household = ave_rooms / ave_occup
bedrooms_ratio = ave_bedrms / ave_rooms
pop_per_household = population / ave_occup

row = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup,
                  latitude, longitude, rooms_per_household, bedrooms_ratio, pop_per_household]])

row_scaled = scaler.transform(row)
pred = model.predict(row_scaled)[0]

st.metric("Predicted Median House Value", f"${pred * 100000:,.0f}")

with st.expander("See raw feature vector"):
    st.write(dict(zip(features, row[0])))