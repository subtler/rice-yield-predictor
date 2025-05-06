import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Rice_Mill_3-Month_Sample_Dataset.csv")  # Now reads from app/ folder
    return df

df = load_data()  # Optional: use it for data insights later

# Load trained model
model = joblib.load('rice_yield_predictor_rf.pkl')  # Also in app/ folder

# App title
st.title("🌾 Rice Yield Predictor (AI-Powered)")
st.markdown("Predict rice output (kg) based on batch input using machine learning")

# Input form
with st.form("yield_form"):
    paddy_type = st.selectbox("Paddy Type", ["CR 1009 Sub 1", "ADT 43", "Iluppai Poo Samba", "Karuppu Kavuni", "CO 51"])
    source = st.selectbox("Source District", ["Tirunelveli", "Thiruvarur"])
    season = st.selectbox("Season", ["Kuruvai", "Samba", "Navarai"])
    moisture = st.slider("Moisture (%)", 13.0, 15.5, 14.0, 0.1)
    input_weight = st.slider("Input Weight (kg)", 800, 1200, 1000)
    milling_time = st.slider("Milling Duration (min)", 30, 60, 45)

    submitted = st.form_submit_button("Predict Yield")

# Predict
if submitted:
    user_input = [[
        moisture,
        input_weight,
        milling_time,
        paddy_type,
        source,
        season
    ]]

    # Prepare input as DataFrame
    input_df = pd.DataFrame(user_input, columns=[
        "Moisture (%)", "Input Weight (kg)", "Milling Duration (min)",
        "Paddy Type", "Source District", "Season"
    ])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Estimated Rice Output: **{int(prediction)} kg**")