import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Agribot - Crop Prediction",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Agribot - Crop Prediction System")
st.write(
    "Enter the soil and environmental conditions below "
    "to predict a suitable crop."
)

# Sample agricultural dataset
data = {
    "N": [90, 85, 60, 74, 78, 69, 55, 45, 40, 30, 20, 95],
    "P": [42, 58, 55, 35, 42, 45, 40, 50, 60, 45, 55, 38],
    "K": [43, 41, 44, 40, 42, 35, 30, 25, 20, 35, 30, 45],
    "temperature": [20, 22, 25, 28, 24, 30, 27, 23, 21, 26, 19, 29],
    "humidity": [80, 82, 78, 65, 70, 75, 68, 72, 85, 60, 88, 66],
    "ph": [6.5, 6.8, 6.2, 7.0, 6.7, 6.0, 6.4, 5.8, 6.9, 7.2, 5.9, 6.6],
    "rainfall": [200, 220, 180, 150, 170, 190, 160, 140, 210, 120, 230, 155],
    "crop": [
        "Rice", "Rice", "Wheat", "Maize",
        "Maize", "Cotton", "Cotton", "Groundnut",
        "Rice", "Wheat", "Rice", "Maize"
    ]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop("crop", axis=1)
y = df["crop"]

# Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.subheader("🌾 Enter Agricultural Conditions")

N = st.number_input(
    "Nitrogen (N)",
    min_value=0.0,
    max_value=150.0,
    value=50.0
)

P = st.number_input(
    "Phosphorus (P)",
    min_value=0.0,
    max_value=150.0,
    value=40.0
)

K = st.number_input(
    "Potassium (K)",
    min_value=0.0,
    max_value=150.0,
    value=40.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

ph = st.number_input(
    "Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=6.5
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=150.0
)

if st.button("🌱 Predict Suitable Crop"):

    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Recommended Crop: **{prediction}**")

    st.info(
        "The recommendation is generated using a "
        "Random Forest machine learning model."
    )
