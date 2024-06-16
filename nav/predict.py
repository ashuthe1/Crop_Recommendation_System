import streamlit as st
import pandas as pd
import sqlite3
from database import log_prediction

# Function to show prediction form
def show(models, metrics, le, feature_names):
    st.title("Crop Recommendation")

    st.subheader("Select Model")
    selected_model = st.selectbox("Choose a model", list(models.keys()))

    if selected_model in metrics:
        st.write(f"Accuracy of {selected_model}: {metrics[selected_model]['accuracy'] * 100:.4f}%")

    # Take feature input from the user
    st.subheader("Select Environmental Factors:")

    df = pd.read_csv('./dataset/crop_recommendation.csv')
    N = st.slider("Nitrogen (N)", int(df["N"].min()), int(df["N"].max()))
    P = st.slider("Phosphorus (P)", int(df["P"].min()), int(df["P"].max()))
    K = st.slider("Potassium (K)", int(df["K"].min()), int(df["K"].max()))
    temperature = st.slider("Temperature", float(df["temperature"].min()), float(df["temperature"].max()))
    humidity = st.slider("Humidity", float(df["humidity"].min()), float(df["humidity"].max()))
    ph = st.slider("pH", float(df["ph"].min()), float(df["ph"].max()))
    rainfall = st.slider("Rainfall", float(df["rainfall"].min()), float(df["rainfall"].max()))

    inputs = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    if st.button("Predict"):
        input_data = pd.DataFrame([inputs])

        model = models[selected_model]
        prediction = model.predict(input_data)
        predicted_crop = le.inverse_transform(prediction)[0]

        # Log prediction into database
        log_prediction(selected_model, inputs, predicted_crop)

        st.success(f"Recommended Crop: {predicted_crop}")

# Main function to execute when script runs directly
if __name__ == "__main__":
    # Ensure database initialization
    from database import initialize_database
    initialize_database()

    # Example of calling show() function directly
    # This is optional and depends on how your app is structured
    # and how you are testing/running it
    models = {}  # Replace with actual models
    metrics = {}  # Replace with actual metrics
    le = {}  # Replace with actual label encoder
    feature_names = []  # Replace with actual feature names
    show(models, metrics, le, feature_names)
