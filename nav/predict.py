import streamlit as st
import pandas as pd

def show(models, metrics, le, feature_names):
    st.title("Crop Recommendation")

    st.subheader("Select Model")
    selected_model = st.selectbox("Choose a model", list(models.keys()))

    if selected_model in metrics:
        st.write(f"Accuracy of {selected_model}: {metrics[selected_model]['accuracy'] * 100:.4f}%")

    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Environmental Factors:")

    df = pd.read_csv('./dataset/crop_recommendation.csv')
    # Take input of features from the user.
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

        st.success(f"Recommended Crop: {predicted_crop}")
