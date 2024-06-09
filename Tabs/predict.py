# Import necessary modules
import streamlit as st
from web_functions import load_data, train_model, predict

def app(df, X, y, model_type):
    """This function creates the prediction page"""

    # Add title to the page
    st.title("Crop Recommendation System")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app recommends crops based on various environmental factors using different machine learning algorithms.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Environmental Factors:")

    # Take input of features from the user.
    N = st.slider("Nitrogen (N)", int(df["N"].min()), int(df["N"].max()))
    P = st.slider("Phosphorus (P)", int(df["P"].min()), int(df["P"].max()))
    K = st.slider("Potassium (K)", int(df["K"].min()), int(df["K"].max()))
    temperature = st.slider("Temperature", float(df["temperature"].min()), float(df["temperature"].max()))
    humidity = st.slider("Humidity", float(df["humidity"].min()), float(df["humidity"].max()))
    ph = st.slider("pH", float(df["ph"].min()), float(df["ph"].max()))
    rainfall = st.slider("Rainfall", float(df["rainfall"].min()), float(df["rainfall"].max()))

    # Create a button to predict
    if st.button("Predict"):
        # Train the model with the selected algorithm
        model = train_model(model_type, X, y)
        # Predict using the trained model
        prediction = predict(model_type, [[N, P, K, temperature, humidity, ph, rainfall]])

        # Print predictions
        st.write(f"Prediction using {model_type}:", prediction[0])

        st.info("Prediction made successfully.")
