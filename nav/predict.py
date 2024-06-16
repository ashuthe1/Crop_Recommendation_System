import streamlit as st
import pandas as pd
from database import log_prediction

def show(models, metrics, le, feature_names):
    st.title("Crop Recommendation")

    # Model selection and accuracy display
    st.subheader("Select Model")
    selected_model = st.selectbox("Choose a model", list(models.keys()))

    if selected_model in metrics:
        st.write(f"Accuracy of {selected_model}: {metrics[selected_model]['accuracy'] * 100:.2f}%")

    st.markdown("<h2 style='text-align: center; padding-bottom: 20px;'>🌤️Enter Environmental Parameters❄️</h2>", unsafe_allow_html=True)

    # User input section
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("N (Nitrogen)", min_value=0.0, max_value=120.0, value=50.0, step=0.1, format="%.1f")
        P = st.slider("P (Phosphorus)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f")
        K = st.slider("K (Potassium)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f")
        rainfall = st.slider("Rainfall", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.1f")
        humidity = st.slider("Humidity", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f")
        ph = st.slider("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1, format="%.1f")

    # Predict button
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        if st.button("Predict", key="predict_button"):
            input_data = {
                'N': N, 'P': P, 'K': K,
                'temperature': temperature, 'humidity': humidity, 'ph': ph,
                'rainfall': rainfall
            }

            # Perform prediction using selected model
            input_df = pd.DataFrame([input_data])
            model = models[selected_model]
            prediction = model.predict(input_df)
            predicted_crop = le.inverse_transform(prediction)[0]

            # Log prediction into database
            log_prediction(selected_model, input_data, predicted_crop)

            # Display prediction result
            st.success(f"Recommended Crop: {predicted_crop}")

            # Display image of the recommended crop (assuming images are stored in an "images" folder)
            image_path = f"images/crops/{predicted_crop.lower()}.jpg"  # adjust filename as per your image naming convention
            st.image(image_path, caption=predicted_crop, use_column_width=True)

        # Apply CSS to set the width of the button
        st.markdown(
            """
            <style>
                div[data-testid="stButton"] > button {
                    width: 200px;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    # Initialize any necessary components (e.g., models, metrics, le)
    models = {}
    metrics = {}
    le = {}  # Replace with actual label encoder
    feature_names = []  # Replace with actual feature names
    show(models, metrics, le, feature_names)
