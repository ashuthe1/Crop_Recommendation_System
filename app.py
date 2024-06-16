import streamlit as st
import pickle
import pandas as pd
from nav import home, predict, visualize
from nav import logs  # Import logs module

# Load the precomputed models, metrics, and label encoder
models_file = 'precomputation/models.pkl'
metrics_file = 'precomputation/metrics.pkl'
le_file = 'precomputation/label_encoder.pkl'

with open(models_file, 'rb') as f:
    models = pickle.load(f)

with open(metrics_file, 'rb') as f:
    metrics = pickle.load(f)

with open(le_file, 'rb') as f:
    le = pickle.load(f)

# Load the dataset for feature names
dataset_path = 'dataset/crop_recommendation.csv'
df = pd.read_csv(dataset_path)
X = df.drop('label', axis=1)

# Custom CSS for better navigation bar
st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: space-around;
        background-color: #f8f9fa;
        padding: 10px;
        border-bottom: 2px solid #dee2e6;
    }
    .navbar a {
        text-decoration: none;
        color: #333;
        font-weight: bold;
    }
    .navbar a:hover {
        color: #007bff;
    }
    </style>
""", unsafe_allow_html=True)

# Create navigation bar
st.markdown("""
    <div class="navbar">
        <a href="?page=home" id="link-home">Home</a>
        <a href="?page=predict" id="link-predict">Predict</a>
        <a href="?page=visualize" id="link-visualize">Visualize</a>
        <a href="?page=logs" id="link-logs">Logs</a>  <!-- Link to Logs page -->
    </div>
""", unsafe_allow_html=True)

# Determine the selected page based on the query parameters
query_params = st.query_params
page = query_params.get("page", "home")

if page == "home":
    home.show()
elif page == "predict":
    predict.show(models, metrics, le, X.columns)
elif page == "visualize":
    visualize.show(models, metrics, le, X)
elif page == "logs":  # Display Logs page when 'logs' is selected
    logs.show_logs()
else:
    home.show()
