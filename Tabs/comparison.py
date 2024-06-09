# Import necessary modules
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from web_functions import load_data, predict, calculate_metrics  # Add import statement for load_data and predict

def app(df, X, y, model_types):
    """This function creates the comparison page"""

    # Add title to the page
    st.title("Model Comparison")

    # Create empty lists to store metrics
    accuracies = []
    precisions = []
    r2_scores = []

    X_test, y_test = X, y  # Use X and y directly from the arguments


    # Calculate metrics for each model
    for model_type in model_types:
        predictions = predict(model_type, X_test)
        accuracy, precision, r2 = calculate_metrics(y_test, predictions)
        accuracies.append(accuracy)
        precisions.append(precision)
        r2_scores.append(r2)

    # Plot comparison graphs
    st.subheader("Model Performance Comparison")

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(model_types, accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison between Models')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Plot precision comparison
    plt.figure(figsize=(10, 6))
    plt.bar(model_types, precisions, color='salmon')
    plt.xlabel('Model')
    plt.ylabel('Precision')
    plt.title('Precision Comparison between Models')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Plot R2 score comparison
    plt.figure(figsize=(10, 6))
    plt.bar(model_types, r2_scores, color='lightgreen')
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.title('R2 Score Comparison between Models')
    plt.xticks(rotation=45)
    st.pyplot(plt)
