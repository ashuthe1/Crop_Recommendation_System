import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_data_analysis(df):
    st.title("Data Analysis")

    st.subheader("Dataset Description")
    st.write(df.describe())

    st.subheader("Correlation Plot")
    plot_correlation(df)

def plot_correlation(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_title("Correlation Matrix")

    st.pyplot(fig)

def main():
    dataset_path = 'dataset/crop_recommendation.csv'
    df = pd.read_csv(dataset_path)

    show_data_analysis(df)

if __name__ == "__main__":
    main()
