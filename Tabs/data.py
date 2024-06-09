# Import necessary modules
import streamlit as st

def app(df):
    """This function creates the data info page"""

    # Add title to the data info page
    st.title("Data Information")

    # Show the data
    st.subheader("Dataset")
    st.write(df.head())

    # Show the description
    st.subheader("Description")
    st.write(df.describe())

    # Show the distribution of the label column
    st.subheader("Label Distribution")
    st.write(df['label'].value_counts())
