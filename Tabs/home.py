# Import necessary modules
import streamlit as st

def app():
    """This function creates the home page"""
    
    # Add title to the home page
    st.title("Welcome to the Crop Recommendation System")
    
    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This application helps farmers and agricultural enthusiasts to get crop recommendations based on various environmental factors using different machine learning algorithms.
            </p>
        """, unsafe_allow_html=True)
