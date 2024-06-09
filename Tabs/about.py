# Import necessary modules
import streamlit as st

def app():
    """This function creates the about page"""

    # Add title to the page
    st.title("About")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app is created as a part of the MLG Project for predicting Cardiac Arrest.
            </p>
        """, unsafe_allow_html=True)
