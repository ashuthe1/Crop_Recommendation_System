# Importing the necessary Python modules.
import streamlit as st

# Import necessary functions from web_functions
from web_functions import load_data, train_model

# Import pages
from Tabs import home, data, predict, visualise

# Configure the app
st.set_page_config(
    page_title='Crop Recommendation System',
    page_icon='🌾',
    layout='wide',
    initial_sidebar_state='auto'
)

activities = ['Decision Tree Classification', 'Random Forest Classification', 'K-Nearest Neighbor(KNN)', 'Support Vector Machine(SVM)']
option = st.sidebar.selectbox('Which model would you like to use?', activities)
st.subheader(option)

# Dictionary for pages
Tabs = {
    "Home": home,
    "Data Info": data,
    "Prediction": predict,
    "Visualisation": visualise
}

# Create a sidebar
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Load the dataset.
df, X, y = load_data()

# Call the app function of selected page to run
if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, X, y, option)
elif page == "Data Info":
    Tabs[page].app(df)
else:
    Tabs[page].app()
