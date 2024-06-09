# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Load the dataset
def load_data():
    """This function returns the preprocessed data"""

    # Load the Crop Recommendation dataset into DataFrame.
    df = pd.read_csv('./dataset/crop_recommendation.csv')

    # Perform feature and target split
    X = df.drop('label', axis=1)  # Assuming 'label' is the target column
    y = df['label']

    return df, X, y

# Train the model
def train_model(model_type, X_train, y_train):
    """This function trains the specified model and returns the trained model"""
    if model_type == 'Decision Tree Classification':
        model = DecisionTreeClassifier()
    elif model_type == 'Random Forest Classification':
        model = RandomForestClassifier()
    elif model_type == 'K-Nearest Neighbor(KNN)':
        model = KNeighborsClassifier()
    elif model_type == 'Support Vector Machine(SVM)':
        model = SVC()

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, f'{model_type.replace(" ", "_").lower()}_model.pkl')

    return model

# Predict using the trained model
def predict(model_type, X_test):
    """This function predicts the labels using the specified trained model"""
    # Load the trained model
    model = joblib.load(f'{model_type.replace(" ", "_").lower()}_model.pkl')

    # Predict the labels
    predictions = model.predict(X_test)

    return predictions
