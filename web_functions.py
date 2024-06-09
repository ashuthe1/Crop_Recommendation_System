# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, r2_score

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

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    """This function calculates accuracy, precision, and R2 score"""
    # Check if there are predictions made
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0, 0, 0

    # Convert y_true and y_pred to numeric if possible
    y_true_numeric = pd.to_numeric(y_true, errors='coerce')
    y_pred_numeric = pd.to_numeric(y_pred, errors='coerce')

    # Drop NaN values
    y_true_numeric = y_true_numeric[~np.isnan(y_true_numeric)]
    y_pred_numeric = y_pred_numeric[~np.isnan(y_pred_numeric)]

    # Check if there are any samples after dropping NaN values
    if len(y_true_numeric) == 0 or len(y_pred_numeric) == 0:
        return 0, 0, 0

    # Calculate metrics
    accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
    precision = precision_score(y_true_numeric, y_pred_numeric, average='weighted')
    r2 = r2_score(y_true_numeric, y_pred_numeric)

    return accuracy, precision, r2
