# FarmEasy

This project provides a machine learning-based crop recommendation system. The system uses environmental parameters such as nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall to predict the most suitable crop for a given set of conditions.

## Folder Structure

```
Crop-Recommendation-System/
│
├── dataset/
│   └── crop_recommendation.csv
│
├── images/
│   └── crops/
│       └── <crop_images>.jpg
│
├── models/
│   └── model_files.pkl
│
├── nav/
│   ├── home.py
│   ├── predict.py
│   └── visualize.py
│
├── app.py
├── train_model.py
├── database.py
├── requirements.txt
└── README.md
```

## CREATE Virtual Environment and Install Requirements

First, create a virtual environment to manage dependencies for this project.

```sh
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Application

To run the application, ensure your virtual environment is active, then execute the following commands:

```sh
source venv/bin/activate
python3 train_model.py
streamlit run app.py
```

## Files and Directories

- **`dataset/`**: Contains the `crop_recommendation.csv` dataset.
- **`images/crops/`**: Contains images of the recommended crops.
- **`models/`**: Directory where model files are stored after training.
- **`nav/`**: Contains navigation modules for the Streamlit application.
  - **`home.py`**: Home page of the application.
  - **`predict.py`**: Prediction page for recommending crops.
  - **`visualize.py`**: Visualization page for displaying model metrics and visualizations.
- **`app.py`**: Main entry point for the Streamlit application.
- **`train_model.py`**: Script for training and saving machine learning models.
- **`database.py`**: Handles logging predictions to the database.
- **`requirements.txt`**: Lists required Python packages for the project.
- **`README.md`**: Project documentation file.

## Additional Information

- **Model Training**: The `train_model.py` script trains three models (Logistic Regression, Decision Tree, and Random Forest) and saves the trained models, metrics, and label encoder in the `models/` directory.
  - **Logistic Regression**: A linear model for binary classification. 
  - **Decision Tree**: A non-linear model that splits the data based on feature values.
  - **Random Forest**: An ensemble model that combines multiple decision trees to improve accuracy and control overfitting.

- **Model Prediction**: The `predict.py` script uses the trained models to recommend the most suitable crop based on user input parameters.
- **Visualization**: The `visualize.py` script provides visualizations for the model metrics and decision tree.

## Model Metrics

The following metrics are used to evaluate the performance of the models:

- **Accuracy**: The ratio of correctly predicted crops to the total crops.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positive observations.
- **R^2 Score**: The proportion of the variance in the dependent variable that is predictable from the independent variables.

After training the models, the metrics are as follows:

- **Logistic Regression**
  - Accuracy: `95.42%`
  - Precision: `96.04%`
  - R^2 Score: `89.67`
  
- **Decision Tree**
  - Accuracy: `98.63%`
  - Precision: `98.71%`
  - R^2 Score: `96.20`
  
- **Random Forest**
  - Accuracy: `99.24%`
  - Precision: `99.34%`
  - R^2 Score: `97.3`
