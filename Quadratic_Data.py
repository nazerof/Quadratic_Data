import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.exceptions import NotFittedError

# Load the pre-trained model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load data from CSV
@st.cache_data
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data['x'].values.reshape(-1, 1), data['y'].values

# Plot data
def plot_data(x, y):
    plt.plot(x, y, 'b.', label='Data Points')

# Plot predictions
def plot_predictions(x, y_pred):
    plt.plot(x, y_pred, 'r-', label='Predicted Polynomial Regression')

def main():
    st.title('Quadratic Regression Model Visualization')

    model = load_model('model.pkl')  # Adjust path as needed
    x, y = load_data('data.csv')     # Adjust path as needed

    # Checkboxes to control plot display
    show_data = st.checkbox('Show Data')
    show_predictions = st.checkbox('Show Predictions')

    plt.figure(figsize=(8, 6))
    
    if show_data:
        plot_data(x, y)
    
    # Generate new x values for predictions
    x_new = np.linspace(0, 1, 100).reshape(-1, 1)
    
    # Create and fit polynomial features
    poly_features = PolynomialFeatures(degree=2)
    try:
        x_new_poly = poly_features.transform(x_new)
    except NotFittedError:
        poly_features.fit(x)
        x_new_poly = poly_features.transform(x_new)
    
    if show_predictions:
        y_pred = model.predict(x_new_poly)
        plot_predictions(x_new, y_pred)
    
    if show_data or show_predictions:
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Data and Model Predictions')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

if __name__ == "__main__":
    main()
