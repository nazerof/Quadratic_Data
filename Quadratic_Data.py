import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Plot predictions
def plot_predictions(x, y_pred):
    plt.figure()
    plt.plot(x, y_pred, 'r-', label='Predicted Polynomial Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression Model Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    st.title('Quadratic Regression Model Visualization')

    model = load_model('model.pkl')  # Adjust path as needed
    poly_features = PolynomialFeatures(degree=2)

    # Slider for user to generate new data points
    x_new = np.linspace(0, 1, 100).reshape(-1, 1)
    x_new_poly = poly_features.transform(x_new)

    # Display the model predictions
    if st.button('Show Predictions'):
        y_pred = model.predict(x_new_poly)
        fig, ax = plt.subplots()
        ax.plot(x_new, y_pred, 'r-', label='Predicted Polynomial Regression')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Regression Model Predictions')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
