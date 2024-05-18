import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def create_data(n, a, b, c, noise):
    x = np.random.rand(n, 1)
    y = a * x**2 + b * x + c + np.random.randn(n, 1) * noise
    return x, y

def save_data(x, y, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for i in range(len(x)):
            writer.writerow([x[i][0], y[i][0]])

def plot_data(x, y):
    plt.figure()
    plt.plot(x, y, 'b.', label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    return plt

def main():
    st.title('Quadratic Data Model with Streamlit')
    n = st.sidebar.slider('Number of points', 50, 500, 100)
    a = st.sidebar.number_input('Coefficient a', value=3.0)
    b = st.sidebar.number_input('Coefficient b', value=0.0)
    c = st.sidebar.number_input('Coefficient c', value=0.25)
    noise = st.sidebar.slider('Noise level', 0.0, 1.0, 0.1)

    output = 'data.csv'
    model_output = 'model.pkl'

    x, y = create_data(n, a, b, c, noise)
    save_data(x, y, output)

    if st.button('Plot Data'):
        fig = plot_data(x, y)
        st.pyplot(fig)

    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    x_new = np.linspace(0, 1, 100).reshape(-1, 1)
    x_new_poly = poly_features.transform(x_new)
    y_pred = model.predict(x_new_poly)

    if st.button('Show Regression Model'):
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'b.', label='Data Points')
        plt.plot(x_new, y_pred, 'r-', label='Regression Line')
        plt.title('Fitted Regression Model')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    with open(model_output, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
