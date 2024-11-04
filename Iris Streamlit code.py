
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved stacking model
model = pickle.load(open('stacking_iris.pkl', 'rb'))

# Set up the page title and description
st.title("Iris Flower Classification App")
st.write("Classify Iris flowers into Setosa, Versicolor, and Virginica based on their measurements.")

# Input fields for each feature
st.sidebar.header("Input Measurements")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Prediction function
def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Create a DataFrame with input data
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    # Make prediction
    prediction = model.predict(input_data)
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    return species[prediction[0]]

# Run the classifier when button is clicked
if st.sidebar.button("Classify"):
    result = classify_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f"The predicted Iris species is: **{result}**")

# Display additional information
st.write("This model uses a Stacking Classifier with Decision Tree, KNN, SVC, and Random Forest as base models and Logistic Regression as the meta model.")
st.write("Note: This application is for educational purposes and demonstrates a basic classification model for Iris species.")

