# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 05:09:28 2024

@author: Sooraj
"""

import streamlit as st
import joblib
import numpy as np

# Load the trained KNN model
model = joblib.load('best_knn_model.pkl')  # Update with your model file path

# Define the Streamlit app
def main():
    # Set title
    st.title('Customer Segment Prediction')

    # Define input widgets for numerical variables
    income = st.slider('Income', min_value=10000, max_value=200000, value=50000)
    expenses = st.slider('Expenses', min_value=0, max_value=10000, value=1000)
    purchases = st.slider('Purchases', min_value=0, max_value=100, value=10)
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    customer_for = st.slider('Customer For (Days)', min_value=0, max_value=365, value=180)
    
    # Define input widgets for categorical variables
    education = st.selectbox('Education', ['Basic', 'Graduation', 'Post Graduate'])  # Update with your categories
    living_with = st.selectbox('Living With', ['Alone', 'Partner'])  # Update with your categories
    is_parent = st.selectbox('Is Parent', ['0', '1'])  # Update with your categories
    
    # Convert categorical input to numerical labels
    education_label = {'Basic': 0, 'Graduation': 1, 'Post Graduate': 2}[education]
    living_with_label = {'Alone': 0, 'Partner': 1}[living_with]
    is_parent_label = int(is_parent)
    
    # Make prediction based on user inputs
    features = np.array([[income, education_label, expenses, purchases, age, customer_for, 0, 0, living_with_label, 0, is_parent_label]])  # Assuming 'Children' and 'Clusters' are not used
    prediction = model.predict(features)
    
    # Display prediction
    st.subheader('Prediction')
    st.write(f'The predicted customer segment is: {prediction[0]}')

# Run the Streamlit app
if __name__ == '__main__':
    main()

