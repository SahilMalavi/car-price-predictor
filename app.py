import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Set up the Streamlit app
st.title('Car Price Prediction')
    
# Dropdown for selecting the company
companies = sorted(car['company'].unique())
companies.insert(0, 'Select Company')
company = st.selectbox('Company', companies)

# Dropdown for selecting the car model
if company != 'Select Company':
    car_models = sorted(car[car['company'] == company]['name'].unique())
else:
    car_models = sorted(car['name'].unique())
car_model = st.selectbox('Car Model', car_models)

# Dropdown for selecting the year
year = list(range(2024, 1999, -1))
selected_year = st.selectbox('Model Year', year)

# Dropdown for selecting the fuel type
fuel_type = car['fuel_type'].unique()
selected_fuel_type = st.selectbox('Fuel Type', fuel_type)

# Input box for kilometers driven
driven = st.number_input('Kilometers Driven', min_value=0)

# Button for prediction
if st.button('Predict Price'):
    # Ensure all selections are made
    if company == 'Select Company':
        st.error('Please select a valid company.')
    else:
        # Perform prediction
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                  data=np.array([car_model, company, selected_year, driven, selected_fuel_type]).reshape(1, 5))
        try:
            prediction = model.predict(input_data)
            st.success(f'The predicted price of the car is ₹ {np.round(prediction[0], 2)}')
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
