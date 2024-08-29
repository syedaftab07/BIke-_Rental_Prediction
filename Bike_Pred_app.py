import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('r1.pkl')

# Function to map season names to numeric values
def map_season(season_name):
    season_mapping = {
        'Spring': 1,
        'Summer': 2,
        'Fall': 3,
        'Winter': 4
    }
    return season_mapping[season_name]

# Function to map weather situations to numeric values
def map_weathersit(weather_name):
    weather_mapping = {
        'Clear': 1,
        'Mist': 2,
        'Light Snow': 3,
        'Heavy Snow': 4
    }
    return weather_mapping[weather_name]

# Function to make prediction
def predict_demand(data):
    return model.predict(data)

# Create Streamlit app
def main():
    st.title("Bike Rental Demand Prediction")

    # User input for features
    season = st.selectbox('Season', options=['Spring', 'Summer', 'Fall', 'Winter'])
    hr = st.slider('Hour', min_value=0, max_value=23, value=12)
    holiday = st.selectbox('Holiday', options=[0, 1])
    weekday = st.selectbox('Weekday', options=[0, 1, 2, 3, 4, 5, 6])
    workingday = st.selectbox('Working Day', options=[0, 1])
    weathersit = st.selectbox('Weather Situation', options=['Clear', 'Mist', 'Light Snow', 'Heavy Snow'])
    temp = st.slider('Temperature (normalized)', min_value=0.0, max_value=1.0, value=0.5)
    atemp = st.slider('Feels Like Temperature (normalized)', min_value=0.0, max_value=1.0, value=0.5)
    hum = st.slider('Humidity (normalized)', min_value=0.0, max_value=1.0, value=0.5)
    windspeed = st.slider('Wind Speed (normalized)', min_value=0.0, max_value=1.0, value=0.5)
    year = st.number_input('Year', min_value=0, max_value=9999, value=2023)
    month = st.selectbox('Month', options=list(range(1, 13)))

    # Map season and weather situation names to numeric values
    season_mapped = map_season(season)
    weathersit_mapped = map_weathersit(weathersit)

    # Create input DataFrame
    input_data = pd.DataFrame({
        'season': [season_mapped], 'hr': [hr], 'holiday': [holiday], 'weekday': [weekday],
        'workingday': [workingday], 'weathersit': [weathersit_mapped], 'temp': [temp],
        'atemp': [atemp], 'hum': [hum], 'windspeed': [windspeed], 'year': [year],
        'month': [month]
    })

    # Make prediction
    if st.button('Predict'):
        prediction = predict_demand(input_data)
        st.write(f"Predicted Bike Rentals: {int(prediction[0])}")

# Run the app
if __name__ == '__main__':
    main()
