import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Flight Price Predictor", layout="centered")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure it is in the project directory.")

# Mapping Dictionaries (Matching your logic)
airline_dict = {'AirAsia': 0, "Indigo": 1, "GO_FIRST": 2, "SpiceJet": 3, "Air_India": 4, "Vistara": 5}
source_dict = {'Delhi': 0, "Hyderabad": 1, "Bangalore": 2, "Mumbai": 3, "Kolkata": 4, "Chennai": 5}
departure_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
stops_dict = {'zero': 0, "one": 1, "two_or_more": 2}
arrival_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
destination_dict = {'Delhi': 0, "Hyderabad": 1, "Mumbai": 2, "Bangalore": 3, "Chennai": 4, "Kolkata": 5}
class_dict = {'Economy': 0, 'Business': 1}

# UI Layout
st.title("✈️ Flight Price Prediction")
st.write("Enter your flight details below to estimate the price.")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", list(airline_dict.keys()))
    source_city = st.selectbox("Source City", list(source_dict.keys()))
    departure_time = st.selectbox("Departure Time", list(departure_dict.keys()))
    departure_date = st.date_input("Date of Departure", min_value=datetime.today())

with col2:
    destination_city = st.selectbox("Destination City", list(destination_dict.keys()))
    travel_class = st.selectbox("Class", list(class_dict.keys()))
    arrival_time = st.selectbox("Arrival Time", list(arrival_dict.keys()))
    stops = st.selectbox("Stops", list(stops_dict.keys()))

if st.button("Predict Price"):
    try:
        # Data Preprocessing
        airline_val = airline_dict[airline]
        source_val = source_dict[source_city]
        dep_val = departure_dict[departure_time]
        stops_val = stops_dict[stops]
        arr_val = arrival_dict[arrival_time]
        dest_val = destination_dict[destination_city]
        class_val = class_dict[travel_class]
        
        # Calculate date difference (Days left until journey)
        date_diff = (departure_date - datetime.today().date()).days + 1

        # Prediction
        features = [airline_val, source_val, dep_val, stops_val, arr_val, dest_val, class_val, date_diff]
        prediction = model.predict([features])[0]

        st.success(f"### Estimated Flight Price: ₹{round(prediction, 2)}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
