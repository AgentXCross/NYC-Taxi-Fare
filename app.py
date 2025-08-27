import streamlit as st
import pandas as pd
import datetime as dt
import joblib
import folium
from streamlit_folium import st_folium

from features import add_date_features, add_trip_distance, add_is_manhattan, add_landmarks, cross_manhattan
from model import load_model

# Load trained model + feature names
@st.cache_resource
def get_model_and_features():
    model = load_model("best_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = get_model_and_features()

st.title("🚕 NYC Taxi Fare Predictor")

# --- Map Interaction ---
st.subheader("Click on the map to select Pickup (green) and Dropoff (red)")

# Initialize state
if "pickup" not in st.session_state:
    st.session_state["pickup"] = None
if "dropoff" not in st.session_state:
    st.session_state["dropoff"] = None
if "selecting" not in st.session_state:
    st.session_state["selecting"] = "pickup"  # start with pickup

# Base map
m = folium.Map(location = [40.75, -73.97], zoom_start = 12)

# Add existing markers if set
if st.session_state["pickup"]:
    folium.Marker(
        st.session_state["pickup"],
        popup = "Pickup",
        icon = folium.Icon(color = "green")
    ).add_to(m)
if st.session_state["dropoff"]:
    folium.Marker(
        st.session_state["dropoff"],
        popup = "Dropoff",
        icon = folium.Icon(color = "red")
    ).add_to(m)

# Render map
output = st_folium(m, width = 700, height = 500)

# Handle clicks
if output["last_clicked"] is not None:
    lat, lon = output["last_clicked"]["lat"], output["last_clicked"]["lng"]

    if st.session_state["selecting"] == "pickup":
        st.session_state["pickup"] = (lat, lon)
        st.session_state["selecting"] = "dropoff"
        st.success(f"✅ Pickup set at {lat:.4f}, {lon:.4f}")
    elif st.session_state["selecting"] == "dropoff":
        st.session_state["dropoff"] = (lat, lon)
        st.session_state["selecting"] = "pickup"
        st.success(f"✅ Dropoff set at {lat:.4f}, {lon:.4f}")

# --- Show current selections ---
if st.session_state["pickup"]:
    st.write("Pickup:", st.session_state["pickup"])
if st.session_state["dropoff"]:
    st.write("Dropoff:", st.session_state["dropoff"])

# --- Datetime ---
pickup_date = st.date_input("Pickup Date", value = dt.date.today())
pickup_time = st.time_input("Pickup Time", value = dt.datetime.now().time())
pickup_datetime = dt.datetime.combine(pickup_date, pickup_time)

# --- Passengers ---
passengers = st.slider("Number of Passengers", min_value = 1, max_value = 6, value = 1)

# --- Predict button ---
if st.button("Predict Fare"):
    if st.session_state["pickup"] and st.session_state["dropoff"]:
        pickup_lat, pickup_lon = st.session_state["pickup"]
        drop_lat, drop_lon = st.session_state["dropoff"]

        # uild input row
        row = pd.DataFrame([{
            "pickup_datetime": pickup_datetime,
            "pickup_latitude": pickup_lat,
            "pickup_longitude": pickup_lon,
            "dropoff_latitude": drop_lat,
            "dropoff_longitude": drop_lon,
            "passenger_count": passengers
        }])

        # Apply same feature engineering
        row = add_date_features(row, "pickup_datetime")
        row = add_trip_distance(row)
        row = add_is_manhattan(row)
        row = add_landmarks(row)
        row = cross_manhattan(row)

        #Drop raw datetime
        row = row.drop(columns = ["pickup_datetime"], errors = "ignore")

        #Align with training features
        row = row.reindex(columns = feature_names, fill_value = 0)

        #Predict
        pred = model.predict(row)[0]
        st.success(f"💵 Estimated Fare: ${pred:.2f}")
    else:
        st.error("Please select both pickup and dropoff points.")