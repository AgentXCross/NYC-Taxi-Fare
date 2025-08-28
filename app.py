import streamlit as st
import joblib
import pandas as pd
from features import apply_features
import time
import requests
import datetime as dt

st.set_page_config(page_title = "NYC Taxi Fare Machine Learning Model", page_icon = "🚕")

# --------------------------
# 1) Load model + feature list 
# --------------------------
@st.cache_resource
def load_artifacts():
    """Loads trained model and its feature names"""
    model = joblib.load("lgbm_model.pkl")  
    feat_names = joblib.load("feature_names.pkl") 
    return model, feat_names

model, feature_names = load_artifacts()

# --------------------------
# 2) Geocoding helper (Nominatim / OpenStreetMap)
# --------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "nyc-taxi-app/1.0"}

@st.cache_data(show_spinner = False, ttl = 3600)
def geocode_candidates(query: str, limit: int = 5):
    """Return up to 5 address candidates with (label, lat, lon)."""
    if not query or len(query.strip()) < 3:
        return []
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
        "countrycodes": "us",
        # "viewbox": "-74.3,40.5,-73.6,41.0",
        # "bounded": 1,
    }
    r = requests.get(NOMINATIM_URL, params = params, headers = HEADERS, timeout = 10)

    time.sleep(1.1)
    r.raise_for_status()
    out = []
    for item in r.json():
        out.append({
            "label": item.get("display_name", "Unknown"),
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
        })
    return out

# --------------------------
# 3) User Interface
# --------------------------
st.title("NYC Taxi 🚕 Fare Predictor")
st.caption("Type a Pickup and Destination Address or Place")

left, right = st.columns(2)

with left:
    start_q = st.text_input("Start (Address or Place)", placeholder = "Ex: Empire State Building")
    start_opts = geocode_candidates(start_q)
    start_choice = None
    if start_opts:
        start_idx = st.selectbox(
            "Choose start match",
            options = list(range(len(start_opts))),
            format_func = lambda i: start_opts[i]["label"],
            key = "start_sel",
        )
        start_choice = start_opts[start_idx]

with right:
    end_q = st.text_input("Destination (Address or Place)", placeholder = "Ex: JFK")
    end_opts = geocode_candidates(end_q)
    end_choice = None
    if end_opts:
        end_idx = st.selectbox(
            "Choose end match",
            options = list(range(len(end_opts))),
            format_func = lambda i: end_opts[i]["label"],
            key = "end_sel",
        )
        end_choice = end_opts[end_idx]

# trip extras
colA, colB = st.columns(2)
with colA:
    passenger_count = st.number_input("Number of Passengers", min_value = 1, max_value = 6, value = 1, step = 1)
with colB:
    d = st.date_input("Pickup Date")
    t = st.time_input("Pickup Time", value = dt.time(12, 0))  
    pickup_dt = pd.to_datetime(dt.datetime.combine(d, t))

# quick map preview
if start_choice and end_choice:
    pts = pd.DataFrame(
        [
            {"name": "start", "lat": start_choice["lat"], "lon": start_choice["lon"]},
            {"name": "end", "lat": end_choice["lat"], "lon": end_choice["lon"]},
        ]
    )
    st.map(pts, latitude = "lat", longitude = "lon", size = 60)

# --------------------------
# 4) Prediction
# --------------------------
if st.button("Predict fare", type = "primary"):
    if not (start_choice and end_choice):
        st.error("Please choose both a start and an end address.")
        st.stop()

    raw = pd.DataFrame([{
        "pickup_latitude": start_choice["lat"],
        "pickup_longitude": start_choice["lon"],
        "dropoff_latitude": end_choice["lat"],
        "dropoff_longitude": end_choice["lon"],
        "passenger_count": passenger_count,
        "pickup_datetime": pd.to_datetime(pickup_dt),
    }])

    #Apply feature engineering
    feats = apply_features(raw, datetime_col = "pickup_datetime")

    # enforce training column order (prevents mismatch)
    X = feats.reindex(columns = feature_names, fill_value = 0)

    #Return predicted fare
    pred = float(model.predict(X)[0])
    st.metric("Predicted Fare", f"${pred:,.2f}")