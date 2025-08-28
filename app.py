import streamlit as st
import joblib
import pandas as pd
from features import apply_features

st.set_page_config(page_title = "NYC Taxi Fare Machine Learning Model", page_icon = "🚕")

@st.cache_resource
def load_artifacts():
    """Loads model information. Must have model already trained and """
    model = joblib.load("lgbm_model.pkl")    
    feat_names = joblib.load("feature_names.pkl")
    return model, feat_names

model, feature_names = load_artifacts()

