import sys
import os

# Setup path for Streamlit Cloud compatibility - MUST be before src imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.data_prep import load_listings, quick_clean, find_listings_path

# Define paths for model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.joblib')
PREPROC_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preproc.joblib')

# Cache data loading to avoid reloading on every interaction
@st.cache_data
def load_data_cached(path):
    """Load and return listings data (cached)"""
    return load_listings(path)

# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_model_cached():
    """Load and return model and preprocessor (cached)"""
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH):
        return joblib.load(MODEL_PATH), joblib.load(PREPROC_PATH)
    return None, None

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")
st.title("Airbnb Price Predictor — Quick Demo")

# Sidebar
st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader("Upload listings.csv (optional)", type=['csv'])
use_pretrained = st.sidebar.checkbox("Use pretrained model (models/rf_model.joblib)", value=os.path.exists(MODEL_PATH))
train_local = st.sidebar.button("Train model locally (slow)")

# Load data (cached)
data_path = None
if uploaded:
    df = pd.read_csv(uploaded)
else:
    data_path = find_listings_path()
    if data_path:
        df = load_data_cached(data_path)
    else:
        df = pd.DataFrame()

if train_local:
    st.info("Training a model locally (this may take several minutes). If dataset is large, consider using --sample.")
    from src.train_model import train
    input_path = uploaded if uploaded else data_path
    if input_path is None:
        st.error("No data supplied for training.")
    else:
        with st.spinner("Training model... Please wait."):
            try:
                train(input_path, model_out=MODEL_PATH, preproc_out=PREPROC_PATH)
                st.cache_resource.clear()  # Clear cache to reload new model
                st.success("Training completed! Refresh the page to use the new model.")
            except Exception as e:
                st.error(f"Training failed: {e}")

# Load model (cached)
model = None
preproc = None
if use_pretrained:
    model, preproc = load_model_cached()

# Tabs
tab1, tab2, tab3 = st.tabs(["EDA", "Predict Single", "Batch Predict"])

with tab1:
    st.header("Quick EDA")
    if df.empty:
        st.info("No data loaded. Upload listings.csv or place it in data/ and restart.")
    else:
        dfc = quick_clean(df)
        st.subheader("Price distribution (clipped at $1000)")
        st.hist_chart = st.bar_chart(dfc['price_clipped'].value_counts(bins=50).sort_index())
        st.subheader("Price by room type")
        if 'room_type' in dfc.columns:
            st.write(dfc.groupby('room_type')['price'].median().sort_values(ascending=False))
        st.subheader("Top 8 neighborhoods by median price")
        if 'neighbourhood' in dfc.columns:
            med = dfc.groupby('neighbourhood')['price'].median().sort_values(ascending=False).head(8)
            st.table(med)

with tab2:
    st.header("Predict single listing")
    if df.empty:
        st.info("No dataset loaded; you can still enter values manually.")
    
    # Initialize session state for coordinates
    if 'selected_lat' not in st.session_state:
        st.session_state.selected_lat = float(df['latitude'].mean()) if 'latitude' in df.columns and not df.empty else 40.7128
    if 'selected_lon' not in st.session_state:
        st.session_state.selected_lon = float(df['longitude'].mean()) if 'longitude' in df.columns and not df.empty else -74.0060
    
    # Interactive map for location selection
    st.subheader("Click on the map to select location")
    import folium
    from streamlit_folium import st_folium
    
    # Create map centered on NYC
    m = folium.Map(
        location=[st.session_state.selected_lat, st.session_state.selected_lon],
        zoom_start=11,
        tiles="CartoDB positron"
    )
    
    # Add marker for current selection
    folium.Marker(
        [st.session_state.selected_lat, st.session_state.selected_lon],
        popup=f"Selected: ({st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f})",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)
    
    # Display map and capture clicks
    map_data = st_folium(m, width=700, height=400, key="location_map")
    
    # Update coordinates if user clicked on map
    if map_data and map_data.get("last_clicked"):
        st.session_state.selected_lat = map_data["last_clicked"]["lat"]
        st.session_state.selected_lon = map_data["last_clicked"]["lng"]
    
    # Display selected coordinates
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latitude", f"{st.session_state.selected_lat:.6f}")
    with col2:
        st.metric("Longitude", f"{st.session_state.selected_lon:.6f}")
    
    latitude = st.session_state.selected_lat
    longitude = st.session_state.selected_lon
    
    st.divider()
    
    # Other form inputs
    neighbourhood = st.text_input("Neighbourhood", value=str(df['neighbourhood'].dropna().unique()[0]) if 'neighbourhood' in df.columns and df['neighbourhood'].dropna().shape[0]>0 else "Unknown")
    room_type = st.selectbox("Room type", options=['Entire home/apt','Private room','Shared room','Hotel room'])
    minimum_nights = st.number_input("Minimum nights", value=int(df['minimum_nights'].median()) if 'minimum_nights' in df.columns else 1)
    number_of_reviews = st.number_input("Number of reviews", value=int(df['number_of_reviews'].median()) if 'number_of_reviews' in df.columns else 0)
    reviews_per_month = st.number_input("Reviews per month", value=float(df['reviews_per_month'].median()) if 'reviews_per_month' in df.columns else 0.0)
    calculated_host_listings_count = st.number_input("Host listings count", value=int(df['calculated_host_listings_count'].median()) if 'calculated_host_listings_count' in df.columns else 1)
    availability_365 = st.number_input("Availability (days)", value=int(df['availability_365'].median()) if 'availability_365' in df.columns else 365)

    if st.button("Predict price"):
        row = pd.DataFrame([{
            'neighbourhood': neighbourhood,
            'room_type': room_type,
            'latitude': latitude,
            'longitude': longitude,
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'calculated_host_listings_count': calculated_host_listings_count,
            'availability_365': availability_365
        }])
        if model is None or preproc is None:
            st.error("Model not loaded. Train model locally or uncheck 'Use pretrained model'.")
        else:
            X = preproc.transform(row)
            yhat_log = model.predict(X)
            yhat = np.expm1(yhat_log)[0]
            st.success(f"Predicted nightly price: ${yhat:.2f}")

with tab3:
    st.header("Batch predict & sample download")
    if df.empty:
        st.info("No data to predict; upload CSV or place listings.csv in data/.")
    else:
        st.write(f"Rows loaded: {df.shape[0]}")
        if model is None or preproc is None:
            st.warning("Model not loaded - run training (src/train_model.py) or enable pretrained model.")
        else:
            dfc = quick_clean(df)
            X = preproc.transform(dfc)
            preds_log = model.predict(X)
            dfc['pred_price'] = np.expm1(preds_log)
            st.dataframe(dfc[['id','neighbourhood','room_type','price','pred_price']].head(15))
            csv = dfc.to_csv(index=False)
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
