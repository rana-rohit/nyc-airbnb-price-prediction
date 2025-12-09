import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from src.data_prep import load_listings, quick_clean, find_listings_path

# Define paths for model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.joblib')
PREPROC_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preproc.joblib')

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")
st.title("Airbnb Price Predictor — Quick Demo")

# Sidebar
st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader("Upload listings.csv (optional)", type=['csv'])
use_pretrained = st.sidebar.checkbox("Use pretrained model (models/rf_model.joblib)", value=os.path.exists(MODEL_PATH))
train_local = st.sidebar.button("Train model locally (slow)")

# Load data
data_path = None
if uploaded:
    df = pd.read_csv(uploaded)
else:
    data_path = find_listings_path()
    if data_path:
        df = load_listings(data_path)
    else:
        df = pd.DataFrame()

if train_local:
    st.info("Training a model locally (this may take several minutes). If dataset is large, consider using --sample.")
    st.spinner()
    from src.train_model import train
    input_path = uploaded if uploaded else data_path
    if input_path is None:
        st.error("No data supplied for training.")
    else:
        train(input_path, model_out=MODEL_PATH, preproc_out=PREPROC_PATH)
        st.success("Training completed. Restart the app or toggle use_pretrained.")

# load model if requested
model = None
preproc = None
if use_pretrained and os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH):
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)

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
    # build simple form
    neighbourhood = st.text_input("Neighbourhood", value=str(df['neighbourhood'].dropna().unique()[0]) if 'neighbourhood' in df.columns and df['neighbourhood'].dropna().shape[0]>0 else "Unknown")
    room_type = st.selectbox("Room type", options=['Entire home/apt','Private room','Shared room','Hotel room'])
    latitude = st.number_input("Latitude", value=float(df['latitude'].mean()) if 'latitude' in df.columns else 0.0)
    longitude = st.number_input("Longitude", value=float(df['longitude'].mean()) if 'longitude' in df.columns else 0.0)
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
