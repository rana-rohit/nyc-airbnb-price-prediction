# src/data_prep.py
import pandas as pd
import numpy as np

DEFAULT_COLS = ['neighbourhood', 'latitude', 'longitude', 'room_type',
                'price', 'minimum_nights', 'number_of_reviews',
                'reviews_per_month', 'calculated_host_listings_count',
                'availability_365', 'id']

def find_listings_path():
    # Use absolute paths relative to this module's location
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, 'data', 'sample_listings.csv'),
        os.path.join(base_dir, 'data', 'listings.csv'),
        os.path.join(base_dir, 'data', 'Data1', 'listings.csv'),
        os.path.join(base_dir, 'data', 'listings_clean.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def load_listings(path=None, nrows=None):
    """
    Load listings.csv and perform lightweight cleaning.
    If path is None, tries default locations.
    """
    if path is None:
        path = find_listings_path()
        if path is None:
            raise FileNotFoundError("Could not find listings.csv in data/ or data/Data1/. Place file there or provide path.")
    df = pd.read_csv(path, nrows=nrows)
    df = df.copy()
    # standardize price field if present
    if 'price' in df.columns:
        df['price'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
        # coerce to numeric, drop bad rows
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'].notnull()]
        df = df[df['price'] > 0]
    # ensure expected columns exist
    for c in DEFAULT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # basic type fixes
    df['neighbourhood'] = df['neighbourhood'].astype(str)
    df['room_type'] = df['room_type'].astype(str)
    return df

def quick_clean(df):
    """
    Additional light cleaning for EDA:
    - cap extreme prices for visuals
    - fill NaNs for numeric cols with median
    """
    df = df.copy()
    if 'price' in df.columns:
        # create clipped view for plotting
        df['price_clipped'] = df['price'].clip(upper=1000)
    num_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                'calculated_host_listings_count', 'availability_365']
    for c in num_cols:
        if c in df.columns:
            median = df[c].median() if df[c].dropna().shape[0] > 0 else 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(median)
    # lat/lon numeric
    for c in ['latitude', 'longitude']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
