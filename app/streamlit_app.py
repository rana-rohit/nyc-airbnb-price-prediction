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
import plotly.express as px
import plotly.graph_objects as go
from src.data_prep import load_listings, quick_clean, find_listings_path

# Define paths for model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.joblib')
PREPROC_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preproc.joblib')

# Page configuration
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Airbnb-themed styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --airbnb-coral: #FF5A5F;
        --airbnb-dark: #484848;
        --airbnb-light: #767676;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #FF5A5F 0%, #FC642D 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #FF5A5F;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF5A5F;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #767676;
        text-transform: uppercase;
    }
    
    /* Prediction result */
    .prediction-result {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    .prediction-price {
        font-size: 3rem;
        font-weight: bold;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF5A5F 0%, #FC642D 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #E5494D 0%, #E5582A 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: #F7F7F7;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #FF5A5F;
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading
@st.cache_data
def load_data_cached(path):
    """Load and return listings data (cached)"""
    return load_listings(path)


# Cache model loading
@st.cache_resource
def load_model_cached():
    """Load and return model and preprocessor (cached)"""
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH):
        return joblib.load(MODEL_PATH), joblib.load(PREPROC_PATH)
    return None, None


# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/512px-Airbnb_Logo_B%C3%A9lo.svg.png", width=150)
    st.markdown("---")
    st.header("⚙️ Options")
    uploaded = st.file_uploader("📁 Upload listings.csv", type=['csv'])
    use_pretrained = st.checkbox("🤖 Use pretrained model", value=os.path.exists(MODEL_PATH))
    train_local = st.button("🔧 Train model locally")
    st.markdown("---")
    st.markdown("**About**")
    st.caption("NYC Airbnb Price Predictor uses machine learning to estimate nightly prices based on listing features.")

# Load data
data_path = None
if uploaded:
    df = pd.read_csv(uploaded)
else:
    data_path = find_listings_path()
    if data_path:
        df = load_data_cached(data_path)
    else:
        df = pd.DataFrame()

# Handle training
if train_local:
    st.info("🔄 Training model locally (this may take several minutes)...")
    from src.train_model import train
    input_path = uploaded if uploaded else data_path
    if input_path is None:
        st.error("❌ No data supplied for training.")
    else:
        with st.spinner("Training model... Please wait."):
            try:
                train(input_path, model_out=MODEL_PATH, preproc_out=PREPROC_PATH)
                st.cache_resource.clear()
                st.success("✅ Training completed! Refresh the page to use the new model.")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

# Load model
model, preproc = None, None
if use_pretrained:
    model, preproc = load_model_cached()

# Hero Header
st.markdown("""
<div class="main-header">
    <h1>🏠 NYC Airbnb Price Predictor</h1>
    <p>Predict nightly prices for New York City Airbnb listings using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Dashboard
if not df.empty:
    dfc = quick_clean(df)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Listings", f"{len(dfc):,}")
    with col2:
        st.metric("💰 Avg Price", f"${dfc['price'].mean():.0f}/night")
    with col3:
        st.metric("🏘️ Neighborhoods", f"{dfc['neighbourhood'].nunique()}")
    with col4:
        st.metric("⭐ Avg Reviews", f"{dfc['number_of_reviews'].mean():.0f}")
    
    st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Analytics", "🎯 Predict Price", "📦 Batch Predict", "🗺️ Price Map"])

# Tab 1: Analytics (Enhanced EDA)
with tab1:
    st.header("📈 Data Analytics")
    
    if df.empty:
        st.info("📁 No data loaded. Upload listings.csv or place it in data/ folder.")
    else:
        dfc = quick_clean(df)
        
        # Row 1: Price Distribution and Room Type
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Price Distribution")
            fig_hist = px.histogram(
                dfc[dfc['price'] <= 500],
                x='price',
                nbins=50,
                color_discrete_sequence=['#FF5A5F'],
                labels={'price': 'Price ($)', 'count': 'Number of Listings'}
            )
            fig_hist.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("🏠 Price by Room Type")
            room_stats = dfc.groupby('room_type')['price'].agg(['median', 'mean', 'count']).reset_index()
            room_stats.columns = ['Room Type', 'Median Price', 'Mean Price', 'Count']
            fig_room = px.bar(
                room_stats.sort_values('Median Price', ascending=True),
                x='Median Price',
                y='Room Type',
                orientation='h',
                color='Median Price',
                color_continuous_scale=['#FFB6C1', '#FF5A5F', '#C41E3A'],
                text='Median Price'
            )
            fig_room.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
            fig_room.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_room, use_container_width=True)
        
        # Row 2: Top Neighborhoods
        st.subheader("🏆 Top 10 Neighborhoods by Median Price")
        top_neighborhoods = dfc.groupby('neighbourhood')['price'].median().sort_values(ascending=False).head(10).reset_index()
        top_neighborhoods.columns = ['Neighbourhood', 'Median Price']
        
        fig_neigh = px.bar(
            top_neighborhoods,
            x='Median Price',
            y='Neighbourhood',
            orientation='h',
            color='Median Price',
            color_continuous_scale=['#FFE4E1', '#FF5A5F', '#8B0000'],
            text='Median Price'
        )
        fig_neigh.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
        fig_neigh.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_neigh, use_container_width=True)
        
        # Row 3: Availability vs Price scatter
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📅 Availability vs Price")
            sample_df = dfc.sample(min(1000, len(dfc)), random_state=42)
            fig_scatter = px.scatter(
                sample_df,
                x='availability_365',
                y='price',
                color='room_type',
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'availability_365': 'Days Available (per year)', 'price': 'Price ($)'}
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.subheader("📊 Room Type Distribution")
            room_counts = dfc['room_type'].value_counts().reset_index()
            room_counts.columns = ['Room Type', 'Count']
            fig_pie = px.pie(
                room_counts,
                values='Count',
                names='Room Type',
                color_discrete_sequence=['#FF5A5F', '#FC642D', '#00A699', '#767676']
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# Tab 2: Single Prediction
with tab2:
    st.header("🎯 Predict Single Listing Price")
    
    if df.empty:
        st.info("📁 No dataset loaded; you can still enter values manually.")
    
    # Initialize session state
    if 'selected_lat' not in st.session_state:
        st.session_state.selected_lat = float(df['latitude'].mean()) if 'latitude' in df.columns and not df.empty else 40.7128
    if 'selected_lon' not in st.session_state:
        st.session_state.selected_lon = float(df['longitude'].mean()) if 'longitude' in df.columns and not df.empty else -74.0060
    
    col_map, col_form = st.columns([1, 1])
    
    with col_map:
        st.subheader("📍 Select Location")
        import folium
        from streamlit_folium import st_folium
        
        m = folium.Map(
            location=[st.session_state.selected_lat, st.session_state.selected_lon],
            zoom_start=11,
            tiles="CartoDB positron"
        )
        
        folium.Marker(
            [st.session_state.selected_lat, st.session_state.selected_lon],
            popup=f"Selected Location",
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)
        
        map_data = st_folium(m, width=None, height=350, key="location_map")
        
        if map_data and map_data.get("last_clicked"):
            st.session_state.selected_lat = map_data["last_clicked"]["lat"]
            st.session_state.selected_lon = map_data["last_clicked"]["lng"]
        
        # Coordinate display
        st.markdown(f"""
        <div style="background: #F7F7F7; padding: 1rem; border-radius: 8px; text-align: center;">
            <strong>📍 Lat:</strong> {st.session_state.selected_lat:.4f} | 
            <strong>Lng:</strong> {st.session_state.selected_lon:.4f}
        </div>
        """, unsafe_allow_html=True)
    
    with col_form:
        st.subheader("📝 Listing Details")
        
        # Get unique neighborhoods for dropdown
        if not df.empty and 'neighbourhood' in df.columns:
            neighborhoods = sorted(df['neighbourhood'].dropna().unique().tolist())
            neighbourhood = st.selectbox("🏘️ Neighbourhood", options=neighborhoods)
        else:
            neighbourhood = st.text_input("🏘️ Neighbourhood", value="Manhattan")
        
        room_type = st.selectbox("🏠 Room Type", options=['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            minimum_nights = st.number_input("🌙 Min Nights", value=2, min_value=1)
            number_of_reviews = st.number_input("⭐ Reviews", value=10, min_value=0)
        with col_b:
            reviews_per_month = st.number_input("📈 Reviews/Month", value=1.0, min_value=0.0, step=0.1)
            availability_365 = st.number_input("📅 Availability (days)", value=180, min_value=0, max_value=365)
        
        calculated_host_listings_count = st.number_input("👤 Host Listings", value=1, min_value=1)
        
        predict_btn = st.button("🔮 Predict Price", use_container_width=True)
    
    if predict_btn:
        row = pd.DataFrame([{
            'neighbourhood': neighbourhood,
            'room_type': room_type,
            'latitude': st.session_state.selected_lat,
            'longitude': st.session_state.selected_lon,
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'calculated_host_listings_count': calculated_host_listings_count,
            'availability_365': availability_365
        }])
        
        if model is None or preproc is None:
            st.error("❌ Model not loaded. Enable 'Use pretrained model' in sidebar.")
        else:
            X = preproc.transform(row)
            yhat_log = model.predict(X)
            yhat = np.expm1(yhat_log)[0]
            
            # Get average price for context
            avg_price = dfc['price'].mean() if not df.empty else 150
            diff_pct = ((yhat - avg_price) / avg_price) * 100
            
            st.markdown(f"""
            <div class="prediction-result">
                <div style="font-size: 1rem; opacity: 0.9;">Predicted Nightly Price</div>
                <div class="prediction-price">${yhat:.2f}</div>
                <div style="font-size: 0.95rem; margin-top: 0.5rem;">
                    {"📈" if diff_pct > 0 else "📉"} {abs(diff_pct):.1f}% {"above" if diff_pct > 0 else "below"} average (${avg_price:.0f})
                </div>
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Batch Prediction
with tab3:
    st.header("📦 Batch Predictions")
    
    if df.empty:
        st.info("📁 No data to predict; upload CSV or place listings.csv in data/.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Loaded:** {df.shape[0]:,} listings")
        
        if model is None or preproc is None:
            st.warning("⚠️ Model not loaded. Enable 'Use pretrained model' in sidebar.")
        else:
            dfc = quick_clean(df)
            X = preproc.transform(dfc)
            preds_log = model.predict(X)
            dfc['pred_price'] = np.expm1(preds_log)
            dfc['diff'] = dfc['pred_price'] - dfc['price']
            dfc['diff_pct'] = (dfc['diff'] / dfc['price'] * 100).round(1)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Avg Actual", f"${dfc['price'].mean():.0f}")
            with col2:
                st.metric("🎯 Avg Predicted", f"${dfc['pred_price'].mean():.0f}")
            with col3:
                mae = np.abs(dfc['diff']).mean()
                st.metric("📏 MAE", f"${mae:.0f}")
            with col4:
                underpriced = (dfc['diff'] > 20).sum()
                st.metric("💡 Underpriced", f"{underpriced:,}")
            
            st.markdown("---")
            
            # Actual vs Predicted chart
            st.subheader("📊 Actual vs Predicted Prices")
            sample = dfc.sample(min(500, len(dfc)), random_state=42)
            fig_compare = px.scatter(
                sample,
                x='price',
                y='pred_price',
                color='room_type',
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'price': 'Actual Price ($)', 'pred_price': 'Predicted Price ($)'}
            )
            # Add diagonal line
            max_price = max(sample['price'].max(), sample['pred_price'].max())
            fig_compare.add_trace(go.Scatter(
                x=[0, max_price], y=[0, max_price],
                mode='lines', name='Perfect Prediction',
                line=dict(color='#FF5A5F', dash='dash')
            ))
            fig_compare.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Data table
            st.subheader("📋 Predictions Table")
            display_cols = ['id', 'neighbourhood', 'room_type', 'price', 'pred_price', 'diff_pct']
            st.dataframe(
                dfc[display_cols].head(50).style.background_gradient(
                    subset=['diff_pct'], cmap='RdYlGn_r', vmin=-50, vmax=50
                ),
                use_container_width=True
            )
            
            # Download button
            csv = dfc.to_csv(index=False)
            st.download_button(
                "📥 Download Full Predictions CSV",
                csv,
                "airbnb_predictions.csv",
                "text/csv",
                use_container_width=True
            )

# Tab 4: Price Heatmap
with tab4:
    st.header("🗺️ NYC Price Heatmap")
    
    if df.empty:
        st.info("📁 No data loaded for map visualization.")
    else:
        dfc = quick_clean(df)
        
        # Sample for performance
        sample_size = st.slider("Sample size for map", 100, 5000, 1000, 100)
        sample_df = dfc.sample(min(sample_size, len(dfc)), random_state=42)
        
        # Price filter
        price_range = st.slider(
            "Price range ($)",
            int(dfc['price'].min()),
            min(int(dfc['price'].max()), 1000),
            (0, 500)
        )
        filtered_df = sample_df[(sample_df['price'] >= price_range[0]) & (sample_df['price'] <= price_range[1])]
        
        st.write(f"Showing {len(filtered_df):,} listings")
        
        # Create map with price heatmap
        fig_map = px.scatter_mapbox(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color='price',
            size='price',
            size_max=15,
            color_continuous_scale='RdYlGn_r',
            zoom=10,
            center={"lat": 40.7128, "lon": -74.0060},
            mapbox_style="carto-positron",
            hover_data=['neighbourhood', 'room_type', 'price']
        )
        fig_map.update_layout(
            height=600,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        st.plotly_chart(fig_map, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #767676; font-size: 0.85rem;">
    Made with ❤️ using Streamlit | Data from Inside Airbnb
</div>
""", unsafe_allow_html=True)
