import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model artifacts ──────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model         = joblib.load(os.path.join(BASE_DIR, 'models', 'house_price_model.pkl'))
    scaler        = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(f"⚠️ Model not loaded yet. Train the model first by running the notebook.\n\n`{e}`")

# ── UI ────────────────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.markdown("Enter the house details below to get an estimated sale price.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📐 Size & Space")
    gr_liv_area   = st.number_input("Above-Ground Living Area (sq ft)", 500, 6000, 1500)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    first_flr_sf  = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
    second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 0)
    garage_area   = st.number_input("Garage Area (sq ft)", 0, 1500, 400)

with col2:
    st.subheader("🏡 House Details")
    year_built    = st.slider("Year Built", 1870, 2010, 1990)
    year_remod    = st.slider("Year Remodelled", 1950, 2010, 2000)
    yr_sold       = st.selectbox("Year Sold", [2006, 2007, 2008, 2009, 2010], index=4)
    overall_qual  = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 7)
    full_bath     = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)
    half_bath     = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
    bedroom_abv   = st.selectbox("Bedrooms Above Ground", [1, 2, 3, 4, 5, 6], index=2)
    garage_cars   = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("💰 Predict Sale Price", use_container_width=True, type="primary"):
    if not model_loaded:
        st.error("Model is not loaded. Please train the model first.")
    else:
        # Build feature dict matching training columns
        input_data = {col: 0 for col in feature_names}

        # Fill in our known features
        known = {
            'GrLivArea':    gr_liv_area,
            'TotalBsmtSF':  total_bsmt_sf,
            '1stFlrSF':     first_flr_sf,
            '2ndFlrSF':     second_flr_sf,
            'GarageArea':   garage_area,
            'YearBuilt':    year_built,
            'YearRemodAdd': year_remod,
            'YrSold':       yr_sold,
            'OverallQual':  overall_qual,
            'FullBath':     full_bath,
            'HalfBath':     half_bath,
            'BedroomAbvGr': bedroom_abv,
            'GarageCars':   garage_cars,
            # Engineered features
            'TotalSF':      total_bsmt_sf + first_flr_sf + second_flr_sf,
            'TotalBath':    full_bath + 0.5 * half_bath,
            'HouseAge':     yr_sold - year_built,
            'RemodelAge':   yr_sold - year_remod,
            'HasGarage':    int(garage_area > 0),
            'HasBasement':  int(total_bsmt_sf > 0),
            'HasPool':      0,
        }

        for k, v in known.items():
            if k in input_data:
                input_data[k] = v

        input_df     = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        log_pred     = model.predict(input_scaled)[0]
        price        = np.expm1(log_pred)

        st.success(f"### 🏷️ Estimated Sale Price: **${price:,.0f}**")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Low Estimate",  f"${price * 0.9:,.0f}")
        col_b.metric("Prediction",    f"${price:,.0f}")
        col_c.metric("High Estimate", f"${price * 1.1:,.0f}")

        st.caption("*Estimates are based on a Gradient Boosting model trained on the Ames Housing dataset.*")

st.divider()
st.markdown("Built with 🐍 Python · scikit-learn · Streamlit")