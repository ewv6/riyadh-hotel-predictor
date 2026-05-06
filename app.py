"""
Riyadh Hotel Price Predictor — Streamlit App
Run: streamlit run app.py
"""
import json
import os
import pickle
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PriceModel

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Riyadh Hotel Price Predictor",
    page_icon="🏨",
    layout="centered",
)

# ── Load model & assets ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return PriceModel.from_state()

@st.cache_data
def load_assets():
    with open(os.path.join(STATE_DIR, "hotel_encoding.json")) as f:
        enc = json.load(f)
    with open(os.path.join(STATE_DIR, "kmeans.pkl"), "rb") as f:
        km = pickle.load(f)
    return enc["hotel_map"], enc["global_mean"], km["kmeans"], km["cluster_median_prices"]

model       = load_model()
hotel_map, global_mean, kmeans, cluster_medians = load_assets()
hotel_names = sorted(hotel_map.keys())

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏨 Riyadh Hotel Price Predictor")
st.caption("Trained on 8,074 Riyadh hotels")
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
hotel_input = st.selectbox(
    "Hotel Name",
    options=["— unknown hotel —"] + hotel_names,
    help="Select a known hotel for highest accuracy. Unknown hotels use the global average.",
)
hotel_name = "" if hotel_input == "— unknown hotel —" else hotel_input

col1, col2, col3 = st.columns(3)
with col1:
    rating = st.slider("Guest Rating", 0.0, 10.0, 7.5, 0.1)
with col2:
    reviews_count = st.number_input("Reviews Count", min_value=0, max_value=10000, value=300, step=50)
with col3:
    stay_duration = st.number_input("Nights", min_value=1, max_value=30, value=2)

col4, col5 = st.columns(2)
with col4:
    checkin_date = st.date_input(
        "Check-in Date",
        value=date.today() + timedelta(days=7),
        min_value=date.today(),
    )
with col5:
    st.write("")
    st.write("")
    checkout_date = checkin_date + timedelta(days=int(stay_duration))
    st.info(f"Check-out: **{checkout_date.strftime('%d %b %Y')}**")

with st.expander("📍 Location (optional — improves accuracy)"):
    lc1, lc2 = st.columns(2)
    with lc1:
        latitude  = st.number_input("Latitude",  value=24.774, format="%.4f")
    with lc2:
        longitude = st.number_input("Longitude", value=46.654, format="%.4f")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):

    cluster        = int(kmeans.predict([[latitude, longitude]])[0])
    cluster_median = cluster_medians.get(cluster, global_mean)
    checkin_month      = checkin_date.month
    checkin_dayofweek  = checkin_date.weekday()
    is_saudi_weekend   = int(checkin_dayofweek in [3, 4])

    result = model.predict(
        hotel_name           = hotel_name,
        rating               = rating,
        reviews_count        = reviews_count,
        stay_duration        = stay_duration,
        checkin_month        = checkin_month,
        checkin_dayofweek    = checkin_dayofweek,
        latitude             = latitude,
        longitude            = longitude,
        location_cluster     = cluster,
        cluster_median_price = cluster_median,
        competition_density  = 15,
        is_saudi_weekend     = is_saudi_weekend,
    )

    ppn   = result["price_per_night"]
    total = result["total_price"]

    st.success("Prediction complete")

    r1, r2 = st.columns(2)
    r1.metric("Price / Night", f"{ppn:,.0f} SAR")
    r2.metric(f"Total ({stay_duration} nights)", f"{total:,.0f} SAR")

    if ppn < 300:
        tier, color = "Budget", "🟢"
    elif ppn < 800:
        tier, color = "Mid-range", "🟡"
    else:
        tier, color = "Luxury", "🔴"

    st.write(f"**Tier:** {color} {tier}")

    if hotel_name and hotel_name in hotel_map:
        st.caption(f"✅ Known hotel — historical avg: {hotel_map[hotel_name]:,.0f} SAR/night")
    elif hotel_name:
        st.caption(f"⚠️ Hotel not in training data — using global average ({global_mean:,.0f} SAR/night)")
    else:
        st.caption(f"ℹ️ No hotel name provided — using global average ({global_mean:,.0f} SAR/night)")

    if is_saudi_weekend:
        st.caption("🕌 Saudi weekend pricing applied (Thu/Fri check-in)")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: HistGradientBoostingRegressor · Dataset: Riyadh Hotels 2026 (Kaggle) · 22 features")
