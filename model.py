"""
Standalone PriceModel — usable independently of the CrewAI pipeline.
Loads two pre-trained models (one per platform) and routes predictions accordingly.
"""
import os
import pickle

import numpy as np
import pandas as pd

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")

SOURCE_MAP = {0: "Booking.com", 1: "TripAdvisor"}


class PriceModel:
    """
    Wraps two platform-specific HistGradientBoostingRegressors.
    Automatically routes each prediction to the correct model.

    Usage
    -----
    model = PriceModel.from_state()

    # Booking.com hotel
    model.predict(source="Booking.com", rating=7.5, reviews_count=300,
                  stay_duration=3, checkin_month=5, checkin_dayofweek=3,
                  latitude=24.77, longitude=46.65,
                  location_cluster=3, cluster_median_price=400,
                  competition_density=20, is_saudi_weekend=1)

    # TripAdvisor hotel
    model.predict(source="TripAdvisor", rating=9.5, reviews_count=1500,
                  stay_duration=2, ...)
    """

    def __init__(self, models: dict):
        # models = {"Booking.com": {"model": ..., "feature_cols": [...]}, ...}
        self._models = models

    @classmethod
    def from_state(cls) -> "PriceModel":
        """Load the Booking.com model and hotel encoding map."""
        path = os.path.join(STATE_DIR, "model_booking.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "model_booking.pkl not found. Run the pipeline first:\n"
                "  python run_pipeline.py  or  python main.py"
            )
        with open(path, "rb") as f:
            models = {"Booking.com": pickle.load(f)}

        # load hotel target-encoding map
        enc_path = os.path.join(STATE_DIR, "hotel_encoding.json")
        hotel_enc = {}
        if os.path.exists(enc_path):
            import json
            with open(enc_path) as f:
                hotel_enc = json.load(f)

        obj = cls(models)
        obj._hotel_map    = hotel_enc.get("hotel_map", {})
        obj._global_mean  = hotel_enc.get("global_mean", 400.0)
        return obj

    def predict(self, source: str = "Booking.com", **kwargs) -> dict:
        """
        Predict nightly rate and total price for a specific platform.

        Parameters
        ----------
        source       : "Booking.com" or "TripAdvisor"
        rating       : guest rating (0–10)
        reviews_count: number of reviews
        stay_duration: number of nights
        checkin_month: 1–12
        checkin_dayofweek: 0=Mon … 6=Sun
        latitude, longitude: hotel coordinates
        location_cluster   : 0–7 (from KMeans)
        cluster_median_price: median price in the hotel's zone
        competition_density : hotels within ~2 km
        is_saudi_weekend    : 1 if Thu/Fri check-in

        Returns
        -------
        dict: price_per_night (SAR), total_price (SAR), platform
        """
        # support source_encoded integer as well
        if isinstance(source, int):
            source = SOURCE_MAP.get(source, "Booking.com")

        # always use Booking.com model as the base
        saved        = self._models["Booking.com"]
        model        = saved["model"]
        feature_cols = saved["feature_cols"]

        # auto-compute derived features if not provided
        reviews_count = kwargs.get("reviews_count", 0)
        rating        = kwargs.get("rating", 0)
        log_rc        = np.log1p(reviews_count)

        kwargs.setdefault("log_reviews_count",  log_rc)
        kwargs.setdefault("rating_trust_score", rating * log_rc)

        # hotel_mean_price: look up from encoding map, fallback to global mean
        if "hotel_mean_price" not in kwargs:
            hotel_name = kwargs.get("hotel_name", "")
            kwargs["hotel_mean_price"] = self._hotel_map.get(hotel_name, self._global_mean)

        row = {col: kwargs.get(col, 0) for col in feature_cols}
        X   = pd.DataFrame([row], columns=feature_cols)

        log_pred        = model.predict(X)[0]
        price_per_night = float(np.expm1(log_pred))
        # Apply per-tier bias correction
        tier_bias = saved.get("tier_bias", {})
        if price_per_night < 300:
            price_per_night -= tier_bias.get("budget", 0.0)
        elif price_per_night < 800:
            price_per_night -= tier_bias.get("mid", 0.0)
        else:
            price_per_night -= tier_bias.get("luxury", 0.0)

        stay_duration = int(kwargs.get("stay_duration", 1))
        total_price   = price_per_night * stay_duration

        print(f"[{source}] Price/night : {price_per_night:,.0f} SAR")
        print(f"[{source}] Total ({stay_duration}n): {total_price:,.0f} SAR")

        return {
            "platform":       source,
            "price_per_night": round(price_per_night, 2),
            "total_price":     round(total_price, 2),
        }

    @property
    def platforms(self) -> list:
        return list(self._models.keys())
