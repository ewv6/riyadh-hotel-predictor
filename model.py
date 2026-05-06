"""
Standalone PriceModel — predicts nightly hotel prices in Riyadh.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")


class PriceModel:

    def __init__(self, saved: dict, hotel_map: dict, global_mean: float):
        self._saved      = saved
        self._hotel_map  = hotel_map
        self._global_mean = global_mean

    @classmethod
    def from_state(cls) -> "PriceModel":
        path = os.path.join(STATE_DIR, "model_booking.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError("model_booking.pkl not found. Run run_pipeline.py first.")
        with open(path, "rb") as f:
            saved = pickle.load(f)

        hotel_enc = {}
        enc_path  = os.path.join(STATE_DIR, "hotel_encoding.json")
        if os.path.exists(enc_path):
            with open(enc_path) as f:
                hotel_enc = json.load(f)

        return cls(saved, hotel_enc.get("hotel_map", {}), hotel_enc.get("global_mean", 400.0))

    def predict(self, **kwargs) -> dict:
        """
        Parameters
        ----------
        hotel_name          : str   (optional — improves accuracy)
        rating              : float (0–10)
        reviews_count       : int
        stay_duration       : int   (nights)
        checkin_month       : int   (1–12)
        checkin_dayofweek   : int   (0=Mon … 6=Sun)
        latitude, longitude : float
        location_cluster    : int   (0–11)
        cluster_median_price: float
        competition_density : int
        is_saudi_weekend    : int   (1 if Thu/Fri)

        Returns
        -------
        dict: price_per_night (SAR), total_price (SAR)
        """
        model        = self._saved["model"]
        feature_cols = self._saved["feature_cols"]

        reviews_count = kwargs.get("reviews_count", 0)
        rating        = kwargs.get("rating", 0)
        log_rc        = np.log1p(reviews_count)

        kwargs.setdefault("log_reviews_count",  log_rc)
        kwargs.setdefault("rating_trust_score", rating * log_rc)

        if "hotel_mean_price" not in kwargs:
            hotel_name = kwargs.get("hotel_name", "")
            kwargs["hotel_mean_price"] = self._hotel_map.get(hotel_name, self._global_mean)

        X = pd.DataFrame([{col: kwargs.get(col, 0) for col in feature_cols}], columns=feature_cols)

        price_per_night = float(np.expm1(model.predict(X)[0]))

        tier_bias = self._saved.get("tier_bias", {})
        if price_per_night < 300:
            price_per_night -= tier_bias.get("budget", 0.0)
        elif price_per_night < 800:
            price_per_night -= tier_bias.get("mid", 0.0)
        else:
            price_per_night -= tier_bias.get("luxury", 0.0)

        stay_duration = int(kwargs.get("stay_duration", 1))
        total_price   = price_per_night * stay_duration

        return {
            "price_per_night": round(price_per_night, 2),
            "total_price":     round(total_price, 2),
        }
