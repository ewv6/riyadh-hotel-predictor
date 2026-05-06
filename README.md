# 🏨 Riyadh Hotel Price Predictor

A machine learning app that predicts nightly hotel prices in Riyadh, Saudi Arabia.

🔗 **[Live Demo](https://ewv6-riyadh-hotel-predictor.streamlit.app)**

---

## Overview

This project uses a **CrewAI multi-agent pipeline** to train a price prediction model on 8,074 Riyadh hotels. The trained model is served through a clean Streamlit web app.

---

## Demo

Enter a hotel name, rating, check-in date, and stay duration — the model returns the predicted nightly rate and total price in SAR.

---

## How It Works

```
hotel_name  ──►  historical avg price (target encoding)  ──┐
rating × log(reviews)  ──►  trust score  ──────────────────┤
lat/lon  ──►  KMeans cluster + neighbourhood median  ───────┤
check-in date  ──►  month, day, Saudi weekend flag  ────────┤
                                                            ▼
                              HistGradientBoostingRegressor
                                                            │
                                  log prediction ──► expm1 ──► bias correction
                                                            │
                                              price / night (SAR)
```

**Top features by importance:**
1. `hotel_mean_price` — hotel's own historical average (strongest signal)
2. `rating_trust_score` — rating × log(reviews count)
3. `stay_duration` — longer stays tend to have lower nightly rates

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML Model | scikit-learn `HistGradientBoostingRegressor` |
| Pipeline | CrewAI multi-agent (4 agents) |
| LLM | Groq `llama-3.3-70b-versatile` (free tier) |
| Web App | Streamlit |
| Dataset | [Riyadh Hotels 2026 — Kaggle](https://www.kaggle.com/datasets/mohammedalsubaie/riyadh-hotels-2026) |

---

## Run Locally

```bash
git clone https://github.com/ewv6/riyadh-hotel-predictor.git
cd riyadh-hotel-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

- **Source:** Kaggle — Riyadh Hotels 2026
- **Raw rows:** 9,849
- **After cleaning:** 8,074
- **Known hotels:** 2,039 with target-encoded historical prices

---

## Key Engineering Decisions

- **Target encoding for hotel names** — 5-fold KFold encoding of `hotel_name` into `hotel_mean_price` with no data leakage; became the #1 most important feature
- **Log-transform target** — raw price skewness was 36.24; after `log1p` it dropped to −1.13, dramatically improving model fit
- **Per-tier bias correction** — separate bias corrections for budget / mid / luxury tiers reduced mean bias from +22.9 SAR to +1.4 SAR
- **Geographic clustering** — KMeans(12) on lat/lon captures neighbourhood-level pricing patterns

---

## Project Structure

```
hotel_predictor/
├── app.py                  ← Streamlit web app
├── model.py                ← PriceModel inference class
├── requirements.txt
└── state/
    ├── model_booking.pkl       ← Trained model (1.9 MB)
    ├── hotel_encoding.json     ← 2,039 hotel name → price map
    └── kmeans.pkl              ← KMeans(12) for location clustering
```
