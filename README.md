# 🚦 Traffic Congestion Analysis & Prediction

A machine learning system that predicts real-time traffic congestion / demand levels for specific geographic zones using historical traffic patterns, weather conditions, road characteristics, and time-based features. The trained model is served through a **FastAPI** REST API for live inference.

## 📌 Overview

This project analyzes traffic data across geohashed locations and builds a predictive model to estimate traffic demand at a given place and time. It combines:

- **Geospatial encoding** of locations using geohashing
- **Temporal feature engineering** (lag features, rolling statistics) to capture time-based traffic trends
- **Categorical feature encoding** for road type, weather, landmarks, and vehicle restrictions
- **Gradient boosting (LightGBM)** for the prediction model
- **A FastAPI backend** that serves predictions via a simple REST endpoint

## ✨ Features

- Predicts traffic congestion/demand as a percentage for a given location and time
- Encodes location using **geohashing** for spatial generalization
- Engineers lag-based demand features (15-min, hourly, daily) and rolling mean/std to capture recent traffic trends
- Accounts for contextual factors: number of lanes, road type (Residential/Street/Highway), large vehicle restrictions, nearby landmarks, temperature, and weather conditions
- Exposes a production-ready REST API (`POST /traffic/`) built with FastAPI
- CORS-enabled for easy integration with frontend clients
- End-to-end workflow documented in a Jupyter Notebook (data exploration, feature engineering, model training/evaluation)

## 🗂️ Project Structure

```
Traffic-Congestion-Analysis/
├── dataset/                    # Traffic data (incl. lag/shift feature dataset)
├── model/                      # Serialized model & encoders (LightGBM model, label encoders, geohash processor)
├── test_model/                 # Model testing / evaluation scripts
├── util/
│   ├── preprocessor/
│   │   ├── labelenc.py         # Label/one-hot encoding utilities
│   │   └── temporal.py         # Time-based feature transformations
│   └── process_geohash.py      # Geohash encoding/decoding processor
├── Traffic_Congestion.ipynb    # EDA, feature engineering & model training notebook
├── app.py                      # FastAPI application serving predictions
└── requirement.txt             # Python dependencies
```

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Modeling | LightGBM, scikit-learn |
| Data Processing | pandas, NumPy |
| Geospatial | pygeohash |
| API | FastAPI, Pydantic |
| Serialization | Pickle |

## ⚙️ How It Works

1. **Data preparation** – Raw traffic records are transformed with temporal features (hour/day extraction) and lag/rolling-window demand statistics per geohash zone.
2. **Encoding** – Categorical fields (road type, weather, landmarks, large-vehicle policy, geohash) are label/one-hot encoded using fitted encoders saved in `model/`.
3. **Model training** – A LightGBM regressor is trained on the engineered feature set to predict traffic demand (see `Traffic_Congestion.ipynb`).
4. **Serving** – `app.py` loads the trained model and encoders at startup and exposes a `/traffic/` POST endpoint that accepts location and contextual details, reconstructs the feature vector (including live lag features pulled from the dataset), and returns a predicted congestion/demand score.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/Rajlxi29/Traffic-Congestion-Analysis.git
cd Traffic-Congestion-Analysis
pip install -r requirement.txt
```

### Run the API

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/traffic/" \
  -H "Content-Type: application/json" \
  -d '{
        "geohash": "tdr1w",
        "day": 3,
        "timestamp": "2026-07-15T14:30:00",
        "NumberofLanes": 2,
        "RoadType": "Highway",
        "LargeVehicles": "Allowed",
        "Landmarks": "Mall",
        "Temprature": 32.5,
        "Weather": "Sunny"
      }'
```

**Response:**
```json
{
  "Predicted_Demand": 62.14
}
```

## 📈 Future Improvements

- Add live traffic data ingestion (real-time streaming instead of static CSV lookups)
- Containerize the API with Docker for easier deployment
- Add a lightweight frontend/dashboard to visualize congestion by zone
- Expand model evaluation metrics and add automated tests

## 👤 Author

**Rajlxi29** — [GitHub Profile](https://github.com/Rajlxi29)

## 📄 License

This project currently has no license specified. Consider adding one (e.g., MIT) to clarify usage rights.
