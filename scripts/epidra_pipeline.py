"""
EPIDRA - Epidemic Risk Assessment Pipeline
============================================
Step 1: Data Collection + Feature Engineering + ML Model + Advanced SHAP

This script:
  1. Fetches real weather data from Open-Meteo API (75 Indian cities, last 120 days)
  2. Engineers meaningful features (rolling stats, interactions, trends)
  3. Labels epidemic risk (Low / Medium / High) based on weather conditions
  4. Trains a high-quality XGBoost classifier with proper evaluation
  5. Performs advanced SHAP analysis with TreeExplainer
  6. Saves all outputs: dataset.csv, model.pkl, shap_profiles.json, plots
"""

import os
import sys
import json
import time
import warnings
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EPIDRA")

# ──────────────────────  PATHS  ──────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────  CITY LIST (75 Indian cities — India only)  ──────────────────────
CITIES = [
    # ══════════ NORTHEAST INDIA — Dense Coverage (17 cities) ══════════
    # Assam (5)
    {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362},
    {"name": "Dibrugarh", "lat": 27.4728, "lon": 94.9120},
    {"name": "Silchar", "lat": 24.8333, "lon": 92.7789},
    {"name": "Tezpur", "lat": 26.6528, "lon": 92.7926},
    {"name": "Tinsukia", "lat": 27.4898, "lon": 95.3599},
    # Manipur (2)
    {"name": "Imphal", "lat": 24.8170, "lon": 93.9368},
    {"name": "Thoubal", "lat": 24.6386, "lon": 93.9974},
    # Meghalaya (2)
    {"name": "Shillong", "lat": 25.5788, "lon": 91.8933},
    {"name": "Tura", "lat": 25.5142, "lon": 90.2021},
    # Mizoram (2)
    {"name": "Aizawl", "lat": 23.7271, "lon": 92.7176},
    {"name": "Lunglei", "lat": 22.88, "lon": 92.73},
    # Nagaland (2)
    {"name": "Kohima", "lat": 25.6751, "lon": 94.1086},
    {"name": "Dimapur", "lat": 25.9063, "lon": 93.7276},
    # Arunachal Pradesh (2)
    {"name": "Itanagar", "lat": 27.0844, "lon": 93.6053},
    {"name": "Naharlagun", "lat": 27.1047, "lon": 93.6950},
    # Tripura (2)
    {"name": "Agartala", "lat": 23.8315, "lon": 91.2868},
    {"name": "Udaipur_Tripura", "lat": 23.5333, "lon": 91.4833},

    # ══════════ REST OF INDIA — Balanced Coverage (58 cities) ══════════
    # Major Metros (15 — as specified)
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
    {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245},
    {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
    {"name": "Indore", "lat": 22.7196, "lon": 75.8577},
    {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558},

    # North India (10)
    {"name": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    {"name": "Amritsar", "lat": 31.6340, "lon": 74.8723},
    {"name": "Dehradun", "lat": 30.3165, "lon": 78.0322},
    {"name": "Shimla", "lat": 31.1048, "lon": 77.1734},
    {"name": "Jammu", "lat": 32.7266, "lon": 74.8570},
    {"name": "Srinagar", "lat": 34.0837, "lon": 74.7973},
    {"name": "Varanasi", "lat": 25.3176, "lon": 82.9739},
    {"name": "Agra", "lat": 27.1767, "lon": 78.0081},
    {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319},
    {"name": "Allahabad", "lat": 25.4358, "lon": 81.8463},

    # West India (8)
    {"name": "Surat", "lat": 21.1702, "lon": 72.8311},
    {"name": "Vadodara", "lat": 22.3072, "lon": 73.1812},
    {"name": "Rajkot", "lat": 22.3039, "lon": 70.8022},
    {"name": "Nashik", "lat": 19.9975, "lon": 73.7898},
    {"name": "Aurangabad", "lat": 19.8762, "lon": 75.3433},
    {"name": "Goa_Panaji", "lat": 15.4909, "lon": 73.8278},
    {"name": "Udaipur", "lat": 24.5854, "lon": 73.7125},
    {"name": "Jodhpur", "lat": 26.2389, "lon": 73.0243},

    # South India (10)
    {"name": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673},
    {"name": "Mangalore", "lat": 12.9141, "lon": 74.8560},
    {"name": "Mysore", "lat": 12.2958, "lon": 76.6394},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    {"name": "Vijayawada", "lat": 16.5062, "lon": 80.6480},
    {"name": "Tiruchirappalli", "lat": 10.7905, "lon": 78.7047},
    {"name": "Madurai", "lat": 9.9252, "lon": 78.1198},
    {"name": "Hubli", "lat": 15.3647, "lon": 75.1240},
    {"name": "Tirupati", "lat": 13.6288, "lon": 79.4192},

    # Central India (7)
    {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126},
    {"name": "Jabalpur", "lat": 23.1815, "lon": 79.9864},
    {"name": "Raipur", "lat": 21.2514, "lon": 81.6296},
    {"name": "Bilaspur", "lat": 22.0797, "lon": 82.1409},
    {"name": "Gwalior", "lat": 26.2183, "lon": 78.1828},
    {"name": "Ujjain", "lat": 23.1765, "lon": 75.7885},
    {"name": "Rewa", "lat": 24.5373, "lon": 81.3042},

    # East India (8)
    {"name": "Ranchi", "lat": 23.3441, "lon": 85.3096},
    {"name": "Jamshedpur", "lat": 22.8046, "lon": 86.2029},
    {"name": "Cuttack", "lat": 20.4625, "lon": 85.8830},
    {"name": "Durgapur", "lat": 23.5204, "lon": 87.3119},
    {"name": "Siliguri", "lat": 26.7271, "lon": 88.3953},
    {"name": "Asansol", "lat": 23.6888, "lon": 86.9661},
    {"name": "Dhanbad", "lat": 23.7957, "lon": 86.4304},
    {"name": "Gangtok", "lat": 27.3389, "lon": 88.6065},
]

log.info(f"Total cities: {len(CITIES)}")

# ════════════════════════════════════════════════════════════════
# 1. DATA COLLECTION — Open-Meteo API
# ════════════════════════════════════════════════════════════════

def fetch_weather_data() -> pd.DataFrame:
    """
    Fetch real weather data from Open-Meteo API.
    Returns a DataFrame with daily weather observations for all cities
    over the last 120 days.
    """
    end_date = datetime.now().date() - timedelta(days=1)  # yesterday
    start_date = end_date - timedelta(days=119)  # 120 days total

    log.info(f"Fetching data from {start_date} to {end_date} ({120} days)")

    all_records = []
    failed_cities = []

    for i, city in enumerate(CITIES):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": str(start_date),
            "end_date": str(end_date),
            "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,relative_humidity_2m_max,relative_humidity_2m_min",
            "timezone": "auto",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            n_days = len(dates)

            for j in range(n_days):
                record = {
                    "city": city["name"],
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "date": dates[j],
                    "precipitation": daily.get("precipitation_sum", [None] * n_days)[j],
                    "temperature_mean": daily.get("temperature_2m_mean", [None] * n_days)[j],
                    "temperature_max": daily.get("temperature_2m_max", [None] * n_days)[j],
                    "temperature_min": daily.get("temperature_2m_min", [None] * n_days)[j],
                    "humidity_mean": daily.get("relative_humidity_2m_mean", [None] * n_days)[j],
                    "humidity_max": daily.get("relative_humidity_2m_max", [None] * n_days)[j],
                    "humidity_min": daily.get("relative_humidity_2m_min", [None] * n_days)[j],
                }
                all_records.append(record)

            log.info(f"  [{i+1}/{len(CITIES)}] {city['name']}: {n_days} days fetched")

        except Exception as e:
            failed_cities.append(city["name"])
            log.warning(f"  [{i+1}/{len(CITIES)}] {city['name']}: FAILED — {e}")

        # Respect rate limits: small pause every 10 cities
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    if failed_cities:
        log.warning(f"Failed cities ({len(failed_cities)}): {failed_cities}")

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Raw data collected: {len(df)} rows from {df['city'].nunique()} cities")
    return df


# ════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful, epidemiologically-relevant features from raw weather data.
    Features:
      - rainfall_7d_sum       : Total precipitation over last 7 days
      - rainfall_3d_avg       : Average daily precipitation over last 3 days
      - temperature_avg       : Mean temperature (already present, alias)
      - temperature_trend     : Slope of temperature over last 5 days
      - humidity_max          : Already present
      - humidity_avg          : Already present (humidity_mean)
      - rainfall_humidity_interaction : rainfall_7d * humidity_avg (interaction)
      - temperature_range     : Daily temperature range (max - min)
      - humidity_range        : Daily humidity range (max - min)
    """
    log.info("Engineering features...")

    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Fill missing values with city-level median, then global median
    numeric_cols = [
        "precipitation", "temperature_mean", "temperature_max",
        "temperature_min", "humidity_mean", "humidity_max", "humidity_min",
    ]
    for col in numeric_cols:
        df[col] = df.groupby("city")[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())

    # Rolling features (per city)
    df["rainfall_7d_sum"] = df.groupby("city")["precipitation"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )
    df["rainfall_3d_avg"] = df.groupby("city")["precipitation"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Temperature trend: slope over last 5 days (vectorized)
    def compute_rolling_slope(group, window=5):
        """Vectorized rolling slope using diff-based approximation."""
        vals = group.values.astype(float)
        # Use simple difference-based slope: (current - value_N_days_ago) / N
        shifted = group.shift(window - 1)
        slope = (group - shifted) / (window - 1)
        # For first few rows, use available window
        for i in range(1, window - 1):
            mask = slope.isna()
            if not mask.any():
                break
            shifted_i = group.shift(i)
            slope = slope.fillna((group - shifted_i) / max(i, 1))
        slope = slope.fillna(0.0)
        return slope

    df["temperature_trend"] = df.groupby("city")["temperature_mean"].transform(
        compute_rolling_slope
    )

    # Rename for clarity
    df["temperature_avg"] = df["temperature_mean"]
    df["humidity_avg"] = df["humidity_mean"]

    # Derived features
    df["temperature_range"] = df["temperature_max"] - df["temperature_min"]
    df["humidity_range"] = df["humidity_max"] - df["humidity_min"]

    # Interaction feature: rainfall × humidity drives vector breeding
    df["rainfall_humidity_interaction"] = df["rainfall_7d_sum"] * df["humidity_avg"]

    # Drop rows where critical features are still NaN
    feature_cols = [
        "rainfall_7d_sum", "rainfall_3d_avg", "temperature_avg",
        "temperature_trend", "humidity_max", "humidity_avg",
        "rainfall_humidity_interaction", "temperature_range", "humidity_range",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    log.info(f"Features engineered. Dataset shape: {df.shape}")
    log.info(f"Feature columns: {feature_cols}")
    return df


# ════════════════════════════════════════════════════════════════
# 3. LABELING — Epidemiologically meaningful risk levels
# ════════════════════════════════════════════════════════════════

def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign epidemic risk labels based on weather conditions.

    Logic (based on epidemiological literature):
    ─────────────────────────────────────────────
    HIGH risk conditions (vector-borne disease outbreaks):
      • Sustained heavy rainfall (7d sum > 50mm) → standing water / breeding sites
      • High humidity (avg > 75%) → mosquito survival
      • Warm temperatures (20–35°C) → optimal vector activity

    MEDIUM risk:
      • Moderate rainfall (7d sum 15–50mm)
      • Moderate humidity (55–75%)
      • Temperature in sub-optimal range

    LOW risk:
      • Low rainfall, low humidity, or extreme temperatures
      • Conditions unfavorable for vector breeding

    A composite score is computed and thresholded to ensure balanced classes.
    """
    log.info("Assigning risk labels...")

    # Normalize features to [0, 1] for scoring
    def normalize(series):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        return (series - s_min) / (s_max - s_min)

    # Rainfall score: higher rainfall → higher risk
    rainfall_norm = normalize(df["rainfall_7d_sum"])

    # Humidity score: higher humidity → higher risk
    humidity_norm = normalize(df["humidity_avg"])

    # Temperature score: peak risk at 25-30°C, lower at extremes
    temp = df["temperature_avg"]
    temp_score = np.exp(-0.5 * ((temp - 27.5) / 7.0) ** 2)  # Gaussian centered at 27.5°C

    # Interaction score
    interaction_norm = normalize(df["rainfall_humidity_interaction"])

    # Composite risk score (weighted)
    risk_score = (
        0.30 * rainfall_norm        # Rainfall is dominant factor
        + 0.25 * humidity_norm       # Humidity is important
        + 0.20 * temp_score          # Temperature modulates risk
        + 0.25 * interaction_norm    # Interaction captures synergy
    )

    # Threshold into classes (aim for ~30% High, ~40% Medium, ~30% Low)
    q_low = risk_score.quantile(0.33)
    q_high = risk_score.quantile(0.67)

    df["risk_score"] = risk_score
    df["risk_label"] = pd.cut(
        risk_score,
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Log distribution
    dist = df["risk_label"].value_counts()
    log.info(f"Risk label distribution:\n{dist}")

    return df


# ════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING — XGBoost Classifier
# ════════════════════════════════════════════════════════════════

FEATURE_COLUMNS = [
    "rainfall_7d_sum",
    "rainfall_3d_avg",
    "temperature_avg",
    "temperature_trend",
    "humidity_max",
    "humidity_avg",
    "rainfall_humidity_interaction",
    "temperature_range",
    "humidity_range",
]


def train_model(df: pd.DataFrame):
    """
    Train an XGBoost classifier for epidemic risk prediction.
    Returns: (model, label_encoder, X_test, y_test, accuracy)
    """
    log.info("Training XGBoost model...")

    X = df[FEATURE_COLUMNS].values
    le = LabelEncoder()
    y = le.fit_transform(df["risk_label"].astype(str))

    log.info(f"Classes: {le.classes_}")
    log.info(f"Feature matrix shape: {X.shape}")

    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost parameters tuned for this task
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "num_class": len(le.classes_),
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(**params)

    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    log.info(f"\n{'='*50}")
    log.info(f"MODEL PERFORMANCE")
    log.info(f"{'='*50}")
    log.info(f"Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    log.info(f"F1 Score : {f1:.4f}")
    log.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save model + label encoder
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump({"model": model, "label_encoder": le, "features": FEATURE_COLUMNS}, model_path)
    log.info(f"Model saved to {model_path}")

    return model, le, X_train, X_test, y_test, accuracy


# ════════════════════════════════════════════════════════════════
# 5. ADVANCED SHAP ANALYSIS
# ════════════════════════════════════════════════════════════════

def run_shap_analysis(model, le, X_train, X_test, y_test):
    """
    Advanced SHAP analysis using TreeExplainer.

    - Computes SHAP values on a representative sample (not the entire dataset)
    - Generates per-class risk profiles with top contributing features
    - Creates SHAP summary and waterfall plots
    - Saves shap_profiles.json
    """
    log.info("Running Advanced SHAP Analysis...")

    # Use a representative sample for SHAP computation (max 500 samples)
    sample_size = min(500, len(X_test))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    log.info(f"SHAP computed on {sample_size} representative samples from test set")

    # TreeExplainer — optimized for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values shape: (n_classes, n_samples, n_features) or list of arrays
    if isinstance(shap_values, list):
        shap_values_array = np.array(shap_values)  # (n_classes, n_samples, n_features)
    else:
        shap_values_array = shap_values
        if shap_values_array.ndim == 3 and shap_values_array.shape[2] == len(le.classes_):
            # Shape is (n_samples, n_features, n_classes) — transpose
            shap_values_array = np.transpose(shap_values_array, (2, 0, 1))

    n_classes = len(le.classes_)

    # ─── Build SHAP Profiles per Risk Class ───
    shap_profiles = {}

    for cls_idx, cls_name in enumerate(le.classes_):
        log.info(f"  Analyzing SHAP for class: {cls_name}")

        # Get SHAP values for this class
        cls_shap = shap_values_array[cls_idx]  # (n_samples, n_features)

        # Filter to samples actually predicted as this class
        cls_mask = y_sample == cls_idx
        if cls_mask.sum() == 0:
            # Fallback: use all samples
            cls_shap_filtered = cls_shap
        else:
            cls_shap_filtered = cls_shap[cls_mask]

        # Mean absolute SHAP value per feature (importance)
        mean_abs_shap = np.mean(np.abs(cls_shap_filtered), axis=0)

        # Mean SHAP value per feature (direction)
        mean_shap = np.mean(cls_shap_filtered, axis=0)

        # Rank features by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        features_profile = []
        for rank, feat_idx in enumerate(sorted_idx):
            feat_name = FEATURE_COLUMNS[feat_idx]
            importance = float(mean_abs_shap[feat_idx])
            direction = "increases_risk" if mean_shap[feat_idx] > 0 else "decreases_risk"

            features_profile.append({
                "rank": rank + 1,
                "feature": feat_name,
                "importance": round(importance, 6),
                "direction": direction,
                "mean_shap_value": round(float(mean_shap[feat_idx]), 6),
            })

        shap_profiles[cls_name] = {
            "class_label": cls_name,
            "n_samples_analyzed": int(cls_mask.sum()) if cls_mask.sum() > 0 else sample_size,
            "top_features": features_profile[:5],  # Top 5 features
            "all_features": features_profile,
        }

    # Save SHAP profiles
    profiles_path = os.path.join(OUTPUT_DIR, "shap_profiles.json")
    with open(profiles_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "model": "XGBoost",
                    "explainer": "TreeExplainer",
                    "n_samples": sample_size,
                    "n_features": len(FEATURE_COLUMNS),
                    "n_classes": n_classes,
                    "classes": list(le.classes_),
                    "feature_names": FEATURE_COLUMNS,
                    "generated_at": datetime.now().isoformat(),
                },
                "profiles": shap_profiles,
            },
            f,
            indent=2,
        )
    log.info(f"SHAP profiles saved to {profiles_path}")

    # ─── SHAP Summary Plot ───
    log.info("Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # For multiclass: show mean |SHAP| per feature across all classes
    mean_abs_all = np.mean([np.mean(np.abs(shap_values_array[c]), axis=0) for c in range(n_classes)], axis=0)
    sorted_features = np.argsort(mean_abs_all)[::-1]

    colors = ["#E74C3C", "#F39C12", "#27AE60"]  # High, Medium, Low → Red, Orange, Green
    class_order = []
    for cls in ["High", "Medium", "Low"]:
        if cls in le.classes_:
            class_order.append(np.where(le.classes_ == cls)[0][0])

    bar_width = 0.25
    y_pos = np.arange(len(FEATURE_COLUMNS))

    for i, cls_idx in enumerate(class_order):
        cls_name = le.classes_[cls_idx]
        cls_importance = np.mean(np.abs(shap_values_array[cls_idx]), axis=0)
        bars = ax.barh(
            y_pos + i * bar_width,
            cls_importance[sorted_features],
            bar_width,
            label=f"{cls_name} Risk",
            color=colors[i],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_yticks(y_pos + bar_width)
    ax.set_yticklabels([FEATURE_COLUMNS[i] for i in sorted_features], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title("EPIDRA — SHAP Feature Importance by Risk Class", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    summary_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"SHAP summary plot saved to {summary_path}")

    # ─── SHAP Waterfall Plot (demo: one sample from each class) ───
    log.info("Generating SHAP waterfall plot...")

    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 8))
    if n_classes == 1:
        axes = [axes]

    for cls_idx, cls_name in enumerate(le.classes_):
        ax = axes[cls_idx]

        # Pick a representative sample for this class
        cls_mask = y_sample == cls_idx
        if cls_mask.sum() > 0:
            # Pick the sample with highest prediction confidence for this class
            probs = model.predict_proba(X_sample[cls_mask])
            best_local = np.argmax(probs[:, cls_idx])
            sample_idx = np.where(cls_mask)[0][best_local]
        else:
            sample_idx = 0

        sample_shap = shap_values_array[cls_idx][sample_idx]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]

        feat_names = [FEATURE_COLUMNS[i] for i in sorted_idx]
        feat_vals = sample_shap[sorted_idx]
        feat_data = X_sample[sample_idx][sorted_idx]

        colors_wf = ["#E74C3C" if v > 0 else "#3498DB" for v in feat_vals]

        bars = ax.barh(range(len(feat_names)), feat_vals, color=colors_wf, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(
            [f"{fn} = {fd:.1f}" for fn, fd in zip(feat_names, feat_data)],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel("SHAP Value", fontsize=10)
        ax.set_title(f"{cls_name} Risk — Sample Waterfall", fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "EPIDRA — SHAP Waterfall Analysis (Representative Samples)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    waterfall_path = os.path.join(OUTPUT_DIR, "shap_waterfall.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"SHAP waterfall plot saved to {waterfall_path}")

    return shap_profiles


# ════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main():
    """Run the complete EPIDRA pipeline."""
    start_time = time.time()

    log.info("=" * 60)
    log.info("  EPIDRA — Epidemic Risk Assessment Pipeline")
    log.info("  Step 1: Data + ML + SHAP")
    log.info("=" * 60)

    # ── Step 1: Data Collection ──
    log.info("\n[STEP 1/6] Collecting weather data from Open-Meteo API...")
    df = fetch_weather_data()

    # ── Step 2: Feature Engineering ──
    log.info("\n[STEP 2/6] Engineering features...")
    df = engineer_features(df)

    # ── Step 3: Risk Labeling ──
    log.info("\n[STEP 3/6] Assigning risk labels...")
    df = assign_risk_labels(df)

    # ── Save dataset ──
    dataset_path = os.path.join(DATA_DIR, "dataset.csv")
    df.to_csv(dataset_path, index=False)
    log.info(f"Dataset saved to {dataset_path} ({len(df)} rows, {df.shape[1]} columns)")

    # ── Step 4: Model Training ──
    log.info("\n[STEP 4/6] Training XGBoost model...")
    model, le, X_train, X_test, y_test, accuracy = train_model(df)

    # ── Step 5: SHAP Analysis ──
    log.info("\n[STEP 5/6] Running advanced SHAP analysis...")
    shap_profiles = run_shap_analysis(model, le, X_train, X_test, y_test)

    # ── Step 6: Summary ──
    elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"  PIPELINE COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Time elapsed    : {elapsed:.1f}s")
    log.info(f"  Dataset rows    : {len(df)}")
    log.info(f"  Model accuracy  : {accuracy*100:.1f}%")
    log.info(f"  Risk classes    : {list(le.classes_)}")
    log.info(f"")
    log.info(f"  Output files:")
    log.info(f"    ├── data/dataset.csv           ({len(df)} rows)")
    log.info(f"    ├── model/model.pkl")
    log.info(f"    ├── outputs/shap_profiles.json")
    log.info(f"    ├── outputs/shap_summary.png")
    log.info(f"    └── outputs/shap_waterfall.png")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
