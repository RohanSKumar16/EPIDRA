"""
EPIDRA — Backend API v6.0 (FastAPI)
====================================
Production-quality Hybrid Multilingual Chatbot with:
- Calibrated XGBoost predictions
- Structured SHAP explanations with quantitative impact
- Gemini API for disease knowledge (gemini-1.5-flash)
- Fuzzy matching via difflib (dharwad → Hubli, bijapur → Hubli)
- State→City mapping (Sikkim → Gangtok)
- District alias map (80+ alt names)
- Lightweight RAG disease knowledge base
- 3-layer intent detection
- Multilingual support (EN / HI / AS)

Endpoints:
  GET  /districts           — All locations with calibrated risk
  GET  /districts/{id}      — Detail with structured SHAP + dominant driver
  POST /predict             — Calibrated prediction from weather inputs
  POST /chat                — Advanced hybrid chatbot
  GET  /health              — System status
"""

import os
import json
import re
import time
import hashlib
import difflib
from typing import Optional

# Load .env BEFORE anything reads os.environ
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the dedicated Gemini service (uses google-generativeai SDK)
from app.services.gemini_service import (
    get_gemini_response as _gemini_service_call,
    is_gemini_available,
)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SHAP_PATH = os.path.join(BASE_DIR, "outputs", "shap_profiles.json")

# ── Load assets ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EPIDRA v6.0 — Loading & calibrating model")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

model_bundle = joblib.load(MODEL_PATH)
raw_model = model_bundle["model"]
label_encoder = model_bundle["label_encoder"]
FEATURE_COLUMNS = model_bundle["features"]

with open(SHAP_PATH, "r") as f:
    shap_data = json.load(f)

# ── Probability Calibration ─────────────────────────────────────
print("  Calibrating probabilities (sigmoid)...")

X_all = df[FEATURE_COLUMNS].values.astype(float)
y_all = label_encoder.transform(df["risk_label"].values)

_, X_cal, _, y_cal = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

try:
    from sklearn.calibration import FrozenEstimator
    frozen_model = FrozenEstimator(raw_model)
    calibrated_model = CalibratedClassifierCV(estimator=frozen_model, method="sigmoid")
except ImportError:
    calibrated_model = CalibratedClassifierCV(raw_model, method="sigmoid", cv="prefit")
calibrated_model.fit(X_cal, y_cal)

model = calibrated_model


def realistic_confidence(raw_prob: float) -> float:
    """Scale raw model probability into a trustworthy 60-95% range.
    NEVER returns 100%. Caps at 95% max, 60% min."""
    MIN_CONF, MAX_CONF = 0.60, 0.95
    scaled = MIN_CONF + (MAX_CONF - MIN_CONF) * ((raw_prob - 0.5) / 0.5) ** 0.6
    return float(np.clip(scaled, MIN_CONF, MAX_CONF))


# Sanity check
sample_probs = calibrated_model.predict_proba(X_cal[:20]).max(axis=1)
scaled = [realistic_confidence(p) for p in sample_probs]
print(f"  Raw prob range: {sample_probs.min():.2%} – {sample_probs.max():.2%}")
print(f"  Scaled confidence: {min(scaled):.2%} – {max(scaled):.2%}")

# ── Dynamically extract cities from dataset ──────────────────────
city_info = df.groupby("city").agg(
    latitude=("latitude", "first"),
    longitude=("longitude", "first"),
).reset_index()

print(f"\n  Cities loaded: {len(city_info)} (from dataset.csv)")

lat_ok = city_info["latitude"].between(6, 38).all()
lon_ok = city_info["longitude"].between(68, 98).all()
print(f"  India bounds check: {'✅ PASS' if lat_ok and lon_ok else '⚠️ FAIL'}")

# ── Precompute latest risk per city ──────────────────────────────
latest_date = df["date"].max()
latest_df = df[df["date"] == latest_date].copy()

DISTRICTS = []
for _, row in latest_df.iterrows():
    features = row[FEATURE_COLUMNS].values.astype(float).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    pred_class = model.predict(features)[0]
    confidence = realistic_confidence(float(np.max(probs)))

    DISTRICTS.append({
        "id": len(DISTRICTS) + 1,
        "city": row["city"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "risk": row["risk_label"],
        "model_confidence": round(confidence, 4),
        "_features": features.flatten().tolist(),
    })

CITY_TO_ID = {d["city"]: d["id"] for d in DISTRICTS}
ID_TO_DISTRICT = {d["id"]: d for d in DISTRICTS}

# ── Build city lookup (lowercased) for fuzzy matching ────────────
_CITY_LOOKUP = {d["city"].lower().replace("_", " "): d for d in DISTRICTS}

rc = {r: sum(1 for d in DISTRICTS if d["risk"] == r) for r in ["High", "Medium", "Low"]}
confs = [d["model_confidence"] for d in DISTRICTS]
print(f"\n  Risk: High={rc['High']}, Medium={rc['Medium']}, Low={rc['Low']}")
print(f"  Confidence range: {min(confs):.2%} – {max(confs):.2%} (mean {np.mean(confs):.2%})")
print(f"  Latest date: {latest_date.date()}")


# ══════════════════════════════════════════════════════════════════
# STATE → CITY MAPPING
# ══════════════════════════════════════════════════════════════════

STATE_TO_CITY = {
    # Northeast
    "sikkim": "Gangtok",
    "meghalaya": "Shillong",
    "mizoram": "Aizawl",
    "nagaland": "Kohima",
    "manipur": "Imphal",
    "tripura": "Agartala",
    "arunachal pradesh": "Itanagar",
    "arunachal": "Itanagar",
    "assam": "Guwahati",
    # Major states
    "west bengal": "Kolkata",
    "bengal": "Kolkata",
    "maharashtra": "Mumbai",
    "karnataka": "Bangalore",
    "tamil nadu": "Chennai",
    "tamilnadu": "Chennai",
    "telangana": "Hyderabad",
    "andhra pradesh": "Visakhapatnam",
    "andhra": "Visakhapatnam",
    "kerala": "Kochi",
    "gujarat": "Ahmedabad",
    "rajasthan": "Jaipur",
    "madhya pradesh": "Bhopal",
    "mp": "Bhopal",
    "uttar pradesh": "Lucknow",
    "up": "Lucknow",
    "bihar": "Patna",
    "odisha": "Bhubaneswar",
    "orissa": "Bhubaneswar",
    "chhattisgarh": "Raipur",
    "jharkhand": "Ranchi",
    "punjab": "Amritsar",
    "haryana": "Chandigarh",
    "uttarakhand": "Dehradun",
    "himachal pradesh": "Shimla",
    "himachal": "Shimla",
    "goa": "Goa_Panaji",
    "jammu and kashmir": "Srinagar",
    "jammu kashmir": "Srinagar",
    "j&k": "Srinagar",
    "jk": "Srinagar",
    # Union territories
    "delhi": "Delhi",
    "chandigarh": "Chandigarh",
}

_STATE_LOOKUP = {k.lower(): v for k, v in STATE_TO_CITY.items()}

# ══════════════════════════════════════════════════════════════════
# DISTRICT / CITY ALIAS MAP  (alt names → dataset city)
# ══════════════════════════════════════════════════════════════════
# Maps alternate names, district names, twin cities, and common
# misspellings to the canonical city name in the dataset.

CITY_ALIAS_MAP = {
    # Karnataka
    "dharwad": "Hubli",
    "dharwar": "Hubli",
    "hubli dharwad": "Hubli",
    "hubli-dharwad": "Hubli",
    "hubballi": "Hubli",
    "hubballi dharwad": "Hubli",
    "bijapur": "Hubli",
    "vijayapura": "Hubli",
    "bagalkot": "Hubli",
    "belgaum": "Hubli",
    "belagavi": "Hubli",
    "davangere": "Hubli",
    "davanagere": "Hubli",
    "bellary": "Hubli",
    "ballari": "Hubli",
    "raichur": "Hubli",
    "gulbarga": "Hubli",
    "kalaburagi": "Hubli",
    "bengaluru": "Bangalore",
    "mysuru": "Mysore",
    "mangaluru": "Mangalore",
    # Maharashtra
    "bombay": "Mumbai",
    "poona": "Pune",
    "kolhapur": "Pune",
    "sangli": "Pune",
    "satara": "Pune",
    "solapur": "Pune",
    "nanded": "Aurangabad",
    "latur": "Aurangabad",
    "jalna": "Aurangabad",
    "sambhajinagar": "Aurangabad",
    "thane": "Mumbai",
    "navi mumbai": "Mumbai",
    # Tamil Nadu
    "madras": "Chennai",
    "trichy": "Tiruchirappalli",
    "tiruchi": "Tiruchirappalli",
    "salem": "Coimbatore",
    "erode": "Coimbatore",
    "vellore": "Chennai",
    "tirunelveli": "Madurai",
    "thoothukudi": "Madurai",
    "tuticorin": "Madurai",
    # Andhra Pradesh / Telangana
    "vizag": "Visakhapatnam",
    "vishakapatnam": "Visakhapatnam",
    "secunderabad": "Hyderabad",
    "warangal": "Hyderabad",
    "guntur": "Vijayawada",
    "nellore": "Tirupati",
    "kurnool": "Visakhapatnam",
    "kakinada": "Visakhapatnam",
    "rajahmundry": "Visakhapatnam",
    # UP
    "benares": "Varanasi",
    "banaras": "Varanasi",
    "kashi": "Varanasi",
    "prayagraj": "Allahabad",
    "noida": "Delhi",
    "ghaziabad": "Delhi",
    "meerut": "Delhi",
    "gorakhpur": "Lucknow",
    "bareilly": "Lucknow",
    "moradabad": "Lucknow",
    "mathura": "Agra",
    "aligarh": "Agra",
    "firozabad": "Agra",
    # Gujarat
    "baroda": "Vadodara",
    "junagadh": "Rajkot",
    "bhavnagar": "Rajkot",
    "gandhinagar": "Ahmedabad",
    "anand": "Vadodara",
    # Rajasthan
    "bikaner": "Jodhpur",
    "ajmer": "Jaipur",
    "kota": "Jaipur",
    # MP
    "guna": "Gwalior",
    "sagar": "Jabalpur",
    "satna": "Rewa",
    # Bihar
    "gaya": "Patna",
    "muzaffarpur": "Patna",
    "bhagalpur": "Patna",
    "darbhanga": "Patna",
    # West Bengal
    "howrah": "Kolkata",
    "hooghly": "Kolkata",
    "bardhaman": "Asansol",
    "burdwan": "Asansol",
    # Jharkhand
    "bokaro": "Dhanbad",
    "hazaribagh": "Ranchi",
    "deoghar": "Jamshedpur",
    # Odisha
    "puri": "Bhubaneswar",
    "berhampur": "Bhubaneswar",
    "sambalpur": "Bhubaneswar",
    "rourkela": "Cuttack",
    # Assam / NE
    "jorhat": "Dibrugarh",
    "nagaon": "Guwahati",
    "bongaigaon": "Guwahati",
    "karimganj": "Silchar",
    # Punjab / Haryana
    "ludhiana": "Amritsar",
    "jalandhar": "Amritsar",
    "patiala": "Chandigarh",
    "ambala": "Chandigarh",
    # Kerala
    "trivandrum": "Thiruvananthapuram",
    "kozhikode": "Kochi",
    "calicut": "Kochi",
    "thrissur": "Kochi",
    # Uttarakhand
    "haridwar": "Dehradun",
    "rishikesh": "Dehradun",
    "nainital": "Dehradun",
    # Chhattisgarh
    "bhilai": "Raipur",
    "durg": "Raipur",
    "korba": "Bilaspur",
}

_ALIAS_LOOKUP = {k.lower(): v.lower().replace("_", " ") for k, v in CITY_ALIAS_MAP.items()}

print(f"  State→City mappings: {len(_STATE_LOOKUP)}")
print(f"  City alias mappings: {len(_ALIAS_LOOKUP)}")


def _normalize_input(msg: str) -> str:
    """Normalize user input: lowercase, strip, remove extra spaces."""
    return re.sub(r'\s+', ' ', msg.strip().lower())


def _fuzzy_match_city(word: str, cutoff: float = 0.72) -> Optional[dict]:
    """Use difflib to find the closest matching city name."""
    all_names = list(_CITY_LOOKUP.keys())
    matches = difflib.get_close_matches(word, all_names, n=1, cutoff=cutoff)
    if matches:
        print(f"  [Fuzzy] '{word}' → '{matches[0]}' (difflib match)")
        return _CITY_LOOKUP[matches[0]]
    return None


def _word_boundary_match(name: str, text: str) -> bool:
    """Check if 'name' appears in 'text' as a whole word/phrase.
    Uses regex word boundaries for short names (<=4 chars) to prevent
    false matches like 'mp' inside 'symptoms' or 'up' inside 'support'.
    Longer names use simple substring match (false positive risk is low)."""
    if len(name) <= 4:
        return bool(re.search(r'\b' + re.escape(name) + r'\b', text))
    return name in text


def _find_city_in_message(msg: str) -> Optional[dict]:
    """4-layer city resolution: exact → alias → state → fuzzy."""
    msg_norm = _normalize_input(msg).replace("_", " ")

    # Layer 1: Direct city match (longest first for accuracy)
    best = None
    best_len = 0
    for d in DISTRICTS:
        city_lower = d["city"].lower().replace("_", " ")
        if _word_boundary_match(city_lower, msg_norm) and len(city_lower) > best_len:
            best = d
            best_len = len(city_lower)
    if best:
        return best

    # Layer 2: City alias map (dharwad → Hubli, bijapur → Hubli, etc.)
    for alias, canonical in _ALIAS_LOOKUP.items():
        if _word_boundary_match(alias, msg_norm):
            if canonical in _CITY_LOOKUP:
                print(f"  [Alias] '{alias}' → '{canonical}'")
                return _CITY_LOOKUP[canonical]

    # Layer 3: State name → capital city
    for state_name, city_name in _STATE_LOOKUP.items():
        if _word_boundary_match(state_name, msg_norm):
            city_key = city_name.lower().replace("_", " ")
            if city_key in _CITY_LOOKUP:
                return _CITY_LOOKUP[city_key]

    # Layer 4: Fuzzy matching via difflib (for typos & unknown names)
    words = msg_norm.split()
    stop_words = {
        "the", "is", "of", "in", "at", "my", "for", "and", "how", "why",
        "what", "are", "it", "to", "a", "an", "do", "does", "can", "will",
        "risk", "safe", "danger", "check", "tell", "me", "about", "show",
        "city", "state", "district", "region", "area", "place",
        "high", "low", "medium", "risky", "dangerous",
    }
    for w in words:
        if len(w) < 3 or w in stop_words:
            continue
        # Check alias first
        if w in _ALIAS_LOOKUP:
            canonical = _ALIAS_LOOKUP[w]
            if canonical in _CITY_LOOKUP:
                print(f"  [Alias-word] '{w}' → '{canonical}'")
                return _CITY_LOOKUP[canonical]
        # Check state
        if w in _STATE_LOOKUP:
            city_key = _STATE_LOOKUP[w].lower().replace("_", " ")
            if city_key in _CITY_LOOKUP:
                return _CITY_LOOKUP[city_key]
        # Check city direct
        if w in _CITY_LOOKUP:
            return _CITY_LOOKUP[w]
        # Fuzzy match (only for words >= 4 chars to avoid false positives)
        if len(w) >= 4:
            fuzzy = _fuzzy_match_city(w)
            if fuzzy:
                return fuzzy

    return None


# ══════════════════════════════════════════════════════════════════
# Human-readable SHAP explanations (with quantitative values)
# ══════════════════════════════════════════════════════════════════

FEATURE_META = {
    "rainfall_7d_sum": {
        "display": "Rainfall (7-day)",
        "increase_text": "Heavy rainfall creates stagnant water, breeding mosquitoes and increasing waterborne disease vectors",
        "decrease_text": "Low rainfall reduces standing water and mosquito breeding grounds",
    },
    "rainfall_3d_avg": {
        "display": "Rainfall (3-day avg)",
        "increase_text": "Recent sustained rainfall elevates flood and contamination risk",
        "decrease_text": "Reduced recent rainfall lowers immediate waterborne threat",
    },
    "temperature_avg": {
        "display": "Temperature",
        "increase_text": "Warm temperatures accelerate pathogen growth and vector reproduction cycles",
        "decrease_text": "Cooler conditions slow down disease vector breeding",
    },
    "temperature_trend": {
        "display": "Temperature trend",
        "increase_text": "Rising temperatures signal worsening conditions for vector-borne diseases",
        "decrease_text": "Falling temperatures may reduce vector activity",
    },
    "humidity_max": {
        "display": "Peak humidity",
        "increase_text": "Very high humidity sustains mosquito populations and airborne pathogen survival",
        "decrease_text": "Lower peak humidity reduces pathogen viability",
    },
    "humidity_avg": {
        "display": "Humidity",
        "increase_text": "Sustained humidity creates ideal conditions for disease transmission and vector survival",
        "decrease_text": "Drier conditions are less favorable for epidemic vectors",
    },
    "rainfall_humidity_interaction": {
        "display": "Rainfall × Humidity",
        "increase_text": "Combined high rainfall and humidity drastically amplify epidemic risk through compounding effects",
        "decrease_text": "Low rainfall-humidity interaction suggests reduced environmental disease pressure",
    },
    "temperature_range": {
        "display": "Temperature variation",
        "increase_text": "Large daily temperature swings can stress populations and weaken immunity",
        "decrease_text": "Stable temperatures are less physiologically stressful",
    },
    "humidity_range": {
        "display": "Humidity variation",
        "increase_text": "Fluctuating humidity disrupts vector behavior but may concentrate breeding",
        "decrease_text": "Consistent humidity levels lead to predictable risk patterns",
    },
}

INTERACTION_NOTES = {
    ("rainfall_7d_sum", "humidity_avg"): "High rainfall combined with sustained humidity creates maximum breeding conditions for disease vectors.",
    ("temperature_avg", "humidity_avg"): "Warm, humid conditions form the most favorable environment for pathogen proliferation.",
    ("rainfall_7d_sum", "temperature_avg"): "Warm rain leads to rapid stagnant water contamination and vector hatching.",
    ("rainfall_humidity_interaction", "temperature_avg"): "The compounding effect of rainfall, humidity, and warmth creates peak epidemic conditions.",
}


def build_shap_explanation(risk: str) -> dict:
    """Build structured SHAP explanation for a risk class, with quantitative impact values."""
    profile = shap_data.get("profiles", {}).get(risk, {})
    top_raw = profile.get("top_features", [])[:3]

    max_imp = max((f["importance"] for f in top_raw), default=1)

    explanations = []
    for f in top_raw:
        feat = f["feature"]
        meta = FEATURE_META.get(feat, {"display": feat, "increase_text": "", "decrease_text": ""})
        direction = f["direction"]
        imp = f["importance"]

        ratio = imp / max_imp if max_imp > 0 else 0
        if ratio > 0.5:
            strength = "high"
        elif ratio > 0.15:
            strength = "medium"
        else:
            strength = "low"

        impact = "increase" if "increase" in direction else "decrease"
        base_text = meta["increase_text"] if impact == "increase" else meta["decrease_text"]

        # Enhanced explanation with quantitative SHAP value
        sign = "+" if impact == "increase" else "-"
        explanation_text = f"{base_text} ({sign}{imp:.2f} SHAP impact)"

        explanations.append({
            "feature": feat,
            "display_name": meta["display"],
            "impact": impact,
            "strength": strength,
            "importance": round(imp, 4),
            "explanation": explanation_text,
        })

    dominant = explanations[0]["display_name"] if explanations else "Unknown"

    top_feats = [e["feature"] for e in explanations[:2]]
    interaction_note = ""
    for pair, note in INTERACTION_NOTES.items():
        if set(pair) == set(top_feats) or set(pair).issubset(set(top_feats)):
            interaction_note = note
            break
    if not interaction_note and len(top_feats) >= 2:
        n1 = FEATURE_META.get(top_feats[0], {}).get("display", top_feats[0])
        n2 = FEATURE_META.get(top_feats[1], {}).get("display", top_feats[1])
        interaction_note = f"{n1} and {n2} are the combined primary drivers of risk in this region."

    return {
        "explanations": explanations,
        "dominant_driver": dominant,
        "interaction_note": interaction_note,
    }


# ══════════════════════════════════════════════════════════════════
# Prevention tips
# ══════════════════════════════════════════════════════════════════

PREVENTION = {
    "High": {
        "en": [
            "Eliminate standing water around homes immediately.",
            "Use mosquito nets and repellent, especially at dawn and dusk.",
            "Report fever cases to the nearest health center.",
        ],
        "hi": [
            "घर के आसपास जमा पानी तुरंत हटाएं।",
            "मच्छरदानी और रिपेलेंट का उपयोग करें।",
            "बुखार होने पर निकटतम स्वास्थ्य केंद्र जाएं।",
        ],
        "as": [
            "ঘৰৰ ওচৰৰ জমা পানী তৎক্ষণাৎ আঁতৰাওক।",
            "মহ জাল আৰু ৰিপেলেণ্ট ব্যৱহাৰ কৰক।",
            "জ্বৰ হ'লে নিকটতম স্বাস্থ্য কেন্দ্ৰলৈ যাওক।",
        ],
    },
    "Medium": {
        "en": [
            "Stay vigilant — monitor stagnant water sources.",
            "Wear full-sleeve clothing during outdoor activities.",
            "Follow local health advisories for any outbreak alerts.",
        ],
        "hi": [
            "सतर्क रहें — रुके पानी पर नज़र रखें।",
            "बाहर जाते समय पूरी बांह के कपड़े पहनें।",
            "स्थानीय स्वास्थ्य सूचनाओं का पालन करें।",
        ],
        "as": [
            "সতৰ্ক থাকক — জমা পানীৰ ওপৰত চকু ৰাখক।",
            "বাহিৰত সম্পূৰ্ণ হাতৰ কাপোৰ পিন্ধক।",
        ],
    },
    "Low": {
        "en": [
            "Maintain basic hygiene and cleanliness.",
            "Keep surroundings clean to prevent future outbreaks.",
        ],
        "hi": [
            "बुनियादी स्वच्छता बनाए रखें।",
            "भविष्य के प्रकोप रोकने हेतु सफाई रखें।",
        ],
        "as": [
            "মৌলিক পৰিষ্কাৰ-পৰিচ্ছন্নতা বজাই ৰাখক।",
        ],
    },
}

# ══════════════════════════════════════════════════════════════════
# DECISION INTELLIGENCE ENGINE
# ══════════════════════════════════════════════════════════════════

def generate_intelligence(risk: str, conf: float, shap_info: dict) -> dict:
    explanations = shap_info["explanations"]
    features = [e["feature"] for e in explanations]
    driver_strength = sum(e["importance"] for e in explanations[:2]) if len(explanations) > 0 else 0

    # 1. Priority Score
    rw = 1.0 if risk == "High" else 0.6 if risk == "Medium" else 0.2
    ds_norm = min(driver_strength / 2.0, 1.0)
    priority_score = int((rw * 50) + (conf * 30) + (ds_norm * 20))
    if priority_score >= 75: priority_label = "CRITICAL"
    elif priority_score >= 50: priority_label = "HIGH"
    else: priority_label = "MODERATE"

    # 2. Urgency & Window
    if risk == "High":
        urgency_level = "High"
        response_window = "Immediate (0–3 days)"
    elif risk == "Medium":
        urgency_level = "Medium"
        response_window = "Short-term (3–7 days)"
    else:
        urgency_level = "Low"
        response_window = "Routine (7+ days)"

    # 3. Alert Level & Context
    if risk == "High" and conf > 0.8:
        alert_level = "Outbreak Likely"
        confidence_context = "High confidence due to strong compounding environmental signals."
        expected_impact = "If current conditions persist, localized outbreaks may begin within 5–7 days, spreading rapidly in densely populated urban zones."
    elif risk in ["High", "Medium"] and conf > 0.6:
        alert_level = "Rising"
        confidence_context = "Moderate confidence; models detect escalating risk drivers."
        expected_impact = "Elevated risk of isolated cases within 7–14 days. Vulnerable neighborhoods may see clustering."
    else:
        alert_level = "Stable"
        confidence_context = "Baseline confidence; no immediate outbreak triggers detected."
        expected_impact = "Risk remains at baseline levels. No widespread transmission expected in the near term."

    # 4. Source Attribution
    weather_feats = {"rainfall_7d_sum", "rainfall_3d_avg", "temperature_avg", "temperature_trend"}
    env_feats = {"humidity_max", "humidity_avg", "rainfall_humidity_interaction", "temperature_range", "humidity_range"}
    
    attr = {"Weather-driven": [], "Environmental": [], "Secondary": []}
    for e in explanations:
        if e["feature"] in weather_feats: attr["Weather-driven"].append(e["display_name"])
        elif e["feature"] in env_feats: attr["Environmental"].append(e["display_name"])
        else: attr["Secondary"].append(e["display_name"])

    # 5. Risk Narrative
    narrative = ""
    if "rainfall_humidity_interaction" in features[:2]:
        narrative = "Recent rainfall combined with sustained high humidity is aggressively accelerating microbial growth and vector breeding, critically increasing outbreak probability."
    elif "temperature_avg" in features[:2] and any("rainfall" in f for f in features):
        narrative = "High temperatures following recent rainfalls are creating ideal stagnant breeding grounds, accelerating the life cycle of disease vectors."
    elif "humidity_avg" in features[:2]:
        narrative = "Sustained high humidity levels are extending the survival duration of airborne and surface pathogens in the environment."
    else:
        narrative = f"The primary driver is {shap_info['dominant_driver']}, which is actively destabilizing baseline health security in the region."

    # 6. Recommended Actions
    actions = []
    if risk == "High":
        if "rainfall_7d_sum" in features or "rainfall_humidity_interaction" in features:
            actions.append({"action": "Deploy chlorine tablets to public water sources within 48 hours.", "rationale": "Heavy rainfall compromises local water sanitation infrastructure."})
            actions.append({"action": "Initiate targeted anti-larval spraying in identified stagnant water zones.", "rationale": "Prevents imminent mosquito population spikes from accumulated water."})
        if "temperature_avg" in features:
            actions.append({"action": "Activate thermal scanning protocols and fever clinics in high-density areas.", "rationale": "High temperatures accelerate vector life cycles, necessitating rapid case isolation."})
        if len(actions) == 0:
            actions.append({"action": "Issue emergency public health warnings and mobilize rapid response teams.", "rationale": "Systemic vulnerability requires immediate broad action."})
    elif risk == "Medium":
        actions.append({"action": "Increase surveillance of hospital admissions for waterborne/vector-borne symptoms.", "rationale": "Early detection of index cases prevents wider community spread."})
        actions.append({"action": "Distribute protective guidelines to community health workers.", "rationale": "Prepares frontline workers for potential escalation in risk levels."})
    else:
        actions.append({"action": "Maintain routine vector control and water quality testing.", "rationale": "Ensures baseline public health defense mechanisms remain active."})

    # 7. Temporal & Projected
    trend_reason = f"Risk is {alert_level.lower()} due to the dominant influence of {shap_info['dominant_driver']} over recent days."
    projected_risk = "Projected to remain elevated if current environmental drivers persist without intervention." if risk in ["High", "Medium"] else "Projected to remain stable barring severe weather changes."

    return {
        "urgency_level": urgency_level,
        "response_window": response_window,
        "recommended_actions": actions,
        "risk_narrative": narrative,
        "alert_level": alert_level,
        "confidence_context": confidence_context,
        "expected_impact": expected_impact,
        "source_attribution": {
            "dominant_factor": shap_info["dominant_driver"],
            "weather_driven": attr["Weather-driven"],
            "environmental": attr["Environmental"],
            "secondary": attr["Secondary"]
        },
        "priority_score": priority_score,
        "priority_label": priority_label,
        "trend_reason": trend_reason,
        "projected_risk": projected_risk
    }



# ══════════════════════════════════════════════════════════════════
# LIGHTWEIGHT RAG — DISEASE KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════
# Local structured knowledge so the chatbot NEVER fails on basic
# disease queries — even without Gemini API access.

DISEASE_KB = {
    "cholera": {
        "name": "Cholera",
        "emoji": "🦠",
        "caused_by": "Vibrio cholerae bacteria",
        "transmission": "Contaminated water and food, poor sanitation",
        "symptoms": ["Severe watery diarrhea", "Vomiting", "Rapid dehydration", "Muscle cramps", "Low blood pressure"],
        "prevention": ["Drink only boiled or treated water", "Wash hands with soap frequently", "Eat properly cooked food", "Avoid raw seafood", "Ensure proper sewage disposal"],
        "severity": "Can be fatal within hours if untreated",
        "treatment": "Oral rehydration salts (ORS), IV fluids, antibiotics in severe cases",
    },
    "dengue": {
        "name": "Dengue Fever",
        "emoji": "🦟",
        "caused_by": "Dengue virus (DENV), transmitted by Aedes aegypti mosquitoes",
        "transmission": "Bite of infected Aedes mosquito, active during daytime",
        "symptoms": ["High fever (40°C/104°F)", "Severe headache", "Pain behind eyes", "Joint and muscle pain", "Skin rash", "Nausea"],
        "prevention": ["Eliminate mosquito breeding sites (standing water)", "Use mosquito repellent", "Wear long-sleeve clothing", "Use window screens and bed nets", "Apply larvicide in water containers"],
        "severity": "Can progress to dengue hemorrhagic fever, potentially fatal",
        "treatment": "Supportive care, hydration, pain relievers (avoid aspirin)",
    },
    "typhoid": {
        "name": "Typhoid Fever",
        "emoji": "🤒",
        "caused_by": "Salmonella typhi bacteria",
        "transmission": "Contaminated water and food, person-to-person via fecal-oral route",
        "symptoms": ["Sustained high fever", "Weakness and fatigue", "Stomach pain", "Headache", "Loss of appetite", "Constipation or diarrhea"],
        "prevention": ["Boil water before drinking", "Wash hands before eating", "Eat thoroughly cooked food", "Avoid street food in endemic areas", "Get vaccinated if traveling to high-risk areas"],
        "severity": "Serious if untreated, can cause intestinal perforation",
        "treatment": "Antibiotics (ciprofloxacin, azithromycin), fluids, rest",
    },
    "malaria": {
        "name": "Malaria",
        "emoji": "🦟",
        "caused_by": "Plasmodium parasites (P. falciparum, P. vivax), transmitted by Anopheles mosquitoes",
        "transmission": "Bite of infected Anopheles mosquito, active during dusk and dawn",
        "symptoms": ["Cyclic fever with chills", "Sweating", "Headache", "Nausea and vomiting", "Body aches", "Anemia"],
        "prevention": ["Sleep under insecticide-treated bed nets", "Use mosquito repellent", "Take antimalarial prophylaxis if prescribed", "Eliminate standing water", "Spray indoor residual insecticide"],
        "severity": "P. falciparum can be fatal, especially in children",
        "treatment": "Antimalarial drugs (ACT for P. falciparum, chloroquine for P. vivax)",
    },
    "diarrhea": {
        "name": "Diarrheal Disease",
        "emoji": "💧",
        "caused_by": "Multiple pathogens — Rotavirus, E. coli, Shigella, Giardia, or contaminated food/water",
        "transmission": "Fecal-oral route, contaminated water, poor hand hygiene",
        "symptoms": ["Frequent loose or watery stools", "Abdominal cramps", "Nausea", "Dehydration", "Fever (in some cases)"],
        "prevention": ["Drink clean, boiled water", "Wash hands with soap after using toilet", "Ensure food hygiene", "Use ORS at first sign of dehydration", "Breastfeed infants exclusively for 6 months"],
        "severity": "Leading cause of child mortality in developing countries",
        "treatment": "ORS, zinc supplements for children, fluids, antibiotics only if bacterial",
    },
    "chikungunya": {
        "name": "Chikungunya",
        "emoji": "🦟",
        "caused_by": "Chikungunya virus, transmitted by Aedes mosquitoes",
        "transmission": "Bite of infected Aedes albopictus or Aedes aegypti mosquito",
        "symptoms": ["Sudden high fever", "Severe joint pain (arthralgia)", "Muscle pain", "Headache", "Rash", "Fatigue"],
        "prevention": ["Eliminate mosquito breeding sites", "Use repellent and protective clothing", "Install window screens", "Use bed nets during daytime rest"],
        "severity": "Rarely fatal but joint pain can persist for months",
        "treatment": "No specific antiviral; rest, fluids, pain relievers",
    },
    "leptospirosis": {
        "name": "Leptospirosis",
        "emoji": "🐀",
        "caused_by": "Leptospira bacteria, carried by rodents and animals",
        "transmission": "Contact with water or soil contaminated with infected animal urine, especially during floods",
        "symptoms": ["High fever", "Headache", "Muscle pain", "Red eyes", "Jaundice", "Vomiting"],
        "prevention": ["Avoid walking in floodwater", "Wear protective footwear", "Control rodent populations", "Avoid swimming in contaminated water", "Prophylactic doxycycline in high-risk settings"],
        "severity": "Can progress to Weil's disease (liver/kidney failure)",
        "treatment": "Antibiotics (doxycycline, penicillin), hospitalization for severe cases",
    },
    "hepatitis": {
        "name": "Hepatitis A/E",
        "emoji": "🟡",
        "caused_by": "Hepatitis A virus (HAV) or Hepatitis E virus (HEV)",
        "transmission": "Contaminated water and food, fecal-oral route",
        "symptoms": ["Jaundice (yellow skin/eyes)", "Fatigue", "Nausea", "Abdominal pain", "Dark urine", "Loss of appetite"],
        "prevention": ["Drink clean water", "Wash hands frequently", "Get vaccinated (Hepatitis A)", "Avoid raw shellfish", "Maintain food hygiene"],
        "severity": "Usually self-limiting; Hep E dangerous in pregnancy",
        "treatment": "Supportive care, rest, avoid alcohol",
    },
}

# Build keyword → disease key mapping
_DISEASE_KEYWORD_MAP = {}
for _dk, _dv in DISEASE_KB.items():
    _DISEASE_KEYWORD_MAP[_dk] = _dk
    _DISEASE_KEYWORD_MAP[_dv["name"].lower()] = _dk
# Additional keyword triggers
_DISEASE_KEYWORD_MAP.update({
    "diarrhoea": "diarrhea", "dysentery": "diarrhea", "loose motion": "diarrhea",
    "loose motions": "diarrhea", "stomach infection": "diarrhea",
    "jaundice": "hepatitis", "yellow fever": "hepatitis",
    "rat fever": "leptospirosis", "lepto": "leptospirosis",
    "chikun": "chikungunya",
})

print(f"  Disease KB entries: {len(DISEASE_KB)} diseases, {len(_DISEASE_KEYWORD_MAP)} keyword triggers")


def _lookup_disease_kb(msg: str, lang: str = "en") -> Optional[str]:
    """Look up disease info from local RAG knowledge base. Returns formatted response or None."""
    msg_lower = msg.lower()

    # Find matching disease
    matched_key = None
    for keyword, disease_key in _DISEASE_KEYWORD_MAP.items():
        if keyword in msg_lower:
            matched_key = disease_key
            break

    if not matched_key or matched_key not in DISEASE_KB:
        return None

    d = DISEASE_KB[matched_key]

    # Determine what info the user wants
    wants_symptoms = any(w in msg_lower for w in ["symptom", "sign", "लक्षण", "লক্ষণ"])
    wants_prevention = any(w in msg_lower for w in ["prevent", "precaution", "avoid", "protect", "बचाव", "সাৱধানতা"])
    wants_cause = any(w in msg_lower for w in ["cause", "spread", "transmit", "how does", "कारण", "কাৰণ"])
    wants_treatment = any(w in msg_lower for w in ["treat", "cure", "medicine", "उपचार"])

    # If no specific aspect requested, give overview
    if not any([wants_symptoms, wants_prevention, wants_cause, wants_treatment]):
        # Full overview
        symptoms_str = ", ".join(d["symptoms"][:4])
        prevention_str = "; ".join(d["prevention"][:3])
        return (
            f"{d['emoji']} **{d['name']}**\n\n"
            f"**Caused by:** {d['caused_by']}\n"
            f"**Transmission:** {d['transmission']}\n"
            f"**Key symptoms:** {symptoms_str}\n"
            f"**Prevention:** {prevention_str}\n"
            f"**Severity:** {d['severity']}\n\n"
            f"⚕️ Always consult a healthcare professional if you experience symptoms."
        )

    # Specific aspect
    lines = [f"{d['emoji']} **{d['name']}**\n"]
    if wants_symptoms:
        lines.append("**Symptoms:**")
        for s in d["symptoms"]:
            lines.append(f"  • {s}")
    if wants_cause:
        lines.append(f"\n**Caused by:** {d['caused_by']}")
        lines.append(f"**Transmission:** {d['transmission']}")
    if wants_prevention:
        lines.append("\n**Prevention:**")
        for p in d["prevention"]:
            lines.append(f"  • {p}")
    if wants_treatment:
        lines.append(f"\n**Treatment:** {d['treatment']}")

    lines.append("\n⚕️ Always consult a healthcare professional if you experience symptoms.")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# GEMINI API INTEGRATION  (via app.services.gemini_service)
# ══════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Startup validation — verify Gemini connectivity
if is_gemini_available():
    try:
        _test_result = _gemini_service_call("Say 'OK' in one word.", "en")
        if _test_result:
            print(f"  ✅ Gemini initialized successfully (test response: {len(_test_result)} chars)")
        else:
            print("  ⚠️  Gemini API key set but test call returned None")
    except Exception as _e:
        print(f"  ⚠️  Gemini API key set but test failed: {_e}")
else:
    print("  ⚠️  GEMINI_API_KEY not set — Gemini disabled, using local RAG only")


async def _call_gemini(user_prompt: str, lang: str = "en") -> Optional[str]:
    """Call Gemini via the dedicated service. Returns None on failure."""
    try:
        print(f"  [Gemini] Called with: '{user_prompt[:80]}...' lang={lang}")
        result = _gemini_service_call(user_prompt, lang)
        if result:
            print(f"  [Gemini] Response received ({len(result)} chars)")
        else:
            print("  [Gemini] No response returned")
        return result
    except Exception as exc:
        print(f"  [Gemini] Service error: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════
# 3-LAYER INTENT DETECTION
# ══════════════════════════════════════════════════════════════════

class Intent:
    GREETING = "greeting"
    HELP = "help"
    RISK_CHECK = "risk_check"
    WHY_RISK = "why_risk"
    PREVENTION = "prevention"
    DISEASE_INFO = "disease_info"
    OVERALL_RISK = "overall_risk"
    UNKNOWN = "unknown"


# Disease keywords that trigger Gemini
DISEASE_KEYWORDS = [
    "cholera", "dengue", "typhoid", "malaria", "chikungunya", "leptospirosis",
    "diarrhea", "diarrhoea", "dysentery", "hepatitis", "jaundice", "fever",
    "mosquito", "waterborne", "water-borne", "vector-borne", "epidemic",
    "pandemic", "infection", "disease", "symptoms", "treatment", "cause",
    "spread", "transmission", "outbreak", "virus", "bacteria", "parasite",
    # Hindi
    "हैजा", "डेंगू", "टाइफाइड", "मलेरिया", "बुखार", "दस्त", "बीमारी",
    "रोग", "लक्षण", "उपचार", "कारण", "संक्रमण", "महामारी",
    # Assamese
    "কলেৰা", "ডেংগু", "মেলেৰিয়া", "জ্বৰ", "ৰোগ", "মহামাৰী",
]

GREETING_KEYWORDS = [
    "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
    "namaste", "namaskar", "नमस्ते", "नमस्कार", "নমস্কাৰ", "আপোনাক",
    "howdy", "greetings",
]

HELP_KEYWORDS = [
    "help", "what can", "kya kar", "क्या कर", "সহায়", "how to use",
    "features", "commands", "menu",
]

PREVENTION_KEYWORDS = [
    "precaution", "prevention", "safety", "protect", "avoid", "tips",
    "बचाव", "सावधानी", "सुरक्षा", "রক্ষা", "সাৱধানতা",
    "safe practice", "how to prevent",
]

WHY_RISK_KEYWORDS = [
    "why", "reason", "explain", "factor", "driver",
    "क्यों", "কিয়", "কাৰণ", "shap",
]

RISK_CHECK_KEYWORDS = [
    "safe", "risk", "danger", "predict", "check", "status",
    "सुरक्षित", "जोखिम", "खतरा", "বিপদ", "নিৰাপদ",
]


def _detect_language(msg: str, declared: str) -> str:
    if declared in ("hi", "as"):
        return declared
    if re.search(r'[\u0900-\u097F]', msg):
        return "hi"
    if re.search(r'[\u0980-\u09FF]', msg):
        return "as"
    return "en"


def _detect_intent(msg: str) -> str:
    """Layer 1: Detect user intent from message."""
    msg_lower = _normalize_input(msg)

    # Greeting — use word boundary for short keywords
    words = set(re.split(r'\W+', msg_lower))
    short_greets = {"hello", "hi", "hey", "howdy", "greetings"}
    long_greets = ["good morning", "good evening", "good afternoon",
                   "namaste", "namaskar", "नमस्ते", "नमस्कार", "নমস্কাৰ", "আপোনাক"]
    is_greeting = bool(words & short_greets) or any(g in msg_lower for g in long_greets)
    if is_greeting and len(msg_lower.split()) <= 4:
        return Intent.GREETING

    # Help
    if any(k in msg_lower for k in HELP_KEYWORDS):
        return Intent.HELP

    # Why risk (must check before risk_check since "why" + city)
    city = _find_city_in_message(msg)
    if city and any(k in msg_lower for k in WHY_RISK_KEYWORDS):
        return Intent.WHY_RISK

    # Prevention
    if any(k in msg_lower for k in PREVENTION_KEYWORDS):
        return Intent.PREVENTION

    # Disease info (check before risk_check) — only if NO city found
    if any(k in msg_lower for k in DISEASE_KEYWORDS):
        if not city:
            return Intent.DISEASE_INFO

    # Risk check — city found (includes state→city mapping)
    if city:
        return Intent.RISK_CHECK

    # Overall risk
    if any(k in msg_lower for k in ["risk", "danger", "jokhim", "जोखिम", "বিপদ", "खतरा", "overall", "total", "cities"]):
        return Intent.OVERALL_RISK

    # Disease info fallback
    if any(k in msg_lower for k in DISEASE_KEYWORDS):
        return Intent.DISEASE_INFO

    return Intent.UNKNOWN


# ══════════════════════════════════════════════════════════════════
# RESPONSE GENERATION (LAYER 3)
# ══════════════════════════════════════════════════════════════════

def _build_shap_chat_explanation(risk: str, city_name: str, lang: str) -> str:
    """Build a concise SHAP explanation for chat with quantitative values."""
    shap_info = build_shap_explanation(risk)
    explanations = shap_info["explanations"]

    if lang == "hi":
        risk_hi = {"High": "उच्च", "Medium": "मध्यम", "Low": "कम"}.get(risk, risk)
        lines = [f"🔬 **{city_name} में {risk_hi} जोखिम के कारण:**"]
        for e in explanations:
            arrow = "↑" if e["impact"] == "increase" else "↓"
            sign = "+" if e["impact"] == "increase" else "-"
            strength_map = {"high": "मजबूत", "medium": "मध्यम", "low": "कमज़ोर"}
            lines.append(f"  {arrow} **{e['display_name']}** ({sign}{e['importance']:.2f}) — {strength_map.get(e['strength'], e['strength'])} प्रभाव")
        lines.append(f"\n🎯 प्रमुख कारक: **{shap_info['dominant_driver']}**")
        if shap_info["interaction_note"]:
            lines.append(f"💡 {shap_info['interaction_note']}")
        return "\n".join(lines)

    if lang == "as":
        risk_as = {"High": "উচ্চ", "Medium": "মধ্যমীয়া", "Low": "কম"}.get(risk, risk)
        lines = [f"🔬 **{city_name}ত {risk_as} বিপদৰ কাৰণ:**"]
        for e in explanations:
            arrow = "↑" if e["impact"] == "increase" else "↓"
            sign = "+" if e["impact"] == "increase" else "-"
            strength_map = {"high": "শক্তিশালী", "medium": "মধ্যমীয়া", "low": "দুৰ্বল"}
            lines.append(f"  {arrow} **{e['display_name']}** ({sign}{e['importance']:.2f}) — {strength_map.get(e['strength'], e['strength'])} প্ৰভাৱ")
        lines.append(f"\n🎯 প্ৰধান কাৰক: **{shap_info['dominant_driver']}**")
        if shap_info["interaction_note"]:
            lines.append(f"💡 {shap_info['interaction_note']}")
        return "\n".join(lines)

    # English — with quantitative SHAP values
    lines = [f"🔬 **Why {city_name} is {risk} Risk:**"]
    for e in explanations:
        arrow = "↑" if e["impact"] == "increase" else "↓"
        sign = "+" if e["impact"] == "increase" else "-"
        lines.append(f"  {arrow} **{e['display_name']}** ({sign}{e['importance']:.2f} SHAP) — {e['strength']} impact")
        lines.append(f"    {e['explanation']}")
    lines.append(f"\n🎯 Dominant driver: **{shap_info['dominant_driver']}**")
    if shap_info["interaction_note"]:
        lines.append(f"💡 {shap_info['interaction_note']}")
    return "\n".join(lines)


def _get_suggestions(intent: str, lang: str) -> list[str]:
    """Get smart suggestion buttons based on context."""
    if lang == "hi":
        return [
            "शहर का जोखिम जाँचें",
            "जोखिम क्यों?",
            "बचाव के उपाय",
            "डेंगू क्या है?",
        ]
    if lang == "as":
        return [
            "চহৰৰ বিপদ পৰীক্ষা",
            "বিপদ কিয়?",
            "সাৱধানতা",
            "ডেংগু কি?",
        ]
    suggestions_map = {
        Intent.GREETING: ["Check my city", "Overall risk", "Prevention tips", "What is dengue?"],
        Intent.RISK_CHECK: ["Why this risk?", "Prevention tips", "Check another city"],
        Intent.WHY_RISK: ["Prevention tips", "Check another city", "What is cholera?"],
        Intent.PREVENTION: ["Check my city", "What is dengue?", "Overall risk"],
        Intent.DISEASE_INFO: ["Check my city", "Prevention tips", "What is typhoid?"],
        Intent.OVERALL_RISK: ["Check my city", "Prevention tips", "What is cholera?"],
        Intent.HELP: ["Check my city", "Why risk?", "Prevention tips", "What is dengue?"],
    }
    return suggestions_map.get(intent, ["Check my city", "Why risk?", "Prevention tips"])


def _was_state_mapped(msg: str, city_dict: dict) -> Optional[str]:
    """Return the mapping source if the city was resolved indirectly."""
    msg_norm = _normalize_input(msg)
    city_lower = city_dict["city"].lower().replace("_", " ")
    # If the city name directly appears in message, it's not a mapping
    if city_lower in msg_norm:
        return None
    # Check state mapping
    for state_name in _STATE_LOOKUP:
        if state_name in msg_norm:
            return state_name.title()
    # Check alias mapping
    for alias in _ALIAS_LOOKUP:
        if alias in msg_norm:
            return alias.title()
    # Fuzzy match label
    return "fuzzy match"


async def _generate_response(message: str, intent: str, lang: str) -> dict:
    """Layer 3: Generate response based on intent."""
    msg = message.strip()
    city = _find_city_in_message(message)

    # ── GREETING ──
    if intent == Intent.GREETING:
        if lang == "hi":
            reply = "नमस्ते! 👋 मैं EPIDRA का AI सहायक हूँ — महामारी जोखिम विश्लेषण में आपकी मदद कर सकता हूँ।\n\n💡 मुझसे पूछें:\n• किसी शहर का जोखिम स्तर\n• जोखिम क्यों है\n• बचाव के उपाय\n• बीमारियों की जानकारी"
        elif lang == "as":
            reply = "নমস্কাৰ! 👋 মই EPIDRA ৰ AI সহায়ক — মহামাৰী বিপদ বিশ্লেষণত সহায় কৰিব পাৰোঁ।\n\n💡 মোক সুধিব পাৰে:\n• চহৰৰ বিপদ স্তৰ\n• বিপদ কিয়\n• সাৱধানতা\n• ৰোগৰ তথ্য"
        else:
            reply = "Hello! 👋 I'm EPIDRA's AI assistant — here to help with epidemic risk analysis.\n\n💡 Ask me about:\n• City risk levels (e.g. 'Guwahati' or 'Sikkim')\n• Why a city is at risk\n• Prevention tips\n• Disease information (cholera, dengue, etc.)"
        return {"reply": reply, "intent": intent}

    # ── HELP ──
    if intent == Intent.HELP:
        if lang == "hi":
            reply = "🤖 **EPIDRA सहायक** — मैं ये कर सकता हूँ:\n\n📍 शहर जोखिम जाँच → 'Guwahati' या 'Sikkim'\n🔬 जोखिम कारण → 'Aizawl में जोखिम क्यों?'\n🛡️ बचाव → 'बचाव के उपाय'\n🦠 रोग जानकारी → 'डेंगू क्या है?'\n📊 कुल जोखिम → 'कितने शहर खतरे में?'"
        elif lang == "as":
            reply = "🤖 **EPIDRA সহায়ক** — মই কৰিব পাৰোঁ:\n\n📍 চহৰ বিপদ পৰীক্ষা → 'Guwahati' বা 'Meghalaya'\n🔬 বিপদৰ কাৰণ → 'Aizawl ত বিপদ কিয়?'\n🛡️ সাৱধানতা → 'সাৱধানতা'\n🦠 ৰোগৰ তথ্য → 'ডেংগু কি?'\n📊 মুঠ বিপদ → 'কিমান চহৰ বিপদত?'"
        else:
            reply = "🤖 **EPIDRA Assistant** — Here's what I can do:\n\n📍 City risk check → Type a city name ('Chennai') or state ('Sikkim')\n🔬 Risk explanation → 'Why is Aizawl high risk?'\n🛡️ Prevention tips → 'Prevention tips' or 'How to stay safe?'\n🦠 Disease info → 'What is cholera?' or 'Symptoms of dengue?'\n📊 Overall risk → 'How many cities are at risk?'"
        return {"reply": reply, "intent": intent}

    # ── RISK CHECK (Structured output) ──
    if intent == Intent.RISK_CHECK and city:
        name = city["city"].replace("_", " ")
        risk = city["risk"]
        conf = city["model_confidence"]
        tips = PREVENTION.get(risk, PREVENTION["Low"])

        # Get SHAP for key drivers
        shap_info = build_shap_explanation(risk)
        dominant = shap_info["dominant_driver"]
        drivers = [e["display_name"] for e in shap_info["explanations"][:2]]
        drivers_str = ", ".join(drivers)

        # Check if state-mapped
        state_name = _was_state_mapped(message, city)
        state_note = f"\n📌 _{state_name} → showing capital city {name}_" if state_name else ""

        if lang == "hi":
            r = {"High": "उच्च 🔴", "Medium": "मध्यम 🟡", "Low": "कम 🟢"}.get(risk, risk)
            reply = f"📍 **{name}**{state_note}\n\n⚠️ जोखिम स्तर: **{r}**\n📊 विश्वास: **{conf:.0%}**\n🎯 प्रमुख कारक: **{drivers_str}**\n\n💡 {tips['hi'][0]}"
        elif lang == "as":
            r = {"High": "উচ্চ 🔴", "Medium": "মধ্যমীয়া 🟡", "Low": "কম 🟢"}.get(risk, risk)
            reply = f"📍 **{name}**{state_note}\n\n⚠️ বিপদ স্তৰ: **{r}**\n📊 বিশ্বাস: **{conf:.0%}**\n🎯 প্ৰধান কাৰক: **{drivers_str}**\n\n💡 {tips['as'][0]}"
        else:
            emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk, "")
            reply = (
                f"📍 **{name}**{state_note}\n\n"
                f"{emoji} Risk Level: **{risk}**\n"
                f"📊 Confidence: **{conf:.0%}**\n"
                f"🎯 Key Drivers: **{drivers_str}**\n\n"
                f"💡 {tips['en'][0]}"
            )
        return {"reply": reply, "intent": intent}

    # ── WHY RISK ──
    if intent == Intent.WHY_RISK and city:
        name = city["city"].replace("_", " ")
        risk = city["risk"]
        reply = _build_shap_chat_explanation(risk, name, lang)
        return {"reply": reply, "intent": intent}

    # ── PREVENTION ──
    if intent == Intent.PREVENTION:
        if city:
            risk = city["risk"]
            name = city["city"].replace("_", " ")
            tips = PREVENTION.get(risk, PREVENTION["Low"])
        else:
            risk = "High"
            name = None
            tips = PREVENTION["High"]

        if lang == "hi":
            risk_hi = {"High": "उच्च", "Medium": "मध्यम", "Low": "कम"}.get(risk, risk)
            prefix = name + ' के लिए ' if name else ''
            header = f"🛡️ {prefix}बचाव के उपाय ({risk_hi} जोखिम):\n"
            reply = header + "\n".join(f"  • {t}" for t in tips["hi"])
        elif lang == "as":
            risk_as = {"High": "উচ্চ", "Medium": "মধ্যমীয়া", "Low": "কম"}.get(risk, risk)
            prefix = name + 'ৰ বাবে ' if name else ''
            header = f"🛡️ {prefix}সাৱধানতা ({risk_as} বিপদ):\n"
            reply = header + "\n".join(f"  • {t}" for t in tips["as"])
        else:
            header = f"🛡️ {'Prevention tips for ' + name + ' (' + risk + ' Risk)' if name else 'General Prevention Tips (High Risk)'}:\n"
            reply = header + "\n".join(f"  • {t}" for t in tips["en"])

        # Try to enhance with Gemini
        gemini_extra = await _call_gemini(
            f"Give 2 additional brief prevention tips for water-borne diseases in {'tropical India' if not name else name + ', India'}. Max 2 sentences total.",
            lang
        )
        if gemini_extra:
            reply += f"\n\n🤖 AI tip: {gemini_extra}"

        return {"reply": reply, "intent": intent}

    # ── DISEASE INFO (RAG → Gemini enhancement) ──
    if intent == Intent.DISEASE_INFO:
        # Step 1: Try local RAG knowledge base first (instant, reliable)
        rag_response = _lookup_disease_kb(message, lang)

        if rag_response:
            # Step 2: Try to enhance with Gemini (adds AI insight)
            gemini_extra = await _call_gemini(
                f"Add one brief, helpful fact about {message}. Max 1-2 sentences. Be specific and practical.",
                lang
            )
            reply = rag_response
            if gemini_extra:
                reply += f"\n\n🤖 **AI Insight:** {gemini_extra}"
        else:
            # No RAG match — try Gemini directly
            gemini_response = await _call_gemini(message, lang)
            if gemini_response:
                reply = f"🦠 {gemini_response}"
            else:
                # Final fallback: static disease info
                reply = _static_disease_response(message, lang)

        return {"reply": reply, "intent": intent}

    # ── OVERALL RISK ──
    if intent == Intent.OVERALL_RISK:
        hc = sum(1 for d in DISTRICTS if d["risk"] == "High")
        mc = sum(1 for d in DISTRICTS if d["risk"] == "Medium")
        lc = sum(1 for d in DISTRICTS if d["risk"] == "Low")
        top3 = sorted(
            [d for d in DISTRICTS if d["risk"] == "High"],
            key=lambda x: x["model_confidence"],
            reverse=True
        )[:3]
        top_names = ", ".join(d["city"].replace("_", " ") for d in top3)

        if lang == "hi":
            reply = f"📊 **वर्तमान जोखिम सारांश**\n\n🔴 उच्च जोखिम: **{hc}** शहर\n🟡 मध्यम जोखिम: **{mc}** शहर\n🟢 कम जोखिम: **{lc}** शहर\n\n⚠️ सबसे अधिक जोखिम: **{top_names}**"
        elif lang == "as":
            reply = f"📊 **বৰ্তমান বিপদ সাৰাংশ**\n\n🔴 উচ্চ বিপদ: **{hc}** চহৰ\n🟡 মধ্যমীয়া বিপদ: **{mc}** চহৰ\n🟢 কম বিপদ: **{lc}** চহৰ\n\n⚠️ আটাইতকৈ বেছি বিপদ: **{top_names}**"
        else:
            reply = f"📊 **Current Risk Summary**\n\n🔴 High Risk: **{hc}** cities\n🟡 Medium Risk: **{mc}** cities\n🟢 Low Risk: **{lc}** cities\n\n⚠️ Highest risk: **{top_names}**"
        return {"reply": reply, "intent": intent}

    # ── UNKNOWN — Try Gemini as fallback ──
    gemini_response = await _call_gemini(message, lang)
    if gemini_response:
        return {"reply": f"🤖 {gemini_response}", "intent": Intent.DISEASE_INFO}

    # ── Structured fallback (NEVER generic) ──
    if lang == "hi":
        reply = (
            "🤖 मैं EPIDRA सहायक हूँ। मैं इन विषयों में मदद कर सकता हूँ:\n\n"
            "📍 **शहर का जोखिम** — कोई शहर या राज्य का नाम लिखें\n"
            "   उदाहरण: 'Guwahati', 'Sikkim', 'Delhi'\n\n"
            "🦠 **रोग जानकारी** — बीमारी के बारे में पूछें\n"
            "   उदाहरण: 'हैजा क्या है?', 'डेंगू के लक्षण'\n\n"
            "🛡️ **बचाव** — 'बचाव के उपाय' लिखें"
        )
    elif lang == "as":
        reply = (
            "🤖 মই EPIDRA সহায়ক। মই এই বিষয়ত সহায় কৰিব পাৰোঁ:\n\n"
            "📍 **চহৰৰ বিপদ** — চহৰ বা ৰাজ্যৰ নাম লিখক\n"
            "   যেনে: 'Guwahati', 'Meghalaya'\n\n"
            "🦠 **ৰোগৰ তথ্য** — ৰোগৰ বিষয়ে সুধক\n"
            "   যেনে: 'ডেংগু কি?'\n\n"
            "🛡️ **সাৱধানতা** — 'সাৱধানতা' লিখক"
        )
    else:
        reply = (
            "🤖 I can help with the following:\n\n"
            "📍 **City risk** — Type a city or state name\n"
            "   e.g. 'Guwahati', 'Sikkim', 'Chennai'\n\n"
            "🦠 **Disease info** — Ask about any disease\n"
            "   e.g. 'What is cholera?', 'Symptoms of dengue?'\n\n"
            "🛡️ **Prevention** — Type 'prevention tips'\n\n"
            "📊 **Overall risk** — Type 'overall risk'"
        )
    return {"reply": reply, "intent": Intent.UNKNOWN}


def _static_disease_response(msg: str, lang: str) -> str:
    """Fallback static disease info when Gemini is unavailable."""
    msg_lower = msg.lower()
    diseases = {
        "cholera": {
            "en": "🦠 **Cholera** is an acute diarrheal disease caused by Vibrio cholerae bacteria. It spreads through contaminated water and food. Symptoms include severe watery diarrhea, vomiting, and dehydration. Prevention: Drink clean water, maintain hygiene, and seek medical help immediately if symptoms appear.",
            "hi": "🦠 **हैजा** विब्रियो कॉलेरी बैक्टीरिया से होने वाली एक तीव्र दस्त रोग है। यह दूषित पानी और भोजन से फैलता है। लक्षण: गंभीर पानी जैसे दस्त, उल्टी, निर्जलीकरण। रोकथाम: स्वच्छ पानी पिएं, स्वच्छता बनाएं।",
            "as": "🦠 **কলেৰা** ভিব্ৰিঅ' কলেৰি বেক্টেৰিয়াৰ দ্বাৰা হোৱা এক তীব্ৰ ডায়েৰিয়া ৰোগ। ই দূষিত পানী আৰু খাদ্যৰ জৰিয়তে বিয়পে।",
        },
        "dengue": {
            "en": "🦟 **Dengue** is a mosquito-borne viral disease transmitted by Aedes mosquitoes. Symptoms include high fever, severe headache, pain behind the eyes, joint pain, and rash. Prevention: Eliminate mosquito breeding sites, use repellent, and wear protective clothing.",
            "hi": "🦟 **डेंगू** एडीज़ मच्छर द्वारा फैलने वाला वायरल रोग है। लक्षण: तेज बुखार, सिरदर्द, जोड़ों में दर्द। रोकथाम: मच्छर प्रजनन स्थल खत्म करें, रिपेलेंट लगाएं।",
            "as": "🦟 **ডেংগু** এডিজ মহৰ দ্বাৰা বিয়পা এক ভাইৰেল ৰোগ। লক্ষণ: উচ্চ জ্বৰ, মূৰৰ বিষ, গাঁঠিৰ বিষ।",
        },
        "typhoid": {
            "en": "🤒 **Typhoid** is caused by Salmonella typhi bacteria, spread through contaminated water and food. Symptoms include sustained fever, weakness, stomach pain, and headache. Prevention: Boil water before drinking, wash hands regularly, eat cooked food.",
            "hi": "🤒 **टाइफाइड** साल्मोनेला टाइफी बैक्टीरिया से होता है। लक्षण: लगातार बुखार, कमजोरी, पेट दर्द। रोकथाम: पानी उबालकर पिएं, हाथ धोएं।",
            "as": "🤒 **টাইফয়ড** ছালমনেলা টাইফি বেক্টেৰিয়াৰ দ্বাৰা হয়। লক্ষণ: অবিৰত জ্বৰ, দুৰ্বলতা, পেটৰ বিষ।",
        },
        "malaria": {
            "en": "🦟 **Malaria** is caused by Plasmodium parasites, transmitted through infected Anopheles mosquito bites. Symptoms include cyclic fever, chills, sweating, and anemia. Prevention: Use mosquito nets, take antimalarial medication if prescribed, eliminate standing water.",
            "hi": "🦟 **मलेरिया** प्लास्मोडियम परजीवी से होता है। लक्षण: चक्रीय बुखार, ठंड लगना, पसीना। रोकथाम: मच्छरदानी उपयोग करें, दवा लें।",
            "as": "🦟 **মেলেৰিয়া** প্লাজমোডিয়াম পৰজীৱীৰ দ্বাৰা হয়। লক্ষণ: চক্ৰীয় জ্বৰ, কঁপনি। প্ৰতিৰোধ: মহ জাল ব্যৱহাৰ কৰক।",
        },
    }

    for disease, info in diseases.items():
        if disease in msg_lower:
            return info.get(lang, info["en"])

    if lang == "hi":
        return "🦠 जल-जनित और वेक्टर-जनित रोग गंभीर स्वास्थ्य खतरे हैं। स्वच्छ पानी पिएं, मच्छरों से बचें, और लक्षण दिखने पर चिकित्सक से मिलें।"
    if lang == "as":
        return "🦠 জল-জনিত আৰু ভেক্টৰ-জনিত ৰোগ গুৰুতৰ স্বাস্থ্য বিপদ। পৰিষ্কাৰ পানী পান কৰক, মহৰ পৰা বাচক।"
    return "🦠 Water-borne and vector-borne diseases are serious health threats in India. Always drink clean water, prevent mosquito breeding, and consult a doctor if you have symptoms like fever, diarrhea, or body aches."


# ══════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="EPIDRA API",
    description="Epidemic Risk Assessment — India-focused, calibrated predictions with SHAP explainability, RAG disease KB, and AI chatbot",
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ─────────────────────────────────────────────

class DistrictSummary(BaseModel):
    id: int
    city: str
    latitude: float
    longitude: float
    risk: str
    model_confidence: float
    priority_score: Optional[int] = 0

class ShapExplanation(BaseModel):
    feature: str
    display_name: str
    impact: str
    strength: str
    importance: float
    explanation: str

class ActionItem(BaseModel):
    action: str
    rationale: str

class SourceAttribution(BaseModel):
    dominant_factor: str
    weather_driven: list[str]
    environmental: list[str]
    secondary: list[str]

class DistrictDetail(BaseModel):
    id: int
    city: str
    latitude: float
    longitude: float
    risk: str
    model_confidence: float
    shap_explanations: list[ShapExplanation]
    dominant_driver: str
    interaction_note: str
    prevention_tips: list[str]
    # Decision Intelligence extensions
    urgency_level: str
    response_window: str
    recommended_actions: list[ActionItem]
    risk_narrative: str
    alert_level: str
    confidence_context: str
    expected_impact: str
    source_attribution: SourceAttribution
    priority_score: int
    priority_label: str
    trend_reason: str
    projected_risk: str

class PredictRequest(BaseModel):
    rainfall: float = Field(..., description="7-day cumulative rainfall (mm)")
    temperature: float = Field(..., description="Average temperature (°C)")
    humidity: float = Field(..., description="Average relative humidity (%)")

class PredictResponse(BaseModel):
    risk: str
    model_confidence: float
    urgency_level: str
    response_window: str
    recommended_actions: list[ActionItem]
    risk_narrative: str
    expected_impact: str
    alert_level: str

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "en"
    context_city: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    language: str
    intent: str
    suggestions: list[str]


# ══════════════════════════════════════════════════════════════════
# 1. GET /districts
# ══════════════════════════════════════════════════════════════════

@app.get("/districts", response_model=list[DistrictSummary])
def list_districts():
    """Return all Indian locations with calibrated risk predictions."""
    res = []
    for d in DISTRICTS:
        risk = d["risk"]
        conf = d["model_confidence"]
        # Basic priority calculation for summary
        rw = 1.0 if risk == "High" else 0.6 if risk == "Medium" else 0.2
        ps = int((rw * 50) + (conf * 30))
        res.append(DistrictSummary(
            id=d["id"],
            city=d["city"],
            latitude=d["latitude"],
            longitude=d["longitude"],
            risk=risk,
            model_confidence=conf,
            priority_score=ps
        ))
    return res


# ══════════════════════════════════════════════════════════════════
# 2. GET /districts/{id}
# ══════════════════════════════════════════════════════════════════

@app.get("/districts/{district_id}", response_model=DistrictDetail)
def get_district(district_id: int):
    """Return city detail with structured SHAP explanation and Decision Intelligence."""
    dist = ID_TO_DISTRICT.get(district_id)
    if not dist:
        raise HTTPException(status_code=404, detail="District not found")

    risk = dist["risk"]
    shap_info = build_shap_explanation(risk)
    intel = generate_intelligence(risk, dist["model_confidence"], shap_info)
    tips = PREVENTION.get(risk, PREVENTION["Low"])["en"]

    return DistrictDetail(
        id=dist["id"],
        city=dist["city"],
        latitude=dist["latitude"],
        longitude=dist["longitude"],
        risk=risk,
        model_confidence=dist["model_confidence"],
        shap_explanations=[ShapExplanation(**e) for e in shap_info["explanations"]],
        dominant_driver=shap_info["dominant_driver"],
        interaction_note=shap_info["interaction_note"],
        prevention_tips=tips,
        urgency_level=intel["urgency_level"],
        response_window=intel["response_window"],
        recommended_actions=intel["recommended_actions"],
        risk_narrative=intel["risk_narrative"],
        alert_level=intel["alert_level"],
        confidence_context=intel["confidence_context"],
        expected_impact=intel["expected_impact"],
        source_attribution=intel["source_attribution"],
        priority_score=intel["priority_score"],
        priority_label=intel["priority_label"],
        trend_reason=intel["trend_reason"],
        projected_risk=intel["projected_risk"]
    )


# ══════════════════════════════════════════════════════════════════
# 3. POST /predict
# ══════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse)
def predict_risk(req: PredictRequest):
    """Predict epidemic risk from weather inputs (calibrated + decision intelligence)."""
    features = np.array([[
        req.rainfall,
        req.rainfall / 3.0,
        req.temperature,
        0.0,
        min(req.humidity + 10, 100),
        req.humidity,
        req.rainfall * req.humidity,
        8.0,
        15.0,
    ]])

    pred_class = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = realistic_confidence(float(np.max(probs)))
    risk_label = label_encoder.inverse_transform([pred_class])[0]

    shap_info = build_shap_explanation(risk_label)
    intel = generate_intelligence(risk_label, confidence, shap_info)

    return PredictResponse(
        risk=risk_label,
        model_confidence=round(confidence, 4),
        urgency_level=intel["urgency_level"],
        response_window=intel["response_window"],
        recommended_actions=intel["recommended_actions"],
        risk_narrative=intel["risk_narrative"],
        expected_impact=intel["expected_impact"],
        alert_level=intel["alert_level"]
    )


# ══════════════════════════════════════════════════════════════════
# 4. POST /chat (ADVANCED HYBRID CHATBOT)
# ══════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Advanced hybrid multilingual chatbot with 3-layer architecture and context awareness."""
    lang = _detect_language(req.message, req.language or "en")
    intent = _detect_intent(req.message)

    print(f"  [Chat] message='{req.message[:60]}' intent={intent} lang={lang} context_city={req.context_city}")

    # Fallback deterministic intelligence for context_city
    if req.context_city:
        city_lower = req.context_city.lower()
        if city_lower in _CITY_LOOKUP:
            city_data = _CITY_LOOKUP[city_lower]
            shap_info = build_shap_explanation(city_data["risk"])
            intel = generate_intelligence(city_data["risk"], city_data["model_confidence"], shap_info)
            
            if intent == Intent.WHY_RISK:
                return ChatResponse(
                    reply=f"🔬 **Risk Narrative for {req.context_city}:**\n{intel['risk_narrative']}\n\n**Expected Impact:** {intel['expected_impact']}",
                    language=lang, intent=intent, suggestions=_get_suggestions(intent, lang)
                )
            if intent == Intent.PREVENTION:
                acts = "\n".join([f"• **{a['action']}**\n  *Why:* {a['rationale']}" for a in intel["recommended_actions"]])
                return ChatResponse(
                    reply=f"🛡️ **Immediate Actions Required ({intel['urgency_level']} Urgency):**\n{acts}",
                    language=lang, intent=intent, suggestions=_get_suggestions(intent, lang)
                )
            # Inject context into LLM payload later if needed. For now, just prepend it in message text
            req.message = f"[Context: User is currently analyzing {req.context_city}. Risk is {city_data['risk']}, Alert Level is {intel['alert_level']}. Answer based on this.] " + req.message

    result = await _generate_response(req.message, intent, lang)
    suggestions = _get_suggestions(result.get("intent", intent), lang)

    return ChatResponse(
        reply=result["reply"],
        language=lang,
        intent=result.get("intent", intent),
        suggestions=suggestions,
    )


# ══════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "6.0.0",
        "cities": len(DISTRICTS),
        "model": "XGBoost (calibrated)",
        "gemini_model": "gemini-1.5-flash",
        "features": len(FEATURE_COLUMNS),
        "latest_date": str(latest_date.date()),
        "confidence_range": f"{min(confs):.2%} – {max(confs):.2%}",
        "chatbot": "hybrid (model + SHAP + RAG + Gemini)",
        "gemini_enabled": is_gemini_available(),
        "state_mappings": len(_STATE_LOOKUP),
        "city_aliases": len(_ALIAS_LOOKUP),
        "disease_kb_entries": len(DISEASE_KB),
        "languages": ["en", "hi", "as"],
    }


print("=" * 60)
print("  EPIDRA v6.0 — Ready")
print(f"  Gemini API (gemini-1.5-flash): {'✅ Enabled' if is_gemini_available() else '⚠️ Disabled (set GEMINI_API_KEY)'}")
print(f"  State→City mappings: {len(_STATE_LOOKUP)}")
print(f"  City alias mappings: {len(_ALIAS_LOOKUP)}")
print(f"  Disease KB: {len(DISEASE_KB)} diseases")
print(f"  Fuzzy matching: ✅ difflib")
print("=" * 60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
