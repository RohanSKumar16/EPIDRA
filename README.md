# EPIDRA 🚀

### Explainable Epidemic Prediction & Risk Analysis System

---

## 📌 Overview

EPIDRA is an AI-powered system that predicts epidemic risk using environmental factors such as temperature, humidity, and rainfall.
It integrates machine learning, explainable AI, and geospatial visualization to provide both accurate predictions and clear reasoning behind them.

---

## ⚙️ Key Features

* 🧠 Risk Prediction — XGBoost-based model for epidemic risk classification
* 🔍 Explainable AI — SHAP-based insights for feature contribution
* 🗺️ Geospatial Visualization — Map-based display of city-wise risk levels
* 💬 Interactive Chatbot — Query risk insights and recommendations
* ⚡ What-if Simulation — Analyze how environmental changes affect risk

---

## 🏗️ Architecture

Frontend → FastAPI Backend → ML Model (XGBoost) → SHAP → Insights

---

## 🧪 Tech Stack

* Backend: FastAPI, Python
* Machine Learning: XGBoost, Scikit-learn
* Explainability: SHAP
* Frontend: HTML, CSS, JavaScript
* Data Handling: Pandas, NumPy

---

## ▶️ Run the Project

### Backend

cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

### Frontend

Open: frontend/index.html

---

## 🧠 Highlights

* Combines prediction + explanation + visualization
* Focuses on interpretability, not just accuracy
* Designed as a practical decision-support system
