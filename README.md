# Matchverse - AI Matrimonial Recommendation System 💍

![Python](https://img.shields.io/badge/Python-3.9-blue) ![XGBoost](https://img.shields.io/badge/AI-XGBoost-orange) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

> 🏆 **Event Participation:** This project was developed for **MatchVerse 2025: AI Matrimonial Challenge** at **Genesis 2025** (JMI Tech Fest).

Matchverse is a personalized recommendation engine designed to predict matrimonial compatibility. It utilizes **XGBoost** for ranking candidate profiles and serves predictions via a **Streamlit** web interface.

## 🚀 Features
- **Smart Matching:** Uses Gradient Boosting to analyze user interactions and profile attributes.
- **Real-time Inference:** Low-latency recommendations served via a serialized JSON model.
- **Interactive UI:** Responsive Streamlit dashboard for browsing matches.

## 🛠️ Technical Architecture
The system follows a standard ML pipeline:
1. **Data Ingestion:** Merges user profiles with interaction history.
2. **Training:** Fine-tuned XGBoost Classifier (optimized for precision).
3. **Deployment:** Model is serialized and served via a lightweight Streamlit frontend.

## ⚠️ Note
*The datasets (`users.csv`, `interactions.csv`) included in this repository are **synthetic/mock data** generated for demonstration purposes. No real personal information is stored.*
