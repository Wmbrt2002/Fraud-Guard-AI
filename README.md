# 🛡️ Fraud Guard AI: Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-red.svg)

## 📌 Project Overview
**Fraud Guard AI** is a graduation project that introduces a Hybrid Machine Learning approach to detect credit card fraud in real-time. By combining Supervised Learning (Random Forest) for known fraud patterns and Unsupervised Learning (Isolation Forest) for novel anomalies, the system ensures high accuracy and a strong recall rate. 

Crucially, the system is designed to be **Privacy-Preserving** by utilizing PCA (Principal Component Analysis) to anonymize sensitive customer data (PII) before analysis.

## 🚀 Key Features
* **Hybrid Architecture:** Integrates Random Forest and Isolation Forest for comprehensive detection.
* **Privacy-First:** Uses PCA-transformed features to maintain PCI DSS compliance.
* **Imbalanced Data Handling:** Employs SMOTE to generate synthetic fraud cases during training, ensuring the model doesn't miss rare fraudulent transactions.
* **Real-Time Live Scanner:** An interactive Streamlit dashboard allowing financial analysts to input transaction details and receive instant Probability Scores (Confidence Levels).
* **High Recall:** Achieved a 97% Recall rate, effectively catching the vast majority of fraudulent activities with minimal false negatives.

## 🛠️ Technology Stack
* **Core Logic:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Model Serialization:** Joblib
* **User Interface:** Streamlit

## 📊 System Performance
Tested on a dataset with a highly imbalanced class distribution, the model achieved:
* **Accuracy:** 99%
* **Recall:** 0.97
* **Precision:** 0.85
* **F1-Score:** 0.91

## 👥 Meet The Team
This project was developed by undergraduate researchers at Al-Zaytoonah University of Jordan, Artificial Intelligence Department:
* **Waleed Abdullah Abdalrahman Hamza**
* **Abedalrhman Mohammed**
* **Abdallah Abu Odeh**
* **Hamza Abu Alhaija**
