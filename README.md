# 🌍 Climate-Aware Malaria Prediction & Explainable Healthcare AI System

## Digital Health Africa Capstone Project

A climate-aware and spatially informed Healthcare AI system developed for predicting malaria incidence across African countries using demographic, healthcare, environmental, temporal, and spatial indicators.

This project integrates:

- Machine Learning
- Climate Intelligence
- Explainable AI (SHAP)
- Spatial Analytics
- Temporal Forecasting
- Public Health Decision Support
- Streamlit Deployment

---

# 🚀 Live Dashboard

## Streamlit Deployment
https://malaria-app-app-hp7r78ub8agnjku4n9stpf.streamlit.app/
---

# 📌 Project Overview

Malaria remains one of the most significant public health challenges across Africa, accounting for approximately 95% of global malaria cases and deaths according to the World Health Organization (WHO).

This project developed a climate-aware and explainable machine learning system capable of forecasting malaria incidence across African countries using:

- healthcare indicators
- demographic variables
- climate data
- spatial information
- temporal forecasting features

The system integrates SHAP explainability and an interactive Streamlit dashboard to support healthcare interpretation, malaria surveillance, and public health decision-making.

---

# ✨ Key Features

- Climate-aware malaria forecasting
- Temporal lag analysis
- Spatial malaria risk mapping
- SHAP explainability
- What-if policy simulation
- Interactive Streamlit dashboard
- Public health recommendation system

---

# 📂 Datasets Used

Three integrated datasets were used:

## 1️⃣ Malaria Dataset

Includes:

- malaria incidence
- healthcare indicators
- sanitation access
- demographic variables
- socio-economic indicators

---

## 2️⃣ Rainfall Dataset

- Annual rainfall patterns
- Rainfall range: `<100 mm to >3,000 mm`

---

## 3️⃣ Temperature Dataset

- Annual temperature patterns
- Temperature range: `~15°C to >30°C`

---

## 📊 Final Integrated Dataset

- 594 records
- 54 African countries
- 30+ engineered features

---

# ⚙️ Feature Engineering

Advanced feature engineering techniques were applied to improve model performance and capture climate-sensitive malaria transmission dynamics.

## ⏳ Temporal Features

- malaria incidence lag variables
- rainfall lag variables
- temperature lag variables
- rolling climate averages

---

## 🌦️ Climate Features

- climate risk index
- rainfall-rural interaction
- latitude-temperature interaction
- longitude-rainfall interaction

---

## 🌍 Spatial Features

- spatial clustering (K-Means)
- country encoding
- geographic interaction variables

---

# 🤖 Machine Learning Models

The following regression models were evaluated:

| Model | Average R² Score |
|---|---|
| Linear Regression | -9.90 |
| Random Forest Regressor | 0.54 |
| Gradient Boosting Regressor | 0.47 |
| XGBoost Regressor | 0.41 |

---

# 📈 Final Model Performance

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Untuned Random Forest | 30.45 | 48.16 | 0.87 |
| Tuned Random Forest | 57.36 | 70.54 | 0.73 |

---

# 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to improve model transparency and identify the strongest drivers of malaria prediction.

## 🔍 Key Findings

- Previous malaria incidence was the strongest predictor
- Climate-related variables significantly influenced malaria burden
- Rainfall effects were amplified in vulnerable rural populations
- Sanitation access contributed to malaria vulnerability
- Spatial heterogeneity strongly influenced transmission patterns

---

## 📌 Example SHAP Insight

Previous malaria incidence contributed approximately:

```text
+209 SHAP units toward one high-risk malaria prediction.

## 🧪 What-If Policy Simulation

A sanitation-access intervention simulation was implemented to evaluate the directional sensitivity of the model to public health improvements.

📍 Example Scenario
Country: Mozambique
Year: 2008
Baseline incidence: 399.79 cases per 1,000 population at risk
20% sanitation increase
Predicted incidence after intervention: 399.19
Estimated reduction: 0.60 cases per 1,000 population at risk

## 🖥️ Streamlit Dashboard Features

The deployed dashboard includes:

Dashboard Overview
Geographic Malaria Risk Mapping
Malaria Incidence Prediction Tool
What-If Policy Simulation
SHAP Explainability
Public Health Policy Recommendations

## ⚖️ Ethical Considerations

This project recognizes important ethical considerations associated with healthcare AI deployment.

Predictions are non-causal
Explainability was incorporated using SHAP
The dashboard supports decision-making rather than replacing healthcare professionals
Responsible deployment requires contextual healthcare interpretation


## ⚠️ Limitations

Relatively small dataset size (594 records)
Country-level aggregation may not fully capture local transmission dynamics
Additional environmental variables such as humidity were not included
Streamlit deployment introduced minor model-pipeline compatibility considerations


## 🛠️ Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
Plotly
Streamlit

## 📁 Repository Structure

malaria-streamlit-app/
│
├── app.py
├── requirements.txt
├── advanced_feature_engineered_malaria_dataset.csv
├── final_malaria_prediction_pipeline.pkl
├── README.md

## Clone Repository
git clone https://github.com/drsam-israel/malaria-streamlit-app.git

## Install Dependencies
pip install -r requirements.txt

## Run Streamlit App
streamlit run app.py

## 📚 References

World Health Organization (WHO)
World Malaria Report 2023
Breiman (2001) – Random Forests
Chen & Guestrin (2016) – XGBoost
Lundberg & Lee (2017) – SHAP Explainability
Scikit-learn Documentation
Streamlit Documentation

## 👨‍⚕️ Author

Dr. Samuel Israel, M.Sc

Healthcare & Clinical Data Scientist
Machine Learning Engineering & Deployable AI
Digital Health Africa Capstone Project
