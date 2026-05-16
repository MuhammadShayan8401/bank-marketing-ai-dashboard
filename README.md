# 💰 Bank Marketing AI Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)

An end-to-end **Machine Learning + Explainable AI (XAI) project** that predicts whether a bank customer will subscribe to a term deposit based on marketing campaign data.

It includes:

* Full ML pipeline
* Data preprocessing & encoding
* Classification models
* Evaluation metrics
* Explainable AI using SHAP
* Interactive Streamlit dashboard

---

## 📌 Project Objective

To predict **customer subscription to term deposits** and help banks optimize marketing strategies by targeting high-probability customers.

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository – Bank Marketing Dataset

### Features:

* Customer Info: age, job, marital status, education
* Financial Info: balance, housing loan, personal loan
* Campaign Info: contact type, duration, campaign count
* Historical Data: pdays, previous outcomes
* Target: y (yes/no subscription)

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Plotly & Matplotlib
* SHAP (Explainable AI)
* Streamlit

---

## 🧠 Machine Learning Models

* Logistic Regression
* Random Forest Classifier

---

## 📈 Evaluation Metrics

* Accuracy Score
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve

---

## 🔍 Explainable AI (SHAP)

This project integrates SHAP (SHapley Additive exPlanations) to:

* Explain individual predictions in real-time
* Show feature impact per customer
* Identify top influencing factors
* Improve model transparency

---

## 📊 Dashboard Features

### 🏠 Overview Page

* Dataset statistics
* Subscription distribution
* Age and job insights

### 🔮 Prediction Page

* User input form
* Real-time prediction
* Probability score
* SHAP explanation (why prediction happened)

### 📉 Model Performance Page

* Confusion Matrix
* ROC Curve
* Model comparison (Logistic Regression vs Random Forest)

### 📊 Feature Importance Page

* Top features visualization
* SHAP global interpretation

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/MuhammadShayan8401/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

```bash
python src/train_model.py
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
bank-marketing-ai/
│
├── data/
│   └── bank.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── feature_columns.pkl
│   ├── model_metrics.pkl
│   └── feature_importance.pkl
│
├── src/
│   └── train_model.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 📌 Key Learnings

* End-to-end ML pipeline development
* Feature encoding & preprocessing
* Model evaluation techniques
* Handling real-world categorical data
* Explainable AI (SHAP)
* Building interactive dashboards with Streamlit

---

## 👨‍💻 Author

**Muhammad Shayan Ahmed**
Data Science Intern 

GitHub: MuhammadShayan8401

---

## 🚀 Future Improvements

* Add XGBoost / LightGBM models
* Deploy on Streamlit Cloud / Render
* Add authentication system
* Add real-time API (FastAPI backend)
* Improve SHAP performance for large datasets
* Add batch prediction (CSV upload)

---

## 🌐 Live Demo

🚀 **Streamlit App:** [Click here to view live demo](https://muhammadshayan8401-bank-marketing-ai-dashboard-appapp-vz2bgu.streamlit.app/)

---

## 📜License

This project is created for educational and internship purposes.
