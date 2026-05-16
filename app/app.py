# =========================================================
# BANK MARKETING AI DASHBOARD
# FINAL PRODUCTION VERSION (STABLE + SHAP FIXED)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Bank Marketing AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH / "bank.csv", sep=';')
df.columns = df.columns.str.strip()

df["y_bin"] = df["y"].map({"yes": 1, "no": 0})

# =========================
# LOAD MODELS
# =========================

rf_model = joblib.load(MODEL_PATH / "random_forest.pkl")
lr_model = joblib.load(MODEL_PATH / "logistic_regression.pkl")
feature_columns = joblib.load(MODEL_PATH / "feature_columns.pkl")
metrics = joblib.load(MODEL_PATH / "model_metrics.pkl")
feature_importance = joblib.load(MODEL_PATH / "feature_importance.pkl")

# =========================
# SAFE SHAP FUNCTION (FIXED)
# =========================

def explain_prediction(model, input_encoded, feature_columns):

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_encoded)

        # binary classification fix
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_columns,
            "Value": input_encoded.iloc[0].values,
            "SHAP Value": shap_values
        })

        shap_df["Abs"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values("Abs", ascending=False)

        return shap_df.drop(columns=["Abs"])

    except Exception as e:
        return None

# =========================
# SIDEBAR NAV
# =========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Dashboard", "Prediction", "Model Performance", "Feature Importance"]
)

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("Bank Marketing AI Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Subscription Rate", f"{df['y_bin'].mean()*100:.2f}%")

    st.divider()

    st.plotly_chart(px.histogram(df, x="y", color="y"), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="age", nbins=30), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="job", color="y"), use_container_width=True)

# =========================================================
# PREDICTION + SHAP
# =========================================================

elif page == "Prediction":

    st.title("Customer Subscription Prediction + Explainable AI")

    st.sidebar.subheader("Customer Input")

    age = st.sidebar.slider("Age", 18, 100, 35)
    balance = st.sidebar.number_input("Balance", 1000)
    day = st.sidebar.slider("Day", 1, 31, 15)
    duration = st.sidebar.slider("Duration", 0, 5000, 300)
    campaign = st.sidebar.slider("Campaign", 1, 50, 2)
    pdays = st.sidebar.slider("Pdays", -1, 500, -1)
    previous = st.sidebar.slider("Previous", 0, 20, 0)

    job = st.sidebar.selectbox("Job", df["job"].unique())
    marital = st.sidebar.selectbox("Marital", df["marital"].unique())
    education = st.sidebar.selectbox("Education", df["education"].unique())
    default = st.sidebar.selectbox("Default", df["default"].unique())
    housing = st.sidebar.selectbox("Housing", df["housing"].unique())
    loan = st.sidebar.selectbox("Loan", df["loan"].unique())
    contact = st.sidebar.selectbox("Contact", df["contact"].unique())
    month = st.sidebar.selectbox("Month", df["month"].unique())
    poutcome = st.sidebar.selectbox("Poutcome", df["poutcome"].unique())

    model_choice = st.sidebar.selectbox(
        "Model",
        ["Random Forest", "Logistic Regression"]
    )

    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    combined = pd.concat(
        [df.drop(["y", "y_bin"], axis=1), input_data],
        ignore_index=True
    )

    encoded = pd.get_dummies(combined, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    input_encoded = encoded.tail(1)

    model = rf_model if model_choice == "Random Forest" else lr_model

    if st.button("Predict Subscription"):

        prediction = int(model.predict(input_encoded)[0])
        probability = model.predict_proba(input_encoded)[0][1]

        st.subheader("Result")

        if prediction == 1:
            st.success("Customer WILL subscribe")
        else:
            st.error("Customer will NOT subscribe")

        st.metric("Probability", f"{probability * 100:.2f}%")

        st.plotly_chart(
            px.bar(x=["Probability"], y=[probability * 100]),
            use_container_width=True
        )

        # =========================
        # REAL-TIME SHAP (FIXED)
        # =========================

        st.subheader("AI Explanation (SHAP)")

        shap_df = explain_prediction(model, input_encoded, feature_columns)

        if shap_df is not None:

            col1, col2 = st.columns(2)

            with col1:
                st.write("Top Influencing Features")
                st.dataframe(shap_df.head(8), use_container_width=True)

            with col2:
                st.write("Least Influencing Features")
                st.dataframe(shap_df.tail(8), use_container_width=True)

            fig = px.bar(
                shap_df.head(10),
                x="SHAP Value",
                y="Feature",
                orientation="h",
                title="Feature Impact (This Customer)"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("SHAP could not be computed for this model.")

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("Model Performance Dashboard")

    df_metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            metrics["Logistic Regression"]["accuracy"],
            metrics["Random Forest"]["accuracy"]
        ],
        "F1 Score": [
            metrics["Logistic Regression"]["f1_score"],
            metrics["Random Forest"]["f1_score"]
        ],
        "ROC-AUC": [
            metrics["Logistic Regression"]["roc_auc"],
            metrics["Random Forest"]["roc_auc"]
        ]
    })

    st.dataframe(df_metrics, use_container_width=True)

    st.plotly_chart(
        px.bar(df_metrics,
               x="Model",
               y=["Accuracy", "F1 Score", "ROC-AUC"],
               barmode="group"),
        use_container_width=True
    )

    # =========================
    # CONFUSION MATRIX
    # =========================

    st.subheader("Confusion Matrix (Random Forest)")

    X = df.drop(["y", "y_bin"], axis=1)
    y = df["y_bin"]

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    y_pred = rf_model.predict(X_encoded)

    cm = confusion_matrix(y, y_pred)

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap="Blues")

    st.pyplot(fig_cm)

    # =========================
    # ROC CURVE
    # =========================

    st.subheader("ROC Curve (Random Forest)")

    y_prob = rf_model.predict_proba(X_encoded)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()

    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")

    ax_roc.legend()

    st.pyplot(fig_roc)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.title("Feature Importance")

    top = feature_importance.head(15)

    st.plotly_chart(
        px.bar(top, x="Importance", y="Feature", orientation="h"),
        use_container_width=True
    )

    st.info("SHAP is shown in Prediction page for real-time explanation.")