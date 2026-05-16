# =========================================================
# BANK MARKETING AI DASHBOARD
# Model Training Pipeline (PROFESSIONAL VERSION)
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================
# PROJECT PATH SETUP (IMPORTANT FIX)
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"

# Create folders if not exist
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH / "bank.csv",
    sep=';'
)

print("Dataset Loaded Successfully")
print(df.head())

# =========================
# DATA PREPROCESSING
# =========================

print("\nPreprocessing dataset...")

# One-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df_encoded.drop("y_yes", axis=1)
y = df_encoded["y_yes"]

# Save feature columns (IMPORTANT FIXED PATH)
joblib.dump(
    X.columns.tolist(),
    MODEL_PATH / "feature_columns.pkl"
)

print("Feature columns saved.")

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================
# LOGISTIC REGRESSION
# =========================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_prob_lr)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", round(lr_accuracy, 4))
print("F1 Score:", round(lr_f1, 4))
print("ROC-AUC:", round(lr_auc, 4))

# =========================
# RANDOM FOREST
# =========================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", round(rf_accuracy, 4))
print("F1 Score:", round(rf_f1, 4))
print("ROC-AUC:", round(rf_auc, 4))

# =========================
# SAVE MODELS (FIXED PATHS)
# =========================

print("\nSaving models...")

joblib.dump(
    lr_model,
    MODEL_PATH / "logistic_regression.pkl"
)

joblib.dump(
    rf_model,
    MODEL_PATH / "random_forest.pkl"
)

# =========================
# SAVE METRICS
# =========================

metrics = {
    "Logistic Regression": {
        "accuracy": lr_accuracy,
        "f1_score": lr_f1,
        "roc_auc": lr_auc
    },
    "Random Forest": {
        "accuracy": rf_accuracy,
        "f1_score": rf_f1,
        "roc_auc": rf_auc
    }
}

joblib.dump(
    metrics,
    MODEL_PATH / "model_metrics.pkl"
)

print("Metrics saved successfully.")

# =========================
# FEATURE IMPORTANCE
# =========================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

joblib.dump(
    feature_importance,
    MODEL_PATH / "feature_importance.pkl"
)

print("Feature importance saved.")

# =========================
# COMPLETION
# =========================

print("\n===================================")
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("===================================")