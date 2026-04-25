import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    recall_score,
    average_precision_score,
    roc_auc_score
)
from xgboost import XGBClassifier
import mlflow
import joblib

# =========================================================
# 1. Load Dataset
# =========================================================
DATA_PATH = "data/hospital_readmission.csv"   # Adjust if filename differs

df = pd.read_csv(DATA_PATH)
target = "readmitted"                         # Adjust if dataset uses another name

X = df.drop(columns=[target])
y = df[target]

# =========================================================
# 2. Train/Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================================================
# 3. Preprocessing Pipeline
# =========================================================
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# =========================================================
# 4. Model Definition (XGBoost)
# =========================================================
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# =========================================================
# 5. Train with MLflow Tracking
# =========================================================
mlflow.set_experiment("readmission_risk")

with mlflow.start_run():
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Log metrics
    mlflow.log_metric("f1", f1_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("auc_pr", average_precision_score(y_test, y_proba))
    mlflow.log_metric("auc_roc", roc_auc_score(y_test, y_proba))

    # Log model artifact
    mlflow.sklearn.log_model(clf, "model")

# =========================================================
# 6. Save Model Locally
# =========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/readmission_model.joblib")

print("Training complete. Model saved to models/readmission_model.joblib")
