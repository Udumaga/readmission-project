Hospital Readmission Risk Prediction — End‑to‑End ML Pipeline
A complete, production‑grade machine learning system for predicting 30‑day hospital readmission risk.
This project demonstrates a full MLOps workflow including:

Data preprocessing

Model training with XGBoost

MLflow experiment tracking

FastAPI model serving

Docker containerization

Drift monitoring with Evidently

Model interpretability with SHAP

Project Overview
Hospital readmissions are costly and often preventable. This project builds a reproducible ML pipeline that predicts whether a patient is likely to be readmitted within 30 days, enabling early intervention and improved care management.

The pipeline includes:

Training pipeline (src/train.py)

Prediction API (api/app.py)

Model artifacts (models/)

Monitoring reports (monitoring/)

Interpretability notebooks (notebooks/)

Containerized deployment (Dockerfile)


For major changes, please open an issue first to discuss what you’d like to modify.
