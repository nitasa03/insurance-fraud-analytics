# Insurance Risk & Claims Fraud Analytics Platform

An end-to-end data engineering and machine learning project simulating a real-world insurance fraud detection and risk scoring system. Built to demonstrate skills relevant to ML Engineer, AI Engineer, and Data Engineer roles in the Canadian banking and insurance sector.

---

## Project Architecture

```
Raw Data (Kaggle)
    │
    ▼
Bronze Layer — Raw ingestion, schema validation, null profiling
    │
    ▼
Silver Layer — Cleaning, outlier capping, Great Expectations validation
    │
    ▼
Gold Layer — Feature engineering, PySpark velocity features, Parquet storage
    │
    ▼
SQL / Data Warehouse — Star schema (fact_claims + 4 dims), SCD Type 2
    │
    ▼
ML Models — XGBoost (fraud) + Random Forest (risk), SHAP explainability
    │
    ▼
Decision Engine — Auto-approve / Manual review / Fraud alert (STP logic)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | Python, pandas, ydata-profiling |
| Data validation | Great Expectations |
| Feature engineering | pandas, PySpark (Window functions) |
| Data warehouse | DuckDB, SQL (star schema, SCD Type 2) |
| ML models | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Notebooks | Jupyter |
| Version control | Git + GitHub |

---

## Datasets

| Dataset | Source | Size | Target |
|---|---|---|---|
| IEEE-CIS Fraud Detection | Kaggle | 590k transactions | `isFraud` (binary) |
| Porto Seguro Safe Driver | Kaggle | 595k policies | `target` (binary, claim filed) |

---

## Model Performance

> Results updated after Day 6

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Fraud detection (XGBoost) | TBD | TBD | TBD | TBD |
| Risk scoring (Random Forest) | TBD | TBD | TBD | TBD |
| Fraud baseline (Logistic Reg.) | TBD | TBD | TBD | TBD |

---

## Repository Structure

```
insurance-fraud-analytics/
├── data/
│   ├── raw/            # gitignored — download from Kaggle
│   ├── silver/         # cleaned CSVs
│   └── gold/           # feature-engineered Parquet files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_etl_pipeline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_fraud_model.ipynb
│   ├── 05_risk_model.ipynb
│   └── 06_decision_engine.ipynb
├── src/
│   ├── etl/            # ingest.py, clean.py, validate.py
│   ├── features/       # fraud_features.py, risk_features.py
│   ├── models/         # fraud_model.py, risk_model.py, evaluate.py
│   ├── decision/       # decision_engine.py
│   └── pyspark/        # velocity_features.py
├── sql/
│   ├── ddl/            # table creation scripts
│   └── analytics/      # business insight queries
├── reports/            # data quality HTML + model evaluation
├── tests/              # unit tests
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/nitasa03/insurance-fraud-analytics.git
cd insurance-fraud-analytics

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets (requires Kaggle API key)
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-cis
kaggle competitions download -c porto-seguro-safe-driver-prediction -p data/raw/porto-seguro

# 5. Run ingestion
python src/etl/ingest.py

# 6. Open notebooks in order
jupyter notebook notebooks/
```

---

## Key Concepts Demonstrated

- **Medallion Architecture** (Bronze / Silver / Gold) — industry-standard pipeline layering used at major Canadian banks
- **SCD Type 2** on `dim_customer` — tracks historical changes to customer attributes with `effective_from` / `effective_to` / `is_current`
- **SMOTE** applied only to training split — prevents data leakage from oversampling
- **SHAP explainability** — each fraud prediction comes with feature-level contribution scores, aligned with OSFI model risk management guidelines
- **Straight-Through Processing (STP)** — decision engine mirrors real P&C insurance claim routing logic
- **PySpark Window functions** — velocity feature computation at scale

---

## Business Context

This project simulates the fraud analytics infrastructure used by Canadian P&C (property and casualty) insurers. Key domain concepts reflected:

- **SIU (Special Investigations Unit)** — fraud alerts are routed to SIU with a 4-hour SLA
- **STP (Straight-Through Processing)** — low-risk claims are auto-approved without human review
- **OSFI guidelines** — model documentation and explainability align with the Office of the Superintendent of Financial Institutions' model risk management expectations

---

## Author

**Nibedita Satapathy**  
MSc in Machine Learning / AI (GenAI specialization)  
