"""
ingest.py — Bronze Layer: Raw Data Ingestion
Supports both local paths and Databricks DBFS paths.
"""
import pandas as pd
import os
import json
from datetime import datetime

# ── Path config ──────────────────────────────────────────
# Change USE_DATABRICKS to True when running on Databricks
USE_DATABRICKS = False

LOCAL_PATHS = {
    "transactions": "data/raw/ieee-cis/train_transaction.csv",
    "identity":     "data/raw/ieee-cis/train_identity.csv",
    "policies":     "data/raw/porto-seguro/train.csv",
}

DBFS_PATHS = {
    "transactions": "/dbfs/FileStore/ieee-cis/train_transaction.csv",
    "identity":     "/dbfs/FileStore/ieee-cis/train_identity.csv",
    "policies":     "/dbfs/FileStore/porto-seguro/train.csv",
}

PATHS = DBFS_PATHS if USE_DATABRICKS else LOCAL_PATHS


# ── Loaders ──────────────────────────────────────────────
def load_ieee_cis() -> dict:
    print("\n[IEEE-CIS] Loading raw data...")
    df_txn = pd.read_csv(PATHS["transactions"])
    df_idn = pd.read_csv(PATHS["identity"])
    print(f"  Transactions : {df_txn.shape[0]:,} rows x {df_txn.shape[1]} cols")
    print(f"  Identity     : {df_idn.shape[0]:,} rows x {df_idn.shape[1]} cols")
    print(f"  Fraud rate   : {df_txn['isFraud'].mean():.4f}  "
          f"({df_txn['isFraud'].sum():,} fraud cases)")
    return {"transactions": df_txn, "identity": df_idn}


def load_porto_seguro() -> dict:
    print("\n[Porto Seguro] Loading raw data...")
    df = pd.read_csv(PATHS["policies"])
    print(f"  Policies   : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Claim rate : {df['target'].mean():.4f}  "
          f"({df['target'].sum():,} claims filed)")
    return {"policies": df}


# ── Reporting ─────────────────────────────────────────────
def null_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    summary = (
        pd.DataFrame({"null_pct": null_pct, "dtype": df.dtypes.astype(str)})
        .query("null_pct > 0")
        .sort_values("null_pct", ascending=False)
    )
    print(f"\n  [Nulls — {name}]  {len(summary)} columns affected")
    print(summary.head(10).to_string())
    return summary


def save_schema_report(tables: dict) -> None:
    os.makedirs("reports", exist_ok=True)
    report = {
        name: {
            "rows": len(df),
            "cols": len(df.columns),
            "numeric": list(df.select_dtypes("number").columns),
            "categorical": list(df.select_dtypes("object").columns),
            "generated_at": datetime.now().isoformat(),
        }
        for name, df in tables.items()
    }
    with open("reports/schema_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Schema report saved → reports/schema_report.json")


# ── Main ──────────────────────────────────────────────────
def run_ingestion() -> dict:
    print("=" * 55)
    print("  BRONZE LAYER — Raw Data Ingestion")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    tables = {**load_ieee_cis(), **load_porto_seguro()}

    for name, df in tables.items():
        null_summary(df, name)

    save_schema_report(tables)

    print("\n  Done. All tables loaded.")
    print("  Next → run clean.py (Silver layer)")
    return tables


if __name__ == "__main__":
    run_ingestion()