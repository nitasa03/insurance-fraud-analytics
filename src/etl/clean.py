import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
SILVER_DIR = BASE_DIR / "data" / "silver"

IEEE_TXN_PATH = RAW_DIR / "ieee-cis" / "train_transaction.csv"
IEEE_IDN_PATH = RAW_DIR / "ieee-cis" / "train_identity.csv"
PORTO_PATH = RAW_DIR / "porto-seguro" / "train.csv"


# ---------- Config from Day 2 EDA ----------
NULL_DROP_THRESHOLD = 70.0

IEEE_KEY_CATEGORICALS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"
]


# ---------- Helpers ----------
def ensure_output_dir() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_txn = pd.read_csv(IEEE_TXN_PATH)
    df_idn = pd.read_csv(IEEE_IDN_PATH)
    df_porto = pd.read_csv(PORTO_PATH)
    return df_txn, df_idn, df_porto


def build_null_strategy(df: pd.DataFrame, threshold: float = NULL_DROP_THRESHOLD) -> tuple[list[str], list[str], pd.DataFrame]:
    null_pct = (df.isnull().mean() * 100).round(2)

    null_df = (
        null_pct[null_pct > 0]
        .reset_index()
        .rename(columns={"index": "column", 0: "null_pct"})
    )

    null_df["dtype"] = null_df["column"].map(df.dtypes.astype(str).to_dict())
    null_df["null_count"] = null_df["column"].map(df.isnull().sum().to_dict())
    null_df["strategy"] = np.where(null_df["null_pct"] > threshold, "Drop", "Impute")
    null_df = null_df.sort_values("null_pct", ascending=False).reset_index(drop=True)

    drop_cols = null_df.loc[null_df["strategy"] == "Drop", "column"].tolist()
    impute_cols = null_df.loc[null_df["strategy"] == "Impute", "column"].tolist()

    return drop_cols, impute_cols, null_df


def impute_dataframe(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    exclude_cols = exclude_cols or []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if df[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    return df


def clean_ieee_transaction(df_txn: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    df_txn = df_txn.copy()

    drop_cols, impute_cols, null_strategy_df = build_null_strategy(df_txn, NULL_DROP_THRESHOLD)

    # Never drop the target even if logic changes later
    if "isFraud" in drop_cols:
        drop_cols.remove("isFraud")

    df_txn = df_txn.drop(columns=drop_cols, errors="ignore")

    # Recompute imputation candidates after dropping
    remaining_impute_cols = [c for c in impute_cols if c in df_txn.columns and c != "isFraud"]
    df_txn = impute_dataframe(df_txn, exclude_cols=["isFraud"])

    # Optional engineered helper from EDA finding
    if "TransactionAmt" in df_txn.columns:
        df_txn["TransactionAmt_log1p"] = np.log1p(df_txn["TransactionAmt"])

    return df_txn, drop_cols, remaining_impute_cols, null_strategy_df


def clean_ieee_identity(df_idn: pd.DataFrame) -> pd.DataFrame:
    df_idn = df_idn.copy()
    df_idn = impute_dataframe(df_idn, exclude_cols=["TransactionID"])
    return df_idn


def clean_porto(df_porto: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_porto = df_porto.copy()

    minus1_cols = []
    for col in df_porto.columns:
        if col in ["id", "target"]:
            continue

        try:
            minus1_count = (df_porto[col] == -1).sum()
            if minus1_count > 0:
                minus1_cols.append(col)
                df_porto[col] = df_porto[col].replace(-1, np.nan)
        except Exception:
            pass

    df_porto = impute_dataframe(df_porto, exclude_cols=["id", "target"])

    return df_porto, minus1_cols


def save_silver_tables(df_txn: pd.DataFrame, df_idn: pd.DataFrame, df_porto: pd.DataFrame) -> None:
    df_txn.to_csv(SILVER_DIR / "ieee_transactions_clean.csv", index=False)
    df_idn.to_csv(SILVER_DIR / "ieee_identity_clean.csv", index=False)
    df_porto.to_csv(SILVER_DIR / "porto_train_clean.csv", index=False)


def run_cleaning() -> dict:
    print("=" * 55)
    print("  SILVER LAYER — Cleaning")
    print("=" * 55)

    ensure_output_dir()

    df_txn, df_idn, df_porto = load_raw_tables()

    print("\n[1] Cleaning IEEE transaction table...")
    df_txn_clean, drop_cols, impute_cols, null_strategy_df = clean_ieee_transaction(df_txn)
    print(f"  Dropped columns : {len(drop_cols)}")
    print(f"  Imputed columns : {len(impute_cols)}")
    print(f"  Output shape    : {df_txn_clean.shape}")

    print("\n[2] Cleaning IEEE identity table...")
    df_idn_clean = clean_ieee_identity(df_idn)
    print(f"  Output shape    : {df_idn_clean.shape}")

    print("\n[3] Cleaning Porto Seguro table...")
    df_porto_clean, minus1_cols = clean_porto(df_porto)
    print(f"  -1 encoded cols : {len(minus1_cols)}")
    print(f"  Output shape    : {df_porto_clean.shape}")

    save_silver_tables(df_txn_clean, df_idn_clean, df_porto_clean)

    null_strategy_df.to_csv(SILVER_DIR / "ieee_null_strategy_from_cleaning.csv", index=False)

    print("\nSaved cleaned tables to data/silver/")

    return {
        "df_txn_clean": df_txn_clean,
        "df_idn_clean": df_idn_clean,
        "df_porto_clean": df_porto_clean,
        "drop_cols": drop_cols,
        "impute_cols": impute_cols,
        "minus1_cols": minus1_cols,
    }


if __name__ == "__main__":
    run_cleaning()