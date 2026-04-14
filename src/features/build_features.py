from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_config import (
    IEEE_ENGINEERED_FEATURES,
    IEEE_GOLD_FILE,
    IEEE_TARGET,
    PORTO_ENGINEERED_FEATURES,
    PORTO_GOLD_FILE,
    PORTO_TARGET,
)
from src.features.fraud_features import build_ieee_fraud_features
from src.features.risk_feature import build_porto_risk_features


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]

SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"

IEEE_SILVER_PATH = SILVER_DIR / "ieee_transactions_clean.csv"
PORTO_SILVER_PATH = SILVER_DIR / "porto_train_clean.csv"


# ---------- Helpers ----------
def ensure_gold_dir() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)


def load_silver_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ieee = pd.read_csv(IEEE_SILVER_PATH)
    df_porto = pd.read_csv(PORTO_SILVER_PATH)
    return df_ieee, df_porto


def existing_columns(df: pd.DataFrame, expected_cols: list[str]) -> list[str]:
    return [col for col in expected_cols if col in df.columns]


def validate_engineered_features(
    df: pd.DataFrame,
    expected_cols: list[str],
    target_col: str,
    table_name: str
) -> None:
    print(f"\nValidating {table_name} engineered features...")

    present_cols = existing_columns(df, expected_cols)
    missing_cols = [col for col in expected_cols if col not in df.columns]

    print(f"  Expected engineered cols : {len(expected_cols)}")
    print(f"  Present engineered cols  : {len(present_cols)}")

    if missing_cols:
        print(f"  Missing engineered cols  : {missing_cols}")

    if target_col not in df.columns:
        raise ValueError(f"{table_name}: target column '{target_col}' is missing.")

    if present_cols:
        null_counts = df[present_cols].isnull().sum().sum()
        inf_counts = np.isinf(df[present_cols].select_dtypes(include=[np.number])).sum().sum()

        print(f"  Null values in engineered cols : {int(null_counts)}")
        print(f"  Inf values in engineered cols  : {int(inf_counts)}")

        if null_counts > 0:
            raise ValueError(f"{table_name}: engineered features contain null values.")

        if inf_counts > 0:
            raise ValueError(f"{table_name}: engineered features contain infinite values.")

    print(f"  {table_name} feature validation passed.")


def save_gold_tables(df_ieee: pd.DataFrame, df_porto: pd.DataFrame) -> None:
    df_ieee.to_csv(GOLD_DIR / IEEE_GOLD_FILE, index=False)
    df_porto.to_csv(GOLD_DIR / PORTO_GOLD_FILE, index=False)


def run_build_features() -> dict:
    print("=" * 60)
    print("  Build Gold Feature Datasets")
    print("=" * 60)

    ensure_gold_dir()

    print("\n[1] Loading Silver datasets...")
    df_ieee, df_porto = load_silver_tables()
    print(f"  IEEE Silver shape  : {df_ieee.shape}")
    print(f"  Porto Silver shape : {df_porto.shape}")

    print("\n[2] Building IEEE engineered features...")
    df_ieee_gold = build_ieee_fraud_features(df_ieee)
    print(f"  IEEE Gold shape    : {df_ieee_gold.shape}")

    print("\n[3] Building Porto engineered features...")
    df_porto_gold = build_porto_risk_features(df_porto)
    print(f"  Porto Gold shape   : {df_porto_gold.shape}")

    print("\n[4] Validating engineered features...")
    validate_engineered_features(
        df=df_ieee_gold,
        expected_cols=IEEE_ENGINEERED_FEATURES,
        target_col=IEEE_TARGET,
        table_name="IEEE-CIS"
    )
    validate_engineered_features(
        df=df_porto_gold,
        expected_cols=PORTO_ENGINEERED_FEATURES,
        target_col=PORTO_TARGET,
        table_name="Porto Seguro"
    )

    print("\n[5] Saving Gold datasets...")
    save_gold_tables(df_ieee_gold, df_porto_gold)
    print(f"  Saved: {GOLD_DIR / IEEE_GOLD_FILE}")
    print(f"  Saved: {GOLD_DIR / PORTO_GOLD_FILE}")

    print("\nGold feature pipeline complete.")

    return {
        "df_ieee_gold": df_ieee_gold,
        "df_porto_gold": df_porto_gold,
    }


if __name__ == "__main__":
    run_build_features()