from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
SILVER_DIR = BASE_DIR / "data" / "silver"

IEEE_TXN_CLEAN_PATH = SILVER_DIR / "ieee_transactions_clean.csv"
IEEE_IDN_CLEAN_PATH = SILVER_DIR / "ieee_identity_clean.csv"
PORTO_CLEAN_PATH = SILVER_DIR / "porto_train_clean.csv"


def load_clean_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_txn = pd.read_csv(IEEE_TXN_CLEAN_PATH)
    df_idn = pd.read_csv(IEEE_IDN_CLEAN_PATH)
    df_porto = pd.read_csv(PORTO_CLEAN_PATH)
    return df_txn, df_idn, df_porto


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name}: missing required columns: {missing}")
    print(f"{table_name}: required columns present.")


def validate_no_nulls_in_target(df: pd.DataFrame, target_col: str, table_name: str) -> None:
    if df[target_col].isnull().sum() > 0:
        raise ValueError(f"{table_name}: target column '{target_col}' contains nulls.")
    print(f"{table_name}: target column has no nulls.")


def validate_duplicates(df: pd.DataFrame, table_name: str) -> None:
    dup_count = int(df.duplicated().sum())
    print(f"{table_name}: duplicate rows = {dup_count}")


def validate_remaining_nulls(df: pd.DataFrame, table_name: str) -> None:
    remaining_null_cols = int((df.isnull().sum() > 0).sum())
    print(f"{table_name}: columns with remaining nulls = {remaining_null_cols}")


def validate_target_distribution(df: pd.DataFrame, target_col: str, table_name: str) -> None:
    rate = round(df[target_col].mean() * 100, 4)
    count = int(df[target_col].sum())
    print(f"{table_name}: positive rate = {rate}% | positive count = {count}")


def run_validation() -> None:
    print("=" * 55)
    print("  SILVER LAYER — Validation")
    print("=" * 55)

    df_txn, df_idn, df_porto = load_clean_tables()

    print("\n[1] Validate IEEE transactions")
    validate_required_columns(df_txn, ["isFraud", "TransactionID"], "IEEE transactions")
    validate_no_nulls_in_target(df_txn, "isFraud", "IEEE transactions")
    validate_duplicates(df_txn, "IEEE transactions")
    validate_remaining_nulls(df_txn, "IEEE transactions")
    validate_target_distribution(df_txn, "isFraud", "IEEE transactions")

    print("\n[2] Validate IEEE identity")
    validate_required_columns(df_idn, ["TransactionID"], "IEEE identity")
    validate_duplicates(df_idn, "IEEE identity")
    validate_remaining_nulls(df_idn, "IEEE identity")

    print("\n[3] Validate Porto Seguro")
    validate_required_columns(df_porto, ["id", "target"], "Porto Seguro")
    validate_no_nulls_in_target(df_porto, "target", "Porto Seguro")
    validate_duplicates(df_porto, "Porto Seguro")
    validate_remaining_nulls(df_porto, "Porto Seguro")
    validate_target_distribution(df_porto, "target", "Porto Seguro")

    print("\nValidation complete.")


if __name__ == "__main__":
    run_validation()