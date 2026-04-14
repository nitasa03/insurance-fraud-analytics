import numpy as np
import pandas as pd


def add_transaction_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"])

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionDT" in df.columns:
        # Approximate hour extraction from seconds-based transaction time
        df["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype(int)
        df["is_night"] = df["hour_of_day"].between(23, 23) | df["hour_of_day"].between(0, 5)
        df["is_night"] = df["is_night"].astype(int)

    return df


#def add_email_domain_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = (
            df["P_emaildomain"].fillna("missing") == df["R_emaildomain"].fillna("missing")
        ).astype(int)

    return df


def add_frequency_encoding(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            freq_map = df[col].value_counts(dropna=False).to_dict()
            df[f"{col}_freq"] = df[col].map(freq_map)

    return df


def add_card_transaction_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "card1" in df.columns:
        df["card1_txn_count"] = df.groupby("card1")["card1"].transform("count")

    return df


def add_card4_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "card4" in df.columns and "TransactionAmt" in df.columns:
        df["card4_amt_mean"] = df.groupby("card4")["TransactionAmt"].transform("mean")
        df["amt_deviation_card4"] = df["TransactionAmt"] - df["card4_amt_mean"]

    return df


def build_ieee_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_transaction_amount_log(df)
    df = add_time_features(df)
    #df = add_email_domain_match(df)
    df = add_frequency_encoding(df, ["ProductCD", "card4", "card6"])
    df = add_card_transaction_count(df)
    df = add_card4_amount_features(df)

    return df