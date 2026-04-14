import numpy as np
import pandas as pd


def add_vehicle_driver_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ps_car_13" in df.columns and "ps_ind_03" in df.columns:
        df["vehicle_driver_interaction"] = df["ps_car_13"] * df["ps_ind_03"]

    return df


def add_binary_risk_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    binary_cols = [
        "ps_ind_06_bin",
        "ps_ind_07_bin",
        "ps_ind_08_bin",
        "ps_ind_09_bin",
        "ps_ind_16_bin",
        "ps_ind_17_bin",
        "ps_ind_18_bin"
    ]

    existing_binary_cols = [col for col in binary_cols if col in df.columns]

    if existing_binary_cols:
        df["binary_risk_count"] = df[existing_binary_cols].sum(axis=1)

    return df


def add_high_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    high_risk_conditions = []

    if "ps_ind_03" in df.columns:
        high_risk_conditions.append(df["ps_ind_03"] > df["ps_ind_03"].median())

    if "ps_car_13" in df.columns:
        high_risk_conditions.append(df["ps_car_13"] > df["ps_car_13"].median())

    if "binary_risk_count" in df.columns:
        high_risk_conditions.append(df["binary_risk_count"] >= 2)

    if high_risk_conditions:
        combined_condition = np.column_stack(high_risk_conditions).sum(axis=1) >= 2
        df["high_risk_flag"] = combined_condition.astype(int)

    return df


def build_porto_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_vehicle_driver_interaction(df)
    df = add_binary_risk_count(df)
    df = add_high_risk_flag(df)

    return df