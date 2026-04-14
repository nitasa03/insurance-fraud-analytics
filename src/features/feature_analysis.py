import os
from pathlib import Path

import pandas as pd


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]

SILVER_DIR = BASE_DIR / "data" / "silver"
REPORTS_DIR = BASE_DIR / "reports"

IEEE_TXN_CLEAN_PATH = SILVER_DIR / "ieee_transactions_clean.csv"
PORTO_CLEAN_PATH = SILVER_DIR / "porto_train_clean.csv"


# ---------- Helpers ----------
def ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_silver_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ieee = pd.read_csv(IEEE_TXN_CLEAN_PATH)
    df_porto = pd.read_csv(PORTO_CLEAN_PATH)
    return df_ieee, df_porto


def analyze_numeric_features(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str],
    top_n: int = 20
) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols and c != target_col]

    rows = []
    for col in numeric_cols:
        try:
            target0_mean = df.loc[df[target_col] == 0, col].mean()
            target1_mean = df.loc[df[target_col] == 1, col].mean()
            abs_diff = abs(target1_mean - target0_mean)

            rows.append({
                "column": col,
                "target0_mean": round(float(target0_mean), 6) if pd.notnull(target0_mean) else None,
                "target1_mean": round(float(target1_mean), 6) if pd.notnull(target1_mean) else None,
                "absolute_mean_diff": round(float(abs_diff), 6) if pd.notnull(abs_diff) else None
            })
        except Exception:
            continue

    result = pd.DataFrame(rows).sort_values("absolute_mean_diff", ascending=False).head(top_n)
    return result


def analyze_categorical_features(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str],
    top_n_categories: int = 10
) -> dict[str, pd.DataFrame]:
    output = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        try:
            summary = (
                df.groupby(col)[target_col]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"mean": "target_rate"})
                .sort_values("target_rate", ascending=False)
                .head(top_n_categories)
            )
            summary["target_rate"] = summary["target_rate"].round(6)
            output[col] = summary
        except Exception:
            continue

    return output


def save_categorical_summaries(summary_dict: dict[str, pd.DataFrame], prefix: str) -> None:
    for col, summary_df in summary_dict.items():
        safe_name = f"{prefix}_{col}_target_rate.csv".replace("/", "_")
        summary_df.to_csv(REPORTS_DIR / safe_name, index=False)


def run_feature_analysis() -> None:
    print("=" * 60)
    print(" Feature Usefulness Analysis")
    print("=" * 60)

    ensure_reports_dir()

    df_ieee, df_porto = load_silver_tables()

    print("\n[1] IEEE-CIS numeric feature analysis")
    ieee_exclude = ["TransactionID"]
    ieee_numeric = analyze_numeric_features(
        df=df_ieee,
        target_col="isFraud",
        exclude_cols=ieee_exclude,
        top_n=20
    )
    print(ieee_numeric.head(10).to_string(index=False))
    ieee_numeric.to_csv(REPORTS_DIR / "ieee_numeric_feature_signal.csv", index=False)

    print("\n[2] IEEE-CIS categorical feature analysis")
    ieee_cats = [
        "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"
    ]
    ieee_cat_summary = analyze_categorical_features(
        df=df_ieee,
        target_col="isFraud",
        categorical_cols=ieee_cats,
        top_n_categories=10
    )
    save_categorical_summaries(ieee_cat_summary, prefix="ieee")
    print(f"Saved IEEE categorical summaries for {len(ieee_cat_summary)} columns.")

    print("\n[3] Porto Seguro numeric feature analysis")
    porto_exclude = ["id"]
    porto_numeric = analyze_numeric_features(
        df=df_porto,
        target_col="target",
        exclude_cols=porto_exclude,
        top_n=20
    )
    print(porto_numeric.head(10).to_string(index=False))
    porto_numeric.to_csv(REPORTS_DIR / "porto_numeric_feature_signal.csv", index=False)

    print("\n[4] Porto Seguro selected categorical/binary-style analysis")
    porto_candidate_cols = [c for c in df_porto.columns if c.startswith("ps_ind_") or c.startswith("ps_car_")]
    porto_candidate_cols = porto_candidate_cols[:12]
    porto_cat_summary = analyze_categorical_features(
        df=df_porto,
        target_col="target",
        categorical_cols=porto_candidate_cols,
        top_n_categories=10
    )
    save_categorical_summaries(porto_cat_summary, prefix="porto")
    print(f"Saved Porto categorical summaries for {len(porto_cat_summary)} columns.")

    print("\nSaved analysis outputs to reports/")
    print("Next: use these results to choose engineered features.")


if __name__ == "__main__":
    run_feature_analysis()