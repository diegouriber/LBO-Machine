import numpy as np
import pandas as pd

from .config import (
    RAW_FINANCIAL_FILE,
    FINANCIAL_METRICS_FILE,
    EXCLUDED_SECTORS,
)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing unnamed separator columns
    and standardizing text formatting.
    """
    df = df.copy()

    # Drop empty separator columns like "Unnamed: 5", "Unnamed: 12", etc.
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Standardize column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def load_raw_financial_screen() -> pd.DataFrame:
    """
    Load the existing financial screening Excel file.

    This file is currently more reliable than yfinance because it already
    contains the calculated LBO financial metrics and scores.
    """
    if not RAW_FINANCIAL_FILE.exists():
        raise FileNotFoundError(f"Missing raw financial file: {RAW_FINANCIAL_FILE}")

    df = pd.read_excel(RAW_FINANCIAL_FILE)
    df = clean_column_names(df)

    return df


def standardize_financial_screen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the financial screening dataset into the format expected
    by the rest of the LBO Machine pipeline.
    """
    df = df.copy()

    if "ticker" not in df.columns:
        raise ValueError("Financial dataset must contain a 'ticker' column.")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip()

    # Remove financial sector firms because traditional LBO ratios
    # are not economically comparable for them.
    if "sector" in df.columns:
        df = df[~df["sector"].isin(EXCLUDED_SECTORS)].copy()

    # Convert key numeric columns
    numeric_cols = [
        "lbo_rank",
        "year",
        "fcf",
        "fcf_margin",
        "fcf_conversion",
        "fcf_score",
        "fcf_margin_score",
        "fcf_conversion_score",
        "cash_flow_strength_score",
        "fcf_to_debt",
        "fcf_to_debt_score",
        "deleveraging_capacity_score",
        "debt_to_ebitda",
        "net_debt_to_ebitda",
        "debt_to_ebitda_score",
        "net_debt_to_ebitda_score",
        "leverage_score",
        "interest_coverage",
        "cash_interest_coverage",
        "interest_coverage_score",
        "cash_interest_coverage_score",
        "coverage_score",
        "capex_to_revenue",
        "capex_to_ebitda",
        "capex_to_revenue_score",
        "capex_to_ebitda_score",
        "capex_burden_score",
        "lbo_financial_score",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert boolean filter columns
    bool_cols = [
        "missing_critical_data",
        "pass_fcf_filter",
        "pass_interest_coverage_filter",
        "pass_debt_to_ebitda_filter",
        "pass_current_ratio_filter",
        "pass_capex_to_ebitda_filter",
        "pass_hard_filters",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Keep latest year if duplicates exist
    if "year" in df.columns:
        df = (
            df.sort_values(["ticker", "year"])
            .groupby("ticker", as_index=False)
            .tail(1)
            .copy()
        )

    # Normalize financial score to 0-1.
    # Original lbo_financial_score is on a 0-100 scale.
    if "lbo_financial_score" in df.columns:
        df["financial_feasibility_score"] = df["lbo_financial_score"] / 100
    else:
        raise ValueError("Financial dataset must contain 'lbo_financial_score'.")

    # Add red flag for financial feasibility concerns
    df["financial_red_flag"] = False

    if "interest_coverage" in df.columns:
        df["financial_red_flag"] = df["financial_red_flag"] | (df["interest_coverage"] < 2.0)

    if "debt_to_ebitda" in df.columns:
        df["financial_red_flag"] = df["financial_red_flag"] | (df["debt_to_ebitda"] > 4.0)

    if "fcf_margin" in df.columns:
        df["financial_red_flag"] = df["financial_red_flag"] | (df["fcf_margin"] <= 0)

    # Data completeness diagnostic
    required_metric_cols = [
        "fcf_margin",
        "fcf_to_debt",
        "debt_to_ebitda",
        "interest_coverage",
        "capex_to_ebitda",
        "lbo_financial_score",
    ]

    available_required_cols = [col for col in required_metric_cols if col in df.columns]
    df["financial_metric_count"] = df[available_required_cols].notna().sum(axis=1)

    df["financial_data_complete"] = (
        df["financial_metric_count"] == len(available_required_cols)
    )

    # Clean sort
    df = df.sort_values("financial_feasibility_score", ascending=False).reset_index(drop=True)

    return df


def save_financial_metrics() -> pd.DataFrame:
    """
    Load, clean, standardize, and save the financial metrics dataset.
    """
    raw_df = load_raw_financial_screen()
    clean_df = standardize_financial_screen(raw_df)

    FINANCIAL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(FINANCIAL_METRICS_FILE, index=False)

    print(f"Loaded raw financial rows: {len(raw_df)}")
    print(f"Saved cleaned financial rows: {len(clean_df)}")
    print(f"Saved file: {FINANCIAL_METRICS_FILE}")

    if "financial_feasibility_score" in clean_df.columns:
        print("\nTop 10 financial candidates:")
        print(
            clean_df[
                [
                    "ticker",
                    "sector",
                    "lbo_financial_score",
                    "financial_feasibility_score",
                    "lbo_status",
                    "lbo_rating",
                    "financial_red_flag",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    return clean_df


if __name__ == "__main__":
    save_financial_metrics()