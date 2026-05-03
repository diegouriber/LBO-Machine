import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import RAW_STAGNATION_FILE, NLP_METRICS_FILE


def load_raw_nlp_metrics() -> pd.DataFrame:
    """
    Load the raw stagnation metrics file created from the filing/NLP pipeline.
    """
    if not RAW_STAGNATION_FILE.exists():
        raise FileNotFoundError(f"Missing raw NLP file: {RAW_STAGNATION_FILE}")

    df = pd.read_csv(RAW_STAGNATION_FILE)

    # Standardize column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def is_empty_for_removal(value) -> bool:
    """
    Identify values that should be treated as empty/unusable.
    Handles NaN, empty strings, stringified lists, and common null strings.
    """
    if pd.isna(value):
        return True

    if isinstance(value, str):
        cleaned = value.strip().lower()

        if cleaned in ["", "nan", "none", "null", "[]", "{}", "na", "n/a"]:
            return True

    return False


def find_ticker_column(df: pd.DataFrame) -> str:
    """
    Find the ticker column in the NLP dataset.
    """
    possible_cols = ["ticker", "symbol", "stock", "company_ticker"]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a ticker column. Expected one of: "
        f"{possible_cols}. Actual columns: {df.columns.tolist()}"
    )


def select_available_nlp_components(df: pd.DataFrame) -> list[str]:
    """
    Select NLP component columns that exist in the current dataset.

    The current project has used different naming versions across notebooks,
    so this function is intentionally flexible.
    """
    preferred_components = [
        "innovation_decay_slope",
        "strategic_decay_slope",
        "strategic_decay",
        "topic_rigidity",
        "sentiment_growth_correlation",
        "sentiment_growth_misalignment",
    ]

    available_components = [col for col in preferred_components if col in df.columns]

    if len(available_components) == 0:
        raise ValueError(
            "No usable NLP component columns found. "
            f"Looked for: {preferred_components}. "
            f"Actual columns: {df.columns.tolist()}"
        )

    return available_components


def clean_nlp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw NLP metrics dataset.
    """
    df = df.copy()

    ticker_col = find_ticker_column(df)

    if ticker_col != "ticker":
        df = df.rename(columns={ticker_col: "ticker"})

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Remove duplicate ticker rows if any exist.
    # Keeps the first available company-level result.
    df = df.drop_duplicates(subset=["ticker"]).copy()

    nlp_components = select_available_nlp_components(df)

    # Remove rows where all selected NLP components are empty/unusable.
    mask_all_empty = np.logical_and.reduce(
        [
            df[col].apply(is_empty_for_removal).to_numpy()
            for col in nlp_components
        ]
    )

    df = df[~mask_all_empty].copy()

    # Convert components to numeric.
    # Non-numeric entries become NaN.
    for col in nlp_components:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where all selected components became NaN after numeric conversion.
    df = df.dropna(subset=nlp_components, how="all").copy()

    return df.reset_index(drop=True)


def score_nlp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize NLP components and calculate nlp_stagnation_score.

    Higher nlp_stagnation_score = stronger stagnation signal.
    """
    df = df.copy()

    nlp_components = select_available_nlp_components(df)

    # Fill component-level missing values with the median.
    # This avoids losing companies that have most, but not all, NLP signals.
    for col in nlp_components:
        median_value = df[col].median()

        if pd.isna(median_value):
            median_value = 0

        df[col] = df[col].fillna(median_value)

    scaled_cols = [f"{col}_scaled" for col in nlp_components]

    scaler = MinMaxScaler()
    df[scaled_cols] = scaler.fit_transform(df[nlp_components])

    # Important direction logic:
    # For innovation_decay_slope and strategic_decay_slope, more negative may mean stronger decay.
    # To keep this project simple and transparent, we assume the raw notebook already calculated
    # these variables in a direction where higher = more stagnation.
    # If later we confirm the opposite, we can flip those columns before scaling.
    df["nlp_stagnation_score"] = df[scaled_cols].mean(axis=1)

    df["nlp_component_count"] = len(nlp_components)
    df["nlp_missing_component_count"] = df[nlp_components].isna().sum(axis=1)

    df = df.sort_values("nlp_stagnation_score", ascending=False).reset_index(drop=True)

    return df


def save_nlp_metrics() -> pd.DataFrame:
    """
    Load, clean, score, and save NLP stagnation metrics.
    """
    raw_df = load_raw_nlp_metrics()
    clean_df = clean_nlp_metrics(raw_df)
    scored_df = score_nlp_metrics(clean_df)

    NLP_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(NLP_METRICS_FILE, index=False)

    print(f"Loaded raw NLP rows: {len(raw_df)}")
    print(f"Rows after NLP cleaning: {len(clean_df)}")
    print(f"Saved scored NLP rows: {len(scored_df)}")
    print(f"Saved file: {NLP_METRICS_FILE}")

    print("\nNLP components used:")
    for col in select_available_nlp_components(scored_df):
        print(f"- {col}")

    print("\nTop 10 NLP stagnation candidates:")
    output_cols = ["ticker", "nlp_stagnation_score"] + select_available_nlp_components(scored_df)
    print(scored_df[output_cols].head(10).to_string(index=False))

    return scored_df


if __name__ == "__main__":
    save_nlp_metrics()