from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import NLP_METRICS_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFINED_STAGNATION_FILE = PROJECT_ROOT / "data" / "interim" / "refined_stagnation_scores.csv"

REFINED_COMPONENTS = [
    "innovation_decay_score",
    "strategic_decay_score",
    "topic_rigidity_score",
    "management_confidence_score",
]

MIN_REFINED_COMPONENTS = 3


def min_max_scale(series: pd.Series) -> pd.Series:
    """
    Convert a numeric series to a 0-1 score.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    min_v = s.min(skipna=True)
    max_v = s.max(skipna=True)

    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        return pd.Series(np.nan, index=s.index)

    return (s - min_v) / (max_v - min_v)


def load_refined_stagnation_scores() -> pd.DataFrame:
    """
    Load Heyman's refined stagnation output and convert it into the format
    expected by the main LBO Machine pipeline.

    Important quality rule:
        A company must have at least 3 of 4 refined NLP components.
        This prevents companies from ranking highly based on only one signal.
    """
    if not REFINED_STAGNATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing refined stagnation file: {REFINED_STAGNATION_FILE}"
        )

    df = pd.read_csv(REFINED_STAGNATION_FILE)

    if "ticker" not in df.columns:
        raise ValueError(
            f"refined_stagnation_scores.csv must contain ticker. "
            f"Actual columns: {df.columns.tolist()}"
        )

    if "final_stagnation_score" not in df.columns:
        raise ValueError(
            "refined_stagnation_scores.csv must contain final_stagnation_score."
        )

    missing_components = [col for col in REFINED_COMPONENTS if col not in df.columns]
    if missing_components:
        raise ValueError(f"Missing refined NLP component columns: {missing_components}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"] != ""].copy()

    for col in REFINED_COMPONENTS + ["final_stagnation_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df["refined_usable_component_count"] = df[REFINED_COMPONENTS].notna().sum(axis=1)
    df["refined_signal_quality"] = np.where(
        df["refined_usable_component_count"] >= MIN_REFINED_COMPONENTS,
        "usable",
        "insufficient_components",
    )

    before_filter = len(df)
    df = df[df["refined_usable_component_count"] >= MIN_REFINED_COMPONENTS].copy()
    after_filter = len(df)

    if df.empty:
        raise ValueError(
            "No companies passed the refined NLP quality filter. "
            "Check refined_stagnation_scores.csv."
        )

    # Keep one row per ticker. If duplicates exist, keep the highest refined score.
    df = (
        df.sort_values("final_stagnation_score", ascending=False)
        .drop_duplicates(subset=["ticker"], keep="first")
        .reset_index(drop=True)
    )

    # Pipeline compatibility:
    # final_stagnation_score is z-score style, so convert to 0-1 for the 70/30 model.
    df["nlp_stagnation_score"] = min_max_scale(df["final_stagnation_score"])

    rename_map = {
        "innovation_decay_score": "innovation_decay_refined",
        "strategic_decay_score": "strategic_decay_refined",
        "topic_rigidity_score": "topic_rigidity_refined",
        "management_confidence_score": "management_confidence_refined",
    }

    for old_col, new_col in rename_map.items():
        df[new_col] = df[old_col]

    df["nlp_source"] = "refined_phrase_based_stagnation_model"
    df["minimum_required_refined_components"] = MIN_REFINED_COMPONENTS
    df["refined_rows_before_quality_filter"] = before_filter
    df["refined_rows_after_quality_filter"] = after_filter

    output_cols = [
        "ticker",
        "nlp_stagnation_score",
        "final_stagnation_score",

        "innovation_decay_refined",
        "strategic_decay_refined",
        "topic_rigidity_refined",
        "management_confidence_refined",

        "innovation_decay_zscore",
        "strategic_decay_zscore",
        "topic_rigidity_zscore",
        "management_confidence_zscore",
        "management_confidence_zscore_aligned",

        "refined_usable_component_count",
        "minimum_required_refined_components",
        "refined_signal_quality",
        "nlp_source",
        "refined_rows_before_quality_filter",
        "refined_rows_after_quality_filter",
    ]

    output_cols = [col for col in output_cols if col in df.columns]

    df = df[output_cols].copy()
    df = df.dropna(subset=["nlp_stagnation_score"]).copy()
    df = df.sort_values("nlp_stagnation_score", ascending=False).reset_index(drop=True)

    return df


def save_nlp_metrics() -> pd.DataFrame:
    """
    Save the refined NLP stagnation score in the format expected by the rest
    of the LBO Machine pipeline.
    """
    scored_df = load_refined_stagnation_scores()

    NLP_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(NLP_METRICS_FILE, index=False)

    print("Using refined phrase-based NLP stagnation scores.")
    print(f"Loaded refined rows before quality filter: {scored_df['refined_rows_before_quality_filter'].iloc[0]}")
    print(f"Rows after quality filter: {len(scored_df)}")
    print(f"Minimum required refined components: {MIN_REFINED_COMPONENTS}")
    print(f"Saved NLP metrics: {NLP_METRICS_FILE}")

    print("\nNLP components used:")
    for col in [
        "innovation_decay_refined",
        "strategic_decay_refined",
        "topic_rigidity_refined",
        "management_confidence_refined",
    ]:
        if col in scored_df.columns:
            print(f"- {col}")

    print("\nTop 10 refined NLP stagnation candidates:")
    display_cols = [
        "ticker",
        "nlp_stagnation_score",
        "final_stagnation_score",
        "refined_usable_component_count",
        "innovation_decay_refined",
        "strategic_decay_refined",
        "topic_rigidity_refined",
        "management_confidence_refined",
    ]
    display_cols = [col for col in display_cols if col in scored_df.columns]

    print(scored_df[display_cols].head(10).to_string(index=False))

    return scored_df


if __name__ == "__main__":
    save_nlp_metrics()
