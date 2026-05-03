import pandas as pd

from .config import (
    NLP_METRICS_FILE,
    FINANCIAL_METRICS_FILE,
    FINAL_RANKING_FILE,
    MERGED_DATASET_FILE,
    TOP_CANDIDATES_FILE,
    SECTOR_SUMMARY_FILE,
    RED_FLAG_SUMMARY_FILE,
    NLP_WEIGHT,
    FINANCIAL_WEIGHT,
    TOP_N,
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load scored NLP metrics and cleaned financial metrics.
    """
    if not NLP_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Missing NLP metrics file: {NLP_METRICS_FILE}. "
            "Run python -m src.lbo_machine.nlp_metrics first."
        )

    if not FINANCIAL_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Missing financial metrics file: {FINANCIAL_METRICS_FILE}. "
            "Run python -m src.lbo_machine.financial_data first."
        )

    nlp_df = pd.read_csv(NLP_METRICS_FILE)
    fin_df = pd.read_csv(FINANCIAL_METRICS_FILE)

    nlp_df["ticker"] = nlp_df["ticker"].astype(str).str.strip().str.upper()
    fin_df["ticker"] = fin_df["ticker"].astype(str).str.strip().str.upper()

    return nlp_df, fin_df


def merge_nlp_and_financials(
    nlp_df: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge NLP stagnation metrics with financial feasibility metrics.

    Inner merge is used because the final ranking should only include companies
    that have both textual evidence and financial evidence.
    """
    merged = nlp_df.merge(
        fin_df,
        on="ticker",
        how="inner",
        suffixes=("_nlp", "_fin"),
    )

    return merged


def calculate_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate final LBO candidate score.

    Higher final_score = stronger candidate under this screening framework.
    """
    df = df.copy()

    required_cols = ["nlp_stagnation_score", "financial_feasibility_score"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for final score: {col}")

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    df["final_score"] = (
        NLP_WEIGHT * df["nlp_stagnation_score"]
        + FINANCIAL_WEIGHT * df["financial_feasibility_score"]
    )

    df["rank"] = df["final_score"].rank(ascending=False, method="first").astype(int)

    df = df.sort_values("rank").reset_index(drop=True)

    return df


def add_candidate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretation labels to make the output easier to understand.
    """
    df = df.copy()

    def classify_candidate(row):
        final_score = row.get("final_score", 0)
        red_flag = bool(row.get("financial_red_flag", False))

        if final_score >= 0.80 and not red_flag:
            return "Strong candidate"
        if final_score >= 0.70 and not red_flag:
            return "Review candidate"
        if final_score >= 0.70 and red_flag:
            return "High NLP signal, financial caution"
        if red_flag:
            return "Financial caution"
        return "Lower priority"

    df["candidate_label"] = df.apply(classify_candidate, axis=1)

    return df


def create_summary_tables(df: pd.DataFrame) -> None:
    """
    Save top candidates and summary tables.
    """
    TOP_CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECTOR_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    RED_FLAG_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    top_cols = [
        "rank",
        "ticker",
        "sector",
        "final_score",
        "nlp_stagnation_score",
        "financial_feasibility_score",
        "lbo_financial_score",
        "lbo_status",
        "lbo_rating",
        "financial_red_flag",
        "candidate_label",
    ]

    top_cols = [col for col in top_cols if col in df.columns]

    df[top_cols].head(TOP_N).to_csv(TOP_CANDIDATES_FILE, index=False)

    if "sector" in df.columns:
        sector_summary = (
            df.groupby("sector", dropna=False)
            .agg(
                company_count=("ticker", "count"),
                avg_final_score=("final_score", "mean"),
                avg_nlp_score=("nlp_stagnation_score", "mean"),
                avg_financial_score=("financial_feasibility_score", "mean"),
                red_flag_rate=("financial_red_flag", "mean"),
            )
            .reset_index()
            .sort_values("avg_final_score", ascending=False)
        )

        sector_summary.to_csv(SECTOR_SUMMARY_FILE, index=False)

    if "financial_red_flag" in df.columns:
        red_flag_summary = (
            df.groupby("financial_red_flag", dropna=False)
            .agg(
                company_count=("ticker", "count"),
                avg_final_score=("final_score", "mean"),
                avg_nlp_score=("nlp_stagnation_score", "mean"),
                avg_financial_score=("financial_feasibility_score", "mean"),
            )
            .reset_index()
        )

        red_flag_summary.to_csv(RED_FLAG_SUMMARY_FILE, index=False)


def save_final_ranking() -> pd.DataFrame:
    """
    Run the full ranking step:
    load inputs, merge, score, label, save outputs.
    """
    nlp_df, fin_df = load_inputs()

    merged = merge_nlp_and_financials(nlp_df, fin_df)
    scored = calculate_final_score(merged)
    labeled = add_candidate_labels(scored)

    FINAL_RANKING_FILE.parent.mkdir(parents=True, exist_ok=True)

    labeled.to_csv(MERGED_DATASET_FILE, index=False)

    output_cols = [
        "rank",
        "ticker",
        "sector",
        "final_score",
        "nlp_stagnation_score",
        "financial_feasibility_score",
        "lbo_financial_score",
        "financial_red_flag",
        "candidate_label",
        "lbo_status",
        "lbo_rating",
        "innovation_decay_slope",
        "strategic_decay_slope",
        "strategic_decay",
        "topic_rigidity",
        "sentiment_growth_correlation",
        "sentiment_growth_misalignment",
        "fcf",
        "fcf_margin",
        "fcf_to_debt",
        "debt_to_ebitda",
        "interest_coverage",
        "capex_to_ebitda",
    ]

    output_cols = [col for col in output_cols if col in labeled.columns]

    final_ranking = labeled[output_cols].copy()
    final_ranking.to_csv(FINAL_RANKING_FILE, index=False)

    create_summary_tables(labeled)

    print(f"NLP rows: {len(nlp_df)}")
    print(f"Financial rows: {len(fin_df)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Final ranked rows: {len(final_ranking)}")
    print(f"Saved final ranking: {FINAL_RANKING_FILE}")
    print(f"Saved merged dataset: {MERGED_DATASET_FILE}")
    print(f"Saved top candidates: {TOP_CANDIDATES_FILE}")

    print("\nTop 15 final candidates:")
    display_cols = [
        "rank",
        "ticker",
        "sector",
        "final_score",
        "nlp_stagnation_score",
        "financial_feasibility_score",
        "financial_red_flag",
        "candidate_label",
    ]
    display_cols = [col for col in display_cols if col in final_ranking.columns]

    print(final_ranking[display_cols].head(15).to_string(index=False))

    return final_ranking


if __name__ == "__main__":
    save_final_ranking()