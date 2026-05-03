import pandas as pd
import matplotlib.pyplot as plt

from .config import (
    FINAL_RANKING_FILE,
    SECTOR_SUMMARY_FILE,
    RED_FLAG_SUMMARY_FILE,
    FIGURES_DIR,
    TOP_N,
)


def load_ranking() -> pd.DataFrame:
    """
    Load final LBO ranking output.
    """
    if not FINAL_RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Missing final ranking file: {FINAL_RANKING_FILE}. "
            "Run python -m src.lbo_machine.ranking first."
        )

    df = pd.read_csv(FINAL_RANKING_FILE)
    return df


def plot_top_candidates(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Bar chart of top candidates by final score.
    """
    top = df.head(top_n).copy()
    top = top.sort_values("final_score", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top["ticker"], top["final_score"])
    plt.xlabel("Final LBO Candidate Score")
    plt.ylabel("Ticker")
    plt.title(f"Top {top_n} LBO Candidates by Final Score")
    plt.tight_layout()

    output_path = FIGURES_DIR / "top_candidates.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_score_distribution(df: pd.DataFrame) -> None:
    """
    Histogram of final scores.
    """
    plt.figure(figsize=(9, 6))
    plt.hist(df["final_score"].dropna(), bins=15)
    plt.xlabel("Final LBO Candidate Score")
    plt.ylabel("Number of Companies")
    plt.title("Distribution of Final LBO Candidate Scores")
    plt.tight_layout()

    output_path = FIGURES_DIR / "score_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_nlp_vs_finance(df: pd.DataFrame) -> None:
    """
    Scatter plot comparing NLP stagnation score and financial feasibility score.
    """
    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["financial_feasibility_score"],
        df["nlp_stagnation_score"],
        alpha=0.75,
    )

    for _, row in df.head(10).iterrows():
        plt.annotate(
            row["ticker"],
            (row["financial_feasibility_score"], row["nlp_stagnation_score"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("Financial Feasibility Score")
    plt.ylabel("NLP Stagnation Score")
    plt.title("NLP Stagnation vs. Financial Feasibility")
    plt.tight_layout()

    output_path = FIGURES_DIR / "nlp_vs_finance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_sector_summary() -> None:
    """
    Bar chart of average final score by sector.
    """
    if not SECTOR_SUMMARY_FILE.exists():
        print(f"Skipping sector chart. Missing {SECTOR_SUMMARY_FILE}")
        return

    sector_df = pd.read_csv(SECTOR_SUMMARY_FILE)

    if sector_df.empty or "sector" not in sector_df.columns:
        print("Skipping sector chart. Sector summary is empty or invalid.")
        return

    sector_df = sector_df.sort_values("avg_final_score", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(sector_df["sector"], sector_df["avg_final_score"])
    plt.xlabel("Average Final Score")
    plt.ylabel("Sector")
    plt.title("Average LBO Candidate Score by Sector")
    plt.tight_layout()

    output_path = FIGURES_DIR / "sector_summary.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_red_flag_breakdown() -> None:
    """
    Bar chart showing count of companies with and without financial red flags.
    """
    if not RED_FLAG_SUMMARY_FILE.exists():
        print(f"Skipping red flag chart. Missing {RED_FLAG_SUMMARY_FILE}")
        return

    red_df = pd.read_csv(RED_FLAG_SUMMARY_FILE)

    if red_df.empty or "financial_red_flag" not in red_df.columns:
        print("Skipping red flag chart. Red flag summary is empty or invalid.")
        return

    red_df["financial_red_flag"] = red_df["financial_red_flag"].astype(str)

    plt.figure(figsize=(7, 5))
    plt.bar(red_df["financial_red_flag"], red_df["company_count"])
    plt.xlabel("Financial Red Flag")
    plt.ylabel("Number of Companies")
    plt.title("Financial Red Flag Breakdown")
    plt.tight_layout()

    output_path = FIGURES_DIR / "red_flag_breakdown.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_component_correlation(df: pd.DataFrame) -> None:
    """
    Correlation heatmap-like chart for main score components.
    Uses matplotlib only.
    """
    possible_cols = [
        "final_score",
        "nlp_stagnation_score",
        "financial_feasibility_score",
        "lbo_financial_score",
        "fcf_margin",
        "fcf_to_debt",
        "debt_to_ebitda",
        "interest_coverage",
        "topic_rigidity",
        "innovation_decay_slope",
        "sentiment_growth_correlation",
    ]

    cols = [col for col in possible_cols if col in df.columns]

    if len(cols) < 3:
        print("Skipping correlation chart. Not enough numeric columns.")
        return

    corr_df = df[cols].copy()

    for col in cols:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")

    corr = corr_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Between Model Components")
    plt.tight_layout()

    output_path = FIGURES_DIR / "component_correlation.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def save_all_plots() -> None:
    """
    Generate all project charts.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ranking()

    print(f"Loaded final ranking rows: {len(df)}")

    plot_top_candidates(df, top_n=min(15, len(df)))
    plot_score_distribution(df)
    plot_nlp_vs_finance(df)
    plot_sector_summary()
    plot_red_flag_breakdown()
    plot_component_correlation(df)

    print("All figures generated.")


if __name__ == "__main__":
    save_all_plots()