from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from .config import (
    FINAL_RANKING_FILE,
    TABLES_DIR,
    FIGURES_DIR,
)


BACKTEST_COMPANY_RETURNS_FILE = TABLES_DIR / "backtest_company_returns.csv"
BACKTEST_SUMMARY_FILE = TABLES_DIR / "backtest_summary.csv"
BACKTEST_BUCKET_CHART = FIGURES_DIR / "backtest_score_buckets.png"
BACKTEST_SCATTER_CHART = FIGURES_DIR / "backtest_score_vs_returns.png"


def load_final_ranking() -> pd.DataFrame:
    """
    Load final LBO ranking.
    """
    if not FINAL_RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Missing final ranking file: {FINAL_RANKING_FILE}. "
            "Run python -m src.lbo_machine.ranking first."
        )

    df = pd.read_csv(FINAL_RANKING_FILE)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    return df


def download_price_history(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.

    Uses group_by='ticker' to handle multiple tickers.
    """
    print(f"Downloading prices for {len(tickers)} tickers...")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    return data


def extract_price_series(price_data: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Extract a single ticker's adjusted close series from yfinance output.
    Handles both single-index and multi-index yfinance outputs.
    """
    if price_data.empty:
        return pd.Series(dtype=float)

    # Multi-ticker output
    if isinstance(price_data.columns, pd.MultiIndex):
        if ticker in price_data.columns.get_level_values(0):
            ticker_data = price_data[ticker]

            if "Close" in ticker_data.columns:
                return ticker_data["Close"].dropna()

        return pd.Series(dtype=float)

    # Single ticker output
    if "Close" in price_data.columns:
        return price_data["Close"].dropna()

    return pd.Series(dtype=float)


def get_price_on_or_after(series: pd.Series, target_date: pd.Timestamp):
    """
    Get the first available price on or after target_date.
    """
    if series.empty:
        return None

    filtered = series[series.index >= target_date]

    if filtered.empty:
        return None

    return filtered.iloc[0]


def calculate_forward_returns_for_ticker(
    ticker: str,
    price_data: pd.DataFrame,
    ranking_date: pd.Timestamp,
) -> dict:
    """
    Calculate 3-month, 6-month, and 12-month forward returns for one ticker.
    """
    series = extract_price_series(price_data, ticker)

    start_price = get_price_on_or_after(series, ranking_date)

    horizons = {
        "return_3m": ranking_date + pd.DateOffset(months=3),
        "return_6m": ranking_date + pd.DateOffset(months=6),
        "return_12m": ranking_date + pd.DateOffset(months=12),
    }

    result = {
        "ticker": ticker,
        "start_price": start_price,
        "return_3m": None,
        "return_6m": None,
        "return_12m": None,
    }

    if start_price is None or start_price == 0:
        return result

    for col, target_date in horizons.items():
        end_price = get_price_on_or_after(series, target_date)

        if end_price is not None:
            result[col] = (end_price / start_price) - 1

    return result


def assign_score_buckets(df: pd.DataFrame, bucket_count: int = 4) -> pd.DataFrame:
    """
    Assign companies into score buckets.

    Bucket 1 = highest score group.
    """
    df = df.copy()

    df["score_bucket"] = pd.qcut(
        df["final_score"],
        q=bucket_count,
        labels=False,
        duplicates="drop",
    )

    # qcut gives 0 to lowest bucket. Flip so 1 = best.
    max_bucket = df["score_bucket"].max()
    df["score_bucket"] = max_bucket - df["score_bucket"] + 1

    df["score_bucket"] = df["score_bucket"].astype(int)

    return df


def run_forward_return_validation(
    ranking_date: str = "2025-01-01",
    max_companies: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate final ranking by checking forward stock returns after ranking_date.

    This is a forward-return diagnostic, not a perfect historical backtest.
    """
    ranking_df = load_final_ranking()

    if max_companies is not None:
        ranking_df = ranking_df.head(max_companies).copy()

    tickers = ranking_df["ticker"].dropna().unique().tolist()

    ranking_date_ts = pd.to_datetime(ranking_date)

    # Need enough price data to cover 12 months forward.
    end_date_ts = ranking_date_ts + pd.DateOffset(months=13)
    end_date = end_date_ts.strftime("%Y-%m-%d")

    price_data = download_price_history(
        tickers=tickers,
        start_date=ranking_date,
        end_date=end_date,
    )

    return_rows = []

    for ticker in tickers:
        returns = calculate_forward_returns_for_ticker(
            ticker=ticker,
            price_data=price_data,
            ranking_date=ranking_date_ts,
        )
        return_rows.append(returns)

    returns_df = pd.DataFrame(return_rows)

    validated = ranking_df.merge(returns_df, on="ticker", how="left")
    validated = assign_score_buckets(validated)

    summary = (
        validated.groupby("score_bucket", dropna=False)
        .agg(
            company_count=("ticker", "count"),
            avg_final_score=("final_score", "mean"),
            avg_nlp_score=("nlp_stagnation_score", "mean"),
            avg_financial_score=("financial_feasibility_score", "mean"),
            avg_return_3m=("return_3m", "mean"),
            avg_return_6m=("return_6m", "mean"),
            avg_return_12m=("return_12m", "mean"),
        )
        .reset_index()
        .sort_values("score_bucket")
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    validated.to_csv(BACKTEST_COMPANY_RETURNS_FILE, index=False)
    summary.to_csv(BACKTEST_SUMMARY_FILE, index=False)

    print(f"Saved company return validation: {BACKTEST_COMPANY_RETURNS_FILE}")
    print(f"Saved backtest summary: {BACKTEST_SUMMARY_FILE}")

    print("\nBacktest / validation summary:")
    print(summary.to_string(index=False))

    return validated, summary


def plot_backtest_score_buckets(summary: pd.DataFrame) -> None:
    """
    Plot average forward returns by score bucket.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = summary.copy()
    plot_df["bucket_label"] = "Bucket " + plot_df["score_bucket"].astype(str)

    plt.figure(figsize=(9, 6))

    x = range(len(plot_df))
    width = 0.25

    plt.bar(
        [i - width for i in x],
        plot_df["avg_return_3m"],
        width=width,
        label="3M",
    )
    plt.bar(
        x,
        plot_df["avg_return_6m"],
        width=width,
        label="6M",
    )
    plt.bar(
        [i + width for i in x],
        plot_df["avg_return_12m"],
        width=width,
        label="12M",
    )

    plt.xticks(x, plot_df["bucket_label"])
    plt.xlabel("Score Bucket; Bucket 1 = Highest Final Score")
    plt.ylabel("Average Forward Return")
    plt.title("Forward Returns by LBO Candidate Score Bucket")
    plt.legend()
    plt.tight_layout()

    plt.savefig(BACKTEST_BUCKET_CHART, dpi=300)
    plt.close()

    print(f"Saved {BACKTEST_BUCKET_CHART}")


def plot_score_vs_returns(validated: pd.DataFrame) -> None:
    """
    Scatter plot of final score vs 12-month forward return.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = validated.dropna(subset=["final_score", "return_12m"]).copy()

    if plot_df.empty:
        print("Skipping score vs returns chart. No valid 12M returns.")
        return

    plt.figure(figsize=(9, 6))
    plt.scatter(plot_df["final_score"], plot_df["return_12m"], alpha=0.75)

    for _, row in plot_df.head(10).iterrows():
        plt.annotate(
            row["ticker"],
            (row["final_score"], row["return_12m"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("Final LBO Candidate Score")
    plt.ylabel("12-Month Forward Return")
    plt.title("Final Score vs. 12-Month Forward Return")
    plt.tight_layout()

    plt.savefig(BACKTEST_SCATTER_CHART, dpi=300)
    plt.close()

    print(f"Saved {BACKTEST_SCATTER_CHART}")


def save_backtest_outputs(
    ranking_date: str = "2025-01-01",
    max_companies: int | None = None,
) -> None:
    """
    Run validation and save tables/charts.
    """
    validated, summary = run_forward_return_validation(
        ranking_date=ranking_date,
        max_companies=max_companies,
    )

    plot_backtest_score_buckets(summary)
    plot_score_vs_returns(validated)

    print("Backtest / validation outputs generated.")


if __name__ == "__main__":
    # Use the current model as if ranking date was Jan 1, 2025.
    # This is a diagnostic validation, not a fully time-aligned historical backtest.
    save_backtest_outputs(ranking_date="2025-01-01", max_companies=None)