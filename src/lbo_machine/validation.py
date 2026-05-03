import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from .config import (
    FINAL_RANKING_FILE,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    TABLES_DIR,
    FIGURES_DIR,
)


# ============================================================
# FILE PATHS
# ============================================================

VALIDATION_EVENTS_FILE = RAW_DATA_DIR / "validation_events.csv"
PRICE_CACHE_FILE = INTERIM_DATA_DIR / "validation_price_cache.csv"

VALIDATION_DASHBOARD_FILE = TABLES_DIR / "top_candidates_validation_dashboard.csv"
VALIDATION_KPI_SUMMARY_FILE = TABLES_DIR / "validation_kpi_summary.csv"
VALIDATION_EVENTS_SCORED_FILE = TABLES_DIR / "validation_events_scored.csv"
VALIDATION_RESEARCH_TEMPLATE_FILE = TABLES_DIR / "validation_research_template.csv"

VALIDATION_SCORE_CHART = FIGURES_DIR / "validation_score_by_candidate.png"
RELATIVE_RETURN_CHART = FIGURES_DIR / "relative_return_vs_spy.png"
VALIDATION_LABEL_BREAKDOWN_CHART = FIGURES_DIR / "validation_label_breakdown.png"
MARKET_KPI_MATRIX_CHART = FIGURES_DIR / "market_kpi_matrix.png"


# ============================================================
# CONFIG
# ============================================================

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2026-05-04"
BENCHMARK_TICKER = "SPY"
TOP_N = 25


# ============================================================
# LOAD CURRENT MODEL OUTPUT
# ============================================================

def load_final_ranking(top_n: int = TOP_N) -> pd.DataFrame:
    """
    Load the latest final LBO ranking and keep the current top N candidates.

    This makes validation dynamic. If the model changes, the validation universe
    updates automatically.
    """
    if not FINAL_RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Missing final ranking file: {FINAL_RANKING_FILE}. "
            "Run python -m src.lbo_machine.ranking first."
        )

    df = pd.read_csv(FINAL_RANKING_FILE)

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values("rank").head(top_n).reset_index(drop=True)

    return df


# ============================================================
# DYNAMIC MANUAL EVENT FILE
# ============================================================

def empty_event_row(ticker: str) -> dict:
    """
    Create a blank manual validation row for one ticker.
    """
    return {
        "ticker": ticker,
        "event_type": "",
        "event_date": "",
        "event_description": "",
        "event_score": 0,
        "source": "",
        "notes": "",
    }


def load_existing_events() -> pd.DataFrame:
    """
    Load existing validation events if the file exists.
    """
    required_cols = [
        "ticker",
        "event_type",
        "event_date",
        "event_description",
        "event_score",
        "source",
        "notes",
    ]

    if not VALIDATION_EVENTS_FILE.exists():
        return pd.DataFrame(columns=required_cols)

    events = pd.read_csv(VALIDATION_EVENTS_FILE)

    for col in required_cols:
        if col not in events.columns:
            events[col] = ""

    events = events[required_cols].copy()
    events["ticker"] = events["ticker"].astype(str).str.strip().str.upper()
    events["event_score"] = pd.to_numeric(events["event_score"], errors="coerce").fillna(0)

    return events


def sync_validation_events_with_current_top_candidates(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sync validation_events.csv with the current top-ranked candidates.

    This does three things:
    1. Keeps existing manual event notes.
    2. Adds new blank rows for new top candidates.
    3. Does not delete old manual rows, because old candidates may still matter historically.

    The dashboard will only use the current top candidates, but the CSV can preserve
    previous research.
    """
    current_tickers = ranking["ticker"].dropna().astype(str).str.upper().tolist()

    existing = load_existing_events()

    if existing.empty:
        synced = pd.DataFrame([empty_event_row(ticker) for ticker in current_tickers])
    else:
        existing_tickers = set(existing["ticker"].dropna().astype(str).str.upper())

        new_rows = [
            empty_event_row(ticker)
            for ticker in current_tickers
            if ticker not in existing_tickers
        ]

        synced = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)

    # Add rank/model fields for easier manual research, but keep event file simple.
    # We save the plain event file, and create a richer research template separately.
    synced["ticker"] = synced["ticker"].astype(str).str.strip().str.upper()
    synced["event_score"] = pd.to_numeric(synced["event_score"], errors="coerce").fillna(0)

    VALIDATION_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    synced.to_csv(VALIDATION_EVENTS_FILE, index=False)

    print(f"Synced manual validation events: {VALIDATION_EVENTS_FILE}")
    print(f"Current top candidates: {len(current_tickers)}")
    print(f"Total event rows preserved: {len(synced)}")

    return synced


def create_research_template(
    ranking: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a richer research template for the current top candidates.

    This file tells the user what to research for each current top candidate.
    """
    event_summary = aggregate_event_scores(events)

    template = ranking.merge(event_summary, on="ticker", how="left")

    fill_cols = ["manual_event_score", "event_count"]
    for col in fill_cols:
        if col in template.columns:
            template[col] = pd.to_numeric(template[col], errors="coerce").fillna(0)

    for col in ["event_types", "event_descriptions", "sources"]:
        if col in template.columns:
            template[col] = template[col].fillna("")

    template["research_questions"] = (
        "Check: acquisition/take-private rumors or deal; activist investor pressure; "
        "major restructuring; layoffs; divestitures; strategic review; major market underperformance."
    )

    output_cols = [
        "rank",
        "ticker",
        "sector",
        "candidate_label",
        "final_score",
        "nlp_stagnation_score",
        "financial_feasibility_score",
        "financial_red_flag",
        "lbo_status",
        "lbo_rating",
        "manual_event_score",
        "event_count",
        "event_types",
        "event_descriptions",
        "sources",
        "research_questions",
    ]

    output_cols = [col for col in output_cols if col in template.columns]
    template = template[output_cols].copy()

    VALIDATION_RESEARCH_TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(VALIDATION_RESEARCH_TEMPLATE_FILE, index=False)

    print(f"Saved dynamic research template: {VALIDATION_RESEARCH_TEMPLATE_FILE}")

    return template


def load_validation_events_for_current_candidates(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load and sync events, then return only rows relevant to current top candidates.
    """
    synced = sync_validation_events_with_current_top_candidates(ranking)

    current_tickers = set(ranking["ticker"].dropna().astype(str).str.upper())
    current_events = synced[synced["ticker"].isin(current_tickers)].copy()

    create_research_template(ranking, synced)

    return current_events


# ============================================================
# PRICE DATA
# ============================================================

def normalize_ticker_for_yfinance(ticker: str) -> str:
    """
    Convert ticker to yfinance format.
    """
    return str(ticker).strip().upper().replace(".", "-")


def load_price_cache() -> pd.DataFrame:
    """
    Load cached price history.
    """
    if not PRICE_CACHE_FILE.exists():
        return pd.DataFrame()

    prices = pd.read_csv(PRICE_CACHE_FILE, index_col=0, parse_dates=True)
    prices.columns = [str(col).strip().upper() for col in prices.columns]

    return prices


def save_price_cache(prices: pd.DataFrame) -> None:
    """
    Save price cache.
    """
    PRICE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICE_CACHE_FILE)


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = 10,
    sleep_seconds: float = 10.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download close prices for candidates plus SPY.

    Uses batching and cache to reduce Yahoo rate-limit problems.
    """
    normalized = [normalize_ticker_for_yfinance(t) for t in tickers]
    normalized = sorted(list(set(normalized + [BENCHMARK_TICKER])))

    if not force_refresh:
        cached = load_price_cache()

        if not cached.empty:
            missing = [ticker for ticker in normalized if ticker not in cached.columns]

            if len(missing) == 0:
                print(f"Using cached validation prices: {PRICE_CACHE_FILE}")
                return cached[normalized]

            print(f"Using partial price cache. Missing tickers: {len(missing)}")
        else:
            cached = pd.DataFrame()
            missing = normalized
    else:
        cached = pd.DataFrame()
        missing = normalized

    all_prices = cached.copy()

    print(f"Downloading validation prices for {len(missing)} tickers.")
    print(f"Date range: {start_date} to {end_date}")

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        print(f"Downloading price batch {i // batch_size + 1}: {batch}")

        try:
            data = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=True,
            )

            closes = pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker]
                        if "Close" in ticker_data.columns:
                            closes[ticker] = ticker_data["Close"]
            else:
                if len(batch) == 1 and "Close" in data.columns:
                    closes[batch[0]] = data["Close"]

            if not closes.empty:
                all_prices = pd.concat([all_prices, closes], axis=1)
                all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
                all_prices = all_prices.sort_index()
                save_price_cache(all_prices)

        except Exception as exc:
            print(f"Price batch failed: {batch}")
            print(f"Error: {exc}")

        time.sleep(sleep_seconds)

    save_price_cache(all_prices)

    return all_prices


# ============================================================
# MARKET KPIS
# ============================================================

def get_price_on_or_after(series: pd.Series, date: pd.Timestamp):
    series = series.dropna()

    if series.empty:
        return np.nan

    filtered = series[series.index >= date]

    if filtered.empty:
        return np.nan

    return filtered.iloc[0]


def get_price_on_or_before(series: pd.Series, date: pd.Timestamp):
    series = series.dropna()

    if series.empty:
        return np.nan

    filtered = series[series.index <= date]

    if filtered.empty:
        return np.nan

    return filtered.iloc[-1]


def calculate_max_drawdown(series: pd.Series) -> float:
    """
    Calculate max drawdown from a price series.
    """
    series = series.dropna()

    if series.empty:
        return np.nan

    running_max = series.cummax()
    drawdown = (series / running_max) - 1

    return drawdown.min()


def calculate_annualized_volatility(series: pd.Series) -> float:
    """
    Calculate annualized volatility from daily returns.
    """
    series = series.dropna()

    if len(series) < 5:
        return np.nan

    returns = series.pct_change().dropna()

    if returns.empty:
        return np.nan

    return returns.std() * np.sqrt(252)


def calculate_return(series: pd.Series, start_date: str, end_date: str) -> float:
    """
    Calculate return between start and end dates.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    start_price = get_price_on_or_after(series, start)
    end_price = get_price_on_or_before(series, end)

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return np.nan

    return (end_price / start_price) - 1


def calculate_market_kpis(
    ranking: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Calculate market-based validation KPIs.
    """
    rows = []

    spy_return = np.nan

    if BENCHMARK_TICKER in prices.columns:
        spy_return = calculate_return(prices[BENCHMARK_TICKER], start_date, end_date)

    for _, row in ranking.iterrows():
        ticker = normalize_ticker_for_yfinance(row["ticker"])

        result = {
            "ticker": row["ticker"],
            "stock_return_since_ranking": np.nan,
            "spy_return_since_ranking": spy_return,
            "relative_return_vs_spy": np.nan,
            "max_drawdown_since_ranking": np.nan,
            "volatility_since_ranking": np.nan,
            "market_pressure_score": 0,
        }

        if ticker not in prices.columns:
            rows.append(result)
            continue

        series = prices[ticker].dropna()

        stock_return = calculate_return(series, start_date, end_date)

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        sliced = series[(series.index >= start_ts) & (series.index <= end_ts)]

        max_drawdown = calculate_max_drawdown(sliced)
        volatility = calculate_annualized_volatility(sliced)

        relative_return = np.nan
        if pd.notna(stock_return) and pd.notna(spy_return):
            relative_return = stock_return - spy_return

        market_pressure_score = 0

        if pd.notna(relative_return) and relative_return <= -0.15:
            market_pressure_score += 1

        if pd.notna(max_drawdown) and max_drawdown <= -0.30:
            market_pressure_score += 1

        if pd.notna(volatility) and volatility >= 0.40:
            market_pressure_score += 1

        result.update(
            {
                "stock_return_since_ranking": stock_return,
                "relative_return_vs_spy": relative_return,
                "max_drawdown_since_ranking": max_drawdown,
                "volatility_since_ranking": volatility,
                "market_pressure_score": market_pressure_score,
            }
        )

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# EVENT VALIDATION
# ============================================================

def aggregate_event_scores(events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate manual event scores by ticker.
    """
    events = events.copy()

    events["event_score"] = pd.to_numeric(events["event_score"], errors="coerce").fillna(0)

    event_summary = (
        events.groupby("ticker", dropna=False)
        .agg(
            manual_event_score=("event_score", "sum"),
            event_count=("event_score", lambda x: (x > 0).sum()),
            event_types=(
                "event_type",
                lambda x: "; ".join(
                    sorted(
                        set(
                            str(v)
                            for v in x
                            if str(v).strip().lower() not in ["", "nan", "none"]
                        )
                    )
                ),
            ),
            event_descriptions=(
                "event_description",
                lambda x: " | ".join(
                    str(v)
                    for v in x
                    if str(v).strip().lower() not in ["", "nan", "none"]
                ),
            ),
            sources=(
                "source",
                lambda x: " | ".join(
                    str(v)
                    for v in x
                    if str(v).strip().lower() not in ["", "nan", "none"]
                ),
            ),
        )
        .reset_index()
    )

    return event_summary


def classify_validation(row: pd.Series) -> str:
    """
    Convert validation score into label.
    """
    score = row.get("validation_score", 0)

    if score >= 5:
        return "Strong validation"
    if score >= 3:
        return "Partial validation"
    if score >= 1:
        return "Weak validation signal"

    return "No validation yet"


def build_validation_dashboard(
    ranking: pd.DataFrame,
    market_kpis: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge ranking, market KPIs, and manual events.
    """
    event_summary = aggregate_event_scores(events)

    dashboard = ranking.merge(market_kpis, on="ticker", how="left")
    dashboard = dashboard.merge(event_summary, on="ticker", how="left")

    fill_zero_cols = [
        "manual_event_score",
        "event_count",
        "market_pressure_score",
    ]

    for col in fill_zero_cols:
        if col in dashboard.columns:
            dashboard[col] = pd.to_numeric(dashboard[col], errors="coerce").fillna(0)

    for col in ["event_types", "event_descriptions", "sources"]:
        if col in dashboard.columns:
            dashboard[col] = dashboard[col].fillna("")

    dashboard["validation_score"] = (
        dashboard["manual_event_score"]
        + dashboard["market_pressure_score"]
    )

    dashboard["validation_label"] = dashboard.apply(classify_validation, axis=1)

    dashboard = dashboard.sort_values(["rank"]).reset_index(drop=True)

    return dashboard, event_summary


def create_validation_kpi_summary(dashboard: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-level KPI summary for model validation.
    """
    summary_rows = [
        {
            "kpi": "top_candidates_evaluated",
            "value": len(dashboard),
            "description": "Number of current top-ranked companies evaluated in validation dashboard.",
        },
        {
            "kpi": "companies_with_manual_events",
            "value": int((dashboard["manual_event_score"] > 0).sum()),
            "description": "Companies with manually verified acquisition, activist, restructuring, or strategic events.",
        },
        {
            "kpi": "companies_with_market_pressure",
            "value": int((dashboard["market_pressure_score"] > 0).sum()),
            "description": "Companies showing underperformance, large drawdown, or high volatility after ranking date.",
        },
        {
            "kpi": "avg_validation_score",
            "value": dashboard["validation_score"].mean(),
            "description": "Average validation score across current top candidates.",
        },
        {
            "kpi": "strong_or_partial_validation_rate",
            "value": dashboard["validation_label"].isin(["Strong validation", "Partial validation"]).mean(),
            "description": "Share of companies with strong or partial validation.",
        },
        {
            "kpi": "avg_relative_return_vs_spy",
            "value": dashboard["relative_return_vs_spy"].mean(),
            "description": "Average stock return relative to SPY after ranking date.",
        },
        {
            "kpi": "avg_max_drawdown",
            "value": dashboard["max_drawdown_since_ranking"].mean(),
            "description": "Average maximum drawdown after ranking date.",
        },
    ]

    return pd.DataFrame(summary_rows)


# ============================================================
# PLOTS
# ============================================================

def plot_validation_score_by_candidate(dashboard: pd.DataFrame) -> None:
    """
    Plot validation score by candidate.
    """
    plot_df = dashboard.head(TOP_N).copy()
    plot_df = plot_df.sort_values("validation_score", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["ticker"], plot_df["validation_score"])
    plt.xlabel("Validation Score")
    plt.ylabel("Ticker")
    plt.title("Validation Score by Current Top Candidate")
    plt.tight_layout()
    plt.savefig(VALIDATION_SCORE_CHART, dpi=300)
    plt.close()

    print(f"Saved {VALIDATION_SCORE_CHART}")


def plot_relative_return_vs_spy(dashboard: pd.DataFrame) -> None:
    """
    Plot relative returns vs SPY.
    """
    plot_df = dashboard.dropna(subset=["relative_return_vs_spy"]).copy()

    if plot_df.empty:
        print("Skipping relative return chart. No valid price data.")
        return

    plot_df = plot_df.head(TOP_N).sort_values("relative_return_vs_spy", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["ticker"], plot_df["relative_return_vs_spy"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Relative Return vs SPY")
    plt.ylabel("Ticker")
    plt.title("Current Top Candidates: Relative Return vs SPY")
    plt.tight_layout()
    plt.savefig(RELATIVE_RETURN_CHART, dpi=300)
    plt.close()

    print(f"Saved {RELATIVE_RETURN_CHART}")


def plot_validation_label_breakdown(dashboard: pd.DataFrame) -> None:
    """
    Plot count by validation label.
    """
    counts = dashboard["validation_label"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Validation Label")
    plt.ylabel("Company Count")
    plt.title("Validation Label Breakdown")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(VALIDATION_LABEL_BREAKDOWN_CHART, dpi=300)
    plt.close()

    print(f"Saved {VALIDATION_LABEL_BREAKDOWN_CHART}")


def plot_market_kpi_matrix(dashboard: pd.DataFrame) -> None:
    """
    Plot heatmap-style matrix of market KPIs.
    """
    cols = [
        "stock_return_since_ranking",
        "relative_return_vs_spy",
        "max_drawdown_since_ranking",
        "volatility_since_ranking",
        "market_pressure_score",
        "manual_event_score",
        "validation_score",
    ]

    cols = [col for col in cols if col in dashboard.columns]

    plot_df = dashboard.head(TOP_N).set_index("ticker")[cols].copy()

    for col in cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    norm_df = plot_df.copy()

    for col in cols:
        col_min = norm_df[col].min()
        col_max = norm_df[col].max()

        if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0

    plt.figure(figsize=(11, 8))
    plt.imshow(norm_df.fillna(0), aspect="auto")
    plt.colorbar(label="Normalized KPI Value")
    plt.xticks(range(len(norm_df.columns)), norm_df.columns, rotation=90)
    plt.yticks(range(len(norm_df.index)), norm_df.index)
    plt.title("Market and Event Validation KPI Matrix")
    plt.tight_layout()
    plt.savefig(MARKET_KPI_MATRIX_CHART, dpi=300)
    plt.close()

    print(f"Saved {MARKET_KPI_MATRIX_CHART}")


# ============================================================
# MAIN RUNNER
# ============================================================

def run_validation_dashboard(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    top_n: int = TOP_N,
    force_price_refresh: bool = False,
) -> pd.DataFrame:
    """
    Run the dynamic validation dashboard.

    The candidate list is pulled from the latest final_lbo_ranking.csv.
    So as the model evolves, the validation universe evolves automatically.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ranking = load_final_ranking(top_n=top_n)
    tickers = ranking["ticker"].tolist()

    events = load_validation_events_for_current_candidates(ranking)

    prices = download_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        batch_size=10,
        sleep_seconds=10.0,
        force_refresh=force_price_refresh,
    )

    market_kpis = calculate_market_kpis(
        ranking=ranking,
        prices=prices,
        start_date=start_date,
        end_date=end_date,
    )

    dashboard, event_summary = build_validation_dashboard(
        ranking=ranking,
        market_kpis=market_kpis,
        events=events,
    )

    kpi_summary = create_validation_kpi_summary(dashboard)

    dashboard.to_csv(VALIDATION_DASHBOARD_FILE, index=False)
    kpi_summary.to_csv(VALIDATION_KPI_SUMMARY_FILE, index=False)
    event_summary.to_csv(VALIDATION_EVENTS_SCORED_FILE, index=False)

    print(f"Saved validation dashboard: {VALIDATION_DASHBOARD_FILE}")
    print(f"Saved KPI summary: {VALIDATION_KPI_SUMMARY_FILE}")
    print(f"Saved scored events: {VALIDATION_EVENTS_SCORED_FILE}")

    print("\nCurrent top candidates validation dashboard:")
    display_cols = [
        "rank",
        "ticker",
        "sector",
        "final_score",
        "nlp_stagnation_score",
        "financial_feasibility_score",
        "stock_return_since_ranking",
        "relative_return_vs_spy",
        "max_drawdown_since_ranking",
        "market_pressure_score",
        "manual_event_score",
        "validation_score",
        "validation_label",
    ]
    display_cols = [col for col in display_cols if col in dashboard.columns]
    print(dashboard[display_cols].to_string(index=False))

    print("\nValidation KPI summary:")
    print(kpi_summary.to_string(index=False))

    plot_validation_score_by_candidate(dashboard)
    plot_relative_return_vs_spy(dashboard)
    plot_validation_label_breakdown(dashboard)
    plot_market_kpi_matrix(dashboard)

    print("\nDynamic validation dashboard complete.")

    return dashboard


if __name__ == "__main__":
    run_validation_dashboard(
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        top_n=TOP_N,
        force_price_refresh=False,
    )