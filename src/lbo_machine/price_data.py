import time
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import (
    FINAL_RANKING_FILE,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
)


VALIDATION_PRICES_FILE = RAW_DATA_DIR / "validation_prices.csv"
PRICE_CACHE_FILE = INTERIM_DATA_DIR / "yahoo_chart_price_cache.csv"

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

BENCHMARK_TICKER = "SPY"
TOP_N = 25

START_DATE = "2025-01-01"
END_DATE = "2026-05-04"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 LBO-Machine academic project",
    "Accept": "application/json,text/plain,*/*",
}


def date_to_unix(date_string: str) -> int:
    """
    Convert YYYY-MM-DD date string to Unix timestamp.
    """
    dt = datetime.strptime(date_string, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def normalize_yahoo_ticker(ticker: str) -> str:
    """
    Convert ticker to Yahoo format.
    BRK.B -> BRK-B
    """
    return str(ticker).strip().upper().replace(".", "-")


def load_current_validation_tickers(top_n: int = TOP_N) -> list[str]:
    """
    Load current top model candidates and add SPY benchmark.
    """
    if not FINAL_RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FINAL_RANKING_FILE}. "
            "Run python -m src.lbo_machine.ranking first."
        )

    ranking = pd.read_csv(FINAL_RANKING_FILE)
    ranking["ticker"] = ranking["ticker"].astype(str).str.strip().str.upper()

    tickers = ranking.sort_values("rank").head(top_n)["ticker"].tolist()
    tickers = sorted(list(set(tickers + [BENCHMARK_TICKER])))

    return tickers


def load_existing_price_cache() -> pd.DataFrame:
    """
    Load already fetched prices, if available.
    """
    if PRICE_CACHE_FILE.exists():
        cache = pd.read_csv(PRICE_CACHE_FILE)
        cache["date"] = pd.to_datetime(cache["date"], errors="coerce")
        cache["ticker"] = cache["ticker"].astype(str).str.strip().str.upper()
        cache["close"] = pd.to_numeric(cache["close"], errors="coerce")
        cache = cache.dropna(subset=["date", "ticker", "close"])
        return cache

    return pd.DataFrame(columns=["date", "ticker", "close"])


def save_price_cache(cache: pd.DataFrame) -> None:
    """
    Save price cache.
    """
    cache = cache.copy()
    cache["date"] = pd.to_datetime(cache["date"], errors="coerce")
    cache["ticker"] = cache["ticker"].astype(str).str.strip().str.upper()
    cache["close"] = pd.to_numeric(cache["close"], errors="coerce")

    cache = cache.dropna(subset=["date", "ticker", "close"])
    cache = cache.drop_duplicates(subset=["date", "ticker"], keep="last")
    cache = cache.sort_values(["ticker", "date"])

    PRICE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(PRICE_CACHE_FILE, index=False)


def filter_validation_window(
    prices: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    Keep only validation date range.
    """
    if prices.empty:
        return prices

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()

    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_cached_tickers(cache: pd.DataFrame) -> set[str]:
    """
    Return tickers already available in cache for the validation window.
    """
    if cache.empty:
        return set()

    window_cache = filter_validation_window(cache)

    available = set()

    for ticker, group in window_cache.groupby("ticker"):
        if len(group) >= 2:
            available.add(ticker)

    return available


def fetch_yahoo_chart_prices(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    Fetch daily prices from Yahoo public chart API.

    This does not use yfinance and does not require an API key.
    """
    yahoo_ticker = normalize_yahoo_ticker(ticker)

    period1 = date_to_unix(start_date)
    period2 = date_to_unix(end_date) + 24 * 60 * 60

    url = YAHOO_CHART_URL.format(ticker=yahoo_ticker)

    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    response = requests.get(
        url,
        params=params,
        headers=REQUEST_HEADERS,
        timeout=60,
    )

    if response.status_code == 429:
        print(f"Yahoo chart API rate-limited on {ticker}.")
        return pd.DataFrame()

    response.raise_for_status()

    data = response.json()

    chart = data.get("chart", {})
    error = chart.get("error")

    if error:
        print(f"Yahoo chart API error for {ticker}: {error}")
        return pd.DataFrame()

    results = chart.get("result", [])

    if not results:
        print(f"No Yahoo chart result for {ticker}.")
        return pd.DataFrame()

    result = results[0]

    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})

    quote_list = indicators.get("quote", [])
    adjclose_list = indicators.get("adjclose", [])

    if not timestamps or not quote_list:
        print(f"Missing price data for {ticker}.")
        return pd.DataFrame()

    quote = quote_list[0]

    close_values = None

    # Prefer adjusted close if Yahoo provides it.
    if adjclose_list:
        adjclose = adjclose_list[0]
        close_values = adjclose.get("adjclose")

    if close_values is None:
        close_values = quote.get("close")

    if close_values is None:
        print(f"No close values for {ticker}.")
        return pd.DataFrame()

    rows = []

    for ts, close in zip(timestamps, close_values):
        if close is None:
            continue

        date = datetime.fromtimestamp(ts, tz=timezone.utc).date()

        rows.append(
            {
                "date": pd.to_datetime(date),
                "ticker": ticker.upper(),
                "close": close,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.sort_values("date")

    return df


def save_validation_prices_from_yahoo_chart(
    top_n: int = TOP_N,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Generate data/raw/validation_prices.csv for the current top candidates.

    Uses direct Yahoo chart API with caching.
    """
    tickers = load_current_validation_tickers(top_n=top_n)

    print(f"Fetching Yahoo chart prices for {len(tickers)} tickers.")
    print(f"Tickers: {tickers}")
    print(f"Validation window: {START_DATE} to {END_DATE}")

    cache = load_existing_price_cache()
    cached_tickers = get_cached_tickers(cache)

    remaining = [ticker for ticker in tickers if ticker not in cached_tickers]

    print(f"Already cached tickers: {len(cached_tickers)}")
    print(f"Remaining tickers: {len(remaining)}")

    all_prices = [cache] if not cache.empty else []

    for i, ticker in enumerate(remaining, start=1):
        print(f"\n[{i}/{len(remaining)}] Fetching {ticker}...")

        ticker_prices = fetch_yahoo_chart_prices(
            ticker=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        if ticker_prices.empty:
            print(f"No usable prices for {ticker}.")
            print("Stopping early to preserve API access. Rerun later if needed.")
            break

        ticker_prices = filter_validation_window(ticker_prices)

        if ticker_prices.empty:
            print(f"No rows inside validation window for {ticker}.")
        else:
            print(f"Rows kept for {ticker}: {len(ticker_prices)}")
            all_prices.append(ticker_prices)

            updated_cache = pd.concat(all_prices, ignore_index=True)
            save_price_cache(updated_cache)

        time.sleep(sleep_seconds)

    if all_prices:
        final_prices = pd.concat(all_prices, ignore_index=True)
    else:
        final_prices = pd.DataFrame(columns=["date", "ticker", "close"])

    final_prices = filter_validation_window(final_prices)
    final_prices = final_prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    final_prices = final_prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if final_prices.empty:
        raise ValueError(
            "No prices were fetched. Yahoo chart API may be blocked or rate-limited."
        )

    VALIDATION_PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_prices.to_csv(VALIDATION_PRICES_FILE, index=False)

    print(f"\nSaved validation prices: {VALIDATION_PRICES_FILE}")
    print(f"Rows saved: {len(final_prices)}")
    print(f"Tickers saved: {final_prices['ticker'].nunique()}")
    print(f"Tickers available: {sorted(final_prices['ticker'].unique().tolist())}")

    return final_prices


if __name__ == "__main__":
    save_validation_prices_from_yahoo_chart(
        top_n=TOP_N,
        sleep_seconds=2.0,
    )