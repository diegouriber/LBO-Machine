import pandas as pd
import requests

from .config import SP500_UNIVERSE_FILE


def fetch_sp500_universe() -> pd.DataFrame:
    """
    Fetch the current S&P 500 company universe from Wikipedia.

    Output columns:
    - ticker
    - company
    - sector
    - sub_industry
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(response.text)
    sp500 = tables[0].copy()

    sp500 = sp500.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )

    sp500["ticker"] = sp500["ticker"].astype(str).str.strip().str.upper()
    sp500["company"] = sp500["company"].astype(str).str.strip()
    sp500["sector"] = sp500["sector"].astype(str).str.strip()
    sp500["sub_industry"] = sp500["sub_industry"].astype(str).str.strip()

    # yfinance uses BRK-B instead of BRK.B
    sp500["ticker"] = sp500["ticker"].str.replace(".", "-", regex=False)

    sp500 = sp500[["ticker", "company", "sector", "sub_industry"]]
    sp500 = sp500.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    return sp500


def save_sp500_universe() -> pd.DataFrame:
    """
    Fetch and save the S&P 500 universe to data/interim.
    """
    sp500 = fetch_sp500_universe()

    SP500_UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sp500.to_csv(SP500_UNIVERSE_FILE, index=False)

    print(f"Saved {len(sp500)} S&P 500 companies to {SP500_UNIVERSE_FILE}")

    return sp500


if __name__ == "__main__":
    save_sp500_universe()