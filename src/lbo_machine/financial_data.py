import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .config import (
    SP500_UNIVERSE_FILE,
    FINANCIAL_METRICS_FILE,
    INTERIM_DATA_DIR,
    TABLES_DIR,
    EXCLUDED_SECTORS,
)


# ============================================================
# SEC CONFIG
# ============================================================

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC asks for a descriptive User-Agent.
# You can replace the email with your own if you want.
SEC_HEADERS = {
    "User-Agent": "LBO-Machine academic project diegouriber@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

SEC_TICKER_HEADERS = {
    "User-Agent": "LBO-Machine academic project diegouriber@example.com",
    "Accept-Encoding": "gzip, deflate",
}

SEC_FINANCIAL_CACHE_FILE = INTERIM_DATA_DIR / "sec_financial_metrics_cache.csv"
SEC_FULL_FINANCIAL_TABLE_FILE = TABLES_DIR / "full_sp500_financial_screen.csv"


# ============================================================
# CONCEPT MAPPING
# ============================================================

CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "DepreciationDepletionAndAmortizationExpense",
    ],
    "interest_expense": [
        "InterestExpenseNonOperating",
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpenditures",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "total_debt": [
        "DebtAndFinanceLeaseObligations",
        "DebtCurrent",
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligations",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "ShortTermDebt",
        "DebtCurrent",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtCurrent",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebt",
    ],
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_ticker(ticker: str) -> str:
    """
    Standardize tickers for matching.
    BRK.B and BRK-B should match.
    """
    return str(ticker).strip().upper().replace(".", "-")


def normalize_ticker_key(ticker: str) -> str:
    """
    More aggressive ticker key for SEC matching.
    Removes punctuation differences.
    """
    return (
        str(ticker)
        .strip()
        .upper()
        .replace(".", "")
        .replace("-", "")
        .replace("/", "")
    )


def safe_float(value: Any) -> float:
    """
    Convert values safely to float.
    """
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def fetch_json(url: str, headers: dict, timeout: int = 30) -> dict:
    """
    Fetch JSON with basic error handling.
    """
    response = requests.get(url, headers=headers, timeout=timeout)

    if response.status_code == 404:
        return {}

    response.raise_for_status()
    return response.json()


# ============================================================
# LOAD S&P 500 UNIVERSE
# ============================================================

def load_sp500_universe() -> pd.DataFrame:
    """
    Load S&P 500 universe created by sp500_universe.py.
    """
    if not SP500_UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SP500_UNIVERSE_FILE}. "
            "Run python -m src.lbo_machine.sp500_universe first."
        )

    universe = pd.read_csv(SP500_UNIVERSE_FILE)

    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["ticker"] = universe["ticker"].str.replace(".", "-", regex=False)

    if "sector" in universe.columns:
        universe["sector"] = universe["sector"].astype(str).str.strip()

    return universe


def exclude_incompatible_sectors(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove sectors that do not fit traditional LBO screening logic.

    Wikipedia S&P 500 sectors use names like "Financials", so we add that
    explicitly in addition to config exclusions.
    """
    universe = universe.copy()

    extra_excluded = [
        "Financials",
        "Financial Services",
        "Banks",
        "Insurance",
        "Capital Markets",
        "Consumer Finance",
    ]

    excluded = set(EXCLUDED_SECTORS + extra_excluded)

    if "sector" in universe.columns:
        universe = universe[~universe["sector"].isin(excluded)].copy()

    return universe.reset_index(drop=True)


# ============================================================
# SEC TICKER / CIK MAP
# ============================================================

def fetch_sec_ticker_cik_map() -> pd.DataFrame:
    """
    Fetch ticker-to-CIK mapping from SEC.
    """
    data = fetch_json(
        SEC_TICKER_CIK_URL,
        headers=SEC_TICKER_HEADERS,
        timeout=30,
    )

    rows = []

    for _, item in data.items():
        ticker = normalize_ticker(item.get("ticker", ""))
        cik = str(item.get("cik_str", "")).zfill(10)
        title = item.get("title", "")

        rows.append(
            {
                "ticker": ticker,
                "ticker_key": normalize_ticker_key(ticker),
                "cik": cik,
                "sec_company_title": title,
            }
        )

    mapping = pd.DataFrame(rows)
    mapping = mapping.drop_duplicates(subset=["ticker_key"]).reset_index(drop=True)

    return mapping


def attach_cik_to_universe(universe: pd.DataFrame, cik_map: pd.DataFrame) -> pd.DataFrame:
    """
    Attach SEC CIK to the S&P 500 universe.
    """
    universe = universe.copy()

    universe["ticker"] = universe["ticker"].apply(normalize_ticker)
    universe["ticker_key"] = universe["ticker"].apply(normalize_ticker_key)

    merged = universe.merge(
        cik_map[["ticker_key", "cik", "sec_company_title"]],
        on="ticker_key",
        how="left",
    )

    return merged


# ============================================================
# SEC COMPANY FACTS PARSING
# ============================================================

def get_us_gaap_facts(company_facts: dict) -> dict:
    """
    Extract us-gaap facts from SEC companyfacts JSON.
    """
    return company_facts.get("facts", {}).get("us-gaap", {})


def extract_latest_annual_fact(
    company_facts: dict,
    possible_concepts: list[str],
    preferred_units: list[str] | None = None,
) -> dict:
    """
    Extract latest annual 10-K value for a list of possible us-gaap concepts.

    Returns:
    {
        "value": float,
        "concept": str,
        "fy": int,
        "filed": str,
        "form": str,
    }
    """
    if preferred_units is None:
        preferred_units = ["USD"]

    us_gaap = get_us_gaap_facts(company_facts)

    best_records = []

    for concept in possible_concepts:
        if concept not in us_gaap:
            continue

        units = us_gaap[concept].get("units", {})

        ordered_units = preferred_units + [
            unit for unit in units.keys() if unit not in preferred_units
        ]

        for unit in ordered_units:
            if unit not in units:
                continue

            entries = units[unit]

            for entry in entries:
                form = str(entry.get("form", ""))

                if form not in ["10-K", "10-K/A"]:
                    continue

                # Prefer annual fiscal-year observations.
                fp = str(entry.get("fp", ""))
                if fp not in ["FY", "CY"]:
                    continue

                value = safe_float(entry.get("val", np.nan))

                if pd.isna(value):
                    continue

                fy = entry.get("fy", np.nan)
                filed = entry.get("filed", "")

                best_records.append(
                    {
                        "value": value,
                        "concept": concept,
                        "unit": unit,
                        "fy": fy,
                        "filed": filed,
                        "form": form,
                    }
                )

    if not best_records:
        return {
            "value": np.nan,
            "concept": "",
            "unit": "",
            "fy": np.nan,
            "filed": "",
            "form": "",
        }

    records = pd.DataFrame(best_records)

    records["fy_numeric"] = pd.to_numeric(records["fy"], errors="coerce")
    records["filed_dt"] = pd.to_datetime(records["filed"], errors="coerce")

    records = records.sort_values(
        ["fy_numeric", "filed_dt"],
        ascending=[False, False],
    )

    best = records.iloc[0].to_dict()

    return {
        "value": best.get("value", np.nan),
        "concept": best.get("concept", ""),
        "unit": best.get("unit", ""),
        "fy": best.get("fy", np.nan),
        "filed": best.get("filed", ""),
        "form": best.get("form", ""),
    }


def extract_metric(company_facts: dict, metric_name: str) -> dict:
    """
    Extract one named metric using CONCEPTS mapping.
    """
    concepts = CONCEPTS.get(metric_name, [])

    return extract_latest_annual_fact(
        company_facts=company_facts,
        possible_concepts=concepts,
        preferred_units=["USD"],
    )


def extract_debt(company_facts: dict) -> dict:
    """
    Extract total debt.

    First tries total debt concepts. If unavailable, sums current and noncurrent debt.
    """
    total_debt_result = extract_metric(company_facts, "total_debt")

    if pd.notna(total_debt_result["value"]):
        return total_debt_result

    short_debt = extract_metric(company_facts, "short_term_debt")
    long_debt = extract_metric(company_facts, "long_term_debt")

    short_value = short_debt["value"]
    long_value = long_debt["value"]

    if pd.notna(short_value) or pd.notna(long_value):
        total_value = 0

        if pd.notna(short_value):
            total_value += short_value

        if pd.notna(long_value):
            total_value += long_value

        return {
            "value": total_value,
            "concept": "short_term_debt + long_term_debt",
            "unit": "USD",
            "fy": max(
                safe_float(short_debt.get("fy", np.nan)),
                safe_float(long_debt.get("fy", np.nan)),
            ),
            "filed": max(
                str(short_debt.get("filed", "")),
                str(long_debt.get("filed", "")),
            ),
            "form": "10-K",
        }

    return total_debt_result


# ============================================================
# COMPANY FINANCIAL METRICS
# ============================================================

def fetch_company_facts(cik: str) -> dict:
    """
    Fetch SEC companyfacts JSON for one CIK.
    """
    url = SEC_COMPANY_FACTS_URL.format(cik=str(cik).zfill(10))

    return fetch_json(
        url,
        headers=SEC_HEADERS,
        timeout=30,
    )


def calculate_company_financials(row: pd.Series) -> dict:
    """
    Fetch SEC facts and calculate LBO financial metrics for one company.
    """
    ticker = normalize_ticker(row.get("ticker", ""))
    cik = row.get("cik", np.nan)

    result = {
        "ticker": ticker,
        "company": row.get("company", np.nan),
        "sector": row.get("sector", np.nan),
        "sub_industry": row.get("sub_industry", np.nan),
        "cik": cik,
        "sec_company_title": row.get("sec_company_title", np.nan),
        "fetch_success": False,
        "error": "",
    }

    if pd.isna(cik) or str(cik).strip() == "":
        result["error"] = "Missing CIK"
        return result

    try:
        company_facts = fetch_company_facts(str(cik).zfill(10))

        revenue_result = extract_metric(company_facts, "revenue")
        gross_profit_result = extract_metric(company_facts, "gross_profit")
        operating_income_result = extract_metric(company_facts, "operating_income")
        da_result = extract_metric(company_facts, "depreciation_amortization")
        interest_expense_result = extract_metric(company_facts, "interest_expense")
        ocf_result = extract_metric(company_facts, "operating_cash_flow")
        capex_result = extract_metric(company_facts, "capex")
        current_assets_result = extract_metric(company_facts, "current_assets")
        current_liabilities_result = extract_metric(company_facts, "current_liabilities")
        cash_result = extract_metric(company_facts, "cash")
        debt_result = extract_debt(company_facts)

        revenue = revenue_result["value"]
        gross_profit = gross_profit_result["value"]
        operating_income = operating_income_result["value"]
        depreciation_amortization = da_result["value"]
        interest_expense = interest_expense_result["value"]
        operating_cash_flow = ocf_result["value"]
        capex = capex_result["value"]
        current_assets = current_assets_result["value"]
        current_liabilities = current_liabilities_result["value"]
        cash = cash_result["value"]
        total_debt = debt_result["value"]

        # SEC capex tag is usually positive cash outflow.
        capex_abs = abs(capex) if pd.notna(capex) else np.nan

        fcf = np.nan
        if pd.notna(operating_cash_flow) and pd.notna(capex_abs):
            fcf = operating_cash_flow - capex_abs

        ebit = operating_income

        ebitda = np.nan
        if pd.notna(operating_income) and pd.notna(depreciation_amortization):
            ebitda = operating_income + depreciation_amortization

        net_debt = np.nan
        if pd.notna(total_debt) and pd.notna(cash):
            net_debt = total_debt - cash

        fcf_margin = np.nan
        if pd.notna(fcf) and pd.notna(revenue) and revenue != 0:
            fcf_margin = fcf / revenue

        fcf_conversion = np.nan
        if pd.notna(fcf) and pd.notna(ebitda) and ebitda != 0:
            fcf_conversion = fcf / ebitda

        fcf_to_debt = np.nan
        if pd.notna(fcf) and pd.notna(total_debt):
            if total_debt == 0:
                fcf_to_debt = np.inf
            else:
                fcf_to_debt = fcf / total_debt

        debt_to_ebitda = np.nan
        if pd.notna(total_debt) and pd.notna(ebitda) and ebitda != 0:
            debt_to_ebitda = total_debt / ebitda

        net_debt_to_ebitda = np.nan
        if pd.notna(net_debt) and pd.notna(ebitda) and ebitda != 0:
            net_debt_to_ebitda = net_debt / ebitda

        interest_coverage = np.nan
        if pd.notna(ebit) and pd.notna(interest_expense) and interest_expense != 0:
            interest_coverage = ebit / abs(interest_expense)

        cash_interest_coverage = np.nan
        if (
            pd.notna(operating_cash_flow)
            and pd.notna(interest_expense)
            and interest_expense != 0
        ):
            cash_interest_coverage = operating_cash_flow / abs(interest_expense)

        current_ratio = np.nan
        if (
            pd.notna(current_assets)
            and pd.notna(current_liabilities)
            and current_liabilities != 0
        ):
            current_ratio = current_assets / current_liabilities

        capex_to_revenue = np.nan
        if pd.notna(capex_abs) and pd.notna(revenue) and revenue != 0:
            capex_to_revenue = capex_abs / revenue

        capex_to_ebitda = np.nan
        if pd.notna(capex_abs) and pd.notna(ebitda) and ebitda != 0:
            capex_to_ebitda = capex_abs / ebitda

        usable_metric_count = sum(
            pd.notna(x)
            for x in [
                revenue,
                gross_profit,
                operating_income,
                depreciation_amortization,
                interest_expense,
                operating_cash_flow,
                capex_abs,
                fcf,
                total_debt,
                cash,
                current_assets,
                current_liabilities,
                fcf_margin,
                fcf_conversion,
                fcf_to_debt,
                debt_to_ebitda,
                interest_coverage,
                current_ratio,
                capex_to_ebitda,
            ]
        )

        result.update(
            {
                "year": revenue_result.get("fy", np.nan),
                "filed": revenue_result.get("filed", ""),
                "revenue": revenue,
                "gross_profit": gross_profit,
                "operating_income": operating_income,
                "depreciation_amortization": depreciation_amortization,
                "ebit": ebit,
                "ebitda": ebitda,
                "interest_expense": interest_expense,
                "operating_cash_flow": operating_cash_flow,
                "capex": capex_abs,
                "fcf": fcf,
                "free_cash_flow": fcf,
                "total_debt": total_debt,
                "cash": cash,
                "net_debt": net_debt,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "fcf_margin": fcf_margin,
                "fcf_conversion": fcf_conversion,
                "fcf_to_debt": fcf_to_debt,
                "debt_to_ebitda": debt_to_ebitda,
                "net_debt_to_ebitda": net_debt_to_ebitda,
                "interest_coverage": interest_coverage,
                "cash_interest_coverage": cash_interest_coverage,
                "current_ratio": current_ratio,
                "capex_to_revenue": capex_to_revenue,
                "capex_to_ebitda": capex_to_ebitda,
                "usable_metric_count": usable_metric_count,
                "revenue_concept": revenue_result.get("concept", ""),
                "debt_concept": debt_result.get("concept", ""),
                "fetch_success": usable_metric_count > 0,
            }
        )

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ============================================================
# SCORING HELPERS
# ============================================================

def score_higher_better(x, bands: list[float]) -> float:
    """
    Original notebook-style scoring:
    for higher-is-better metrics.
    """
    if pd.isna(x):
        return np.nan

    if x == np.inf:
        return 5.0

    for i, bound in enumerate(bands):
        if x < bound:
            return float(i)

    return 5.0


def score_lower_better(x, bands: list[float]) -> float:
    """
    Original notebook-style scoring:
    for lower-is-better metrics.
    """
    if pd.isna(x):
        return np.nan

    if x == -np.inf:
        return 5.0

    for i, bound in enumerate(bands):
        if x <= bound:
            return float(5 - i)

    return 0.0


def add_lbo_financial_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply LBO financial scoring logic to SEC financial metrics.
    """
    df = df.copy()

    df["fcf_score"] = df["fcf"].apply(
        lambda x: score_higher_better(
            x,
            [0, 100_000_000, 500_000_000, 1_000_000_000, 5_000_000_000],
        )
    )

    df["fcf_margin_score"] = df["fcf_margin"].apply(
        lambda x: score_higher_better(
            x,
            [0, 0.05, 0.10, 0.15, 0.20],
        )
    )

    df["fcf_conversion_score"] = df["fcf_conversion"].apply(
        lambda x: score_higher_better(
            x,
            [0, 0.20, 0.40, 0.60, 0.80],
        )
    )

    df["fcf_to_debt_score"] = df["fcf_to_debt"].apply(
        lambda x: score_higher_better(
            x,
            [0, 0.05, 0.10, 0.15, 0.25],
        )
    )

    df["debt_to_ebitda_score"] = df["debt_to_ebitda"].apply(
        lambda x: score_lower_better(
            x,
            [1, 2, 3, 4, 5],
        )
    )

    df["net_debt_to_ebitda_score"] = df["net_debt_to_ebitda"].apply(
        lambda x: score_lower_better(
            x,
            [1, 2, 3, 4, 5],
        )
    )

    df["interest_coverage_score"] = df["interest_coverage"].apply(
        lambda x: score_higher_better(
            x,
            [1, 2, 3, 5, 8],
        )
    )

    df["cash_interest_coverage_score"] = df["cash_interest_coverage"].apply(
        lambda x: score_higher_better(
            x,
            [1, 2, 3, 5, 8],
        )
    )

    df["capex_to_revenue_score"] = df["capex_to_revenue"].apply(
        lambda x: score_lower_better(
            x,
            [0.02, 0.05, 0.10, 0.15, 0.25],
        )
    )

    df["capex_to_ebitda_score"] = df["capex_to_ebitda"].apply(
        lambda x: score_lower_better(
            x,
            [0.05, 0.10, 0.20, 0.35, 0.50],
        )
    )

    df["cash_flow_strength_score"] = df[
        ["fcf_score", "fcf_margin_score", "fcf_conversion_score"]
    ].mean(axis=1)

    df["deleveraging_capacity_score"] = df[
        ["fcf_to_debt_score"]
    ].mean(axis=1)

    df["leverage_score"] = df[
        ["debt_to_ebitda_score", "net_debt_to_ebitda_score"]
    ].mean(axis=1)

    df["coverage_score"] = df[
        ["interest_coverage_score", "cash_interest_coverage_score"]
    ].mean(axis=1)

    df["capex_burden_score"] = df[
        ["capex_to_revenue_score", "capex_to_ebitda_score"]
    ].mean(axis=1)

    df["lbo_financial_score"] = (
        0.30 * df["cash_flow_strength_score"]
        + 0.25 * df["deleveraging_capacity_score"]
        + 0.15 * df["leverage_score"]
        + 0.15 * df["coverage_score"]
        + 0.15 * df["capex_burden_score"]
    ) * 20

    df["financial_feasibility_score"] = df["lbo_financial_score"] / 100

    critical_cols = [
        "fcf",
        "interest_coverage",
        "debt_to_ebitda",
        "current_ratio",
        "capex_to_ebitda",
    ]

    df["missing_critical_data"] = df[critical_cols].isna().any(axis=1)

    df["pass_fcf_filter"] = df["fcf"] > 0
    df["pass_interest_coverage_filter"] = df["interest_coverage"] >= 2.0
    df["pass_debt_to_ebitda_filter"] = df["debt_to_ebitda"] <= 5.0
    df["pass_current_ratio_filter"] = df["current_ratio"] >= 1.0
    df["pass_capex_to_ebitda_filter"] = df["capex_to_ebitda"] <= 0.50

    df["pass_hard_filters"] = (
        (~df["missing_critical_data"])
        & df["pass_fcf_filter"]
        & df["pass_interest_coverage_filter"]
        & df["pass_debt_to_ebitda_filter"]
        & df["pass_current_ratio_filter"]
        & df["pass_capex_to_ebitda_filter"]
    )

    df["financial_red_flag"] = (
        (df["interest_coverage"] < 2.0)
        | (df["debt_to_ebitda"] > 5.0)
        | (df["fcf_margin"] <= 0)
        | (df["fcf"] <= 0)
        | df["missing_critical_data"]
    )

    def assign_rating(row):
        score = row["lbo_financial_score"]

        if row["missing_critical_data"] or pd.isna(score):
            return "Insufficient data"

        if score < 70:
            return "Marginal"

        if row["pass_hard_filters"] and score >= 85:
            return "Ideal"

        if row["pass_hard_filters"] and score >= 70:
            return "Attractive"

        if (not row["pass_hard_filters"]) and score >= 85:
            return "Situational"

        return "Marginal"

    df["lbo_rating"] = df.apply(assign_rating, axis=1)

    df["lbo_status"] = df["pass_hard_filters"].map(
        {
            True: "Pass",
            False: "Fail",
        }
    )

    df = df.sort_values("lbo_financial_score", ascending=False).reset_index(drop=True)
    df.insert(0, "lbo_rank", range(1, len(df) + 1))

    return df


# ============================================================
# CACHE + RUNNER
# ============================================================

def load_existing_cache() -> pd.DataFrame:
    """
    Load SEC financial cache if it exists.
    """
    if SEC_FINANCIAL_CACHE_FILE.exists():
        cache = pd.read_csv(SEC_FINANCIAL_CACHE_FILE)
        cache["ticker"] = cache["ticker"].astype(str).str.strip().str.upper()
        return cache

    return pd.DataFrame()


def save_cache(df: pd.DataFrame) -> None:
    """
    Save SEC financial cache.
    """
    SEC_FINANCIAL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEC_FINANCIAL_CACHE_FILE, index=False)


def fetch_full_sp500_financials(
    sleep_seconds: float = 0.15,
    stop_after_rate_limit: bool = True,
) -> pd.DataFrame:
    """
    Fetch full S&P 500 financial metrics from SEC Company Facts API.

    Uses cache so rerunning continues from previous progress.
    """
    universe = load_sp500_universe()
    universe = exclude_incompatible_sectors(universe)

    print(f"S&P 500 universe after sector exclusions: {len(universe)}")

    cik_map = fetch_sec_ticker_cik_map()
    universe = attach_cik_to_universe(universe, cik_map)

    missing_cik = universe["cik"].isna().sum()
    print(f"Companies missing SEC CIK: {missing_cik}")

    cache = load_existing_cache()

    completed_tickers = set()
    if not cache.empty:
        completed_tickers = set(cache["ticker"].dropna().astype(str).str.upper())

    remaining = universe[~universe["ticker"].isin(completed_tickers)].copy()

    print(f"Already cached: {len(completed_tickers)}")
    print(f"Remaining to fetch: {len(remaining)}")

    rows = [] if cache.empty else cache.to_dict("records")

    for _, row in tqdm(remaining.iterrows(), total=len(remaining)):
        ticker = row["ticker"]

        try:
            result = calculate_company_financials(row)

            rows.append(result)

            cache_df = pd.DataFrame(rows)
            save_cache(cache_df)

            time.sleep(sleep_seconds)

        except requests.exceptions.HTTPError as exc:
            error_text = str(exc)

            if "429" in error_text and stop_after_rate_limit:
                print(f"\nSEC rate limit hit at ticker {ticker}. Cache saved.")
                break

            rows.append(
                {
                    "ticker": ticker,
                    "company": row.get("company", np.nan),
                    "sector": row.get("sector", np.nan),
                    "sub_industry": row.get("sub_industry", np.nan),
                    "cik": row.get("cik", np.nan),
                    "fetch_success": False,
                    "error": error_text,
                }
            )

            save_cache(pd.DataFrame(rows))

        except Exception as exc:
            rows.append(
                {
                    "ticker": ticker,
                    "company": row.get("company", np.nan),
                    "sector": row.get("sector", np.nan),
                    "sub_industry": row.get("sub_industry", np.nan),
                    "cik": row.get("cik", np.nan),
                    "fetch_success": False,
                    "error": str(exc),
                }
            )

            save_cache(pd.DataFrame(rows))

    final_raw = pd.DataFrame(rows)
    final_raw = final_raw.drop_duplicates(subset=["ticker"], keep="last")

    return final_raw


def save_financial_metrics() -> pd.DataFrame:
    """
    Fetch SEC financial facts for the S&P 500, calculate LBO financial scores,
    and save the standardized financial dataset.
    """
    raw_financials = fetch_full_sp500_financials()

    if raw_financials.empty:
        raise ValueError("No financial data available. SEC fetch/cache is empty.")

    scored = add_lbo_financial_scores(raw_financials)

    FINANCIAL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    scored.to_csv(FINANCIAL_METRICS_FILE, index=False)
    scored.to_csv(SEC_FULL_FINANCIAL_TABLE_FILE, index=False)

    print(f"\nSaved SEC financial cache: {SEC_FINANCIAL_CACHE_FILE}")
    print(f"Saved standardized financial metrics: {FINANCIAL_METRICS_FILE}")
    print(f"Saved full financial screen table: {SEC_FULL_FINANCIAL_TABLE_FILE}")

    print(f"\nRows saved: {len(scored)}")
    print(f"Successful fetches: {int(scored['fetch_success'].sum())}")
    print(f"Companies with complete critical data: {int((~scored['missing_critical_data']).sum())}")

    display_cols = [
        "lbo_rank",
        "ticker",
        "sector",
        "year",
        "lbo_financial_score",
        "financial_feasibility_score",
        "lbo_status",
        "lbo_rating",
        "financial_red_flag",
        "usable_metric_count",
    ]

    display_cols = [col for col in display_cols if col in scored.columns]

    print("\nTop 20 financial candidates:")
    print(scored[display_cols].head(20).to_string(index=False))

    return scored


if __name__ == "__main__":
    save_financial_metrics()