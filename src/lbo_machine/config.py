from pathlib import Path

# Project root: LBO-Machine/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Input files currently in the repo
RAW_STAGNATION_FILE = RAW_DATA_DIR / "sp500_stagnation_metric_df.csv"
RAW_FINANCIAL_FILE = RAW_DATA_DIR / "lbo_finscreening_reordered.xlsx"

# Pipeline outputs
SP500_UNIVERSE_FILE = INTERIM_DATA_DIR / "sp500_universe.csv"
FINANCIAL_METRICS_FILE = INTERIM_DATA_DIR / "financial_metrics_yfinance.csv"
NLP_METRICS_FILE = INTERIM_DATA_DIR / "nlp_stagnation_metrics.csv"

FINAL_RANKING_FILE = PROCESSED_DATA_DIR / "final_lbo_ranking.csv"
MERGED_DATASET_FILE = PROCESSED_DATA_DIR / "merged_lbo_dataset.csv"
TOP_CANDIDATES_FILE = TABLES_DIR / "top_25_candidates.csv"
SECTOR_SUMMARY_FILE = TABLES_DIR / "sector_summary.csv"
RED_FLAG_SUMMARY_FILE = TABLES_DIR / "red_flag_summary.csv"

# Model weights
NLP_WEIGHT = 0.70
FINANCIAL_WEIGHT = 0.30

# Financial red flag thresholds
MIN_INTEREST_COVERAGE = 2.0
MAX_DEBT_TO_EBITDA = 4.0
MIN_FCF_MARGIN = 0.0

# Sectors excluded from traditional LBO screening
EXCLUDED_SECTORS = [
    "Financial Services",
    "Banks",
    "Insurance",
    "Capital Markets",
    "Consumer Finance",
]

# NLP columns expected from the current stagnation file
NLP_COMPONENTS = [
    "innovation_decay_slope",
    "topic_rigidity",
    "sentiment_growth_correlation",
]

# Optional columns, depending on available data
OPTIONAL_NLP_COMPONENTS = [
    "innovation_freq",
    "strategic_decay",
    "strategic_decay_slope",
]

# Plot settings
TOP_N = 25