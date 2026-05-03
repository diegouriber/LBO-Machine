from .sp500_universe import save_sp500_universe
from .financial_data import save_financial_metrics
from .nlp_metrics import save_nlp_metrics
from .ranking import save_final_ranking
from .plots import save_all_plots
from .backtest import save_backtest_outputs


def run_full_pipeline(
    run_universe: bool = True,
    run_financials: bool = True,
    run_nlp: bool = True,
    run_ranking: bool = True,
    run_plots: bool = True,
    run_backtest: bool = False,
) -> None:
    """
    Run the full LBO Machine pipeline.

    Steps:
    1. Fetch S&P 500 universe.
    2. Clean financial data.
    3. Clean and score NLP stagnation metrics.
    4. Merge NLP + financial layers and rank companies.
    5. Generate analytical charts.
    6. Optionally run forward-return validation.

    Note:
    The backtest is turned off by default because yfinance can rate-limit
    price downloads. The backtest can still be run manually with:

        python -m src.lbo_machine.backtest
    """

    print("=" * 80)
    print("LBO MACHINE PIPELINE STARTED")
    print("=" * 80)

    if run_universe:
        print("\n[1/6] Fetching S&P 500 universe...")
        save_sp500_universe()
    else:
        print("\n[1/6] Skipping S&P 500 universe step.")

    if run_financials:
        print("\n[2/6] Cleaning financial data...")
        save_financial_metrics()
    else:
        print("\n[2/6] Skipping financial data step.")

    if run_nlp:
        print("\n[3/6] Cleaning and scoring NLP metrics...")
        save_nlp_metrics()
    else:
        print("\n[3/6] Skipping NLP metrics step.")

    if run_ranking:
        print("\n[4/6] Creating final LBO ranking...")
        save_final_ranking()
    else:
        print("\n[4/6] Skipping ranking step.")

    if run_plots:
        print("\n[5/6] Generating analytical plots...")
        save_all_plots()
    else:
        print("\n[5/6] Skipping plots step.")

    if run_backtest:
        print("\n[6/6] Running forward-return validation...")
        save_backtest_outputs(ranking_date="2025-01-01", max_companies=None)
    else:
        print("\n[6/6] Skipping backtest step.")
        print("To run validation manually: python -m src.lbo_machine.backtest")

    print("\n" + "=" * 80)
    print("LBO MACHINE PIPELINE FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline(
        run_universe=True,
        run_financials=True,
        run_nlp=True,
        run_ranking=True,
        run_plots=True,
        run_backtest=False,
    )