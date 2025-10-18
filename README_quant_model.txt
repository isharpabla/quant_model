# Quant Model Backtester (Weekly/Monthly)

## What this does
- Ranks assets by momentum over [6, 12] months, excluding the most recent 1 month.
- Filters out assets with negative last 1-month return (configurable).
- Buys top N equal-weight. Rebalances monthly by default.
- Applies transaction costs via turnover.
- Outputs equity curve, returns, weights, trades, and summary metrics.

## How to run
1. Install Python 3.10+
2. `pip install pandas numpy yfinance`
3. Place daily CSVs in a folder with files named TICKER.csv, each with columns: Date, Adj Close
4. Edit Config in main():
   - tickers
   - freq='M' or 'W-FRI'
   - data_dir='/path/to/csvs' or ONLINE=True for yfinance
5. Run:
   ```bash
   python quant_weekly_monthly_model.py
   ```

## Switching to CapIQ
- Export daily adjusted close for each ticker to CSV.
- Point data_dir to the export folder.
- Ensure each CSV has Date and Adj Close columns.

## Next steps
- Add inverse-volatility sizing.
- Add volatility targeting.
- Add walk-forward validation and out-of-sample tests.
