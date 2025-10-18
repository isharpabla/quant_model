Quant Model Backtester (Weekly/Monthly)

What this does

Downloads or loads daily Adj Close for a universe of tickers.
Resamples to rebalance frequency (BM = Business Month-End by default; try W-FRI for weekly).
Computes momentum score as the average of [6, 12]-month returns excluding the most recent month.
Applies a 1-month return filter (min_1m_ret, default 0.0).
Buys top-N equal-weight assets; shifts weights by one rebalance to avoid look-ahead.
Cash fallback (BIL by default): if nothing qualifies, leftover weight goes to cash.
Transaction costs applied via turnover (bps).
Scale-safe compounding (caps extreme daily moves and sanitizes NaNs/infs).
Outputs equity curve, daily returns, weights, trades, summary metrics, and equity vs SPY plot.

Requirements

Python 3.10+
Install deps:
pip install pandas numpy yfinance matplotlib

How to run

Open quant_weekly_monthly_model.py.
Edit the Config in main():
tickers=[...] (see Indian stocks note below)
freq="BM" (monthly) or freq="W-FRI" (weekly)
min_1m_ret=0.0 (try 0.02 to require positive 1-month momentum)
cash_ticker="BIL" (set to None to disable cash)
Choose one data source:
ONLINE=True for yfinance or
data_dir="/path/to/csvs" for local CSVs (files must be TICKER.csv with columns Date, Adj Close)

Run:
python quant_weekly_monthly_model.py

Outputs (saved next to the script):
backtest_output/
  equity_curve.csv
  daily_returns.csv
  weights_by_period.csv
  trades_by_period.csv
  summary_metrics.csv
  equity_vs_benchmark.png
