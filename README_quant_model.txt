What the model is doing

Build universe (NSE only)
Pulls a broad NSE list, fetches market caps, and splits into 3 size buckets by quantiles: large / mid / small.
Target mix: 5 large + 5 mid + 5 small (15 total).
Get prices & set rebalance clock
Downloads daily adjusted prices (yfinance).
Resamples to your chosen rebalance frequency (you have W-FRI → weekly bars; if you switch to BM, they become monthly bars).
Compute raw momentum (skip latest period)
For each stock and each rebalance date, compute returns over the last 1 and 3 periods (period = week with W-FRI, month with BM), excluding the most recent period to avoid look-ahead.
Average those two → raw momentum score.
Professional refinements (applied historically, every date)
Volatility adjust: compute 60-day daily volatility; divide raw momentum by same-date vol → risk-adjusted momentum.
Cross-sectional z-scores per date: standardize both:
z_voladj = z-score of risk-adjusted momentum
z_raw = z-score of raw momentum
Hybrid signal: final score = 0.7 * z_voladj + 0.3 * z_raw
(balances stability with some punch from fast movers).

Rank & select inside each cap bucket
Rank by the final score within each bucket.
Apply a simple gate r1m >= min_1m_ret (yours is 0.0, so no extra filter).
Pick up to 5 per bucket. If a bucket can’t fill (data gaps, etc.), the leftover stays in cash (ticker BIL).

Weights, costs, performance
Equal-weight within each bucket, then equal-weight across buckets.
Apply 10 bps transaction cost on rebalance days.
Build equity curve; compare vs NIFTYBEES; save CSVs and a benchmark plot.

How the shortlist is produced

After the backtest computes the new week’s weights, we take the latest rebalance row of weights.
Drop cash (BIL).
Any ticker with weight > 0 is included.
Save to backtest_output/shortlist_latest_15.csv with two columns: Ticker and its Bucket (large/mid/small).
That printed list in the console under [SHORTLIST @ last rebalance] is the same set.

Quick notes you might care about

With W-FRI, your “1 and 3” lookbacks are 1-week & 3-weeks. Switch to BM if you truly want 1- & 3-month momentum.
The hybrid (70% vol-adjusted + 30% raw, both z-scored) is why the list is more stable and avoids “lottery” small caps, but can still catch strong new movers.
If a bucket has too few valid names, you’ll see <15 tickers; the remainder is kept as cash by design (risk-aware).
