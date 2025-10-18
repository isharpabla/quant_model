#!/usr/bin/env python3
# Quant momentum + filter backtester for stocks/ETFs
# Timeframe: weekly or monthly. Default monthly.
# Data: CSV files (one per ticker) or yfinance (optional).

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf  # optional; only used if ONLINE=True
except Exception:
    yf = None

# --------- Configuration ---------
@dataclass
class Config:
    tickers: List[str]
    start: str = "2012-01-01"
    end: Optional[str] = None  # None = today
    freq: str = "M"            # 'W-FRI' for weekly, 'M' for monthly
    lookback_months: List[int] = None  # momentum windows excluding last 1 month
    top_n: int = 4
    min_1m_ret: float = 0.0    # filter: require last 1-month return >= this
    cash_ticker: Optional[str] = None  # e.g., 'BIL' as cash proxy
    transaction_cost_bps: float = 5.0  # one-way cost per trade in bps
    data_dir: Optional[str] = None     # directory with CSVs (Date, Adj Close)
    ONLINE: bool = False        # if True and yfinance present, pull data online

    def __post_init__(self):
        if self.lookback_months is None:
            self.lookback_months = [6, 12]


# --------- Data loading ---------
def load_prices(cfg: Config) -> pd.DataFrame:
    """
    Return daily Adj Close price DataFrame with columns=cfg.tickers.
    CSV mode: expects files like data_dir/TICKER.csv with columns Date, Adj Close.
    YF mode: uses yfinance download if ONLINE=True.
    """
    if cfg.data_dir:
        frames = []
        for t in cfg.tickers:
            path = os.path.join(cfg.data_dir, f"{t}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing CSV for {t}: {path}")
            df = pd.read_csv(path, parse_dates=["Date"])            
            if "Adj Close" not in df.columns:
                raise ValueError(f"{t} CSV missing 'Adj Close' column")
            df = df[["Date", "Adj Close"]].rename(columns={"Adj Close": t}).set_index("Date")
            frames.append(df)
        prices = pd.concat(frames, axis=1).sort_index()
    elif cfg.ONLINE and yf is not None:
        data = yf.download(cfg.tickers, start=cfg.start, end=cfg.end, auto_adjust=True, progress=False)
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            prices = data["Adj Close"].copy()
        else:
            prices = data.copy()
        prices = prices.dropna(how="all")
    else:
        raise RuntimeError("No data. Provide data_dir with CSVs or set ONLINE=True with yfinance installed.")

    prices = prices.loc[cfg.start:cfg.end].ffill().dropna(how="all")
    return prices


# --------- Signal generation ---------
def momentum_score(prices_m: pd.DataFrame, lookbacks: List[int]) -> pd.DataFrame:
    """Average of L-month returns excluding last 1 month. Uses prices at t-1 and t-(L+1)."""
    score = pd.DataFrame(0.0, index=prices_m.index, columns=prices_m.columns)
    for L in lookbacks:
        ret_ex_last1 = prices_m.shift(1) / prices_m.shift(L + 1) - 1.0
        score = score.add(ret_ex_last1, fill_value=0.0)
    score /= float(len(lookbacks))
    return score


def last_1m_return(prices_m: pd.DataFrame) -> pd.DataFrame:
    r1 = prices_m.pct_change(1)
    return r1


def build_weights(scores: pd.DataFrame, r1m: pd.DataFrame, top_n: int, min_1m_ret: float, cash_col: Optional[str] = None) -> pd.DataFrame:
    """Rank by score, keep top_n with r1m >= min_1m_ret. Equal weights. Leftover to cash if provided."""
    ranks = scores.rank(axis=1, ascending=False, method="first")
    selected = (ranks <= top_n) & (r1m >= min_1m_ret)
    weights = selected.astype(float)
    row_sums = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sums, axis=0).fillna(0.0)

    if cash_col is not None and cash_col not in weights.columns:
        weights[cash_col] = 0.0

    if cash_col is not None:
        leftover = 1.0 - weights.sum(axis=1)
        weights[cash_col] = weights[cash_col].add(leftover, fill_value=0.0)

    return weights


# --------- Backtest ---------
def backtest(prices: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    """Backtest quant strategy. Returns dict of outputs."""
    prices_f = prices.resample(cfg.freq).last().dropna(how="all")
    # drop assets with missing history for fair ranking
    prices_f = prices_f.dropna(axis=1, how="any")

    scores = momentum_score(prices_f, cfg.lookback_months)
    r1m = last_1m_return(prices_f)

    # Build target weights using information available at time t, then apply from t+1
    weights = build_weights(scores, r1m, cfg.top_n, cfg.min_1m_ret, cfg.cash_ticker).shift(1).fillna(0.0)

    # Period returns on signal frequency
    period_rets = prices_f.pct_change().fillna(0.0)
    port_rets_period = (weights * period_rets).sum(axis=1)

    # Daily path
    weights_daily = weights.reindex(prices.index).ffill().fillna(0.0)
    rets_daily = prices.pct_change().fillna(0.0)
    port_rets_daily = (weights_daily * rets_daily).sum(axis=1)

    # Transaction costs on rebalance dates
    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    tc = turnover * (cfg.transaction_cost_bps / 10000.0)
    port_rets_daily.loc[weights.index] = port_rets_daily.loc[weights.index] - tc

    equity = (1.0 + port_rets_daily).cumprod()

    trades = (weights - w_prev).fillna(0.0)

    return {
        "prices": prices,
        "prices_freq": prices_f,
        "scores": scores,
        "weights": weights,
        "weights_daily": weights_daily,
        "turnover": turnover,
        "returns_daily": port_rets_daily,
        "equity": equity,
        "trades": trades,
    }


# --------- Metrics ---------
def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


def annualize_return(daily_rets: pd.Series) -> float:
    mean = daily_rets.mean()
    return float((1 + mean) ** 252 - 1)


def annualize_vol(daily_rets: pd.Series) -> float:
    return float(daily_rets.std() * np.sqrt(252))


def sharpe(daily_rets: pd.Series, rf: float = 0.0) -> float:
    rf_daily = (1 + rf) ** (1/252) - 1
    excess = daily_rets - rf_daily
    vol = annualize_vol(excess)
    return float(np.nan) if vol == 0 else float(annualize_return(excess) / vol)


def summarize(bt: Dict[str, pd.DataFrame], rf: float = 0.0) -> pd.Series:
    eq = bt["equity"].copy()
    rets = bt["returns_daily"].copy()

    stats = pd.Series({
        "Total Return": float(eq.iloc[-1] - 1.0),
        "CAGR": float(eq.iloc[-1] ** (252/len(eq)) - 1) if len(eq) > 0 else np.nan,
        "Max Drawdown": max_drawdown(eq),
        "Ann Return": annualize_return(rets),
        "Ann Vol": annualize_vol(rets),
        "Sharpe": sharpe(rets, rf=rf),
        "Worst Day": float(rets.min()),
        "Win Rate (daily)": float((rets > 0).mean()),
        "Avg Turnover (per rebalance)": float(bt["turnover"].mean()) if "turnover" in bt else np.nan,
    })
    return stats


# --------- Example main ---------
EXAMPLE_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD"]  # edit as needed

def main():
    cfg = Config(
        tickers=EXAMPLE_TICKERS,
        start="2012-01-01",
        end=None,
        freq="M",               # 'M' monthly or 'W-FRI' weekly
        lookback_months=[6, 12],
        top_n=3,
        min_1m_ret=0.0,
        cash_ticker=None,       # set to 'BIL' if you include a cash ETF in data
        transaction_cost_bps=5.0,
        data_dir=None,          # set to a folder with CSVs or leave None for ONLINE
        ONLINE=False            # set True to fetch via yfinance
    )

    prices = load_prices(cfg)
    bt = backtest(prices, cfg)
    stats = summarize(bt, rf=0.0)

    outdir = os.path.abspath("backtest_output")
    os.makedirs(outdir, exist_ok=True)
    bt["equity"].to_frame("equity").to_csv(os.path.join(outdir, "equity_curve.csv"))
    bt["returns_daily"].to_frame("daily_returns").to_csv(os.path.join(outdir, "daily_returns.csv"))
    bt["weights"].to_csv(os.path.join(outdir, "weights_by_period.csv"))
    bt["trades"].to_csv(os.path.join(outdir, "trades_by_period.csv"))
    stats.to_csv(os.path.join(outdir, "summary_metrics.csv"), header=False)

    print("Summary metrics:")
    print(stats.to_string())
    print(f"\nSaved CSVs in: {outdir}")
    print("\nTo switch to weekly: set cfg.freq='W-FRI'.")
    print("To use CapIQ: export Adj Close to CSV per ticker and set cfg.data_dir to that folder.")
    print("CSV format required: columns ['Date','Adj Close'] with daily rows.")

if __name__ == "__main__":
    main()
