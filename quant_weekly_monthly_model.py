#!/usr/bin/env python3
# Quant momentum + filter backtester for stocks/ETFs
# Uses yfinance online data by default (Business Month-End safe version)

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    yf = None


# ---------- Config ----------
@dataclass
class Config:
    tickers: List[str]
    start: str = "2012-01-01"
    end: Optional[str] = None
    freq: str = "BM"                 # 'BM' = business month-end
    lookback_months: List[int] = None
    top_n: int = 3
    min_1m_ret: float = 0.0
    cash_ticker: Optional[str] = None
    transaction_cost_bps: float = 5.0
    data_dir: Optional[str] = None
    ONLINE: bool = True               # online mode

    def __post_init__(self):
        if self.lookback_months is None:
            self.lookback_months = [6, 12]


# ---------- Data ----------
def load_prices(cfg: Config) -> pd.DataFrame:
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
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(1):
                prices = data.xs("Adj Close", axis=1, level=1)
            else:
                prices = data.copy()
        else:
            prices = data.copy()
        prices = prices.dropna(how="all")
    else:
        raise RuntimeError("No data source. Either set ONLINE=True with yfinance installed or provide data_dir with CSVs.")
    prices = prices.loc[cfg.start:cfg.end].ffill().dropna(how="all")
    return prices


# ---------- Signals ----------
def momentum_score(prices_m: pd.DataFrame, lookbacks: List[int]) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=prices_m.index, columns=prices_m.columns)
    for L in lookbacks:
        # exclude the most recent 1 month: P_{t-1} / P_{t-(L+1)} - 1
        ret_ex_last1 = prices_m.shift(1) / prices_m.shift(L + 1) - 1.0
        score = score.add(ret_ex_last1, fill_value=0.0)
    score /= float(len(lookbacks))
    return score


def last_1m_return(prices_m: pd.DataFrame) -> pd.DataFrame:
    return prices_m.pct_change(1)


def build_weights(
    scores: pd.DataFrame,
    r1m: pd.DataFrame,
    top_n: int,
    min_1m_ret: float,
    cash_col: Optional[str] = None
) -> pd.DataFrame:
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


# ---------- Backtest ----------
def backtest(prices: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    # Rebalance prices (end-of-period for chosen frequency)
    prices_f = prices.resample(cfg.freq).last().dropna(how="all")
    prices_f = prices_f.dropna(axis=1, how="any")

    # Signals on the rebalance grid
    scores = momentum_score(prices_f, cfg.lookback_months)
    r1m = last_1m_return(prices_f)

    # Weights live on the rebalance grid; shift to trade next period
    weights = (
        build_weights(scores, r1m, cfg.top_n, cfg.min_1m_ret, cfg.cash_ticker)
        .shift(1)
        .fillna(0.0)
    )

    # Extend weights to DAILY calendar via forward-fill
    weights_daily = weights.reindex(prices.index, method="ffill").fillna(0.0)

    # Daily portfolio returns (scale-safe)
    rets_daily = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port_rets_daily = (weights_daily * rets_daily).sum(axis=1)

    # Safety rails: avoid unrealistic compounding due to bad ticks
    port_rets_daily = port_rets_daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port_rets_daily = port_rets_daily.clip(lower=-0.30, upper=0.30)  # cap daily moves at ±30%

    # Transaction costs on rebalance dates
    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    tc = turnover * (cfg.transaction_cost_bps / 10000.0)

    # Align TC to daily index (no KeyError)
    tc_daily = tc.reindex(port_rets_daily.index).fillna(0.0)
    port_rets_daily = port_rets_daily - tc_daily

    # Equity curve & trades
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


# ---------- Metrics ----------
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
    rf_daily = (1 + rf) ** (1 / 252) - 1
    excess = daily_rets - rf_daily
    vol = annualize_vol(excess)
    return float(np.nan) if vol == 0 else float(annualize_return(excess) / vol)


def summarize(bt: Dict[str, pd.DataFrame], rf: float = 0.0) -> pd.Series:
    eq = bt["equity"].copy()
    rets = bt["returns_daily"].copy()
    return pd.Series({
        "Total Return": float(eq.iloc[-1] - 1.0),
        "CAGR": float(eq.iloc[-1] ** (252 / len(eq)) - 1) if len(eq) > 0 else np.nan,
        "Max Drawdown": max_drawdown(eq),
        "Ann Return": annualize_return(rets),
        "Ann Vol": annualize_vol(rets),
        "Sharpe": sharpe(rets, rf=rf),
        "Worst Day": float(rets.min()),
        "Win Rate (daily)": float((rets > 0).mean()),
        "Avg Turnover (per rebalance)": float(bt["turnover"].mean()),
    })


# ---------- Main ----------
def main():
    cfg = Config(
        tickers=["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD"],
        start="2012-01-01",
        end=None,
        freq="BM",                # Business month-end
        lookback_months=[6, 12],
        top_n=3,
        min_1m_ret=0.0,
        cash_ticker=None,
        transaction_cost_bps=5.0,
        data_dir=None,
        ONLINE=True
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
    print("To switch to weekly: set cfg.freq='W-FRI'.")


if __name__ == "__main__":
    main()
