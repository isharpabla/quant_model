#!/usr/bin/env python3
# Robust quant momentum backtester for Indian stocks (weekly/monthly)
# Includes RSI/volatility/liquidity filters, safe fallbacks, and benchmark plot.

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
    cash_ticker: Optional[str] = "BIL"
    transaction_cost_bps: float = 5.0
    data_dir: Optional[str] = None
    ONLINE: bool = True

    def __post_init__(self):
        if self.lookback_months is None:
            self.lookback_months = [6, 12]


# ---------- Data ----------
def _download(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(1):
            data = data.xs("Adj Close", axis=1, level=1)
    return data


def load_prices(cfg: Config) -> pd.DataFrame:
    """
    Robust loader for NSE (.NS) tickers:
    - Fetch each ticker with yf.Ticker(...).history(auto_adjust=True)
    - Force DatetimeIndex
    - Log what loaded vs failed
    - Return only tickers that actually loaded
    """
    want = list(cfg.tickers)
    if cfg.cash_ticker and cfg.cash_ticker not in want:
        want.append(cfg.cash_ticker)

    frames = []
    failed = []

    # CSV mode
    if cfg.data_dir:
        for t in want:
            path = os.path.join(cfg.data_dir, f"{t}.csv")
            if not os.path.exists(path):
                failed.append((t, "csv_missing"))
                continue
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                if "Adj Close" not in df.columns:
                    failed.append((t, "no_adj_close_col"))
                    continue
                s = df.set_index("Date")["Adj Close"].rename(t)
                frames.append(s)
            except Exception as e:
                failed.append((t, f"csv_error:{e}"))

    # ONLINE mode — per-ticker fetch (more reliable for .NS)
    elif cfg.ONLINE and yf is not None:
        for t in want:
            try:
                tk = yf.Ticker(t)
                h = tk.history(start=cfg.start, end=cfg.end, auto_adjust=True)
                if isinstance(h, pd.DataFrame) and not h.empty:
                    # prefer Close (already adjusted when auto_adjust=True)
                    col = "Close" if "Close" in h.columns else ("Adj Close" if "Adj Close" in h.columns else None)
                    if col is None:
                        failed.append((t, "no_price_col"))
                        continue
                    s = h[col].copy()
                    # ensure datetime index
                    if not isinstance(s.index, pd.DatetimeIndex):
                        s.index = pd.to_datetime(s.index, errors="coerce")
                    s = s.sort_index().dropna()
                    s = s.rename(t)
                    frames.append(s)
                else:
                    failed.append((t, "empty_history"))
            except Exception as e:
                failed.append((t, f"hist_error:{e}"))
    else:
        raise RuntimeError("No data source. Set ONLINE=True with yfinance installed, or provide data_dir with CSVs.")

    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    if prices.empty:
        print("[ERROR] No price series loaded. Check tickers or data source.")
        if failed:
            print("[DETAIL] Failures:", failed)
        return prices

    # Force DatetimeIndex + clean
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.sort_index()
    prices = prices.loc[cfg.start:cfg.end].ffill().dropna(how="all")

    loaded = list(prices.columns)
    missing = [t for t in want if t not in loaded]
    print(f"[LOAD] Loaded {len(loaded)} tickers. Missing: {len(missing)}")
    if missing:
        print("[LOAD] Missing tickers:", ", ".join(missing))
    if failed:
        brief = ", ".join([f"{t}({reason})" for t, reason in failed])
        print("[LOAD] Failure reasons:", brief)

    return prices

# ---------- Signals ----------
def momentum_score(prices_m: pd.DataFrame, lookbacks: List[int]) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=prices_m.index, columns=prices_m.columns)
    for L in lookbacks:
        ret_ex_last1 = prices_m.shift(1) / prices_m.shift(L + 1) - 1.0
        score = score.add(ret_ex_last1, fill_value=0.0)
    score /= float(len(lookbacks))
    return score


def last_1m_return(prices_m: pd.DataFrame) -> pd.DataFrame:
    return prices_m.pct_change(1)


def build_weights(scores: pd.DataFrame, r1m: pd.DataFrame, top_n: int,
                  min_1m_ret: float, cash_col: Optional[str]) -> pd.DataFrame:
    ranks = scores.rank(axis=1, ascending=False, method="first")
    selected = (ranks <= top_n) & (r1m >= min_1m_ret)
    weights = selected.astype(float)
    row_sums = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sums, axis=0)

    if cash_col is not None and cash_col not in weights.columns:
        weights[cash_col] = 0.0

    weights = weights.fillna(0.0)
    if cash_col is not None:
        leftover = 1.0 - weights.sum(axis=1)
        weights[cash_col] = weights[cash_col].add(leftover, fill_value=0.0)
    return weights


# ---------- Backtest ----------

def backtest(prices: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    # Ensure a proper DatetimeIndex for resampling
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.sort_index()
    prices_f = prices.resample(cfg.freq).last().dropna(how="all")
    prices_f = prices_f.dropna(axis=1, how="all")

    scores = momentum_score(prices_f, cfg.lookback_months)
    r1m = last_1m_return(prices_f)

    weights = build_weights(scores, r1m, cfg.top_n, cfg.min_1m_ret, cfg.cash_ticker).shift(1)
    weights = weights.fillna(0.0)

    weights_daily = weights.reindex(prices.index, method="ffill").fillna(0.0)

    rets_daily = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port_rets_daily = (weights_daily * rets_daily).sum(axis=1)
    port_rets_daily = port_rets_daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port_rets_daily = port_rets_daily.clip(lower=-0.30, upper=0.30)

    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    tc = turnover * (cfg.transaction_cost_bps / 10000.0)

    tc_daily = tc.reindex(port_rets_daily.index).fillna(0.0)
    port_rets_daily = port_rets_daily - tc_daily

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


# ---------- Plot ----------
def plot_equity_vs_spy(bt: Dict[str, pd.DataFrame], outdir: str, spy_ticker: str = "SPY"):
    import matplotlib.pyplot as plt

def compare_to_nifty(bt: Dict[str, pd.DataFrame], outdir: str, benchmark_ticker: str = "NIFTYBEES.NS"):
    import matplotlib.pyplot as plt  # 👈 add this line

    """
    Print strategy vs NIFTYBEES cumulative performance and relative out/under-performance.
    If NIFTYBEES isn't in bt['prices'], we try to fetch it (ONLINE mode required).
    """
    prices = bt.get("prices")
    eq = bt.get("equity")
    if prices is None or prices.empty or eq is None or eq.empty:
        print("[WARN] Missing prices or equity; cannot compare to benchmark.")
        return

    # Ensure we have the benchmark series
    if benchmark_ticker not in prices.columns:
        try:
            if yf is None:
                print(f"[INFO] {benchmark_ticker} not in prices and yfinance unavailable; skipping compare.")
                return
            bench = yf.Ticker(benchmark_ticker).history(
                start=str(prices.index.min().date()),
                end=str(prices.index.max().date()),
                auto_adjust=True
            )
            if bench is None or bench.empty:
                print(f"[INFO] Could not fetch {benchmark_ticker}; skipping compare.")
                return
            col = "Close" if "Close" in bench.columns else ("Adj Close" if "Adj Close" in bench.columns else None)
            if col is None:
                print(f"[INFO] No usable price column for {benchmark_ticker}; skipping compare.")
                return
            prices = prices.copy()
            prices[benchmark_ticker] = bench[col]
            prices = prices.sort_index().ffill()
        except Exception as e:
            print(f"[INFO] Benchmark fetch failed: {e}; skipping compare.")
            return

    # Build cumulative curves (normalized to 1 at start)
    bench_curve = (prices[benchmark_ticker].pct_change().fillna(0.0) + 1.0).cumprod()
    bench_curve = bench_curve / bench_curve.iloc[0]
    strat_curve = eq / eq.iloc[0]

    # Align indexes
    common_idx = strat_curve.index.intersection(bench_curve.index)
    strat_curve = strat_curve.reindex(common_idx).ffill()
    bench_curve = bench_curve.reindex(common_idx).ffill()

    # Compute relative performance
    rel = (strat_curve / bench_curve) - 1.0
    strat_ret = float(strat_curve.iloc[-1] - 1.0)
    bench_ret = float(bench_curve.iloc[-1] - 1.0)
    rel_ret = float(rel.iloc[-1])

    print("\n[Relative Performance vs NIFTYBEES]")
    print(f"Strategy cumulative return: {strat_ret*100:.2f}%")
    print(f"NIFTYBEES cumulative return: {bench_ret*100:.2f}%")
    print(f"Strategy vs NIFTYBEES (final): {rel_ret*100:.2f}%")

    eq = bt.get("equity")
    prices = bt.get("prices")

    if prices is None or prices.empty or len(prices.columns) == 0:
        print("[WARN] No price data available to plot. Skipping equity plot.")
        return

    if benchmark_ticker in prices.columns:
        bench = (prices[spy_ticker].pct_change().fillna(0.0) + 1.0).cumprod()
        label_bench = spy_ticker
    else:
        first = prices.columns[0]
        bench = (prices[first].pct_change().fillna(0.0) + 1.0).cumprod()
        label_bench = first

    if eq is None or eq.empty:
        print("[WARN] No equity series available to plot.")
        return

    ax = eq.plot(figsize=(10, 5), label="Strategy")
    bench.reindex(eq.index).plot(ax=ax, label=label_bench)
    ax.set_title("Equity Curve vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalized)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    png_path = os.path.join(outdir, "equity_vs_benchmark.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved plot: {png_path}")


# ---------- Main ----------
def main():
    import numpy as np
    import pandas as pd

    base_universe = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ASIANPAINT.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS",
        "SUNPHARMA.NS", "MARUTI.NS", "ULTRACEMCO.NS", "BHARTIARTL.NS", "AXISBANK.NS",
        "TITAN.NS", "ITC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS"
    ]
    benchmark = "NIFTYBEES.NS"

    cfg = Config(
        tickers=base_universe,
        start="2022-01-01",
        end=None,
        freq="W-FRI",
        lookback_months=[1, 3],
        top_n=10,
        min_1m_ret=0.0,
        cash_ticker="BIL",
        transaction_cost_bps=10.0,
        data_dir=None,
        ONLINE=True,
    )

    prices = load_prices(cfg)

    # Add this guard with correct indentation
    if prices is None or prices.empty:
        print("[FATAL] No prices loaded; aborting run.")
        return

    # ---------- Realism filters ----------
    lookback_days = 60
    window = prices.tail(lookback_days)

    def rsi14(series: pd.Series, period: int = 14) -> float:
        diff = series.diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if rsi.notna().any() else np.nan

    vol_map, miss_map, rsi_map = {}, {}, {}
    for t in base_universe:
        if t not in window.columns:
            continue
        s = window[t]
        miss_map[t] = float(s.isna().mean())
        rets = s.pct_change().dropna()
        vol_map[t] = float(rets.std()) if len(rets) > 5 else np.inf
        rsi_map[t] = rsi14(s.dropna())

    candidates = [t for t in base_universe if t in window.columns and miss_map.get(t, 1.0) <= 0.10]

    if candidates:
        vols = np.array([vol_map[t] for t in candidates if np.isfinite(vol_map.get(t, np.inf))])
        if vols.size > 0:
            vol_cut = float(np.quantile(vols, 0.90))
            candidates = [t for t in candidates if vol_map.get(t, np.inf) <= vol_cut]

    def apply_rsi(cands, lo, hi):
        return [t for t in cands if np.isfinite(rsi_map.get(t, np.nan)) and lo <= rsi_map[t] <= hi]

    filtered = apply_rsi(candidates, 50.0, 70.0)
    if len(filtered) == 0:
        filtered = apply_rsi(candidates, 45.0, 75.0)

    final_universe = filtered if len(filtered) > 0 else [t for t in base_universe if t in prices.columns]
    if len(final_universe) == 0:
        print("[WARN] No tickers passed filters. Using base universe.")
        final_universe = [t for t in base_universe if t in prices.columns]

    top_n_effective = min(cfg.top_n, max(1, len(final_universe)))
    if top_n_effective != cfg.top_n:
        print(f"[INFO] Reducing top_n from {cfg.top_n} to {top_n_effective} (universe size {len(final_universe)}).")
        cfg.top_n = top_n_effective

    print("\n[UNIVERSE] Using tickers after realism filters:")
    print(", ".join(final_universe))

    cfg.tickers = final_universe

    bt = backtest(prices[final_universe], cfg)
    stats = summarize(bt, rf=0.0)

    # ---------- Save outputs ----------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base_dir, "backtest_output")
    os.makedirs(outdir, exist_ok=True)

    bt["equity"].to_frame("equity").to_csv(os.path.join(outdir, "equity_curve.csv"))
    bt["returns_daily"].to_frame("daily_returns").to_csv(os.path.join(outdir, "daily_returns.csv"))
    bt["weights"].to_csv(os.path.join(outdir, "weights_by_period.csv"))
    bt["trades"].to_csv(os.path.join(outdir, "trades_by_period.csv"))
    stats.to_csv(os.path.join(outdir, "summary_metrics.csv"), header=False)

    # plot
    if benchmark in prices.columns:
        plot_equity_vs_spy(bt, outdir, spy_ticker=benchmark)
    else:
        plot_equity_vs_spy(bt, outdir)

    print("\nSummary metrics:")
    print(stats.to_string())
    print(f"\nSaved CSVs and plot in: {outdir}")
    print("Config: India large caps, W-FRI, lookbacks=[1,3], top_n=10, with RSI/vol/liquidity realism filters.")
    
    compare_to_nifty(bt, outdir, benchmark_ticker="NIFTYBEES.NS")

if __name__ == "__main__":
    main()
