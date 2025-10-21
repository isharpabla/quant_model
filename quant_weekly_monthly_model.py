#!/usr/bin/env python3
# Robust quant momentum backtester for Indian stocks (weekly/monthly)
# Dynamic NSE universe → market-cap buckets (5 large, 5 mid, 5 small)
# Includes RSI/vol/liquidity filters, caching, and benchmark comparison/plot.

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    freq: str = "BM"  # e.g. 'W-FRI'
    lookback_months: List[int] = None
    top_n: int = 3
    min_1m_ret: float = 0.0
    cash_ticker: Optional[str] = "BIL"
    transaction_cost_bps: float = 10.0
    data_dir: Optional[str] = None
    ONLINE: bool = True

    # Buckets
    cap_buckets: Optional[Dict[str, List[str]]] = None
    per_bucket_top_n: Optional[Dict[str, int]] = None  # {"large":5,"mid":5,"small":5}

    # Cache
    cache_dir: str = "cache"
    mcap_cache_days: int = 3

    def __post_init__(self):
        if self.lookback_months is None:
            self.lookback_months = [6, 12]
        if self.per_bucket_top_n is None and self.cap_buckets is not None:
            self.per_bucket_top_n = {k: 5 for k in self.cap_buckets.keys()}


# ---------- Utilities ----------
def cdfloat(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _is_equity_symbol(sym_raw: str) -> bool:
    """
    Keep only real equity symbols (no index labels like 'NIFTY 500').
    """
    sym = sym_raw.strip().upper()
    if " " in sym:
        return False
    for bad in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCAP", "SMALLCAP", "LARGECAP"):
        if sym.startswith(bad):
            return False
    return True


# ---------- Prices ----------
def load_prices(cfg: Config) -> pd.DataFrame:
    want = list(cfg.tickers)
    if cfg.cash_ticker and cfg.cash_ticker not in want:
        want.append(cfg.cash_ticker)

    frames, failed = [], []

    if cfg.data_dir:
        for t in want:
            path = os.path.join(cfg.data_dir, f"{t}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                s = df.set_index("Date")["Adj Close"].rename(t)
                frames.append(s)
            except Exception as e:
                failed.append((t, str(e)))
    elif cfg.ONLINE and yf is not None:
        for t in want:
            try:
                tk = yf.Ticker(t)
                h = tk.history(start=cfg.start, end=cfg.end, auto_adjust=True)
                if h is None or h.empty:
                    continue
                col = "Close" if "Close" in h.columns else "Adj Close"
                s = h[col].copy()
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.sort_index().dropna().rename(t)
                frames.append(s)
            except Exception as e:
                failed.append((t, str(e)))
    else:
        raise RuntimeError("No data source available.")

    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if prices.empty:
        print("[ERROR] No price series loaded.")
        return prices

    prices = prices.loc[cfg.start:cfg.end].sort_index().ffill().dropna(how="all")
    print(f"[LOAD] Loaded {len(prices.columns)} tickers. Missing: {len(want) - len(prices.columns)}")
    if failed:
        print("[LOAD] Some tickers failed:", failed[:5], "...")
    return prices


# ---------- Signals ----------
def momentum_score(prices_m: pd.DataFrame, lookbacks: List[int]) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=prices_m.index, columns=prices_m.columns)
    for L in lookbacks:
        ret = prices_m.shift(1) / prices_m.shift(L + 1) - 1.0
        score = score.add(ret, fill_value=0.0)
    return score / float(len(lookbacks))


def last_1m_return(prices_m: pd.DataFrame) -> pd.DataFrame:
    return prices_m.pct_change(1)


def build_weights_by_bucket(
    scores: pd.DataFrame,
    r1m: pd.DataFrame,
    bucket_map: Dict[str, str],
    per_bucket_top_n: Dict[str, int],
    min_1m_ret: float,
    cash_col: Optional[str],
) -> pd.DataFrame:
    # Rank within each bucket on each date
    weights_list = []
    for bucket, k in per_bucket_top_n.items():
        cols = [c for c, b in bucket_map.items() if b == bucket and c in scores.columns]
        if not cols:
            continue
        ranks_b = scores[cols].rank(axis=1, ascending=False, method="first")
        sel_b = (ranks_b <= int(k)) & (r1m[cols] >= min_1m_ret)
        w_b = sel_b.astype(float)

        # normalize within bucket (so each bucket sums to 1)
        row_sums = w_b.sum(axis=1).replace(0, np.nan)
        w_b = w_b.div(row_sums, axis=0).fillna(0.0)
        weights_list.append(w_b)

    if not weights_list:
        w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    else:
        nb = len(weights_list)
        w = pd.concat(weights_list, axis=1).reindex(scores.columns, axis=1).fillna(0.0)
        w = w * (1.0 / nb)

    if cash_col and cash_col not in w.columns:
        w[cash_col] = 0.0
    if cash_col:
        leftover = 1.0 - w.sum(axis=1)
        w[cash_col] = w[cash_col].add(leftover, fill_value=0.0)
    return w


# ---------- Backtest ----------
def backtest(prices: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    # Resample to rebalance frequency
    prices = prices.sort_index()
    prices_f = prices.resample(cfg.freq).last().dropna(how="all")
    scores = momentum_score(prices_f, cfg.lookback_months)
    r1m = last_1m_return(prices_f)

    # --- Professional Momentum Refinements (historical, every rebalance) ---

    # 1) Daily volatility (60 trading days) for each stock
    daily_rets = prices.pct_change()
    vol_daily = daily_rets.rolling(60, min_periods=20).std()

    # 2) Align volatility to rebalance dates (weekly/monthly index of prices_f)
    vol_rb = vol_daily.reindex(prices_f.index, method="ffill")

    # 3) Risk-normalize momentum: divide each row of 'scores' by same-date vol
    #    (scores is on the rebalance index already)
    mom_adj = scores.divide(vol_rb, axis=1)

    # 4) Row-wise z-score: standardize across all stocks *for each rebalance date*
    def _rowwise_z(df: pd.DataFrame) -> pd.DataFrame:
        m = df.mean(axis=1)
        s = df.std(axis=1).replace(0, np.nan)
        return df.sub(m, axis=0).div(s, axis=0).fillna(0.0)
    
    # 4️⃣ Create hybrid: blend raw and vol-adjusted momentum
    #    70% risk-adjusted + 30% raw momentum
    z_voladj = _rowwise_z(mom_adj)
    z_raw = _rowwise_z(scores)          # standardize the unadjusted momentum
    scores = 0.7 * z_voladj + 0.3 * z_raw

    if cfg.cap_buckets and cfg.per_bucket_top_n:
        bucket_map = {t: b for b, lst in cfg.cap_buckets.items() for t in lst if t in scores.columns}
        weights = build_weights_by_bucket(
            scores, r1m, bucket_map, cfg.per_bucket_top_n, cdfloat(cfg.min_1m_ret), cfg.cash_ticker
        ).shift(1)
    else:
        # Global top-N (fallback)
        ranks = scores.rank(axis=1, ascending=False, method="first")
        sel = (ranks <= cfg.top_n) & (r1m >= cfg.min_1m_ret)
        w = sel.astype(float)
        row_sums = w.sum(axis=1).replace(0, np.nan)
        weights = w.div(row_sums, axis=0).fillna(0.0).shift(1)

        if cfg.cash_ticker and cfg.cash_ticker not in weights.columns:
            weights[cfg.cash_ticker] = 0.0
        if cfg.cash_ticker:
            leftover = 1.0 - weights.sum(axis=1)
            weights[cfg.cash_ticker] = weights[cfg.cash_ticker].add(leftover, fill_value=0.0)

    weights = weights.fillna(0.0)
    weights_daily = weights.reindex(prices.index, method="ffill").fillna(0.0)

    rets_daily = prices.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    port_rets = (weights_daily * rets_daily).sum(axis=1)

    # Simple transaction costs on rebalance days
    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    tc = turnover * (cfg.transaction_cost_bps / 10000.0)
    tc_daily = tc.reindex(port_rets.index).fillna(0.0)
    port_rets = port_rets - tc_daily

    equity = (1 + port_rets).cumprod()

    trades = (weights - w_prev).fillna(0.0)

    return {
        "prices": prices,
        "prices_freq": prices_f,
        "scores": scores,
        "weights": weights,
        "weights_daily": weights_daily,
        "returns_daily": port_rets,
        "equity": equity,
        "turnover": turnover,
        "trades": trades,
    }

# ---------- Metrics ----------
def max_drawdown(s): return float((s / s.cummax() - 1).min())
def annualize_return(r): return float((1 + r.mean())**252 - 1)
def annualize_vol(r): return float(r.std() * np.sqrt(252))
def sharpe(r): 
    vol = annualize_vol(r)
    return float(annualize_return(r) / vol) if vol else np.nan

def summarize(bt):
    r = bt["returns_daily"]; e = bt["equity"]
    return pd.Series({
        "Total Return": float(e.iloc[-1] - 1),
        "CAGR": float(e.iloc[-1]**(252 / len(e)) - 1),
        "Max Drawdown": max_drawdown(e),
        "Ann Return": annualize_return(r),
        "Ann Vol": annualize_vol(r),
        "Sharpe": sharpe(r),
        "Worst Day": float(r.min()),
        "Win Rate (daily)": float((r > 0).mean())
    })

# ---------- NSE Symbols + MCAP (with cache) ----------
def fetch_all_nse_symbols() -> List[str]:
    """
    Returns a broad list of NSE symbols with .NS suffix.
    Filters out non-equity index names (fixes 'NIFTY 500' 404).
    """
    symbols = []
    try:
        from nsetools import Nse
        nse = Nse()
        codes = nse.get_stock_codes()
        syms = [k for k in codes.keys() if k and k != "SYMBOL"]
        syms = [s for s in syms if _is_equity_symbol(s)]
        symbols.extend([s + ".NS" for s in syms])
    except Exception:
        pass

    if not symbols:
        try:
            from nsepython import nsefetch
            data = nsefetch("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500")
            syms = [row["symbol"] for row in data.get("data", []) if "symbol" in row]
            syms = [s for s in syms if _is_equity_symbol(s)]
            symbols.extend([s + ".NS" for s in syms])
        except Exception:
            pass

    return sorted(set(symbols))


def _fetch_one_mcap(ticker: str) -> Tuple[str, Optional[float]]:
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        mcap = getattr(fi, "market_cap", None) if fi else None
        if mcap is None:
            info = tk.info or {}
            mcap = info.get("marketCap")
        return ticker, float(mcap) if mcap and mcap > 0 else None
    except Exception:
        return ticker, None


def _mcaps_cache_paths(cache_dir: str) -> Tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    latest = os.path.join(cache_dir, "mcaps_latest.csv")
    dated = os.path.join(cache_dir, f"mcaps_{pd.Timestamp.now().date().isoformat()}.csv")
    return latest, dated


def fetch_market_caps_with_cache(tickers, cache_dir="cache", staleness_days=3, max_workers=24) -> pd.Series:
    latest, dated = _mcaps_cache_paths(cache_dir)

    # Load cache if fresh
    if os.path.exists(latest):
        try:
            df = pd.read_csv(latest)
            if {"ticker", "market_cap", "asof"}.issubset(df.columns):
                asof = pd.to_datetime(df["asof"].iloc[0]).date()
                age = (pd.Timestamp.now().date() - asof).days
                if age <= staleness_days:
                    print(f"[CACHE] Loaded market caps (age {age}d)")
                    s = pd.Series(df.market_cap.values, index=df.ticker.values, dtype="float64")
                    return s.sort_values(ascending=False)
        except Exception:
            pass

    print("[INFO] Fetching market caps...")
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one_mcap, t): t for t in tickers}
        for fut in as_completed(futures):
            t, mcap = fut.result()
            if mcap:
                out[t] = mcap

    s = pd.Series(out, dtype="float64").sort_values(ascending=False)
    try:
        df = pd.DataFrame({"ticker": s.index, "market_cap": s.values, "asof": pd.Timestamp.now().date()})
        df.to_csv(latest, index=False)
        df.to_csv(dated, index=False)
        print(f"[CACHE] Saved market caps to {latest} and {dated}")
    except Exception as e:
        print(f"[CACHE] Save failed: {e}")
    return s


def split_buckets_by_mcap(mcaps, q_large=0.67, q_mid=0.33, per_bucket_cap=250):
    q_hi = mcaps.quantile(q_large)
    q_lo = mcaps.quantile(q_mid)
    large = mcaps[mcaps >= q_hi].index[:per_bucket_cap].tolist()
    mid   = mcaps[(mcaps < q_hi) & (mcaps >= q_lo)].index[:per_bucket_cap].tolist()
    small = mcaps[mcaps < q_lo].index[:per_bucket_cap].tolist()
    return {"large": large, "mid": mid, "small": small}


# ---------- Benchmark comparison/plot ----------
def plot_equity_vs_benchmark(bt: Dict[str, pd.DataFrame], outdir: str, benchmark_ticker: str = "NIFTYBEES.NS"):
    import matplotlib.pyplot as plt

    prices = bt.get("prices")
    eq = bt.get("equity")
    if prices is None or prices.empty or eq is None or eq.empty:
        print("[WARN] Missing prices/equity; cannot plot vs benchmark.")
        return

    # Ensure benchmark present
    if benchmark_ticker not in prices.columns and yf is not None:
        try:
            bench = yf.Ticker(benchmark_ticker).history(
                start=str(prices.index.min().date()),
                end=str(prices.index.max().date()),
                auto_adjust=True
            )
            if bench is not None and not bench.empty:
                col = "Close" if "Close" in bench.columns else "Adj Close"
                prices = prices.copy()
                prices[benchmark_ticker] = bench[col]
                prices = prices.sort_index().ffill()
            else:
                print(f"[INFO] Could not fetch {benchmark_ticker}; skipping plot.")
                return
        except Exception as e:
            print(f"[INFO] Benchmark fetch failed: {e}; skipping plot.")
            return

    bench_curve = (prices[benchmark_ticker].pct_change().fillna(0.0) + 1.0).cumprod()
    bench_curve = bench_curve / bench_curve.iloc[0]
    strat_curve = eq / eq.iloc[0]

    common_idx = strat_curve.index.intersection(bench_curve.index)
    strat_curve = strat_curve.reindex(common_idx).ffill()
    bench_curve = bench_curve.reindex(common_idx).ffill()

    ax = strat_curve.plot(figsize=(10, 5), label="Strategy")
    bench_curve.plot(ax=ax, label=benchmark_ticker)
    ax.set_title("Equity Curve vs Benchmark")
    ax.set_xlabel("Date"); ax.set_ylabel("Equity (normalized)")
    ax.grid(True); ax.legend()
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, "equity_vs_benchmark.png")
    import matplotlib.pyplot as plt  # ensure scope
    plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.close()
    print(f"Saved plot: {png_path}")


def print_relative_to_benchmark(bt: Dict[str, pd.DataFrame], benchmark_ticker: str = "NIFTYBEES.NS"):
    prices = bt.get("prices"); eq = bt.get("equity")
    if prices is None or prices.empty or eq is None or eq.empty:
        return

    # ensure benchmark exists
    if benchmark_ticker not in prices.columns and yf is not None:
        try:
            bench = yf.Ticker(benchmark_ticker).history(
                start=str(prices.index.min().date()),
                end=str(prices.index.max().date()),
                auto_adjust=True
            )
            if bench is not None and not bench.empty:
                col = "Close" if "Close" in bench.columns else "Adj Close"
                prices = prices.copy()
                prices[benchmark_ticker] = bench[col]
                prices = prices.sort_index().ffill()
            else:
                print(f"[INFO] Could not fetch {benchmark_ticker}; skipping relative performance print.")
                return
        except Exception as e:
            print(f"[INFO] Benchmark fetch failed: {e}; skipping relative performance print.")
            return

    bench_curve = (prices[benchmark_ticker].pct_change().fillna(0.0) + 1.0).cumprod()
    bench_curve = bench_curve / bench_curve.iloc[0]
    strat_curve = eq / eq.iloc[0]

    idx = strat_curve.index.intersection(bench_curve.index)
    strat_curve = strat_curve.reindex(idx).ffill()
    bench_curve = bench_curve.reindex(idx).ffill()

    rel = (strat_curve / bench_curve) - 1.0
    strat_ret = float(strat_curve.iloc[-1] - 1.0)
    bench_ret = float(bench_curve.iloc[-1] - 1.0)
    rel_ret = float(rel.iloc[-1])

    print("\n[Relative Performance vs NIFTYBEES]")
    print(f"Strategy cumulative return: {strat_ret*100:.2f}%")
    print(f"NIFTYBEES cumulative return: {bench_ret*100:.2f}%")
    print(f"Strategy vs NIFTYBEES (final): {rel_ret*100:.2f}%")


# ---------- Main ----------
def main():
    # --- Dynamic NSE universe
    all_syms = fetch_all_nse_symbols()
    if not all_syms:
        print("[FATAL] No NSE symbols found.")
        return

    # --- Market caps → buckets
    mcaps = fetch_market_caps_with_cache(all_syms, cache_dir="cache", staleness_days=3, max_workers=24)
    buckets = split_buckets_by_mcap(mcaps, q_large=0.67, q_mid=0.33, per_bucket_cap=250)
    base_universe = sorted(set(buckets["large"] + buckets["mid"] + buckets["small"]))

    # Simple sanity print
    print(f"[BUCKETS] large={len(buckets['large'])}, mid={len(buckets['mid'])}, small={len(buckets['small'])}")

    cfg = Config(
        tickers=base_universe,
        start="2022-01-01",
        end=None,
        freq="W-FRI",
        lookback_months=[1, 3],
        top_n=15,  # unused when bucketed
        min_1m_ret=0.0,
        cash_ticker="BIL",
        transaction_cost_bps=10.0,
        data_dir=None,
        ONLINE=True,
        cap_buckets=buckets,
        per_bucket_top_n={"large": 5, "mid": 5, "small": 5},
        cache_dir="cache",
        mcap_cache_days=3,
    )

    # Load prices
    prices = load_prices(cfg)
    if prices.empty:
        print("[FATAL] No prices loaded.")
        return

    # Backtest
    bt = backtest(prices, cfg)
    stats = summarize(bt)

    # Outputs
    outdir = "backtest_output"
    os.makedirs(outdir, exist_ok=True)
    bt["equity"].to_frame("equity").to_csv(f"{outdir}/equity_curve.csv")
    bt["returns_daily"].to_frame("daily_returns").to_csv(f"{outdir}/daily_returns.csv")
    bt["weights"].to_csv(f"{outdir}/weights_by_period.csv")
    bt["trades"].to_csv(f"{outdir}/trades_by_period.csv")
    stats.to_csv(f"{outdir}/summary_metrics.csv")

    # Plot + relative vs NIFTYBEES
    plot_equity_vs_benchmark(bt, outdir, benchmark_ticker="NIFTYBEES.NS")
    print_relative_to_benchmark(bt, benchmark_ticker="NIFTYBEES.NS")

    # Shortlist with bucket labels
    w_last = bt["weights"].iloc[-1].dropna()
    if cfg.cash_ticker in w_last.index:
        w_last = w_last.drop(cfg.cash_ticker, errors="ignore")
    shortlist = list(w_last[w_last > 0].sort_values(ascending=False).index)

    bucket_of = {t: b for b, lst in cfg.cap_buckets.items() for t in lst}
    bucket_labels = [bucket_of.get(t, "unknown") for t in shortlist]

    df_short = pd.DataFrame({"Ticker": shortlist, "Bucket": bucket_labels})
    df_short.to_csv(f"{outdir}/shortlist_latest_15.csv", index=False)

    print("\n[SHORTLIST @ last rebalance]")
    for t, b in zip(shortlist, bucket_labels):
        print(f"{t}  ({b})")

    print("\nSummary metrics:")
    print(stats.to_string())
    print(f"\nSaved outputs in: {os.path.abspath(outdir)}")
    print("Config: dynamic NSE universe → 5 large, 5 mid, 5 small momentum picks.")


if __name__ == "__main__":
    main()
