#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage0: Enrich raw LOB CSVs with additional microstructure factors (NO future leakage).
[Optimized Version: Uses PyArrow for high-speed IO and ProcessPoolExecutor]

Outputs:
- Enriched CSV per input file
"""

import argparse
import json
import os
import sys
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

# Use ProcessPoolExecutor for CPU/IO intensive tasks to bypass GIL
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_existing_factor_count(cfg: dict) -> int:
    try:
        c = int(cfg["data"]["required_columns"]["factors"]["count"])
        return c
    except Exception:
        pass
    try:
        return int(cfg["features"]["num_factors"])
    except Exception:
        return 0


def infer_next_index_from_existing(cols: List[str], rename_prefix: str, fallback: int) -> int:
    pat = re.compile(rf"^{re.escape(rename_prefix)}(\d+)$")
    mx = None
    for c in cols:
        m = pat.match(str(c))
        if not m:
            continue
        try:
            v = int(m.group(1))
        except Exception:
            continue
        mx = v if mx is None else max(mx, v)
    return (mx + 1) if mx is not None else int(fallback)


def ensure_finite(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).fillna(fill)


def get_float_shares_from_cfg(cfg: dict, stock_code: Optional[str]) -> float:
    data = cfg.get("data", {}) or {}
    stocks = data.get("stocks") or []
    if not stock_code:
        raise KeyError("float_shares not found in CSV and stock_code is None")

    for s in stocks:
        if str(s.get("stock_code")) == str(stock_code):
            fs = s.get("float_shares")
            if fs is None:
                raise KeyError(f"cfg.data.stocks[{stock_code}].float_shares is missing")
            try:
                fs = float(fs)
            except Exception:
                raise ValueError(f"cfg.data.stocks[{stock_code}].float_shares is not a number: {fs}")
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError(f"cfg.data.stocks[{stock_code}].float_shares must be >0")
            return fs

    raise KeyError(f"stock_code={stock_code} not found in cfg.data.stocks")

# -----------------------------
# Helpers
# -----------------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def resolve_path(project_root: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(project_root, p)

def ts_to_seconds(ts_val, assume_digits: int = 9) -> float:
    try:
        x = int(float(ts_val))
    except Exception:
        return float("nan")
    s = f"{x:0{assume_digits}d}"
    if len(s) != assume_digits:
        s = s[-assume_digits:].rjust(assume_digits, "0")
    hh = int(s[0:2]); mm = int(s[2:4]); ss = int(s[4:6]); ms = int(s[6:9])
    return hh * 3600.0 + mm * 60.0 + ss + ms / 1000.0

def safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def inv_level_weights(levels: int) -> np.ndarray:
    return np.array([1.0 / (i + 1) for i in range(levels)], dtype=np.float64)

def choose_price_by_cum(depth: pd.Series,
                        prices: List[pd.Series],
                        cum_depths: List[pd.Series]) -> pd.Series:
    if len(prices) == 0 or len(prices) != len(cum_depths):
        raise ValueError(f"prices/cum_depths length mismatch")

    conds = []
    choices = []
    for i in range(len(prices)):
        if i == 0:
            cond = depth <= cum_depths[i]
        else:
            cond = (depth > cum_depths[i - 1]) & (depth <= cum_depths[i])
        conds.append(cond)
        choices.append(prices[i])

    out = np.select(conds, choices, default=np.nan)
    return pd.Series(out, index=prices[0].index)

def compute_ofi_level(bid_p: pd.Series, bid_q: pd.Series,
                      ask_p: pd.Series, ask_q: pd.Series) -> pd.Series:
    pb = bid_p
    qb = bid_q
    pa = ask_p
    qa = ask_q

    pb_prev = pb.shift(1)
    qb_prev = qb.shift(1)
    pa_prev = pa.shift(1)
    qa_prev = qa.shift(1)

    e_bid = np.where(pb > pb_prev, qb,
             np.where(pb == pb_prev, qb - qb_prev,
             -qb_prev))
    e_ask = np.where(pa < pa_prev, qa,
             np.where(pa == pa_prev, qa - qa_prev,
             -qa_prev))
    return pd.Series(e_bid - e_ask, index=pb.index)

def run_length_of_sign(sign_s: pd.Series) -> pd.Series:
    s = sign_s.fillna(0).astype(int)
    change = (s != s.shift(1)) | (s == 0)
    grp = change.cumsum()
    runlen = grp.groupby(grp).cumcount() + 1
    runlen = runlen.where(s != 0, 0)
    return runlen


def normalize_keep_features(keep: List[str], prefix: str) -> List[str]:
    out = []
    for k in keep:
        k = str(k).strip()
        if not k:
            continue
        if k.startswith(prefix):
            out.append(k)
        else:
            out.append(prefix + k)
    return out


def build_factor_map(keep_cols: List[str], out_cols: List[str]) -> dict:
    return {
        "version": 1,
        "mapping": [
            {"out": out_cols[i], "source": keep_cols[i]}
            for i in range(len(keep_cols))
        ],
    }

# -----------------------------
# Factor computation
# -----------------------------

def enrich_df(df: pd.DataFrame, cfg: dict, stock_code: Optional[str] = None) -> pd.DataFrame:
    s0 = cfg.get("stage0", {}) or {}
    levels = int(s0.get("levels", 4))
    include_level0 = bool(s0.get("include_level0", True))
    prefix = s0.get("prefix", "s0_")

    l0_cols = ["bid", "ask", "bz", "az"]
    has_l0 = all(c in df.columns for c in l0_cols)
    if include_level0 and not has_l0:
         # Fallback or error handled by caller usually, but let's strictly check
         # For performance we assume happy path mostly
         pass
    use_l0 = include_level0 and has_l0

    def _cols(base: str) -> List[str]:
        if use_l0:
            return [base] + [f"{base}{i}" for i in range(1, levels + 1)]
        return [f"{base}{i}" for i in range(1, levels + 1)]

    bid_cols = _cols("bid")
    ask_cols = _cols("ask")
    bz_cols = _cols("bz")
    az_cols = _cols("az")
    total_levels = len(bid_cols)

    need_cols = [
        "timestamp", "last",
        *bid_cols,
        *ask_cols,
        *bz_cols,
        *az_cols,
        "volume", "acc_volume", "turnover", "acc_turnover",
    ]

    # Fast path: numeric conversion not needed if using PyArrow engine usually,
    # but strictly ensuring types is good for vectorization
    for c in need_cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
             df[c] = pd.to_numeric(df[c], errors="coerce")

    if "float_shares" in df.columns:
        df["float_shares"] = pd.to_numeric(df["float_shares"], errors="coerce")
        fs_series = df["float_shares"]
    else:
        fs_val = get_float_shares_from_cfg(cfg, stock_code)
        fs_series = pd.Series(np.full(len(df), fs_val, dtype=np.float64), index=df.index)

    vol = df["volume"]
    amt = df["turnover"]
    trade_flag = (vol > 0).astype(np.float64)

    t_sec = df["timestamp"].apply(ts_to_seconds).astype(np.float64)
    dt = t_sec.diff()
    dt_pos = dt.where(dt > 0)
    dt_med = float(np.nanmedian(dt_pos.values)) if np.isfinite(np.nanmedian(dt_pos.values)) else np.nan
    dt_filled = dt_pos.fillna(dt_med)
    df[prefix + "t_sec"] = t_sec
    df[prefix + "dt_sec"] = dt_filled

    bid_p = [df[c] for c in bid_cols]
    ask_p = [df[c] for c in ask_cols]
    bid_q = [df[c] for c in bz_cols]
    ask_q = [df[c] for c in az_cols]

    bid_top = bid_p[0]
    ask_top = ask_p[0]
    lastp = df["last"]

    has_bidask = bid_top.notna() & ask_top.notna()
    mid = ((bid_top + ask_top) / 2.0).where(has_bidask, lastp)
    df[prefix + "mid"] = mid
    spread = (ask_top - bid_top)
    df[prefix + "spread"] = spread
    df[prefix + "rel_spread"] = safe_div(spread, mid)

    bz_top = bid_q[0]
    az_top = ask_q[0]
    micro = safe_div(ask_top * bz_top + bid_top * az_top, (bz_top + az_top))
    df[prefix + "microprice"] = micro
    df[prefix + "microprice_dev"] = safe_div(micro - mid, mid)

    depth_bid_L1 = bid_q[0]
    depth_ask_L1 = ask_q[0]
    depth_bid_L4 = sum(bid_q)
    depth_ask_L4 = sum(ask_q)
    df[prefix + "depth_bid_L1"] = depth_bid_L1
    df[prefix + "depth_ask_L1"] = depth_ask_L1
    df[prefix + "depth_bid_L4"] = depth_bid_L4
    df[prefix + "depth_ask_L4"] = depth_ask_L4
    df[prefix + "depth_total_L4"] = depth_bid_L4 + depth_ask_L4

    df[prefix + "log_depth_total_L4"] = np.log1p(df[prefix + "depth_total_L4"])
    df[prefix + "log_depth_bid_L1"] = np.log1p(depth_bid_L1.clip(lower=0.0))
    df[prefix + "log_depth_ask_L1"] = np.log1p(depth_ask_L1.clip(lower=0.0))

    obi_L1 = safe_div(depth_bid_L1 - depth_ask_L1, depth_bid_L1 + depth_ask_L1)
    obi_L4 = safe_div(depth_bid_L4 - depth_ask_L4, depth_bid_L4 + depth_ask_L4)
    df[prefix + "obi_L1"] = obi_L1
    df[prefix + "obi_L4"] = obi_L4

    w = inv_level_weights(total_levels)
    w_bid = sum(w[i] * bid_q[i] for i in range(total_levels))
    w_ask = sum(w[i] * ask_q[i] for i in range(total_levels))
    wobi = safe_div(w_bid - w_ask, w_bid + w_ask)
    df[prefix + "wobi_L4"] = wobi

    bid_deep = bid_p[-1]
    ask_deep = ask_p[-1]
    dp_bid = (mid - bid_deep)
    dp_ask = (ask_deep - mid)
    slope_bid = safe_div(dp_bid, depth_bid_L4)
    slope_ask = safe_div(dp_ask, depth_ask_L4)
    df[prefix + "slope_bid_L4"] = slope_bid
    df[prefix + "slope_ask_L4"] = slope_ask
    df[prefix + "slope_asym"] = slope_ask - slope_bid

    dp_bid_L1 = (mid - bid_top)
    dp_ask_L1 = (ask_top - mid)
    slope_bid_L1 = safe_div(dp_bid_L1, depth_bid_L1)
    slope_ask_L1 = safe_div(dp_ask_L1, depth_ask_L1)
    df[prefix + "curv_bid_L1_over_L4"] = safe_div(slope_bid_L1, slope_bid)
    df[prefix + "curv_ask_L1_over_L4"] = safe_div(slope_ask_L1, slope_ask)

    churn_bid = sum(bid_q[i].diff().abs() for i in range(total_levels))
    churn_ask = sum(ask_q[i].diff().abs() for i in range(total_levels))
    df[prefix + "churn_bid_L4"] = safe_div(churn_bid, depth_bid_L4.shift(1))
    df[prefix + "churn_ask_L4"] = safe_div(churn_ask, depth_ask_L4.shift(1))
    df[prefix + "churn_asym"] = df[prefix + "churn_bid_L4"] - df[prefix + "churn_ask_L4"]

    persistent_m = int(s0.get("persistent_m", 5))
    sign_obi = np.sign(obi_L4.fillna(0.0)).astype(int)
    runlen = run_length_of_sign(sign_obi)
    df[prefix + "obi_persistent"] = obi_L4.where((runlen >= persistent_m) & (sign_obi != 0), 0.0)
    df[prefix + "obi_runlen"] = runlen

    flicker_w = int(s0.get("flicker_window", 50))
    flips = ((sign_obi != sign_obi.shift(1)) & (sign_obi != 0) & (sign_obi.shift(1) != 0)).astype(float)
    df[prefix + "obi_flicker_cnt"] = flips.rolling(flicker_w, min_periods=max(3, flicker_w // 5)).sum()
    df[prefix + "obi_flicker_rate"] = safe_div(df[prefix + "obi_flicker_cnt"], float(flicker_w))

    ofi_all: List[pd.Series] = []
    for i in range(total_levels):
        ofi_i = compute_ofi_level(bid_p[i], bid_q[i], ask_p[i], ask_q[i])
        ofi_all.append(ofi_i)

    for i in range(4):
        if i < total_levels:
            df[prefix + f"ofi_L{i+1}"] = ofi_all[i]
        else:
            df[prefix + f"ofi_L{i+1}"] = 0.0

    ofi_w = inv_level_weights(total_levels)
    iofi = sum(ofi_w[i] * ofi_all[i] for i in range(total_levels))
    df[prefix + "iofi"] = iofi

    ofi_span = int(s0.get("ofi_ema_span", 50))
    iofi_ema = iofi.ewm(span=ofi_span, adjust=False).mean()
    iofi_ema_past = iofi_ema.shift(1)
    df[prefix + "iofi_ema_past"] = iofi_ema_past
    df[prefix + "iofi_surprise"] = iofi - iofi_ema_past

    ofi_vol_w = int(s0.get("ofi_vol_window", 50))
    df[prefix + "iofi_vol"] = iofi.rolling(ofi_vol_w, min_periods=max(5, ofi_vol_w // 5)).std()

    df[prefix + "vol_rate"] = safe_div(vol, dt_filled)
    df[prefix + "amt_rate"] = safe_div(amt, dt_filled)
    df[prefix + "trade_flag"] = trade_flag

    vwap_tick = safe_div(amt, vol.replace(0, np.nan))
    df[prefix + "vwap_tick"] = vwap_tick
    vwap_dev = safe_div(vwap_tick - mid, mid)
    df[prefix + "vwap_dev"] = vwap_dev.where(trade_flag > 0, 0.0)

    r_mid = mid.pct_change()
    df[prefix + "r_mid"] = r_mid
    illiq = safe_div(r_mid.abs(), amt)
    df[prefix + "illiq"] = illiq.where(amt > 0, 0.0)

    float_shares = fs_series
    mcap = mid * float_shares
    df[prefix + "vol_ratio_tick"] = safe_div(vol, float_shares.replace(0, np.nan))
    df[prefix + "amt_ratio_tick"] = safe_div(amt, mcap.replace(0, np.nan))

    ofi_norm_cfg = (s0.get("ofi_norm") or {})
    if bool(ofi_norm_cfg.get("enabled", False)):
        dv_span = int(ofi_norm_cfg.get("dv_ema_span", 200))
        dv_floor = float(ofi_norm_cfg.get("dv_floor", 1.0))
        dv = df["acc_volume"].diff()
        dv = dv.where(dv > 0, 0.0)
        dv_ema = dv.ewm(span=dv_span, adjust=False).mean()
        denom = np.maximum(dv_ema, dv_floor)

        for i in range(4):
            c = prefix + f"ofi_L{i+1}"
            df[prefix + f"ofi_norm_L{i+1}"] = safe_div(df[c], denom)
        df[prefix + "iofi_norm"] = safe_div(df[prefix + "iofi"], denom)
        df[prefix + "iofi_surprise_norm"] = safe_div(df[prefix + "iofi_surprise"], denom)
        df[prefix + "iofi_vol_norm"] = safe_div(df[prefix + "iofi_vol"], denom)

    impact_fracs = s0.get("impact_fracs", [0.25, 0.5, 0.75])
    impact_fracs = [float(x) for x in impact_fracs]

    cum_asks: List[pd.Series] = []
    cum_bids: List[pd.Series] = []
    for i in range(total_levels):
        cum_asks.append(ask_q[i] if i == 0 else (cum_asks[i - 1] + ask_q[i]))
        cum_bids.append(bid_q[i] if i == 0 else (cum_bids[i - 1] + bid_q[i]))

    for frac in impact_fracs:
        v_buy = frac * depth_ask_L4
        p_buy = choose_price_by_cum(v_buy, prices=ask_p, cum_depths=cum_asks)
        imp_buy = safe_div(p_buy - mid, mid)
        v_sell = frac * depth_bid_L4
        p_sell = choose_price_by_cum(v_sell, prices=bid_p, cum_depths=cum_bids)
        imp_sell = safe_div(mid - p_sell, mid)

        tag = str(frac).replace(".", "_")
        df[prefix + f"impact_buy_f{tag}"] = imp_buy
        df[prefix + f"impact_sell_f{tag}"] = imp_sell
        df[prefix + f"impact_asym_f{tag}"] = imp_buy - imp_sell

    if bool(s0.get("float32_new_cols", True)):
        for c in df.columns:
            if c.startswith(prefix):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    delta_like = [
        prefix + "churn_bid_L4", prefix + "churn_ask_L4", prefix + "churn_asym",
        prefix + "r_mid",
        prefix + "ofi_L1", prefix + "ofi_L2", prefix + "ofi_L3", prefix + "ofi_L4",
        prefix + "iofi", prefix + "iofi_surprise",
    ]
    rolling_like = [
        prefix + "obi_flicker_cnt", prefix + "obi_flicker_rate",
        prefix + "iofi_vol",
    ]
    if bool((s0.get("ofi_norm") or {}).get("enabled", False)):
        delta_like += [
            prefix + "ofi_norm_L1", prefix + "ofi_norm_L2", prefix + "ofi_norm_L3", prefix + "ofi_norm_L4",
            prefix + "iofi_norm", prefix + "iofi_surprise_norm",
        ]
        rolling_like += [prefix + "iofi_vol_norm"]

    for c in delta_like + rolling_like + [prefix + "vwap_dev", prefix + "illiq", prefix + "trade_flag"]:
        if c in df.columns:
            df[c] = ensure_finite(df[c], fill=0.0)

    return df


# -----------------------------
# Baseline
# -----------------------------

def _baseline_cfg(stage0_cfg: dict) -> dict:
    return (stage0_cfg.get("baseline") or {})

def _baseline_stats_path(project_root: str, s0: dict, stock_code: str, date: str, session: Optional[int]) -> str:
    bcfg = _baseline_cfg(s0)
    base_root = resolve_path(project_root, bcfg.get("baseline_root")) or os.path.join(project_root, "data", "stage0_baseline_stats")
    sess_tag = f"s{int(session)}" if session is not None else "s?"
    odir = os.path.join(base_root, str(stock_code), str(date))
    os.makedirs(odir, exist_ok=True)
    return os.path.join(odir, f"baseline_{sess_tag}.npz")

def _bucket_id_from_tsec(t_sec: pd.Series, bucket_seconds: int) -> np.ndarray:
    bs = max(int(bucket_seconds), 1)
    n_buckets = int(np.ceil(86400.0 / bs))
    bid = np.floor(np.asarray(t_sec, dtype=np.float64) / float(bs)).astype(np.int32)
    bid = np.clip(bid, 0, n_buckets - 1)
    return bid

def _combine_stats(n: np.ndarray, mean: np.ndarray, M2: np.ndarray,
                   n2: np.ndarray, mean2: np.ndarray, M2_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = n.astype(np.float64, copy=False)
    n2 = n2.astype(np.float64, copy=False)
    tot = n + n2
    out_mean = mean.copy()
    out_M2 = M2.copy()
    mask = tot > 0
    mask_both = (n > 0) & (n2 > 0)
    only2 = (n == 0) & (n2 > 0)
    out_mean[only2] = mean2[only2]
    out_M2[only2] = M2_2[only2]
    if np.any(mask_both):
        delta = mean2[mask_both] - mean[mask_both]
        out_mean[mask_both] = mean[mask_both] + delta * (n2[mask_both] / tot[mask_both])
        out_M2[mask_both] = M2[mask_both] + M2_2[mask_both] + (delta * delta) * (n[mask_both] * n2[mask_both] / tot[mask_both])
    return tot.astype(np.float64), out_mean, out_M2

def _build_index(rows_all: List[dict]) -> Dict[Tuple[str, str, Optional[int]], str]:
    idx = {}
    for r in rows_all:
        sc = r.get("stock_code")
        d = r.get("date")
        sess = r.get("session")
        p = r.get("path")
        if sc is None or d is None or p is None:
            continue
        try:
            sess_i = int(sess) if sess is not None else None
        except Exception:
            sess_i = None
        idx[(str(sc), str(d), sess_i)] = str(p)
    return idx


def build_baseline_stats_for_key(
    *,
    cfg: dict,
    project_root: str,
    s0: dict,
    idx_all: Dict[Tuple[str, str, Optional[int]], str],
    stock_code: str,
    date: str,
    session: Optional[int],
) -> str:
    bcfg = _baseline_cfg(s0)
    if not bool(bcfg.get("enabled", False)):
        raise ValueError("baseline.enabled is false")

    lookback_days = int(bcfg.get("lookback_days", 20))
    bucket_seconds = int(bcfg.get("bucket_seconds", 60))
    sigma_floor = float(bcfg.get("sigma_floor", 1e-6))

    # Whether to apply per-metric absolute clipping (metric_specs[].clip_abs)
    use_metric_clip_abs = bool(bcfg.get("use_metric_clip_abs", True))

    metric_specs = bcfg.get("metric_specs") or []
    if not metric_specs:
        raise KeyError("stage0.baseline.metric_specs is empty")

    all_dates = sorted({d for (sc, d, sess) in idx_all.keys() if sc == str(stock_code) and (sess == int(session) if session is not None else sess is None)})
    hist_dates = [d for d in all_dates if d < str(date)]
    hist_dates = hist_dates[-lookback_days:]

    out_path = _baseline_stats_path(project_root, s0, stock_code, date, session)
    bs = max(bucket_seconds, 1)
    n_buckets = int(np.ceil(86400.0 / bs))

    n_dict = {}
    mean_dict = {}
    M2_dict = {}
    for ms in metric_specs:
        name = str(ms.get("name"))
        n_dict[name] = np.zeros(n_buckets, dtype=np.float64)
        mean_dict[name] = np.zeros(n_buckets, dtype=np.float64)
        M2_dict[name] = np.zeros(n_buckets, dtype=np.float64)

    prefix = str(s0.get("prefix", "s0_"))

    for hd in hist_dates:
        key = (str(stock_code), str(hd), int(session) if session is not None else None)
        in_path = idx_all.get(key)
        if not in_path or (not os.path.isfile(in_path)):
            continue

        try:
            # OPTIMIZATION: Use PyArrow for reading baseline source files too
            df = pd.read_csv(in_path, engine="pyarrow")
        except Exception:
            # Fallback
            try:
                df = pd.read_csv(in_path)
            except Exception:
                continue

        try:
            df2 = enrich_df(df, cfg, stock_code=str(stock_code))
        except Exception:
            continue

        t_sec = df2.get(prefix + "t_sec")
        if t_sec is None:
            continue
        bid = _bucket_id_from_tsec(t_sec, bucket_seconds=bs)
        df2[prefix + "bucket_id"] = bid

        for ms in metric_specs:
            name = str(ms.get("name"))
            src = str(ms.get("source") or name)
            clip_abs = ms.get("clip_abs") if use_metric_clip_abs else None

            col = prefix + src
            if col not in df2.columns:
                if bool(bcfg.get("strict_metrics", True)):
                    raise KeyError(f"baseline metric source column missing: {col}")
                else:
                    continue

            x = pd.to_numeric(df2[col], errors="coerce")
            x = x.replace([np.inf, -np.inf], np.nan)
            if clip_abs is not None:
                try:
                    ca = float(clip_abs)
                    x = x.clip(lower=-ca, upper=ca)
                except Exception:
                    pass

            tmp = pd.DataFrame({"bucket_id": bid, "x": x})
            tmp = tmp.dropna()
            if tmp.empty:
                continue

            g = tmp.groupby("bucket_id")["x"]
            cnt = g.size()
            mu = g.mean()
            var0 = g.var(ddof=0)
            M2_2 = (var0 * cnt).reindex(range(n_buckets), fill_value=0.0).to_numpy(dtype=np.float64)
            n2 = cnt.reindex(range(n_buckets), fill_value=0).to_numpy(dtype=np.float64)
            mean2 = mu.reindex(range(n_buckets), fill_value=0.0).to_numpy(dtype=np.float64)

            n_new, mean_new, M2_new = _combine_stats(n_dict[name], mean_dict[name], M2_dict[name], n2, mean2, M2_2)
            n_dict[name], mean_dict[name], M2_dict[name] = n_new, mean_new, M2_new

    npz = {
        "bucket_seconds": np.array([bs], dtype=np.int32),
        "lookback_days": np.array([lookback_days], dtype=np.int32),
        "date": np.array([int(date)], dtype=np.int32) if str(date).isdigit() else np.array([0], dtype=np.int32),
        "session": np.array([int(session) if session is not None else -1], dtype=np.int32),
        "metrics": np.array([str(ms.get("name")) for ms in metric_specs], dtype=object),
        "hist_dates": np.array([int(d) for d in hist_dates if str(d).isdigit()], dtype=np.int32),
    }

    for ms in metric_specs:
        name = str(ms.get("name"))
        n = n_dict[name]
        mu = mean_dict[name]
        sigma = np.sqrt(np.where(n > 0, M2_dict[name] / np.maximum(n, 1.0), 0.0))
        sigma = np.where(n >= 2, np.maximum(sigma, sigma_floor), np.nan)

        npz[f"cnt__{name}"] = n.astype(np.float32)
        npz[f"mu__{name}"] = mu.astype(np.float32)
        npz[f"sigma__{name}"] = sigma.astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **npz)
    return out_path


def build_baseline_stats_for_rows(*, cfg: dict, project_root: str, s0: dict, rows_need: List[dict], rows_all: List[dict]) -> None:
    # Note: Logic moved to stage0_build_baseline_stats.py for parallelism.
    # This remains for single-threaded fallback if called directly.
    bcfg = _baseline_cfg(s0)
    if not bool(bcfg.get("enabled", False)):
        return

    overwrite_stats = bool(bcfg.get("overwrite_stats", False))
    idx_all = _build_index(rows_all)

    keys = []
    for r in rows_need:
        sc = r.get("stock_code")
        d = r.get("date")
        sess = r.get("session")
        if sc is None or d is None:
            continue
        try:
            sess_i = int(sess) if sess is not None else None
        except Exception:
            sess_i = None
        keys.append((str(sc), str(d), sess_i))

    keys = sorted(set(keys), key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2]))

    for (sc, d, sess_i) in keys:
        out_path = _baseline_stats_path(project_root, s0, sc, d, sess_i)
        if (not overwrite_stats) and os.path.isfile(out_path):
            continue
        build_baseline_stats_for_key(cfg=cfg, project_root=project_root, s0=s0, idx_all=idx_all, stock_code=sc, date=d, session=sess_i)


def apply_baseline_deviation(
    df: pd.DataFrame,
    *,
    cfg: dict,
    project_root: str,
    stock_code: str,
    date: Optional[str],
    session: Optional[int],
) -> pd.DataFrame:
    s0 = cfg.get("stage0", {}) or {}
    bcfg = _baseline_cfg(s0)
    if not bool(bcfg.get("enabled", False)):
        return df

    if not date:
        return df

    prefix = str(s0.get("prefix", "s0_"))
    lookback_days = int(bcfg.get("lookback_days", 20))
    bucket_seconds = int(bcfg.get("bucket_seconds", 60))
    z_clip = float(bcfg.get("z_clip", 8.0))
    eps = float(bcfg.get("eps", s0.get("eps", 1e-12)))
    min_bucket_samples = int(bcfg.get("min_bucket_samples", 30))

    # Controls for baseline deviation clipping
    use_z_clip = bool(bcfg.get("use_z_clip", True))
    use_metric_clip_abs = bool(bcfg.get("use_metric_clip_abs", True))

    metric_specs = bcfg.get("metric_specs") or []
    if not metric_specs:
        raise KeyError("stage0.baseline.metric_specs is empty")

    stats_path = _baseline_stats_path(project_root, s0, stock_code, str(date), session)
    if not os.path.isfile(stats_path):
        for ms in metric_specs:
            name = str(ms.get("name"))
            out_col = prefix + f"base_{name}_z{lookback_days}"
            df[out_col] = 0.0
        return df

    try:
        z = np.load(stats_path, allow_pickle=True)
    except Exception:
        for ms in metric_specs:
            name = str(ms.get("name"))
            out_col = prefix + f"base_{name}_z{lookback_days}"
            df[out_col] = 0.0
        return df

    t_sec = df.get(prefix + "t_sec")
    if t_sec is None:
        return df

    bid = _bucket_id_from_tsec(t_sec, bucket_seconds=bucket_seconds)
    df[prefix + "bucket_id"] = bid

    for ms in metric_specs:
        name = str(ms.get("name"))
        src = str(ms.get("source") or name)
        clip_abs = ms.get("clip_abs") if use_metric_clip_abs else None

        src_col = prefix + src
        out_col = prefix + f"base_{name}_z{lookback_days}"

        if src_col not in df.columns:
            if bool(bcfg.get("strict_metrics", True)):
                raise KeyError(f"baseline metric source column missing in df: {src_col}")
            df[out_col] = 0.0
            continue
        k_mu = f"mu__{name}"
        k_sig = f"sigma__{name}"
        k_cnt = f"cnt__{name}"
        if (k_mu not in z) or (k_sig not in z) or (k_cnt not in z):
            df[out_col] = 0.0
            continue

        mu = z[k_mu]
        sigma = z[k_sig]
        cnt = z[k_cnt]

        x = pd.to_numeric(df[src_col], errors="coerce").astype(np.float64)
        x = x.replace([np.inf, -np.inf], np.nan)
        if clip_abs is not None:
            try:
                ca = float(clip_abs)
                x = x.clip(lower=-ca, upper=ca)
            except Exception:
                pass

        mu_b = mu[bid]
        sigma_b = sigma[bid]
        cnt_b = cnt[bid]

        zz = (x.to_numpy(dtype=np.float64) - mu_b) / (sigma_b + eps)
        if use_z_clip and np.isfinite(z_clip) and (z_clip > 0):
            zz = np.clip(zz, -z_clip, z_clip)
        zz = np.where(cnt_b >= min_bucket_samples, zz, 0.0)
        zz = np.nan_to_num(zz, nan=0.0, posinf=0.0, neginf=0.0)

        df[out_col] = zz.astype(np.float32)

    return df

# -----------------------------
# IO & orchestration
# -----------------------------

def iter_stage1_ok_manifest(manifest_ok_path: str) -> List[dict]:
    rows = []
    with open(manifest_ok_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def infer_stock_date_session_from_manifest_row(r: dict) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    sc = r.get("stock_code")
    d = r.get("date")
    sess = r.get("session")
    try:
        sess = int(sess) if sess is not None else None
    except Exception:
        sess = None
    return (str(sc) if sc is not None else None, str(d) if d is not None else None, sess)

def get_stock_specs(cfg: dict) -> List[dict]:
    data = cfg.get("data", {}) or {}
    stocks = data.get("stocks")
    if isinstance(stocks, list) and stocks:
        out = []
        for s in stocks:
            rr = s.get("raw_root")
            if not rr:
                continue
            out.append({
                "stock_code": s.get("stock_code"),
                "raw_root": rr,
                "out_root": s.get("out_root"),
            })
        if out:
            return out
    rr = data.get("raw_root")
    if not rr:
        raise KeyError("config.data.raw_root missing and config.data.stocks empty")
    return [{
        "stock_code": data.get("stock_code"),
        "raw_root": rr,
        "out_root": data.get("out_root"),
    }]


def scan_raw_roots(
    stock_specs: List[dict],
    date_start: str,
    date_end: str,
    sessions_allowed: Optional[List[int]] = None,
) -> List[dict]:
    files: List[dict] = []
    pat = re.compile(r"(?P<code>\d+)_(?P<date>\d{8})_(?P<session>\d+)(?:_.*)?\.csv$", re.IGNORECASE)
    sessions_set = set(int(x) for x in sessions_allowed) if sessions_allowed else None

    for s in stock_specs:
        sc = s.get("stock_code")
        raw_root = s["raw_root"]
        out_root = s.get("out_root")

        if not os.path.isdir(raw_root):
            continue

        for d in sorted(os.listdir(raw_root)):
            if not d.isdigit():
                continue
            if not (date_start <= d <= date_end):
                continue

            dpath = os.path.join(raw_root, d)
            if not os.path.isdir(dpath):
                continue

            for fn in sorted(os.listdir(dpath)):
                if not fn.lower().endswith(".csv"):
                    continue

                m = pat.match(fn)
                if m:
                    sess = int(m.group("session"))
                else:
                    sess = None
                    try:
                        parts = fn.rsplit(".", 1)[0].split("_")
                        if parts and parts[-1].isdigit():
                            sess = int(parts[-1])
                        elif len(parts) >= 2 and parts[-2].isdigit():
                            sess = int(parts[-2])
                    except Exception:
                        sess = None

                if sessions_set is not None and sess is not None and sess not in sessions_set:
                    continue

                files.append({
                    "path": os.path.join(dpath, fn),
                    "stock_code": sc,
                    "raw_root": raw_root,
                    "out_root": out_root,
                    "date": d,
                    "session": sess,
                })

    return files


def build_out_path(
    out_root: str,
    stock_code: Optional[str],
    date: Optional[str],
    in_path: str,
    suffix: str,
    per_stock_root: bool = False,
) -> str:
    base = os.path.basename(in_path)
    stem = base[:-4] if base.lower().endswith(".csv") else base
    out_name = stem + suffix + ".csv"

    if per_stock_root:
        if date:
            odir = os.path.join(out_root, str(date))
        else:
            odir = out_root
    else:
        if stock_code and date:
            odir = os.path.join(out_root, str(stock_code), str(date))
        elif stock_code:
            odir = os.path.join(out_root, str(stock_code))
        else:
            odir = out_root

    os.makedirs(odir, exist_ok=True)
    return os.path.join(odir, out_name)



def process_one_row(
    r: dict,
    cfg: dict,
    project_root: str,
    base_out_root: str,
    suffix: str,
    overwrite: bool,
    s0: dict,
    keep_cols: Optional[List[str]] = None,
    out_cols: Optional[List[str]] = None,
) -> dict:
    in_path = r.get("path")
    sc = r.get("stock_code")
    d = r.get("date")
    sess = r.get("session")

    if not in_path or not os.path.isfile(in_path):
        return {"status": "failed", "error": "missing_input", "path": in_path, "stock_code": sc, "date": d, "session": sess}

    row_out_root = r.get("out_root")
    out_root = resolve_path(project_root, row_out_root) if row_out_root else base_out_root
    per_stock_root = bool(row_out_root)
    out_path = build_out_path(out_root, sc, d, in_path, suffix, per_stock_root=per_stock_root)

    if (not overwrite) and os.path.isfile(out_path):
        return {"status": "skipped_exists", "in_path": in_path, "out_path": out_path, "stock_code": sc, "date": d, "session": sess}

    try:
        # OPTIMIZATION: Use pyarrow engine for fast reading
        try:
            df = pd.read_csv(in_path, engine="pyarrow")
        except Exception:
            # Fallback to standard if pyarrow fails on malformed CSV
            df = pd.read_csv(in_path)

        df2 = enrich_df(df, cfg, stock_code=sc)

        if bool(((s0.get("baseline") or {}).get("enabled", False))):
            df2 = apply_baseline_deviation(
                df2,
                cfg=cfg,
                project_root=project_root,
                stock_code=str(sc),
                date=str(d) if d is not None else None,
                session=sess,
            )

        output_mode = str(s0.get("output_mode", "append"))
        if output_mode not in ("append", "factors_only"):
            raise ValueError(f"stage0.output_mode must be 'append' or 'factors_only', got: {output_mode}")

        if output_mode == "append":
            df_out = df2
            if bool(s0.get("append_rename_as_gpmain", False)):
                prefix = s0.get("prefix", "s0_")
                keep = s0.get("keep_features") or []
                if not keep:
                    raise KeyError("stage0.append_rename_as_gpmain=1 requires non-empty stage0.keep_features")

                keep_cols_local = keep_cols if keep_cols is not None else normalize_keep_features(list(keep), prefix=prefix)

                strict_keep = bool(s0.get("strict_keep_features", True))
                missing_keep = [c for c in keep_cols_local if c not in df_out.columns]
                if missing_keep and strict_keep:
                    raise KeyError(f"missing keep_features columns (append mode): {missing_keep}")
                keep_cols_eff = [c for c in keep_cols_local if c in df_out.columns]

                rename_cfg = s0.get("rename") or {}
                rename_prefix = str(rename_cfg.get("prefix", "gpmain_"))

                if "start_index" in rename_cfg and rename_cfg.get("start_index") is not None:
                    start_index = int(rename_cfg.get("start_index"))
                else:
                    start_index = infer_next_index_from_existing(
                        cols=list(df.columns),
                        rename_prefix=rename_prefix,
                        fallback=get_existing_factor_count(cfg),
                    )

                out_cols_eff = out_cols if out_cols is not None else [
                    f"{rename_prefix}{start_index + i}" for i in range(len(keep_cols_local))
                ]
                if len(keep_cols_eff) != len(keep_cols_local):
                    out_cols_eff = out_cols_eff[:len(keep_cols_eff)]

                rename_map = {keep_cols_eff[i]: out_cols_eff[i] for i in range(len(keep_cols_eff))}
                df_out = df_out.rename(columns=rename_map)

                if bool(s0.get("append_drop_unmapped_s0", False)):
                    unmapped = [c for c in df_out.columns if str(c).startswith(prefix) and c not in keep_cols_eff]
                    if unmapped:
                        df_out = df_out.drop(columns=unmapped)

            # OPTIMIZATION: Use PyArrow for fast writing
            table = pa.Table.from_pandas(df_out)
            pa_csv.write_csv(table, out_path)

            n_rows, n_cols = int(df_out.shape[0]), int(df_out.shape[1])
            return {
                "status": "processed",
                "in_path": in_path, "out_path": out_path, "stock_code": sc, "date": d, "session": sess,
                "n_rows": n_rows, "n_cols": n_cols,
            }

        # factors_only
        keep = s0.get("keep_features") or []
        prefix = s0.get("prefix", "s0_")
        strict_keep = bool(s0.get("strict_keep_features", True))

        if keep_cols is None:
            keep_cols = normalize_keep_features(list(keep), prefix=prefix)

        missing_keep = [c for c in keep_cols if c not in df2.columns]
        if missing_keep and strict_keep:
            raise KeyError(f"missing keep_features columns: {missing_keep}")

        keep_cols_eff = [c for c in keep_cols if c in df2.columns]

        if out_cols is None:
            rename_cfg = s0.get("rename") or {}
            rename_prefix = str(rename_cfg.get("prefix", "gpmain_"))
            start_index = int(rename_cfg.get("start_index", get_existing_factor_count(cfg)))
            out_cols_eff = [f"{rename_prefix}{start_index + i}" for i in range(len(keep_cols_eff))]
        else:
            if len(keep_cols_eff) != len(keep_cols):
                out_cols_eff = out_cols[:len(keep_cols_eff)]
            else:
                out_cols_eff = out_cols

        df_out = df2[keep_cols_eff].copy()
        df_out.columns = out_cols_eff

        for c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
            df_out[c] = ensure_finite(df_out[c], fill=0.0)

        if bool(s0.get("float32_new_cols", True)):
            for c in df_out.columns:
                df_out[c] = df_out[c].astype(np.float32)

        # OPTIMIZATION: Use PyArrow for fast writing
        table = pa.Table.from_pandas(df_out)
        pa_csv.write_csv(table, out_path)

        return {
            "status": "processed",
            "in_path": in_path, "out_path": out_path, "stock_code": sc, "date": d, "session": sess,
            "n_rows": int(df_out.shape[0]), "n_cols": int(df_out.shape[1]),
        }

    except Exception as e:
        return {"status": "failed", "error": str(e), "path": in_path, "stock_code": sc, "date": d, "session": sess}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", type=str, required=True)
    ap.add_argument("--stock-code", type=str, default=None, help="optional filter")
    ap.add_argument("--date", type=str, default=None, help="optional filter YYYYMMDD")
    ap.add_argument("--session", type=int, default=None, help="optional filter session int")
    ap.add_argument("--max-files", type=int, default=None, help="optional override stage0.max_files")
    args = ap.parse_args()

    cfg = load_json(args.config_path)
    project_root = cfg["project"]["project_root"]
    date_start = cfg["data"]["date_range"]["start"]
    date_end = cfg["data"]["date_range"]["end"]
    s0 = cfg.get("stage0", {}) or {}

    enabled = bool(s0.get("enabled", True))
    if not enabled:
        print("[Stage0] disabled in config.stage0.enabled=0")
        return 0

    base_out_root = resolve_path(project_root, s0.get("out_root")) or os.path.join(project_root, "results", "stage0_enriched")
    suffix = str(s0.get("suffix", ""))
    overwrite = bool(s0.get("overwrite", False))
    use_manifest = bool(s0.get("use_stage1_ok_manifest", False))

    sessions_allowed = None
    try:
        sessions_allowed = cfg.get("data", {}).get("file_pattern", {}).get("sessions")
    except Exception:
        sessions_allowed = None

    rows: List[dict] = []
    if use_manifest:
        stage1_cfg = cfg.get("stage1", {}) or {}
        ok_path = resolve_path(project_root, stage1_cfg.get("manifest_ok_path"))
        if (not ok_path) or (not os.path.isfile(ok_path)):
            print(f"[Stage0] WARN: use_stage1_ok_manifest=1 but manifest not found ({ok_path}). Falling back to raw_root scan.")
            use_manifest = False
        else:
            rows = iter_stage1_ok_manifest(ok_path)

    if not use_manifest:
        specs = get_stock_specs(cfg)
        rows = scan_raw_roots(specs, date_start, date_end, sessions_allowed=sessions_allowed)

    rows_all = list(rows)

    def keep(r: dict) -> bool:
        sc, d, sess = infer_stock_date_session_from_manifest_row(r) if "date" in r else (r.get("stock_code"), r.get("date"), r.get("session"))
        if args.stock_code is not None and sc is not None and str(sc) != str(args.stock_code):
            return False
        if args.date is not None and d is not None and str(d) != str(args.date):
            return False
        if args.session is not None and sess is not None and int(sess) != int(args.session):
            return False
        return True

    rows = [r for r in rows if keep(r)]
    max_files = args.max_files if args.max_files is not None else int(s0.get("max_files", 0) or 0)
    if max_files and len(rows) > max_files:
        rows = rows[:max_files]

    # Note: If baseline building is required, using the separate parallel script is recommended.
    # We keep this sequential call for backward compatibility.
    baseline_cfg = (s0.get("baseline") or {})
    if bool(baseline_cfg.get("enabled", False)) and bool(baseline_cfg.get("build_stats_before_enrich", True)):
        print("[Stage0] building baseline stats (sequential fallback)...")
        build_baseline_stats_for_rows(cfg=cfg, project_root=project_root, s0=s0, rows_need=rows, rows_all=rows_all)

    print(f"[Stage0] inputs={len(rows)} base_out_root={base_out_root} use_manifest={int(use_manifest)}")

    summary = {
        "stage": "stage0",
        "config_path": os.path.abspath(args.config_path),
        "base_out_root": base_out_root,
        "suffix": suffix,
        "overwrite": overwrite,
        "use_stage1_ok_manifest": use_manifest,
        "filters": {"stock_code": args.stock_code, "date": args.date, "session": args.session},
        "sessions_allowed": sessions_allowed,
        "processed": 0,
        "skipped_exists": 0,
        "failed": 0,
        "errors": [],
        "outputs": [],
    }

    keep_cols = None
    out_cols = None
    output_mode = str(s0.get("output_mode", "append"))
    if output_mode == "factors_only" or bool(s0.get("append_rename_as_gpmain", False)):
        prefix = s0.get("prefix", "s0_")
        keep = s0.get("keep_features") or []
        keep_cols = normalize_keep_features(list(keep), prefix=prefix)
        strict_keep = bool(s0.get("strict_keep_features", True))
        if strict_keep and not keep_cols:
            raise KeyError("stage0.keep_features is empty under strict_keep_features=1")

        rename_cfg = s0.get("rename") or {}
        rename_prefix = str(rename_cfg.get("prefix", "gpmain_"))

        if output_mode == "factors_only":
            start_index_cfg = rename_cfg.get("start_index", None)
            start_index = get_existing_factor_count(cfg) if start_index_cfg is None else int(start_index_cfg)
            out_cols = [f"{rename_prefix}{start_index + i}" for i in range(len(keep_cols))]

            map_path = s0.get("export_factor_map_path")
            if map_path:
                map_abs = resolve_path(project_root, map_path)
                os.makedirs(os.path.dirname(map_abs), exist_ok=True)
                fmap = build_factor_map(keep_cols, out_cols)
                save_json(map_abs, fmap)
                summary["factor_map_path"] = map_abs
                summary["keep_features_resolved"] = keep_cols
                summary["output_columns"] = out_cols

        elif bool(s0.get("append_rename_as_gpmain", False)):
            start_index_cfg = rename_cfg.get("start_index", None)
            if start_index_cfg is not None:
                start_index = int(start_index_cfg)
                out_cols = [f"{rename_prefix}{start_index + i}" for i in range(len(keep_cols))]

    num_workers = int(s0.get("num_workers", 1) or 1)
    summary["num_workers"] = num_workers

    if num_workers <= 1:
        for r in rows:
            res = process_one_row(
                r=r, cfg=cfg, project_root=project_root, base_out_root=base_out_root,
                suffix=suffix, overwrite=overwrite, s0=s0,
                keep_cols=keep_cols, out_cols=out_cols,
            )
            if res["status"] == "processed":
                summary["processed"] += 1
                summary["outputs"].append({
                    "stock_code": res.get("stock_code"),
                    "date": res.get("date"),
                    "session": res.get("session"),
                })
            elif res["status"] == "skipped_exists":
                summary["skipped_exists"] += 1
            else:
                summary["failed"] += 1
                summary["errors"].append({
                    "path": res.get("path"),
                    "error": res.get("error"),
                })
    else:
        # OPTIMIZATION: ProcessPoolExecutor instead of ThreadPoolExecutor for CPU isolation
        print(f"[Stage0] Using ProcessPoolExecutor with {num_workers} workers.")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = []
            for r in rows:
                futs.append(ex.submit(
                    process_one_row,
                    r, cfg, project_root, base_out_root, suffix, overwrite, s0, keep_cols, out_cols
                ))

            # Simple progress tracker
            total = len(futs)
            done_cnt = 0
            for fut in as_completed(futs):
                res = fut.result()
                done_cnt += 1
                if done_cnt % 50 == 0:
                    print(f"[Stage0] progress: {done_cnt}/{total}")

                if res["status"] == "processed":
                    summary["processed"] += 1
                    summary["outputs"].append({
                        "stock_code": res.get("stock_code"),
                        "date": res.get("date"),
                        "session": res.get("session"),
                    })
                elif res["status"] == "skipped_exists":
                    summary["skipped_exists"] += 1
                else:
                    summary["failed"] += 1
                    summary["errors"].append({
                        "path": res.get("path"),
                        "error": res.get("error"),
                    })

    summary_path = os.path.join(base_out_root, "stage0_summary.json")
    save_json(summary_path, summary)
    print(f"[Stage0] done. processed={summary['processed']} skipped_exists={summary['skipped_exists']} failed={summary['failed']}")
    print(f"[Stage0] summary: {summary_path}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())