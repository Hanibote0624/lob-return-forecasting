#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1.5: Factor scale audit across multiple stocks (combined dataset preparation).

What it does:
- Reads a combined Stage1 manifest (sessions_ok*.jsonl).
- For common factor columns (default: gpmain_0..gpmain_108 based on config),
  estimates robust scale per factor using q-quantile of abs(x): q=0.90 default.
- Checks whether scales are aligned across ALL stocks under:
    (1) raw
    (2) volume-normalized: x / max(EMA(Δacc_volume), dv_floor)
    (3) marketcap-normalized: x / max(price * float_shares, mcap_floor)
- Chooses the best alignment method per factor:
    raw -> volume_norm -> mcap_norm -> none

Outputs (to stage1_5.out_dir):
- factor_scale_report.csv
- factor_norm_map.json  (machine-readable; later Stage3 will use this)

Notes:
- Recommended: use only TRAIN split sessions for this audit to avoid any leakage
  (stage1_5.use_train_split_only=true).
"""

from __future__ import annotations

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# config helpers
# ----------------------------
def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get(d: Dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def expand_glob(path_or_glob: str) -> List[str]:
    if not path_or_glob:
        return []
    if any(ch in path_or_glob for ch in ["*", "?", "["]):
        return sorted(glob.glob(path_or_glob, recursive=True))
    return [path_or_glob]


def read_jsonl_records(jsonl_path: str) -> List[Dict]:
    recs: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def get_stage1_ok_manifest(cfg: Dict) -> Optional[str]:
    # Stage1.5 uses Stage1 combined manifest by default
    return _get(cfg, ["stage1_5", "manifest_ok_path"], None) or _get(cfg, ["stage1", "manifest_ok_path"], None)


def get_stage1_5_out_dir(cfg: Dict) -> str:
    return _get(cfg, ["stage1_5", "out_dir"], "data/stage1_5_scale_audit")


def get_use_train_only(cfg: Dict) -> bool:
    return bool(_get(cfg, ["stage1_5", "use_train_split_only"], True))


def get_train_range(cfg: Dict) -> Tuple[str, str]:
    tr = _get(cfg, ["data", "splits", "train"], {}) or {}
    return str(tr.get("start", "")), str(tr.get("end", ""))


def get_scale_audit_cfg(cfg: Dict) -> Dict:
    return _get(cfg, ["audit", "scale_audit"], {}) or {}


@dataclass
class MarketCols:
    acc_volume: str = "acc_volume"
    bid: str = "bid"
    ask: str = "ask"
    last: str = "last"
    mid: Optional[str] = None


def get_market_cols(cfg: Dict) -> MarketCols:
    cols = _get(cfg, ["data", "market", "columns"], {}) or {}
    return MarketCols(
        acc_volume=cols.get("acc_volume", "acc_volume"),
        bid=cols.get("bid", "bid"),
        ask=cols.get("ask", "ask"),
        last=cols.get("last", "last"),
        mid=cols.get("mid", None),
    )


def expected_factor_cols(cfg: Dict) -> List[str]:
    fcfg = _get(cfg, ["data", "required_columns", "factors"], {}) or {}
    prefix = fcfg.get("prefix", "gpmain_")
    count = int(fcfg.get("count", 109))
    return [f"{prefix}{i}" for i in range(count)]


def get_stock_list(cfg: Dict) -> List[Dict]:
    stocks = _get(cfg, ["data", "stocks"], []) or []
    if not isinstance(stocks, list) or not stocks:
        raise ValueError("config.data.stocks must be a non-empty list")
    return stocks


# ----------------------------
# manifest parsing helpers
# ----------------------------
_RE_DATE = re.compile(r"_(\d{8})_")
_RE_STOCK = re.compile(r"^(\d{6})_")


def infer_stock_code_from_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = _RE_STOCK.match(base)
    return m.group(1) if m else None


def infer_date_from_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = _RE_DATE.search(base)
    return m.group(1) if m else None


def get_record_stock_and_date(rec: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns: (stock_code, date_yyyymmdd, path)
    """
    path = rec.get("path") or rec.get("csv_path") or rec.get("file") or rec.get("filepath")
    if not path:
        return None, None, None

    stock = rec.get("stock_code") or rec.get("stock") or infer_stock_code_from_path(path)
    date = rec.get("date") or rec.get("yyyymmdd") or rec.get("YYYYMMDD") or infer_date_from_path(path)
    if date is not None:
        date = str(date)
    return (str(stock) if stock is not None else None), date, str(path)


def filter_records_by_train(records: List[Dict], train_start: str, train_end: str) -> List[Dict]:
    if not train_start or not train_end:
        return records
    out = []
    for rec in records:
        stock, date, path = get_record_stock_and_date(rec)
        if date is None:
            continue
        if train_start <= date <= train_end:
            out.append(rec)
    return out


def group_paths_by_stock(records: List[Dict], allowed_stocks: List[str]) -> Dict[str, List[str]]:
    allowed = set(allowed_stocks)
    out: Dict[str, List[str]] = {s: [] for s in allowed_stocks}
    for rec in records:
        stock, _, path = get_record_stock_and_date(rec)
        if not stock or not path:
            continue
        if stock not in allowed:
            continue
        out[stock].append(path)
    # dedupe keep order
    for k in list(out.keys()):
        out[k] = list(dict.fromkeys(out[k]))
    return out


# ----------------------------
# normalization denominators
# ----------------------------
def choose_price_array(df: pd.DataFrame, bid_col: str, ask_col: str, last_col: str, mid_col: Optional[str]) -> np.ndarray:
    if mid_col and mid_col in df.columns:
        return df[mid_col].to_numpy(dtype=np.float64, copy=False)
    if bid_col in df.columns and ask_col in df.columns:
        bid = df[bid_col].to_numpy(dtype=np.float64, copy=False)
        ask = df[ask_col].to_numpy(dtype=np.float64, copy=False)
        return 0.5 * (bid + ask)
    if last_col in df.columns:
        return df[last_col].to_numpy(dtype=np.float64, copy=False)
    raise ValueError(f"Cannot find price columns: mid={mid_col}, bid={bid_col}, ask={ask_col}, last={last_col}")


def ema_with_carry(dv: np.ndarray, span: int, ema_last: Optional[float]) -> Tuple[np.ndarray, float]:
    if span <= 1:
        out = dv.astype(np.float64, copy=False)
        last = float(out[-1]) if out.size > 0 else (float(ema_last) if ema_last is not None else 0.0)
        return out, last

    dv = dv.astype(np.float64, copy=False)
    if ema_last is None:
        e = pd.Series(dv).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)
        return e, float(e[-1]) if e.size > 0 else 0.0

    dv_ext = np.concatenate([[float(ema_last)], dv])
    e_ext = pd.Series(dv_ext).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)
    e = e_ext[1:]
    return e, float(e[-1]) if e.size > 0 else float(ema_last)


# ----------------------------
# reservoir sampler
# ----------------------------
@dataclass
class Reservoir:
    k: int
    n_seen: int
    buf: np.ndarray  # [k, F]


def reservoir_init(k: int, nf: int) -> Reservoir:
    return Reservoir(k=k, n_seen=0, buf=np.zeros((k, nf), dtype=np.float32))


def reservoir_update(res: Reservoir, rows: np.ndarray, rng: np.random.Generator) -> None:
    if rows is None or rows.size == 0:
        return
    m = rows.shape[0]
    for i in range(m):
        res.n_seen += 1
        if res.n_seen <= res.k:
            res.buf[res.n_seen - 1] = rows[i]
        else:
            j = int(rng.integers(0, res.n_seen))
            if j < res.k:
                res.buf[j] = rows[i]


@dataclass
class SampledData:
    raw: np.ndarray
    vol: np.ndarray
    mcap: Optional[np.ndarray]
    n_rows_seen: int
    n_rows_sampled: int


def stream_sample_from_csvs(
    csv_paths: List[str],
    factor_cols: List[str],
    market_cols: MarketCols,
    float_shares: Optional[float],
    chunksize: int,
    per_chunk_sample: int,
    sample_max: int,
    dv_floor: float,
    ema_span: int,
    mcap_floor: float,
    seed: int,
) -> SampledData:
    rng = np.random.default_rng(seed)

    nf = len(factor_cols)
    res_raw = reservoir_init(sample_max, nf)
    res_vol = reservoir_init(sample_max, nf)
    res_mcap = reservoir_init(sample_max, nf) if float_shares is not None else None

    base_cols = [market_cols.acc_volume]
    for c in [market_cols.bid, market_cols.ask, market_cols.last]:
        if c and c not in base_cols:
            base_cols.append(c)
    if market_cols.mid and market_cols.mid not in base_cols:
        base_cols.append(market_cols.mid)
    usecols = list(dict.fromkeys(base_cols + factor_cols))

    n_rows_seen_total = 0
    n_rows_sampled_total = 0

    for fp in csv_paths:
        if not os.path.exists(fp):
            continue

        prev_acc = None
        ema_last = None

        for df in pd.read_csv(fp, usecols=lambda c: c in usecols, chunksize=chunksize):
            n = len(df)
            if n == 0:
                continue
            n_rows_seen_total += n

            X = df[factor_cols].to_numpy(dtype=np.float32, copy=False)
            X = X.astype(np.float32, copy=False)
            X[~np.isfinite(X)] = np.nan

            if market_cols.acc_volume not in df.columns:
                raise ValueError(f"Missing acc_volume_col={market_cols.acc_volume} in {fp}")

            acc = df[market_cols.acc_volume].to_numpy(dtype=np.float64, copy=False)
            dv = np.empty_like(acc, dtype=np.float64)
            if prev_acc is None:
                dv[0] = 0.0
            else:
                dv[0] = acc[0] - prev_acc
            dv[1:] = acc[1:] - acc[:-1]

            dv[~np.isfinite(dv)] = 0.0
            dv[dv < 0] = 0.0
            prev_acc = float(acc[-1])

            dv_ema, ema_last = ema_with_carry(dv, span=ema_span, ema_last=ema_last)
            denom_vol = np.maximum(dv_ema, dv_floor)

            denom_mcap = None
            if float_shares is not None:
                price = choose_price_array(df, market_cols.bid, market_cols.ask, market_cols.last, market_cols.mid)
                denom_mcap = np.maximum(price * float_shares, mcap_floor)

            k = min(per_chunk_sample, n)
            if k <= 0:
                continue
            idx = rng.choice(n, size=k, replace=False)
            n_rows_sampled_total += k

            Xs = X[idx, :]
            reservoir_update(res_raw, Xs, rng)

            vol_s = Xs / denom_vol[idx, None].astype(np.float32)
            reservoir_update(res_vol, vol_s, rng)

            if res_mcap is not None and denom_mcap is not None:
                mcap_s = Xs / denom_mcap[idx, None].astype(np.float32)
                reservoir_update(res_mcap, mcap_s, rng)

    def final_buf(res: Reservoir) -> np.ndarray:
        n = min(res.n_seen, res.k)
        return res.buf[:n].copy()

    raw = final_buf(res_raw)
    vol = final_buf(res_vol)
    mcap = final_buf(res_mcap) if res_mcap is not None else None

    return SampledData(raw=raw, vol=vol, mcap=mcap, n_rows_seen=n_rows_seen_total, n_rows_sampled=n_rows_sampled_total)


# ----------------------------
# scale & alignment
# ----------------------------
def robust_scale_qabs(X: np.ndarray, q: float) -> np.ndarray:
    if X is None or X.size == 0:
        return np.array([], dtype=np.float64)
    A = np.abs(X.astype(np.float64, copy=False))
    A[~np.isfinite(A)] = np.nan
    return np.nanquantile(A, q, axis=0)


def ratio_max_over_min(scales: np.ndarray, tiny: float = 1e-12) -> np.ndarray:
    s = np.maximum(scales, tiny)
    return np.nanmax(s, axis=0) / np.nanmin(s, axis=0)


def dispersion_logratio(scales: np.ndarray, tiny: float = 1e-12) -> np.ndarray:
    r = ratio_max_over_min(scales, tiny=tiny)
    return np.abs(np.log(np.maximum(r, tiny)))


def choose_best_method(
    r_raw: np.ndarray,
    r_vol: np.ndarray,
    r_mcap: Optional[np.ndarray],
    ratio_tol: float,
    disp_raw: np.ndarray,
    disp_vol: np.ndarray,
    disp_mcap: Optional[np.ndarray],
) -> np.ndarray:
    n = r_raw.shape[0]
    best = np.array(["none"] * n, dtype=object)

    same_raw = r_raw <= ratio_tol
    same_vol = r_vol <= ratio_tol
    same_mcap = (r_mcap <= ratio_tol) if r_mcap is not None else np.zeros(n, dtype=bool)

    best[same_raw] = "raw"

    for i in range(n):
        if best[i] == "raw":
            continue
        candidates = []
        if same_vol[i]:
            candidates.append(("volume_norm", float(disp_vol[i])))
        if r_mcap is not None and disp_mcap is not None and same_mcap[i]:
            candidates.append(("mcap_norm", float(disp_mcap[i])))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            best[i] = candidates[0][0]
    return best


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, type=str)
    ap.add_argument("--manifest-ok", default=None, type=str, help="override sessions_ok jsonl or glob")
    ap.add_argument("--out-dir", default=None, type=str)

    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--ratio-tol", type=float, default=None)

    ap.add_argument("--chunksize", type=int, default=None)
    ap.add_argument("--per-chunk-sample", type=int, default=None)
    ap.add_argument("--sample-max", type=int, default=None)

    ap.add_argument("--dv-floor", type=float, default=None)
    ap.add_argument("--ema-span", type=int, default=None)
    ap.add_argument("--mcap-floor", type=float, default=None)

    args = ap.parse_args()

    cfg = load_config(args.config_path)
    stocks = get_stock_list(cfg)
    stock_codes = [str(s["stock_code"]) for s in stocks]

    manifest_ok = args.manifest_ok or get_stage1_ok_manifest(cfg)
    if not manifest_ok:
        raise ValueError("No manifest_ok provided. Set stage1.manifest_ok_path or stage1_5.manifest_ok_path, or pass --manifest-ok")

    out_dir = args.out_dir or get_stage1_5_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    audit = get_scale_audit_cfg(cfg)
    q = args.q if args.q is not None else float(audit.get("q", 0.90))
    ratio_tol = args.ratio_tol if args.ratio_tol is not None else float(audit.get("ratio_tol", 2.0))
    chunksize = args.chunksize if args.chunksize is not None else int(audit.get("chunksize", 200_000))
    per_chunk_sample = args.per_chunk_sample if args.per_chunk_sample is not None else int(audit.get("per_chunk_sample", 5000))
    sample_max = args.sample_max if args.sample_max is not None else int(audit.get("sample_max", 200_000))
    dv_floor = args.dv_floor if args.dv_floor is not None else float(audit.get("dv_floor", 1.0))
    ema_span = args.ema_span if args.ema_span is not None else int(audit.get("ema_span", 200))
    mcap_floor = args.mcap_floor if args.mcap_floor is not None else float(audit.get("mcap_floor", 1.0))

    market_cols = get_market_cols(cfg)
    use_train_only = get_use_train_only(cfg)
    train_start, train_end = get_train_range(cfg)

    # read manifest records (support glob)
    recs: List[Dict] = []
    for p in expand_glob(manifest_ok):
        recs.extend(read_jsonl_records(p))

    if use_train_only:
        recs = filter_records_by_train(recs, train_start, train_end)

    paths_by_stock = group_paths_by_stock(recs, stock_codes)
    for sc in stock_codes:
        if not paths_by_stock.get(sc):
            raise ValueError(f"No CSV paths found for stock {sc} in manifest_ok (after filtering).")

    # float shares per stock (preferred) with fallback
    float_by_stock: Dict[str, Optional[float]] = {}
    for s in stocks:
        sc = str(s["stock_code"])
        v = s.get("float_shares", None)
        float_by_stock[sc] = None if v is None else float(v)

    have_mcap = all(float_by_stock[sc] is not None for sc in stock_codes)
    if not have_mcap:
        print("[WARN] float_shares missing for at least one stock -> mcap_norm will be skipped.")

    # determine common factor columns
    expected = expected_factor_cols(cfg)
    common = set(expected)
    for sc in stock_codes:
        head = pd.read_csv(paths_by_stock[sc][0], nrows=1)
        cols = set(head.columns.tolist())
        common &= set([c for c in expected if c in cols])

    factor_cols = [c for c in expected if c in common]
    if not factor_cols:
        raise ValueError("No common factor columns found across stocks. Check your CSV headers and config.data.required_columns.factors.")

    print(f"[INFO] stocks={stock_codes}, common_factors={len(factor_cols)}")
    print(f"[INFO] train_only={use_train_only} train_range=[{train_start},{train_end}]")

    # sample per stock
    sampled: Dict[str, SampledData] = {}
    for i, sc in enumerate(stock_codes):
        sd = stream_sample_from_csvs(
            paths_by_stock[sc],
            factor_cols,
            market_cols,
            float_by_stock[sc] if have_mcap else None,
            chunksize,
            per_chunk_sample,
            sample_max,
            dv_floor,
            ema_span,
            mcap_floor,
            seed=42 + i * 97,
        )
        sampled[sc] = sd
        print(f"[INFO] {sc}: seen_rows={sd.n_rows_seen:,} sampled_rows={sd.n_rows_sampled:,} reservoir={sd.raw.shape[0]:,}")

    # compute scales: [S, F]
    S = len(stock_codes)
    F = len(factor_cols)
    scales_raw = np.zeros((S, F), dtype=np.float64)
    scales_vol = np.zeros((S, F), dtype=np.float64)
    scales_mcap = np.zeros((S, F), dtype=np.float64) if have_mcap else None

    for si, sc in enumerate(stock_codes):
        sd = sampled[sc]
        scales_raw[si] = robust_scale_qabs(sd.raw, q=q)
        scales_vol[si] = robust_scale_qabs(sd.vol, q=q)
        if have_mcap and sd.mcap is not None and scales_mcap is not None:
            scales_mcap[si] = robust_scale_qabs(sd.mcap, q=q)

    r_raw = ratio_max_over_min(scales_raw)
    r_vol = ratio_max_over_min(scales_vol)
    r_mcap = ratio_max_over_min(scales_mcap) if have_mcap and scales_mcap is not None else None

    same_raw = r_raw <= ratio_tol
    same_vol = r_vol <= ratio_tol
    same_mcap = (r_mcap <= ratio_tol) if r_mcap is not None else np.zeros(F, dtype=bool)

    disp_raw = dispersion_logratio(scales_raw)
    disp_vol = dispersion_logratio(scales_vol)
    disp_mcap = dispersion_logratio(scales_mcap) if r_mcap is not None and scales_mcap is not None else None

    best = choose_best_method(r_raw, r_vol, r_mcap, ratio_tol, disp_raw, disp_vol, disp_mcap)

    out_csv = os.path.join(out_dir, "factor_scale_report.csv")
    out_map = os.path.join(out_dir, "factor_norm_map.json")

    # Build report dataframe (wide: per-stock scale columns)
    df = pd.DataFrame({"factor": factor_cols})
    for si, sc in enumerate(stock_codes):
        df[f"q{int(q*100)}_abs_raw_{sc}"] = scales_raw[si]
        df[f"q{int(q*100)}_abs_vol_{sc}"] = scales_vol[si]
        if have_mcap and scales_mcap is not None:
            df[f"q{int(q*100)}_abs_mcap_{sc}"] = scales_mcap[si]

    df["ratio_raw_max_over_min"] = r_raw
    df["ratio_vol_max_over_min"] = r_vol
    df["ratio_mcap_max_over_min"] = r_mcap if r_mcap is not None else np.nan

    df["same_raw"] = same_raw
    df["same_vol"] = same_vol
    df["same_mcap"] = same_mcap

    df["disp_log_raw"] = disp_raw
    df["disp_log_vol"] = disp_vol
    df["disp_log_mcap"] = disp_mcap if disp_mcap is not None else np.nan

    df["best_alignment"] = best
    df.to_csv(out_csv, index=False)
    print(f"[OK] report saved -> {out_csv}")

    raw_list = df.loc[df["best_alignment"] == "raw", "factor"].tolist()
    vol_list = df.loc[df["best_alignment"] == "volume_norm", "factor"].tolist()
    mcap_list = df.loc[df["best_alignment"] == "mcap_norm", "factor"].tolist()
    none_list = df.loc[df["best_alignment"] == "none", "factor"].tolist()

    norm_map = {
        "meta": {
            "stocks": stock_codes,
            "q_abs": q,
            "ratio_tol": ratio_tol,
            "train_only": bool(use_train_only),
            "train_range": [train_start, train_end],
            "chunksize": chunksize,
            "per_chunk_sample": per_chunk_sample,
            "sample_max": sample_max,
            "ema_span": ema_span,
            "dv_floor": dv_floor,
            "mcap_floor": mcap_floor,
            "have_mcap": bool(have_mcap),
        },
        "groups": {
            "raw": raw_list,
            "volume_norm": vol_list,
            "mcap_norm": mcap_list,
            "none": none_list,
        },
    }
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(norm_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] norm map saved -> {out_map}")

    print(f"\n[SUMMARY] raw={len(raw_list)} vol={len(vol_list)} mcap={len(mcap_list)} none={len(none_list)}")


if __name__ == "__main__":
    main()