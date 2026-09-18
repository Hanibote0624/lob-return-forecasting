#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2: Build regression labels for active horizon (min_sec, max_sec), session-by-session.

Key design:
- NEVER cross session: each CSV is a session unit.
- Train-only stats: q90_abs, q99_abs computed ONLY from train sessions (raw labels).
- Two-step outputs to avoid re-reading huge CSVs:
  1) Write raw label files for ALL sessions: r_raw + masks + mid + t_sec
  2) Compute train stats from raw label files
  3) Write final label files for ALL sessions: r_raw + r_scaled + same masks (no CSV read)

Label definition (per index i):
  - mid_i = (ask1_i + bid1_i)/2 if available else LastPrice
  - F(i) = {j : t[j]-t[i] in [min_sec, max_sec]}
  - avg_mid = mean(mid[j] for j in F(i), excluding NaNs)
  - r_raw = avg_mid / mid_i - 1



Additional label mode (no interface change):
  - cfg.stage2.label_mode == 'tick_mid_delta'
    label (r_raw) per index i:
      fut = i + cfg.stage2.tick_offset (default 10)
      r_raw = (mid[fut] - mid[i]) * cfg.stage2.label_multiplier (default 200.0)
    validity requires fut < N and both mid[i], mid[fut] finite and >0.
    In this mode, Stage2 will ignore CLI --scale (forces scale=1.0) so that r_scaled==r_raw.

Notes:
- Handles NaN mids using prefix sums of (mid, valid_mask).
- Optional fix for non-monotonic timestamps (disabled by default): can sort by t_sec,
  but that would require Stage3 to follow same order. So default is strict error.
"""

import argparse
import json
import logging
import re
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required for this script. Please install pandas.", file=sys.stderr)
    raise


# -----------------------------
# IO helpers
# -----------------------------

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def yyyymmdd_in_range(d: str, start: str, end: str) -> bool:
    return start <= d <= end


def infer_split(date: str, splits: dict) -> Optional[str]:
    for k in ["train", "val", "test"]:
        s = splits.get(k, {})
        if s.get("start") and s.get("end") and yyyymmdd_in_range(date, s["start"], s["end"]):
            return k
    return None


def get_row_stock_code(row: dict) -> str:
    """Resolve stock_code for a manifest row (supports combined multi-stock manifests).

    Priority:
      1) row['stock_code'] if present
      2) parse from basename of row['path'] (e.g. SAMPLE_20250101_1.csv)
      3) 'unknown'
    """
    sc = row.get("stock_code")
    if sc is not None and str(sc).strip():
        return str(sc)
    path = str(row.get("path") or "")
    base = os.path.basename(path)
    m = re.match(r"(?P<code>\d+)_\d{8}_[12]\.csv$", base, flags=re.IGNORECASE)
    if m:
        return m.group("code")
    return "unknown"


# -----------------------------
# Core math: timestamp -> seconds
# -----------------------------

def hhmmssmmm_to_seconds_vec(ts: np.ndarray) -> np.ndarray:
    """
    ts: float or int array with HHMMSSmmm (9 digits)
    returns t_sec float64 array, NaN for invalid.
    """
    out = np.full(ts.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(ts) & (ts > 0)
    if not np.any(mask):
        return out
    x = ts[mask].astype(np.int64, copy=False)
    hh = x // 10_000_000
    mm = (x // 100_000) % 100
    ss = (x // 1_000) % 100
    ms = x % 1_000
    out[mask] = hh * 3600.0 + mm * 60.0 + ss + ms / 1000.0
    return out


def prefix_range_sum(ps: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    ps: prefix sum (float64/int64), shape [N]
    lo/hi: int64 arrays shape [N], where hi may be < lo for invalid
    Returns sum over [lo..hi] inclusive for each i (invalid positions yield 0).
    """
    res = np.zeros_like(ps, dtype=ps.dtype)
    valid = (hi >= lo) & (hi >= 0) & (lo >= 0)
    if not np.any(valid):
        return res
    lo_v = lo[valid]
    hi_v = hi[valid]
    s_hi = ps[hi_v]
    s_lo_1 = np.where(lo_v > 0, ps[lo_v - 1], 0)
    res_v = s_hi - s_lo_1
    res[valid] = res_v
    return res


# -----------------------------
# Worker
# -----------------------------

@dataclass
class Task:
    path: str
    date: str
    session: int
    stock_code: str
    split: str
    horizon_id: str
    min_sec: float
    max_sec: float
    cfg: dict


def compute_mid(bid1: np.ndarray, ask1: np.ndarray, lastp: np.ndarray, prefer_bidask: bool = True) -> np.ndarray:
    """Compute mid price vector.

    If prefer_bidask:
      - use (bid1+ask1)/2 where bid/ask are both finite and >0
      - otherwise fallback to lastp
    Else:
      - always use lastp
    """
    if not prefer_bidask:
        mid = lastp
    else:
        ba = (bid1 + ask1) * 0.5
        valid_ba = np.isfinite(bid1) & np.isfinite(ask1) & (bid1 > 0) & (ask1 > 0)
        mid = np.where(valid_ba, ba, lastp)
    mid = np.round(mid.astype(np.float64, copy=False), 3)
    return mid


def worker_build_raw(task: Task) -> dict:
    cfg = task.cfg
    req = cfg["data"]["required_columns"]
    ts_col = req["timestamp"]
    mid_cfg = req["mid_price"]
    bid1_col = mid_cfg["bid1"]
    ask1_col = mid_cfg["ask1"]
    last_col = mid_cfg["fallback_last"]
    prefer_bidask = bool(mid_cfg.get("prefer_bidask", True))

    # outputs
    project_root = cfg["project"]["project_root"]
    raw_dir = os.path.join(project_root, cfg["stage2"]["labels_raw_dir"], task.horizon_id, task.date)
    ensure_dir(raw_dir)
    out_raw_path = os.path.join(raw_dir, f"{task.stock_code}_{task.date}_{task.session}.npz")

    t0 = time.time()
    try:
        usecols = [ts_col, bid1_col, ask1_col, last_col]
        df = pd.read_csv(task.path, usecols=usecols, engine="c")
    except Exception as e:
        return {"ok": False, "path": task.path, "error": f"read_csv_failed: {e}"}

    try:
        ts = df[ts_col].to_numpy(dtype=np.float64, copy=False)
        bid1 = df[bid1_col].to_numpy(dtype=np.float64, copy=False)
        ask1 = df[ask1_col].to_numpy(dtype=np.float64, copy=False)
        lastp = df[last_col].to_numpy(dtype=np.float64, copy=False)
    except Exception as e:
        return {"ok": False, "path": task.path, "error": f"extract_cols_failed: {e}"}

    t_sec = hhmmssmmm_to_seconds_vec(ts)

    # strict monotonic check (no reorder by default)
    if cfg["stage2"].get("strict_monotonic", True):
        if (not np.all(np.isfinite(t_sec))) or np.any(np.diff(t_sec) < 0):
            return {"ok": False, "path": task.path, "error": "timestamp_not_monotonic_or_nan (strict_monotonic=1)"}

    # mid
    mid = compute_mid(bid1, ask1, lastp, prefer_bidask=prefer_bidask)
    mid_valid = np.isfinite(mid) & (mid > 0) & np.isfinite(t_sec)

    # label mode
    label_mode = str(cfg.get('stage2', {}).get('label_mode', 'time_window_avg_mid')).strip()
    if label_mode in ('tick_mid_delta', 'tick_mid_delta_x200'):
        tick_offset = int(cfg.get('stage2', {}).get('tick_offset', 10))
        label_mult = float(cfg.get('stage2', {}).get('label_multiplier', 200.0))
        N = mid.shape[0]
        fut_idx = (np.arange(N, dtype=np.int64) + tick_offset)
        fut_ok = fut_idx < N
        mid_fut = np.full((N,), np.nan, dtype=np.float64)
        if np.any(fut_ok):
            mid_fut[fut_ok] = mid[fut_idx[fut_ok]]
        r_raw = (mid_fut - mid) * label_mult
        is_valid = mid_valid & fut_ok & np.isfinite(mid_fut) & (mid_fut > 0)
        r_raw = r_raw.astype(np.float32)
        r_raw[~is_valid] = np.nan

        # lo/hi keep interface: point to the single future tick index
        lo = np.where(fut_ok, fut_idx, -1).astype(np.int64)
        hi = lo.copy()

        # optional: session_rules max_window_span_seconds (avoid giant gaps)
        max_span = float(cfg['data']['session_rules'].get('max_window_span_seconds', 0.0))
        if max_span and max_span > 0:
            # compare current index time to future tick time
            vmask = is_valid & (lo >= 0) & (lo < N)
            if np.any(vmask):
                idx = np.flatnonzero(vmask)
                span_v = t_sec[lo[idx]] - t_sec[idx]
                bad_v = span_v > (max_span * 10.0)
                if np.any(bad_v):
                    bad_idx = idx[bad_v]
                    is_valid[bad_idx] = False
                    r_raw[bad_idx] = np.nan

        # train sample for quantiles (abs raw)
        sample_n = int(cfg.get('stage2', {}).get('quantile_sample_per_session', 5000))
        samples = np.empty((0,), dtype=np.float32)
        if task.split == 'train' and sample_n > 0:
            rr = r_raw[is_valid]
            if rr.size > 0:
                aa = np.abs(rr)
                if aa.size <= sample_n:
                    samples = aa.astype(np.float32, copy=False)
                else:
                    stride = max(1, aa.size // sample_n)
                    samples = aa[::stride][:sample_n].astype(np.float32, copy=False)

        # write raw label file
        try:
            np.savez(
                out_raw_path,
                t_sec=t_sec.astype(np.float32),
                mid=np.round(mid.astype(np.float32), 3),
                r_raw=r_raw,
                is_valid=is_valid.astype(np.uint8),
                lo=lo.astype(np.int32),
                hi=hi.astype(np.int32),
            )
        except Exception as e:
            return {'ok': False, 'path': task.path, 'error': f'save_raw_failed: {e}'}

        dt = time.time() - t0
        return {
            'ok': True,
            'path': task.path,
            'stock_code': task.stock_code,
            'date': task.date,
            'session': task.session,
            'split': task.split,
            'raw_out': out_raw_path,
            'n': int(t_sec.shape[0]),
            'valid': int(is_valid.sum()),
            'seconds': dt,
            'train_samples': samples
        }

    # future window indices
    # (searchsorted requires t_sec nondecreasing; we assume strict)
    tmin = t_sec + task.min_sec
    tmax = t_sec + task.max_sec

    # for NaN t_sec, searchsorted gives undefined; set invalid later
    # if np.all(np.isfinite(t_sec)):
    #     print(task.path, 'good')
    t_sec_for_search = np.where(np.isfinite(t_sec), t_sec, -1.0)
    lo = np.searchsorted(t_sec_for_search, tmin, side="left").astype(np.int64)
    hi_excl = np.searchsorted(t_sec_for_search, tmax, side="right").astype(np.int64)
    hi = hi_excl - 1

    # prefix sums of mid (nan->0) and valid counts
    mid0 = np.nan_to_num(mid, nan=0.0).astype(np.float64, copy=False)
    vm = np.isfinite(mid).astype(np.int64, copy=False)
    ps_mid = np.cumsum(mid0)
    ps_cnt = np.cumsum(vm)

    sum_mid = prefix_range_sum(ps_mid, lo, hi).astype(np.float64, copy=False)
    sum_cnt = prefix_range_sum(ps_cnt, lo, hi).astype(np.int64, copy=False)
    avg_mid = np.where(sum_cnt > 0, sum_mid / np.maximum(sum_cnt, 1), np.nan)

    r_raw = avg_mid / mid - 1.0

    #### NEW 中间label赋值从nan变为0###
    is_valid = tmin <= t_sec[-1]
    r_raw = r_raw.astype(np.float32)
    r_raw = np.where(sum_cnt==0, 0, r_raw)
    r_raw[~is_valid] = np.nan
    ###################################

    ####OLD####
    # # final validity: current mid valid + future cnt>0 + index valid
    # is_valid = mid_valid & (hi >= lo) & (sum_cnt > 0)

    # # cleanup invalid
    # r_raw = r_raw.astype(np.float32)
    # r_raw[~is_valid] = np.nan
    ####OLD####

    # optional: session_rules max_window_span_seconds (avoid giant gaps)
    max_span = float(cfg["data"]["session_rules"].get("max_window_span_seconds", 0.0))
    if max_span and max_span > 0:
        N = t_sec.shape[0]
        # 只对索引合法且 is_valid 的位置计算 span，避免 np.where 的“提前求值”越界
        vmask = is_valid & (lo >= 0) & (hi >= 0) & (lo < N) & (hi < N)

        bad_span = np.zeros_like(is_valid, dtype=bool)
        if np.any(vmask):
            idx = np.flatnonzero(vmask)
            span_v = t_sec[hi[idx]] - t_sec[lo[idx]]
            bad_v = span_v > (max_span * 10.0)
            if np.any(bad_v):
                bad_span[idx[bad_v]] = True

        if np.any(bad_span):
            is_valid[bad_span] = False
            r_raw[bad_span] = np.nan


    # train sample for quantiles (abs raw)
    sample_n = int(cfg["stage2"].get("quantile_sample_per_session", 5000))
    samples = np.empty((0,), dtype=np.float32)
    if task.split == "train" and sample_n > 0:
        rr = r_raw[is_valid]
        if rr.size > 0:
            aa = np.abs(rr)
            if aa.size <= sample_n:
                samples = aa.astype(np.float32, copy=False)
            else:
                stride = max(1, aa.size // sample_n)
                samples = aa[::stride][:sample_n].astype(np.float32, copy=False)

    # write raw label file
    try:
        np.savez(
            out_raw_path,
            t_sec=t_sec.astype(np.float32),
            mid=np.round(mid.astype(np.float32), 3),
            r_raw=r_raw,
            is_valid=is_valid.astype(np.uint8),
            lo=lo.astype(np.int32),
            hi=hi.astype(np.int32),
        )
    except Exception as e:
        return {"ok": False, "path": task.path, "error": f"save_raw_failed: {e}"}

    dt = time.time() - t0
    return {
        "ok": True,
        "path": task.path,
        "stock_code": task.stock_code,
        "date": task.date,
        "session": task.session,
        "split": task.split,
        "raw_out": out_raw_path,
        "n": int(t_sec.shape[0]),
        "valid": int(is_valid.sum()),
        "seconds": dt,
        "train_samples": samples  # small array
    }


def worker_scale_from_raw(args: tuple) -> dict:
    """
    args = (raw_path, out_path, scale, clip_abs_scaled)
    """
    raw_path, out_path, scale, clip_abs_scaled = args
    try:
        z = np.load(raw_path)
        r_raw = z["r_raw"].astype(np.float32, copy=False)
        is_valid = z["is_valid"].astype(np.uint8, copy=False).astype(bool)

        r_scaled = (r_raw.astype(np.float64) * scale).astype(np.float32)
        r_scaled[~is_valid] = np.nan

        np.savez(
            out_path,
            t_sec=z["t_sec"].astype(np.float32, copy=False),
            mid=z["mid"].astype(np.float32, copy=False),
            r_raw=r_raw,
            r_scaled=r_scaled,
            is_valid=z["is_valid"].astype(np.uint8, copy=False),
            lo=z["lo"].astype(np.int32, copy=False),
            hi=z["hi"].astype(np.int32, copy=False),
        )
        return {"ok": True, "raw_path": raw_path, "out_path": out_path}
    except Exception as e:
        return {"ok": False, "raw_path": raw_path, "error": str(e)}


# -----------------------------
# Main
# -----------------------------

def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", type=str, required=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--only-scale", type=int, default=0, help="1=skip CSV pass, only scale from existing raw label files")
    ap.add_argument("--scale", type=int, default=1)
    args = ap.parse_args()

    cfg = load_json(args.config_path)
    project_root = cfg["project"]["project_root"]
    manifests_dir = os.path.join(project_root, cfg["paths"]["manifests_dir"])

    # Stage1 ok manifest
    ok_manifest = cfg["stage1"]["manifest_ok_path"]
    ok_manifest_path = ok_manifest if os.path.isabs(ok_manifest) else os.path.join(project_root, ok_manifest)
    rows = read_jsonl(ok_manifest_path)
    ok_rows = [r for r in rows if r.get("ok")]

    if not ok_rows:
        logging.error(f"No ok sessions in manifest: {ok_manifest_path}")
        sys.exit(2)

    horizon_id = cfg["horizons"]["active_horizon_id"]
    hdef = cfg["horizons"]["definitions"][horizon_id]
    min_sec = float(hdef["min_seconds"])
    max_sec = float(hdef["max_seconds"])
    assert max_sec > min_sec > 0

    splits = cfg["data"]["splits"]

    # Legacy single-stock config may set data.stock_code.
    # In combined multi-stock mode, each manifest row should carry its own stock_code.
    cfg_stock_code = cfg["data"].get("stock_code", None)
    cfg_stock_code = str(cfg_stock_code) if cfg_stock_code is not None else None

    def resolve_stock_code(row: dict) -> str:
        return cfg_stock_code if cfg_stock_code is not None else get_row_stock_code(row)

    # Stage2 paths
    labels_raw_base = os.path.join(project_root, cfg["stage2"]["labels_raw_dir"], horizon_id)
    labels_final_base = os.path.join(project_root, cfg["stage2"]["labels_final_dir"], horizon_id)
    stats_dir = os.path.join(project_root, cfg["paths"]["stats_dir"])
    ensure_dir(labels_raw_base)
    ensure_dir(labels_final_base)
    ensure_dir(stats_dir)

    # workers
    num_workers = args.num_workers if args.num_workers > 0 else int(cfg["stage2"].get("num_workers", 16))
    num_workers = max(1, min(num_workers, 128))
    label_mode_log = str(cfg.get('stage2', {}).get('label_mode', 'time_window_avg_mid')).strip()
    tick_off_log = cfg.get('stage2', {}).get('tick_offset', None)
    logging.info(f"[Stage2] horizon={horizon_id} ({min_sec}-{max_sec}s), label_mode={label_mode_log}, tick_offset={tick_off_log}, workers={num_workers}, only_scale={bool(args.only_scale)}")

    index_rows = []

    # 1) CSV pass -> raw label files
    train_samples_all = []
    raw_paths = []
    if not bool(args.only_scale):
        tasks = []
        for r in ok_rows:
            date = r["date"]
            session = int(r["session"])
            split = infer_split(date, splits) or "out_of_split"
            if split == "out_of_split":
                continue
            path = r["path"]
            tasks.append(Task(
                path=path,
                date=date,
                session=session,
                stock_code=resolve_stock_code(r),
                split=split,
                horizon_id=horizon_id,
                min_sec=min_sec,
                max_sec=max_sec,
                cfg=cfg
            ))

        if not tasks:
            logging.error("No tasks after split filtering. Check cfg.data.splits.")
            sys.exit(3)

        t0 = time.time()
        with Pool(processes=num_workers) as pool:
            results = list(pool.map(worker_build_raw, tasks))
        dt = time.time() - t0

        bad = [x for x in results if not x.get("ok")]
        good = [x for x in results if x.get("ok")]

        logging.info(f"[Stage2] raw pass done. good={len(good)} bad={len(bad)} in {dt:.1f}s")
        if bad:
            # print first few
            for b in bad[:10]:
                logging.error(f"[Stage2] bad: {b.get('path')} err={b.get('error')}")
            if cfg["stage2"].get("strict", True):
                logging.error("[Stage2] strict=1 and bad exists. Fix raw data/manifest.")
                sys.exit(4)

        for g in good:
            raw_paths.append(g["raw_out"])
            index_rows.append({
                "stock_code": g.get("stock_code", "unknown"),
                "date": g["date"],
                "session": g["session"],
                "split": g["split"],
                "raw_label_path": g["raw_out"],
                "n": g["n"],
                "valid": g["valid"]
            })
            if g["split"] == "train":
                s = g.get("train_samples")
                if isinstance(s, np.ndarray) and s.size > 0:
                    train_samples_all.append(s)

    else:
        # only_scale: enumerate raw label files from disk
        for r in ok_rows:
            date = r["date"]
            session = int(r["session"])
            split = infer_split(date, splits) or "out_of_split"
            if split == "out_of_split":
                continue
            sc = resolve_stock_code(r)
            raw_path = os.path.join(project_root, cfg["stage2"]["labels_raw_dir"], horizon_id, date, f"{sc}_{date}_{session}.npz")
            if os.path.exists(raw_path):
                raw_paths.append(raw_path)
                index_rows.append({"stock_code": sc, "date": date, "session": session, "split": split, "raw_label_path": raw_path})
        if not raw_paths:
            logging.error("only_scale=1 but no raw label files found. Run without --only-scale first.")
            sys.exit(5)

        # gather train samples from raw files (fast)
        sample_n = int(cfg["stage2"].get("quantile_sample_per_session", 5000))
        for item in index_rows:
            if item["split"] != "train":
                continue
            z = np.load(item["raw_label_path"])
            r_raw = z["r_raw"].astype(np.float32, copy=False)
            is_valid = z["is_valid"].astype(np.uint8).astype(bool)
            rr = r_raw[is_valid]
            if rr.size == 0:
                continue
            aa = np.abs(rr)
            if aa.size <= sample_n:
                train_samples_all.append(aa.astype(np.float32, copy=False))
            else:
                stride = max(1, aa.size // sample_n)
                train_samples_all.append(aa[::stride][:sample_n].astype(np.float32, copy=False))

    # 2) Compute train-only stats (q90_abs, q99_abs)
    if not train_samples_all:
        logging.error("No train samples collected for quantiles. Check split ranges or data.")
        sys.exit(6)

    samples_cat = np.concatenate(train_samples_all, axis=0).astype(np.float64, copy=False)
    samples_cat = samples_cat[np.isfinite(samples_cat)]
    if samples_cat.size < 1000:
        logging.error(f"Too few samples for quantiles: {samples_cat.size}")
        sys.exit(7)

    q90 = float(np.quantile(samples_cat, 0.90))
    q99 = float(np.quantile(samples_cat, 0.99))
    q100 = float(np.quantile(samples_cat, 1))
    eps = float(cfg["label"].get("eps", 1e-12))
    q90 = max(q90, eps)
    q99 = max(q99, q90)
    q100 = max(q100, q99)
    # scale = 1.0 / q90
    label_mode = str(cfg.get('stage2', {}).get('label_mode', 'time_window_avg_mid')).strip()
    if label_mode in ('tick_mid_delta', 'tick_mid_delta_x200') and bool(cfg.get('stage2', {}).get('ignore_cli_scale_in_tick_mode', True)):
        scale = 1.0
    else:
        scale = args.scale

    # clip_abs_scaled based on q99
    clip_abs_scaled = float(q100 * scale)
    max_clip = float(cfg["stage2"].get("max_clip_abs_scaled", 10.0))
    clip_abs_scaled = min(max_clip, max(1.0, clip_abs_scaled))

    stats = {
        "horizon_id": horizon_id,
        "min_seconds": min_sec,
        "max_seconds": max_sec,
        "train_only_quantiles": {
            "q90_abs_r_raw": q90,
            "q99_abs_r_raw": q99,
            "q100_abs_r_raw": q100
        },
        "scale": scale,
        "clip_abs_scaled": clip_abs_scaled,
        "quantile_sample_per_session": int(cfg["stage2"].get("quantile_sample_per_session", 5000)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    stats_path = os.path.join(stats_dir, f"label_stats_{horizon_id}.json")
    save_json(stats_path, stats)
    logging.info(f"[Stage2] label stats saved: {stats_path}")
    logging.info(f"[Stage2] q90_abs={q90:.6g}, q99_abs={q99:.6g}, scale={scale:.6g}, clip_abs_scaled={clip_abs_scaled:.3f}")

    # 3) Scale from raw label files -> final label files
    # build list
    scale_jobs = []
    for item in index_rows:
        date = item["date"]
        session = int(item["session"])
        split = item["split"]
        if split == "out_of_split":
            continue
        raw_path = item["raw_label_path"]
        out_dir = os.path.join(project_root, cfg["stage2"]["labels_final_dir"], horizon_id, date)
        ensure_dir(out_dir)
        sc = str(item.get("stock_code") or "unknown")
        out_path = os.path.join(out_dir, f"{sc}_{date}_{session}.npz")
        item["final_label_path"] = out_path
        scale_jobs.append((raw_path, out_path, scale, clip_abs_scaled))

    t1 = time.time()
    with Pool(processes=num_workers) as pool:
        scale_res = list(pool.map(worker_scale_from_raw, scale_jobs))
    dt2 = time.time() - t1
    bad2 = [x for x in scale_res if not x.get("ok")]
    logging.info(f"[Stage2] scale pass done. bad={len(bad2)} in {dt2:.1f}s")
    if bad2:
        for b in bad2[:10]:
            logging.error(f"[Stage2] scale bad: raw={b.get('raw_path')} err={b.get('error')}")
        sys.exit(8)

    # write index + summary
    index_path = os.path.join(project_root, cfg["stage2"]["labels_index_path"].format(horizon_id=horizon_id))
    append_jsonl(index_path, index_rows)

    summary = {
        "horizon_id": horizon_id,
        "stats_path": stats_path,
        "labels_index_path": index_path,
        "counts": {
            "sessions": len(index_rows),
            "train_sessions": sum(1 for x in index_rows if x.get("split") == "train"),
            "val_sessions": sum(1 for x in index_rows if x.get("split") == "val"),
            "test_sessions": sum(1 for x in index_rows if x.get("split") == "test")
        }
    }
    summary_path = os.path.join(project_root, cfg["stage2"]["stage2_summary_path"].format(horizon_id=horizon_id))
    save_json(summary_path, summary)
    logging.info(f"[Stage2] summary: {summary_path}")
    logging.info(f"[Stage2] labels index: {index_path}")
    logging.info("[Stage2] DONE.")


if __name__ == "__main__":
    main()
