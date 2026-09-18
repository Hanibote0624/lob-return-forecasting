#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightGBM quick A/B test: Stage0 new factors vs baseline factors.

Goal
----
Train two LightGBM regressors on the SAME samples/labels, but with different
feature sets:
  A) baseline: only the "old" factors (e.g. gpmain_0..gpmain_108)
  B) stage0+:  old factors + Stage0-added factors (e.g. gpmain_0..gpmain_124)

This matches your pipeline layout:
- CSVs live under cfg.data.stocks[*].out_root/{YYYYMMDD}/{stock}_{YYYYMMDD}_{session}.csv
  (Stage1 scans out_root, not raw_root)
- Labels index:
    cfg.stage2.labels_index_path (jsonl)
  Each row includes stock_code/date/session/split and final_label_path.

Outputs
-------
- Prints a compact comparison report to stdout.
- Writes a JSON report to: <out_dir>/lgbm_compare_report.json
- Saves both trained models to: <out_dir>/model_baseline.txt and model_stage0.txt

Usage
-----
python3 lgbm_compare_stage0_factors.py \
  --config-path /path/to/gp_lit_regression_v6_gpmain_64.json \
  --out-dir results/lgbm_ab \
  --label-field r_scaled \
  --stride 3 \
  --max-sessions 0

Tips
----
- For a quick sanity check, use: --max-sessions 10 --stride 5
- For a fairer test, keep the same splits as Stage2 labels_index.
 - To inspect high-confidence subsets, use --top-fracs (default: 0.1%,0.5%,1%,5%)
   where "top" means top-|pred| rows within each split (val/test).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required.", file=sys.stderr)
    raise

try:
    import lightgbm as lgb
except Exception as e:
    print("ERROR: lightgbm is required. Install with: pip install lightgbm", file=sys.stderr)
    raise


# ----------------------------
# Utils
# ----------------------------

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def pearson_corr(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64, copy=False)
    p = p.astype(np.float64, copy=False)
    y = y - np.nanmean(y)
    p = p - np.nanmean(p)
    denom = np.sqrt(np.nanmean(y * y) * np.nanmean(p * p))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.nanmean(y * p) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Simple rankdata (average ranks for ties) without scipy."""
    a = a.astype(np.float64, copy=False)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)

    # Handle ties: average ranks for equal values
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 3:
        return float("nan")
    ry = _rankdata(y[mask])
    rp = _rankdata(p[mask])
    return pearson_corr(ry, rp)


def dir_acc(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y[mask] >= 0) == (p[mask] >= 0)))


def mse(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    d = y[mask] - p[mask]
    return float(np.mean(d * d))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    m = mse(y, p)
    return float(np.sqrt(m)) if np.isfinite(m) else float("nan")


def mae(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - p[mask])))


def _fmt_pct_key(frac: float) -> str:
    """frac=0.001 -> '0p1' (means 0.1%) ; frac=0.05 -> '5' (means 5%)."""
    pct = frac * 100.0
    s = f"{pct:g}"  # '0.1', '0.5', '1', '5'
    s = s.replace(".", "p")
    return s


def _subset_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Compute standard metrics on (y,p) with finite mask applied."""
    mask = np.isfinite(y) & np.isfinite(p)
    yy = y[mask]
    pp = p[mask]
    return {
        "n": float(mask.sum()),
        "mse": mse(yy, pp),
        "rmse": rmse(yy, pp),
        "mae": mae(yy, pp),
        "pearson": pearson_corr(yy, pp),
        "spearman": spearman_corr(yy, pp),
        "dir_acc": dir_acc(yy, pp),
    }


def top_abs_pred_metrics(
    y: np.ndarray,
    p: np.ndarray,
    top_fracs: List[float],
    split_name: str,
) -> Dict[str, float]:
    """Metrics for top-|pred| subsets inside a split.

    For each frac in top_fracs (e.g., 0.001), select samples with abs(pred)
    in the top frac proportion (within finite y/p samples), then compute metrics.
    Returns a FLAT dict with keys like:
      val_top0p1pct_n, val_top0p1pct_rmse, ...
    """
    out: Dict[str, float] = {}
    base_mask = np.isfinite(y) & np.isfinite(p)
    if base_mask.sum() == 0:
        for frac in top_fracs:
            tag = _fmt_pct_key(frac)
            for k in ("n", "mse", "rmse", "mae", "pearson", "spearman", "dir_acc"):
                out[f"{split_name}_top{tag}pct_{k}"] = float("nan") if k != "n" else 0.0
        return out

    yy = y[base_mask]
    pp = p[base_mask]
    abs_p = np.abs(pp)
    n = abs_p.size

    for frac in top_fracs:
        frac = float(frac)
        tag = _fmt_pct_key(frac)
        if not (0.0 < frac <= 1.0) or n == 0:
            for k in ("n", "mse", "rmse", "mae", "pearson", "spearman", "dir_acc"):
                out[f"{split_name}_top{tag}pct_{k}"] = float("nan") if k != "n" else 0.0
            continue

        # threshold at (1-frac) quantile; select abs(pred) >= threshold
        q = 1.0 - frac
        thr = float(np.quantile(abs_p, q))
        sel = abs_p >= thr
        y_sel = yy[sel]
        p_sel = pp[sel]

        sm = _subset_metrics(y_sel, p_sel)
        for k, v in sm.items():
            out[f"{split_name}_top{tag}pct_{k}"] = float(v)

    return out


# ----------------------------
# Data loading
# ----------------------------

@dataclass
class StockSpec:
    stock_code: str
    out_root: str


def build_stock_map(cfg: dict) -> Dict[str, StockSpec]:
    stocks = cfg.get("data", {}).get("stocks", [])
    if not stocks:
        raise KeyError("cfg.data.stocks is empty; please use multi-stock style config.")
    out: Dict[str, StockSpec] = {}
    for s in stocks:
        sc = str(s.get("stock_code"))
        oroot = str(s.get("out_root"))
        if not sc or not oroot:
            continue
        out[sc] = StockSpec(stock_code=sc, out_root=oroot)
    if not out:
        raise KeyError("No valid entries in cfg.data.stocks with stock_code/out_root")
    return out


def build_csv_path(cfg: dict, stock_code: str, date: str, session: int) -> str:
    """Match Stage1 convention: out_root/{YYYYMMDD}/{stock}_{YYYYMMDD}_{session}.csv"""
    stock_map = build_stock_map(cfg)
    st = stock_map.get(str(stock_code))
    if st is None:
        raise KeyError(f"stock_code not found in cfg.data.stocks: {stock_code}")
    daily_dir = cfg["data"]["file_pattern"]["daily_dir"].format(YYYYMMDD=date)
    csv_name = cfg["data"]["file_pattern"]["csv_name"].format(stock_code=stock_code, YYYYMMDD=date, session=session)
    return os.path.join(st.out_root, daily_dir, csv_name)


def get_factor_cols(cfg: dict) -> List[str]:
    req = cfg["data"]["required_columns"]["factors"]
    prefix = str(req["prefix"])
    count = int(req["count"])
    return [f"{prefix}{i}" for i in range(count)]


def infer_old_count(cfg: dict, factor_count: int) -> int:
    """Infer baseline factor count.

    Preferred:
    - cfg.stage0.rename.start_index (where Stage0 starts writing gpmain indices)

    Fallback:
    - factor_count - len(cfg.stage0.keep_features)

    Final fallback:
    - 109
    """
    s0 = cfg.get("stage0", {})
    rename = s0.get("rename", {})
    if isinstance(rename, dict) and "start_index" in rename:
        try:
            return int(rename["start_index"])
        except Exception:
            pass
    keep = s0.get("keep_features", [])
    if isinstance(keep, list) and keep:
        oc = factor_count - len(keep)
        if oc > 0:
            return int(oc)
    return 109


def load_session_xy(
    cfg: dict,
    item: dict,
    factor_cols: List[str],
    label_field: str,
    stride: int,
    max_rows_per_session: int,
    fill_na: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one session (CSV factors + label npz) and return (X, y) after masking/stride."""

    stock_code = str(item["stock_code"])
    date = str(item["date"])
    session = int(item["session"])

    # CSV path
    csv_path = build_csv_path(cfg, stock_code=stock_code, date=date, session=session)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # label path
    label_path = item.get("final_label_path")
    if not label_path:
        # reconstruct if missing
        project_root = cfg["project"]["project_root"]
        horizon_id = cfg["horizons"]["active_horizon_id"]
        labels_final_dir = cfg["stage2"]["labels_final_dir"]
        label_path = os.path.join(project_root, labels_final_dir, horizon_id, date, f"{stock_code}_{date}_{session}.npz")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label npz not found: {label_path}")

    z = np.load(label_path)
    if label_field not in z.files:
        raise KeyError(f"Label field '{label_field}' not found in npz={label_path}, available={z.files}")

    y = z[label_field].astype(np.float32, copy=False)
    is_valid = z["is_valid"].astype(np.uint8, copy=False).astype(bool)

    # Read factors
    try:
        df = pd.read_csv(csv_path, usecols=factor_cols, engine="c")
    except ValueError as e:
        # Column missing — give a clearer message
        # (common if you try to use stage0+ cols but csv doesn't have them)
        raise ValueError(f"Reading CSV failed for {csv_path}. Likely missing some factor columns. Error: {e}")

    x = df.to_numpy(dtype=np.float32, copy=False)
    # lightgbm handles NaN; but never allow inf
    x[~np.isfinite(x)] = np.nan

    # Optional filling
    if fill_na == "zero":
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    elif fill_na == "ffill0":
        # forward-fill per column then fill remaining with 0
        x2 = pd.DataFrame(x).ffill().to_numpy(dtype=np.float32, copy=False)
        x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
        x = x2
    elif fill_na == "none":
        pass
    else:
        raise ValueError(f"Unknown --fill-na: {fill_na}")

    if x.shape[0] != y.shape[0]:
        raise RuntimeError(f"Row count mismatch: csv_rows={x.shape[0]} label_rows={y.shape[0]} for {csv_path}")

    mask = is_valid & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if stride > 1:
        x = x[::stride]
        y = y[::stride]

    if max_rows_per_session > 0 and x.shape[0] > max_rows_per_session:
        x = x[:max_rows_per_session]
        y = y[:max_rows_per_session]

    return x, y


def build_split_matrices(
    cfg: dict,
    labels_index_rows: List[dict],
    factor_cols: List[str],
    label_field: str,
    stride: int,
    max_rows_per_session: int,
    max_sessions: int,
    fill_na: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load (X,y) for each split and concatenate."""

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        n_sess = 0
        for item in labels_index_rows:
            if item.get("split") != split:
                continue
            try:
                x, y = load_session_xy(
                    cfg=cfg,
                    item=item,
                    factor_cols=factor_cols,
                    label_field=label_field,
                    stride=stride,
                    max_rows_per_session=max_rows_per_session,
                    fill_na=fill_na,
                )
            except Exception as e:
                print(f"[WARN] skip session (split={split}) {item.get('stock_code')} {item.get('date')} s{item.get('session')}: {e}", file=sys.stderr)
                continue

            if x.size == 0:
                continue
            xs.append(x)
            ys.append(y)
            n_sess += 1
            if max_sessions > 0 and n_sess >= max_sessions:
                break

        if not xs:
            raise RuntimeError(f"No data loaded for split={split}. Check paths / splits / masks.")

        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        out[split] = (X, Y)
        print(f"[INFO] Loaded split={split}: sessions={n_sess}, rows={X.shape[0]}, feats={X.shape[1]}")

    return out


# ----------------------------
# Training + evaluation
# ----------------------------

def train_and_eval(
    name: str,
    params: dict,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    top_fracs: List[float],
) -> Dict[str, float]:

    Xtr, ytr = data["train"]
    Xv, yv = data["val"]
    Xt, yt = data["test"]

    train_data = lgb.Dataset(data=Xtr, label=ytr)
    val_data = lgb.Dataset(data=Xv, label=yv)

    # Keep silent by default; change log_evaluation period if you want.
    booster = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[val_data],
        valid_names=["val"],
    )

    pv = booster.predict(Xv)
    pt = booster.predict(Xt)

    metrics = {
        "val_mse": mse(yv, pv),
        "val_rmse": rmse(yv, pv),
        "val_mae": mae(yv, pv),
        "val_pearson": pearson_corr(yv, pv),
        "val_spearman": spearman_corr(yv, pv),
        "val_dir_acc": dir_acc(yv, pv),
        "test_mse": mse(yt, pt),
        "test_rmse": rmse(yt, pt),
        "test_mae": mae(yt, pt),
        "test_pearson": pearson_corr(yt, pt),
        "test_spearman": spearman_corr(yt, pt),
        "test_dir_acc": dir_acc(yt, pt),
    }
    # Top-|pred| subset metrics (val/test)
    if top_fracs:
        metrics.update(top_abs_pred_metrics(yv, pv, top_fracs=top_fracs, split_name="val"))
        metrics.update(top_abs_pred_metrics(yt, pt, top_fracs=top_fracs, split_name="test"))

    return {"_booster": booster, **metrics}


def format_metrics_line(m: Dict[str, float]) -> str:
    def f(x: float) -> str:
        return "nan" if (x is None or not np.isfinite(x)) else f"{x:.6g}"

    return (
        f"val_rmse={f(m['val_rmse'])} val_pearson={f(m['val_pearson'])} val_dir={f(m['val_dir_acc'])} | "
        f"test_rmse={f(m['test_rmse'])} test_pearson={f(m['test_pearson'])} test_dir={f(m['test_dir_acc'])}"
    )


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="results/lgbm_ab")
    ap.add_argument("--label-field", type=str, default="r_scaled", choices=["r_scaled", "r_raw"])
    ap.add_argument("--stride", type=int, default=1, help="Use every k-th valid row (downsample).")
    ap.add_argument("--max-rows-per-session", type=int, default=0, help="Cap rows per session after masking/stride (0=all).")
    ap.add_argument("--max-sessions", type=int, default=0, help="Cap sessions per split (0=all).")
    ap.add_argument("--fill-na", type=str, default="none", choices=["none", "zero", "ffill0"], help="How to handle NaNs in X.")
    ap.add_argument("--old-count", type=int, default=0, help="Baseline feature count (0=auto infer).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--top-fracs",
        type=str,
        default="0.1%,0.5%,1%,5%",
        help="Comma-separated top-|pred| fractions. Supports percent tokens like '0.1%'. Example: '0.1%,0.5%,1%,5%'.",
    )
    # LightGBM params overrides
    ap.add_argument("--num-iterations", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.085)
    ap.add_argument("--num-leaves", type=int, default=128)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--num-threads", type=int, default=40)
    ap.add_argument("--bagging-fraction", type=float, default=0.75)
    ap.add_argument("--feature-fraction", type=float, default=0.30)
    ap.add_argument("--min-data-in-leaf", type=int, default=2)

    args = ap.parse_args()

    # Parse top_fracs
    top_fracs: List[float] = []
    if args.top_fracs:
        for tok in str(args.top_fracs).split(","):
            t = tok.strip()
            if not t:
                continue
            if t.endswith("%"):
                v = float(t[:-1].strip()) / 100.0
            else:
                v = float(t)
            if v > 0:
                top_fracs.append(v)
    # Keep stable ordering, de-dup
    top_fracs = sorted(set(top_fracs))

    cfg = load_json(args.config_path)
    project_root = cfg["project"]["project_root"]

    # Resolve labels index
    horizon_id = cfg["horizons"]["active_horizon_id"]
    labels_index_rel = cfg["stage2"]["labels_index_path"].format(horizon_id=horizon_id)
    labels_index_path = labels_index_rel if os.path.isabs(labels_index_rel) else os.path.join(project_root, labels_index_rel)
    if not os.path.exists(labels_index_path):
        raise FileNotFoundError(f"labels_index not found: {labels_index_path}")

    labels_index_rows = read_jsonl(labels_index_path)
    labels_index_rows = [r for r in labels_index_rows if r.get("split") in ("train", "val", "test")]
    if not labels_index_rows:
        raise RuntimeError(f"No usable rows in labels_index: {labels_index_path}")

    # Factor columns
    all_factor_cols = get_factor_cols(cfg)
    factor_count = len(all_factor_cols)

    old_count = int(args.old_count) if args.old_count and args.old_count > 0 else infer_old_count(cfg, factor_count)
    old_count = max(1, min(old_count, factor_count))

    baseline_cols = all_factor_cols[:old_count]
    stage0_cols = all_factor_cols  # all

    print(f"[INFO] horizon={horizon_id}, label={args.label_field}")
    print(f"[INFO] baseline features: {len(baseline_cols)} (0..{len(baseline_cols)-1})")
    print(f"[INFO] stage0+ features: {len(stage0_cols)} (0..{len(stage0_cols)-1})")

    ensure_dir(args.out_dir)

    # LightGBM params (your params1 as default)
    params1 = {
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "objective": "regression",
        "metric": "mse",
        "num_iterations": int(args.num_iterations),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "max_depth": int(args.max_depth),
        "num_threads": int(args.num_threads),
        "bagging_fraction": float(args.bagging_fraction),
        "feature_fraction": float(args.feature_fraction),
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "verbosity": -1,

        # seeds for reproducibility
        "seed": int(args.seed),
        "feature_fraction_seed": int(args.seed),
        "bagging_seed": int(args.seed),
        "data_random_seed": int(args.seed),
    }

    # Load splits once per feature-set (because usecols differ)
    print("\n[INFO] Loading baseline data...")
    data_base = build_split_matrices(
        cfg=cfg,
        labels_index_rows=labels_index_rows,
        factor_cols=baseline_cols,
        label_field=args.label_field,
        stride=max(1, args.stride),
        max_rows_per_session=max(0, args.max_rows_per_session),
        max_sessions=max(0, args.max_sessions),
        fill_na=args.fill_na,
    )

    print("\n[INFO] Loading stage0+ data...")
    data_s0 = build_split_matrices(
        cfg=cfg,
        labels_index_rows=labels_index_rows,
        factor_cols=stage0_cols,
        label_field=args.label_field,
        stride=max(1, args.stride),
        max_rows_per_session=max(0, args.max_rows_per_session),
        max_sessions=max(0, args.max_sessions),
        fill_na=args.fill_na,
    )

    print("\n[INFO] Training baseline model...")
    res_base = train_and_eval("baseline", params1, data_base, top_fracs=top_fracs)
    booster_base = res_base.pop("_booster")

    print("[RESULT] baseline:")
    print("  " + format_metrics_line(res_base))

    print("\n[INFO] Training stage0+ model...")
    res_s0 = train_and_eval("stage0", params1, data_s0, top_fracs=top_fracs)
    booster_s0 = res_s0.pop("_booster")

    print("[RESULT] stage0+:")
    print("  " + format_metrics_line(res_s0))

    # Save models
    base_model_path = os.path.join(args.out_dir, "model_baseline.txt")
    s0_model_path = os.path.join(args.out_dir, "model_stage0.txt")
    booster_base.save_model(base_model_path)
    booster_s0.save_model(s0_model_path)

    report = {
        "config_path": os.path.abspath(args.config_path),
        "labels_index_path": os.path.abspath(labels_index_path),
        "horizon_id": horizon_id,
        "label_field": args.label_field,
        "stride": int(args.stride),
        "max_rows_per_session": int(args.max_rows_per_session),
        "max_sessions": int(args.max_sessions),
        "fill_na": args.fill_na,
        "params": params1,
        "feature_sets": {
            "baseline": {
                "count": len(baseline_cols),
                "cols": baseline_cols,
            },
            "stage0_plus": {
                "count": len(stage0_cols),
                "cols": stage0_cols,
            },
        },
        "metrics": {
            "baseline": res_base,
            "stage0_plus": res_s0,
            "delta_stage0_minus_baseline": {
                k: float(res_s0.get(k, float("nan")) - res_base.get(k, float("nan")))
                for k in sorted(set(res_base.keys()) | set(res_s0.keys()))
            },
        },
        "artifacts": {
            "model_baseline": os.path.abspath(base_model_path),
            "model_stage0": os.path.abspath(s0_model_path),
        },
    }

    report_path = os.path.join(args.out_dir, "lgbm_compare_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Saved:")
    print(f"  report: {report_path}")
    print(f"  baseline model: {base_model_path}")
    print(f"  stage0+ model: {s0_model_path}")


if __name__ == "__main__":
    main()
