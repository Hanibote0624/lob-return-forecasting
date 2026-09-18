#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage5: Manifest integrity and sampled dataset healthcheck (NO training).

Checks (sample-based, fast):
1) Stage4 blocks guarantee:
   - end_idx in [seg_start, seg_start+seg_len)
   - w[end_idx] > 0 and finite
2) Window materialization:
   - X[base:end_idx+1] has shape [W,F], no NaN/Inf
3) Label/weight sanity:
   - y finite when w>0
   - y_raw finite when w>0 (if exists)
   - weight range, quantiles, effective sample size
4) Split integrity:
   - report per split: end positions, blocks, (optional) per-stock coverage
5) Write report json for reproducibility.

Output:
  data/stage5/{horizon_id}/stage5_report.json
"""

try:
    from .target_contract import load_target_binding, require_target_binding
except ImportError:
    from target_contract import load_target_binding, require_target_binding

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import hashlib
import json
import os
import time
import logging
from typing import Dict, Optional, Tuple, List, Any

import numpy as np


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: str, obj: dict):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def abspath(project_root: str, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(project_root, p)


def _get(d: dict, path: str, default=None):
    """Tiny nested getter: _get(cfg, 'stage5.strict', False)."""
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or (k not in cur):
            return default
        cur = cur[k]
    return cur


class LRUMmapCache:
    def __init__(self, cap: int):
        self.cap = max(1, int(cap))
        self.cache: Dict[str, np.ndarray] = {}
        self.lru: List[str] = []

    def get(self, path: str) -> np.ndarray:
        if path in self.cache:
            self.lru.remove(path)
            self.lru.append(path)
            return self.cache[path]
        arr = np.load(path, mmap_mode="r")
        self.cache[path] = arr
        self.lru.append(path)
        while len(self.lru) > self.cap:
            old = self.lru.pop(0)
            try:
                del self.cache[old]
            except KeyError:
                pass
        return arr


def sample_end_indices(blocks: list, n: int, rng: np.random.Generator):
    """Sample (block, end_idx) pairs."""
    out = []
    if not blocks or n <= 0:
        return out
    for _ in range(n):
        b = blocks[int(rng.integers(0, len(blocks)))]
        end_len = int(b["end_len"])
        off = int(rng.integers(0, end_len))
        end_idx = int(b["end_start"]) + off
        out.append((b, end_idx))
    return out


def quantiles(x: np.ndarray, ps=(0.5, 0.9, 0.99)):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {str(p): None for p in ps}
    q = np.quantile(x, list(ps))
    return {str(p): float(v) for p, v in zip(ps, q)}


def check_split_structure(st3, st4, split, root, window):
    """Check every block against the original Stage3 segment metadata."""
    errors = []
    source_shards = {str(sh["shard_id"]): sh for sh in st3.get("shards", {}).get(split, [])}
    split_obj = st4.get("splits", {}).get(split, {})
    shards = split_obj.get("shards", {})
    blocks = split_obj.get("blocks", [])
    if set(shards) != set(source_shards):
        errors.append("Stage3/4 shard identities differ")
    segment_lookup = {}
    total_segments = 0
    for sid, source in source_shards.items():
        if sid not in shards:
            continue
        target = shards[sid]
        for name, path in source["files"].items():
            if abspath(root, target.get(name)) != abspath(root, path):
                errors.append(f"shard {sid}: {name} path differs from Stage3")
        for name in ("valid_rows", "capacity_rows", "num_segments"):
            if source.get(name) != target.get(name):
                errors.append(f"shard {sid}: {name} differs from Stage3")
        with open(abspath(root, source["files"]["segments"]), encoding="utf-8") as handle:
            segments = [json.loads(line) for line in handle if line.strip()]
        total_segments += len(segments)
        if len(segments) != source["num_segments"]:
            errors.append(f"shard {sid}: segment count mismatch")
        cursor = 0
        for seg in segments:
            if seg["start_row"] != cursor or seg["length"] <= 0:
                errors.append(f"shard {sid}: noncontiguous or empty segment")
            cursor = seg["start_row"] + seg["length"]
            segment_lookup[(sid, seg["start_row"], seg["length"])] = seg
        if cursor != source["valid_rows"] or not 0 <= cursor <= source["capacity_rows"]:
            errors.append(f"shard {sid}: used-row bounds mismatch")
    intervals = {}
    total_ends = 0
    for block in blocks:
        sid = str(block["shard_id"])
        start, length = int(block["end_start"]), int(block["end_len"])
        total_ends += length
        seg_start, seg_length = int(block["seg_start_row"]), int(block["seg_length"])
        segment = segment_lookup.get((sid, seg_start, seg_length))
        if segment is None:
            errors.append(f"shard {sid}: block has no matching Stage3 segment")
            continue
        if block.get("split") != split or length <= 0 or start - window + 1 < seg_start or start + length > seg_start + seg_length:
            errors.append(f"shard {sid}: block crosses segment bounds or has an invalid length/split")
        for name in ("date", "session", "session_id", "stock_code", "csv_path", "label_path"):
            if block.get(f"seg_{name}") != segment.get(name):
                errors.append(f"shard {sid}: block {name} differs from its Stage3 segment")
        intervals.setdefault(sid, []).append((start, start + length))
    for sid, ranges in intervals.items():
        ranges.sort()
        if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
            errors.append(f"shard {sid}: overlapping/duplicate endpoint ranges")
    expected_counts = {
        "total_blocks": len(blocks), "total_end_positions": total_ends,
        "total_shards": len(shards), "total_segments": total_segments,
    }
    for name, value in expected_counts.items():
        if st4.get("counts", {}).get(split, {}).get(name) != value:
            errors.append(f"Stage4 {name} disagrees with its contents")
    if total_ends != st3.get("counts", {}).get("valid_end_rows_for_windowing", {}).get(split):
        errors.append("Stage3/4 endpoint counts differ")
    return errors


def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, type=str)
    args = ap.parse_args()

    cfg = load_config(args.config_path)
    root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]

    # locate manifests
    st3_path = cfg["stage3"]["packs_manifest_path"].format(horizon_id=horizon_id)
    st3_path = abspath(root, st3_path)
    st4_path = cfg["stage4"]["stage4_manifest_path"].format(horizon_id=horizon_id)
    st4_path = abspath(root, st4_path)

    st3 = load_json(st3_path)
    st4 = load_json(st4_path)

    # contract (shape)
    W = int(st4["window_W"])
    F3 = int(st3["num_factors"])
    F4 = int(st4.get("num_factors", F3))
    F = int(F3)
    rules = cfg.get("data", {}).get("session_rules", {})
    constraints = {
        "max_history_span_seconds": float(rules.get("max_history_span_seconds", 0.0)),
        "max_inter_event_gap_seconds": float(rules.get("max_inter_event_gap_seconds", 0.0)),
    }

    # stage5 cfg
    st5 = cfg.get("stage5", {})
    seed = int(st5.get("seed", 42))
    rng = np.random.default_rng(seed)
    mmap_cache_items = int(st5.get("mmap_cache_items", 16))
    strict = bool(st5.get("strict", False))
    per_stock_breakdown = bool(st5.get("per_stock_breakdown", True))

    row_samples = st5.get("row_samples_per_split", {"train": 50000, "val": 20000, "test": 20000})
    win_samples = st5.get("window_samples_per_split", {"train": 5000, "val": 2000, "test": 2000})

    report_path = st5["report_path"].format(horizon_id=horizon_id)
    report_path = abspath(root, report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    cache = LRUMmapCache(mmap_cache_items)

    target_error = None
    try:
        binding = load_target_binding(cfg)
        require_target_binding(st3, binding)
        require_target_binding(st4, binding)
    except (ValueError, KeyError, OSError) as exc:
        target_error = str(exc)

    report: Dict[str, Any] = {
        "target_contract_error": target_error,
        "project_root": root,
        "horizon_id": horizon_id,
        "window_W": W,
        "num_factors": F,
        "stage3_manifest": st3_path,
        "stage4_manifest": st4_path,
        "schema": {
            "stage3": {
                "num_factors": F3,
                "factor_schema_path": st3.get("factor_schema_path"),
                "base_factor_schema_path": st3.get("base_factor_schema_path"),
                "norm_map_path": st3.get("norm_map_path"),
            },
            "stage4": {
                "num_factors": st4.get("num_factors"),
                "factor_schema_path": st4.get("factor_schema_path"),
                "base_factor_schema_path": st4.get("base_factor_schema_path"),
                "norm_map_path": st4.get("norm_map_path"),
            },
        },
        "consistency": {
            "target_contract_match": target_error is None,
            "num_factors_match": (F3 == F4),
            "factor_schema_match": (st3.get("factor_schema_path") == st4.get("factor_schema_path")) if (st3.get("factor_schema_path") and st4.get("factor_schema_path")) else None,
            "window_W_match": st3.get("window_W") == W == cfg["features"]["window_W"],
            "horizon_match": st3.get("horizon_id") == st4.get("horizon_id") == horizon_id,
            "window_constraints_match": st4.get("window_constraints") == constraints
            and all(st3.get("packing", {}).get(key) == value for key, value in constraints.items()),
        },
        "md5": {
            "stage3_manifest": md5_file(st3_path),
            "stage4_manifest": md5_file(st4_path),
        },
        "splits": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strict": strict,
    }

    global_ok = True
    global_warnings: List[str] = []

    for name, matches in report["consistency"].items():
        if matches is False:
            global_ok = False
            global_warnings.append(f"manifest/configuration consistency failed: {name}")

    for split in ["train", "val", "test"]:
        split_obj = st4["splits"].get(split)
        if split_obj is None:
            report["splits"][split] = {"ok": False, "reason": "missing split"}
            global_ok = False
            continue

        blocks = split_obj.get("blocks", [])
        shards = split_obj.get("shards", {})
        try:
            structural_errors = check_split_structure(st3, st4, split, root, W)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            structural_errors = [f"unreadable/invalid manifest or segment metadata: {exc}"]
        if structural_errors:
            report["splits"][split] = {"ok": False, "structural_errors": structural_errors}
            global_ok = False
            continue

        n_row = int(row_samples.get(split, 20000))
        n_win = int(win_samples.get(split, 2000))

        pairs = sample_end_indices(blocks, n_row, rng)
        if not pairs:
            report["splits"][split] = {"ok": False, "reason": "no blocks"}
            global_ok = False
            continue

        # accumulators
        ys = []
        yraws = []
        ws = []
        violations = {
            "end_outside_segment": 0,
            "base_outside_segment": 0,
            "w_nonpos_or_nan": 0,
            "y_nan_when_wpos": 0,
            "yraw_nan_when_wpos": 0,
            "missing_shard": 0,
            "row_outside_shard": 0,
         }
        stock_counts: Dict[str, int] = {}

        # row-level checks
        for b, end_idx in pairs:
            sid = str(b.get("shard_id"))
            sh = shards.get(sid)
            if sh is None:
                violations["missing_shard"] += 1
                continue

            if per_stock_breakdown and ("seg_stock_code" in b):
                sc = str(b.get("seg_stock_code"))
                stock_counts[sc] = stock_counts.get(sc, 0) + 1

            y_path = abspath(root, sh["y"])
            w_path = abspath(root, sh["w"])
            yraw_path = abspath(root, sh.get("y_raw"))

            try:
                y_mm = cache.get(y_path)
                w_mm = cache.get(w_path)
            except (OSError, ValueError):
                violations["missing_shard"] += 1
                continue
            if y_mm.ndim != 1 or w_mm.ndim != 1 or not 0 <= end_idx < min(len(y_mm), len(w_mm)):
                violations["row_outside_shard"] += 1
                continue
            y = float(y_mm[end_idx])
            w = float(w_mm[end_idx])

            seg_start = int(b["seg_start_row"])
            seg_len = int(b["seg_length"])
            seg_end_excl = seg_start + seg_len

            if not (seg_start <= end_idx < seg_end_excl):
                violations["end_outside_segment"] += 1

            if (not np.isfinite(w)) or (w <= 0.0):
                violations["w_nonpos_or_nan"] += 1
            else:
                if (not np.isfinite(y)):
                    violations["y_nan_when_wpos"] += 1
                else:
                    ys.append(y)
                    ws.append(w)

                if yraw_path is not None:
                    try:
                        yraw_mm = cache.get(yraw_path)
                    except (OSError, ValueError):
                        violations["missing_shard"] += 1
                        continue
                    if yraw_mm.ndim != 1 or end_idx >= len(yraw_mm):
                        violations["row_outside_shard"] += 1
                        continue
                    yr = float(yraw_mm[end_idx])
                    if not np.isfinite(yr):
                        violations["yraw_nan_when_wpos"] += 1
                    else:
                        yraws.append(yr)

        ys_arr = np.asarray(ys, dtype=np.float64)
        ws_arr = np.asarray(ws, dtype=np.float64)
        yraw_arr = np.asarray(yraws, dtype=np.float64) if yraws else np.empty((0,), dtype=np.float64)

        # window-level checks (materialize)
        win_pairs = sample_end_indices(blocks, n_win, rng)
        X_nan = 0
        X_inf = 0
        X_badshape = 0
        win_eff = 0
        time_checks = dict.fromkeys((
            "time_missing_or_badshape", "time_nonfinite", "time_nonmonotonic",
            "history_span_exceeded", "inter_event_gap_exceeded",
        ), 0)

        for b, end_idx in win_pairs:
            sid = str(b.get("shard_id"))
            sh = shards.get(sid)
            if sh is None:
                violations["missing_shard"] += 1
                continue

            X_path = abspath(root, sh["X"])
            try:
                X_mm = cache.get(X_path)
            except (OSError, ValueError):
                violations["missing_shard"] += 1
                continue

            seg_start = int(b["seg_start_row"])
            seg_len = int(b["seg_length"])
            seg_end_excl = seg_start + seg_len

            base = end_idx - (W - 1)
            if base < seg_start:
                violations["base_outside_segment"] += 1
                continue
            if end_idx >= seg_end_excl:
                violations["end_outside_segment"] += 1
                continue

            # Make sure we can index; we keep it as a view (no full copy).
            if X_mm.ndim != 2:
                X_badshape += 1
                continue
            Xw = np.asarray(X_mm[base:end_idx + 1, :], dtype=np.float32)
            win_eff += 1
            if Xw.shape != (W, F):
                X_badshape += 1
                continue
            if np.isnan(Xw).any():
                X_nan += 1
            if np.isinf(Xw).any():
                X_inf += 1
            try:
                time_path = abspath(root, sh.get("t_sec"))
                times = np.asarray(cache.get(time_path)[base:end_idx + 1], dtype=np.float64)
                if times.shape != (W,):
                    raise ValueError("timestamp window shape mismatch")
            except (OSError, ValueError, TypeError):
                time_checks["time_missing_or_badshape"] += 1
                continue
            if not np.isfinite(times).all():
                time_checks["time_nonfinite"] += 1
                continue
            gaps = np.diff(times)
            time_checks["time_nonmonotonic"] += int(np.any(gaps < 0))
            max_span = constraints["max_history_span_seconds"]
            max_gap = constraints["max_inter_event_gap_seconds"]
            time_checks["history_span_exceeded"] += int(max_span > 0 and times[-1] - times[0] > max_span + 1e-12)
            time_checks["inter_event_gap_exceeded"] += int(max_gap > 0 and np.any(gaps > max_gap + 1e-12))

        # summary stats
        weighted_mae0 = None
        ess_ratio = None
        if ys_arr.size > 0 and ws_arr.size > 0:
            wsum = float(np.sum(ws_arr))
            if wsum > 0:
                weighted_mae0 = float(np.sum(ws_arr * np.abs(ys_arr)) / wsum)
            # effective sample size ratio
            s1 = float(np.sum(ws_arr))
            s2 = float(np.sum(ws_arr * ws_arr))
            if s2 > 0:
                ess = (s1 * s1) / s2
                ess_ratio = float(ess / max(1.0, ws_arr.size))

        # Report observed failures even in diagnostic (non-strict) mode.
        split_ok = not (
            any(v > 0 for v in violations.values()) or any(time_checks.values())
            or X_badshape or X_nan or X_inf
        )

        report["splits"][split] = {
            "ok": bool(split_ok),
            "structural_errors": [],
            "counts": st4.get("counts", {}).get(split),
            "samples": {
                "row_samples_target": n_row,
                "row_samples_effective": int(ys_arr.size),
                "window_samples_target": n_win,
                "window_samples_effective": int(win_eff),
            },
            "per_stock_samples": stock_counts if per_stock_breakdown else None,
            "y_scaled": {
                "mean": float(np.mean(ys_arr)) if ys_arr.size else None,
                "std": float(np.std(ys_arr)) if ys_arr.size else None,
                "abs_quantiles": quantiles(np.abs(ys_arr)) if ys_arr.size else None,
                "quantiles": quantiles(ys_arr) if ys_arr.size else None,
            },
            "y_raw": {
                "abs_quantiles": quantiles(np.abs(yraw_arr)) if yraw_arr.size else None,
                "quantiles": quantiles(yraw_arr) if yraw_arr.size else None,
            },
            "w": {
                "min": float(np.min(ws_arr)) if ws_arr.size else None,
                "max": float(np.max(ws_arr)) if ws_arr.size else None,
                "quantiles": quantiles(ws_arr) if ws_arr.size else None,
                "ess_ratio": ess_ratio,
            },
            "baseline": {
                "weighted_mae_predict0": weighted_mae0
            },
            "window_checks": {
                "X_badshape": int(X_badshape),
                "X_nan_windows": int(X_nan),
                "X_inf_windows": int(X_inf),
                **time_checks,
            },
            "violations": violations
        }

        if not split_ok:
            global_ok = False

    if global_warnings:
        report["warnings"] = global_warnings

    report["ok"] = bool(global_ok)
    report["status"] = "passed" if global_ok else "failed"
    report["check_scope"] = "full_manifest_structure_and_sampled_values"
    save_json(report_path, report)
    logging.info(f"[Stage5] report saved: {report_path}")

    if strict and (not global_ok):
        logging.error("[Stage5] STRICT FAILED. See report for details.")
        raise SystemExit(2)

    logging.info("[Stage5] DONE.")


if __name__ == "__main__":
    main()
