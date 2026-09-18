#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3: Pack per-row features (factors) + labels into mmap .npy shards, split by train/val/test.

WHY per-row packs (not per-sample windows):
- If we precompute sliding windows X (W=64) for every tick, disk explodes (tens~hundreds TB).
- Row-pack stores only (N, F) features and (N,) labels; Stage4 builds windows on-the-fly by slicing,
  with "block shuffle + continuous slice + mmap cache".

Outputs (per split, per shard):
- X.npy      float32 [R, F]
- y.npy      float32 [R]      (scaled, clipped; from Stage2)
- y_raw.npy  float32 [R]      (raw return; from Stage2)   (optional)
- w.npy      float32 [R]      sample_weight per row       (0 for invalid rows)
- is_valid.npy uint8 [R]      validity from Stage2
- t_sec.npy  float64 [R]      timestamp seconds (preserves millisecond spacing) (optional)

And segment metadata (per shard):
- segments.jsonl: each line describes a session segment placed in this shard, with start/len/date/session,
  and counts helpful for later step sizing.

Manifest:
- packs_manifest.json: list shards and their files & segment files.

Dependencies:
- Stage1 ok manifest: to map (date,session)->csv path
- Stage1 schema_factors.json: factor column order
- Stage2 labels index: to map (date,session)->label path & split
- Stage2 label_stats_{horizon}.json: to compute weights using train q90
"""

try:
    from .target_contract import load_target_binding, require_label_binding
except ImportError:
    from target_contract import load_target_binding, require_label_binding

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .artifact_contract import sha256_file
    from .data_contract import valid_window_end_mask
    from .feature_preprocessing import (
        parse_norm_groups,
        preprocessing_spec,
        preprocess_session_features,
        required_feature_columns,
    )
except ImportError:
    from artifact_contract import sha256_file
    from data_contract import valid_window_end_mask
    from feature_preprocessing import (
        parse_norm_groups,
        preprocessing_spec,
        preprocess_session_features,
        required_feature_columns,
    )

try:
    import pandas as pd
except Exception:
    print("ERROR: pandas is required.", file=sys.stderr)
    raise


# -----------------------------
# small utils
# -----------------------------

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


def read_jsonl(p: str) -> List[dict]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(p: str, rows: List[dict]):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def open_memmap_npy(path: str, dtype, shape):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def infer_split(date: str, splits: dict) -> Optional[str]:
    for k in ["train", "val", "test"]:
        s = splits.get(k, {})
        if s.get("start") and s.get("end") and (s["start"] <= date <= s["end"]):
            return k
    return None


def abs_path(project_root: str, p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(project_root, p)


def load_norm_map(norm_map_path: Optional[str]) -> Tuple[Optional[dict], dict]:
    """Load factor normalization map from Stage1.5.

    Returns:
      norm_map (dict or None), meta (dict, possibly empty)
    """
    if not norm_map_path:
        return None, {}
    if not os.path.exists(norm_map_path):
        raise FileNotFoundError(f"Configured normalization map is missing: {norm_map_path}")
    nm = load_json(norm_map_path)
    meta = nm.get("meta", {}) if isinstance(nm, dict) else {}
    return nm, meta


# -----------------------------
# Pack writer
# -----------------------------

class RowPackWriter:
    def __init__(
        self,
        out_dir: str,
        split: str,
        shard_rows: int,
        num_factors: int,
        write_y_raw: bool,
        write_t_sec: bool,
        write_is_valid: bool,
    ):
        self.out_dir = out_dir
        self.split = split
        self.shard_rows = int(shard_rows)
        self.F = int(num_factors)
        self.write_y_raw = bool(write_y_raw)
        self.write_t_sec = bool(write_t_sec)
        self.write_is_valid = bool(write_is_valid)

        self.shard_id = -1
        self.offset = 0
        self._mm_X = None
        self._mm_y = None
        self._mm_yraw = None
        self._mm_w = None
        self._mm_valid = None
        self._mm_t = None

        self._segments = []
        self._shards_manifest = []

        ensure_dir(out_dir)

    def _start_new_shard(self):
        self.shard_id += 1
        self.offset = 0
        self._segments = []

        sid = f"{self.shard_id:05d}"
        prefix = os.path.join(self.out_dir, f"{self.split}_pack_{sid}")

        self.paths = {
            "X": prefix + "_X.npy",
            "y": prefix + "_y.npy",
            "w": prefix + "_w.npy",
            "segments": prefix + "_segments.jsonl",
        }
        if self.write_y_raw:
            self.paths["y_raw"] = prefix + "_y_raw.npy"
        if self.write_is_valid:
            self.paths["is_valid"] = prefix + "_is_valid.npy"
        if self.write_t_sec:
            self.paths["t_sec"] = prefix + "_t_sec.npy"

        self._mm_X = open_memmap_npy(self.paths["X"], np.float32, (self.shard_rows, self.F))
        self._mm_y = open_memmap_npy(self.paths["y"], np.float32, (self.shard_rows,))
        self._mm_w = open_memmap_npy(self.paths["w"], np.float32, (self.shard_rows,))

        if self.write_y_raw:
            self._mm_yraw = open_memmap_npy(self.paths["y_raw"], np.float32, (self.shard_rows,))
        if self.write_is_valid:
            self._mm_valid = open_memmap_npy(self.paths["is_valid"], np.uint8, (self.shard_rows,))
        if self.write_t_sec:
            self._mm_t = open_memmap_npy(self.paths["t_sec"], np.float64, (self.shard_rows,))

    def _finalize_shard(self):
        if self._mm_X is None:
            return

        # flush
        for mm in [self._mm_X, self._mm_y, self._mm_w, self._mm_yraw, self._mm_valid, self._mm_t]:
            if mm is not None:
                mm.flush()

        # write segments
        write_jsonl(self.paths["segments"], self._segments)

        # record shard manifest
        shard_entry = {
            "split": self.split,
            "shard_id": self.shard_id,
            "capacity_rows": self.shard_rows,
            "valid_rows": self.offset,  # important: only first offset rows are meaningful
            "files": dict(self.paths),
            "num_segments": len(self._segments),
        }
        self._shards_manifest.append(shard_entry)

        # close refs
        self._mm_X = None
        self._mm_y = None
        self._mm_w = None
        self._mm_yraw = None
        self._mm_valid = None
        self._mm_t = None
        self._segments = []

    def add_session(
        self,
        X: np.ndarray,          # [N, F] float32
        y: np.ndarray,          # [N] float32
        y_raw: Optional[np.ndarray],   # [N] float32
        w: np.ndarray,          # [N] float32
        is_valid: np.ndarray,   # [N] bool
        t_sec: Optional[np.ndarray],   # [N] float64
        meta: dict,             # date/session/path etc
        window_W: int,
        max_history_span_seconds: float,
        max_inter_event_gap_seconds: float,
    ) -> int:
        if self._mm_X is None:
            self._start_new_shard()

        N = int(X.shape[0])
        if N > self.shard_rows:
            raise RuntimeError(f"Session too large for a shard_rows={self.shard_rows}. N={N} (increase shard_rows).")

        # ensure session is not split across shards
        if self.offset > 0 and (self.offset + N) > self.shard_rows:
            self._finalize_shard()
            self._start_new_shard()
            print(meta['csv_path'], "new shard!!!")

        start = self.offset
        end = start + N

        self._mm_X[start:end, :] = X
        self._mm_y[start:end] = y
        self._mm_w[start:end] = w
        if self.write_y_raw and y_raw is not None:
            self._mm_yraw[start:end] = y_raw
        if self.write_is_valid:
            self._mm_valid[start:end] = is_valid.astype(np.uint8)
        if self.write_t_sec and t_sec is not None:
            self._mm_t[start:end] = t_sec

        # Use the same contract as Stage4 so manifest counts cannot drift.
        valid_end = int(valid_window_end_mask(
            weights=w,
            t_sec=t_sec,
            window_size=window_W,
            max_history_span_seconds=max_history_span_seconds,
            max_inter_event_gap_seconds=max_inter_event_gap_seconds,
        ).sum())

        seg = {
            "shard_id": self.shard_id,
            "start_row": start,
            "length": N,
            "valid_end_count": valid_end,
            **meta,
        }
        self._segments.append(seg)
        self.offset = end
        return valid_end

    def finalize(self):
        self._finalize_shard()
        return self._shards_manifest


# -----------------------------
# main
# -----------------------------

def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", type=str, required=True)
    ap.add_argument("--num-workers", type=int, default=0)  # reserved; Stage3 is single-process IO-optimized
    args = ap.parse_args()

    cfg = load_config(args.config_path)
    project_root = cfg["project"]["project_root"]

    # horizon
    horizon_id = cfg["horizons"]["active_horizon_id"]
    window_W = int(cfg["features"]["window_W"])

    # paths
    manifests_dir = os.path.join(project_root, cfg["paths"]["manifests_dir"])
    stats_dir = os.path.join(project_root, cfg["paths"]["stats_dir"])

    stage1_ok_path = cfg["stage1"]["manifest_ok_path"]
    stage1_ok_path = abs_path(project_root, stage1_ok_path)

    # stage3 config (needed early)
    stage3 = cfg.get("stage3", {})

    # Stage1 factor schema path (combined factors order)
    schema_cfg_path = (cfg.get("stage1", {}) or {}).get("schema_path")
    schema_path = abs_path(project_root, schema_cfg_path) if schema_cfg_path else os.path.join(stats_dir, "schema_factors.json")

    if not os.path.exists(schema_path):
        logging.error(f"Missing factor schema at {schema_path} (Stage1 output).")
        sys.exit(2)

    base_schema = load_json(schema_path)
    base_factor_cols = base_schema.get("factor_cols")
    if not base_factor_cols:
        logging.error("Factor schema has no factor_cols. Check Stage1 schema output.")
        sys.exit(3)

    base_F = int(base_schema.get("num_factors", len(base_factor_cols)))
    if len(base_factor_cols) != base_F:
        logging.warning(f"[Stage3] base factor_cols length mismatch: len={len(base_factor_cols)} vs num_factors={base_F}")

    # Stage1.5 normalization map
    norm_map_path = stage3.get("feature_norm_map_path") or ((cfg.get("stage1_5", {}) or {}).get("factor_norm_map_path"))
    norm_map_path = abs_path(project_root, norm_map_path) if norm_map_path else None
    norm_map, nm_meta = load_norm_map(norm_map_path)
    raw_set, vol_set, mcap_set, none_set = parse_norm_groups(norm_map)

    # final factor columns: drop "none", keep stable order from Stage1 schema
    factor_cols = [c for c in base_factor_cols if c not in none_set]
    F = int(len(factor_cols))
    if F == 0:
        logging.error("After dropping 'none' factors, zero factors remain. Check Stage1.5 norm map.")
        sys.exit(4)

    # factor -> method
    method_by_factor = {}
    for c in factor_cols:
        if c in vol_set:
            method_by_factor[c] = "volume_norm"
        elif c in mcap_set:
            method_by_factor[c] = "mcap_norm"
        else:
            method_by_factor[c] = "raw"

    # stage2 index + stats
    labels_index_path = cfg["stage2"]["labels_index_path"].format(horizon_id=horizon_id)
    labels_index_path = abs_path(project_root, labels_index_path)
    if not os.path.exists(labels_index_path):
        logging.error(f"Missing labels_index.jsonl: {labels_index_path} (Stage2 output).")
        sys.exit(5)

    label_stats_path = abs_path(project_root, cfg["stage2"]["label_stats_path"].format(horizon_id=horizon_id))
    if not os.path.exists(label_stats_path):
        logging.error(f"Missing label stats: {label_stats_path} (Stage2 output).")
        sys.exit(6)
    binding = load_target_binding(cfg)
    label_stats = load_json(label_stats_path)
    q90 = float(label_stats["train_only_quantiles"]["q90_abs_r_raw"])
    q90 = max(q90, float(cfg["label"].get("eps", 1e-12)))

    # weight params
    sw = cfg["sample_weight"]
    alpha = float(sw.get("alpha", 0.75))
    clip_max = float(sw.get("clip_max", 3.0))
    min_w = float(sw.get("min_weight", 1.0))
    max_w = float(sw.get("max_weight", 4.0))

    # fill rules
    ffill = bool(cfg["data"]["session_rules"].get("ffill_within_session", True))
    fill_value_raw = cfg["data"]["session_rules"].get("fill_remaining_nan", 0.0)
    fill_value = None if fill_value_raw is None else float(fill_value_raw)
    session_rules = cfg["data"]["session_rules"]
    max_history_span_seconds = float(session_rules.get(
        "max_history_span_seconds",
        session_rules.get("max_window_span_seconds", 0.0),
    ))
    max_inter_event_gap_seconds = float(session_rules.get("max_inter_event_gap_seconds", 0.0))

    # pack params
    packed_root = stage3["packed_root"]
    packed_root = abs_path(project_root, packed_root)
    out_dir = os.path.join(packed_root, horizon_id)
    ensure_dir(out_dir)

    shard_rows = int(stage3.get("row_shard_rows", 2_000_000))
    write_y_raw = bool(stage3.get("write_y_raw", True))
    write_t_sec = bool(stage3.get("write_t_sec", True))
    write_is_valid = bool(stage3.get("write_is_valid", True))
    strict_match = bool(stage3.get("strict_rowcount_match", True))
    strict_no_nan_X = bool(stage3.get("strict_no_nan_X", True))
    if (max_history_span_seconds > 0.0 or max_inter_event_gap_seconds > 0.0) and not write_t_sec:
        raise RuntimeError("Stage4 time constraints require stage3.write_t_sec=true")

    # Build map session_id -> csv path from Stage1 ok (multi-stock safe)
    ok_rows = read_jsonl(stage1_ok_path)
    path_map: Dict[str, str] = {}
    for r in ok_rows:
        if not r.get("ok"):
            continue
        date_r = str(r["date"])
        session_r = int(r["session"])
        sc_r = str(r.get("stock_code") or "")
        sid_r = str(r.get("session_id") or (f"{sc_r}_{date_r}_{session_r}" if sc_r else f"{date_r}_{session_r}"))
        path = r["path"]
        path_map[sid_r] = path
        # compatibility aliases
        path_map[f"{date_r}_{session_r}"] = path
        if sc_r:
            path_map[f"{sc_r}_{date_r}_{session_r}"] = path

    # Load label index rows (canonical session list + split)
    idx_rows = read_jsonl(labels_index_path)

    # Pack writers
    writers = {
        "train": RowPackWriter(out_dir, "train", shard_rows, F, write_y_raw, write_t_sec, write_is_valid),
        "val":   RowPackWriter(out_dir, "val",   shard_rows, F, write_y_raw, write_t_sec, write_is_valid),
        "test":  RowPackWriter(out_dir, "test",  shard_rows, F, write_y_raw, write_t_sec, write_is_valid),
    }

    # Train-only factor stats
    # write final factor schema used by Stage3 (after dropping/normalizing)
    final_schema_path = stage3.get("final_schema_path") or os.path.join(stats_dir, f"schema_factors_stage3_{horizon_id}.json")
    final_schema_path = final_schema_path.format(horizon_id=horizon_id)
    final_schema_path = abs_path(project_root, final_schema_path)
    final_schema = {
        "horizon_id": horizon_id,
        "base_schema_path": schema_path,
        "norm_map_path": norm_map_path,
        "num_factors": F,
        "factor_cols": factor_cols,
        "dropped_factors": sorted(list(none_set)),
        "method_by_factor": method_by_factor,
        "preprocessing": preprocessing_spec(cfg, factor_cols, norm_map),
        "norm_map_sha256": sha256_file(norm_map_path) if norm_map_path else None,
        "groups": {
            "raw": sorted(list(raw_set)) if raw_set else [],
            "volume_norm": sorted(list(vol_set)) if vol_set else [],
            "mcap_norm": sorted(list(mcap_set)) if mcap_set else [],
            "none": sorted(list(none_set)) if none_set else [],
        },
    }
    save_json(final_schema_path, final_schema)

    do_input_stats = bool(stage3.get("compute_input_stats_train", True))
    mean = np.zeros((F,), dtype=np.float64)
    m2 = np.zeros((F,), dtype=np.float64)
    n_stat = 0

    def welford_update_batch(x: np.ndarray):
        nonlocal mean, m2, n_stat
        # x float32 [N, F]
        if x.size == 0:
            return
        x64 = x.astype(np.float64, copy=False)
        n_b = int(x64.shape[0])
        mean_b = x64.mean(axis=0)
        m2_b = ((x64 - mean_b) ** 2).sum(axis=0)

        n_a = int(n_stat)
        if n_a == 0:
            mean[:] = mean_b
            m2[:] = m2_b
            n_stat = n_b
            return

        delta = mean_b - mean
        n = n_a + n_b
        mean[:] = mean + delta * (n_b / n)
        m2[:] = m2 + m2_b + (delta ** 2) * (n_a * n_b / n)
        n_stat = n

    # Iterate sessions
    t0 = time.time()
    total_rows = {"train": 0, "val": 0, "test": 0}
    total_valid_end = {"train": 0, "val": 0, "test": 0}
    total_sessions = {"train": 0, "val": 0, "test": 0}

    for k, item in enumerate(idx_rows):
        split = item.get("split")
        if split not in writers:
            continue

        date = item["date"]
        session = int(item["session"])
        sc = str(item.get("stock_code") or (cfg.get("data", {}) or {}).get("stock_code") or "unknown")
        sid = str(item.get("session_id") or f"{sc}_{date}_{session}")
        csv_path = path_map.get(sid) or path_map.get(f"{sc}_{date}_{session}") or path_map.get(f"{date}_{session}")
        if csv_path is None:
            raise RuntimeError(f"CSV path not found for session_id={sid}. Check Stage1 manifest_ok vs Stage2 labels_index.")

        # label path: prefer final_label_path; else reconstruct
        label_path = item.get("final_label_path")
        if not label_path:
            # reconstruct
            label_path = os.path.join(
                project_root,
                cfg["stage2"]["labels_final_dir"],
                horizon_id,
                date,
                f"{sc}_{date}_{session}.npz"
            )
        label_path = abs_path(project_root, label_path)
        if not os.path.exists(label_path):
            raise RuntimeError(f"Label file missing: {label_path}")

        usecols = required_feature_columns(factor_cols, cfg, vol_set, mcap_set)

        try:
            df = pd.read_csv(csv_path, usecols=usecols, engine="c")
        except Exception as e:
            raise RuntimeError(f"read_csv failed (usecols={len(usecols)}): {csv_path} err={e}")

        try:
            X = preprocess_session_features(
                df,
                factor_cols,
                cfg,
                volume_norm_factors=vol_set,
                mcap_norm_factors=mcap_set,
                norm_meta=nm_meta,
                stock_code=sc,
                strict=strict_no_nan_X,
            )
        except Exception as exc:
            raise RuntimeError(f"feature preprocessing failed: {csv_path}: {exc}") from exc

        # read labels
        with np.load(label_path, allow_pickle=False) as z:
            require_label_binding(z, binding, sha256_file(csv_path))
            y = z["r_scaled"].astype(np.float32, copy=False)
            y_raw = z["r_raw"].astype(np.float32, copy=False)
            is_valid = z["is_valid"].astype(np.uint8, copy=False).astype(bool)
            t_sec = z["t_sec"].astype(np.float64, copy=False) if ("t_sec" in z.files) else None

        lengths = {
            "X": int(X.shape[0]),
            "y": int(y.shape[0]),
            "y_raw": int(y_raw.shape[0]),
            "is_valid": int(is_valid.shape[0]),
        }
        if t_sec is not None:
            lengths["t_sec"] = int(t_sec.shape[0])
        if strict_match and len(set(lengths.values())) != 1:
            raise RuntimeError(f"Rowcount mismatch session_id={sid}: {lengths}")

        N = int(min(lengths.values()))
        X = X[:N]
        y = y[:N]
        y_raw = y_raw[:N]
        is_valid = is_valid[:N]
        if t_sec is not None:
            t_sec = t_sec[:N]

        finite_label = np.isfinite(y) & np.isfinite(y_raw)
        if t_sec is not None:
            finite_label &= np.isfinite(t_sec)
        inconsistent_valid = is_valid & ~finite_label
        if np.any(inconsistent_valid):
            if bool(stage3.get("strict_label_integrity", True)):
                raise RuntimeError(
                    f"Label validity mismatch session_id={sid}: "
                    f"{int(inconsistent_valid.sum())} rows marked valid contain non-finite values"
                )
            is_valid = is_valid & finite_label

        # weights (align to “strong-signal trading”)
        # w = 1 + alpha * clip(|y_raw| / q90, 0, clip_max), then clip to [min_w, max_w]
        abs_ratio = np.abs(y_raw).astype(np.float64) / q90
        abs_ratio = np.clip(abs_ratio, 0.0, clip_max)
        w = (1.0 + alpha * abs_ratio).astype(np.float32)
        w = np.clip(w, min_w, max_w, out=w)
        # invalid rows weight=0 to avoid accidental usage
        w[~is_valid] = 0.0

        # input stats on TRAIN rows only (row-level stats, no future leak)
        if do_input_stats and split == "train":
            only_valid = bool(stage3.get("input_stats_only_valid", True))
            X_stat = X[is_valid] if only_valid else X
            welford_update_batch(X_stat)

        meta = {
            "date": date,
            "session": session,
            "session_id": sid,
            "stock_code": sc,
            "csv_path": csv_path,
            "label_path": label_path
        }

        valid_end_count = writers[split].add_session(
            X=X,
            y=y,
            y_raw=y_raw if write_y_raw else None,
            w=w,
            is_valid=is_valid,
            t_sec=t_sec,
            meta=meta,
            window_W=window_W,
            max_history_span_seconds=max_history_span_seconds,
            max_inter_event_gap_seconds=max_inter_event_gap_seconds,
        )

        total_rows[split] += N
        total_sessions[split] += 1
        total_valid_end[split] += valid_end_count

        if (k + 1) % int(stage3.get("log_every_sessions", 20)) == 0:
            logging.info(f"[Stage3] processed {k+1}/{len(idx_rows)} sessions...")

    # finalize writers
    shards = {}
    for sp, wtr in writers.items():
        shards[sp] = wtr.finalize()

    # input stats
    input_stats = None
    if do_input_stats and n_stat >= 2:
        var = m2 / max(1, (n_stat - 1))
        std = np.sqrt(np.maximum(var, 1e-12))
        input_stats = {
            **binding,
            "horizon_id": horizon_id,
            "num_factors": F,
            "train_row_count": int(n_stat),
            "factor_schema_sha256": sha256_file(final_schema_path),
            "mean": mean.tolist(),
            "std": std.tolist()
        }
        input_stats_path = stage3["input_stats_path"].format(horizon_id=horizon_id)
        input_stats_path = input_stats_path if os.path.isabs(input_stats_path) else os.path.join(project_root, input_stats_path)
        save_json(input_stats_path, input_stats)
    else:
        input_stats_path = None

    # manifest
    manifest = {
        **binding,
        "project_root": project_root,
        "horizon_id": horizon_id,
        "window_W": window_W,
        "num_factors": F,
        "factor_schema_path": final_schema_path,
        "base_factor_schema_path": schema_path,
        "norm_map_path": norm_map_path,
        "label_stats_path": label_stats_path,
        "packing": {
            "row_shard_rows": shard_rows,
            "write_y_raw": write_y_raw,
            "write_t_sec": write_t_sec,
            "write_is_valid": write_is_valid,
            "ffill": ffill,
            "fill_value": fill_value,
            "timestamp_dtype": "float64",
            "max_history_span_seconds": max_history_span_seconds,
            "max_inter_event_gap_seconds": max_inter_event_gap_seconds,
        },
        "sample_weight": {
            "q90_abs_r_raw_train": q90,
            "alpha": alpha,
            "clip_max": clip_max,
            "min_weight": min_w,
            "max_weight": max_w
        },
        "counts": {
            "rows": total_rows,
            "sessions": total_sessions,
            "valid_end_rows_for_windowing": total_valid_end
        },
        "shards": shards,
        "input_stats_path": input_stats_path
    }

    manifest_path = stage3["packs_manifest_path"].format(horizon_id=horizon_id)
    manifest_path = manifest_path if os.path.isabs(manifest_path) else os.path.join(project_root, manifest_path)
    save_json(manifest_path, manifest)

    dt = time.time() - t0
    logging.info(f"[Stage3] DONE in {dt:.1f}s")
    logging.info(f"[Stage3] manifest: {manifest_path}")
    logging.info(f"[Stage3] rows: {total_rows}, sessions: {total_sessions}, valid_end: {total_valid_end}")
    if input_stats_path:
        logging.info(f"[Stage3] input stats: {input_stats_path}")


if __name__ == "__main__":
    main()
