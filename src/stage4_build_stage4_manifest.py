#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4: Build stage4_manifest for fast window sampling without crossing sessions.

Input:
- Stage3 packs_manifest.json (row-level packs + segments per shard)
- shard files: X.npy, y.npy, w.npy, (optional is_valid.npy, t_sec.npy)
- segments.jsonl per shard

Output:
- stage4_manifest.json:
  - lists blocks (contiguous valid end-index ranges) per split
  - records total end positions usable for windowing (end_idx>=W-1 AND w>0)
  - stores shard file paths for loader
Notes (multi-stock):
- Stage3 segments may contain stock_code/session_id/csv_path/label_path; Stage4 will carry them into blocks.

Compatibility:
- If Stage3 manifest provides factor_schema_path / num_factors, Stage4 will copy them into stage4_manifest
  (can be disabled by config stage4.copy_stage3_schema=false).
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
import json
import os
import sys
import time
import logging

import numpy as np

try:
    from .data_contract import valid_window_end_mask
except ImportError:
    from data_contract import valid_window_end_mask


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


def read_jsonl(p: str):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)



def abs_path(project_root: str, p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(project_root, p)


def build_runs_from_end_ok(end_ok: np.ndarray):
    """
    end_ok: bool array length L (segment-local indices).
    Return list of (start, length) runs where end_ok True and contiguous.
    """
    runs = []
    L = int(end_ok.shape[0])
    in_run = False
    start = 0
    for i in range(L):
        v = bool(end_ok[i])
        if v and (not in_run):
            in_run = True
            start = i
        elif (not v) and in_run:
            runs.append((start, i - start))
            in_run = False
    if in_run:
        runs.append((start, L - start))
    return runs


def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, type=str)
    args = ap.parse_args()

    cfg = load_config(args.config_path)
    project_root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]
    W = int(cfg["features"]["window_W"])

    stage3_manifest_path = abs_path(project_root, cfg["stage3"]["packs_manifest_path"].format(horizon_id=horizon_id))
    if not os.path.exists(stage3_manifest_path):
        logging.error(f"Missing Stage3 manifest: {stage3_manifest_path}")
        sys.exit(2)

    st3 = load_json(stage3_manifest_path)
    binding = load_target_binding(cfg)
    require_target_binding(st3, binding)

    st4 = cfg.get("stage4", {})
    block_size_ends = int(st4.get("block_size_ends", 65536))
    assert block_size_ends > 0
    strict = bool(st4.get("strict", True))
    copy_stage3_schema = bool(st4.get("copy_stage3_schema", True))
    session_rules = cfg.get("data", {}).get("session_rules", {}) or {}
    max_history_span_seconds = float(session_rules.get(
        "max_history_span_seconds",
        session_rules.get("max_window_span_seconds", 0.0),
    ))
    max_inter_event_gap_seconds = float(session_rules.get("max_inter_event_gap_seconds", 0.0))

    out_path = abs_path(project_root, st4["stage4_manifest_path"].format(horizon_id=horizon_id))
    ensure_dir(os.path.dirname(out_path))

    t0 = time.time()

    out = {
        **binding,
        "project_root": project_root,
        "horizon_id": horizon_id,
        "window_W": W,
        "source_stage3_manifest": stage3_manifest_path,
        "block_size_ends": block_size_ends,
        "window_constraints": {
            "max_history_span_seconds": max_history_span_seconds,
            "max_inter_event_gap_seconds": max_inter_event_gap_seconds,
        },
        "splits": {},
        "counts": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Copy schema/meta from Stage3 so downstream stages can avoid relying on stale config.features.num_factors
    if copy_stage3_schema:
        if "num_factors" in st3:
            out["num_factors"] = int(st3["num_factors"])
        if "factor_schema_path" in st3:
            out["factor_schema_path"] = st3["factor_schema_path"]
        if "base_factor_schema_path" in st3:
            out["base_factor_schema_path"] = st3["base_factor_schema_path"]
        if "norm_map_path" in st3 and st3["norm_map_path"]:
            out["norm_map_path"] = st3["norm_map_path"]


    for split in ["train", "val", "test"]:
        shards = (st3.get("shards") or {}).get(split, [])
        out["splits"][split] = {"shards": {}, "blocks": []}
        split_blocks = []
        split_total_ends = 0
        split_total_segments = 0

        for sh in shards:
            files = sh["files"]

            seg_path = abs_path(project_root, files["segments"])
            w_path = abs_path(project_root, files["w"])
            t_path = abs_path(project_root, files.get("t_sec")) if files.get("t_sec") else None

            # mmap row validity and high-precision event time.
            w_mm = np.load(w_path, mmap_mode="r")  # shape [capacity_rows]
            t_mm = np.load(t_path, mmap_mode="r") if t_path and os.path.exists(t_path) else None
            if (max_history_span_seconds > 0.0 or max_inter_event_gap_seconds > 0.0) and t_mm is None:
                raise RuntimeError(f"Missing t_sec shard required by window constraints: {t_path}")
            segs = read_jsonl(seg_path)
            split_total_segments += len(segs)

            shard_blocks = []
            for seg in segs:
                seg_start = int(seg["start_row"])
                seg_len = int(seg["length"])
                if seg_len < W:
                    continue

                w_local = np.asarray(w_mm[seg_start: seg_start + seg_len], dtype=np.float32)
                t_local = (
                    np.asarray(t_mm[seg_start: seg_start + seg_len], dtype=np.float64)
                    if t_mm is not None else None
                )
                end_ok = valid_window_end_mask(
                    weights=w_local,
                    t_sec=t_local,
                    window_size=W,
                    max_history_span_seconds=max_history_span_seconds,
                    max_inter_event_gap_seconds=max_inter_event_gap_seconds,
                )

                runs = build_runs_from_end_ok(end_ok)
                for (run_s, run_len) in runs:
                    remain = int(run_len)
                    cur = int(run_s)
                    while remain > 0:
                        take = min(block_size_ends, remain)
                        end_start_global = seg_start + cur
                        block = {
                            "shard_id": sh["shard_id"],
                            "split": split,
                            "end_start": int(end_start_global),
                            "end_len": int(take),
                            # carry segment meta (optional fields)
                            "seg_date": seg.get("date"),
                            "seg_session": seg.get("session"),
                            "seg_session_id": seg.get("session_id"),
                            "seg_stock_code": seg.get("stock_code"),
                            "seg_csv_path": seg.get("csv_path"),
                            "seg_label_path": seg.get("label_path"),
                            "seg_start_row": seg_start,
                            "seg_length": seg_len,
                        }
                        shard_blocks.append(block)
                        split_total_ends += int(take)
                        cur += take
                        remain -= take

            split_blocks.extend(shard_blocks)

            # store shard file paths needed by loader (keep relative if Stage3 kept relative)
            out["splits"][split]["shards"][str(sh["shard_id"])] = {
                "X": files["X"],
                "y": files["y"],
                "w": files["w"],
                "y_raw": files.get("y_raw"),
                "is_valid": files.get("is_valid"),
                "t_sec": files.get("t_sec"),
                "segments": files["segments"],
                "capacity_rows": sh.get("capacity_rows"),
                "valid_rows": sh.get("valid_rows"),
                "num_segments": sh.get("num_segments"),
            }

        out["splits"][split]["blocks"] = split_blocks
        out["counts"][split] = {
            "total_blocks": len(split_blocks),
            "total_end_positions": int(split_total_ends),
            "total_segments": int(split_total_segments),
            "total_shards": int(len(shards)),
        }

        logging.info(
            f"[Stage4] split={split} shards={len(shards)} blocks={len(split_blocks)} "
            f"end_positions={split_total_ends} segments={split_total_segments}"
        )

        # sanity: compare with Stage3's valid_end_rows_for_windowing (should match exactly)
        st3_end = (((st3.get("counts") or {}).get("valid_end_rows_for_windowing") or {}).get(split))

        if st3_end is not None and strict:
            if int(st3_end) != int(split_total_ends):
                raise RuntimeError(
                    f"Stage4 end_positions mismatch vs Stage3! split={split} "
                    f"stage3={st3_end} stage4={split_total_ends}. "
                    f"(If expected, set stage4.strict=false)"
                )

    dt = time.time() - t0
    out["build_seconds"] = float(dt)
    save_json(out_path, out)
    logging.info(f"[Stage4] manifest saved: {out_path}")
    logging.info(f"[Stage4] DONE in {dt:.1f}s")


if __name__ == "__main__":
    main()
