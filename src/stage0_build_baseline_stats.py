#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage0 Baseline Builder
[Optimized Version: Multiprocessing Support]

This script precomputes the historical baseline statistics used by Stage0's
"baseline deviation" factors.

Optimizations:
- Parallel execution using ProcessPoolExecutor.
- Depends on the PyArrow optimizations in stage0_enrich_raw_csv.py.
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

# Multiprocessing support
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def dynamic_import(module_name: str, file_path: str):
    """Import a python file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def infer_stock_date_session(r: dict) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    sc = r.get("stock_code")
    d = r.get("date")
    sess = r.get("session")
    try:
        sess_i = int(sess) if sess is not None else None
    except Exception:
        sess_i = None
    return (str(sc) if sc is not None else None, str(d) if d is not None else None, sess_i)


# ---------------------------------------------------------
# Worker Function for Parallel Execution
# Must be at top-level for pickling
# ---------------------------------------------------------
def _worker_process_key(
    key_tuple: Tuple,
    cfg: dict,
    project_root: str,
    s0: dict,
    idx_all: Dict,
    stage0_script_path: str
):
    """
    Worker function to build stats for a single key (stock, date, session).
    It re-imports stage0 locally to ensure isolation and proper context in subprocess.
    """
    sc, d, sess = key_tuple

    # Import stage0 logic inside the worker
    try:
        stage0 = dynamic_import("stage0_enrich_raw_csv", stage0_script_path)
    except Exception as e:
        return {
            "key": key_tuple,
            "status": "failed",
            "error": f"Import failed: {str(e)}"
        }

    try:
        out_path = stage0.build_baseline_stats_for_key(
            cfg=cfg,
            project_root=project_root,
            s0=s0,
            idx_all=idx_all,
            stock_code=sc,
            date=d,
            session=sess,
        )
        return {
            "key": key_tuple,
            "status": "processed",
            "out_path": out_path
        }
    except Exception as e:
        return {
            "key": key_tuple,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(limit=3)
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", type=str, default ="config/gp_lit_regression_v6_gpmain_64.json")
    ap.add_argument("--stage0-script-path", type=str, default=None,
                    help="Path to Stage0 implementation file (default: <config_dir>/../src/stage0_enrich_raw_csv.py if exists, else ./src/stage0_enrich_raw_csv.py)")
    ap.add_argument("--stock-code", type=str, default=None, help="optional filter")
    ap.add_argument("--date", type=str, default=None, help="optional filter YYYYMMDD")
    ap.add_argument("--session", type=int, default=None, help="optional filter session int")
    ap.add_argument("--max-files", type=int, default=None, help="optional override")
    ap.add_argument("--workers", type=int, default=None, help="Manual override for worker count (default: stage0.num_workers or 4)")
    args = ap.parse_args()

    cfg = load_json(args.config_path)

    project_root = cfg.get("project", {}).get("project_root")
    if not project_root:
        raise KeyError("config.project.project_root is missing")

    s0 = cfg.get("stage0", {}) or {}
    bcfg = (s0.get("baseline") or {})
    if not bool(bcfg.get("enabled", False)):
        print("[Stage0Baseline] baseline.enabled=0, nothing to do")
        return 0

    s0b = cfg.get("stage0_baseline_stats", {}) or {}
    summary_path = s0b.get("summary_path") or "data/stage0_baseline_stats/stage0_baseline_stats_summary.json"
    summary_path = resolve_path(project_root, summary_path)

    overwrite_stats = bool(s0b.get("overwrite", False)) or bool(bcfg.get("overwrite_stats", False))

    stage0_script_path = args.stage0_script_path
    if stage0_script_path is None:
        cand1 = os.path.join(project_root, "src", "stage0_enrich_raw_csv.py")
        cand2 = os.path.join(os.getcwd(), "src", "stage0_enrich_raw_csv.py")
        cand3 = os.path.join(os.path.dirname(os.path.abspath(args.config_path)), "..", "src", "stage0_enrich_raw_csv.py")
        for c in (cand1, cand3, cand2):
            if os.path.isfile(c):
                stage0_script_path = c
                break
    if stage0_script_path is None or (not os.path.isfile(stage0_script_path)):
        raise FileNotFoundError(f"Stage0 script not found. Provide --stage0-script-path. Tried: {stage0_script_path}")

    # Initial import to setup scanning
    stage0 = dynamic_import("stage0_enrich_raw_csv", stage0_script_path)

    date_start = str(cfg.get("data", {}).get("date_range", {}).get("start", "00000000"))
    date_end = str(cfg.get("data", {}).get("date_range", {}).get("end", "99999999"))

    sessions_allowed = None
    try:
        sessions_allowed = cfg.get("data", {}).get("file_pattern", {}).get("sessions")
    except Exception:
        sessions_allowed = None

    use_manifest = bool(s0b.get("use_stage1_ok_manifest", s0.get("use_stage1_ok_manifest", False)))

    rows: List[dict] = []
    if use_manifest:
        stage1_cfg = cfg.get("stage1", {}) or {}
        ok_path = resolve_path(project_root, stage1_cfg.get("manifest_ok_path"))
        if (not ok_path) or (not os.path.isfile(ok_path)):
            print(f"[Stage0Baseline] WARN: use_stage1_ok_manifest=1 but manifest not found. Falling back to raw_root scan.")
            use_manifest = False
        else:
            rows = stage0.iter_stage1_ok_manifest(ok_path)

    if not use_manifest:
        specs = stage0.get_stock_specs(cfg)
        rows = stage0.scan_raw_roots(specs, date_start, date_end, sessions_allowed=sessions_allowed)

    rows_all = list(rows)

    def _keep(r: dict) -> bool:
        sc, d, sess = infer_stock_date_session(r) if "date" in r else (r.get("stock_code"), r.get("date"), r.get("session"))
        if args.stock_code is not None and sc is not None and str(sc) != str(args.stock_code):
            return False
        if args.date is not None and d is not None and str(d) != str(args.date):
            return False
        if args.session is not None and sess is not None and int(sess) != int(args.session):
            return False
        return True

    rows_need = [r for r in rows if _keep(r)]

    max_files = args.max_files if args.max_files is not None else int(s0b.get("max_files", 0) or 0)
    if max_files and len(rows_need) > max_files:
        rows_need = rows_need[:max_files]

    idx_all = stage0._build_index(rows_all)

    keys = []
    for r in rows_need:
        sc, d, sess = infer_stock_date_session(r)
        if sc is None or d is None:
            continue
        keys.append((str(sc), str(d), sess))
    keys = sorted(set(keys), key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2]))

    summary = {
        "stage": "stage0_baseline_stats",
        "config_path": os.path.abspath(args.config_path),
        "stage0_script_path": os.path.abspath(stage0_script_path),
        "keys_total": len(keys),
        "processed": 0,
        "skipped_exists": 0,
        "failed": 0,
        "errors": [],
        "outputs": [],
    }

    if overwrite_stats:
        if "baseline" not in s0:
            s0["baseline"] = {}
        s0["baseline"]["overwrite_stats"] = True
        cfg["stage0"] = s0

    # Determine num_workers
    num_workers = args.workers
    if num_workers is None:
        num_workers = int(s0.get("num_workers", 4))

    print(f"[Stage0Baseline] Processing {len(keys)} items with {num_workers} workers...")

    # Tasks to run
    tasks = []
    for k in keys:
        sc, d, sess = k
        out_path = stage0._baseline_stats_path(project_root, s0, sc, d, sess)
        if (not overwrite_stats) and os.path.isfile(out_path):
            summary["skipped_exists"] += 1
            continue
        tasks.append(k)

    if not tasks:
        print("[Stage0Baseline] Nothing to do (all exist and overwrite=False).")
        return 0

    # Parallel Execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _worker_process_key,
                task_key, cfg, project_root, s0, idx_all, stage0_script_path
            ): task_key for task_key in tasks
        }

        done_count = 0
        total_tasks = len(tasks)

        for future in as_completed(futures):
            done_count += 1
            if done_count % 10 == 0:
                print(f"[Stage0Baseline] Progress: {done_count}/{total_tasks}")

            res = future.result()
            key_tuple = res["key"]
            sc, d, sess = key_tuple

            if res["status"] == "processed":
                summary["processed"] += 1
                summary["outputs"].append({
                    "stock_code": sc, "date": d, "session": sess, "baseline_path": res.get("out_path")
                })
            else:
                summary["failed"] += 1
                summary["errors"].append({
                    "stock_code": sc,
                    "date": d,
                    "session": sess,
                    "error": res.get("error"),
                    "traceback": res.get("traceback")
                })

    if summary_path:
        save_json(summary_path, summary)
        print(f"[Stage0Baseline] summary saved: {summary_path}")

    print(f"[Stage0Baseline] keys={len(keys)} processed={summary['processed']} skipped={summary['skipped_exists']} failed={summary['failed']}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())