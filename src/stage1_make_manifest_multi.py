#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1: Scan raw CSVs -> build session manifest (jsonl) + summary (json) + split manifests.
- Robust to missing days/sessions.
- Validates required columns & factor schema.
- Computes fast row counts, first/last timestamps, and small timestamp quality stats.
- Prepares outputs used by Stage2/3 (no future leaks; split lists frozen here).

Assumptions:
- Files like: local/raw_data/SAMPLE/20250102/SAMPLE_20250102_1.csv
- timestamp like HHMMSSmmm as float (e.g. 130000000.0) -> convert to seconds.
"""

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import csv  # Added for robust CSV parsing (handles quotes)
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Utilities
# -----------------------------

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def append_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def yyyymmdd_in_range(d: str, start: str, end: str) -> bool:
    return start <= d <= end

def date_to_int(d: str) -> int:
    return int(d)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def count_lines_fast(path: str, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Count '\n' quickly in binary mode. Returns number of lines including header line."""
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            n += b.count(b"\n")
    return n

def tail_last_nonempty_line(path: str, max_bytes: int = 2 * 1024 * 1024) -> Optional[str]:
    """Read last non-empty line. Avoid loading entire file."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        read_size = min(size, max_bytes)
        f.seek(size - read_size, os.SEEK_SET)
        buf = f.read(read_size)
    # Splitlines handles different endings; we want last non-empty
    lines = buf.splitlines()
    for line in reversed(lines):
        if line.strip():
            try:
                return line.decode("utf-8", errors="ignore")
            except Exception:
                return None
    return None

def parse_header_columns(header_line: str) -> List[str]:
    """
    Parse CSV header using the csv module to correctly handle quotes.
    e.g. '"col1","col2"' -> ['col1', 'col2']
    """
    if not header_line:
        return []
    # Use csv.reader to handle quotes automatically
    reader = csv.reader([header_line.strip()])
    try:
        row = next(reader)
        return [c.strip() for c in row if c]
    except StopIteration:
        return []

def parse_csv_line_simple(line: str) -> List[str]:
    """
    Parse a CSV data line using the csv module to correctly handle quotes.
    """
    if not line:
        return []
    reader = csv.reader([line.strip()])
    try:
        row = next(reader)
        return [x.strip() for x in row]
    except StopIteration:
        return []

def ts_hhmmssmmm_to_seconds(ts_val: float, assume_digits: int = 9) -> float:
    """
    Convert e.g. 130000000 -> 13:00:00.000 -> seconds.
    assume_digits=9 => HHMMSSmmm
    """
    try:
        x = int(float(ts_val))
    except Exception:
        return float("nan")
    s = f"{x:0{assume_digits}d}"
    if len(s) != assume_digits:
        # best-effort: pad/trim
        s = s[-assume_digits:].rjust(assume_digits, "0")
    hh = int(s[0:2])
    mm = int(s[2:4])
    ss = int(s[4:6])
    ms = int(s[6:9])
    return hh * 3600.0 + mm * 60.0 + ss + ms / 1000.0

def detect_factor_schema(
    columns: List[str],
    preferred_prefix: str,
    count: int,
    allow_prefixes: List[str],
) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Returns:
      (chosen_prefix, factor_cols_ordered, missing_factor_cols)
    Chooses a prefix that yields complete [prefix0..prefix{count-1}] if possible.
    """
    def build_expected(prefix: str) -> List[str]:
        return [f"{prefix}{i}" for i in range(count)]

    colset = set(columns)

    # try preferred first
    for prefix in [preferred_prefix] + [p for p in allow_prefixes if p != preferred_prefix]:
        expected = build_expected(prefix)
        if all(c in colset for c in expected):
            return prefix, expected, []

    # fallback: find best matching prefix by regex-like presence
    best_prefix = None
    best_present = -1
    best_cols = []
    for prefix in [preferred_prefix] + [p for p in allow_prefixes if p != preferred_prefix]:
        expected = build_expected(prefix)
        present = sum(1 for c in expected if c in colset)
        if present > best_present:
            best_present = present
            best_prefix = prefix
            best_cols = expected

    missing = [c for c in best_cols if c not in colset]
    if best_present <= 0:
        return None, [], best_cols  # treat all as missing if nothing matched
    return best_prefix, best_cols, missing

def infer_split(date: str, splits: dict) -> Optional[str]:
    for k in ["train", "val", "test"]:
        s = splits.get(k, {})
        if s.get("start") and s.get("end") and yyyymmdd_in_range(date, s["start"], s["end"]):
            return k
    return None


def resolve_path(project_root: str, p: Optional[str]) -> Optional[str]:
    """Resolve a possibly-relative path against project_root."""
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(project_root, p)


@dataclass
class StockSpec:
    stock_code: Optional[str]
    out_root: str


def get_stock_specs(cfg: dict) -> List[StockSpec]:
    """Return list of StockSpec.

    Backward compatible:
      - If cfg['data']['stocks'] exists and is non-empty, use it.
      - Else fall back to single-stock fields: data.out_root + data.stock_code.
    """
    data = cfg.get("data", {})
    stocks = data.get("stocks", None)
    s0 = cfg.get("stage0",{})
    s0_enable = s0.get("enabled", False)
    if isinstance(stocks, list) and len(stocks) > 0:
        out: List[StockSpec] = []
        for s in stocks:
            if not isinstance(s, dict):
                continue
            if bool(s0_enable):
                out_root = s.get("out_root")
            else:
                out_root = s.get("raw_root")
            if not out_root:
                continue
            out.append(StockSpec(stock_code=s.get("stock_code"), out_root=out_root))
        if out:
            return out

    # fallback to legacy single-stock config
    out_root = data.get("out_root")
    if not out_root:
        raise KeyError("config.data.out_root missing and config.data.stocks empty")
    return [StockSpec(stock_code=data.get("stock_code", None), out_root=out_root)]


def derive_manifest_tag(cfg: dict, stock_specs: List[StockSpec]) -> str:
    """Derive a stable tag for manifest filenames."""
    stage1 = cfg.get("stage1", {}) or {}
    tag = stage1.get("manifest_tag")
    if isinstance(tag, str) and tag.strip():
        return tag.strip()

    codes = sorted({str(s.stock_code) for s in stock_specs if s.stock_code})
    if len(codes) >= 2:
        return "combined_" + "_".join(codes)
    if len(codes) == 1:
        return codes[0]
    return "all"

# -----------------------------
# Worker inspect
# -----------------------------

@dataclass
class InspectArgs:
    path: str
    stock_code: Optional[str]
    out_root: Optional[str]
    date: str
    session: int
    cfg: dict
    strict: bool

def inspect_one(arg: InspectArgs) -> dict:
    path = arg.path
    cfg = arg.cfg
    strict = arg.strict

    out = {
        "path": path,
        "stock_code": arg.stock_code,
        "out_root": arg.out_root,
        "date": arg.date,
        "session": arg.session,
        # Important: must be unique across stocks
        "session_id": f"{arg.stock_code}_{arg.date}_{arg.session}" if arg.stock_code else f"{arg.date}_{arg.session}",
        "ok": False,
        "error": None,
        "n_rows": None,
        "size_bytes": None,
        "mtime_ns": None,
        "columns": None,
        "missing_required_cols": [],
        "factor_prefix": None,
        "num_factors_expected": cfg["data"]["required_columns"]["factors"]["count"],
        "num_factors_found": None,
        "missing_factor_cols": [],
        "timestamp": {
            "t_first_sec": None,
            "t_last_sec": None,
            "sample_dt_median_ms": None,
            "sample_dt_p99_ms": None,
            "sample_monotonic_non_decreasing": None
        }
    }

    try:
        st = os.stat(path)
        out["size_bytes"] = st.st_size
        out["mtime_ns"] = st.st_mtime_ns
    except Exception as e:
        out["error"] = f"stat_failed: {e}"
        return out

    # Read header
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
            if not header:
                out["error"] = "empty_file"
                return out
            columns = parse_header_columns(header)
    except Exception as e:
        out["error"] = f"read_header_failed: {e}"
        return out

    out["columns"] = columns
    colset = set(columns)

    # Required columns
    req = cfg["data"]["required_columns"]
    ts_col = req["timestamp"]
    mid_cfg = req["mid_price"]
    bid1 = mid_cfg["bid1"]
    ask1 = mid_cfg["ask1"]
    lastp = mid_cfg["fallback_last"]

    missing = []
    if ts_col not in colset:
        missing.append(ts_col)

    has_bidask = (bid1 in colset) and (ask1 in colset)
    has_last = lastp in colset

    if mid_cfg.get("prefer_bidask", True):
        if not has_bidask and not has_last:
            missing.extend([bid1, ask1, lastp])
    else:
        if not has_last and not has_bidask:
            missing.extend([lastp, bid1, ask1])

    # Factors
    fcfg = req["factors"]
    pref_prefix = fcfg["prefix"]
    allow_prefixes = fcfg.get("allow_prefixes", [pref_prefix])
    fcount = int(fcfg["count"])

    factor_prefix, factor_cols, missing_factors = detect_factor_schema(
        columns=columns,
        preferred_prefix=pref_prefix,
        count=fcount,
        allow_prefixes=allow_prefixes
    )

    out["factor_prefix"] = factor_prefix
    out["num_factors_found"] = len(factor_cols) if factor_cols else 0
    out["missing_factor_cols"] = missing_factors

    # strict required col checks
    out["missing_required_cols"] = missing

    if missing:
        out["error"] = f"missing_required_cols: {missing}"
        if strict:
            return out

    if factor_prefix is None or len(missing_factors) > 0:
        out["error"] = out["error"] or f"factor_schema_incomplete: missing={len(missing_factors)}"
        if strict:
            return out

    # Row count (fast)
    try:
        n_lines = count_lines_fast(path)
        out["n_rows"] = max(0, n_lines - 1)
    except Exception as e:
        out["error"] = out["error"] or f"count_lines_failed: {e}"
        if strict:
            return out

    # First & last timestamp + small sample dt stats
    ts_idx = columns.index(ts_col) if ts_col in colset else None
    assume_digits = int(cfg["data"]["timestamp_parse"].get("assume_int_digits", 9))
    sample_n = int(cfg.get("stage1", {}).get("timestamp_sample_rows", 5000))

    if ts_idx is not None:
        t_list = []
        t_first = None

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()  # header
                for _k in range(sample_n):
                    line = f.readline()
                    if not line:
                        break
                    parts = parse_csv_line_simple(line)
                    if ts_idx >= len(parts):
                        continue
                    tsec = ts_hhmmssmmm_to_seconds(parts[ts_idx], assume_digits=assume_digits)
                    if t_first is None:
                        t_first = tsec
                    if tsec == tsec:  # not nan
                        t_list.append(tsec)
        except Exception as e:
            out["error"] = out["error"] or f"read_sample_failed: {e}"
            if strict:
                return out

        # tail last line timestamp
        t_last = None
        try:
            last_line = tail_last_nonempty_line(path)
            if last_line:
                parts = parse_csv_line_simple(last_line)
                if ts_idx < len(parts):
                    t_last = ts_hhmmssmmm_to_seconds(parts[ts_idx], assume_digits=assume_digits)
        except Exception as e:
            out["error"] = out["error"] or f"tail_failed: {e}"
            if strict:
                return out

        out["timestamp"]["t_first_sec"] = t_first
        out["timestamp"]["t_last_sec"] = t_last

        # dt stats
        if len(t_list) >= 3:
            diffs = []
            mono_ok = True
            prev = t_list[0]
            for x in t_list[1:]:
                if x < prev:
                    mono_ok = False
                d = x - prev
                if d > 0:
                    diffs.append(d)
                prev = x
            out["timestamp"]["sample_monotonic_non_decreasing"] = mono_ok

            if diffs:
                diffs.sort()
                mid = diffs[len(diffs)//2]
                p99 = diffs[int(0.99 * (len(diffs)-1))]
                out["timestamp"]["sample_dt_median_ms"] = mid * 1000.0
                out["timestamp"]["sample_dt_p99_ms"] = p99 * 1000.0

    out["ok"] = (out["error"] is None) or (not strict and out["missing_required_cols"] == [] and len(out["missing_factor_cols"]) == 0)
    if out["ok"]:
        out["error"] = None
    return out

# -----------------------------
# Main
# -----------------------------

def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", type=str, required=True)
    ap.add_argument("--strict", type=int, default=1, help="1=strict (fail on schema issues), 0=non-strict")
    ap.add_argument("--num-workers", type=int, default=0, help="override packing.num_workers for Stage1 scan")
    args = ap.parse_args()

    cfg = load_config(args.config_path)

    project_root = cfg["project"]["project_root"]
    stock_specs = get_stock_specs(cfg)
    date_start = cfg["data"]["date_range"]["start"]
    date_end = cfg["data"]["date_range"]["end"]
    splits = cfg["data"]["splits"]
    manifest_tag = derive_manifest_tag(cfg, stock_specs)

    out_paths = cfg["paths"]
    manifests_dir = os.path.join(project_root, out_paths["manifests_dir"])
    stats_dir = os.path.join(project_root, out_paths["stats_dir"])
    ensure_dir(manifests_dir)
    ensure_dir(stats_dir)

    sessions_allowed = cfg["data"]["file_pattern"].get("sessions", [1, 2])

    # Find all csv files under out_root/*/*.csv
    # Parse filenames like: {code}_{YYYYMMDD}_{session}.csv
    pat = re.compile(r"(?P<code>\d+)_(?P<date>\d{8})_(?P<session>\d+)(?:_.*)?\.csv$", re.IGNORECASE)

    all_files = []
    for spec in stock_specs:
        out_root = spec.out_root
        code_filter = str(spec.stock_code) if spec.stock_code is not None else None

        if not os.path.isdir(out_root):
            logging.warning(f"[Stage1] out_root not found/dir: {out_root} (stock_code={code_filter})")
            continue

        for dname in os.listdir(out_root):
            dpath = os.path.join(out_root, dname)
            if not os.path.isdir(dpath):
                continue
            # dname may be YYYYMMDD but don't assume; we parse filename anyway
            try:
                for fn in os.listdir(dpath):
                    if not fn.lower().endswith(".csv"):
                        continue
                    m = pat.search(fn)
                    if not m:
                        continue
                    code = m.group("code")
                    date = m.group("date")
                    session = int(m.group("session"))
                    if code_filter is not None and str(code) != code_filter:
                        continue
                    if sessions_allowed and session not in sessions_allowed:
                        continue
                    if not yyyymmdd_in_range(date, date_start, date_end):
                        continue
                    full = os.path.join(dpath, fn)
                    all_files.append((full, code, out_root, date, session))
            except FileNotFoundError:
                continue

    all_files.sort(key=lambda x: (x[3], x[4], x[1], x[0]))
    if not all_files:
        logging.error("No matching csv files found under configured data.stocks / data.out_root")
        sys.exit(2)

    num_workers = args.num_workers if args.num_workers > 0 else int(cfg.get("stage1", {}).get("num_workers", 48))
    num_workers = max(1, min(num_workers, 128))

    stock_codes = sorted({str(s.stock_code) for s in stock_specs if s.stock_code is not None})
    out_roots_by_stock = {str(s.stock_code): s.out_root for s in stock_specs if s.stock_code is not None}
    out_roots_all = sorted({s.out_root for s in stock_specs})

    stage1_cfg = cfg.get("stage1", {}) or {}
    manifest_path = resolve_path(
        project_root,
        stage1_cfg.get("manifest_all_path"),
    ) or os.path.join(manifests_dir, f"sessions_manifest_{manifest_tag}_{date_start}_{date_end}.jsonl")
    ok_path = resolve_path(
        project_root,
        stage1_cfg.get("manifest_ok_path"),
    ) or os.path.join(manifests_dir, f"sessions_ok_{manifest_tag}_{date_start}_{date_end}.jsonl")
    bad_path = resolve_path(
        project_root,
        stage1_cfg.get("manifest_bad_path"),
    ) or os.path.join(manifests_dir, f"sessions_bad_{manifest_tag}_{date_start}_{date_end}.jsonl")
    summary_path = resolve_path(
        project_root,
        stage1_cfg.get("summary_path"),
    ) or os.path.join(manifests_dir, f"stage1_summary_{manifest_tag}_{date_start}_{date_end}.json")
    schema_path = resolve_path(
        project_root,
        stage1_cfg.get("schema_path"),
    ) or os.path.join(stats_dir, f"schema_factors_{manifest_tag}.json")

    logging.info(f"[Stage1] project_root={project_root}")
    logging.info(f"[Stage1] manifest_tag={manifest_tag}")
    logging.info(f"[Stage1] stock_codes={stock_codes if stock_codes else 'auto'}")
    logging.info(f"[Stage1] out_roots={out_roots_all}")
    logging.info(f"[Stage1] date_range={date_start}-{date_end}")
    logging.info(f"[Stage1] found_files={len(all_files)}, num_workers={num_workers}, strict={bool(args.strict)}")
    logging.info(f"[Stage1] outputs: ok={ok_path}")

    t0 = time.time()
    tasks = [
        InspectArgs(path=p, stock_code=c, out_root=rr, date=d, session=s, cfg=cfg, strict=bool(args.strict))
        for (p, c, rr, d, s) in all_files
    ]
    with Pool(processes=num_workers) as pool:
        results = list(pool.map(inspect_one, tasks))
    dt = time.time() - t0

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]

    # Infer factor schema (per stock)
    schema = {
        "manifest_tag": manifest_tag,
        "stock_codes": stock_codes,
        "date_range": {"start": date_start, "end": date_end},
        "factor_prefix": None,
        "factor_prefix_by_stock": {},
        "num_factors": cfg["data"]["required_columns"]["factors"]["count"],
        "factor_cols": None,
        "factor_cols_by_stock": {},
        "required_columns": cfg["data"]["required_columns"],
        "timestamp_parse": cfg["data"]["timestamp_parse"]
    }

    if ok:
        # first seen prefix per stock
        for r in ok:
            sc = r.get("stock_code")
            fp = r.get("factor_prefix")
            if sc and fp and sc not in schema["factor_prefix_by_stock"]:
                schema["factor_prefix_by_stock"][sc] = fp
        # common prefix?
        uniq = sorted(set(schema["factor_prefix_by_stock"].values()))
        if len(uniq) == 1:
            schema["factor_prefix"] = uniq[0]
            cnt = int(schema["num_factors"])
            schema["factor_cols"] = [f"{uniq[0]}{i}" for i in range(cnt)]
        # always fill factor_cols_by_stock
        cnt = int(schema["num_factors"])
        for sc, fp in schema["factor_prefix_by_stock"].items():
            schema["factor_cols_by_stock"][sc] = [f"{fp}{i}" for i in range(cnt)]

    # Coverage check by stock/date/session
    coverage_by_stock: Dict[str, dict] = {}
    for sc in sorted(set([x[1] for x in all_files])):
        date_to_sessions: Dict[str, set] = {}
        for r in ok:
            if str(r.get("stock_code")) != str(sc):
                continue
            date_to_sessions.setdefault(r["date"], set()).add(int(r["session"]))

        dates_with_any_file = sorted(set([x[3] for x in all_files if str(x[1]) == str(sc)]))
        missing_days = []
        missing_sessions = []
        for d in dates_with_any_file:
            ss = date_to_sessions.get(d, set())
            for s in sessions_allowed:
                if s not in ss:
                    missing_sessions.append(f"{sc}_{d}_{s}")
            if not ss:
                missing_days.append(d)

        coverage_by_stock[str(sc)] = {
            "dates_with_any_file": dates_with_any_file,
            "missing_sessions": missing_sessions[:2000],
            "missing_sessions_count": len(missing_sessions),
            "missing_days": missing_days,
        }

    # Split manifests
    split_rows = {"train": [], "val": [], "test": []}
    for r in ok:
        sp = infer_split(r["date"], splits)
        r2 = dict(r)
        r2["split"] = sp
        if sp in split_rows:
            split_rows[sp].append(r2)

    append_jsonl(manifest_path, results)
    append_jsonl(ok_path, ok)
    append_jsonl(bad_path, bad)

    for sp, rows in split_rows.items():
        sp_path = os.path.join(manifests_dir, f"sessions_{sp}_{manifest_tag}_{splits[sp]['start']}_{splits[sp]['end']}.jsonl")
        append_jsonl(sp_path, rows)
    save_json(schema_path, schema)

    summary = {
        "project_root": project_root,
        "manifest_tag": manifest_tag,
        "stock_codes": stock_codes,
        "out_roots_by_stock": out_roots_by_stock,
        "date_range": {"start": date_start, "end": date_end},
        "scan": {
            "found_files": len(all_files),
            "ok_files": len(ok),
            "bad_files": len(bad),
            "seconds": dt,
            "num_workers": num_workers,
            "strict": bool(args.strict)
        },
        "factor_schema": {
            "factor_prefix": schema["factor_prefix"],
            "num_factors": schema["num_factors"],
            "schema_path": schema_path
        },
        "coverage_by_stock": coverage_by_stock,
        "outputs": {
            "manifest_all": manifest_path,
            "manifest_ok": ok_path,
            "manifest_bad": bad_path,
            "manifests_dir": manifests_dir
        },
        "splits": {
            "train_count": len(split_rows["train"]),
            "val_count": len(split_rows["val"]),
            "test_count": len(split_rows["test"])
        }
    }
    save_json(summary_path, summary)

    logging.info(f"[Stage1] done. ok={len(ok)} bad={len(bad)} in {dt:.1f}s")
    logging.info(f"[Stage1] manifest_all: {manifest_path}")
    logging.info(f"[Stage1] summary: {summary_path}")

    # Strict mode: fail fast if any bad exists
    if bool(args.strict) and bad:
        logging.error("[Stage1] strict=1 and bad files exist. See sessions_bad*.jsonl and summary.")
        sys.exit(3)

if __name__ == "__main__":
    main()
