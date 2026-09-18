"""Small deterministic CSV fixtures and independent reference calculations.

The reference uses scalar loops and integer milliseconds, never the production
label, preprocessing, or window-validity helpers. No market data is embedded.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FACTORS = [f"gpmain_{i}" for i in range(3)]
SPLITS = ("train", "val", "test")
ROWS_PER_SESSION = 384


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def artifact_path(cfg, section, field):
    return Path(cfg["project"]["project_root"]) / cfg[section][field].format(
        horizon_id=cfg["horizons"]["active_horizon_id"], run_name=cfg["train"]["run_name"]
    )


def clock_from_ms(total):
    hour, rest = divmod(total, 3_600_000)
    minute, rest = divmod(rest, 60_000)
    second, millisecond = divmod(rest, 1000)
    return hour * 10_000_000 + minute * 100_000 + second * 1000 + millisecond


def ms_from_clock(value):
    clock = int(float(value))
    hour, rest = divmod(clock, 10_000_000)
    minute, rest = divmod(rest, 100_000)
    second, millisecond = divmod(rest, 1000)
    return ((hour * 60 + minute) * 60 + second) * 1000 + millisecond


def create_fixture(root, *, held_out_variant=False):
    """Create a fresh single-stock case; refuse to overwrite an existing tree."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = read_json(ROOT / "config/gp_lit_regression_v6_gpmain_64.example.json")
    cfg["project"].update(project_root=str(root), name="synthetic_contract_check")
    cfg["data"]["stocks"] = [{"stock_code": "000000", "raw_root": "raw/000000"}]
    cfg["data"]["date_range"] = {"start": "20250106", "end": "20250109"}
    cfg["data"]["splits"] = {
        "train": {"start": "20250106", "end": "20250107"},
        "val": {"start": "20250108", "end": "20250108"},
        "test": {"start": "20250109", "end": "20250109"},
    }
    cfg["data"]["required_columns"]["factors"]["count"] = len(FACTORS)
    cfg["data"]["session_rules"].update(
        max_history_span_seconds=1.0, max_inter_event_gap_seconds=0.45
    )
    cfg["features"].update(window_W=8, num_factors=len(FACTORS))
    cfg["horizons"] = {
        "active_horizon_id": "h2_5_3_5",
        "definitions": {"h2_5_3_5": {"min_seconds": 2.5, "max_seconds": 3.5}},
    }
    cfg["label"].update(
        min_future_observations=2,
        fixed_scale=37.0,
        require_full_horizon=True,
        clip_mode="q99_abs_train_only",
    )
    cfg["stage0"]["enabled"] = False
    cfg["stage1"].update(num_workers=1, manifest_tag="synthetic")
    cfg["stage2"].update(num_workers=1, quantile_sample_per_session=10_000)
    cfg["stage3"].update(row_shard_rows=900, feature_norm_map_path=None)
    cfg["stage1_5"]["factor_norm_map_path"] = None
    cfg["stage4"]["block_size_ends"] = 5
    cfg["stage5"].update(
        row_samples_per_split=dict.fromkeys(SPLITS, 64),
        window_samples_per_split=dict.fromkeys(SPLITS, 64),
        strict=True,
    )
    cfg["train"]["global_batch"] = 64
    sessions = []
    steps = [0, 10, 30, 50, 100, 200, 350, 400]
    for day_index, (date, split) in enumerate(
        (
            ("20250106", "train"),
            ("20250107", "train"),
            ("20250108", "val"),
            ("20250109", "test"),
        )
    ):
        for session in (1, 2):
            path = root / "raw/000000" / date / f"000000_{date}_{session}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Start close to a minute boundary to exercise HHMMSSmmm parsing.
            total_ms = (9 * 3600 + 30 * 60 + 59) * 1000 + 980
            if session == 2:
                total_ms = (13 * 3600 + 59) * 1000 + 980
            amplitude = 0.035 if split == "train" else 4.0 + day_index
            offset = 100.0 * (2 * day_index + session)
            if split != "train":
                offset += 10_000.0 * day_index
                if held_out_variant:
                    amplitude *= 2.0
                    offset += 50_000.0
            with path.open("w", newline="", encoding="utf-8") as handle:
                # Deliberately permute the CSV factors relative to schema order.
                writer = csv.writer(handle)
                writer.writerow(
                    ["timestamp", "bid", "ask", "last", FACTORS[2], FACTORS[0], FACTORS[1]]
                )
                for row in range(ROWS_PER_SESSION):
                    if row:
                        total_ms += 6000 if row == 190 else steps[row % len(steps)]
                    # A half-mill price exercises exact midpoint handling.
                    bid = round(100.0 + amplitude * math.sin(row / 13.0) + row * 0.0005, 3)
                    ask = bid + 0.011
                    last = bid + 0.0055
                    if row == 30:  # crossed quotes, recoverable via last
                        bid, ask = ask, bid
                    if row == 31 or 60 <= row < 68:  # unrecoverable prices
                        bid, ask, last = 0.0, 0.0, 0.0
                    f0 = float("nan") if row in (0, 21) else offset + row * 0.5
                    f1 = float("nan") if row < 2 else (row % 17 - 8) * 0.25
                    if row == 48:
                        f1 = float("inf")
                    f2 = float("-inf") if row == 70 else 7.0
                    writer.writerow([clock_from_ms(total_ms), bid, ask, last, f2, f0, f1])
            sessions.append(
                {
                    "stock_code": "000000",
                    "date": date,
                    "session": session,
                    "split": split,
                    "path": str(path),
                }
            )
    config_path = root / "synthetic.local.json"
    write_json(config_path, cfg)
    return cfg, config_path, sessions


def reference_session(path, cfg):
    """Brute-force labels and manual session-local feature filling."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    times_ms = np.array([ms_from_clock(row["timestamp"]) for row in rows], dtype=np.int64)
    mids, features = [], []
    previous = [0.0] * len(FACTORS)
    for row in rows:
        bid, ask, last = (float(row[key]) for key in ("bid", "ask", "last"))
        good_quote = all(math.isfinite(v) and v > 0 for v in (bid, ask)) and ask >= bid
        mids.append((bid + ask) / 2 if good_quote else last)
        for column, factor in enumerate(FACTORS):
            value = float(row[factor])
            if math.isfinite(value):
                previous[column] = value
        features.append(previous.copy())
    mids = np.asarray(mids, dtype=np.float64)
    horizon = cfg["horizons"]["definitions"][cfg["horizons"]["active_horizon_id"]]
    lo_ms, hi_ms = (round(horizon[name] * 1000) for name in ("min_seconds", "max_seconds"))
    raw = np.full(len(rows), np.nan, dtype=np.float32)
    reasons = np.zeros(len(rows), dtype=np.uint8)
    counts, lows, highs = [], [], []
    for index, time in enumerate(times_ms):
        future = [j for j, stamp in enumerate(times_ms) if time + lo_ms <= stamp <= time + hi_ms]
        good = [mids[j] for j in future if math.isfinite(mids[j]) and mids[j] > 0]
        counts.append(len(good))
        lows.append(sum(stamp < time + lo_ms for stamp in times_ms))
        highs.append(sum(stamp <= time + hi_ms for stamp in times_ms) - 1)
        if not math.isfinite(mids[index]) or mids[index] <= 0:
            reasons[index] |= 1
        if time + hi_ms > times_ms[-1]:
            reasons[index] |= 2
        if len(good) < cfg["label"]["min_future_observations"]:
            reasons[index] |= 4
        if reasons[index] == 0:
            raw[index] = math.fsum(good) / len(good) / mids[index] - 1.0
    valid = reasons == 0
    endpoints, excluded_span, excluded_gap = [], 0, 0
    window = cfg["features"]["window_W"]
    rules = cfg["data"]["session_rules"]
    for end in range(window - 1, len(rows)):
        if not valid[end]:
            continue
        history = times_ms[end - window + 1 : end + 1]
        span_bad = history[-1] - history[0] > round(rules["max_history_span_seconds"] * 1000)
        gap_bad = any(
            int(b - a) > round(rules["max_inter_event_gap_seconds"] * 1000)
            for a, b in zip(history, history[1:])
        )
        excluded_span += int(span_bad)
        excluded_gap += int(gap_bad)
        if not span_bad and not gap_bad:
            endpoints.append(end)
    return {
        "t_sec": times_ms.astype(np.float64) / 1000,
        "mid": mids,
        "X": np.asarray(features, dtype=np.float32),
        "r_raw": raw,
        "is_valid": valid.astype(np.uint8),
        "invalid_reason": reasons,
        "future_count": np.asarray(counts, dtype=np.int32),
        "lo": np.asarray(lows, dtype=np.int32),
        "hi": np.asarray(highs, dtype=np.int32),
        "endpoints": endpoints,
        "span_exclusions": excluded_span,
        "gap_exclusions": excluded_gap,
    }
