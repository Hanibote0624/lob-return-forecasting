#!/usr/bin/env python3
"""Strict row-aligned offline signal diagnostics for prediction bundles."""

try:
    from .target_contract import load_target_binding, require_target_binding, require_label_binding, load_completed_model_contract
except ImportError:
    from target_contract import load_target_binding, require_target_binding, require_label_binding, load_completed_model_contract

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    from .prediction_io import load_prediction_bundle, prediction_filename, validate_bundle_identity
except ImportError:
    from prediction_io import load_prediction_bundle, prediction_filename, validate_bundle_identity


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def abspath(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def safe_format(template: str, **kwargs) -> str:
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


def load_label_index(path: str) -> Dict[str, dict]:
    result = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = f"{item.get('stock_code', '')}_{item.get('date', '')}_{item.get('session', '')}"
            label_path = item.get("final_label_path") or item.get("raw_label_path")
            if label_path:
                if key in result:
                    raise ValueError(f"duplicate session in label index: {key}")
                result[key] = {
                    "path": label_path, "split": item.get("split"),
                    "stock": str(item["stock_code"]), "date": str(item["date"]),
                    "session": int(item["session"]),
                }
    return result


def calc_ic(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    if pred.size < 2 or np.std(pred) == 0.0 or np.std(true) == 0.0:
        return 0.0, 0.0
    pearson, _ = pearsonr(pred, true)
    spearman, _ = spearmanr(pred, true)
    return float(pearson), float(spearman)


def align_prediction_to_labels(
    bundle: dict,
    true_full: np.ndarray,
    valid_full: np.ndarray,
    label_time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align by explicit original row IDs and verify endpoint timestamps."""

    truth_array = np.asarray(true_full, dtype=np.float32).reshape(-1)
    validity_array = np.asarray(valid_full, dtype=bool).reshape(-1)
    time_array = np.asarray(label_time, dtype=np.float64).reshape(-1)
    if not (truth_array.size == validity_array.size == time_array.size):
        raise ValueError("label arrays have inconsistent row counts")
    prediction = np.asarray(bundle["pred"], dtype=np.float32).reshape(-1)
    end_rows = np.asarray(bundle["end_row"], dtype=np.int64).reshape(-1)
    prediction_time = np.asarray(bundle["t_sec"], dtype=np.float64).reshape(-1)
    if not (prediction.size == end_rows.size == prediction_time.size):
        raise ValueError("prediction bundle arrays have inconsistent row counts")
    if end_rows.size == 0 or end_rows[0] < 0 or end_rows[-1] >= truth_array.size:
        raise ValueError("prediction endpoint rows exceed the label array")
    if np.any(np.diff(end_rows) <= 0):
        raise ValueError("prediction endpoint rows must be strictly increasing")
    if not np.allclose(prediction_time, time_array[end_rows], rtol=0.0, atol=1e-9):
        raise ValueError("prediction timestamps do not match Stage2 rows")
    truth = truth_array[end_rows]
    mask = validity_array[end_rows] & np.isfinite(prediction) & np.isfinite(truth)
    return prediction, truth, mask, end_rows


def find_events(mask: np.ndarray, end_rows: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Return true runs, breaking whenever original endpoint rows are not adjacent."""

    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if end_rows is None:
        rows = np.arange(selected.size, dtype=np.int64)
    else:
        rows = np.asarray(end_rows, dtype=np.int64).reshape(-1)
        if rows.size != selected.size:
            raise ValueError("mask/end_rows length mismatch")
    events = []
    start = None
    for index, is_selected in enumerate(selected):
        contiguous = index > 0 and rows[index] == rows[index - 1] + 1
        if is_selected and (start is None or not contiguous):
            if start is not None:
                events.append((start, index - start))
            start = index
        elif not is_selected and start is not None:
            events.append((start, index - start))
            start = None
    if start is not None:
        events.append((start, selected.size - start))
    return events


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    project_root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]
    eval_cfg = cfg.get("accuracy_eval", {})
    run_name = cfg.get("predict", {}).get("run_name") or cfg.get("train", {}).get("run_name")
    eval_splits = set(eval_cfg.get("splits", ["test"]))
    if not eval_splits or eval_splits - set(cfg["data"]["splits"]):
        raise ValueError("unknown or empty evaluation splits")
    strict_complete = bool(eval_cfg.get("strict_complete", True))

    pred_root = abspath(
        project_root,
        safe_format(
            eval_cfg.get("pred_root", "predictions/{horizon_id}/{run_name}"),
            horizon_id=horizon_id,
            run_name=run_name,
        ),
    )
    report_path = abspath(
        project_root,
        safe_format(
            eval_cfg.get("report_path", "predictions/{horizon_id}/{run_name}/report_metrics.json"),
            horizon_id=horizon_id,
            run_name=run_name,
        ),
    )
    detail_path = abspath(
        project_root,
        safe_format(
            eval_cfg.get("detail_csv_path", "predictions/{horizon_id}/{run_name}/detail_metrics.csv"),
            horizon_id=horizon_id,
            run_name=run_name,
        ),
    )
    topk_fractions = [float(value) for value in eval_cfg.get("topk", {}).get("drops", [0.01])]
    if any(not np.isfinite(value) or value <= 0.0 or value >= 0.5 for value in topk_fractions):
        raise ValueError("accuracy_eval.topk.drops must be between 0 and 0.5")
    cost_rate = float(eval_cfg.get("cost_rate", 0.0))

    label_index_path = abspath(
        project_root,
        safe_format(cfg["stage2"]["labels_index_path"], horizon_id=horizon_id),
    )
    binding = load_target_binding(cfg)
    _, model_fingerprint = load_completed_model_contract(cfg, binding)
    label_index = load_label_index(label_index_path)

    tasks = []
    for item in label_index.values():
        if item["split"] in eval_splits:
            bounds = cfg["data"]["splits"][item["split"]]
            if not bounds["start"] <= item["date"] <= bounds["end"]:
                raise ValueError("label index split conflicts with configured dates")
            tasks.append({
                "stock": item["stock"], "date": item["date"], "session": item["session"],
                "split": item["split"],
                "pred": os.path.join(pred_root, item["stock"], prediction_filename(item["date"], item["session"])),
                "label": abspath(project_root, item["path"]),
            })
    if not tasks:
        raise RuntimeError(f"no prediction bundles found for evaluation splits {sorted(eval_splits)}")
    missing_splits = eval_splits - {task["split"] for task in tasks}
    if missing_splits:
        raise ValueError(f"label index has no sessions for evaluation splits: {sorted(missing_splits)}")

    all_data = []
    pooled_predictions = []
    pooled_truth = []
    session_metrics = []
    failures = []
    contract_fingerprints = set()
    for task in tasks:
        try:
            bundle = load_prediction_bundle(task["pred"])
            metadata = bundle["metadata"]
            fingerprint = validate_bundle_identity(
                bundle, stock=task["stock"], date=task["date"], session=task["session"],
                split=task["split"], run_name=run_name, horizon_id=horizon_id,
            )
            if fingerprint != model_fingerprint:
                raise ValueError("prediction/model contract fingerprint mismatch")
            require_target_binding(metadata, binding)
            contract_fingerprints.add(fingerprint)

            with np.load(task["label"], allow_pickle=False) as labels:
                require_label_binding(labels, binding, metadata.get("source_sha256"))
                true_full = np.asarray(labels["r_raw"], dtype=np.float32)
                valid_full = np.asarray(labels["is_valid"], dtype=np.uint8).astype(bool)
                label_time = np.asarray(labels["t_sec"], dtype=np.float64)
            if metadata["source_row_count"] != true_full.size:
                raise ValueError("source row count differs from Stage2 labels")
            prediction, truth, mask, end_rows = align_prediction_to_labels(
                bundle,
                true_full,
                valid_full,
                label_time,
            )
            if not np.any(mask):
                raise ValueError("session has no finite, label-valid prediction rows")
            ic, rank_ic = calc_ic(prediction[mask], truth[mask])
            session_metrics.append(
                {
                    "stock": task["stock"],
                    "date": task["date"],
                    "session": task["session"],
                    "split": task["split"],
                    "n_predictions": int(prediction.size),
                    "n_valid_labels": int(np.sum(mask)),
                    "ic": ic,
                    "rank_ic": rank_ic,
                }
            )
            pooled_predictions.append(prediction[mask])
            pooled_truth.append(truth[mask])
            all_data.append(
                {
                    "pred": prediction,
                    "true": truth,
                    "mask": mask,
                    "end_row": end_rows,
                }
            )
        except Exception as exc:
            failures.append({"prediction": task["pred"], "error": f"{type(exc).__name__}: {exc}"})
            logging.error("Evaluation failed for %s: %s", task["pred"], exc)

    if failures and strict_complete:
        save_json(report_path, {"status": "failed", "failures": failures})
        raise RuntimeError(f"{len(failures)} evaluation session(s) failed strict validation")
    if len(contract_fingerprints) != 1:
        raise ValueError("evaluation bundles must all come from one model contract")
    if not pooled_predictions:
        raise RuntimeError("no valid predictions remained after strict alignment")

    flat_prediction = np.concatenate(pooled_predictions)
    flat_truth = np.concatenate(pooled_truth)
    pooled_ic, pooled_rank_ic = calc_ic(flat_prediction, flat_truth)
    threshold_report = {}
    for fraction in topk_fractions:
        threshold_long = float(np.quantile(flat_prediction, 1.0 - fraction))
        threshold_short = float(np.quantile(flat_prediction, fraction))
        direction_totals = {
            "long": [0, 0.0, 0],
            "short": [0, 0.0, 0],
        }
        for data in all_data:
            for direction, selected in (
                ("long", data["mask"] & (data["pred"] > threshold_long)),
                ("short", data["mask"] & (data["pred"] < threshold_short)),
            ):
                for start, length in find_events(selected, data["end_row"]):
                    event_truth = data["true"][start : start + length]
                    signed_return = float(np.mean(event_truth)) * (1.0 if direction == "long" else -1.0)
                    totals = direction_totals[direction]
                    totals[0] += 1
                    totals[1] += signed_return
                    totals[2] += length

        def event_metrics(values):
            count, return_sum, row_count = values
            if count == 0:
                return None
            average = return_sum / count
            return {
                "n_events": int(count),
                "avg_event_rows": float(row_count / count),
                "avg_label_return": float(average),
                "avg_label_return_after_assumed_cost": float(average - cost_rate),
            }

        long_metrics = event_metrics(direction_totals["long"])
        short_metrics = event_metrics(direction_totals["short"])
        combined_values = [
            direction_totals["long"][index] + direction_totals["short"][index]
            for index in range(3)
        ]
        threshold_report[f"Top_{fraction * 100}%"] = {
            "threshold_long": threshold_long,
            "threshold_short": threshold_short,
            "long": long_metrics,
            "short": short_metrics,
            "combined": event_metrics(combined_values),
        }

    detail = pd.DataFrame(session_metrics)
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)
    detail.to_csv(detail_path, index=False)
    mean_ic = float(detail["ic"].mean())
    ic_std = float(detail["ic"].std()) if len(detail) > 1 else 0.0
    icir = mean_ic / ic_std if np.isfinite(ic_std) and ic_std > 0.0 else 0.0
    report = {
        **binding,
        "prediction_units": "scaled_clipped_return",
        "truth_units": "raw_unclipped_return",
        "status": "partial" if failures else "complete",
        "run_name": run_name,
        "horizon_id": horizon_id,
        "evaluation_splits": sorted(eval_splits),
        "metrics": {
            "session_ic_mean": mean_ic,
            "session_icir": float(icir),
            "session_rank_ic_mean": float(detail["rank_ic"].mean()),
            "pooled_ic": pooled_ic,
            "pooled_rank_ic": pooled_rank_ic,
            "valid_prediction_rows": int(flat_prediction.size),
        },
        "offline_pooled_signal_thresholds": threshold_report,
        "failed_sessions": failures,
        "notes": [
            "Thresholds in this report are descriptive and estimated on the evaluated split.",
            "They must not be reused as causal trading thresholds; the backtest calibrates on validation only.",
            "Event duration is reported in observed rows because source events are irregularly spaced.",
        ],
        "config": {"assumed_cost_per_event": cost_rate},
    }
    save_json(report_path, report)
    logging.info(
        "[Accuracy] sessions=%d rows=%d pooled_ic=%.4f report=%s",
        len(detail),
        flat_prediction.size,
        pooled_ic,
        report_path,
    )


if __name__ == "__main__":
    main()
