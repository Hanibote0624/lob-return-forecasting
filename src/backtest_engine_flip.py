#!/usr/bin/env python3
"""Optional flip research backtest with prior-split threshold calibration."""

try:
    from .target_contract import load_target_binding, require_target_binding, load_completed_model_contract
except ImportError:
    from target_contract import load_target_binding, require_target_binding, load_completed_model_contract

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .artifact_contract import require_fingerprint
    from .data_contract import hhmmssmmm_to_seconds_vec
    from .backtest_core import (
        build_position_segments,
        calibrate_thresholds,
        require_prior_calibration,
        generate_positions,
    )
    from .prediction_io import load_prediction_bundle, validate_bundle_identity
except ImportError:
    from artifact_contract import require_fingerprint
    from data_contract import hhmmssmmm_to_seconds_vec
    from backtest_core import (
        build_position_segments,
        calibrate_thresholds,
        require_prior_calibration,
        generate_positions,
    )
    from prediction_io import load_prediction_bundle, validate_bundle_identity


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


def load_csv_map(path: str) -> Dict[str, str]:
    result = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("ok") or not item.get("path"):
                continue
            key = f"{item.get('stock_code', '')}_{item.get('date', '')}_{item.get('session', '')}"
            if key in result:
                raise ValueError(f"duplicate session in Stage1 manifest: {key}")
            result[key] = item["path"]
    return result


def split_for_date(date: str, split_cfg: dict) -> str:
    matches = [
        name
        for name, bounds in split_cfg.items()
        if bounds.get("start") <= date <= bounds.get("end")
    ]
    if len(matches) > 1:
        raise ValueError(f"date {date} maps to {len(matches)} chronological splits: {matches}")
    return matches[0] if matches else ""


def load_aligned_quotes(task, cfg):
    bundle = load_prediction_bundle(task["prediction_path"])
    binding = load_target_binding(cfg)
    _, fingerprint = load_completed_model_contract(cfg, binding)
    require_target_binding(bundle["metadata"], binding)
    if bundle["metadata"].get("model_contract_sha256") != fingerprint:
        raise ValueError("prediction/model contract fingerprint mismatch")
    path = task["csv_path"]
    if not path:
        raise FileNotFoundError("raw CSV is missing from the Stage1 manifest")
    require_fingerprint(path, bundle["metadata"].get("source_sha256"))
    required = cfg["data"]["required_columns"]
    bid_col = required["mid_price"].get("bid1", "bid")
    ask_col = required["mid_price"].get("ask1", "ask")
    ts_col = required["timestamp"]
    frame = pd.read_csv(path, usecols=list(dict.fromkeys([bid_col, ask_col, ts_col])), engine="c")
    if len(frame) != bundle["metadata"]["source_row_count"]:
        raise ValueError("raw CSV row count differs from prediction metadata")
    times = hhmmssmmm_to_seconds_vec(frame[ts_col].to_numpy(dtype=np.float64))
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("invalid raw quote timestamps")
    if not np.allclose(times[bundle["end_row"]], bundle["t_sec"], rtol=0.0, atol=1e-9):
        raise ValueError("prediction timestamps differ from raw quote rows")
    bid = frame[bid_col].to_numpy(dtype=np.float64)
    ask = frame[ask_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0) & (ask >= bid)
    return bundle, bid, ask, times, valid


@dataclass
class TradeSegment:
    stock: str
    date: str
    session: int
    direction: int
    start_idx: int
    end_idx: int
    start_row: int
    end_row: int
    start_time_sec: float
    end_time_sec: float
    holding_prediction_rows: int
    holding_seconds: float
    entry_price: float
    exit_price: float
    avg_pred: float
    gross_ret: float
    cost: float
    net_ret: float


class CostModel:
    def __init__(self, cfg: dict):
        self.commission = float(cfg.get("commission_rate", 0.00015))
        self.stamp_tax = float(cfg.get("stamp_tax_rate", 0.0005))
        self.slippage = float(cfg.get("slippage_rate", 0.0))
        rates = [self.commission, self.stamp_tax, self.slippage]
        if not np.isfinite(rates).all() or min(rates) < 0.0:
            raise ValueError("backtest costs must be finite and nonnegative")

    def calc_trade_cost(self, direction: int, entry_price: float, exit_price: float) -> float:
        open_rate = self.commission + self.slippage
        close_rate = self.commission + self.slippage
        if direction == 1:
            close_rate += self.stamp_tax
        elif direction == -1:
            open_rate += self.stamp_tax
        else:
            raise ValueError(f"unexpected trade direction: {direction}")
        price_ratio = exit_price / entry_price
        return float(open_rate + close_rate * price_ratio)


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    project_root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]
    backtest_cfg = cfg.get("backtest", {})
    run_name = (
        backtest_cfg.get("run_name")
        or cfg.get("predict", {}).get("run_name")
        or cfg.get("train", {}).get("run_name")
    )
    trade_splits = set(backtest_cfg.get("splits", ["test"]))
    if not trade_splits or trade_splits - set(cfg["data"]["splits"]):
        raise ValueError("unknown or empty trading splits")
    strict_complete = bool(backtest_cfg.get("strict_complete", True))
    strategy_cfg = backtest_cfg.get("strategy", {})
    topk_long = float(strategy_cfg.get("topk_long", 0.01))
    topk_short = float(strategy_cfg.get("topk_short", 0.01))
    threshold_cfg = backtest_cfg.get("thresholds", {})
    threshold_mode = str(threshold_cfg.get("mode", "calibration_split"))
    calibration_split = str(threshold_cfg.get("source_split", "val"))
    if threshold_mode not in {"calibration_split", "fixed"}:
        raise ValueError("backtest.thresholds.mode must be 'calibration_split' or 'fixed'")
    if threshold_mode == "calibration_split" and calibration_split in trade_splits:
        raise ValueError("threshold calibration split must be disjoint from trading splits")
    if threshold_mode == "calibration_split":
        require_prior_calibration(cfg["data"]["splits"], calibration_split, trade_splits)

    output_cfg = backtest_cfg.get("output", {})
    formatting = {"horizon_id": horizon_id, "run_name": run_name}
    report_path = abspath(
        project_root,
        safe_format(output_cfg.get("report_path", "backtest/{horizon_id}/{run_name}/report.json"), **formatting),
    )
    trade_path = abspath(
        project_root,
        safe_format(output_cfg.get("trade_detail_csv", "backtest/{horizon_id}/{run_name}/trades.csv"), **formatting),
    )
    curve_path = abspath(
        project_root,
        safe_format(
            output_cfg.get(
                "curve_csv",
                "backtest/{horizon_id}/{run_name}/cumulative_trade_returns.csv",
            ),
            **formatting,
        ),
    )
    pred_root = abspath(
        project_root,
        safe_format(cfg["accuracy_eval"]["pred_root"], **formatting),
    )
    csv_map = load_csv_map(abspath(project_root, cfg["stage1"]["manifest_ok_path"]))

    bundle_pattern = re.compile(r"^(\d{8})_(\d+)_pred\.npz$")
    tasks = []
    seen_keys = set()
    scan_failures = []
    contract_fingerprints = set()
    if os.path.exists(pred_root):
        for stock in sorted(os.listdir(pred_root)):
            stock_dir = os.path.join(pred_root, stock)
            if not os.path.isdir(stock_dir):
                continue
            for filename in sorted(os.listdir(stock_dir)):
                match = bundle_pattern.match(filename)
                if not match:
                    continue
                date, session_text = match.groups()
                session = int(session_text)
                key = f"{stock}_{date}_{session}"
                path = os.path.join(stock_dir, filename)
                try:
                    if key in seen_keys:
                        raise ValueError(f"duplicate prediction session: {key}")
                    seen_keys.add(key)
                    chronological_split = split_for_date(date, cfg["data"]["splits"])
                    if chronological_split in trade_splits or (
                        threshold_mode == "calibration_split" and chronological_split == calibration_split
                    ):
                        bundle = load_prediction_bundle(path)
                        fingerprint = validate_bundle_identity(
                            bundle, stock=stock, date=date, session=session, split=chronological_split,
                            run_name=run_name, horizon_id=horizon_id,
                        )
                        contract_fingerprints.add(fingerprint)
                        tasks.append(
                            {
                                "stock": stock,
                                "date": date,
                                "session": session,
                                "split": chronological_split,
                                "prediction_path": path,
                                "csv_path": abspath(project_root, csv_map[key]) if key in csv_map else None,
                            }
                        )
                except Exception as exc:
                    scan_failures.append({"prediction": path, "error": f"{type(exc).__name__}: {exc}"})
    if scan_failures and strict_complete:
        raise RuntimeError(f"{len(scan_failures)} prediction bundle(s) failed backtest validation")
    if len(contract_fingerprints) != 1:
        raise ValueError("backtest bundles must all come from one model contract")

    required_splits = trade_splits | ({calibration_split} if threshold_mode == "calibration_split" else set())
    expected_keys = {
        key for key in csv_map
        if split_for_date(key.rsplit("_", 2)[1], cfg["data"]["splits"]) in required_splits
    }
    task_keys = {f"{t['stock']}_{t['date']}_{t['session']}" for t in tasks}
    if strict_complete and expected_keys != task_keys:
        raise ValueError(f"backtest prediction coverage mismatch: missing={sorted(expected_keys - task_keys)}")

    trade_tasks = [task for task in tasks if task["split"] in trade_splits]
    calibration_tasks = [task for task in tasks if task["split"] == calibration_split]
    if not trade_tasks:
        raise RuntimeError(f"no prediction bundles found for trading splits: {sorted(trade_splits)}")

    if threshold_mode == "calibration_split":
        if not calibration_tasks:
            raise RuntimeError(f"no prediction bundles found for calibration split: {calibration_split}")
        calibration_predictions = []
        for task in calibration_tasks:
            bundle, _, _, _, quote_valid = load_aligned_quotes(task, cfg)
            calibration_rows = bundle["end_row"]
            executable = quote_valid[calibration_rows]
            if not np.any(executable):
                raise ValueError(
                    f"calibration session has no executable prediction rows: "
                    f"{task['stock']} {task['date']}/{task['session']}"
                )
            calibration_predictions.append(bundle["pred"][executable])
        threshold_long, threshold_short = calibrate_thresholds(
            calibration_predictions,
            topk_long,
            topk_short,
        )
        calibration_row_count = int(sum(values.size for values in calibration_predictions))
        threshold_provenance = {
            "mode": threshold_mode,
            "source_split": calibration_split,
            "session_count": len(calibration_tasks),
            "executable_prediction_rows": calibration_row_count,
            "topk_long": topk_long,
            "topk_short": topk_short,
        }
    else:
        if "long" not in threshold_cfg or "short" not in threshold_cfg:
            raise ValueError("fixed threshold mode requires backtest.thresholds.long and .short")
        threshold_long = float(threshold_cfg["long"])
        threshold_short = float(threshold_cfg["short"])
        threshold_provenance = {"mode": threshold_mode}
    if not np.isfinite(threshold_long) or not np.isfinite(threshold_short):
        raise ValueError("backtest thresholds must be finite")
    if threshold_long < threshold_short:
        raise ValueError("long threshold must not be below short threshold")

    logging.info(
        "[Backtest] calibration=%s long>%.6g short<%.6g trade_sessions=%d",
        threshold_provenance,
        threshold_long,
        threshold_short,
        len(trade_tasks),
    )
    cost_model = CostModel(backtest_cfg.get("cost", {}))
    segments: List[TradeSegment] = []
    processing_failures = []

    for task in sorted(trade_tasks, key=lambda item: (item["date"], item["stock"], item["session"])):
        try:
            bundle, bid, ask, endpoint_time, valid_execution = load_aligned_quotes(task, cfg)
            end_rows = bundle["end_row"]
            # Missing history-valid windows mean no new signal, not a command
            # to backdate liquidation before the gap. Keep the full raw timeline.
            prediction = np.full(bid.size, np.nan, dtype=np.float32)
            prediction[end_rows] = bundle["pred"]
            positions = generate_positions(prediction, valid_execution, threshold_long, threshold_short)
            executable_segments = build_position_segments(positions, valid_execution)

            for direction, start, end in executable_segments:
                if direction == 1:
                    entry_price = float(ask[start])
                    exit_price = float(bid[end])
                    gross_return = (exit_price - entry_price) / entry_price
                else:
                    entry_price = float(bid[start])
                    exit_price = float(ask[end])
                    gross_return = (entry_price - exit_price) / entry_price
                cost = cost_model.calc_trade_cost(direction, entry_price, exit_price)
                finite_prediction = prediction[start : end + 1]
                finite_prediction = finite_prediction[np.isfinite(finite_prediction)]
                segments.append(
                    TradeSegment(
                        stock=task["stock"],
                        date=task["date"],
                        session=task["session"],
                        direction=direction,
                        start_idx=start,
                        end_idx=end,
                        start_row=start,
                        end_row=end,
                        start_time_sec=float(endpoint_time[start]),
                        end_time_sec=float(endpoint_time[end]),
                        holding_prediction_rows=int(np.isfinite(prediction[start:end + 1]).sum()),
                        holding_seconds=float(max(0.0, endpoint_time[end] - endpoint_time[start])),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        avg_pred=float(np.mean(finite_prediction)),
                        gross_ret=float(gross_return),
                        cost=float(cost),
                        net_ret=float(gross_return - cost),
                    )
                )
        except Exception as exc:
            processing_failures.append(
                {
                    "stock": task["stock"],
                    "date": task["date"],
                    "session": task["session"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logging.error(
                "Backtest failed for %s %s/%d: %s",
                task["stock"],
                task["date"],
                task["session"],
                exc,
            )
    if processing_failures and strict_complete:
        save_json(report_path, {"status": "failed", "failures": processing_failures})
        raise RuntimeError(f"{len(processing_failures)} trading session(s) failed")

    trade_columns = list(TradeSegment.__dataclass_fields__)
    trades = pd.DataFrame([asdict(segment) for segment in segments], columns=trade_columns)
    os.makedirs(os.path.dirname(trade_path), exist_ok=True)
    trades.to_csv(trade_path, index=False)

    if trades.empty:
        daily = pd.DataFrame(
            columns=[
                "date",
                "summed_net_ret",
                "summed_gross_ret",
                "summed_cost",
                "trade_count",
                "cumulative_summed_net_ret",
            ]
        )
        metrics = {
            "total_trades": 0,
            "win_rate": None,
            "average_trade_net_ret": None,
            "sum_trade_net_returns": 0.0,
            "annualized_ratio_of_daily_summed_trade_returns": None,
            "max_drawdown_of_cumulative_summed_trade_returns": 0.0,
        }
    else:
        daily = (
            trades.groupby("date", as_index=False)
            .agg(
                summed_net_ret=("net_ret", "sum"),
                summed_gross_ret=("gross_ret", "sum"),
                summed_cost=("cost", "sum"),
                trade_count=("stock", "count"),
            )
            .sort_values("date")
        )
        daily["cumulative_summed_net_ret"] = daily["summed_net_ret"].cumsum()
        daily_std = float(daily["summed_net_ret"].std()) if len(daily) > 1 else 0.0
        annualized_ratio = (
            float(daily["summed_net_ret"].mean() / daily_std * np.sqrt(242.0))
            if np.isfinite(daily_std) and daily_std > 0.0
            else None
        )
        cumulative = daily["cumulative_summed_net_ret"]
        max_drawdown = float((cumulative - cumulative.cummax().clip(lower=0.0)).min())
        metrics = {
            "total_trades": int(len(trades)),
            "win_rate": float((trades["net_ret"] > 0.0).mean()),
            "average_trade_net_ret": float(trades["net_ret"].mean()),
            "sum_trade_net_returns": float(trades["net_ret"].sum()),
            "annualized_ratio_of_daily_summed_trade_returns": annualized_ratio,
            "max_drawdown_of_cumulative_summed_trade_returns": max_drawdown,
        }
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    daily.to_csv(curve_path, index=False)

    report = {
        **load_target_binding(cfg),
        "status": "partial" if (scan_failures or processing_failures) else "complete",
        "run_name": run_name,
        "horizon_id": horizon_id,
        "trading_splits": sorted(trade_splits),
        "thresholds": {
            "long": threshold_long,
            "short": threshold_short,
            "provenance": threshold_provenance,
        },
        "cost_model": backtest_cfg.get("cost", {}),
        "metrics": metrics,
        "failed_bundle_validation": scan_failures,
        "failed_trading_sessions": processing_failures,
        "notes": [
            "Threshold provenance is recorded above; split calibration must precede every trading period.",
            "Invalid or crossed quotes cannot open or flip a position; "
            "terminal liquidation fails if the actual session-terminal quote is invalid.",
            "Missing prediction rows hold the current position; they never trigger backdated exits.",
            "Same-row fills and terminal-row closure are idealized offline assumptions, not live execution guarantees.",
            "The reported cumulative series is a sum of per-trade returns, not a capital-account equity curve.",
            "This simplified research backtest omits latency, queue position, "
            "fill probability, and market impact beyond configured costs.",
        ],
    }
    save_json(report_path, report)
    logging.info("[Backtest] trades=%d report=%s", len(trades), report_path)


if __name__ == "__main__":
    main()
