#!/usr/bin/env python3
"""Stream row-aligned predictions from raw session CSVs using the training contract."""

try:
    from .target_contract import MODEL_CONTRACT_VERSION, load_target_binding, require_target_binding
except ImportError:
    from target_contract import MODEL_CONTRACT_VERSION, load_target_binding, require_target_binding

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import json
import logging
import os
import sys
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .artifact_contract import require_fingerprint, sha256_file
    from .data_contract import hhmmssmmm_to_seconds_vec, valid_window_end_mask
    from .feature_preprocessing import (
        PREPROCESSING_VERSION,
        parse_norm_groups,
        preprocess_session_features,
        required_feature_columns,
        require_preprocessing_spec,
    )
    from .prediction_io import (
        iter_window_batches,
        load_prediction_bundle,
        prediction_filename,
        save_prediction_bundle,
    )
    from .stage6_train_regression import FixedStandardize, TimeAwarePositionalEncoding
except ImportError:
    from artifact_contract import require_fingerprint, sha256_file
    from data_contract import hhmmssmmm_to_seconds_vec, valid_window_end_mask
    from feature_preprocessing import (
        PREPROCESSING_VERSION,
        parse_norm_groups,
        preprocess_session_features,
        required_feature_columns,
        require_preprocessing_spec,
    )
    from prediction_io import (
        iter_window_batches,
        load_prediction_bundle,
        prediction_filename,
        save_prediction_bundle,
    )
    from stage6_train_regression import FixedStandardize, TimeAwarePositionalEncoding


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def abspath(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def fmt(template: str, **kwargs) -> str:
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


def split_for_date(date: str, split_names: List[str], split_cfg: dict) -> str:
    matches = [
        name
        for name in split_names
        if name in split_cfg and split_cfg[name]["start"] <= date <= split_cfg[name]["end"]
    ]
    if len(matches) > 1:
        raise ValueError(f"date {date} belongs to overlapping requested splits: {matches}")
    return matches[0] if matches else ""


def configure_gpu(gpu_id: int) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logging.warning("No GPU detected; inference will use the available TensorFlow device.")
        return
    if gpu_id < 0 or gpu_id >= len(gpus):
        raise ValueError(f"gpu_id={gpu_id} is out of range for {len(gpus)} visible GPU(s)")
    tf.config.set_visible_devices(gpus[gpu_id], "GPU")
    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    project_root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]
    pred_cfg = cfg.get("predict", {})
    run_name = pred_cfg.get("run_name") or cfg["train"]["run_name"]
    output_root = abspath(
        project_root,
        fmt(
            pred_cfg.get("output_root", "predictions/{horizon_id}/{run_name}"),
            horizon_id=horizon_id,
            run_name=run_name,
        ),
    )
    split_names = list(pred_cfg.get("splits", ["val", "test"]))
    unknown_splits = set(split_names).difference(cfg["data"]["splits"])
    if unknown_splits:
        raise ValueError(f"unknown prediction splits: {sorted(unknown_splits)}")
    batch_size = int(pred_cfg.get("batch_size", 4096))
    if batch_size < 1:
        raise ValueError("predict.batch_size must be positive")
    strict_complete = bool(pred_cfg.get("strict_complete", True))
    configure_gpu(int(pred_cfg.get("gpu_id", 0)))

    contract_path = abspath(
        project_root,
        fmt(
            pred_cfg.get("contract_path", "results/{run_name}/model_contract.json"),
            run_name=run_name,
            horizon_id=horizon_id,
        ),
    )
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"model contract not found: {contract_path}; retrain with the current Stage 6")
    contract = load_json(contract_path)
    contract_sha256 = sha256_file(contract_path)
    if int(contract.get("contract_version", -1)) != MODEL_CONTRACT_VERSION:
        raise ValueError("unsupported model contract version")
    if int(contract.get("preprocessing_version", -1)) != PREPROCESSING_VERSION:
        raise ValueError("model preprocessing version is incompatible with this predictor")
    if contract.get("status") != "trained" or contract.get("run_name") != run_name:
        raise ValueError("model contract must identify a completed training run with the requested run_name")
    if contract.get("horizon_id") != horizon_id:
        raise ValueError(
            f"model horizon={contract.get('horizon_id')} does not match config horizon={horizon_id}"
        )

    binding = load_target_binding(cfg)
    require_target_binding(contract, binding)

    packs_path = abspath(
        project_root,
        cfg["stage3"]["packs_manifest_path"].format(horizon_id=horizon_id),
    )
    packs = load_json(packs_path)
    require_target_binding(packs, binding)
    schema_path = abspath(project_root, packs["factor_schema_path"])
    schema = load_json(schema_path)
    if sha256_file(schema_path) != contract.get("factor_schema_sha256"):
        raise ValueError("Stage3 factor schema differs from the schema used to train this model")
    factor_cols = list(schema.get("factor_cols") or [])
    if factor_cols != list(contract.get("factor_cols") or []):
        raise ValueError("factor order differs from the training contract")
    window_size = int(contract["window_W"])
    num_factors = int(contract["num_factors"])
    if len(factor_cols) != num_factors:
        raise ValueError("factor count differs from the training contract")
    if int(cfg["features"]["window_W"]) != window_size:
        raise ValueError("configured window_W differs from the training contract")

    raw_set, volume_set, mcap_set, dropped_set = parse_norm_groups(schema)
    del raw_set, dropped_set
    norm_meta: Dict[str, object] = {}
    norm_map = None
    norm_map_path = packs.get("norm_map_path") or schema.get("norm_map_path")
    if volume_set or mcap_set:
        if not norm_map_path:
            raise ValueError("normalized factors require a normalization-map path")
        norm_map_abs = abspath(project_root, norm_map_path)
        if not os.path.exists(norm_map_abs):
            raise FileNotFoundError(f"normalization map not found: {norm_map_abs}")
        if sha256_file(norm_map_abs) != contract.get("norm_map_sha256"):
            raise ValueError("normalization map differs from the map used to train this model")
        norm_map = load_json(norm_map_abs)
        norm_meta = norm_map.get("meta", {})
    require_preprocessing_spec(cfg, factor_cols, norm_map, contract.get("preprocessing"))

    model_path = abspath(
        project_root,
        fmt(
            pred_cfg.get("model_path", "results/{run_name}/models/final.keras"),
            run_name=run_name,
            horizon_id=horizon_id,
        ),
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    require_fingerprint(model_path, contract.get("model_sha256"))
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "FixedStandardize": FixedStandardize,
            "TimeAwarePositionalEncoding": TimeAwarePositionalEncoding,
        },
        compile=False,
    )
    if len(model.inputs) != 2:
        raise ValueError(f"expected two model inputs, found {len(model.inputs)}")
    if tuple(model.inputs[1].shape[1:]) != (window_size,) or tuple(model.output_shape[1:]) != (1,):
        raise ValueError("model time input or scalar output differs from contract")
    model_window = int(model.inputs[0].shape[-2])
    model_factors = int(model.inputs[0].shape[-1])
    if (model_window, model_factors) != (window_size, num_factors):
        raise ValueError(
            f"model input {(model_window, model_factors)} differs from contract "
            f"{(window_size, num_factors)}"
        )

    if bool(pred_cfg.get("use_best_weights", True)):
        weights_path = abspath(
            project_root,
            fmt(
                pred_cfg.get("best_weights_path", "results/{run_name}/models/best.weights.h5"),
                run_name=run_name,
                horizon_id=horizon_id,
            ),
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"best weights requested but not found: {weights_path}")
        require_fingerprint(weights_path, contract.get("best_weights_sha256"))
        model.load_weights(weights_path)

    timestamp_col = cfg["data"]["required_columns"]["timestamp"]
    read_columns = required_feature_columns(factor_cols, cfg, volume_set, mcap_set)
    if timestamp_col not in read_columns:
        read_columns.append(timestamp_col)

    session_rules = cfg["data"].get("session_rules", {}) or {}
    max_history_span = float(
        session_rules.get(
            "max_history_span_seconds",
            session_rules.get("max_window_span_seconds", 0.0),
        )
    )
    max_inter_event_gap = float(session_rules.get("max_inter_event_gap_seconds", 0.0))
    contract_constraints = contract.get("window_constraints") or {}
    expected_constraints = {
        "max_history_span_seconds": max_history_span,
        "max_inter_event_gap_seconds": max_inter_event_gap,
    }
    for key, value in expected_constraints.items():
        if not np.isclose(float(contract_constraints.get(key, 0.0)), value):
            raise ValueError(f"config {key} differs from the training contract")

    manifest_path = abspath(project_root, cfg["stage1"]["manifest_ok_path"])
    target_sessions = []
    for session in load_jsonl(manifest_path):
        if not session.get("ok"):
            continue
        date = str(session["date"])
        split = split_for_date(date, split_names, cfg["data"]["splits"])
        if split:
            copied = dict(session)
            copied["prediction_split"] = split
            target_sessions.append(copied)
    target_sessions.sort(
        key=lambda item: (str(item.get("stock_code", "")), str(item["date"]), int(item["session"]))
    )
    if not target_sessions:
        raise RuntimeError(f"no Stage1 sessions found for requested splits: {split_names}")
    found_splits = {session["prediction_split"] for session in target_sessions}
    if set(split_names) - found_splits:
        raise ValueError(f"requested prediction splits have no sessions: {sorted(set(split_names) - found_splits)}")
    session_keys = [(str(s.get("stock_code", "")), str(s["date"]), int(s["session"])) for s in target_sessions]
    if len(set(session_keys)) != len(session_keys):
        raise ValueError("duplicate sessions in Stage1 manifest")

    logging.info(
        "[Predict] run=%s sessions=%d splits=%s W=%d F=%d",
        run_name,
        len(target_sessions),
        split_names,
        window_size,
        num_factors,
    )
    successes = []
    failures = []
    skipped_existing = 0

    for session in target_sessions:
        stock = str(session.get("stock_code", ""))
        date = str(session["date"])
        session_number = int(session["session"])
        split = str(session["prediction_split"])
        csv_path = abspath(project_root, session["path"])
        out_file = os.path.join(output_root, stock, prediction_filename(date, session_number))

        try:
            source_sha256 = sha256_file(csv_path)
            if os.path.exists(out_file) and not bool(pred_cfg.get("overwrite", False)):
                existing = load_prediction_bundle(out_file)["metadata"]
                require_target_binding(existing, binding)
                expected = {
                    "run_name": run_name, "horizon_id": horizon_id, "split": split,
                    "stock_code": stock, "date": date, "session": session_number,
                    "model_contract_sha256": contract_sha256, "source_sha256": source_sha256,
                    "use_best_weights": bool(pred_cfg.get("use_best_weights", True)),
                }
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise ValueError("existing predictions have stale provenance; enable overwrite to regenerate")
                skipped_existing += 1
                continue

            # Never sort here: Stage 1 has already validated session order, and
            # reordering raw rows would break row identity with Stage 2 labels.
            frame = pd.read_csv(csv_path, usecols=read_columns, engine="c")
            X = preprocess_session_features(
                frame,
                factor_cols,
                cfg,
                volume_norm_factors=volume_set,
                mcap_norm_factors=mcap_set,
                norm_meta=norm_meta,
                stock_code=stock,
                strict=True,
            )
            t_sec = hhmmssmmm_to_seconds_vec(
                frame[timestamp_col].to_numpy(dtype=np.float64, copy=False)
            )
            if not np.isfinite(t_sec).all():
                raise ValueError("timestamp contains invalid HHMMSSmmm values")
            if np.any(np.diff(t_sec) < 0.0):
                raise ValueError("session timestamps are not nondecreasing")

            history_ok = valid_window_end_mask(
                weights=np.ones(len(frame), dtype=np.float32),
                t_sec=t_sec,
                window_size=window_size,
                max_history_span_seconds=max_history_span,
                max_inter_event_gap_seconds=max_inter_event_gap,
            )
            end_rows = np.flatnonzero(history_ok).astype(np.int64, copy=False)
            if end_rows.size == 0:
                raise ValueError("session has no history-valid prediction windows")

            prediction_parts = []
            emitted_rows = []
            for X_batch, t_batch, batch_end_rows in iter_window_batches(
                X,
                t_sec,
                window_size,
                batch_size,
                end_rows=end_rows,
            ):
                batch_prediction = np.asarray(
                    model.predict_on_batch([X_batch, t_batch]), dtype=np.float32
                ).reshape(-1)
                if batch_prediction.size != batch_end_rows.size:
                    raise ValueError("model returned a different number of predictions than windows")
                prediction_parts.append(batch_prediction)
                emitted_rows.append(batch_end_rows)

            predictions = np.concatenate(prediction_parts)
            emitted_end_rows = np.concatenate(emitted_rows)
            save_prediction_bundle(
                out_file,
                predictions,
                emitted_end_rows,
                t_sec[emitted_end_rows],
                metadata={
                    **binding,
                    "run_name": run_name,
                    "horizon_id": horizon_id,
                    "split": split,
                    "stock_code": stock,
                    "date": date,
                    "session": session_number,
                    "source_row_count": int(len(frame)),
                    "window_W": window_size,
                    "num_factors": num_factors,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "model_contract_sha256": contract_sha256,
                    "source_sha256": source_sha256,
                    "use_best_weights": bool(pred_cfg.get("use_best_weights", True)),
                    "model_contract_path": contract_path,
                    "factor_schema_sha256": contract["factor_schema_sha256"],
                },
            )
            successes.append(
                {
                    "stock_code": stock,
                    "date": date,
                    "session": session_number,
                    "split": split,
                    "prediction_count": int(predictions.size),
                    "output_path": out_file,
                }
            )
            if len(successes) % 10 == 0:
                logging.info("Predicted %d sessions", len(successes))
        except Exception as exc:
            failures.append(
                {
                    "stock_code": stock,
                    "date": date,
                    "session": session_number,
                    "split": split,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logging.error("Prediction failed for %s %s/%d: %s", stock, date, session_number, exc)
            logging.debug(traceback.format_exc())

    summary_path = os.path.join(output_root, "prediction_summary.json")
    save_json(
        summary_path,
        {
            "run_name": run_name,
            "horizon_id": horizon_id,
            "requested_splits": split_names,
            "target_sessions": len(target_sessions),
            "successful_sessions": len(successes),
            "skipped_existing_sessions": skipped_existing,
            "failed_sessions": len(failures),
            "successes": successes,
            "failures": failures,
        },
    )
    logging.info(
        "[Predict] success=%d skipped=%d failed=%d summary=%s",
        len(successes),
        skipped_existing,
        len(failures),
        summary_path,
    )
    if failures and strict_complete:
        sys.exit(2)


if __name__ == "__main__":
    main()
