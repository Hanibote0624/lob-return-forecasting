"""Fail-closed target provenance shared by preparation, training and evaluation.

Hashes detect stale/mixed artifacts, not deliberate tampering. No TensorFlow import.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .artifact_contract import sha256_file
except ImportError:
    from artifact_contract import sha256_file


MODEL_CONTRACT_VERSION = 2
BINDING_KEYS = ("target_contract", "target_contract_sha256", "label_stats_sha256")


def json_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def label_definition(cfg):
    horizon = cfg["horizons"]["active_horizon_id"]
    bounds = cfg["horizons"]["definitions"][horizon]
    label = cfg.get("label", {})
    required = cfg["data"]["required_columns"]
    price = required["mid_price"]
    return {
        "version": 2, "horizon_id": horizon,
        "min_seconds": float(bounds["min_seconds"]),
        "max_seconds": float(bounds["max_seconds"]),
        "formula": "future_event_mean_mid/current_mid-1",
        "future_aggregation": label.get("future_aggregation", "event_mean"),
        "future_interval_inclusive": True,
        "require_full_horizon": bool(label.get("require_full_horizon", True)),
        "min_future_observations": int(label.get("min_future_observations", 1)),
        "eps": float(label.get("eps", 1e-12)),
        "valid_price_rule": "finite_and_strictly_positive",
        "mid_price_rounding": "none", "mid_price_storage": "float64",
        "timestamp_storage": "float64_seconds", "return_storage": "float32",
        "timestamp_column": required["timestamp"],
        "timestamp_parse": "HHMMSSmmm_to_seconds",
        "price_source": {
            "prefer_bidask": bool(price.get("prefer_bidask", True)),
            "bid1": price["bid1"], "ask1": price["ask1"],
            "fallback_last": price["fallback_last"],
            "bidask_rule": "finite_positive_ask_ge_bid_else_last",
        },
    }


def target_spec(cfg, stats):
    """Include actual training-calibrated clipping, weighting statistics and settings."""
    mode = str(cfg.get("label", {}).get("clip_mode", "q99_abs_train_only")).lower()
    mode = "none" if mode in {"none", "off", "disabled"} else mode
    scale = float(cfg.get("label", {}).get("fixed_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0 or stats.get("scale") != scale:
        raise ValueError("label scale differs from configuration; rebuild Stage2 and downstream artifacts")
    if stats.get("label_contract") != label_definition(cfg):
        raise ValueError("label definition differs or lacks V6 provenance; rebuild Stage2 from CSV")
    quantiles = stats["train_only_quantiles"]
    q99 = float(quantiles["q99_abs_r_raw"])
    if mode == "none":
        limit = None
    elif mode == "q99_abs_train_only":
        limit = q99 * scale
        cap = float(cfg["stage2"].get("max_clip_abs_scaled", 0.0))
        if cap > 0:
            limit = min(limit, cap)
        limit = max(float(cfg.get("label", {}).get("eps", 1e-12)) * scale, limit)
    else:
        raise ValueError(f"unsupported clipping rule: {mode}")
    actual_mode = stats.get("clip_mode")
    actual_mode = "none" if actual_mode in {"none", "off", "disabled"} else actual_mode
    if actual_mode != mode or stats.get("clip_abs_scaled") != limit:
        raise ValueError("label clipping differs from configuration/statistics")
    return {
        "version": 1, "label_definition": label_definition(cfg),
        "output_field": "r_scaled", "output_units": "scaled_clipped_return",
        "scale": scale, "clip_mode": mode, "clip_abs_scaled": limit,
        "train_only_quantiles": quantiles,
        "calibration": {
            "train_split": cfg["data"]["splits"].get("train"),
            "stocks": sorted(str(s["stock_code"]) for s in cfg["data"].get("stocks", [])),
            "quantile_sample_per_session": int(cfg["stage2"].get("quantile_sample_per_session", 5000)),
            "max_clip_abs_scaled": float(cfg["stage2"].get("max_clip_abs_scaled", 0.0)),
        },
    }


def binding_from_stats(cfg, stats, stats_path):
    expected = target_spec(cfg, stats)
    if stats.get("target_contract") != expected:
        raise ValueError("target contract differs from configuration; rebuild Stage2 and downstream")
    digest = json_sha256(expected)
    if stats.get("target_contract_sha256") != digest:
        raise ValueError("invalid target contract fingerprint")
    return {"target_contract": expected, "target_contract_sha256": digest,
            "label_stats_sha256": sha256_file(str(stats_path))}


def load_target_binding(cfg):
    horizon = cfg["horizons"]["active_horizon_id"]
    path = Path(cfg["project"]["project_root"]) / cfg["stage2"]["label_stats_path"].format(horizon_id=horizon)
    with path.open(encoding="utf-8") as handle:
        return binding_from_stats(cfg, json.load(handle), path)


def require_target_binding(artifact, expected):
    for key in BINDING_KEYS:
        if key not in artifact or artifact[key] != expected[key]:
            raise ValueError(f"{key} mismatch/missing; rebuild affected artifacts with the current pipeline")


def require_raw_definition(labels, definition, source_sha256=None):
    if ("label_contract_version" not in labels.files or int(labels["label_contract_version"]) != 2
            or labels["mid"].dtype != np.float64 or labels["t_sec"].dtype != np.float64):
        raise ValueError("labels lack the unrounded float64 midpoint/time contract; rebuild Stage2 from CSV")
    if "label_definition_sha256" not in labels.files or str(labels["label_definition_sha256"].item()) != json_sha256(definition):
        raise ValueError("raw label definition mismatch; rebuild Stage2 from CSV without --only-scale")
    if source_sha256 is not None and (
        "source_sha256" not in labels.files or str(labels["source_sha256"].item()) != source_sha256
    ):
        raise ValueError("label source fingerprint differs from raw/predicted CSV")


def require_label_binding(labels, binding, source_sha256=None):
    require_raw_definition(labels, binding["target_contract"]["label_definition"], source_sha256)
    for key in ("target_contract_sha256", "label_stats_sha256"):
        if key not in labels.files or str(labels[key].item()) != binding[key]:
            raise ValueError(f"label {key} mismatch; rebuild Stage2 and downstream artifacts")


def load_completed_model_contract(cfg, binding):
    run = cfg.get("predict", {}).get("run_name") or cfg["train"]["run_name"]
    horizon = cfg["horizons"]["active_horizon_id"]
    template = cfg.get("predict", {}).get("contract_path", "results/{run_name}/model_contract.json")
    path = Path(cfg["project"]["project_root"]) / template.format(run_name=run, horizon_id=horizon)
    with path.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("contract_version") != MODEL_CONTRACT_VERSION:
        raise ValueError("old model contract: retrain after rebuilding Stage2–5")
    if contract.get("status") != "trained" or contract.get("run_name") != run or contract.get("horizon_id") != horizon:
        raise ValueError("model contract must identify the requested completed training run")
    require_target_binding(contract, binding)
    return contract, sha256_file(str(path))
