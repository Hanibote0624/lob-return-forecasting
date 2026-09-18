"""Shared configuration loading and preflight checks; standard library only."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from string import Formatter
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
HORIZON_PATH_FIELDS = {
    "stage2.label_stats_path",
    "stage2.labels_index_path",
    "stage2.stage2_summary_path",
    "stage3.input_stats_path",
    "stage3.packs_manifest_path",
    "stage3.final_schema_path",
    "stage4.stage4_manifest_path",
    "stage5.report_path",
}
PATH_FIELDS = {
    "paths": (
        "manifests_dir",
        "labels_dir",
        "packs_dir",
        "stats_dir",
        "models_dir",
        "logs_dir",
        "eval_dir",
        "results_dir",
    ),
    "stage1": (
        "manifest_all_path",
        "manifest_ok_path",
        "manifest_bad_path",
        "summary_path",
        "schema_path",
    ),
    "stage1_5": ("out_dir", "manifest_ok_path", "factor_norm_map_path"),
    "stage2": (
        "labels_raw_dir",
        "labels_final_dir",
        "label_stats_path",
        "labels_index_path",
        "stage2_summary_path",
    ),
    "stage3": (
        "packed_root",
        "input_stats_path",
        "packs_manifest_path",
        "feature_norm_map_path",
        "final_schema_path",
    ),
    "stage4": ("stage4_manifest_path",),
    "stage5": ("report_path",),
    "predict": ("output_root", "contract_path", "model_path", "best_weights_path"),
    "accuracy_eval": ("pred_root", "report_path", "detail_csv_path"),
    "backtest.output": ("report_path", "trade_detail_csv", "curve_csv"),
    "stage0": ("out_root", "export_factor_map_path"),
    "stage0.baseline": ("baseline_root",),
}
# Only producer paths: prediction model paths and accuracy.pred_root are readers
# and deliberately refer to outputs from other stages.
OUTPUT_FILE_FIELDS = HORIZON_PATH_FIELDS | {
    *(f"stage1.{field}" for field in PATH_FIELDS["stage1"]),
    "accuracy_eval.report_path",
    "accuracy_eval.detail_csv_path",
    *(f"backtest.output.{field}" for field in PATH_FIELDS["backtest.output"]),
}


class ConfigError(ValueError):
    """An invalid or internally inconsistent research configuration."""


def _object_at(cfg, dotted):
    value = cfg
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ConfigError(f"non-finite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _reject_constant(value)
    return number


def resolve_path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("paths must be nonempty strings")
    if os.name != "nt" and PureWindowsPath(value).drive:
        raise ConfigError(
            f"Windows path cannot be used on this platform: {value}; use the Linux/WSL path"
        )
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(root) / path).resolve())


def load_config(path):
    """Resolve project_root relative to this checkout, independent of caller CWD.

    Known artifact paths and stock roots are then relative to project_root.
    The source JSON is never rewritten. Full semantic validation is explicit,
    allowing individual stage scripts to load their smaller test fixtures.
    """

    with open(path, encoding="utf-8-sig") as handle:
        cfg = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    if not isinstance(cfg, dict) or not isinstance(cfg.get("project"), dict):
        raise ConfigError("configuration must contain a project object")
    root = resolve_path(REPO_ROOT, cfg["project"].get("project_root", "."))
    cfg["project"]["project_root"] = root
    for section, fields in PATH_FIELDS.items():
        obj = _object_at(cfg, section)
        if not isinstance(obj, dict):
            continue
        for field in fields:
            if obj.get(field) is not None:
                obj[field] = resolve_path(root, obj[field])
    data = cfg.get("data", {})
    if not isinstance(data, dict):
        raise ConfigError("data must be an object")
    stocks = data.get("stocks", [])
    if not isinstance(stocks, list):
        raise ConfigError("data.stocks must be a list")
    for obj in [data, *stocks]:
        if not isinstance(obj, dict):
            raise ConfigError("data.stocks entries must be objects")
        for field in ("raw_root", "out_root"):
            if obj.get(field) is not None:
                obj[field] = resolve_path(root, obj[field])
    return cfg


def validate_config(cfg):
    """Check the supported single-target main pipeline, without opening data."""

    errors = []
    for key, default, upper_inclusive in (("warmup_ratio", 0.05, False), ("min_lr_ratio", 0.05, True)):
        train_options = cfg.get("train", {})
        value = train_options.get(key, default) if isinstance(train_options, dict) else default
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0 or (value > 1 if upper_inclusive else value >= 1)):
            errors.append(f"train.{key} is outside its supported range")


    def require(path, kind):
        value = _object_at(cfg, path)
        valid = isinstance(value, kind) and not (kind is int and isinstance(value, bool))
        if not valid or (kind is str and not value.strip()):
            errors.append(
                f"{path} must be a nonempty {kind.__name__}"
                if kind is str
                else f"{path} must be {kind.__name__}"
            )
        return value

    def positive(path, *, zero=False, integer=False):
        value = _object_at(cfg, path)
        valid = isinstance(value, int if integer else (int, float)) and not isinstance(value, bool)
        if not valid or not math.isfinite(value) or (value < 0 if zero else value <= 0):
            errors.append(
                f"{path} must be a finite {'nonnegative' if zero else 'positive'} {'integer' if integer else 'number'}"
            )

    for section in (
        "project",
        "data",
        "features",
        "horizons",
        "label",
        "sample_weight",
        "train",
        "model",
        "paths",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
        "stage5",
        "dataloader",
        "predict",
        "accuracy_eval",
        "backtest",
    ):
        require(section, dict)
    for section in (
        "data.session_rules",
        "data.required_columns",
        "data.splits",
        "data.timestamp_parse",
        "data.required_columns.factors",
        "train.optimizer",
        "backtest.thresholds",
        "backtest.output",
    ):
        require(section, dict)
    for section in ("stage0", "stage1_5"):
        if section in cfg:
            require(section, dict)
    if errors:
        raise ConfigError("\n".join(errors))

    for section in (
        "paths",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
        "stage5",
        "predict",
        "accuracy_eval",
        "backtest.output",
    ):
        for field in PATH_FIELDS[section]:
            if field == "feature_norm_map_path":
                continue
            require(f"{section}.{field}", str)
    for field in (
        "features.window_W",
        "features.num_factors",
        "train.global_batch",
        "train.epochs",
        "stage1.num_workers",
        "stage2.num_workers",
        "stage3.row_shard_rows",
        "stage4.block_size_ends",
        "predict.batch_size",
        "model.d_model",
        "model.num_heads",
        "model.num_layers",
        "model.ff_dim",
        "model.lstm_units",
        "model.head_hidden",
        "label.min_future_observations",
    ):
        positive(field, integer=True)
    for field in ("train.steps_per_epoch", "train.val_steps", "train.test_steps"):
        positive(field, integer=True, zero=True)
    for field in (
        "label.fixed_scale",
        "label.eps",
        "train.base_lr",
        "model.time_scale",
        "model.max_timescale",
    ):
        positive(field)
    for field in (
        "data.session_rules.max_history_span_seconds",
        "data.session_rules.max_inter_event_gap_seconds",
    ):
        positive(field, zero=True)
    for field in (
        "data.session_rules.ffill_within_session",
        "data.session_rules.no_cross_session_windows",
        "data.session_rules.no_cross_day_windows",
        "label.require_full_horizon",
        "dataloader.drop_remainder_train",
        "dataloader.drop_remainder_eval",
        "model.use_time_aware_pos",
        "model.use_lstm",
        "train.mixed_precision",
        "predict.strict_complete",
        "accuracy_eval.strict_complete",
        "backtest.strict_complete",
    ):
        require(field, bool)
    # These switches have defaults in their stages, but explicit values must
    # be JSON booleans: bool("false") would otherwise silently enable them.
    for section, fields in {
        "stage0": ("enabled",),
        "stage1_5": ("use_train_split_only",),
        "stage2": ("strict",),
        "stage3": (
            "write_y_raw", "write_t_sec", "write_is_valid", "strict_rowcount_match",
            "strict_label_integrity", "strict_no_nan_X", "compute_input_stats_train",
            "input_stats_only_valid",
        ),
        "stage4": ("strict", "copy_stage3_schema"),
        "stage5": ("strict", "per_stock_breakdown"),
        "train": ("require_input_stats", "jit_compile"),
        "dataloader": ("shuffle_blocks_train",),
        "predict": ("use_best_weights", "overwrite"),
    }.items():
        for field in fields:
            if field in cfg.get(section, {}):
                require(f"{section}.{field}", bool)
    for section, fields, zero in (
        ("sample_weight", ("alpha", "clip_max"), True),
        ("sample_weight", ("min_weight", "max_weight"), False),
        ("train.optimizer", ("weight_decay", "beta1", "beta2"), True),
        ("train.optimizer", ("epsilon", "clipnorm"), False),
    ):
        for field in fields:
            if field in _object_at(cfg, section):
                positive(f"{section}.{field}", zero=zero)
    require("data.required_columns.timestamp", str)
    require("data.required_columns.mid_price", dict)
    require("data.file_pattern", dict)
    require("horizons.active_horizon_id", str)
    if errors:
        raise ConfigError("\n".join(errors))

    def identifier(value, name):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            errors.append(f"{name} must be a path-safe identifier")

    run_name = cfg["train"].get("run_name")
    identifier(run_name, "train.run_name")
    for section in ("predict", "backtest"):
        if cfg[section].get("run_name") != run_name:
            errors.append(f"{section}.run_name must match train.run_name")
    horizon_id = cfg["horizons"]["active_horizon_id"]
    identifier(horizon_id, "horizons.active_horizon_id")
    definitions = cfg["horizons"].get("definitions", {})
    horizon = definitions.get(horizon_id) if isinstance(definitions, dict) else None
    if not isinstance(horizon, dict):
        errors.append("active horizon must exist in horizons.definitions")
    else:
        lo, hi = horizon.get("min_seconds"), horizon.get("max_seconds")
        if (
            any(
                not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                for v in (lo, hi)
            )
            or not 0 < lo < hi
        ):
            errors.append("active horizon must satisfy 0 < min_seconds < max_seconds")

    splits = cfg["data"].get("splits", {})
    dates = {}
    for name in ("train", "val", "test", "date_range"):
        bounds = (
            cfg["data"].get("date_range")
            if name == "date_range"
            else splits.get(name)
            if isinstance(splits, dict)
            else None
        )
        try:
            start, end = bounds["start"], bounds["end"]
            for date in (start, end):
                if not isinstance(date, str) or len(date) != 8:
                    raise ValueError()
                datetime.strptime(date, "%Y%m%d")
            if start > end:
                raise ValueError()
            dates[name] = (start, end)
        except (TypeError, KeyError, ValueError):
            errors.append(f"data.{name} needs ordered YYYYMMDD start/end dates")
    if len(dates) == 4:
        if not dates["train"][1] < dates["val"][0] or not dates["val"][1] < dates["test"][0]:
            errors.append("train, val and test must be chronological and non-overlapping")
        if any(
            not dates["date_range"][0] <= dates[name][0] <= dates[name][1] <= dates["date_range"][1]
            for name in ("train", "val", "test")
        ):
            errors.append("all splits must fit inside data.date_range")

    stocks = cfg["data"].get("stocks")
    if not isinstance(stocks, list) or not stocks:
        errors.append("data.stocks must contain at least one stock")
    else:
        seen = set()
        for stock in stocks:
            code = stock.get("stock_code") if isinstance(stock, dict) else None
            identifier(code, "stock_code")
            if isinstance(code, str):
                if code in seen:
                    errors.append(f"duplicate stock_code: {code}")
                seen.add(code)
            if (
                not isinstance(stock, dict)
                or not isinstance(stock.get("raw_root"), str)
                or not stock["raw_root"]
            ):
                errors.append("each stock requires a raw_root")
    if "enable" in cfg.get("stage0", {}):
        errors.append("stage0.enable is a legacy typo; use stage0.enabled")
    if cfg.get("stage0", {}).get("enabled", False) and isinstance(stocks, list):
        for stock in stocks:
            if not isinstance(stock, dict) or not isinstance(stock.get("out_root"), str) or not stock["out_root"].strip():
                errors.append("stage0.enabled=true requires an out_root for every stock")
    if not cfg["stage3"].get("write_t_sec", True):
        errors.append("stage3.write_t_sec must be true: the training window loader requires timestamps")
    if cfg["label"].get("scale_mode") != "fixed_multiplier":
        errors.append("main Stage2 supports label.scale_mode=fixed_multiplier")
    if cfg["label"].get("future_aggregation") != "event_mean":
        errors.append("main Stage2 supports label.future_aggregation=event_mean")
    clip_mode = cfg["label"].get("clip_mode", "q99_abs_train_only")
    if not isinstance(clip_mode, str) or clip_mode.lower() not in (
        "q99_abs_train_only", "none", "off", "disabled"
    ):
        errors.append("label.clip_mode must be q99_abs_train_only, none, off, or disabled")
    sw = cfg["sample_weight"]
    if sw.get("mode", "abs_r_q90_train_only") != "abs_r_q90_train_only":
        errors.append("Stage3 supports sample_weight.mode=abs_r_q90_train_only")
    if sw.get("min_weight", 1.0) > sw.get("max_weight", 4.0):
        errors.append("sample_weight.min_weight must not exceed max_weight")
    optimizer = cfg["train"]["optimizer"]
    if optimizer.get("name", "adamw") != "adamw":
        errors.append("Stage6 supports train.optimizer.name=adamw")
    if "eps" in optimizer:
        errors.append("train.optimizer.eps is unused; remove it and set train.optimizer.epsilon explicitly")
    for field in ("beta1", "beta2"):
        if optimizer.get(field, 0.0) >= 1.0:
            errors.append(f"train.optimizer.{field} must be less than 1")
    if cfg["data"].get("timestamp_parse", {}).get("mode") != "HHMMSSmmm_to_seconds":
        errors.append("timestamp_parse.mode must be HHMMSSmmm_to_seconds")
    if cfg["train"].get("loss") not in ("ccc", "mse", "huber", "logcosh"):
        errors.append("train.loss must be ccc, mse, huber, or logcosh")
    if cfg["train"].get("strategy") not in ("mirrored", "single"):
        errors.append("train.strategy must be mirrored or single")
    if cfg["dataloader"]["drop_remainder_eval"]:
        errors.append("dataloader.drop_remainder_eval must be false")
    if cfg["train"]["val_steps"] or cfg["train"]["test_steps"]:
        errors.append("set val_steps=test_steps=0 for complete evaluation coverage")
    if (
        not cfg["data"]["session_rules"]["no_cross_session_windows"]
        or not cfg["data"]["session_rules"]["no_cross_day_windows"]
    ):
        errors.append("cross-session/day windows are not supported")
    if cfg["model"]["d_model"] % cfg["model"]["num_heads"]:
        errors.append("model.d_model must be divisible by model.num_heads")
    if cfg["stage3"]["row_shard_rows"] < cfg["features"]["window_W"]:
        errors.append("stage3.row_shard_rows must be at least features.window_W")

    for section in ("predict", "accuracy_eval", "backtest"):
        chosen = cfg[section].get("splits")
        if (
            not isinstance(chosen, list)
            or not chosen
            or any(s not in ("train", "val", "test") for s in chosen)
            or len(set(chosen)) != len(chosen)
        ):
            errors.append(f"{section}.splits must be a nonempty, unique list of train/val/test")
        elif section != "predict" and not set(chosen) <= set(cfg["predict"].get("splits", [])):
            errors.append(f"{section}.splits must be included in predict.splits")
        if errors:
            raise ConfigError("\n".join(errors))
    thresholds = cfg["backtest"].get("thresholds", {})
    if thresholds.get("mode") == "calibration_split":
        source = thresholds.get("source_split")
        if not isinstance(source, str) or source not in ("train", "val", "test"):
            errors.append("backtest.thresholds.source_split must be train, val, or test")
        elif source not in cfg["predict"].get("splits", []):
            errors.append("threshold calibration split must be included in predict.splits")
        elif source in dates:
            for target in cfg["backtest"].get("splits", []):
                if target in dates and dates[source][1] >= dates[target][0]:
                    errors.append("threshold calibration must precede every trading split")
    elif thresholds.get("mode") == "fixed":
        lo, hi = thresholds.get("short"), thresholds.get("long")
        if (
            any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in (lo, hi))
            or hi < lo
        ):
            errors.append("fixed thresholds require finite short <= long")
    else:
        errors.append("backtest.thresholds.mode must be calibration_split or fixed")

    resolved = {}
    for section, fields in PATH_FIELDS.items():
        obj = _object_at(cfg, section)
        if not isinstance(obj, dict):
            continue
        for field in fields:
            value = obj.get(field)
            if value is None:
                continue
            try:
                allowed = (
                    {"horizon_id", "run_name"}
                    if section in ("predict", "accuracy_eval", "backtest.output")
                    else {"horizon_id"}
                    if f"{section}.{field}" in HORIZON_PATH_FIELDS
                    else set()
                )
                for _, name, spec, conv in Formatter().parse(value):
                    if name is not None and (name not in allowed or spec or conv):
                        raise ValueError()
                resolved[f"{section}.{field}"] = value.format(
                    horizon_id=horizon_id, run_name=run_name
                )
            except (ValueError, KeyError, TypeError):
                errors.append(f"unsupported path placeholder in {section}.{field}")
    if not errors:
        producers = {}
        for field in sorted(OUTPUT_FILE_FIELDS):
            path = Path(resolved[field])
            if path in producers:
                errors.append(f"output path collision: {field} and {producers[path]} must be distinct")
            producers[path] = field
        if resolved["stage2.labels_raw_dir"] == resolved["stage2.labels_final_dir"]:
            errors.append("stage2.labels_raw_dir and labels_final_dir must be distinct")
        model_dir = Path(resolved["paths.results_dir"]) / run_name
        for field, tail in (
            ("contract_path", "model_contract.json"),
            ("model_path", "models/final.keras"),
            ("best_weights_path", "models/best.weights.h5"),
        ):
            if Path(resolved[f"predict.{field}"]) != model_dir / tail:
                errors.append(
                    f"predict.{field} must point to the matching Stage6 run under paths.results_dir"
                )
        if resolved["predict.output_root"] != resolved["accuracy_eval.pred_root"]:
            errors.append("accuracy_eval.pred_root must equal predict.output_root")
    if errors:
        raise ConfigError("\n".join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate configuration without opening data or importing TensorFlow."
    )
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config_path)
        validate_config(cfg)
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"Configuration valid; project_root={cfg['project']['project_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
