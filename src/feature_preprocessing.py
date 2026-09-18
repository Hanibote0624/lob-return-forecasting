"""Shared session-level feature preprocessing for packing and inference."""

from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PREPROCESSING_VERSION = 1


def preprocessing_spec(cfg: dict, factor_cols: Sequence[str], norm_map: Optional[dict]) -> dict:
    """Freeze all configuration values that can change packed feature values."""

    _, volume_set, mcap_set, _ = parse_norm_groups(norm_map)
    factors = set(factor_cols)
    volume_set &= factors
    mcap_set &= factors
    if volume_set & mcap_set:
        raise ValueError("a factor cannot belong to both normalization groups")
    rules = cfg.get("data", {}).get("session_rules", {}) or {}
    market = cfg.get("data", {}).get("market", {}).get("columns", {}) or {}
    scale = cfg.get("audit", {}).get("scale_audit", {}) or {}
    meta = (norm_map or {}).get("meta", {}) or {}
    fill = rules.get("fill_remaining_nan", 0.0)
    spec = {
        "version": PREPROCESSING_VERSION,
        "factor_cols": list(factor_cols),
        "ffill_within_session": bool(rules.get("ffill_within_session", True)),
        "fill_remaining_nan": None if fill is None else float(fill),
        "timestamp_column": cfg.get("data", {}).get("required_columns", {}).get("timestamp", "timestamp"),
        "volume_norm": sorted(volume_set),
        "mcap_norm": sorted(mcap_set),
    }
    if volume_set:
        spec["volume_parameters"] = {
            "column": str(market.get("acc_volume") or "acc_volume"),
            "ema_span": int(meta.get("ema_span", scale.get("ema_span", 200))),
            "dv_floor": float(meta.get("dv_floor", scale.get("dv_floor", 1.0))),
        }
    if mcap_set:
        spec["price_columns"] = list(_price_columns(cfg))
        spec["mcap_floor"] = float(meta.get("mcap_floor", scale.get("mcap_floor", 1.0)))
        spec["float_shares"] = {
            str(stock["stock_code"]): float(stock.get("float_shares", 0.0) or 0.0)
            for stock in cfg.get("data", {}).get("stocks", [])
        }
    return spec


def require_preprocessing_spec(cfg: dict, factor_cols: Sequence[str], norm_map: Optional[dict], expected: dict):
    if not expected or preprocessing_spec(cfg, factor_cols, norm_map) != expected:
        raise ValueError("preprocessing parameters differ from Stage3; rebuild packs or restore the configuration")


def parse_norm_groups(norm_map: Optional[dict]) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Return ``(raw, volume_norm, mcap_norm, dropped)`` factor sets."""

    if not norm_map:
        return set(), set(), set(), set()
    groups = norm_map.get("groups") or {}
    return (
        set(groups.get("raw", [])),
        set(groups.get("volume_norm", [])),
        set(groups.get("mcap_norm", [])),
        set(groups.get("none", [])),
    )


def _price_columns(cfg: dict) -> Tuple[Optional[str], str, str, str, bool]:
    market_cols = ((cfg.get("data", {}).get("market", {}) or {}).get("columns", {}) or {})
    price_cfg = ((cfg.get("data", {}).get("required_columns", {}) or {}).get("mid_price", {}) or {})
    mid_col = market_cols.get("mid")
    bid_col = str(price_cfg.get("bid1", market_cols.get("bid", "bid")))
    ask_col = str(price_cfg.get("ask1", market_cols.get("ask", "ask")))
    last_col = str(price_cfg.get("fallback_last", market_cols.get("last", "last")))
    prefer_bidask = bool(price_cfg.get("prefer_bidask", True))
    return mid_col, bid_col, ask_col, last_col, prefer_bidask


def required_feature_columns(
    factor_cols: Sequence[str],
    cfg: dict,
    volume_norm_factors: Iterable[str] = (),
    mcap_norm_factors: Iterable[str] = (),
) -> List[str]:
    """Columns required to reproduce Stage 3 preprocessing for one session."""

    columns = list(factor_cols)
    if volume_norm_factors:
        market_cols = ((cfg.get("data", {}).get("market", {}) or {}).get("columns", {}) or {})
        acc_col = str(market_cols.get("acc_volume") or "acc_volume")
        if acc_col not in columns:
            columns.append(acc_col)

    if mcap_norm_factors:
        mid_col, bid_col, ask_col, last_col, prefer_bidask = _price_columns(cfg)
        candidates = []
        if mid_col:
            candidates.append(str(mid_col))
        if prefer_bidask:
            candidates.extend([bid_col, ask_col])
        if last_col:
            candidates.append(last_col)
        for column in candidates:
            if column not in columns:
                columns.append(column)
    return columns


def compute_reference_price(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Build a finite positive price when possible without accepting crossed quotes.

    A configured explicit mid has first priority. Remaining rows use a valid
    bid/ask midpoint when requested, then a positive last-trade price.
    Unrecoverable rows remain NaN so strict callers fail loudly.
    """

    mid_col, bid_col, ask_col, last_col, prefer_bidask = _price_columns(cfg)
    price = np.full(len(df), np.nan, dtype=np.float64)

    if mid_col and str(mid_col) in df.columns:
        explicit = df[str(mid_col)].to_numpy(dtype=np.float64, copy=False)
        valid = np.isfinite(explicit) & (explicit > 0.0)
        price[valid] = explicit[valid]

    missing = ~np.isfinite(price) | (price <= 0.0)
    if prefer_bidask and bid_col in df.columns and ask_col in df.columns and np.any(missing):
        bid = df[bid_col].to_numpy(dtype=np.float64, copy=False)
        ask = df[ask_col].to_numpy(dtype=np.float64, copy=False)
        valid_quote = (
            missing
            & np.isfinite(bid)
            & np.isfinite(ask)
            & (bid > 0.0)
            & (ask > 0.0)
            & (ask >= bid)
        )
        price[valid_quote] = 0.5 * (bid[valid_quote] + ask[valid_quote])

    missing = ~np.isfinite(price) | (price <= 0.0)
    if last_col in df.columns and np.any(missing):
        last = df[last_col].to_numpy(dtype=np.float64, copy=False)
        valid_last = missing & np.isfinite(last) & (last > 0.0)
        price[valid_last] = last[valid_last]
    return price


def preprocess_session_features(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    cfg: dict,
    volume_norm_factors: Iterable[str] = (),
    mcap_norm_factors: Iterable[str] = (),
    norm_meta: Optional[Mapping[str, object]] = None,
    stock_code: str = "",
    strict: bool = True,
) -> np.ndarray:
    """Apply the exact same fill and optional scale rules in packing/inference."""

    missing = [column for column in factor_cols if column not in df.columns]
    if missing:
        raise ValueError(f"missing factor columns: {missing[:10]}")

    volume_set = set(volume_norm_factors)
    mcap_set = set(mcap_norm_factors)
    norm_meta = dict(norm_meta or {})
    cleaned = df.replace([np.inf, -np.inf], np.nan)

    session_rules = (cfg.get("data", {}).get("session_rules", {}) or {})
    if bool(session_rules.get("ffill_within_session", True)):
        cleaned = cleaned.ffill()
    fill_value = session_rules.get("fill_remaining_nan", 0.0)
    if fill_value is not None:
        cleaned = cleaned.fillna(float(fill_value))

    X = cleaned[list(factor_cols)].to_numpy(dtype=np.float32, copy=True)
    volume_idx = [index for index, name in enumerate(factor_cols) if name in volume_set]
    mcap_idx = [index for index, name in enumerate(factor_cols) if name in mcap_set]
    scale_cfg = (cfg.get("audit", {}).get("scale_audit", {}) or {})

    if volume_idx:
        market_cols = ((cfg.get("data", {}).get("market", {}) or {}).get("columns", {}) or {})
        acc_col = str(market_cols.get("acc_volume") or "acc_volume")
        if acc_col not in cleaned.columns:
            raise ValueError(f"volume normalization requires column: {acc_col}")
        accumulated = cleaned[acc_col].to_numpy(dtype=np.float64, copy=False)
        delta = np.diff(accumulated, prepend=accumulated[:1])
        delta = np.maximum(delta, 0.0)
        if delta.size:
            delta[0] = 0.0
        span = int(norm_meta.get("ema_span", scale_cfg.get("ema_span", 200)))
        floor = float(norm_meta.get("dv_floor", scale_cfg.get("dv_floor", 1.0)))
        if span < 1 or not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("invalid volume-normalization span or floor")
        ema = pd.Series(delta).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64, copy=False)
        denominator = np.maximum(ema, floor).astype(np.float32, copy=False)
        X[:, volume_idx] /= denominator.reshape(-1, 1)

    if mcap_idx:
        shares_by_stock = {
            str(item.get("stock_code")): float(item.get("float_shares", 0.0) or 0.0)
            for item in (cfg.get("data", {}).get("stocks", []) or [])
        }
        float_shares = float(shares_by_stock.get(str(stock_code), 0.0))
        if not np.isfinite(float_shares) or float_shares <= 0.0:
            raise ValueError(
                f"market-cap normalization requires positive float_shares for stock {stock_code!r}"
            )
        price = compute_reference_price(cleaned, cfg)
        if strict and np.any(~np.isfinite(price) | (price <= 0.0)):
            bad_count = int(np.sum(~np.isfinite(price) | (price <= 0.0)))
            raise ValueError(f"cannot build market-cap denominator for {bad_count} rows")
        floor = float(norm_meta.get("mcap_floor", scale_cfg.get("mcap_floor", 1.0)))
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("invalid market-cap normalization floor")
        denominator = np.maximum(price * float_shares, floor).astype(np.float32, copy=False)
        X[:, mcap_idx] /= denominator.reshape(-1, 1)

    if strict and not np.isfinite(X).all():
        raise ValueError("features contain NaN/Inf after shared preprocessing")
    return X
