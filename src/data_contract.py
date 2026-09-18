"""Shared, NumPy-only data-contract helpers for the LOB pipeline."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


INVALID_CURRENT_MID = np.uint8(1)
INCOMPLETE_HORIZON = np.uint8(2)
INSUFFICIENT_FUTURE = np.uint8(4)
NONFINITE_RETURN = np.uint8(8)


@dataclass(frozen=True)
class FutureReturnLabels:
    r_raw: np.ndarray
    is_valid: np.ndarray
    invalid_reason: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    future_count: np.ndarray


def hhmmssmmm_to_seconds_vec(ts: np.ndarray) -> np.ndarray:
    """Convert numeric ``HHMMSSmmm`` timestamps to seconds as float64.

    Invalid clock values become NaN. Keeping float64 until a window has been
    made relative preserves millisecond differences at intraday clock values.
    """

    values = np.asarray(ts)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out

    x = values[finite].astype(np.int64, copy=False)
    hh = x // 10_000_000
    mm = (x // 100_000) % 100
    ss = (x // 1_000) % 100
    ms = x % 1_000
    valid_clock = (x >= 0) & (hh < 24) & (mm < 60) & (ss < 60) & (ms < 1_000)

    converted = hh * 3600.0 + mm * 60.0 + ss + ms / 1000.0
    target = np.flatnonzero(finite)
    out.flat[target[valid_clock]] = converted[valid_clock]
    return out


def build_event_mean_return_labels(
    t_sec: np.ndarray,
    mid: np.ndarray,
    min_seconds: float,
    max_seconds: float,
    *,
    require_full_horizon: bool = True,
    min_future_observations: int = 1,
) -> FutureReturnLabels:
    """Build event-weighted future-window returns.

    The future interval is inclusive: ``[t + min_seconds, t + max_seconds]``.
    Valid future prices are finite and strictly positive. This is an
    observation-weighted arithmetic mean, not a time-weighted mean.
    """

    t = np.asarray(t_sec, dtype=np.float64)
    price = np.asarray(mid, dtype=np.float64)
    if t.ndim != 1 or price.ndim != 1 or t.shape != price.shape:
        raise ValueError("t_sec and mid must be one-dimensional arrays of equal length")
    if not (float(max_seconds) > float(min_seconds) > 0.0):
        raise ValueError("expected max_seconds > min_seconds > 0")
    min_obs = int(min_future_observations)
    if min_obs < 1:
        raise ValueError("min_future_observations must be at least 1")
    if t.size == 0:
        empty_i64 = np.empty((0,), dtype=np.int64)
        return FutureReturnLabels(
            r_raw=np.empty((0,), dtype=np.float32),
            is_valid=np.empty((0,), dtype=bool),
            invalid_reason=np.empty((0,), dtype=np.uint8),
            lo=empty_i64,
            hi=empty_i64.copy(),
            future_count=empty_i64.copy(),
        )
    if not np.all(np.isfinite(t)):
        raise ValueError("t_sec contains non-finite values")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("t_sec must be nondecreasing")

    t_min = t + float(min_seconds)
    t_max = t + float(max_seconds)
    lo = np.searchsorted(t, t_min, side="left").astype(np.int64)
    hi_exclusive = np.searchsorted(t, t_max, side="right").astype(np.int64)
    hi = hi_exclusive - 1

    future_price_valid = np.isfinite(price) & (price > 0.0)
    price_for_sum = np.where(future_price_valid, price, 0.0)
    prefix_sum = np.concatenate(([0.0], np.cumsum(price_for_sum, dtype=np.float64)))
    prefix_count = np.concatenate(([0], np.cumsum(future_price_valid, dtype=np.int64)))
    future_sum = prefix_sum[hi_exclusive] - prefix_sum[lo]
    future_count = prefix_count[hi_exclusive] - prefix_count[lo]

    current_valid = np.isfinite(price) & (price > 0.0)
    if require_full_horizon:
        horizon_covered = t_max <= (t[-1] + 1e-12)
    else:
        horizon_covered = t_min <= (t[-1] + 1e-12)
    enough_future = future_count >= min_obs

    invalid_reason = np.zeros(t.shape, dtype=np.uint8)
    invalid_reason[~current_valid] |= INVALID_CURRENT_MID
    invalid_reason[~horizon_covered] |= INCOMPLETE_HORIZON
    invalid_reason[~enough_future] |= INSUFFICIENT_FUTURE

    candidate = current_valid & horizon_covered & enough_future
    r64 = np.full(t.shape, np.nan, dtype=np.float64)
    if np.any(candidate):
        avg_future = future_sum[candidate] / future_count[candidate]
        r64[candidate] = avg_future / price[candidate] - 1.0

    finite_return = np.isfinite(r64)
    invalid_reason[candidate & ~finite_return] |= NONFINITE_RETURN
    is_valid = candidate & finite_return
    r64[~is_valid] = np.nan

    return FutureReturnLabels(
        r_raw=r64.astype(np.float32),
        is_valid=is_valid,
        invalid_reason=invalid_reason,
        lo=lo,
        hi=hi,
        future_count=future_count.astype(np.int64, copy=False),
    )


def relative_time_windows(t_windows: np.ndarray) -> np.ndarray:
    """Subtract each window's first time in float64, then return float32."""

    t = np.asarray(t_windows, dtype=np.float64)
    if t.ndim < 1 or t.shape[-1] == 0:
        raise ValueError("t_windows must have a non-empty final dimension")
    return (t - t[..., :1]).astype(np.float32)


def valid_window_end_mask(
    weights: np.ndarray,
    t_sec: Optional[np.ndarray],
    window_size: int,
    *,
    max_history_span_seconds: float = 0.0,
    max_inter_event_gap_seconds: float = 0.0,
) -> np.ndarray:
    """Return valid rolling-window end positions for one session.

    A valid end has a positive finite sample weight, at least ``window_size``
    rows of same-session history, finite nondecreasing timestamps, and any
    configured history-span/gap constraints.
    """

    w = np.asarray(weights)
    if w.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    size = int(window_size)
    if size < 1:
        raise ValueError("window_size must be at least 1")

    end_ok = np.isfinite(w) & (w > 0.0)
    if w.size == 0:
        return end_ok
    end_ok[: min(size - 1, w.size)] = False
    if w.size < size:
        return end_ok

    max_span = float(max_history_span_seconds)
    max_gap = float(max_inter_event_gap_seconds)
    if t_sec is None:
        if max_span > 0.0 or max_gap > 0.0:
            raise ValueError("timestamp data is required for active window-time constraints")
        return end_ok

    t = np.asarray(t_sec, dtype=np.float64)
    if t.ndim != 1 or t.shape != w.shape:
        raise ValueError("t_sec and weights must be one-dimensional arrays of equal length")

    ends = np.arange(size - 1, w.size, dtype=np.int64)
    starts = ends - (size - 1)

    bad_point = ~np.isfinite(t)
    point_prefix = np.concatenate(([0], np.cumsum(bad_point, dtype=np.int64)))
    point_bad_count = point_prefix[ends + 1] - point_prefix[starts]
    time_ok = point_bad_count == 0

    if size > 1:
        gaps = np.diff(t)
        bad_gap = ~np.isfinite(gaps) | (gaps < 0.0)
        if max_gap > 0.0:
            bad_gap |= gaps > (max_gap + 1e-12)
        gap_prefix = np.concatenate(([0], np.cumsum(bad_gap, dtype=np.int64)))
        gap_bad_count = gap_prefix[ends] - gap_prefix[starts]
        time_ok &= gap_bad_count == 0

    if max_span > 0.0:
        spans = t[ends] - t[starts]
        time_ok &= np.isfinite(spans) & (spans <= (max_span + 1e-12))

    end_ok[ends] &= time_ok
    return end_ok
