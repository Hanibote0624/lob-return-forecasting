"""Signal state and trade-segment primitives for the optional flip backtest."""

from typing import Iterable, List, Tuple

import numpy as np


def require_prior_calibration(split_cfg: dict, calibration_split: str, trade_splits):
    """Calibration must finish before every trading period, not merely be disjoint."""

    if calibration_split not in split_cfg or not trade_splits:
        raise ValueError("missing calibration or trading split")
    calibration = split_cfg[calibration_split]
    if calibration["start"] > calibration["end"]:
        raise ValueError("reversed calibration date range")
    for name in trade_splits:
        bounds = split_cfg[name]
        if bounds["start"] > bounds["end"] or calibration["end"] >= bounds["start"]:
            raise ValueError("calibration must end before every trading split starts")


def calibrate_thresholds(
    prediction_arrays: Iterable[np.ndarray],
    topk_long: float,
    topk_short: float,
) -> Tuple[float, float]:
    if not 0.0 < float(topk_long) < 0.5 or not 0.0 < float(topk_short) < 0.5:
        raise ValueError("top-k fractions must be between 0 and 0.5")
    finite_parts = []
    for values in prediction_arrays:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        raise ValueError("calibration predictions contain no finite values")
    pooled = np.concatenate(finite_parts)
    return (
        float(np.quantile(pooled, 1.0 - float(topk_long))),
        float(np.quantile(pooled, float(topk_short))),
    )


def generate_positions(
    predictions: np.ndarray,
    valid_execution: np.ndarray,
    threshold_long: float,
    threshold_short: float,
) -> np.ndarray:
    """Generate persistent positions; invalid quotes can never open or flip."""

    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_execution, dtype=bool).reshape(-1)
    if pred.size != valid.size:
        raise ValueError("predictions/valid_execution length mismatch")
    if not np.isfinite([threshold_long, threshold_short]).all() or threshold_long < threshold_short:
        raise ValueError("invalid signal thresholds")
    positions = np.zeros(pred.size, dtype=np.int8)
    state = 0
    for index in range(pred.size):
        if valid[index] and np.isfinite(pred[index]):
            if pred[index] > threshold_long:
                state = 1
            elif pred[index] < threshold_short:
                state = -1
        positions[index] = state
    return positions


def build_position_segments(
    positions: np.ndarray,
    valid_execution: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Return ``(direction, entry_index, exit_index)`` executable segments.

    The supplied timeline must contain every raw session row, including rows
    without predictions. Terminal closure uses only the actual final row.
    An invalid terminal quote raises: we never backdate liquidation to an
    earlier quote selected using future knowledge. Terminal signals do not
    open a new round trip. Same-row fills remain an idealized research assumption.
    """

    position = np.asarray(positions, dtype=np.int8).reshape(-1)
    valid = np.asarray(valid_execution, dtype=bool).reshape(-1)
    if position.size != valid.size:
        raise ValueError("positions/valid_execution length mismatch")
    segments: List[Tuple[int, int, int]] = []
    current = 0
    start = -1

    for index in range(position.size):
        if index == position.size - 1:
            if current != 0:
                if not valid[index]:
                    raise ValueError("cannot liquidate open position at invalid session-terminal quote")
                segments.append((current, start, index))
            break
        new_position = int(position[index])
        if new_position == current:
            continue
        if not valid[index]:
            raise ValueError("position changed on a non-executable row")
        if current != 0:
            segments.append((current, start, index))
        current = new_position
        start = index if current != 0 else -1
    return segments
