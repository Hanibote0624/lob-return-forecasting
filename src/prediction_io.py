"""Prediction window streaming and strict row-aligned bundle I/O."""

import json
import os
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

try:
    from .data_contract import relative_time_windows
except ImportError:
    from data_contract import relative_time_windows


PREDICTION_FORMAT_VERSION = 2


def validate_bundle_arrays(pred, end_row, t_sec):
    prediction = np.asarray(pred)
    rows = np.asarray(end_row)
    times = np.asarray(t_sec)
    if prediction.ndim != 1 or rows.ndim != 1 or times.ndim != 1:
        raise ValueError("prediction bundle arrays must be one-dimensional")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("end_row must use an integer dtype")
    if not (prediction.size == rows.size == times.size) or rows.size == 0:
        raise ValueError("prediction bundle arrays must be nonempty and have equal lengths")
    if rows[0] < 0 or np.any(rows[1:] <= rows[:-1]):
        raise ValueError("end_row must be nonnegative, strictly increasing and unique")
    if not np.isfinite(prediction).all() or not np.isfinite(times).all():
        raise ValueError("prediction bundles cannot contain NaN/Inf")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("prediction endpoint times must be nondecreasing")


def validate_bundle_identity(bundle: dict, *, stock: str, date: str, session: int,
                             split: str, run_name: str, horizon_id: str) -> str:
    metadata = bundle["metadata"]
    expected = {
        "stock_code": stock, "date": date, "session": int(session),
        "split": split, "run_name": run_name, "horizon_id": horizon_id,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"prediction metadata {key} mismatch: expected {value!r}")
    fingerprint = metadata.get("model_contract_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("prediction metadata is missing model_contract_sha256")
    source_hash = metadata.get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("prediction metadata is missing source_sha256")
    source_rows = metadata.get("source_row_count")
    if not isinstance(source_rows, int) or source_rows <= int(bundle["end_row"][-1]):
        raise ValueError("prediction endpoint exceeds source_row_count")
    return fingerprint


def iter_window_batches(
    X: np.ndarray,
    t_sec: np.ndarray,
    window_size: int,
    batch_size: int,
    end_rows: Optional[np.ndarray] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield bounded-memory window batches and their original endpoint rows."""

    features = np.asarray(X)
    times = np.asarray(t_sec, dtype=np.float64)
    if features.ndim != 2 or times.ndim != 1 or features.shape[0] != times.shape[0]:
        raise ValueError("X must be [N,F] and t_sec must be [N] with matching rows")
    if not np.isfinite(features).all() or not np.isfinite(times).all():
        raise ValueError("window inputs must be finite")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("t_sec must be nondecreasing within a session")

    window = int(window_size)
    batch = int(batch_size)
    if window < 1 or batch < 1:
        raise ValueError("window_size and batch_size must be positive")
    if end_rows is None:
        endpoints = np.arange(window - 1, features.shape[0], dtype=np.int64)
    else:
        endpoints = np.asarray(end_rows, dtype=np.int64).reshape(-1)
    if endpoints.size:
        if np.any(np.diff(endpoints) <= 0):
            raise ValueError("end_rows must be strictly increasing and unique")
        if endpoints[0] < window - 1 or endpoints[-1] >= features.shape[0]:
            raise ValueError("end_rows are outside valid window bounds")

    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    for start in range(0, endpoints.size, batch):
        batch_end = endpoints[start : start + batch]
        row_index = batch_end[:, None] - offsets[None, :]
        X_batch = np.asarray(features[row_index], dtype=np.float32)
        t_absolute = times[row_index]
        yield X_batch, relative_time_windows(t_absolute), batch_end.copy()


def prediction_filename(date: str, session: int) -> str:
    return f"{date}_{int(session)}_pred.npz"


def save_prediction_bundle(
    path: str,
    pred: np.ndarray,
    end_row: np.ndarray,
    t_sec: np.ndarray,
    metadata: Optional[dict] = None,
) -> None:
    validate_bundle_arrays(pred, end_row, t_sec)
    prediction = np.asarray(pred, dtype=np.float32)
    endpoint = np.asarray(end_row, dtype=np.int64)
    endpoint_time = np.asarray(t_sec, dtype=np.float64)
    validate_bundle_arrays(prediction, endpoint, endpoint_time)

    payload = dict(metadata or {})
    payload["prediction_format_version"] = PREDICTION_FORMAT_VERSION
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = path + ".tmp.npz"
    np.savez_compressed(
        temp_path,
        pred=prediction,
        end_row=endpoint,
        t_sec=endpoint_time,
        metadata_json=np.asarray(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    )
    os.replace(temp_path, path)


def load_prediction_bundle(path: str) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"pred", "end_row", "t_sec", "metadata_json"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"prediction bundle missing arrays: {sorted(missing)}")
        validate_bundle_arrays(bundle["pred"], bundle["end_row"], bundle["t_sec"])
        pred = np.asarray(bundle["pred"], dtype=np.float32)
        end_row = np.asarray(bundle["end_row"], dtype=np.int64)
        t_sec = np.asarray(bundle["t_sec"], dtype=np.float64)
        raw_metadata = str(np.asarray(bundle["metadata_json"]).item())
    validate_bundle_arrays(pred, end_row, t_sec)
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid prediction metadata JSON: {path}") from exc
    if not isinstance(metadata, dict) or metadata.get("prediction_format_version") != PREDICTION_FORMAT_VERSION:
        raise ValueError(f"unsupported prediction bundle version: {path}")
    return {"pred": pred, "end_row": end_row, "t_sec": t_sec, "metadata": metadata}
