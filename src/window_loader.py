"""TensorFlow-independent mmap window loader used by Stage 6 and tests."""

import json
import os
import queue
import random
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np

try:
    from .data_contract import relative_time_windows
except ImportError:
    from data_contract import relative_time_windows


@dataclass(frozen=True)
class LoaderConfig:
    project_root: str
    stage4_manifest_path: str
    split: str
    batch_size: int
    window_W: int
    num_factors: int
    shuffle_blocks: bool = False
    seed: int = 42
    drop_remainder: bool = True
    mmap_cache_items: int = 16


def _abspath(root: str, path: str) -> str:
    if not path:
        raise ValueError("required shard path is missing")
    return path if os.path.isabs(path) else os.path.join(root, path)


class LRUMmapCache:
    def __init__(self, capacity: int):
        self.capacity = int(max(1, capacity))
        self._cache: Dict[str, np.ndarray] = {}
        self._lru: List[str] = []

    def get(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._lru.remove(path)
            self._lru.append(path)
            return self._cache[path]
        if not os.path.exists(path):
            raise FileNotFoundError(f"mmap shard not found: {path}")
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as exc:
            raise RuntimeError(f"failed to load mmap shard {path}: {exc}") from exc
        self._cache[path] = arr
        self._lru.append(path)
        while len(self._lru) > self.capacity:
            old = self._lru.pop(0)
            self._cache.pop(old, None)
        return arr


def count_batches_from_blocks(blocks: List[dict], batch_size: int, drop_remainder: bool) -> int:
    batch = int(batch_size)
    if batch < 1:
        raise ValueError("batch_size must be at least 1")
    total = 0
    for block in blocks:
        length = int(block["end_len"])
        if length < 0:
            raise ValueError(f"negative block length: {length}")
        total += length
    return int(total // batch if drop_remainder else (total + batch - 1) // batch)


def count_split_batches(stage4_manifest: dict, split: str, batch_size: int, drop_remainder: bool) -> int:
    if split not in stage4_manifest.get("splits", {}):
        raise KeyError(f"split not found in Stage4 manifest: {split}")
    return count_batches_from_blocks(
        list(stage4_manifest["splits"][split].get("blocks", [])),
        batch_size,
        drop_remainder,
    )


def iter_batches_with_time(
    cfg: LoaderConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield one finite pass over a split.

    Evaluation with ``drop_remainder=False`` emits every valid end position
    exactly once. Training may drop only one final short batch for the split;
    artificial Stage4 block boundaries do not discard additional samples.
    Relative time is computed in float64 before conversion to float32.
    """

    with open(cfg.stage4_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if cfg.split not in manifest.get("splits", {}):
        raise KeyError(f"split not found in Stage4 manifest: {cfg.split}")

    split_obj = manifest["splits"][cfg.split]
    blocks = list(split_obj.get("blocks", []))
    shards = split_obj.get("shards", {})

    rng = random.Random(cfg.seed)
    if cfg.shuffle_blocks:
        rng.shuffle(blocks)

    batch = int(cfg.batch_size)
    window = int(cfg.window_W)
    factors = int(cfg.num_factors)
    if batch < 1 or window < 1 or factors < 1:
        raise ValueError("batch_size, window_W, and num_factors must be positive")

    cache = LRUMmapCache(cfg.mmap_cache_items)

    def chunks():
        for block in blocks:
            shard_id = str(block["shard_id"])
            if shard_id not in shards:
                raise KeyError(f"shard {shard_id} referenced by block is missing")
            shard = shards[shard_id]
            X = cache.get(_abspath(cfg.project_root, shard.get("X")))
            y = cache.get(_abspath(cfg.project_root, shard.get("y")))
            weight = cache.get(_abspath(cfg.project_root, shard.get("w")))
            t_sec = cache.get(_abspath(cfg.project_root, shard.get("t_sec")))

            if X.ndim != 2 or X.shape[1] != factors:
                raise ValueError(
                    f"unexpected X shape for shard {shard_id}: {X.shape}, expected (*, {factors})"
                )
            row_count = X.shape[0]
            if (
                y.shape[0] != row_count
                or weight.shape[0] != row_count
                or t_sec.shape[0] != row_count
            ):
                raise ValueError(
                    f"row-count mismatch in shard {shard_id}: "
                    f"X={row_count}, y={y.shape[0]}, w={weight.shape[0]}, t={t_sec.shape[0]}"
                )

            end_start = int(block["end_start"])
            end_len = int(block["end_len"])
            if end_len < 0:
                raise ValueError(f"negative block length: {end_len}")
            positions = list(range(0, end_len, batch))
            if cfg.shuffle_blocks:
                rng.shuffle(positions)

            for pos in positions:
                take = min(batch, end_len - pos)
                first_end = end_start + pos
                last_end = first_end + take - 1
                base = first_end - (window - 1)
                stop = last_end + 1
                seg_start = int(block.get("seg_start_row", 0))
                seg_stop = seg_start + int(block.get("seg_length", row_count - seg_start))
                if base < seg_start or stop > seg_stop or base < 0 or stop > row_count:
                    raise ValueError(
                        f"window block is out of segment bounds: shard={shard_id}, "
                        f"base={base}, stop={stop}, segment=[{seg_start}, {seg_stop})"
                    )

                big_X = X[base:stop]
                big_t = t_sec[base:stop]
                expected_rows = take + window - 1
                if big_X.shape[0] != expected_rows or big_t.shape[0] != expected_rows:
                    raise ValueError(f"short shard slice for shard={shard_id}, block={block}")

                X_view = np.lib.stride_tricks.as_strided(
                    big_X,
                    shape=(take, window, factors),
                    strides=(big_X.strides[0], big_X.strides[0], big_X.strides[1]),
                    writeable=False,
                )
                t_view = np.lib.stride_tricks.as_strided(
                    big_t,
                    shape=(take, window),
                    strides=(big_t.strides[0], big_t.strides[0]),
                    writeable=False,
                )
                yield (
                    np.asarray(X_view, dtype=np.float32).copy(),
                    relative_time_windows(t_view),
                    np.asarray(y[first_end:stop], dtype=np.float32).copy(),
                    np.asarray(weight[first_end:stop], dtype=np.float32).copy(),
                )

    pending_X: List[np.ndarray] = []
    pending_t: List[np.ndarray] = []
    pending_y: List[np.ndarray] = []
    pending_w: List[np.ndarray] = []
    pending_count = 0

    for X_chunk, t_chunk, y_chunk, w_chunk in chunks():
        offset = 0
        while offset < y_chunk.shape[0]:
            remaining = y_chunk.shape[0] - offset
            if pending_count == 0 and remaining == batch:
                yield X_chunk[offset:], t_chunk[offset:], y_chunk[offset:], w_chunk[offset:]
                offset += batch
                continue

            take = min(batch - pending_count, remaining)
            stop = offset + take
            pending_X.append(X_chunk[offset:stop])
            pending_t.append(t_chunk[offset:stop])
            pending_y.append(y_chunk[offset:stop])
            pending_w.append(w_chunk[offset:stop])
            pending_count += take
            offset = stop
            if pending_count == batch:
                yield (
                    np.concatenate(pending_X, axis=0),
                    np.concatenate(pending_t, axis=0),
                    np.concatenate(pending_y, axis=0),
                    np.concatenate(pending_w, axis=0),
                )
                pending_X.clear()
                pending_t.clear()
                pending_y.clear()
                pending_w.clear()
                pending_count = 0

    if pending_count and not cfg.drop_remainder:
        yield (
            np.concatenate(pending_X, axis=0),
            np.concatenate(pending_t, axis=0),
            np.concatenate(pending_y, axis=0),
            np.concatenate(pending_w, axis=0),
        )


@dataclass(frozen=True)
class _PrefetchFailure:
    error: BaseException


def iter_batches_prefetch(base_iter: Iterator, prefetch: int):
    """Prefetch while propagating producer exceptions to the consumer."""

    if prefetch <= 0:
        yield from base_iter
        return

    output = queue.Queue(maxsize=int(prefetch))
    sentinel = object()
    stop_event = threading.Event()

    def put_unless_stopped(item) -> bool:
        while not stop_event.is_set():
            try:
                output.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def worker():
        try:
            for item in base_iter:
                if not put_unless_stopped(item):
                    break
        except BaseException as exc:  # propagated on the consuming thread
            put_unless_stopped(_PrefetchFailure(exc))
        finally:
            put_unless_stopped(sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            item = output.get()
            if item is sentinel:
                break
            if isinstance(item, _PrefetchFailure):
                raise item.error
            yield item
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
