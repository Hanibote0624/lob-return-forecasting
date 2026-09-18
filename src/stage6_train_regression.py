#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import time
import random
from typing import Optional

import numpy as np
import tensorflow as tf

try:
    from .artifact_contract import require_fingerprint, sha256_file
    from .feature_preprocessing import PREPROCESSING_VERSION, require_preprocessing_spec
    from .window_loader import (
        LoaderConfig,
        count_split_batches,
        iter_batches_prefetch,
        iter_batches_with_time,
    )
except ImportError:
    from artifact_contract import require_fingerprint, sha256_file
    from feature_preprocessing import PREPROCESSING_VERSION, require_preprocessing_spec
    from window_loader import (
        LoaderConfig,
        count_split_batches,
        iter_batches_prefetch,
        iter_batches_with_time,
    )

# ---------------------------
# utils
# ---------------------------

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: str, obj: dict):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def abspath(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, p)


def set_global_seeds(seed: int):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------
# LR schedule & Metrics
# ---------------------------

@tf.keras.utils.register_keras_serializable(package="lit")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        if (not np.isfinite(self.base_lr) or self.base_lr <= 0 or self.total_steps < 1
                or not 0 <= self.warmup_steps < self.total_steps or not 0 <= self.min_lr_ratio <= 1):
            raise ValueError("invalid warmup/cosine schedule bounds")

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)
        lr_warm = self.base_lr * tf.minimum(1.0, step / tf.maximum(1.0, warm))
        progress = (step - warm) / tf.maximum(1.0, total - warm)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * progress))
        min_lr = self.base_lr * self.min_lr_ratio
        lr_cos = min_lr + (self.base_lr - min_lr) * cosine
        return tf.where(step < warm, lr_warm, lr_cos)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }


def _weighted_vectors(y_true, y_pred, sample_weight, dtype=tf.float64):
    true = tf.reshape(tf.cast(y_true, dtype), [-1])
    pred = tf.reshape(tf.cast(y_pred, dtype), [-1])
    weight = tf.ones_like(true) if sample_weight is None else tf.broadcast_to(
        tf.reshape(tf.cast(sample_weight, dtype), [-1]), tf.shape(true)
    )
    tf.debugging.assert_equal(tf.shape(true), tf.shape(pred))
    tf.debugging.assert_all_finite(true, "nonfinite targets")
    tf.debugging.assert_all_finite(pred, "nonfinite predictions")
    tf.debugging.assert_all_finite(weight, "nonfinite sample weights")
    tf.debugging.assert_non_negative(weight, "negative sample weights")
    return true, pred, weight


@tf.keras.utils.register_keras_serializable(package="lit")
class WeightedDirAcc(tf.keras.metrics.Metric):
    def __init__(self, name="dir_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct_sum = self.add_weight(name="correct_sum", initializer="zeros", dtype=tf.float64)
        self.weight_sum = self.add_weight(name="weight_sum", initializer="zeros", dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred, weight = _weighted_vectors(y_true, y_pred, sample_weight)
        self.correct_sum.assign_add(tf.reduce_sum(weight * tf.cast(tf.sign(true) == tf.sign(pred), tf.float64)))
        self.weight_sum.assign_add(tf.reduce_sum(weight))

    def result(self):
        return tf.math.divide_no_nan(self.correct_sum, self.weight_sum)

    def reset_state(self):
        for variable in self.variables:
            variable.assign(tf.zeros_like(variable))


class _MomentMetric(tf.keras.metrics.Metric):
    """Float64 pooled raw moments; additive states also aggregate across replicas.

    Avoids the previous float32 cancellation for normal return-scale inputs.
    Arbitrarily large offsets are not supported: normalize regression targets.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        for field in ("mass", "sum_true", "sum_pred", "sum_tt", "sum_pp", "sum_tp", "sse"):
            setattr(self, field, self.add_weight(name=field, initializer="zeros", dtype=tf.float64))

    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred, weight = _weighted_vectors(y_true, y_pred, sample_weight)
        for field, value in (
            ("mass", weight), ("sum_true", weight * true), ("sum_pred", weight * pred),
            ("sum_tt", weight * true * true), ("sum_pp", weight * pred * pred),
            ("sum_tp", weight * true * pred), ("sse", weight * tf.square(true - pred)),
        ):
            getattr(self, field).assign_add(tf.reduce_sum(value))

    def reset_state(self):
        for variable in self.variables:
            variable.assign(tf.zeros_like(variable))


@tf.keras.utils.register_keras_serializable(package="lit")
class R2Score(_MomentMetric):
    """Weighted pooled R²; empty/zero-weight=0, constant target perfect=1 else=0."""
    def __init__(self, name="r2", **kwargs):
        super().__init__(name=name, **kwargs)

    def result(self):
        total = tf.maximum(self.sum_tt - tf.math.divide_no_nan(tf.square(self.sum_true), self.mass), 0.)
        score = tf.where(total > 0., 1. - tf.math.divide_no_nan(self.sse, total),
                         tf.cast(self.sse == 0., tf.float64))
        return tf.where(self.mass > 0., score, 0.)


@tf.keras.utils.register_keras_serializable(package="lit")
class PearsonCorrelation(_MomentMetric):
    """Pooled IC, unweighted when registered in compile(metrics=...)."""
    def __init__(self, name="pearson", **kwargs):
        super().__init__(name=name, **kwargs)

    def result(self):
        cov = self.sum_tp - tf.math.divide_no_nan(self.sum_true * self.sum_pred, self.mass)
        tt = tf.maximum(self.sum_tt - tf.math.divide_no_nan(tf.square(self.sum_true), self.mass), 0.)
        pp = tf.maximum(self.sum_pp - tf.math.divide_no_nan(tf.square(self.sum_pred), self.mass), 0.)
        return tf.clip_by_value(tf.math.divide_no_nan(cov, tf.sqrt(tt * pp)), -1., 1.)


def require_supported_loss_strategy(loss_name, replicas):
    if loss_name == "ccc" and int(replicas) != 1:
        raise ValueError("weighted CCC supports one replica only; expose one GPU or select mse/huber/logcosh")


@tf.keras.utils.register_keras_serializable(package="lit")
class WeightedCCCLoss(tf.keras.losses.Loss):
    """Single-replica batch CCC with weights INSIDE the moments.

    This batch statistic deliberately bypasses the elementwise Loss reduction.
    A zero-weight batch contributes zero. A singleton/constant target has no
    useful covariance gradient; use ordinary regression losses for such data.
    Loss values averaged across batches are not a dataset-wide CCC statistic.
    """
    def __init__(self, name="weighted_ccc_loss", reduction="sum_over_batch_size", **kwargs):
        if reduction != "sum_over_batch_size":
            raise ValueError("WeightedCCCLoss has a fixed batch-statistic reduction")
        super().__init__(name=name, reduction=reduction, **kwargs)

    def call(self, y_true, y_pred):
        return self.__call__(y_true, y_pred)

    def __call__(self, y_true, y_pred, sample_weight=None):
        require_supported_loss_strategy("ccc", tf.distribute.get_strategy().num_replicas_in_sync)
        true, pred, weight = _weighted_vectors(y_true, y_pred, sample_weight, tf.float32)
        mass = tf.reduce_sum(weight)
        mean_t = tf.math.divide_no_nan(tf.reduce_sum(weight * true), mass)
        mean_p = tf.math.divide_no_nan(tf.reduce_sum(weight * pred), mass)
        dt, dp = true - mean_t, pred - mean_p
        var_t = tf.math.divide_no_nan(tf.reduce_sum(weight * tf.square(dt)), mass)
        var_p = tf.math.divide_no_nan(tf.reduce_sum(weight * tf.square(dp)), mass)
        cov = tf.math.divide_no_nan(tf.reduce_sum(weight * dt * dp), mass)
        ccc = 2. * cov / (var_t + var_p + tf.square(mean_t - mean_p) + 1e-8)
        return tf.where(mass > 0., 1. - ccc, 0.)


# ---------------------------
# model & layers
# ---------------------------

@tf.keras.utils.register_keras_serializable(package="lit")
class FixedStandardize(tf.keras.layers.Layer):
    """(x-mean)/std with constants."""
    def __init__(self, mean, std, eps=1e-6, **kwargs):
        kwargs["dtype"] = "float32"
        kwargs["autocast"] = False
        super().__init__(**kwargs)
        mean_np = np.asarray(mean, dtype=np.float32).reshape((-1,))
        std_np  = np.asarray(std, dtype=np.float32).reshape((-1,))
        if (mean_np.size == 0 or mean_np.shape != std_np.shape or not np.isfinite(mean_np).all()
                or not np.isfinite(std_np).all() or np.any(std_np <= 0) or not np.isfinite(eps) or eps <= 0):
            raise ValueError("normalization requires matching finite means and positive standard deviations/eps")
        self.mean_list = [float(v) for v in mean_np.tolist()]
        self.std_list  = [float(v) for v in std_np.tolist()]
        self.eps = float(eps)
        mean_t = np.asarray(self.mean_list, dtype=np.float32).reshape((1, 1, -1))
        std_t  = np.asarray(self.std_list, dtype=np.float32).reshape((1, 1, -1))
        self.mean = tf.constant(mean_t, dtype=tf.float32)
        self.std  = tf.constant(std_t, dtype=tf.float32)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        return (x - self.mean) / tf.maximum(self.std, self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mean": self.mean_list, "std": self.std_list, "eps": self.eps})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="lit")
class TimeAwarePositionalEncoding(tf.keras.layers.Layer):
    """
    Continuous Time-Aware Positional Encoding.
    """
    def __init__(
        self,
        d_model: int,
        max_timescale: float = 10000.0,
        time_scale: float = 100.0,
        **kwargs,
    ):
        kwargs["autocast"] = False  # preserve float32 times before call(), even under mixed policies
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.max_timescale = float(max_timescale)
        self.time_scale = float(time_scale)

        if self.d_model < 2 or not np.isfinite(self.max_timescale) or self.max_timescale <= 0 or not np.isfinite(self.time_scale) or self.time_scale <= 0:
            raise ValueError("invalid time encoding dimensions/scales")
        num_freqs = self.d_model // 2
        # 标准频率生成 (最高频固定为1)
        exponents = tf.range(num_freqs, dtype=tf.float32) * (-np.log(self.max_timescale) / num_freqs)
        self.inv_freqs = tf.exp(exponents)  # [d_model/2]

    def call(self, x, t_secs):
        """
        x: [Batch, W, d_model]
        t_secs: [Batch, W] in seconds (float32)
        """
        # Callers must subtract absolute time in float64 before this layer.
        # Do trigonometry in float32 even under a mixed-precision policy.
        t_secs = tf.cast(t_secs, tf.float32)
        t_rel = t_secs - t_secs[:, 0:1]
        t_scaled = t_rel * self.time_scale

        # 3. 扩展维度
        t_rel_expanded = tf.expand_dims(t_scaled, -1)
        freqs_expanded = tf.reshape(self.inv_freqs, (1, 1, -1))

        # 4. 计算
        args = t_rel_expanded * freqs_expanded
        pe_sin = tf.sin(args)
        pe_cos = tf.cos(args)
        pos_enc = tf.concat([pe_sin, pe_cos], axis=-1)

        if self.d_model % 2 != 0:
            pos_enc = tf.pad(pos_enc, [[0, 0], [0, 0], [0, 1]])

        return x + tf.cast(pos_enc, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "max_timescale": self.max_timescale,
            "time_scale": self.time_scale,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_transformer_lstm_regressor(
    W: int,
    F: int,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    model_cfg: dict,
) -> tf.keras.Model:
    d_model = int(model_cfg["d_model"])
    num_heads = int(model_cfg["num_heads"])
    num_layers = int(model_cfg["num_layers"])
    ff_dim = int(model_cfg["ff_dim"])
    dropout = float(model_cfg.get("dropout", 0.1))
    use_lstm = bool(model_cfg.get("use_lstm", True))
    lstm_units = int(model_cfg.get("lstm_units", 128))
    head_hidden = int(model_cfg.get("head_hidden", 128))
    use_time_aware_pos = bool(model_cfg.get("use_time_aware_pos", True))
    max_timescale = float(model_cfg.get("max_timescale", 10000.0))
    time_scale = float(model_cfg.get("time_scale", 100.0))

    if d_model < 2 or num_heads < 1 or d_model % num_heads != 0:
        raise ValueError("d_model must be at least 2 and divisible by a positive num_heads")
    if num_layers < 1 or ff_dim < 1 or lstm_units < 1 or head_hidden < 1:
        raise ValueError("model layer counts and hidden dimensions must be positive")
    if not np.isfinite(max_timescale) or max_timescale <= 0.0:
        raise ValueError("model.max_timescale must be finite and positive")
    if not np.isfinite(time_scale) or time_scale <= 0.0:
        raise ValueError("model.time_scale must be finite and positive")

    # Inputs: Features and Time
    inp_x = tf.keras.Input(shape=(W, F), name="x_factors")   # [B, W, F]
    inp_t = tf.keras.Input(shape=(W,), name="t_secs")        # [B, W]

    x = inp_x

    # Standardization
    if (mean is not None) and (std is not None):
        x = FixedStandardize(mean, std, eps=1e-6, name="fixed_norm")(x)
    else:
        x = tf.keras.layers.Rescaling(1.0, dtype="float32", name="cast_fp32")(x)

    # Projection to d_model
    x = tf.keras.layers.Dense(d_model, name="proj")(x)

    if use_time_aware_pos:
        x = TimeAwarePositionalEncoding(
            d_model,
            max_timescale=max_timescale,
            time_scale=time_scale,
            name="time_aware_pos",
        )(x, inp_t)
    else:
        # Keep the time input connected to the Functional graph without
        # injecting time information.
        time_zero = tf.keras.layers.Reshape((W, 1), name="disabled_time_reshape")(inp_t)
        time_zero = tf.keras.layers.Dense(
            d_model,
            use_bias=False,
            trainable=False,
            kernel_initializer="zeros",
            name="disabled_time_projection",
        )(time_zero)
        x = tf.keras.layers.Add(name="disabled_time_passthrough")([x, time_zero])

    x = tf.keras.layers.Dropout(dropout)(x)

    # Transformer Blocks
    for i in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"mha_{i}",
        )(x, x)
        x = tf.keras.layers.Add(name=f"res_attn_{i}")([x, attn])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln_attn_{i}")(x)

        ff = tf.keras.layers.Dense(ff_dim, activation="gelu", name=f"ff1_{i}")(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(d_model, name=f"ff2_{i}")(ff)
        x = tf.keras.layers.Add(name=f"res_ff_{i}")([x, ff])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln_ff_{i}")(x)

    # LSTM or Pooling
    if use_lstm:
        x = tf.keras.layers.LSTM(lstm_units, name="lstm")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    # Prediction Head
    x = tf.keras.layers.Dense(head_hidden, activation="gelu", name="head_fc")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    out = tf.keras.layers.Dense(1, dtype="float32", name="y_pred")(x)

    model = tf.keras.Model(inputs=[inp_x, inp_t], outputs=out, name="lit_regression_v6_time")
    return model


# ---------------------------
# dataset builder
# ---------------------------

def make_tf_dataset(
    project_root: str,
    stage4_manifest_path: str,
    split: str,
    batch_size: int,
    window_W: int,
    num_factors: int,
    shuffle_blocks: bool,
    seed: int,
    mmap_cache_items: int,
    prefetch_batches: int,
    drop_remainder: bool,
) -> tf.data.Dataset:

    def gen():
        epoch = 0
        while True:
            lc = LoaderConfig(
                project_root=project_root,
                stage4_manifest_path=stage4_manifest_path,
                split=split,
                batch_size=batch_size,
                window_W=window_W,
                num_factors=num_factors,
                shuffle_blocks=shuffle_blocks,
                seed=seed + epoch * 10007,
                drop_remainder=drop_remainder,
                mmap_cache_items=mmap_cache_items,
            )
            it = iter_batches_with_time(lc)
            it = iter_batches_prefetch(it, prefetch_batches)

            yielded = 0
            for X, t, y, w in it:
                yielded += 1
                yield (
                    (X.astype(np.float32, copy=False),
                     t.astype(np.float32, copy=False)),
                    y.astype(np.float32, copy=False).reshape((-1, 1)),
                    w.astype(np.float32, copy=False)
                )
            if yielded == 0:
                raise RuntimeError(
                    f"No batches produced for split={split}; check Stage4 blocks, "
                    f"batch_size={batch_size}, and drop_remainder={drop_remainder}"
                )
            epoch += 1

    batch_dim = batch_size if drop_remainder else None
    output_signature = (
        (
            tf.TensorSpec(shape=(batch_dim, window_W, num_factors), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_dim, window_W), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(batch_dim, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_dim,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------
# main
# ---------------------------

def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, type=str)
    ap.add_argument("--smoke-test", action="store_true", help="Run a tiny training (few steps) and exit.")
    ap.add_argument("--run-test", action="store_true", help="Evaluate on test split after training.")
    args = ap.parse_args()

    cfg = load_config(args.config_path)
    root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]

    # stage4 manifest
    st4_path = cfg["stage4"]["stage4_manifest_path"].format(horizon_id=horizon_id)
    st4_path = abspath(root, st4_path)
    st4 = load_json(st4_path)
    if st4.get("horizon_id") != horizon_id:
        raise ValueError("Stage4 horizon differs from the training configuration")
    W = int(st4["window_W"])
    if W != int(cfg["features"]["window_W"]):
        raise ValueError("Stage4 window size differs from the training configuration")
    F4 = int(st4.get("num_factors", 0)) if isinstance(st4.get("num_factors", 0), (int, float)) else 0
    factor_schema_path = st4.get("factor_schema_path")

    # input stats
    input_stats_path = cfg["stage3"].get("input_stats_path", "data/stats/input_factors_stats_{horizon_id}.json")
    input_stats_path = abspath(root, input_stats_path.format(horizon_id=horizon_id))
    mean = std = None
    F_stats = 0
    if os.path.exists(input_stats_path):
        st = load_json(input_stats_path)
        mean = np.asarray(st["mean"], dtype=np.float32)
        std = np.asarray(st["std"], dtype=np.float32)
        F_stats = int(st.get("num_factors", len(mean)))

    if bool(cfg.get("train", {}).get("require_input_stats", True)) and mean is None:
        raise RuntimeError(
            f"Training input statistics are required but missing: {input_stats_path}. Run Stage 3 first."
        )

    if F4: F = F4
    elif F_stats: F = F_stats
    else:
        packs_path = abspath(root, cfg["stage3"]["packs_manifest_path"].format(horizon_id=horizon_id))
        packs = load_json(packs_path)
        F = int(packs["num_factors"])

    if mean is not None:
        if mean.shape != (F,) or std.shape != (F,):
            raise RuntimeError(
                f"Input-stat dimension mismatch: mean={mean.shape}, std={std.shape}, expected ({F},)"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
            raise RuntimeError("Input statistics must contain finite means and positive finite standard deviations")

    if not factor_schema_path:
        raise RuntimeError("Stage4 manifest does not identify the Stage3 factor schema")
    factor_schema_abs = abspath(root, factor_schema_path)
    if not os.path.exists(factor_schema_abs):
        raise RuntimeError(f"Stage3 factor schema not found: {factor_schema_abs}")
    factor_schema = load_json(factor_schema_abs)
    factor_cols = list(factor_schema.get("factor_cols") or [])
    if len(factor_cols) != F:
        raise RuntimeError(
            f"Factor schema length={len(factor_cols)} does not match model input F={F}"
        )
    norm_path = factor_schema.get("norm_map_path")
    norm_map = load_json(abspath(root, norm_path)) if norm_path else None
    if norm_path:
        require_fingerprint(abspath(root, norm_path), factor_schema.get("norm_map_sha256"))
    require_preprocessing_spec(cfg, factor_cols, norm_map, factor_schema.get("preprocessing"))
    if mean is not None:
        require_fingerprint(factor_schema_abs, st.get("factor_schema_sha256"))

    binding = load_target_binding(cfg)
    require_target_binding(st4, binding)
    if mean is not None:
        require_target_binding(st, binding)

    # train cfg
    tr = cfg["train"]
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Stage6 training requires a visible GPU and a compatible TensorFlow environment")
    seed = int(tr.get("seed", 42))
    set_global_seeds(seed)

    # Reset any inherited policy explicitly.
    # mixed precision
    mp = bool(tr.get("mixed_precision", True))
    if mp:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        logging.info("[Stage6] mixed_precision enabled: mixed_bfloat16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        logging.info("[Stage6] mixed_precision disabled")

    # strategy
    strategy_name = tr.get("strategy", "mirrored")
    if strategy_name == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    nrep = strategy.num_replicas_in_sync
    require_supported_loss_strategy(tr.get("loss", "ccc"), nrep)

    global_batch = int(tr["global_batch"])
    if global_batch % nrep != 0:
        raise RuntimeError(f"global_batch={global_batch} must be divisible by replicas={nrep}")

    # dataloader cfg
    dl = cfg["dataloader"]
    mmap_cache_items = int(dl.get("mmap_cache_items", 16))
    prefetch_batches = int(dl.get("prefetch_batches", 4))
    drop_remainder_train = bool(dl.get("drop_remainder_train", dl.get("drop_remainder", True)))
    drop_remainder_eval = bool(dl.get("drop_remainder_eval", False))
    shuffle_blocks_train = bool(dl.get("shuffle_blocks_train", True))

    available_train_steps = count_split_batches(
        st4, "train", global_batch, drop_remainder_train
    )
    available_val_steps = count_split_batches(
        st4, "val", global_batch, drop_remainder_eval
    )
    if available_train_steps <= 0:
        raise RuntimeError(
            "Training split produces zero batches. Reduce global_batch or set "
            "dataloader.drop_remainder_train=false."
        )
    if available_val_steps <= 0:
        raise RuntimeError("Validation split contains no usable Stage4 windows.")

    configured_train_steps = int(tr.get("steps_per_epoch", 0))
    configured_val_steps = int(tr.get("val_steps", 0))
    if configured_train_steps < 0 or configured_val_steps < 0 or int(tr.get("test_steps", 0)) < 0:
        raise ValueError("step counts must be zero (automatic) or positive")
    steps_per_epoch = configured_train_steps or available_train_steps
    val_steps = configured_val_steps or available_val_steps
    if steps_per_epoch > available_train_steps:
        raise RuntimeError(
            f"steps_per_epoch={steps_per_epoch} exceeds one-pass batches={available_train_steps}"
        )
    if val_steps > available_val_steps:
        raise RuntimeError(f"val_steps={val_steps} exceeds one-pass batches={available_val_steps}")
    if not args.smoke_test and (drop_remainder_eval or val_steps != available_val_steps):
        raise ValueError("normal validation must cover every row: drop_remainder_eval=false and val_steps=0")

    if args.smoke_test:
        steps_per_epoch = min(30, steps_per_epoch)
        val_steps = min(10, val_steps)

    logging.info(
        f"[Stage6] W={W} F={F} global_batch={global_batch} "
        f"train_steps={steps_per_epoch}/{available_train_steps} "
        f"val_steps={val_steps}/{available_val_steps}"
    )

    train_ds = make_tf_dataset(
        project_root=root,
        stage4_manifest_path=st4_path,
        split="train",
        batch_size=global_batch,
        window_W=W,
        num_factors=F,
        shuffle_blocks=shuffle_blocks_train,
        seed=seed,
        mmap_cache_items=mmap_cache_items,
        prefetch_batches=prefetch_batches,
        drop_remainder=drop_remainder_train,
    )
    val_ds = make_tf_dataset(
        project_root=root,
        stage4_manifest_path=st4_path,
        split="val",
        batch_size=global_batch,
        window_W=W,
        num_factors=F,
        shuffle_blocks=False,
        seed=seed,
        mmap_cache_items=mmap_cache_items,
        prefetch_batches=prefetch_batches,
        drop_remainder=drop_remainder_eval,
    )

    out_root = abspath(root, cfg["paths"]["results_dir"])
    run_name = tr.get("run_name", f"{horizon_id}_{time.strftime('%Y%m%d_%H%M%S')}_time_aware")
    run_dir = os.path.join(out_root, run_name)
    if os.path.exists(os.path.join(run_dir, "model_contract.json")):
        raise FileExistsError("run_name already has a model contract; use a new run_name to avoid stale checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    save_json(os.path.join(run_dir, "config_snapshot.json"), cfg)
    model_contract_path = os.path.join(run_dir, "model_contract.json")
    contract_norm_map_path = st4.get("norm_map_path") or factor_schema.get("norm_map_path")
    contract_norm_map_abs = abspath(root, contract_norm_map_path) if contract_norm_map_path else None
    model_contract = {
        "contract_version": MODEL_CONTRACT_VERSION,
        **binding,
        "status": "training",
        "model_component_version": 2,
        "precision_policy": tf.keras.mixed_precision.global_policy().name,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "run_name": run_name,
        "horizon_id": horizon_id,
        "window_W": W,
        "num_factors": F,
        "factor_cols": factor_cols,
        "factor_schema_path": factor_schema_path,
        "factor_schema_sha256": sha256_file(factor_schema_abs),
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing": factor_schema["preprocessing"],
        "norm_map_path": contract_norm_map_path,
        "norm_map_sha256": (
            sha256_file(contract_norm_map_abs)
            if contract_norm_map_abs and os.path.exists(contract_norm_map_abs)
            else None
        ),
        "input_stats_path": input_stats_path if mean is not None else None,
        "input_stats_sha256": sha256_file(input_stats_path) if mean is not None else None,
        "stage4_manifest_path": st4_path,
        "stage4_manifest_sha256": sha256_file(st4_path),
        "window_constraints": st4.get("window_constraints", {}),
        "model": cfg["model"],
    }
    save_json(model_contract_path, model_contract)

    with strategy.scope():
        # Build new model
        model = build_transformer_lstm_regressor(
            W=W, F=F, mean=mean, std=std, model_cfg=cfg["model"]
        )

        epochs = int(tr["epochs"])
        total_steps = int(epochs * steps_per_epoch)
        warmup_ratio = float(tr.get("warmup_ratio", 0.05))
        warmup_steps = min(total_steps - 1, int(total_steps * warmup_ratio))
        lr = WarmupCosine(
            base_lr=float(tr["base_lr"]),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=float(tr.get("min_lr_ratio", 0.05)),
        )

        opt_cfg = cfg["train"]["optimizer"]
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
            beta_1=float(opt_cfg.get("beta1", 0.9)),
            beta_2=float(opt_cfg.get("beta2", 0.999)),
            epsilon=float(opt_cfg.get("epsilon", 1e-7)),
            clipnorm=float(opt_cfg.get("clipnorm", 1.0)),
        )

        # [MODIFIED] Loss Selection: Use WeightedCCCLoss by default
        loss_name = tr.get("loss", "ccc")
        if loss_name == "mse":
            loss = tf.keras.losses.MeanSquaredError()
            logging.info("[Stage6] Using MeanSquaredError loss")
        elif loss_name == "huber":
            loss = tf.keras.losses.Huber(delta=float(tr.get("huber_delta", 1.0)))
            logging.info("[Stage6] Using Huber loss")
        elif loss_name == "logcosh":
            loss = tf.keras.losses.LogCosh()
            logging.info("[Stage6] Using LogCosh loss")
        elif loss_name == "ccc":
            # Explicitly utilize WeightedCCCLoss which overrides __call__ to capture sample_weight
            loss = WeightedCCCLoss()
            logging.info("[Stage6] Using WeightedCCCLoss (1 - Weighted_CCC)")
        else:
            raise ValueError(f"unsupported loss: {loss_name}")

        # [MODIFIED] Added PearsonCorrelation (Unweighted) for validation
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[PearsonCorrelation(name="pearson")],
            weighted_metrics=[
                WeightedDirAcc(name="dir_acc"),
                R2Score(name="r2"),
            ],
            jit_compile=bool(tr.get("jit_compile", False)),
        )

    model.summary(print_fn=logging.info)

    cb = []
    cb.append(tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "logs", "train_log.csv")))
    cb.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "logs", "tb")))

    # [MODIFIED] Monitor val_pearson for EarlyStopping
    cb.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_pearson", patience=int(tr.get("early_stop_patience", 10)), restore_best_weights=True, mode="max"
    ))
    ckpt_path = os.path.join(run_dir, "models", "best.weights.h5")
    cb.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, monitor="val_pearson", save_best_only=True, save_weights_only=True, mode="max"
    ))

    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs if not args.smoke_test else 2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cb,
        verbose=1
    )
    dt = time.time() - t0
    logging.info(f"[Stage6] train finished in {dt/60:.1f} min")

    # Explicitly restore the selected checkpoint, independently of callback lifecycle.
    model.load_weights(ckpt_path)
    model_path = os.path.join(run_dir, "models", "final.keras")
    model.save(model_path)
    logging.info(f"[Stage6] saved model: {model_path}")
    model_contract.update({
        "status": "smoke_test" if args.smoke_test else "trained",
        "model_sha256": sha256_file(model_path),
        "best_weights_sha256": sha256_file(ckpt_path),
        "replicas": int(nrep),
        "ccc_reduction": "single_replica_weighted_batch" if loss_name == "ccc" else None,
    })
    save_json(model_contract_path, model_contract)

    # Test
    if args.run_test and (not args.smoke_test):
        available_test_steps = count_split_batches(
            st4, "test", global_batch, drop_remainder_eval
        )
        if available_test_steps <= 0:
            raise RuntimeError("Test split contains no usable Stage4 windows.")
        test_steps = int(tr.get("test_steps", 0)) or available_test_steps
        if test_steps != available_test_steps:
            raise RuntimeError(
                f"test_steps={test_steps} must cover all {available_test_steps} batches"
            )
        test_ds = make_tf_dataset(
            project_root=root,
            stage4_manifest_path=st4_path,
            split="test",
            batch_size=global_batch,
            window_W=W,
            num_factors=F,
            shuffle_blocks=False,
            seed=seed,
            mmap_cache_items=mmap_cache_items,
            prefetch_batches=prefetch_batches,
            drop_remainder=drop_remainder_eval,
        )
        metrics = model.evaluate(test_ds, steps=test_steps, return_dict=True, verbose=1)
        save_json(os.path.join(run_dir, "test_metrics.json"), metrics)
        logging.info(f"[Stage6] test metrics saved.")

    summary = {
        "run_dir": run_dir,
        "horizon_id": horizon_id,
        "W": W,
        "F": F,
        "train_seconds": dt,
        "history_keys": list(hist.history.keys()),
        "time_aware": bool(cfg["model"].get("use_time_aware_pos", True)),
        "model_contract_path": model_contract_path,
        "data_coverage": {
            "train_batches_available": available_train_steps,
            "train_batches_used": steps_per_epoch,
            "val_batches_available": available_val_steps,
            "val_batches_used": val_steps,
            "drop_remainder_train": drop_remainder_train,
            "drop_remainder_eval": drop_remainder_eval,
        },
        "loss_type": "weighted_ccc" if isinstance(loss, WeightedCCCLoss) else loss_name
    }
    save_json(os.path.join(run_dir, "run_summary.json"), summary)


if __name__ == "__main__":
    main()
