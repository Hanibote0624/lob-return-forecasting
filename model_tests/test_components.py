"""Small numerical/serialization checks; no research-data training."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from src.stage6_train_regression import (
    FixedStandardize, TimeAwarePositionalEncoding, WarmupCosine, WeightedCCCLoss,
    WeightedDirAcc, PearsonCorrelation, R2Score, build_transformer_lstm_regressor,
    require_supported_loss_strategy,
)


def reference_ccc(true, pred, weight):
    true, pred, weight = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (true, pred, weight))
    mt, mp = np.average(true, weights=weight), np.average(pred, weights=weight)
    vt = np.average((true - mt) ** 2, weights=weight)
    vp = np.average((pred - mp) ** 2, weights=weight)
    cov = np.average((true - mt) * (pred - mp), weights=weight)
    return 1 - 2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-8)


def small_model(time_aware=True):
    return build_transformer_lstm_regressor(4, 3, np.zeros(3), np.ones(3), {
        "d_model": 8, "num_heads": 2, "num_layers": 1, "ff_dim": 12,
        "dropout": 0., "use_lstm": True, "lstm_units": 6, "head_hidden": 5,
        "use_time_aware_pos": time_aware, "time_scale": 100.,
    })


class Components(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(42)
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")

    def test_fixed_standardization_preserves_small_differences_under_mixed_policy(self):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        layer = FixedStandardize([10000.], [1.])
        result = layer(tf.constant([[[10000.], [10001.]]]))
        np.testing.assert_array_equal(result.numpy().reshape(-1), [0., 1.])
        self.assertEqual(result.dtype, tf.float32)
        clone = tf.keras.layers.deserialize(tf.keras.layers.serialize(layer))
        np.testing.assert_array_equal(clone(tf.constant([[[10001.]]])).numpy(), [[[1.]]])

    def test_time_encoding_matches_reference_including_odd_width(self):
        for width in (4, 5):
            times = np.array([[0., .01, .01, .3]], np.float32)
            layer = TimeAwarePositionalEncoding(width)
            value = layer(tf.zeros([1, 4, width]), tf.constant(times)).numpy()
            frequency = np.exp(np.arange(width // 2) * (-np.log(10000) / (width // 2)))
            angle = times[..., None] * 100 * frequency
            expected = np.concatenate((np.sin(angle), np.cos(angle)), axis=-1)
            if width % 2:
                expected = np.pad(expected, ((0, 0), (0, 0), (0, 1)))
            np.testing.assert_allclose(value, expected, atol=2e-6)

    def test_millisecond_time_difference_survives_mixed_policy(self):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        layer = TimeAwarePositionalEncoding(4)
        times = tf.constant([[0., 8., 8.001]])
        value = layer(tf.zeros([1, 3, 4], dtype=tf.bfloat16), times).numpy().astype(np.float32)
        self.assertGreater(float(np.max(np.abs(value[:, 1] - value[:, 2]))), .01)

    def test_weighted_ccc_value_and_gradient(self):
        true = tf.constant([[-.5], [.2], [.6], [1.]])
        pred = tf.Variable([[-.2], [.4], [.9], [.1]])
        weight = tf.constant([1., 2., 0., 4.])
        with tf.GradientTape() as tape:
            value = WeightedCCCLoss()(true, pred, weight)
        gradient = tape.gradient(value, pred).numpy()
        self.assertAlmostEqual(float(value), reference_ccc(true.numpy(), pred.numpy(), weight.numpy()), places=6)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertNotEqual(float(np.linalg.norm(gradient)), 0.)
        self.assertEqual(float(gradient[2, 0]), 0.)

    def test_zero_weight_loss_and_metric_do_not_use_unweighted_fallback(self):
        true, pred, weight = [1., 2.], [2., 4.], [0., 0.]
        self.assertEqual(float(WeightedCCCLoss()(true, pred, weight)), 0.)
        metric = R2Score()
        metric.update_state(true, pred, weight)
        self.assertEqual(float(metric.result()), 0.)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            WeightedCCCLoss()(true, pred, [-1., 1.])

    def test_pooled_metrics_match_reference_across_uneven_batches(self):
        true = np.array([-.4, .001, .3, .8, -.2, .7], dtype=np.float32)
        pred = np.array([-.3, -.002, .7, .2, -.1, .8], dtype=np.float32)
        weight = np.array([1., 0., 3., 2., 1., 4.], dtype=np.float32)
        metrics = [PearsonCorrelation(), R2Score(), WeightedDirAcc()]
        for a, b in ((0, 2), (2, 5), (5, 6)):
            metrics[0].update_state(true[a:b], pred[a:b])
            for m in metrics[1:]:
                m.update_state(true[a:b], pred[a:b], weight[a:b])
        self.assertAlmostEqual(float(metrics[0].result()), np.corrcoef(true, pred)[0, 1], places=7)
        mean = np.average(true.astype(np.float64), weights=weight)
        r2 = 1 - np.sum(weight * (true.astype(np.float64) - pred) ** 2) / np.sum(weight * (true - mean) ** 2)
        self.assertAlmostEqual(float(metrics[1].result()), r2, places=7)
        expected_dir = np.average(np.sign(true) == np.sign(pred), weights=weight)
        self.assertAlmostEqual(float(metrics[2].result()), expected_dir, places=7)
        for metric in metrics:
            metric.reset_state()
            self.assertEqual(float(metric.result()), 0.)

    def test_tiny_return_variance_does_not_clamp_pearson(self):
        metric = PearsonCorrelation()
        metric.update_state([0., 1e-5, 2e-5], [0., 2e-5, 4e-5])
        self.assertAlmostEqual(float(metric.result()), 1., places=7)

    def test_constant_target_r2_policy(self):
        for pred, expected in (([1., 1.], 1.), ([1., 2.], 0.)):
            metric = R2Score()
            metric.update_state([1., 1.], pred)
            self.assertEqual(float(metric.result()), expected)

    def test_schedule_boundaries_and_serialization(self):
        schedule = WarmupCosine(.01, 10, 2, .1)
        np.testing.assert_allclose([schedule(i) for i in (0, 2, 10, 100)], [0., .01, .001, .001], rtol=1e-6)
        clone = tf.keras.optimizers.schedules.deserialize(tf.keras.optimizers.schedules.serialize(schedule))
        self.assertAlmostEqual(float(clone(5)), float(schedule(5)), places=8)
        self.assertAlmostEqual(float(WarmupCosine(.01, 1, 0)(0)), .01, places=7)
        with self.assertRaises(ValueError):
            WarmupCosine(.01, 2, 2)

    def test_ccc_multiple_replicas_is_explicitly_rejected(self):
        require_supported_loss_strategy("ccc", 1)
        with self.assertRaisesRegex(ValueError, "one replica"):
            require_supported_loss_strategy("ccc", 2)
        require_supported_loss_strategy("mse", 2)

    def test_time_switch_changes_only_the_enabled_model(self):
        features = np.random.default_rng(4).normal(size=(3, 4, 3)).astype(np.float32)
        early = np.tile(np.array([0., .01, .02, .03], np.float32), (3, 1))
        late = early * 3
        for enabled in (False, True):
            model = small_model(enabled)
            first = model([features, early], training=False).numpy()
            second = model([features, late], training=False).numpy()
            if enabled:
                self.assertGreater(float(np.max(np.abs(first - second))), 1e-7)
            else:
                np.testing.assert_array_equal(first, second)

    def test_compiled_weighted_losses_update_and_round_trip(self):
        rng = np.random.default_rng(5)
        inputs = [rng.normal(size=(5, 4, 3)).astype(np.float32),
                  np.tile(np.array([0., .001, .07, .2], np.float32), (5, 1))]
        true = np.array([-.3, .1, .6, -.1, .7], np.float32)[:, None]
        weight = np.array([1., 2., 0., 3., 1.], np.float32)
        for policy in ("float32", "mixed_bfloat16"):
            for loss_name in ("ccc", "mse", "huber", "logcosh"):
                with self.subTest(policy=policy, loss=loss_name), tempfile.TemporaryDirectory() as temp:
                    tf.keras.mixed_precision.set_global_policy(policy)
                    model = small_model()
                    loss = {"ccc": WeightedCCCLoss, "mse": tf.keras.losses.MeanSquaredError,
                            "huber": tf.keras.losses.Huber, "logcosh": tf.keras.losses.LogCosh}[loss_name]()
                    model.compile(optimizer=tf.keras.optimizers.AdamW(WarmupCosine(.001, 10, 0)), loss=loss,
                                  metrics=[PearsonCorrelation()], weighted_metrics=[R2Score(), WeightedDirAcc()],
                                  jit_compile=False)
                    before = model(inputs, training=False).numpy()
                    value = model.test_on_batch(inputs, true, sample_weight=weight, return_dict=True)
                    if loss_name == "ccc":
                        self.assertAlmostEqual(value["loss"], reference_ccc(true, before, weight), places=5)
                    model.reset_metrics()
                    update = model.train_on_batch(inputs, true, sample_weight=weight, return_dict=True)
                    self.assertTrue(all(np.isfinite(v) for v in update.values()))
                    expected = model(inputs, training=False).numpy()
                    self.assertGreater(float(np.max(np.abs(before - expected))), 0.)
                    model_path = Path(temp) / "model.keras"
                    weight_path = Path(temp) / "best.weights.h5"
                    model.save(model_path)
                    model.save_weights(weight_path)
                    restored = tf.keras.models.load_model(model_path, compile=True)
                    np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected, atol=1e-6, rtol=1e-5)
                    restored.load_weights(weight_path)
                    np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual(int(restored.optimizer.iterations), 1)
                    self.assertTrue(np.isfinite(restored.test_on_batch(inputs, true, sample_weight=weight)).all())


if __name__ == "__main__":
    unittest.main()
