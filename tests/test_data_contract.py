import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_contract import (
    INCOMPLETE_HORIZON,
    INSUFFICIENT_FUTURE,
    INVALID_CURRENT_MID,
    build_event_mean_return_labels,
    hhmmssmmm_to_seconds_vec,
    relative_time_windows,
    valid_window_end_mask,
)
from src.stage2_build_labels_clear import compute_mid, worker_scale_from_raw
from src.stage3_pack_rows import RowPackWriter


class TimestampTests(unittest.TestCase):
    def test_millisecond_spacing_survives_relative_conversion(self):
        raw = np.array([93_000_000, 93_000_010, 93_000_020], dtype=np.int64)
        t_sec = hhmmssmmm_to_seconds_vec(raw)
        relative = relative_time_windows(t_sec[None, :])
        np.testing.assert_allclose(relative[0], [0.0, 0.01, 0.02], atol=1e-7)

    def test_invalid_clock_components_become_nan(self):
        raw = np.array([93_000_000, 96_000_000, 93_060_000], dtype=np.int64)
        converted = hhmmssmmm_to_seconds_vec(raw)
        self.assertTrue(np.isfinite(converted[0]))
        self.assertTrue(np.isnan(converted[1]))
        self.assertTrue(np.isnan(converted[2]))


class FutureLabelTests(unittest.TestCase):
    def test_half_tick_midpoint_and_last_price_are_not_rounded(self):
        actual = compute_mid(
            bid1=np.array([100.0, 101.0]), ask1=np.array([100.011, 100.0]),
            lastp=np.array([99.0, 100.0035]), prefer_bidask=True,
        )
        np.testing.assert_allclose(actual, [100.0055, 100.0035], rtol=0, atol=1e-12)

    def test_crossed_quote_falls_back_to_last_trade(self):
        mid = compute_mid(
            bid1=np.array([101.0]),
            ask1=np.array([100.0]),
            lastp=np.array([99.5]),
            prefer_bidask=True,
        )
        np.testing.assert_allclose(mid, [99.5])

    def test_empty_future_interval_is_invalid_not_zero(self):
        labels = build_event_mean_return_labels(
            t_sec=np.array([0.0, 1.0, 2.0, 10.0]),
            mid=np.array([100.0, 101.0, 102.0, 110.0]),
            min_seconds=3.0,
            max_seconds=4.0,
            require_full_horizon=True,
        )
        self.assertFalse(labels.is_valid[0])
        self.assertTrue(np.isnan(labels.r_raw[0]))
        self.assertNotEqual(labels.invalid_reason[0] & INSUFFICIENT_FUTURE, 0)

    def test_invalid_current_mid_is_invalid(self):
        labels = build_event_mean_return_labels(
            t_sec=np.array([0.0, 3.0, 4.0, 5.0]),
            mid=np.array([0.0, 103.0, 104.0, 105.0]),
            min_seconds=2.5,
            max_seconds=3.5,
        )
        self.assertFalse(labels.is_valid[0])
        self.assertTrue(np.isnan(labels.r_raw[0]))
        self.assertNotEqual(labels.invalid_reason[0] & INVALID_CURRENT_MID, 0)

    def test_full_horizon_policy_is_explicit(self):
        t = np.array([0.0, 2.7, 3.1])
        mid = np.array([100.0, 102.0, 104.0])
        strict = build_event_mean_return_labels(t, mid, 2.5, 3.5, require_full_horizon=True)
        partial = build_event_mean_return_labels(t, mid, 2.5, 3.5, require_full_horizon=False)

        self.assertFalse(strict.is_valid[0])
        self.assertNotEqual(strict.invalid_reason[0] & INCOMPLETE_HORIZON, 0)
        self.assertTrue(partial.is_valid[0])
        self.assertAlmostEqual(float(partial.r_raw[0]), 0.03, places=7)

    def test_future_mean_uses_only_positive_finite_prices(self):
        labels = build_event_mean_return_labels(
            t_sec=np.array([0.0, 2.5, 3.0, 3.5, 4.0]),
            mid=np.array([100.0, 102.0, 0.0, np.nan, 105.0]),
            min_seconds=2.5,
            max_seconds=3.5,
            min_future_observations=1,
        )
        self.assertTrue(labels.is_valid[0])
        self.assertEqual(int(labels.future_count[0]), 1)
        self.assertAlmostEqual(float(labels.r_raw[0]), 0.02, places=7)

    def test_minimum_future_observation_count(self):
        labels = build_event_mean_return_labels(
            t_sec=np.array([0.0, 2.5, 4.0]),
            mid=np.array([100.0, 102.0, 104.0]),
            min_seconds=2.5,
            max_seconds=3.5,
            min_future_observations=2,
        )
        self.assertFalse(labels.is_valid[0])
        self.assertNotEqual(labels.invalid_reason[0] & INSUFFICIENT_FUTURE, 0)


class WindowValidityTests(unittest.TestCase):
    def test_history_span_limit_is_enforced(self):
        weights = np.ones(5, dtype=np.float32)
        t_sec = np.array([0.0, 0.01, 0.02, 10.0, 10.01], dtype=np.float64)
        mask = valid_window_end_mask(
            weights,
            t_sec,
            3,
            max_history_span_seconds=1.0,
        )
        np.testing.assert_array_equal(mask, [False, False, True, False, False])

    def test_inter_event_gap_limit_is_enforced(self):
        weights = np.ones(5, dtype=np.float32)
        t_sec = np.array([0.0, 0.1, 0.2, 1.5, 1.6], dtype=np.float64)
        mask = valid_window_end_mask(
            weights,
            t_sec,
            3,
            max_inter_event_gap_seconds=0.5,
        )
        np.testing.assert_array_equal(mask, [False, False, True, False, False])

    def test_nonpositive_weight_invalidates_only_its_end(self):
        weights = np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        t_sec = np.arange(5, dtype=np.float64) * 0.1
        mask = valid_window_end_mask(weights, t_sec, 3)
        np.testing.assert_array_equal(mask, [False, False, True, False, True])


class ScalingTests(unittest.TestCase):
    def test_scale_pass_clips_and_preserves_float64_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.npz"
            out_path = Path(tmp) / "scaled.npz"
            from src.target_contract import json_sha256
            definition = {"fixture": "scaling-unit"}
            binding = {"target_contract": {"label_definition": definition},
                       "target_contract_sha256": "a" * 64, "label_stats_sha256": "b" * 64}
            np.savez(
                raw_path,
                label_definition_sha256=np.asarray(json_sha256(definition)),
                source_sha256=np.asarray("c" * 64),
                label_contract_version=np.array(2, dtype=np.int16),
                t_sec=np.array([34_200.0, 34_200.01], dtype=np.float64),
                mid=np.array([100.0055, 100.0035], dtype=np.float64),
                r_raw=np.array([0.001, 0.02], dtype=np.float32),
                is_valid=np.array([1, 1], dtype=np.uint8),
                invalid_reason=np.array([0, 0], dtype=np.uint8),
                future_count=np.array([2, 2], dtype=np.int32),
                lo=np.array([1, 1], dtype=np.int32),
                hi=np.array([1, 1], dtype=np.int32),
            )

            result = worker_scale_from_raw((str(raw_path), str(out_path), 200.0, 1.0, binding))
            self.assertTrue(result["ok"], result)
            with np.load(out_path) as scaled:
                self.assertEqual(scaled["t_sec"].dtype, np.dtype(np.float64))
                self.assertEqual(scaled["mid"].dtype, np.dtype(np.float64))
                np.testing.assert_allclose(scaled["mid"], [100.0055, 100.0035], rtol=0, atol=1e-12)
                np.testing.assert_allclose(scaled["r_scaled"], [0.2, 1.0])
                np.testing.assert_array_equal(scaled["was_clipped"], [0, 1])


class Stage3WriterTests(unittest.TestCase):
    def test_writer_uses_float64_time_and_shared_window_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = RowPackWriter(
                out_dir=tmp,
                split="train",
                shard_rows=8,
                num_factors=1,
                write_y_raw=True,
                write_t_sec=True,
                write_is_valid=True,
            )
            valid_end_count = writer.add_session(
                X=np.arange(5, dtype=np.float32).reshape(-1, 1),
                y=np.ones(5, dtype=np.float32),
                y_raw=np.ones(5, dtype=np.float32) * 0.001,
                w=np.ones(5, dtype=np.float32),
                is_valid=np.ones(5, dtype=bool),
                t_sec=np.array([0.0, 0.01, 0.02, 10.0, 10.01], dtype=np.float64),
                meta={"csv_path": "synthetic.csv", "date": "20250101", "session": 1},
                window_W=3,
                max_history_span_seconds=1.0,
                max_inter_event_gap_seconds=0.0,
            )
            shards = writer.finalize()

            self.assertEqual(valid_end_count, 1)
            t_saved = np.load(shards[0]["files"]["t_sec"], mmap_mode="r")
            self.assertEqual(t_saved.dtype, np.dtype(np.float64))
            segment_line = Path(shards[0]["files"]["segments"]).read_text().strip()
            self.assertEqual(json.loads(segment_line)["valid_end_count"], 1)


if __name__ == "__main__":
    unittest.main()
