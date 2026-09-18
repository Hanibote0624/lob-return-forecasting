import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.feature_preprocessing import (
    compute_reference_price,
    preprocess_session_features,
    preprocessing_spec,
    require_preprocessing_spec,
    required_feature_columns,
)
from src.prediction_io import (
    iter_window_batches,
    load_prediction_bundle,
    save_prediction_bundle,
)


def example_config():
    return {
        "data": {
            "stocks": [{"stock_code": "AAA", "float_shares": 100.0}],
            "session_rules": {
                "ffill_within_session": True,
                "fill_remaining_nan": 0.0,
            },
            "market": {
                "columns": {
                    "acc_volume": "acc_volume",
                    "bid": "bid",
                    "ask": "ask",
                    "last": "last",
                    "mid": None,
                }
            },
            "required_columns": {
                "mid_price": {
                    "prefer_bidask": True,
                    "bid1": "bid",
                    "ask1": "ask",
                    "fallback_last": "last",
                }
            },
        },
        "audit": {"scale_audit": {"ema_span": 2, "dv_floor": 1.0, "mcap_floor": 1.0}},
    }


class SharedPreprocessingTests(unittest.TestCase):
    def test_contract_rejects_changed_fill_or_scaling_parameters(self):
        cfg = example_config()
        norm_map = {"groups": {"volume_norm": ["f_volume"]}}
        expected = preprocessing_spec(cfg, ["f_volume"], norm_map)
        require_preprocessing_spec(cfg, ["f_volume"], norm_map, expected)
        cfg["data"]["session_rules"]["fill_remaining_nan"] = 5.0
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            require_preprocessing_spec(cfg, ["f_volume"], norm_map, expected)
        cfg["data"]["session_rules"]["fill_remaining_nan"] = 0.0
        cfg["audit"]["scale_audit"]["ema_span"] = 99
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            require_preprocessing_spec(cfg, ["f_volume"], norm_map, expected)

    def test_required_columns_are_deduplicated(self):
        columns = required_feature_columns(
            ["f_volume", "f_mcap"],
            example_config(),
            ["f_volume"],
            ["f_mcap"],
        )
        self.assertEqual(
            columns,
            ["f_volume", "f_mcap", "acc_volume", "bid", "ask", "last"],
        )

    def test_crossed_quote_uses_positive_last_trade(self):
        frame = pd.DataFrame(
            {"bid": [10.0, 12.0], "ask": [10.2, 11.0], "last": [10.1, 11.5]}
        )
        np.testing.assert_allclose(compute_reference_price(frame, example_config()), [10.1, 11.5])

    def test_volume_and_market_cap_rules_are_shared(self):
        frame = pd.DataFrame(
            {
                "f_volume": [2.0, np.nan, 8.0],
                "f_mcap": [1000.0, 2000.0, 3000.0],
                "acc_volume": [100.0, 102.0, 106.0],
                "bid": [9.9, 19.9, 29.9],
                "ask": [10.1, 20.1, 30.1],
                "last": [10.0, 20.0, 30.0],
            }
        )
        result = preprocess_session_features(
            frame,
            ["f_volume", "f_mcap"],
            example_config(),
            volume_norm_factors=["f_volume"],
            mcap_norm_factors=["f_mcap"],
            norm_meta={"ema_span": 2, "dv_floor": 1.0, "mcap_floor": 1.0},
            stock_code="AAA",
        )
        # dv=[0,2,4], pandas EWM(span=2, adjust=False)=[0,4/3,28/9].
        expected_volume = np.asarray([2.0, 1.5, 8.0 / (28.0 / 9.0)], dtype=np.float32)
        expected_mcap = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(result[:, 0], expected_volume, rtol=1e-6)
        np.testing.assert_allclose(result[:, 1], expected_mcap, rtol=1e-6)

    def test_missing_float_shares_fails(self):
        cfg = example_config()
        cfg["data"]["stocks"] = []
        frame = pd.DataFrame({"f": [1.0], "bid": [1.0], "ask": [1.1], "last": [1.0]})
        with self.assertRaisesRegex(ValueError, "float_shares"):
            preprocess_session_features(
                frame,
                ["f"],
                cfg,
                mcap_norm_factors=["f"],
                stock_code="AAA",
            )


class PredictionBundleTests(unittest.TestCase):
    def test_bundle_rejects_fractional_or_negative_row_ids(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "bundle.npz")
            for rows in ([1.5, 2.5], [-1, 2]):
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    save_prediction_bundle(path, [0.1, 0.2], rows, [1.0, 2.0])

    def test_streaming_windows_keep_endpoint_rows_and_relative_time(self):
        X = np.arange(30, dtype=np.float32).reshape(10, 3)
        t_sec = 40_000.0 + np.arange(10, dtype=np.float64) * 0.01
        endpoints = np.asarray([3, 5, 9], dtype=np.int64)
        batches = list(iter_window_batches(X, t_sec, 4, 2, endpoints))
        self.assertEqual([len(item[2]) for item in batches], [2, 1])
        np.testing.assert_array_equal(np.concatenate([item[2] for item in batches]), endpoints)
        np.testing.assert_array_equal(batches[0][0][1], X[2:6])
        np.testing.assert_allclose(batches[0][1][:, 0], 0.0, atol=1e-7)
        np.testing.assert_allclose(np.diff(batches[0][1], axis=1), 0.01, atol=2e-6)

    def test_prediction_bundle_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "bundle.npz")
            save_prediction_bundle(
                path,
                pred=np.asarray([0.1, -0.2]),
                end_row=np.asarray([63, 70]),
                t_sec=np.asarray([3600.1, 3600.2]),
                metadata={"stock_code": "AAA"},
            )
            loaded = load_prediction_bundle(path)
        np.testing.assert_allclose(loaded["pred"], [0.1, -0.2])
        np.testing.assert_array_equal(loaded["end_row"], [63, 70])
        self.assertEqual(loaded["metadata"]["stock_code"], "AAA")

    def test_prediction_bundle_rejects_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "bundle.npz")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                save_prediction_bundle(path, [0.1, 0.2], [5, 5], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
