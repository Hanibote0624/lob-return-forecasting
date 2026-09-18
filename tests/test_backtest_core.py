import unittest

import numpy as np

from src.backtest_core import (
    build_position_segments,
    calibrate_thresholds,
    generate_positions,
    require_prior_calibration,
)


class BacktestCoreTests(unittest.TestCase):
    def test_quantiles_use_the_finite_calibration_population(self):
        validation = [np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])]
        long, short = calibrate_thresholds(validation + [np.asarray([np.nan])], 0.2, 0.2)
        self.assertAlmostEqual(long, 1.2)
        self.assertAlmostEqual(short, -1.2)

    def test_disjoint_but_later_calibration_is_rejected(self):
        splits = {
            "val": {"start": "20250501", "end": "20250531"},
            "test": {"start": "20250601", "end": "20250630"},
        }
        require_prior_calibration(splits, "val", {"test"})
        with self.assertRaisesRegex(ValueError, "before"):
            require_prior_calibration(splits, "test", {"val"})

    def test_invalid_quote_cannot_open_or_flip(self):
        pred = np.asarray([1.0, -2.0, -2.0])
        valid = np.asarray([True, False, True])
        positions = generate_positions(pred, valid, 0.5, -0.5)
        np.testing.assert_array_equal(positions, [1, 1, -1])

    def test_invalid_terminal_quote_cannot_backdate_liquidation(self):
        pred = np.asarray([1.0, 0.0, -1.0])
        valid = np.asarray([True, True, False])
        positions = generate_positions(pred, valid, 0.5, -0.5)
        with self.assertRaisesRegex(ValueError, "terminal"):
            build_position_segments(positions, valid)

    def test_no_new_round_trip_is_opened_on_final_executable_row(self):
        pred = np.asarray([1.0, -1.0])
        valid = np.asarray([True, True])
        positions = generate_positions(pred, valid, 0.5, -0.5)
        self.assertEqual(build_position_segments(positions, valid), [(1, 0, 1)])

    def test_missing_signals_hold_position_until_an_executable_flip(self):
        pred = np.asarray([1.0, np.nan, np.nan, -1.0, np.nan])
        valid = np.ones(5, dtype=bool)
        positions = generate_positions(pred, valid, 0.5, -0.5)
        np.testing.assert_array_equal(positions, [1, 1, 1, -1, -1])
        self.assertEqual(build_position_segments(positions, valid), [(1, 0, 3), (-1, 3, 4)])


if __name__ == "__main__":
    unittest.main()
