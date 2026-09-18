import unittest

import numpy as np

from src.accuracy import align_prediction_to_labels, find_events


class EvaluationAlignmentTests(unittest.TestCase):
    def test_explicit_endpoint_rows_select_labels_without_truncation(self):
        label_time = np.arange(8, dtype=np.float64) * 0.01
        bundle = {
            "pred": np.asarray([0.2, -0.1], dtype=np.float32),
            "end_row": np.asarray([3, 7], dtype=np.int64),
            "t_sec": label_time[[3, 7]],
        }
        prediction, truth, mask, rows = align_prediction_to_labels(
            bundle,
            true_full=np.arange(8, dtype=np.float32),
            valid_full=np.asarray([True] * 7 + [False]),
            label_time=label_time,
        )
        np.testing.assert_array_equal(rows, [3, 7])
        np.testing.assert_array_equal(truth, [3.0, 7.0])
        np.testing.assert_array_equal(mask, [True, False])
        np.testing.assert_allclose(prediction, [0.2, -0.1])

    def test_timestamp_mismatch_fails_instead_of_silent_alignment(self):
        bundle = {
            "pred": np.asarray([0.2]),
            "end_row": np.asarray([2]),
            "t_sec": np.asarray([9.0]),
        }
        with self.assertRaisesRegex(ValueError, "timestamps"):
            align_prediction_to_labels(
                bundle,
                true_full=np.arange(4),
                valid_full=np.ones(4, dtype=bool),
                label_time=np.arange(4, dtype=np.float64),
            )

    def test_signal_events_break_across_missing_original_rows(self):
        mask = np.asarray([True, True, True, True])
        rows = np.asarray([10, 11, 20, 21])
        self.assertEqual(find_events(mask, rows), [(0, 2), (2, 2)])


if __name__ == "__main__":
    unittest.main()
