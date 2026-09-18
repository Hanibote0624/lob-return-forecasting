import json
import os
import tempfile
import unittest

import numpy as np

from src.window_loader import (
    LoaderConfig,
    count_batches_from_blocks,
    count_split_batches,
    iter_batches_prefetch,
    iter_batches_with_time,
)


class WindowLoaderTests(unittest.TestCase):
    def _build_fixture(self, root: str) -> str:
        rows = 20
        factors = 2
        np.save(os.path.join(root, "X.npy"), np.arange(rows * factors, dtype=np.float32).reshape(rows, factors))
        np.save(os.path.join(root, "y.npy"), np.arange(rows, dtype=np.float32))
        np.save(os.path.join(root, "w.npy"), np.ones(rows, dtype=np.float32))
        # Large absolute seconds expose float32 cancellation if subtraction is
        # performed after a premature cast. Event spacing is exactly 10 ms.
        np.save(os.path.join(root, "t.npy"), 50_000.0 + np.arange(rows, dtype=np.float64) * 0.01)

        manifest = {
            "window_W": 4,
            "num_factors": factors,
            "splits": {
                "val": {
                    "shards": {
                        "0": {
                            "X": "X.npy",
                            "y": "y.npy",
                            "w": "w.npy",
                            "t_sec": "t.npy",
                        }
                    },
                    "blocks": [
                        {
                            "shard_id": 0,
                            "end_start": 3,
                            "end_len": 9,
                            "seg_start_row": 0,
                            "seg_length": rows,
                        }
                    ],
                }
            },
        }
        path = os.path.join(root, "stage4.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        return path

    def _config(self, root: str, manifest: str, drop_remainder: bool) -> LoaderConfig:
        return LoaderConfig(
            project_root=root,
            stage4_manifest_path=manifest,
            split="val",
            batch_size=4,
            window_W=4,
            num_factors=2,
            drop_remainder=drop_remainder,
        )

    def test_eval_tail_is_emitted_once_and_time_precision_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._build_fixture(root)
            batches = list(iter_batches_with_time(self._config(root, manifest, False)))

        self.assertEqual([len(batch[2]) for batch in batches], [4, 4, 1])
        np.testing.assert_array_equal(
            np.concatenate([batch[2] for batch in batches]),
            np.arange(3, 12, dtype=np.float32),
        )
        for _, rel_t, _, _ in batches:
            np.testing.assert_allclose(rel_t[:, 0], 0.0, atol=1e-7)
            np.testing.assert_allclose(np.diff(rel_t, axis=1), 0.01, atol=2e-6)

    def test_training_can_drop_only_the_short_tail(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._build_fixture(root)
            batches = list(iter_batches_with_time(self._config(root, manifest, True)))
            with open(manifest, "r", encoding="utf-8") as handle:
                manifest_obj = json.load(handle)

        self.assertEqual(len(batches), 2)
        self.assertEqual(sum(len(batch[2]) for batch in batches), 8)
        self.assertEqual(count_split_batches(manifest_obj, "val", 4, True), 2)
        self.assertEqual(count_split_batches(manifest_obj, "val", 4, False), 3)

    def test_zero_batch_count_is_not_hidden(self):
        blocks = [{"end_len": 3}]
        self.assertEqual(count_batches_from_blocks(blocks, 4, True), 0)
        self.assertEqual(count_batches_from_blocks(blocks, 4, False), 1)

    def test_short_stage4_blocks_are_batched_across_artificial_boundaries(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = self._build_fixture(root)
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["splits"]["val"]["blocks"] = [
                {
                    "shard_id": 0,
                    "end_start": 3,
                    "end_len": 3,
                    "seg_start_row": 0,
                    "seg_length": 20,
                },
                {
                    "shard_id": 0,
                    "end_start": 6,
                    "end_len": 3,
                    "seg_start_row": 0,
                    "seg_length": 20,
                },
            ]
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            batches = list(iter_batches_with_time(self._config(root, manifest_path, True)))

        self.assertEqual([len(batch[2]) for batch in batches], [4])
        np.testing.assert_array_equal(batches[0][2], [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(count_batches_from_blocks(manifest["splits"]["val"]["blocks"], 4, True), 1)

    def test_missing_shard_raises(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._build_fixture(root)
            os.remove(os.path.join(root, "X.npy"))
            with self.assertRaises(FileNotFoundError):
                next(iter_batches_with_time(self._config(root, manifest, False)))

    def test_prefetch_propagates_producer_error(self):
        def broken_iterator():
            yield "first"
            raise ValueError("synthetic producer failure")

        iterator = iter_batches_prefetch(broken_iterator(), prefetch=1)
        self.assertEqual(next(iterator), "first")
        with self.assertRaisesRegex(ValueError, "synthetic producer failure"):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
