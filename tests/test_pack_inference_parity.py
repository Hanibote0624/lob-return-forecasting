"""Verify Stage3/4 CLI wiring against the preprocessing used by inference."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.artifact_contract import sha256_file
from src.target_contract import label_definition, target_spec, json_sha256, binding_from_stats
from src.feature_preprocessing import preprocess_session_features, require_preprocessing_spec
from src.prediction_io import iter_window_batches
from src.window_loader import LoaderConfig, iter_batches_with_time


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackingParityTests(unittest.TestCase):
    def test_packed_windows_match_raw_preprocessing_and_stats_use_only_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads((REPO_ROOT / "config/gp_lit_regression_v6_gpmain_64.example.json").read_text())
            cfg["project"]["project_root"] = str(root)
            cfg["features"].update(window_W=3, num_factors=2)
            cfg["data"]["stocks"] = [{"stock_code": "AAA", "float_shares": 100.0}]
            cfg["stage3"]["row_shard_rows"] = 16
            cfg["stage3"]["feature_norm_map_path"] = "norm.json"
            cfg["stage4"]["block_size_ends"] = 2  # smaller than loader batch size
            horizon = cfg["horizons"]["active_horizon_id"]
            factors = ["f_volume", "f_mcap"]
            norm = {
                "groups": {"volume_norm": ["f_volume"], "mcap_norm": ["f_mcap"]},
                "meta": {"ema_span": 2, "dv_floor": 1.0, "mcap_floor": 1.0},
            }

            def path(section, key):
                return root / cfg[section][key].format(horizon_id=horizon)

            def write_json(target, content):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(content), encoding="utf-8")

            write_json(root / "norm.json", norm)
            write_json(path("stage1", "schema_path"), {"factor_cols": factors, "num_factors": 2})
            label_stats = {
                "scale": 200., "clip_mode": "q99_abs_train_only", "clip_abs_scaled": 2.,
                "train_only_quantiles": {"q90_abs_r_raw": 0.01, "q99_abs_r_raw": 0.01},
                "label_contract": label_definition(cfg),
            }
            label_stats["target_contract"] = target_spec(cfg, label_stats)
            label_stats["target_contract_sha256"] = json_sha256(label_stats["target_contract"])
            write_json(path("stage2", "label_stats_path"), label_stats)
            binding = binding_from_stats(cfg, label_stats, path("stage2", "label_stats_path"))
            times = 34200.0 + np.arange(8, dtype=np.float64) * 0.01
            manifests, index, expected = [], [], {}
            for split, date, offset in [("train", "20250412", 0), ("val", "20250512", 1000), ("test", "20250612", 2000)]:
                frame = pd.DataFrame({
                    "timestamp": 93000000 + np.arange(8) * 10,
                    "f_volume": [np.nan, 2.0, np.nan, 6.0, 8.0, 10.0, 12.0, 14.0],
                    "f_mcap": 1000.0 + offset + np.arange(8) * 100.0,
                    "acc_volume": [100, 101, 103, 106, 110, 115, 121, 128],
                    "bid": np.full(8, 9.9), "ask": np.full(8, 10.1), "last": np.full(8, 10.0),
                })
                raw_path = root / f"{date}.csv"
                frame.to_csv(raw_path, index=False)
                expected[split] = preprocess_session_features(
                    frame, factors, cfg, volume_norm_factors=["f_volume"],
                    mcap_norm_factors=["f_mcap"], norm_meta=norm["meta"], stock_code="AAA",
                )
                label_path = root / f"{date}.npz"
                np.savez(label_path, label_contract_version=np.array(2), mid=np.ones(8, dtype=np.float64),
                         label_definition_sha256=np.asarray(json_sha256(label_definition(cfg))),
                         source_sha256=np.asarray(sha256_file(str(raw_path))),
                         target_contract_sha256=np.asarray(binding["target_contract_sha256"]),
                         label_stats_sha256=np.asarray(binding["label_stats_sha256"]), r_raw=np.arange(8) * 0.001, r_scaled=np.arange(8) * 0.2,
                         t_sec=times, is_valid=np.ones(8, dtype=np.uint8))
                identity = {"stock_code": "AAA", "date": date, "session": 1, "split": split}
                manifests.append(dict(identity, ok=True, path=str(raw_path)))
                index.append(dict(identity, final_label_path=str(label_path)))
            for target, items in [(path("stage1", "manifest_ok_path"), manifests),
                                  (path("stage2", "labels_index_path"), index)]:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
            write_json(root / "config.json", cfg)
            for script in ["stage3_pack_rows", "stage4_build_stage4_manifest"]:
                result = subprocess.run(
                    [sys.executable, "-m", f"src.{script}", "--config-path", str(root / "config.json")],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            stats = json.loads(path("stage3", "input_stats_path").read_text())
            np.testing.assert_allclose(stats["mean"], expected["train"].mean(axis=0), rtol=1e-6)
            self.assertEqual(stats["train_row_count"], 8)
            schema_path = path("stage3", "final_schema_path")
            self.assertEqual(stats["factor_schema_sha256"], sha256_file(str(schema_path)))
            schema = json.loads(schema_path.read_text())
            require_preprocessing_spec(cfg, factors, norm, schema["preprocessing"])
            self.assertEqual(schema["norm_map_sha256"], sha256_file(str(root / "norm.json")))

            for split in ["train", "val", "test"]:
                loaded = list(iter_batches_with_time(LoaderConfig(
                    project_root=str(root), stage4_manifest_path=str(path("stage4", "stage4_manifest_path")),
                    split=split, batch_size=4, window_W=3, num_factors=2,
                    shuffle_blocks=False, drop_remainder=False,
                )))
                inferred = list(iter_window_batches(expected[split], times, 3, 4))
                self.assertEqual([len(batch[2]) for batch in loaded], [4, 2])
                for packed, raw in zip(loaded, inferred):
                    np.testing.assert_array_equal(packed[0], raw[0])
                    np.testing.assert_array_equal(packed[1], raw[1])
                np.testing.assert_allclose(np.concatenate([batch[2] for batch in loaded]), np.arange(2, 8) * 0.2)


if __name__ == "__main__":
    unittest.main()
