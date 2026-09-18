"""Real predictor/evaluator with a tiny synthetic checkpoint fixture.

This prepares its test contract explicitly and does NOT exercise Stage6.fit.
The temporary contract is not a research training result.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from scripts.check_synthetic_pipeline import ROOT, run_pipeline
from scripts.synthetic_data import create_fixture, artifact_path, read_json, write_json
from src.artifact_contract import sha256_file
from src.feature_preprocessing import PREPROCESSING_VERSION
from src.stage6_train_regression import build_transformer_lstm_regressor
from src.target_contract import load_target_binding
from src.window_loader import LoaderConfig, iter_batches_with_time


class PredictionCLI(unittest.TestCase):
    def test_prediction_generation_reuse_evaluation_and_stale_bundle_rejection(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        with tempfile.TemporaryDirectory() as temp:
            cfg, config_path, _ = create_fixture(Path(temp) / "fixture")
            cfg["train"]["run_name"] = cfg["predict"]["run_name"] = cfg["backtest"]["run_name"] = "unit_checkpoint_fixture"
            cfg["model"].update(d_model=8, num_heads=2, num_layers=1, ff_dim=12,
                                lstm_units=6, head_hidden=5, dropout=0.)
            cfg["predict"]["batch_size"] = 64
            write_json(config_path, cfg)
            prepared = run_pipeline(config_path)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            binding = load_target_binding(cfg)
            stats = read_json(artifact_path(cfg, "stage3", "input_stats_path"))
            st4_path = artifact_path(cfg, "stage4", "stage4_manifest_path")
            st4 = read_json(st4_path)
            schema_path = Path(st4["factor_schema_path"])
            schema = read_json(schema_path)
            model = build_transformer_lstm_regressor(8, 3, stats["mean"], stats["std"], cfg["model"])
            model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse", jit_compile=False)
            batch = next(iter_batches_with_time(LoaderConfig(
                project_root=cfg["project"]["project_root"], stage4_manifest_path=str(st4_path),
                split="train", batch_size=8, window_W=8, num_factors=3, shuffle_blocks=False,
            )))
            x, t, y, weight = batch
            model.train_on_batch([x, t], y[:, None], sample_weight=weight)
            model_path = artifact_path(cfg, "predict", "model_path")
            weights_path = artifact_path(cfg, "predict", "best_weights_path")
            model_path.parent.mkdir(parents=True)
            model.save(model_path)
            model.save_weights(weights_path)
            contract = {
                **binding, "contract_version": 2, "status": "trained",
                "fixture_only": True, "run_name": cfg["train"]["run_name"],
                "horizon_id": cfg["horizons"]["active_horizon_id"], "window_W": 8, "num_factors": 3,
                "preprocessing_version": PREPROCESSING_VERSION,
                "factor_cols": schema["factor_cols"], "factor_schema_sha256": sha256_file(str(schema_path)),
                "preprocessing": schema["preprocessing"], "norm_map_sha256": None,
                "window_constraints": st4["window_constraints"],
                "model_sha256": sha256_file(str(model_path)),
                "best_weights_sha256": sha256_file(str(weights_path)),
            }
            write_json(artifact_path(cfg, "predict", "contract_path"), contract)

            def invoke(module, success=True):
                env = dict(os.environ, TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
                result = subprocess.run([sys.executable, "-m", f"src.{module}", "--config-path", str(config_path)],
                                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            invoke("batch_predict_raw")
            root = artifact_path(cfg, "predict", "output_root")
            first = read_json(root / "prediction_summary.json")
            self.assertEqual(first["successful_sessions"], 4)
            paths = sorted(root.glob("*/*_pred.npz"))
            original_hashes = {str(path): sha256_file(str(path)) for path in paths}
            cfg["predict"]["overwrite"] = False
            write_json(config_path, cfg)
            invoke("batch_predict_raw")
            self.assertEqual(read_json(root / "prediction_summary.json")["skipped_existing_sessions"], 4)
            self.assertEqual(original_hashes, {str(path): sha256_file(str(path)) for path in paths})
            invoke("accuracy")
            self.assertEqual(read_json(artifact_path(cfg, "accuracy_eval", "report_path"))["status"], "complete")
            with np.load(paths[0]) as bundle:
                contents = {key: bundle[key] for key in bundle.files}
            metadata = json.loads(str(contents["metadata_json"].item()))
            metadata["target_contract_sha256"] = "f" * 64
            contents["metadata_json"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(paths[0], **contents)
            failed = invoke("batch_predict_raw", success=False)
            self.assertIn("target_contract_sha256", failed.stderr)
