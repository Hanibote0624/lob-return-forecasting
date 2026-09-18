"""Exercise the real evaluation/backtest entry points with synthetic artifacts."""

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
from src.prediction_io import load_prediction_bundle, save_prediction_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]


class EvaluationCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = np.arange(2, 7, dtype=np.int64)
        self.times = 34200.0 + np.arange(8, dtype=np.float64) * 0.01
        self.config = {
            "project": {"project_root": str(self.root)},
            "horizons": {"active_horizon_id": "synthetic", "definitions": {"synthetic": {"min_seconds": 0.01, "max_seconds": 0.02}}},
            "label": {"fixed_scale": 200., "clip_mode": "none"},
            "train": {"run_name": "fixture"},
            "predict": {"run_name": "fixture"},
            "data": {
                "splits": {
                    "val": {"start": "20250501", "end": "20250531"},
                    "test": {"start": "20250601", "end": "20250630"},
                },
                "required_columns": {
                    "timestamp": "timestamp", "mid_price": {"bid1": "bid", "ask1": "ask", "fallback_last": "last"},
                },
            },
            "stage1": {"manifest_ok_path": "sessions.jsonl"},
            "stage2": {"labels_index_path": "labels.jsonl", "label_stats_path": "stats.json"},
            "accuracy_eval": {
                "pred_root": "predictions", "report_path": "accuracy/report.json",
                "detail_csv_path": "accuracy/detail.csv", "splits": ["test"],
                "strict_complete": True, "topk": {"drops": [0.2]},
            },
            "backtest": {
                "splits": ["test"], "strict_complete": True,
                "thresholds": {"mode": "calibration_split", "source_split": "val"},
                "strategy": {"topk_long": 0.2, "topk_short": 0.2},
                "cost": {"commission_rate": 0.0, "stamp_tax_rate": 0.0},
                "output": {"report_path": "backtest/report.json", "trade_detail_csv": "backtest/trades.csv",
                           "curve_csv": "backtest/curve.csv"},
            },
        }
        stats = {"scale": 200., "clip_mode": "none", "clip_abs_scaled": None,
                 "train_only_quantiles": {"q90_abs_r_raw": .01, "q99_abs_r_raw": .02},
                 "label_contract": label_definition(self.config)}
        stats["target_contract"] = target_spec(self.config, stats)
        stats["target_contract_sha256"] = json_sha256(stats["target_contract"])
        stats_path = self.root / "stats.json"
        stats_path.write_text(json.dumps(stats))
        self.binding = binding_from_stats(self.config, stats, stats_path)
        contract_path = self.root / "results/fixture/model_contract.json"
        contract_path.parent.mkdir(parents=True)
        contract_path.write_text(json.dumps({**self.binding, "contract_version": 2,
            "status": "trained", "run_name": "fixture", "horizon_id": "synthetic"}))
        self.contract_path = contract_path
        manifests, labels = [], []
        self.paths = {}
        for split, date, pred in [
            ("val", "20250512", [-2, -1, 0, 1, 2]),
            ("test", "20250612", [2, 0, -2, 0, 2]),
        ]:
            raw = self.root / f"{date}.csv"
            pd.DataFrame({
                "timestamp": 93000000 + np.arange(8) * 10,
                "bid": 10.0 + np.arange(8) * 0.01,
                "ask": 10.02 + np.arange(8) * 0.01,
            }).to_csv(raw, index=False)
            bundle_path = self.root / "predictions" / "AAA" / f"{date}_1_pred.npz"
            save_prediction_bundle(str(bundle_path), np.asarray(pred), self.rows, self.times[self.rows], {
                "stock_code": "AAA", "date": date, "session": 1, "split": split,
                "run_name": "fixture", "horizon_id": "synthetic", "source_row_count": 8,
                "source_sha256": sha256_file(str(raw)), "model_contract_sha256": sha256_file(str(contract_path)),
                **self.binding,
            })
            label_path = self.root / f"{date}.npz"
            valid = np.ones(8, dtype=np.uint8)
            valid[6] = 0  # A finite label must still be excluded when is_valid=0.
            np.savez(label_path, label_contract_version=np.array(2), mid=np.ones(8, dtype=np.float64),
                     label_definition_sha256=np.asarray(json_sha256(label_definition(self.config))),
                     target_contract_sha256=np.asarray(self.binding["target_contract_sha256"]),
                     label_stats_sha256=np.asarray(self.binding["label_stats_sha256"]),
                     source_sha256=np.asarray(sha256_file(str(raw))), r_raw=np.arange(8, dtype=np.float32) * 0.001,
                     t_sec=self.times, is_valid=valid)
            manifests.append({"ok": True, "stock_code": "AAA", "date": date, "session": 1, "path": str(raw)})
            labels.append({"stock_code": "AAA", "date": date, "session": 1,
                           "split": split, "final_label_path": str(label_path)})
            self.paths[split] = {"raw": raw, "bundle": bundle_path, "label": label_path}
        for name, items in [("sessions.jsonl", manifests), ("labels.jsonl", labels)]:
            (self.root / name).write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def run_cli(self, module, success=True):
        result = subprocess.run(
            [sys.executable, "-m", f"src.{module}", "--config-path", str(self.config_path)],
            cwd=REPO_ROOT, text=True, capture_output=True, timeout=30,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def rewrite_bundle(self, split, *, pred=None, metadata=None):
        path = str(self.paths[split]["bundle"])
        bundle = load_prediction_bundle(path)
        bundle["metadata"].update(metadata or {})
        save_prediction_bundle(path, bundle["pred"] if pred is None else pred,
                               bundle["end_row"], bundle["t_sec"], bundle["metadata"])

    def test_evaluation_and_threshold_calibration_use_the_correct_populations(self):
        self.run_cli("accuracy")
        report = json.loads((self.root / "accuracy/report.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["metrics"]["valid_prediction_rows"], 4)
        self.run_cli("backtest_engine_flip")
        first = json.loads((self.root / "backtest/report.json").read_text())
        self.assertAlmostEqual(first["thresholds"]["long"], 1.2)
        self.assertAlmostEqual(first["thresholds"]["short"], -1.2)
        trades = pd.read_csv(self.root / "backtest/trades.csv")
        self.assertEqual(trades.iloc[0]["start_row"], 2)
        # Session liquidation uses raw row 7 even though predictions end at row 6.
        self.assertEqual(trades.iloc[-1]["end_row"], 7)
        self.rewrite_bundle("test", pred=np.asarray([200, 100, -200, -100, 200]))
        self.run_cli("backtest_engine_flip")
        second = json.loads((self.root / "backtest/report.json").read_text())
        self.assertEqual(first["thresholds"], second["thresholds"])

    def test_changed_scale_is_rejected_even_with_same_horizon_name(self):
        self.config["label"]["fixed_scale"] = 100.
        self.config_path.write_text(json.dumps(self.config))
        for module in ("accuracy", "backtest_engine_flip"):
            result = self.run_cli(module, success=False)
            self.assertIn("scale differs", result.stderr)

    def test_prediction_with_stale_target_fails_both_entry_points(self):
        self.rewrite_bundle("test", metadata={"target_contract_sha256": "b" * 64})
        for module in ("accuracy", "backtest_engine_flip"):
            result = self.run_cli(module, success=False)
            self.assertIn("target_contract_sha256", result.stderr)

    def test_legacy_model_contract_is_rejected(self):
        contract = json.loads(self.contract_path.read_text())
        contract["contract_version"] = 1
        self.contract_path.write_text(json.dumps(contract))
        self.run_cli("accuracy", success=False)
        self.run_cli("backtest_engine_flip", success=False)

    def test_labels_from_another_scale_are_rejected(self):
        path = self.paths["test"]["label"]
        with np.load(path) as labels:
            data = {key: labels[key] for key in labels.files}
        data["label_stats_sha256"] = np.asarray("b" * 64)
        np.savez(path, **data)
        result = self.run_cli("accuracy", success=False)
        self.assertIn("label_stats_sha256", result.stderr)

    def test_missing_expected_session_fails_both_entry_points(self):
        self.paths["test"]["bundle"].unlink()
        self.run_cli("accuracy", success=False)
        self.run_cli("backtest_engine_flip", success=False)

    def test_wrong_run_metadata_fails_both_entry_points(self):
        self.rewrite_bundle("test", metadata={"run_name": "different_run"})
        self.run_cli("accuracy", success=False)
        self.run_cli("backtest_engine_flip", success=False)

    def test_changed_raw_prices_fail_the_source_fingerprint(self):
        raw = self.paths["test"]["raw"]
        frame = pd.read_csv(raw)
        frame.loc[2, "bid"] = 5.0
        frame.to_csv(raw, index=False)
        result = self.run_cli("backtest_engine_flip", success=False)
        self.assertIn("fingerprint", result.stderr)

    def test_invalid_terminal_quote_cannot_produce_a_successful_report(self):
        raw = self.paths["test"]["raw"]
        frame = pd.read_csv(raw)
        frame.loc[7, "ask"] = 0.0
        frame.to_csv(raw, index=False)
        self.rewrite_bundle("test", metadata={"source_sha256": sha256_file(str(raw))})
        result = self.run_cli("backtest_engine_flip", success=False)
        self.assertIn("terminal", result.stderr)
        report = json.loads((self.root / "backtest/report.json").read_text())
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
