"""Real Stage1–5 runs, independent references, and corrupted-artifact checks."""

from contextlib import contextmanager
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.check_synthetic_pipeline import ROOT, run_pipeline, run_scenarios
from scripts.synthetic_data import (
    artifact_path,
    clock_from_ms,
    create_fixture,
    ms_from_clock,
    read_json,
    read_jsonl,
    write_json,
)


@contextmanager
def restore_files(*paths):
    originals = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, content in originals.items():
            path.write_bytes(content)


def edit_csv(path, change):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    change(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


class SyntheticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lob-v5-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name) / "case with spaces"
        # These two full runs are reused by read-only checks and restored fault
        # injection cases. No artifacts or oracle values are mocked.
        cls.summary = run_scenarios(cls.root)
        cls.baseline = cls.root / "baseline"
        cls.config_path = cls.baseline / "synthetic.local.json"
        cls.cfg = read_json(cls.config_path)
        cls.stage4_path = artifact_path(cls.cfg, "stage4", "stage4_manifest_path")
        cls.report_path = artifact_path(cls.cfg, "stage5", "report_path")

    def test_complete_pipeline_matches_independent_labels_and_windows(self):
        self.assertEqual(self.summary["status"], "passed")
        baseline = self.summary["cases"]["baseline"]
        self.assertEqual(sum(p["rows"] for p in baseline["splits"].values()), 3072)
        self.assertGreaterEqual(baseline["splits"]["train"]["valid_labels"], 1000)
        self.assertGreater(baseline["splits"]["train"]["shards"], 1)
        for split in ("train", "val", "test"):
            part = baseline["splits"][split]
            self.assertGreater(part["windows"], 0)
            self.assertTrue(0 < part["last_batch_size"] < 64)
        for name in ("results", "predictions", "backtest"):
            self.assertFalse((self.baseline / name).exists())

    def test_held_out_changes_do_not_affect_training_statistics(self):
        baseline, changed = (
            self.summary["cases"][name] for name in ("baseline", "held_out_changed")
        )
        self.assertEqual(baseline["training_fingerprint"], changed["training_fingerprint"])
        self.assertEqual(baseline["train_quantiles"], changed["train_quantiles"])
        self.assertNotEqual(baseline["source_csv_sha256"], changed["source_csv_sha256"])
        self.assertNotEqual(baseline["numeric_artifact_sha256"], changed["numeric_artifact_sha256"])

    def test_same_horizon_name_cannot_reuse_raw_labels_after_definition_change(self):
        stats_path = artifact_path(self.cfg, "stage2", "label_stats_path")
        original_stats = stats_path.read_bytes()
        with restore_files(self.config_path):
            changed = read_json(self.config_path)
            horizon = changed["horizons"]["active_horizon_id"]
            changed["horizons"]["definitions"][horizon]["min_seconds"] += .1
            write_json(self.config_path, changed)
            result = subprocess.run(
                [sys.executable, "-m", "src.stage2_build_labels_clear", "--config-path",
                 str(self.config_path), "--only-scale", "1"], cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("raw label definition mismatch", result.stderr)
        self.assertEqual(stats_path.read_bytes(), original_stats)

    def test_stage4_and_healthcheck_reject_stale_target_provenance(self):
        stage3_path = artifact_path(self.cfg, "stage3", "packs_manifest_path")
        with restore_files(stage3_path, self.report_path):
            manifest = read_json(stage3_path)
            manifest["target_contract_sha256"] = "a" * 64
            write_json(stage3_path, manifest)
            result = run_pipeline(self.config_path, ["windows"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("target_contract_sha256", result.stderr)
            result = run_pipeline(self.config_path, ["healthcheck"])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(read_json(self.report_path)["consistency"]["target_contract_match"])

    def test_stage3_rejects_stale_final_label_provenance(self):
        index = read_jsonl(artifact_path(self.cfg, "stage2", "labels_index_path"))
        path = Path(index[0]["final_label_path"])
        # Preserve all pack outputs: Stage3 opens its output shards before reading sessions.
        pack_root = artifact_path(self.cfg, "stage3", "packs_manifest_path").parent
        packed_files = list(pack_root.rglob("*"))
        with restore_files(path, *[p for p in packed_files if p.is_file()]):
            with np.load(path) as labels:
                contents = {key: labels[key] for key in labels.files}
            contents["target_contract_sha256"] = np.asarray("a" * 64)
            np.savez(path, **contents)
            result = run_pipeline(self.config_path, ["pack"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("target_contract_sha256", result.stderr)

    def test_healthcheck_rejects_inconsistent_manifest_structure(self):
        for case in ("schema", "counts", "duplicate_block", "segment_boundary", "window_size"):
            with self.subTest(case=case), restore_files(self.stage4_path, self.report_path):
                manifest = read_json(self.stage4_path)
                if case == "schema":
                    manifest["factor_schema_path"] = str(
                        artifact_path(self.cfg, "stage1", "schema_path")
                    )
                elif case == "counts":
                    manifest["counts"]["train"]["total_end_positions"] += 1
                elif case == "window_size":
                    manifest["window_W"] -= 1
                else:
                    blocks = manifest["splits"]["train"]["blocks"]
                    if case == "duplicate_block":
                        blocks.append(dict(blocks[0]))
                        manifest["counts"]["train"]["total_blocks"] += 1
                        manifest["counts"]["train"]["total_end_positions"] += blocks[0]["end_len"]
                    else:
                        blocks[0]["seg_start_row"] += 1
                write_json(self.stage4_path, manifest)
                result = run_pipeline(self.config_path, ("healthcheck",))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                report = read_json(self.report_path)
                self.assertFalse(report["ok"])
                self.assertEqual(report["status"], "failed")
                self.assertNotIn("Traceback", result.stderr)

    def test_healthcheck_rejects_invalid_window_times(self):
        manifest = read_json(self.stage4_path)
        time_path = Path(manifest["splits"]["train"]["shards"]["0"]["t_sec"])
        for case, violation in (
            ("nan", "time_nonfinite"),
            ("reverse", "time_nonmonotonic"),
            ("gap", "inter_event_gap_exceeded"),
            ("missing", "time_missing_or_badshape"),
        ):
            with self.subTest(case=case), restore_files(time_path, self.report_path):
                times = np.load(time_path, allow_pickle=False)
                if case == "missing":
                    time_path.unlink()
                else:
                    if case == "nan":
                        times[:] = np.nan
                    elif case == "reverse":
                        times[:] = -np.arange(len(times), dtype=np.float64)
                    else:
                        times[:] = np.arange(len(times), dtype=np.float64)
                    np.save(time_path, times)
                result = run_pipeline(self.config_path, ("healthcheck",))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                report = read_json(self.report_path)
                self.assertFalse(report["ok"])
                self.assertGreater(report["splits"]["train"]["window_checks"][violation], 0)
                self.assertNotIn("Traceback", result.stderr)

    def test_diagnostic_healthcheck_still_reports_detected_failure(self):
        with restore_files(self.config_path, self.stage4_path, self.report_path):
            cfg = read_json(self.config_path)
            cfg["stage5"]["strict"] = False
            write_json(self.config_path, cfg)
            manifest = read_json(self.stage4_path)
            manifest["factor_schema_path"] = str(artifact_path(cfg, "stage1", "schema_path"))
            write_json(self.stage4_path, manifest)
            result = run_pipeline(self.config_path, ("healthcheck",))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(read_json(self.report_path)["ok"])

    def test_bad_timestamp_stops_pipeline_before_packing(self):
        cfg, config_path, sessions = create_fixture(self.root / "bad timestamp")

        def break_time(rows):
            rows[201][0] = clock_from_ms(ms_from_clock(rows[200][0]) - 1)

        edit_csv(Path(sessions[0]["path"]), break_time)
        result = run_pipeline(config_path)
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("Stage labels failed", result.stderr)
        self.assertIn("timestamp_not_monotonic_or_nan", result.stderr)
        self.assertFalse(artifact_path(cfg, "stage3", "packs_manifest_path").exists())

    def test_missing_factor_stops_pipeline_before_label_generation(self):
        cfg, config_path, sessions = create_fixture(self.root / "missing factor")

        def remove_column(rows):
            for row in rows:
                row.pop()

        edit_csv(Path(sessions[0]["path"]), remove_column)
        result = run_pipeline(config_path)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("Stage manifest failed", result.stderr)
        self.assertFalse(artifact_path(cfg, "stage2", "label_stats_path").exists())
        self.assertFalse(artifact_path(cfg, "stage3", "packs_manifest_path").exists())

    def test_legacy_raw_labels_cannot_receive_the_new_contract_by_rescaling(self):
        index = read_jsonl(artifact_path(self.cfg, "stage2", "labels_index_path"))
        raw_path = Path(index[0]["raw_label_path"])
        stats_path = artifact_path(self.cfg, "stage2", "label_stats_path")
        final_path = Path(index[0]["final_label_path"])
        original_stats, original_final = stats_path.read_bytes(), final_path.read_bytes()
        with restore_files(raw_path, stats_path, final_path):
            with np.load(raw_path, allow_pickle=False) as source:
                legacy = {
                    key: source[key] for key in source.files if key != "label_contract_version"
                }
            legacy["mid"] = np.round(legacy["mid"], 3).astype(np.float32)
            np.savez(raw_path, **legacy)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.stage2_build_labels_clear",
                    "--config-path",
                    str(self.config_path),
                    "--only-scale",
                    "1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
            self.assertIn("rebuild Stage2 from CSV", result.stderr)
            self.assertEqual(stats_path.read_bytes(), original_stats)
            self.assertEqual(final_path.read_bytes(), original_final)


if __name__ == "__main__":
    unittest.main()
