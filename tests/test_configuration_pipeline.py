"""Configuration and orchestration failures must stop before expensive stages."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.configuration import ConfigError, REPO_ROOT, load_config, validate_config
from src.pipeline import BY_NAME, execute_plan, main, select_stages
from src.stage1_make_manifest_multi import get_stock_specs


EXAMPLE = REPO_ROOT / "config/gp_lit_regression_v6_gpmain_64.example.json"


class ConfigurationTests(unittest.TestCase):
    def test_public_example_and_paths_are_independent_of_caller_directory(self):
        original = EXAMPLE.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            old = Path.cwd()
            try:
                os.chdir(temp)
                cfg = load_config(EXAMPLE)
            finally:
                os.chdir(old)
        validate_config(cfg)
        self.assertEqual(cfg["project"]["project_root"], str(REPO_ROOT))
        self.assertEqual(
            cfg["data"]["stocks"][0]["raw_root"], str(REPO_ROOT / "local/raw_data/SAMPLE")
        )
        self.assertEqual(EXAMPLE.read_bytes(), original)

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for text in (
                '{"project":{},"project":{}}',
                '{"project":{},"x":NaN}',
                '{"project":{},"x":1e999}',
            ):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ConfigError):
                    load_config(path)

    def test_overlap_invalid_calendar_date_and_unknown_horizon_are_rejected(self):
        for mode in ("overlap", "calendar", "horizon"):
            cfg = load_config(EXAMPLE)
            if mode == "overlap":
                cfg["data"]["splits"]["val"]["start"] = "20250401"
            elif mode == "calendar":
                cfg["data"]["splits"]["train"]["start"] = "20250230"
            else:
                cfg["horizons"]["active_horizon_id"] = "unknown"
            with self.subTest(mode=mode), self.assertRaises(ConfigError):
                validate_config(cfg)

    def test_unknown_path_placeholder_and_wrong_run_are_rejected(self):
        for field, value in (
            ("model_path", "results/{rnu_name}/final.keras"),
            ("run_name", "different_run"),
            ("model_path", "/other/model.keras"),
        ):
            cfg = load_config(EXAMPLE)
            cfg["predict"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ConfigError):
                validate_config(cfg)

    def test_invalid_types_fail_as_configuration_errors(self):
        for section, field, value in (
            ("train", "global_batch", True),
            ("predict", "splits", None),
            ("data", "splits", []),
            ("model", "num_heads", 0),
        ):
            cfg = load_config(EXAMPLE)
            cfg[section][field] = value
            with self.subTest(field=field), self.assertRaises(ConfigError):
                validate_config(cfg)

    def test_stage0_enabled_selects_enriched_input_and_legacy_typo_fails(self):
        cfg = load_config(EXAMPLE)
        cfg["stage0"]["enabled"] = True
        self.assertEqual(get_stock_specs(cfg)[0].out_root, cfg["data"]["stocks"][0]["out_root"])
        cfg["stage0"]["enable"] = True
        with self.assertRaisesRegex(ConfigError, "legacy typo"):
            validate_config(cfg)

    def test_optional_switches_reject_truthy_strings_and_non_booleans(self):
        for section, field, value in (
            ("stage0", "enabled", "false"),
            ("stage3", "write_t_sec", "false"),
            ("stage3", "strict_label_integrity", 0),
            ("stage3", "compute_input_stats_train", None),
            ("train", "jit_compile", "false"),
            ("predict", "use_best_weights", "false"),
        ):
            cfg = load_config(EXAMPLE)
            cfg[section][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "must be bool"):
                validate_config(cfg)

    def test_training_loader_requires_timestamps_even_without_time_constraints(self):
        cfg = load_config(EXAMPLE)
        cfg["stage3"]["write_t_sec"] = False
        cfg["model"]["use_time_aware_pos"] = False
        cfg["data"]["session_rules"]["max_history_span_seconds"] = 0
        cfg["data"]["session_rules"]["max_inter_event_gap_seconds"] = 0
        with self.assertRaisesRegex(ConfigError, "write_t_sec must be true"):
            validate_config(cfg)

    def test_enriched_input_requires_output_root_for_every_stock(self):
        cfg = load_config(EXAMPLE)
        cfg["stage0"]["enabled"] = True
        cfg["data"]["stocks"].append({"stock_code": "OTHER", "raw_root": "unused"})
        with self.assertRaisesRegex(ConfigError, "out_root for every stock"):
            validate_config(cfg)

    def test_unsupported_or_ambiguous_training_settings_are_rejected(self):
        for keys, value, message in (
            (("train", "optimizer", "name"), "sgd", "optimizer.name"),
            (("train", "optimizer", "eps"), 1e-8, "optimizer.eps is unused"),
            (("train", "optimizer", "beta1"), 1.0, "beta1 must be less than 1"),
            (("train", "optimizer", "epsilon"), False, "epsilon must be"),
            (("label", "clip_mode"), "unsupported", "label.clip_mode"),
            (("sample_weight", "mode"), "uniform", "sample_weight.mode"),
            (("sample_weight", "min_weight"), 9, "must not exceed"),
            (("sample_weight", "alpha"), -1, "alpha must be"),
        ):
            cfg = load_config(EXAMPLE)
            obj = cfg
            for key in keys[:-1]:
                obj = obj[key]
            obj[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaisesRegex(ConfigError, message):
                validate_config(cfg)

    def test_independent_outputs_cannot_overwrite_each_other(self):
        for left, right in (
            (("stage3", "input_stats_path"), ("stage2", "label_stats_path")),
            (("stage3", "final_schema_path"), ("stage1", "schema_path")),
            (("stage1", "manifest_ok_path"), ("stage1", "manifest_all_path")),
            (("accuracy_eval", "report_path"), ("accuracy_eval", "detail_csv_path")),
            (("stage2", "labels_raw_dir"), ("stage2", "labels_final_dir")),
        ):
            cfg = load_config(EXAMPLE)
            cfg[left[0]][left[1]] = cfg[right[0]][right[1]]
            with self.subTest(left=left), self.assertRaisesRegex(ConfigError, "must be distinct"):
                validate_config(cfg)

    def test_malformed_thresholds_raise_configuration_errors(self):
        for source in ([], {}, None, True, "unknown"):
            cfg = load_config(EXAMPLE)
            cfg["backtest"]["thresholds"]["source_split"] = source
            with self.subTest(source=source), self.assertRaisesRegex(ConfigError, "source_split"):
                validate_config(cfg)
        cfg = load_config(EXAMPLE)
        cfg["backtest"]["thresholds"] = {"mode": "fixed", "short": False, "long": True}
        with self.assertRaisesRegex(ConfigError, "finite short"):
            validate_config(cfg)

    def test_supported_disabled_options_and_reader_path_aliases_remain_valid(self):
        cfg = load_config(EXAMPLE)
        cfg["stage3"].update(
            write_y_raw=False, write_is_valid=False, compute_input_stats_train=False
        )
        cfg["train"]["optimizer"] = {}
        cfg["sample_weight"].update(alpha=0, clip_max=0, min_weight=1, max_weight=1)
        cfg["backtest"]["thresholds"] = {"mode": "fixed", "short": 0, "long": 0}
        for mode in ("none", "off", "disabled", "Q99_ABS_TRAIN_ONLY"):
            cfg["label"]["clip_mode"] = mode
            validate_config(cfg)
        self.assertEqual(cfg["predict"]["output_root"], cfg["accuracy_eval"]["pred_root"])

    def test_stage1_entry_point_uses_resolved_stock_root_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads(EXAMPLE.read_text())
            cfg["project"]["project_root"] = str(root / "work")
            cfg["data"]["stocks"][0]["raw_root"] = "raw"
            cfg["data"]["stocks"][0]["stock_code"] = "000000"
            cfg["data"]["required_columns"]["factors"]["count"] = 2
            cfg["features"]["num_factors"] = 2
            cfg["stage1"]["num_workers"] = 1
            raw = root / "work/raw/20250401/000000_20250401_1.csv"
            raw.parent.mkdir(parents=True)
            raw.write_text(
                "timestamp,bid,ask,last,gpmain_0,gpmain_1\n93000000,10,10.1,10,1,2\n93000010,10,10.1,10,2,3\n",
                encoding="utf-8",
            )
            config_path = root / "settings.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "src/stage1_make_manifest_multi.py"),
                    "--config-path",
                    str(config_path),
                    "--strict",
                    "1",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = root / "work" / cfg["stage1"]["manifest_ok_path"]
            records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            self.assertEqual(len(records), 1)
            self.assertEqual(Path(records[0]["path"]), raw)


class PipelineTests(unittest.TestCase):
    def test_invalid_config_exits_before_gpu_preflight_or_artifact_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "settings.json"
            for failure in ("time", "optimizer", "calibration", "collision"):
                cfg = json.loads(EXAMPLE.read_text())
                cfg["project"]["project_root"] = str(root / "no_outputs")
                if failure == "time":
                    cfg["stage3"]["write_t_sec"] = False
                elif failure == "optimizer":
                    cfg["train"]["optimizer"]["name"] = "sgd"
                elif failure == "calibration":
                    cfg["backtest"]["thresholds"]["source_split"] = []
                else:
                    cfg["stage3"]["input_stats_path"] = cfg["stage2"]["label_stats_path"]
                config_path.write_text(json.dumps(cfg), encoding="utf-8")
                result = subprocess.run(
                    [
                        sys.executable, "-S", str(REPO_ROOT / "scripts/run_pipeline.py"),
                        "--config-path", str(config_path),
                    ],
                    cwd=root, capture_output=True, text=True, timeout=10,
                )
                with self.subTest(failure=failure):
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("Configuration error:", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertNotIn("missing dependency", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertFalse((root / "no_outputs").exists())

    @unittest.skipUnless(
        shutil.which("bash"), "Bash wrapper is only checked where Bash is available"
    )
    def test_shell_wrapper_preserves_relative_config_path_and_rejects_scale_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads(EXAMPLE.read_text())
            cfg["project"]["project_root"] = str(root / "outputs")
            config = root / "settings with spaces.json"
            config.write_text(json.dumps(cfg), encoding="utf-8")
            env = dict(
                os.environ,
                PYTHON_BIN=sys.executable,
                RUN_BACKTEST="0",
                RUN_SCALE_AUDIT="0",
                RUN_VISUALIZATION="0",
            )
            env.pop("LABEL_SCALE", None)
            command = [
                shutil.which("bash"),
                str(REPO_ROOT / "run_all.sh"),
                config.name,
                "--dry-run",
                "--stages",
                "evaluate",
            ]
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [item["name"] for item in json.loads(result.stdout)["stages"]], ["evaluate"]
            )
            self.assertFalse((root / "outputs").exists())
            env["LABEL_SCALE"] = "200"
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("label.fixed_scale", result.stderr)

    def test_dry_run_needs_no_site_packages_and_preserves_configured_label_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads(EXAMPLE.read_text())
            cfg["project"]["project_root"] = str(root / "no_outputs")
            cfg["label"]["fixed_scale"] = 37.0
            path = root / "config with spaces.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(REPO_ROOT / "scripts/run_pipeline.py"),
                    "--config-path",
                    path.name,
                    "--dry-run",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["label_scale"], 37.0)
            self.assertEqual(
                [stage["name"] for stage in plan["stages"]],
                [
                    "manifest",
                    "labels",
                    "pack",
                    "windows",
                    "healthcheck",
                    "train",
                    "predict",
                    "evaluate",
                ],
            )
            self.assertNotIn("--scale", plan["stages"][1]["argv"])
            self.assertFalse((root / "no_outputs").exists())

    def test_optional_scale_audit_requires_train_only_map_with_multiple_stocks(self):
        cfg = load_config(EXAMPLE)
        with self.assertRaisesRegex(ConfigError, "multi-stock"):
            select_stages(cfg, scale_audit=True)
        cfg["data"]["stocks"].append({"stock_code": "OTHER", "raw_root": "unused"})
        with self.assertRaisesRegex(ConfigError, "feature_norm_map_path"):
            select_stages(cfg, scale_audit=True)
        cfg["stage3"]["feature_norm_map_path"] = str(
            Path(cfg["stage1_5"]["out_dir"]) / "factor_norm_map.json"
        )
        self.assertEqual(select_stages(cfg, scale_audit=True)[1].name, "scale-audit")
        cfg["stage1_5"]["use_train_split_only"] = False
        with self.assertRaisesRegex(ConfigError, "train_split_only"):
            select_stages(cfg, scale_audit=True)

    def test_failure_stops_before_any_downstream_stage(self):
        results = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 7)]
        with (
            patch("src.pipeline.subprocess.run", side_effect=results) as run,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            code = execute_plan([BY_NAME[name] for name in ("manifest", "labels", "pack")], EXAMPLE)
        self.assertEqual(code, 7)
        self.assertEqual(run.call_count, 2)

    def test_failed_gpu_preflight_never_starts_preprocessing(self):
        with (
            patch("src.pipeline.environment_report", return_value={"errors": ["GPU unavailable"]}),
            patch("src.pipeline.execute_plan") as execute,
            redirect_stderr(io.StringIO()),
        ):
            code = main(["--config-path", str(EXAMPLE)])
        self.assertEqual(code, 2)
        execute.assert_not_called()

    def test_subset_runs_in_dependency_order_and_rejects_duplicates(self):
        cfg = load_config(EXAMPLE)
        self.assertEqual(
            [s.name for s in select_stages(cfg, ["evaluate", "predict"])], ["predict", "evaluate"]
        )
        with self.assertRaises(ConfigError):
            select_stages(cfg, ["train", "train"])


if __name__ == "__main__":
    unittest.main()
