"""Configuration-driven stage orchestration, with a dependency-free dry run."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from .configuration import ConfigError, REPO_ROOT, load_config, validate_config
from .runtime_environment import environment_report


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    packages: tuple[str, ...] = ()
    gpu: bool = False


STAGES = (
    Stage("manifest", "src.stage1_make_manifest_multi"),
    Stage("scale-audit", "src.stage1_5_factor_scale_audit", ("numpy", "pandas")),
    Stage("labels", "src.stage2_build_labels_clear", ("numpy", "pandas")),
    Stage("pack", "src.stage3_pack_rows", ("numpy", "pandas")),
    Stage("windows", "src.stage4_build_stage4_manifest", ("numpy",)),
    Stage("healthcheck", "src.stage5_dataset_healthcheck", ("numpy",)),
    Stage("train", "src.stage6_train_regression", ("numpy", "tensorflow"), True),
    Stage("predict", "src.batch_predict_raw", ("numpy", "pandas", "tensorflow"), True),
    Stage("evaluate", "src.accuracy", ("numpy", "pandas", "scipy")),
    Stage("backtest", "src.backtest_engine_flip", ("numpy", "pandas")),
    Stage("visualize", "src.visualize_flip_trades", ("numpy", "pandas", "plotly")),
)
BY_NAME = {stage.name: stage for stage in STAGES}
DEFAULT_STAGES = (
    "manifest",
    "labels",
    "pack",
    "windows",
    "healthcheck",
    "train",
    "predict",
    "evaluate",
)


def select_stages(cfg, names=None, *, scale_audit=False, backtest=False, visualize=False):
    selected = list(DEFAULT_STAGES if names is None else names)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in BY_NAME for name in selected)
    ):
        raise ConfigError("stages must be a nonempty list of unique known stage names")
    for enabled, name in (
        (scale_audit, "scale-audit"),
        (backtest, "backtest"),
        (visualize, "visualize"),
    ):
        if enabled and name not in selected:
            selected.append(name)
    plan = [stage for stage in STAGES if stage.name in selected]
    if "scale-audit" in selected:
        if len(cfg["data"]["stocks"]) < 2:
            raise ConfigError(
                "scale-audit is a multi-stock experiment; configure at least two stocks"
            )
        audit = cfg.get("stage1_5", {})
        if audit.get("use_train_split_only") is not True:
            raise ConfigError("scale-audit requires stage1_5.use_train_split_only=true")
        expected = Path(audit.get("out_dir", "")) / "factor_norm_map.json"
        actual = cfg["stage3"].get("feature_norm_map_path")
        if not actual or Path(actual) != expected:
            raise ConfigError(
                "stage3.feature_norm_map_path must reference stage1_5.out_dir/factor_norm_map.json"
            )
    return plan


def stage_command(stage, config_path):
    command = [sys.executable, "-u", "-m", stage.module, "--config-path", str(config_path)]
    if stage.name == "manifest":
        command.extend(["--strict", "1"])
    return command


def execute_plan(plan, config_path):
    for index, stage in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] {stage.name}", flush=True)
        result = subprocess.run(stage_command(stage, config_path), cwd=REPO_ROOT, check=False)
        if result.returncode:
            print(
                f"Stage {stage.name} failed (exit {result.returncode}); pipeline stopped.",
                file=sys.stderr,
            )
            return result.returncode if result.returncode > 0 else 1
    print("Selected stages completed.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(BY_NAME),
        help="Run only these stages in dependency order; prerequisites must already exist.",
    )
    parser.add_argument("--scale-audit", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without importing ML libraries or writing outputs.",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config_path).expanduser().resolve()
    try:
        cfg = load_config(config_path)
        validate_config(cfg)
        plan = select_stages(
            cfg,
            args.stages,
            scale_audit=args.scale_audit,
            backtest=args.backtest,
            visualize=args.visualize,
        )
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_path": str(config_path),
                    "project_root": cfg["project"]["project_root"],
                    "label_scale": cfg["label"]["fixed_scale"],
                    "stages": [
                        {
                            "name": stage.name,
                            "requires_gpu": stage.gpu,
                            "argv": stage_command(stage, config_path),
                        }
                        for stage in plan
                    ],
                },
                indent=2,
            )
        )
        return 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    required = {name for stage in plan for name in stage.packages}
    report = environment_report(gpu=any(stage.gpu for stage in plan), required=required)
    if report["errors"]:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2
    try:
        return execute_plan(plan, config_path)
    except KeyboardInterrupt:
        print("Pipeline interrupted.", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"Could not start stage: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
