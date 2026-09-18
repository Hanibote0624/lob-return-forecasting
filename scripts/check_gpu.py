#!/usr/bin/env python3
"""Single-GPU synthetic acceptance: components, real Stage1–6, checkpoint replay.

This command refuses CPU fallback. It produces smoke-test artifacts, never a
publishable research result or a contract accepted by batch_predict_raw.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new, ignored directory")
    parser.add_argument("--mixed-precision", action="store_true", help="exercise mixed_bfloat16")
    args = parser.parse_args()
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU verification blocked: no visible GPU. No CPU fallback.", file=sys.stderr)
        return 2
    if len(gpus) != 1:
        print("Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0, for weighted CCC verification.", file=sys.stderr)
        return 2
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "synthetic_single_gpu_acceptance",
              "mixed_precision": args.mixed_precision, "real_data_performance_verified": False,
              "gpu_details": tf.config.experimental.get_device_details(gpus[0]),
              "tensorflow_build": tf.sysconfig.get_build_info(), "steps": []}
    report_path = output / "report.json"

    def persist():
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    def run(name, command):
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=900, check=False)
        report["steps"].append({"name": name, "returncode": result.returncode})
        persist()
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {name}.log")

    persist()
    try:
        run("components", [sys.executable, str(ROOT / "scripts/check_model.py"), "--device", "gpu",
                           "--report-path", str(output / "components.json")])
        from scripts.synthetic_data import create_fixture, write_json, artifact_path, read_json
        from scripts.check_synthetic_pipeline import verify_outputs
        cfg, config_path, sessions = create_fixture(output / "pipeline")
        cfg["train"].update(run_name="synthetic_gpu_smoke", mixed_precision=args.mixed_precision,
                            epochs=2, loss="ccc", steps_per_epoch=0, val_steps=0)
        cfg["predict"]["run_name"] = cfg["backtest"]["run_name"] = cfg["train"]["run_name"]
        cfg["model"].update(d_model=16, num_heads=2, num_layers=1, ff_dim=24,
                            lstm_units=12, head_hidden=8, dropout=0.)
        write_json(config_path, cfg)
        run("stage1_5", [sys.executable, str(ROOT / "scripts/run_pipeline.py"), "--config-path",
                          str(config_path), "--stages", "manifest", "labels", "pack", "windows", "healthcheck"])
        report["preparation"] = verify_outputs(cfg, sessions)
        run("stage6", [sys.executable, "-m", "src.stage6_train_regression", "--config-path",
                       str(config_path), "--smoke-test"])
        # Replay the saved best model on all validation windows including the tail.
        import numpy as np
        from src.artifact_contract import require_fingerprint
        from src.target_contract import load_target_binding, require_target_binding
        from src.window_loader import LoaderConfig, iter_batches_with_time
        from src import stage6_train_regression  # registers the serialized components
        run_dir = Path(cfg["project"]["project_root"]) / cfg["paths"]["results_dir"] / cfg["train"]["run_name"]
        contract = read_json(run_dir / "model_contract.json")
        if contract["status"] != "smoke_test":
            raise AssertionError("GPU acceptance artifacts must be marked smoke_test")
        require_target_binding(contract, load_target_binding(cfg))
        model_path, weights_path = run_dir / "models/final.keras", run_dir / "models/best.weights.h5"
        require_fingerprint(str(model_path), contract["model_sha256"])
        require_fingerprint(str(weights_path), contract["best_weights_sha256"])
        model = tf.keras.models.load_model(model_path, compile=False)
        batches = list(iter_batches_with_time(LoaderConfig(
            project_root=cfg["project"]["project_root"],
            stage4_manifest_path=str(artifact_path(cfg, "stage4", "stage4_manifest_path")),
            split="val", batch_size=64, window_W=8, num_factors=3,
            shuffle_blocks=False, drop_remainder=False,
        )))
        before = [model([x, t], training=False).numpy() for x, t, _, _ in batches]
        model.load_weights(weights_path)
        for expected, (x, t, _, _) in zip(before, batches):
            actual = model([x, t], training=False).numpy()
            if not np.isfinite(actual).all():
                raise AssertionError("nonfinite saved-model predictions")
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        report["validation_rows_replayed"] = sum(len(batch[2]) for batch in batches)
        if report["validation_rows_replayed"] != report["preparation"]["splits"]["val"]["windows"]:
            raise AssertionError("incomplete validation replay")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("src", "scripts", "model_tests") for path in sorted((ROOT / folder).glob("*.py"))
        }
        persist()
    print(f"GPU synthetic verification passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
