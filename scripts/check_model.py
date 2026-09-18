#!/usr/bin/env python3
"""Run bounded TensorFlow component checks on an explicit device."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_soft_device_placement(False)
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu" and not gpus:
        raise RuntimeError("GPU validation requires a visible CUDA GPU; no CPU fallback")
    if gpus:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    device = "/GPU:0" if args.device == "gpu" else "/CPU:0"
    with tf.device(device):
        probe = tf.linalg.matmul(tf.ones([4, 4]), tf.ones([4, 4]))
    if args.device.upper() not in probe.device:
        raise RuntimeError("device placement probe failed")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "model_tests"))
    started = time.monotonic()
    # CPU placement is explicit in CPU mode. In GPU mode Keras places supported
    # model operations on the sole visible GPU; host-only bookkeeping stays CPU.
    tf.config.set_soft_device_placement(True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [str(p.relative_to(ROOT)) for folder in ("src", "scripts", "model_tests")
               for p in sorted((ROOT / folder).glob("*.py"))]
    report = {
        "status": "passed" if result.wasSuccessful() else "failed", "scope": "small_model_components",
        "device": args.device, "placement_probe": probe.device.split("/device:")[-1],
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "duration_seconds": round(time.monotonic() - started, 3),
        "packages": {p: importlib.metadata.version(p) for p in ("tensorflow", "keras", "numpy", "h5py")},
        "python": sys.version.split()[0],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "research_training_executed": False,
        "real_data_performance_verified": False,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
