"""Lightweight environment inventory with an explicit, optional GPU probe."""

import importlib.metadata
import json
import platform
import subprocess
import sys


KNOWN_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "ruff",
    "tensorflow",
    "keras",
    "lightgbm",
    "pyarrow",
    "plotly",
)


def environment_report(*, gpu=False, required=()):
    versions = {}
    for name in KNOWN_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("the declared development baseline is Python 3.12")
    for name in sorted(set(required)):
        if versions.get(name) is None:
            errors.append(f"missing dependency: {name}")
    report = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
        "gpu_checked": False,
        "gpu": None,
        "errors": errors,
    }
    if gpu:
        if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            errors.append("GPU profile targets Linux/WSL2 x86-64")
        elif not versions.get("tensorflow"):
            errors.append("GPU execution requires TensorFlow; see requirements/gpu.txt")
        else:
            probe = (
                "import json, tensorflow as tf; "
                "devices=tf.config.list_physical_devices('GPU'); "
                "print(json.dumps({'devices':[d.name for d in devices], "
                "'tensorflow':tf.__version__, 'build':tf.sysconfig.get_build_info()},default=str)); "
                "raise SystemExit(0 if devices else 1)"
            )
            try:
                result = subprocess.run(
                    [sys.executable, "-c", probe], capture_output=True, text=True, timeout=45
                )
                report["gpu_checked"] = True
                if result.stdout.strip():
                    report["gpu"] = json.loads(result.stdout.strip().splitlines()[-1])
                if result.returncode:
                    errors.append("TensorFlow GPU probe failed or found no visible GPU")
                    report["gpu_error"] = result.stderr[-4000:]
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                errors.append(f"GPU probe failed: {type(exc).__name__}: {exc}")
    report["status"] = "failed" if errors else "passed"
    return report
