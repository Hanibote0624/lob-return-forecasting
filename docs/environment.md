# Environments and validation status

The V6 development baseline is Python 3.12. Keep development checks and GPU research execution in separate virtual environments. Historical Python, TensorFlow, CUDA, and driver versions were not preserved; the files below do not reconstruct that original server.

| File | Purpose | Validation status |
| --- | --- | --- |
| `requirements/core.txt` | NumPy, pandas, SciPy for preparation/evaluation and synthetic tests | Installed and tested in a clean Linux Python 3.12 environment |
| `requirements/core-constraints.txt` | The core environment's four transitive dependencies | Versions recorded from that clean installation |
| `requirements/dev.txt` | Core dependencies plus Ruff | Used for the local checks and declared CI job |
| `requirements/model-check.txt` | Core plus TensorFlow 2.20.0 / Keras 3.15.1 | CPU component checks passed; no research training |
| `requirements/gpu.txt` | Core plus TensorFlow 2.20.0 / Keras 3.15.1 with CUDA extras | Dependency resolution checked; GPU installation and execution unverified |
| `requirements/experiments.txt` | LightGBM comparison, Stage0/PyArrow, and Plotly viewer | Optional pinned candidates; experimental results unverified |
| `requirements-test.txt` | Compatibility with the V3 test command | Includes `requirements/core.txt` |

`gpu.txt` pins TensorFlow and Keras, not its entire dependency graph or the NVIDIA driver. It is a candidate for future verification, not a GPU lockfile or a claim that this model works with all resolved versions. TensorFlow publishes a Python 3.12 Linux wheel and the `and-cuda` extra for the selected release. [TensorFlow 2.20.0 package](https://pypi.org/project/tensorflow/2.20.0/).

## Development checks

On Linux/WSL, from the checkout:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.txt
.venv/bin/python scripts/check.py
```

No activation script is required. In Windows PowerShell the corresponding commands are:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe scripts/check.py
```

The recorded test run was on Linux; Windows execution has not been verified in this workspace. These checks do not train a model on CPU. They compile Python source without importing optional modules, validate example configuration, check installed dependency compatibility, run Ruff correctness rules, and execute the synthetic regression tests.

Preview the pipeline using only the Python standard library:

```bash
python scripts/run_pipeline.py --config-path config/gp_lit_regression_v6_gpmain_64.example.json --dry-run
```

## Future GPU verification

The declared GPU target is Linux/WSL2 on x86-64 with an NVIDIA GPU. Modern TensorFlow GPU installations use the `and-cuda` extra; native Windows GPU support ended after TensorFlow 2.10. [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

In a separate Python 3.12 environment on that machine:

```bash
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install -r requirements/gpu.txt
.venv-gpu/bin/python -m pip check
.venv-gpu/bin/python scripts/check_environment.py --gpu --output local/environment.json
```

The probe imports TensorFlow in a child process and requires at least one visible GPU. A successful probe establishes visibility only; it does not establish kernel, serialization, loss, or training correctness. Driver and library requirements must match the selected TensorFlow release and the hardware.

Before labeling this environment verified, run the custom-layer/model round trip, metric/loss reference checks, a tiny GPU smoke test, and then a clean research experiment. Record the commit, configuration, `pip freeze`, environment inventory, GPU/driver information, random seed, and data identifiers privately with that run. Do not transfer packages from a developer's shared environment by an unreviewed global `pip freeze`.

## CI scope

The GitHub workflow installs only `requirements/dev.txt` on Ubuntu with Python 3.12 and runs `scripts/check.py`, Bash syntax checks, and a dry run. V5's full synthetic Stage1–5 runs and fault-injection tests are included in that check command. It needs neither secrets nor private market data. It has read-only repository permissions and does not train, deploy, or upload datasets. The action usage follows the official [checkout](https://github.com/actions/checkout) and [setup-python](https://github.com/actions/setup-python) documentation.

The same commands have been checked locally. The GitHub-hosted workflow has not been run before the repository is uploaded; no passing GitHub badge is claimed.

V6 adds [separate model/GPU verification commands](model_verification.md). CPU component success does not validate the CUDA-extra dependency graph or a GPU run.
