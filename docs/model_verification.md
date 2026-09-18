# Model review and verification boundaries

V6 binds the target definition to artifacts and adds numerical checks for the time-aware Transformer–LSTM. Stage6 still requires a visible GPU. CPU component checks are small synthetic forward/backward operations, not a supported research-training pipeline.

## Reviewed behavior

| Area | V6 behavior | Evidence / remaining boundary |
| --- | --- | --- |
| Standardization | Compute `(x-mean)/std` in float32 before any mixed-precision projection | Regression fixture preserves the difference between 10000 and 10001; the previous mixed policy collapsed both |
| Elapsed-time encoding | Disable input autocasting before computing float32 sinusoidal phases; cast encoding only when adding to features | 8.000 and 8.001 seconds remain distinguishable; even/odd widths checked against NumPy |
| Attention and LSTM | Attend over the complete historical window, then a forward LSTM and scalar head | Both time-enabled and disabled models tested; no future label rows enter the feature window |
| Weighted CCC | Weights enter means, variances and covariance; zero-total-weight batch returns zero | NumPy reference and finite, nonzero gradients checked; only one replica supported |
| Pearson | Unweighted pooled validation metric using float64 accumulators | Uneven batches and very small return variance checked |
| R² / direction | Weighted pooled metrics; zero-weight observations contribute nothing | NumPy references, zero mass, reset, and constant-target policy checked |
| Checkpoints | Explicitly reload best validation-Pearson weights before saving `final.keras` | Component `.keras` and `.weights.h5` round trips tested; full callback/Stage6 execution awaits GPU |
| Other losses | MSE, Huber and LogCosh remain selectable | One small synthetic optimizer update and serialization checked for each |
| Precision policies | Float32 and `mixed_bfloat16` exercised locally | These checks do not verify NVIDIA kernels, CUDA/cuDNN, throughput or memory use |

The original mixed-policy failures were reproduced before fixing them: standardizing `[10000,10001]` with mean 10000/std 1 produced `[-16,-16]`, and elapsed-time encodings for 8 and 8.001 seconds became equal. Both now have regression tests. The public example continues to default to float32.

CCC is a batch statistic, so its logged epoch loss is an average of batch losses, not dataset-wide CCC. A singleton or constant-target batch has no useful covariance gradient. Multi-GPU CCC is rejected explicitly; using MSE/Huber/LogCosh avoids that restriction, but multi-GPU execution itself remains unverified. R² for constant targets is 1 for a perfect prediction and 0 otherwise; empty/zero-weight R² is 0. Pearson is 0 when either variance is zero. Float64 pooled raw moments suit normalized return-scale values; they do not guarantee accuracy for arbitrarily large offsets.

`final.keras` is an inference snapshot of the selected weights. Optimizer state is from the end of fitting, so it is not an exact training-resume checkpoint for the selected epoch. The workflow does not claim exact resume support.

## Run the local component checks

Use a separate Python 3.12 environment:

```bash
python -m venv .venv-model-check
source .venv-model-check/bin/activate
python -m pip install -r requirements/model-check.txt
python scripts/check_model.py --device cpu --report-path local/model-check/report.json
```

The recorded run used TensorFlow 2.20.0, Keras 3.15.1 and NumPy 2.3.5. The [component report](verification/v6_model_components.json) includes actual package versions and source hashes. All 13 test methods passed. The suite includes eight loss/policy combinations for compiled updates and save/load, plus a real prediction-CLI integration fixture that checks generation, reuse, evaluation and stale-bundle rejection. Its tiny checkpoint and contract are constructed explicitly in the test; this does not execute Stage6.fit. No real data or GPU was used. The dependency file pins direct model dependencies; the report is not a complete lockfile for every transitive dependency.

The lightweight `scripts/check.py` still runs without TensorFlow. It compiles the model-check sources but does not execute them. GitHub-hosted lightweight CI [passed for commit `57b1ead`](https://github.com/Hanibote0624/lob-return-forecasting/actions/runs/35343126972) on 2026-09-18. This run does not cover TensorFlow model tests, GPU training or Windows execution.

## Run the separate GPU acceptance check

On the configured Linux/WSL2 GPU environment, expose exactly one GPU for CCC. Each output directory must be new:

```bash
python -m pip install -r requirements/gpu.txt
CUDA_VISIBLE_DEVICES=0 python scripts/check_gpu.py --output-dir local/gpu-check-fp32
CUDA_VISIBLE_DEVICES=0 python scripts/check_gpu.py --output-dir local/gpu-check-bf16 --mixed-precision
```

The command checks GPU visibility, runs the model-component suite, generates only synthetic market rows, executes real Stage1–5 with independent reference comparisons, and runs Stage6's two-epoch smoke mode. It verifies saved file hashes, replays every validation window including the final batch, and compares `final.keras` against the best weights. Logs, configuration, environment/build details and source hashes stay in the chosen ignored directory. Failure exits nonzero; the command does not fall back to CPU.

Smoke artifacts retain `status=smoke_test` and cannot be passed off as completed research models by the normal prediction/evaluation entry points. The component suite exercises the raw-prediction CLI using its temporary checkpoint fixture; this acceptance command does not run that CLI against the Stage6 smoke model. That requires a subsequent clean GPU research run and its complete train → predict → evaluate path.

Only the no-GPU rejection path of this GPU command has been checked locally. GPU acceptance, real-data training, historical result reproduction, driver compatibility, XLA and multi-GPU execution remain pending. Synthetic success is not evidence of forecasting quality or trading profitability.
