# Research runbook

[English README](../README.md) · [中文说明](../README.zh-CN.md) · [Environment](environment.md)

This is the private-data GPU workflow. The public example is a schema/template, not a bundled research dataset. GPU success-path validation remains pending; inspect the [verification boundaries](model_verification.md) before interpreting a run.

## 1. Prepare a separate configuration

From the repository root, on Linux/WSL, copy the example to a new local file. If you already have that local file, edit it rather than replacing it:

```bash
cp -n config/gp_lit_regression_v6_gpmain_64.example.json config/research.local.json
```

Set the actual numeric stock identifier, raw-data root, date range and non-overlapping date splits. Keep a single stock and `stage3.feature_norm_map_path=null` for the main path. The template's `SAMPLE` is a placeholder, not a valid dataset identity for discovery.

Set `train.run_name`, `predict.run_name` and `backtest.run_name` to the same new name. Prediction defaults to validation and test; default accuracy diagnostics use test only. Review all output paths so runs do not overwrite each other's prerequisite artifacts. The template has server-sized worker counts and batch size 4096; choose values appropriate to the machine.

Each raw CSV represents one stock/session, ordered by event time, with configured price/time columns and existing factor columns. See [configuration](../config/README.md) and [data contract](data_contract.md). Verify the supplied factors' causal construction separately.

## 2. Inspect the plan

```bash
python -m src.configuration --config-path config/research.local.json
python scripts/run_pipeline.py --config-path config/research.local.json --dry-run
```

A dry run checks configuration and prints commands. It does not certify the data or GPU. Relative `project_root` is resolved against the checkout; known data/artifact paths are resolved against that root.

## 3. Verify the GPU environment separately

Use the candidate GPU environment from [environment setup](environment.md). The default CCC objective requires one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --gpu --output local/environment.json
CUDA_VISIBLE_DEVICES=0 python scripts/check_gpu.py --output-dir local/gpu-acceptance
```

The acceptance output directory must be new. This check uses synthetic data, marks Stage6 artifacts as `smoke_test`, and cannot establish real-data performance. A second independent directory can exercise `--mixed-precision`; the research example defaults to float32.

## 4. Run the main path

With the GPU environment active and the private configuration ready:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_all.sh config/research.local.json
```

The wrapper invokes the shared Python planner. When training or prediction is selected, GPU/package preflight occurs before launching the first stage. `label.fixed_scale` is the scaling source; the old Bash `LABEL_SCALE` override is rejected.

Selected stages can be run with existing prerequisites:

```bash
python scripts/run_pipeline.py --config-path config/research.local.json --stages manifest labels pack windows healthcheck
CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py --config-path config/research.local.json --stages train predict evaluate
```

Selection does not rebuild prerequisites automatically and does not resume an interrupted fit. A failed stage stops subsequent stages; partial files from the failed stage may need rebuilding.

## 5. Inspect outputs before interpreting results

| Artifact | Check |
| --- | --- |
| Stage2 statistics and labels | Actual interval, precision, scaling, clipping, training quantiles and invalid-label reasons |
| Stage3/4 manifests and Stage5 report | Schema, counts, session boundaries, target fingerprints and health status |
| Run configuration, logs and model contract | Correct run identity, target, model settings, checkpoint hashes and completed status |
| Prediction summary | Required validation/test sessions completed without stale or missing outputs |
| Accuracy report | Evaluation split, valid aligned row counts and metric definitions |

`final.keras` uses the selected validation checkpoint's weights. Its optimizer state is not an exact resume state for that selected epoch. Keep the configuration, environment and code revision with any private research results.

## Optional branches

Stage0 enrichment, alternative tick labels and LightGBM comparisons are separate historical/experimental commands, outside the default runner. Setting `stage0.enabled=true` changes Stage1's input source; it does not run enrichment for you.

For a deliberate multi-stock experiment, configure at least two stocks and a training-only scale audit, then point `stage3.feature_norm_map_path` to the audit's `factor_norm_map.json` and add `--scale-audit`. This is not needed for the public single-stock example.

For the optional backtest, with existing predictions:

```bash
python scripts/run_pipeline.py --config-path config/research.local.json --stages backtest
```

Its default thresholds must be calibrated on an earlier validation split. Understand the [execution assumptions](inference_and_backtest_contract.md) before presenting any output as trading evidence. Visualization is another optional stage and is not part of the V6 numerical model verification.

## Migration

Historical/V5 labels and checkpoints lack the current target provenance chain. Rebuild Stage2–5 from CSV, train with a new run name and regenerate predictions/evaluation. V6 labels retain label version 2, while the model-contract and prediction-format versions are 2; the numbers describe different artifact types.

`--only-scale` may reuse only compatible current raw-label bundles. It verifies the raw definition and source CSV fingerprint, so source files are still needed. Changing horizon semantics or editing a metadata version field cannot upgrade a historical artifact.
