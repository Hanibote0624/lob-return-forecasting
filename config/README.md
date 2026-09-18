# Configuration files

`gp_lit_regression_v6_gpmain_64.example.json` is the public, sanitized template. Its stock identifier, paths, dates, and float-share value are placeholders.

Use a numeric stock code for actual Stage 1 file discovery; `SAMPLE` in the template is intentionally not a dataset identifier.

Create a separate `*.local.json` file for real paths and metadata. Files matching that suffix are ignored by Git. Never force-add them because they can reveal machine paths, dataset identities, or research parameters that are not intended for publication.

Before a run, verify at least:

- `project.project_root` resolves correctly from the repository root;
- `data.stocks` contains only the intended stock(s);
- raw roots, output roots, and date splits exist and do not overlap;
- `float_shares` is correct for the relevant dates if scale normalization is used;
- `stage3.feature_norm_map_path` is `null` for a single-stock run, or points to a training-only Stage 1.5 output for a deliberate multi-stock run;
- `horizons.active_horizon_id` matches the experiment;
- `train.run_name`, `predict.run_name`, and `backtest.run_name` are identical;
- model and prediction paths resolve to the same run;
- prediction includes `val` when the backtest uses validation-calibrated thresholds, while accuracy evaluation remains restricted to `test`;
- worker counts, batch size, precision mode, and visible GPUs match the machine.

The example deliberately keeps Stage 0 disabled. The main dataset already contains precomputed factors; Stage 0 is an optional factor-engineering experiment. `stage0.enabled=true` tells Stage 1 to read each stock's `out_root`; it does not execute enrichment. The misspelled historical `stage0.enable` is rejected by the validator.

The example also keeps cross-stock factor normalization disabled. Stage 6 emits a `model_contract.json`, and prediction validates that contract before loading raw data. Do not point the cleaned predictor at a historical model that lacks this contract; retrain it with the current pipeline.

## Resolution and validation

- The command-line configuration path is relative to the caller's current directory.
- A relative `project.project_root` is resolved relative to this checkout, not relative to the configuration file or caller.
- Stock roots and known artifact paths are resolved against `project_root`. Absolute paths remain absolute.
- On Linux/WSL, use Linux paths such as `/mnt/c/...`; Windows drive paths are rejected rather than treated as relative Linux paths.
- The source JSON is never rewritten. Training snapshots contain resolved paths and remain private artifacts.

Validate before running:

```bash
python -m src.configuration --config-path config/research.local.json
python scripts/run_pipeline.py --config-path config/research.local.json --dry-run
```

The validator checks the complete main-pipeline configuration: required fields, positive dimensions/batch sizes, chronological splits, horizon definitions, matching run names and model locations, evaluation coverage, and supported path placeholders. It rejects duplicate JSON keys and non-finite JSON numbers. This is a semantic preflight for supported fields, not an exhaustive schema for every archived experiment. It does not check file contents or certify an environment.

The V4 closing review adds these checks before dependency probes or data processing:

- Supported control switches must use JSON `true`/`false`, not strings or integers. When Stage0 input selection is enabled, every stock needs an `out_root`.
- Keep `stage3.write_t_sec=true`: the current training window loader requires timestamp arrays even with time-aware positional encoding disabled. `write_y_raw` and `write_is_valid` remain optional.
- Stage2 accepts `label.clip_mode` values `q99_abs_train_only`, `none`, `off`, and `disabled` (case-insensitive). Stage3 implements `sample_weight.mode=abs_r_q90_train_only`, with nonnegative `alpha`/`clip_max` and `0 < min_weight <= max_weight`.
- Stage6 implements AdamW; `train.optimizer.name`, if supplied, must be `adamw`. Use `epsilon` for its numerical-stability parameter. The historical `eps` key was never read by Stage6 and is now rejected to prevent an apparently configured value from being ignored. Remove `eps` from migrated private configurations and explicitly retain the intended `epsilon`; the public template keeps the previously effective `epsilon=1e-7`.
- Explicit manifest, schema, statistics and report output files must have distinct resolved paths. Raw and final label directories must also differ. Intended reader references, such as `accuracy_eval.pred_root=predict.output_root`, remain valid.
- Backtest calibration splits must name `train`, `val` or `test`; fixed thresholds must be ordered finite numbers, not booleans.

Individual active stage scripts share path loading but do not run the entire semantic validator, allowing smaller stage-specific fixtures and diagnostics. The shared runner is the supported full-pipeline entry point.

`{horizon_id}` is supported by Stage2 statistics/index/summary paths, Stage3 statistics/manifest/final-schema paths, Stage4 manifests and Stage5 reports. Prediction, accuracy and backtest output paths also support `{run_name}`. Directory roots and Stage1 paths must be concrete. `data.file_pattern` describes the existing naming convention; it is not a general filename-regex customization interface.

Set `label.fixed_scale` in JSON. `run_all.sh` no longer imposes a scale of 200 or accepts `LABEL_SCALE`. Existing configurations with `fixed_scale=200` keep that behavior, and the configuration snapshot records the effective setting.
