# Development workflow

This repository is a checkout-based research project. `pyproject.toml` configures tools; the project is not a published Python package, and `pip install .` is not the installation path.

## Source boundaries

| Area | Responsibility |
| --- | --- |
| `src/stage1` through `src/stage6` entry points | Historical research stages, retained under their existing names for traceability |
| `src/data_contract.py`, `feature_preprocessing.py`, `window_loader.py`, `prediction_io.py`, `artifact_contract.py`, `backtest_core.py` | Shared data, inference and execution rules |
| `src/configuration.py` | Strict JSON loading, path resolution, and main-pipeline configuration checks |
| `src/pipeline.py`, `runtime_environment.py` | Stage selection, dependency/GPU preflight, and failure propagation |
| `scripts/` | User-facing development and execution commands |
| `requirements/` | Core, development, GPU candidate and optional experiment dependencies |
| `tests/` | Synthetic unit and targeted integration tests |
| `docs/` | Data/inference contracts, environment status, migration notes and release procedure |

Stage0 scripts, `merge_txt_into_csv.py`, `lgbm_compare_stage0_factors.py`, and `stage2_build_labels_tick10_mid_x200.py` remain historical utilities in `src/`. They are outside the default pipeline. Their presence is not a statement that their methods or results have passed the main pipeline's validation. Optional historical commands should be launched from the repository root.

## Change and verify

1. Make a focused change in a branch and keep real configuration in ignored `*.local.json` files.
2. If labels, ordering, masks, timestamps, or preprocessing semantics change, update the corresponding contract and add a test that distinguishes the old and new behavior.
3. Run `python scripts/check.py` in the development environment.
4. Run `bash -n run_all.sh` after editing the shell wrapper, and inspect a `--dry-run` for orchestration changes.
5. Review the diff and `git diff --check`. Report GPU-dependent verification separately.

Ruff checks syntax-related issues, invalid comparisons/control flow, and undefined names throughout source, tests and scripts. It is deliberately a correctness gate, not a claim that all archived code has been reformatted or fully reviewed. Configuration is in `pyproject.toml`, following [Ruff's configuration format](https://docs.astral.sh/ruff/configuration/).

## Execution and resuming

The Python entry point and `run_all.sh` invoke the same planner. Explicit stage selection runs a subset in the original dependency order:

```bash
python scripts/run_pipeline.py --config-path config/research.local.json --stages manifest labels pack windows healthcheck
python scripts/run_pipeline.py --config-path config/research.local.json --stages train predict evaluate
python scripts/run_pipeline.py --config-path config/research.local.json --stages backtest
```

Selecting stages does not regenerate their prerequisites or resume an interrupted training checkpoint. The required inputs must already exist, and the individual stage checks still apply. A failing stage prevents all downstream selected stages from starting. When a selected plan includes training or prediction, GPU preflight happens before the first stage.

Stage0 enrichment and LightGBM remain separate commands. For Stage1.5, configure at least two stocks, train-only auditing, and a Stage3 normalization map pointing to `stage1_5.out_dir/factor_norm_map.json`, then add `--scale-audit`.

## Maintainer review

Review public changes for experiment provenance as well as code correctness. Do not turn historical metrics into current benchmark claims. V5's [complete synthetic Stage1–5 verification](synthetic_verification.md) is included in unittest discovery and therefore in `scripts/check.py`. Its references use scalar operations and integer milliseconds independently of the production label/preprocessing/window helpers. GPU model validation and a real experiment remain separate milestones.

## V6 model and provenance checks

`src/target_contract.py` binds Stage2 target settings and statistics through the main artifact chain. Tests inject stale statistics, label files, model versions and prediction metadata. The independent TensorFlow suite is under `model_tests/` and runs through `scripts/check_model.py`; see [model verification](model_verification.md) for commands and limits. It is deliberately separate from lightweight CI.

## Portfolio documentation

Start with the [evidence guide](evidence.md) when updating research claims. The [architecture](architecture.md), [Chinese walkthrough](project_walkthrough.zh-CN.md), and [runbook](research_runbook.md) separate the project explanation from operational detail. V7 preserves V6 implementation bytes and existing numerical evidence; its documentation check does not replace a GPU experiment.
