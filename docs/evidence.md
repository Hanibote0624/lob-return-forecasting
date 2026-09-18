# Evidence guide

[English README](../README.md) · [中文说明](../README.zh-CN.md)

This page distinguishes code behavior, recorded synthetic verification and pending empirical work. V7 changes presentation materials; V6 source, configuration, tests and recorded numerical evidence remain unchanged.

## Read the records

| Record | Observed result | What it supports |
| --- | --- | --- |
| [V6 development checks](verification/v6_checks_summary.json) | 80 non-TensorFlow tests; dependency, configuration and static checks passed | The covered preparation, alignment, orchestration and backtest behaviors |
| [V6 synthetic preparation](verification/v6_synthetic_summary.json) | Two complete Stage1–5 scenarios passed independent reference comparisons | Correctness of generated labels/windows and unchanged training statistics after held-out perturbations |
| [V6 model checks](verification/v6_model_components.json) | 13 checks passed; TensorFlow 2.20.0, Keras 3.15.1, CPU | Numerical components, small optimizer updates, serialization and a temporary prediction/evaluation fixture |
| [V7 documentation check](verification/v7_documentation_summary.json) | See report for final packaging checks | Documentation/source consistency, relative links and preservation of the V6 implementation |

These counts overlap in purpose: the full preparation fixture is also used in the 80-test suite. They must not be presented as independent experiments or added together as 95 research runs. The 13 model checks include eight loss/precision combinations inside one test method.

The temporary predictor checkpoint is explicitly constructed by the test harness. It is not produced by a complete Stage6 research run. GitHub-hosted lightweight CI [passed for commit `57b1ead`](https://github.com/Hanibote0624/lob-return-forecasting/actions/runs/35343126972) on 2026-09-18. This run does not cover TensorFlow model tests, GPU training or Windows execution.

## Concrete claims and their evidence

| Claim | Inspect | What remains outside the claim |
| --- | --- | --- |
| Future labels use the intended event-time interval | [`build_event_mean_return_labels`](../src/data_contract.py), [independent reference](../scripts/synthetic_data.py) | Economic appropriateness of that label |
| Time information survives preprocessing and mixed policies | [time conversion](../src/data_contract.py), [component tests](../model_tests/test_components.py) | Forecast improvement from time encoding |
| Held-out perturbations leave training statistics unchanged | [integration tests](../tests/test_synthetic_pipeline.py), [record](verification/v6_synthetic_summary.json) | Causality of externally supplied factors; all possible leakage modes |
| Training and raw inference share preprocessing | [shared transform](../src/feature_preprocessing.py), [packing/inference parity test](../tests/test_pack_inference_parity.py) | Every future data vendor/schema variant |
| Labels, models and predictions cannot be mixed solely by matching filenames | [target contract](../src/target_contract.py), [CLI tests](../tests/test_evaluation_cli.py) | Deliberate tampering or undetected edits to all large packed arrays |
| Existing predictions can be reused only with matching provenance | [predictor](../src/batch_predict_raw.py), [real CLI fixture](../model_tests/test_prediction_cli.py) | Production throughput or a trained-model GPU deployment |
| Backtest thresholds come from a prior split | [calibration rules](../src/backtest_core.py), [tests](../tests/test_backtest_core.py) | Execution feasibility, market impact and profitability |

## A worked verification record

In **each** synthetic preparation scenario, the generator creates 8 sessions and 3,072 source rows. After label and history constraints, the reference counts 472 training, 236 validation and 236 test windows. All 944 windows are compared. A second scenario changes only validation/test values; the training-statistics fingerprint remains identical.

These are controlled test data. The deliberate held-out perturbation clips all valid held-out labels in this fixture, which stresses handling of scale shifts. That is a test construction, not a measured property of real securities. The counts are not the size of the original research dataset.

Production Stage5 checks every manifest block/segment reference but samples array values and windows. Exhaustive value comparisons apply to the small synthetic fixture, not automatically to a large private dataset.

## Research results are a separate milestone

The public package currently provides no provenance-matched table of real-data IC, directional accuracy, baseline outperformance or trading returns. Historical logs and recollections are not substituted for current-version evidence, especially after label-precision and model-precision changes.

Before adding a result, retain its code version, configuration, data/split identity, label/statistics fingerprints, checkpoint identity, prediction coverage and evaluation definition. Compare methods on matched target and endpoint populations. Dense overlapping windows should not be described as independent observations; report variation across sessions/time periods as well as pooled summaries.

A profitable execution claim additionally requires a defensible execution and capital model. The current optional backtest does not supply that evidence.

## Reproduce the checks that are available

```bash
python scripts/check.py
python scripts/check_synthetic_pipeline.py --output-dir local/evidence-review
```

The output directory for the second command must not already exist. These commands use the development environment. The separate model suite and GPU acceptance entry point are documented in [model verification](model_verification.md).

The V5 report remains historical. Use the V6 records for the implementation preserved here; use the V7 report only for this presentation update's checks.
