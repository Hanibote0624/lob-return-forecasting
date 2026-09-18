# Synthetic Stage1–5 verification

The preparation check, introduced in V5 and repeated with V6 provenance checks, runs the real pipeline against generated CSVs and compares outputs with independent references. It verifies data preparation and window loading without private market data, TensorFlow or GPU training. It does not measure predictive performance.

## Run the check

Use the Python 3.12 development environment described in [environment.md](environment.md):

```bash
python -m pip install -r requirements/dev.txt
python scripts/check_synthetic_pipeline.py
```

The default command uses a temporary directory, prints a JSON summary, and removes generated files when finished. To inspect CSVs, labels, packs, logs and health reports, choose a **new** output directory:

```bash
python scripts/check_synthetic_pipeline.py --output-dir local/synthetic-review
```

Existing output directories are rejected rather than overwritten. Relative output paths are relative to the caller. If a retained run fails, inspect its `pipeline.log`; choose another directory for the next attempt. Generated files stay under the ignored `local/` directory in this example.

`python scripts/check.py` also runs the integration tests and fault-injection cases. The existing GitHub workflow invokes this same command. No separate service, secret, GPU or dataset download is required. GitHub-hosted lightweight CI [passed for commit `57b1ead`](https://github.com/Hanibote0624/lob-return-forecasting/actions/runs/35343126972) on 2026-09-18. This run does not cover TensorFlow model tests, GPU training or Windows execution.

## Fixture

The fixture contains one fictional stock (`000000`), three synthetic factors and eight sessions. Stage0 enrichment and Stage1.5 normalization are disabled. Generated factor formulas have no relationship to the original private factor definitions.

| Split | Sessions | Source rows | Valid labels | Eligible windows | Shards | Last evaluation batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 4 | 1,536 | 1,284 | 472 | 2 | 24 |
| Validation | 2 | 768 | 642 | 236 | 1 | 44 |
| Test | 2 | 768 | 642 | 236 | 1 | 44 |

These are verification counts, not research results. The fixture uses an eight-event window, a 2.5–3.5 second future interval, multiplier 37, two required future observations, a one-second maximum history span and a 450 ms maximum adjacent gap. Production Stage2's minimum of 1,000 training samples is retained. Packs hold whole sessions and contain unused capacity; five-endpoint blocks are deliberately smaller than the 64-window evaluation batch.

The data include 0–400 ms ordinary event gaps, six-second interruptions, duplicate timestamps, minute rollovers, half-tick midpoints, crossed quotes with a valid last price, unrecoverable prices, and missing/infinite factor values. CSV column order differs from factor-schema order. Validation/test factors and price movements differ substantially from training.

## Independent checks

`scripts/synthetic_data.py` provides the reference calculation. It enumerates future rows with integer-millisecond comparisons, computes prices and means directly, fills factors in scalar session-local loops, and enumerates eligible history windows. It does not call the production label, preprocessing or window-mask functions.

`scripts/check_synthetic_pipeline.py` invokes the public runner for `manifest labels pack windows healthcheck`, then checks:

- Session identities, row counts, split membership and factor ordering.
- Every raw/scaled label, validity bit, invalidity reason, future count, interval boundary and clipping flag.
- Training-only label quantiles, feature means, sample standard deviations and schema fingerprint.
- Every used packed row, every Stage4 endpoint, session isolation, shard rollover and unused tails.
- Every materialized loader window, relative time, target and weight, including incomplete evaluation batches.
- Stage5's strict result and recorded violations.

A second full run changes only validation/test prices and factors. Its held-out artifacts must change while the training-statistics fingerprint stays identical. Training feature statistics are also independently calculated from valid training rows; a matching fingerprint alone is insufficient evidence.

## Failure checks

The unittest suite additionally injects errors into generated files:

| Error | Required behavior |
| --- | --- |
| Missing factor column | Stage1 fails; labels and packs are not generated |
| Decreasing timestamp inside a session | Stage2 fails; packing does not start |
| Old rounded raw labels passed to `--only-scale` | Stage2 refuses before replacing statistics or final labels |
| Changed factor-schema path or window size | Strict Stage5 fails and writes a failed report |
| Incorrect count, overlapping blocks or changed segment boundary | Strict Stage5 rejects manifest structure |
| NaN, decreasing, overlarge-gap or missing window timestamps | Strict Stage5 fails its sampled time checks |
| A detected error with `stage5.strict=false` | The process may finish, but the report still says `ok=false` |

Production Stage5 checks all manifest blocks and Stage3 segment references, then samples array values and windows. It does **not** exhaustively certify a large dataset. Exhaustive comparisons are practical here because the synthetic fixture is small.

## Recorded evidence and limits

[v6_synthetic_summary.json](verification/v6_synthetic_summary.json) records the current implementation’s observed local run, dependency versions, counts, generated-data/artifact fingerprints and checked source-file hashes. It contains no machine paths or real market data. Logs and generated binary artifacts are not committed.

The artifact fingerprint covers decoded array contents in a stable order. Incidental manifest fields, including timestamps, absolute paths and file modification times, are not a claim of byte-for-byte reproducibility. The report is evidence for the recorded source and environment; future revisions may produce different outputs.

This preparation fixture covers the single-stock main path. Multi-stock normalization, Stage0, the alternative tick target, LightGBM, model construction, model serialization, GPU kernels, training, real-data generalization and trading results remain outside this milestone. Separate unit tests cover parts of inference/evaluation/backtesting; this fixture does not run a trained-model end-to-end experiment.

## V6 continuation

[V6 evidence](verification/v6_synthetic_summary.json) re-runs these same independent references with target-provenance checks enabled. The V5 report remains a historical snapshot; its source hashes refer to V5. Model-component evidence is separate in [model verification](model_verification.md).
