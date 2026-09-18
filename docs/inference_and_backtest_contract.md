# Inference and backtest contract

This document records the safeguards that connect a trained model to raw-session inference, offline diagnostics, and the optional research backtest.

## Model contract

Stage 6 writes `results/<run_name>/model_contract.json` before training. The contract binds a run to:

- the horizon, lookback width, factor count, and ordered factor names;
- SHA-256 fingerprints of the Stage 3 factor schema, normalization map (when used), input statistics, and Stage 4 manifest;
- the feature-preprocessing version, effective fill/normalization parameters, and history-window time constraints; and
- the time-aware model configuration.

After training and saving succeed, Stage 6 adds the final model and best-weights fingerprints and marks the contract `trained`. The predictor requires this completed status; a `smoke_test` artifact is not accepted as a completed research run. Reusing an existing run name with a contract is rejected to avoid stale checkpoints.

Inference refuses to combine a model with a different factor order, normalization map, effective fill/scaling parameters, horizon, window width, or history constraint. Models created by the historical pre-contract script must be retrained before using the cleaned predictor. Fingerprints detect artifact changes; they do not replace a code-version record or establish data provenance.

## Shared preprocessing

`src/feature_preprocessing.py` is the single implementation used by both Stage 3 packing and raw-CSV inference. It owns within-session forward fill, the remaining-value fill, optional accumulated-volume normalization, and optional float-market-cap normalization. Inference retains source row order; it never sorts a CSV to repair invalid input.

The public single-stock configuration does not use the cross-stock normalization map. Stage 1.5 is an explicit multi-stock experiment and must use training dates only.

## Prediction bundle

Each session produces `{date}_{session}_pred.npz` with four fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `pred` | `float32[N]` | Model outputs in the scaled-label units used for training |
| `end_row` | `int64[N]` | Original CSV row at the end of each model window |
| `t_sec` | `float64[N]` | Absolute event time at `end_row` |
| `metadata_json` | JSON string | Run, split, session, row count, model-contract/source-CSV fingerprints, and format identity |

Windows are generated in bounded-memory batches. Only windows satisfying the configured history-span and inter-event-gap rules are emitted, so `end_row` need not be consecutive. Evaluation and backtesting align with labels and quotes using these explicit row IDs. Length truncation and implicit `W-1` offsets are forbidden.

Prediction defaults to both validation and test sessions because validation outputs are needed for causal threshold calibration. `accuracy.py` reports test diagnostics only by default.

## Offline accuracy diagnostics

Accuracy evaluation enumerates expected sessions from the Stage 2 label index, then verifies bundle identity, endpoint bounds, timestamp equality, and Stage 2 label validity. It requires a single model-contract fingerprint across the evaluated bundles. With `strict_complete=true`, a missing or invalid session fails evaluation. Its pooled top/bottom prediction thresholds are descriptive summaries estimated on the evaluated data; they are not trading parameters.

## Optional flip backtest

The default backtest calibrates long and short thresholds from executable rows in the validation split, then holds those thresholds fixed on the test split. Calibration must end before every trading split starts. A fixed-threshold mode is available when thresholds come from a separately documented source.

The backtest checks prediction coverage against the Stage 1 manifest, verifies source-CSV fingerprints, and processes the complete raw-session timeline. Different runs, source files, or endpoint timestamps cannot be silently combined.

At each raw row:

- a quote is executable when bid and ask are finite, both prices are positive, and `ask >= bid`;
- opening or flipping additionally requires a finite prediction;
- an invalid quote or missing prediction holds the existing position;
- session-end liquidation uses the actual final raw row, including when that row has no prediction;
- an invalid terminal quote with an open position fails the session; liquidation is never moved retrospectively to an earlier quote; and
- the final raw row cannot open a new round trip.

Longs enter at ask and exit at bid; shorts enter at bid and exit at ask. Configured commission, stamp tax, and additional slippage are then applied. Signal generation and execution on the same row remain an idealized assumption. Prior-split threshold calibration alone does not make this a realistic execution simulator.

The reported cumulative series is the cumulative sum of per-trade returns. It is deliberately not called an equity curve because this script does not define capital allocation across overlapping stocks or sessions. It also omits latency, queue position, fill probability, and endogenous market impact.

## V6 target checks

Prediction metadata carries the full target contract, its SHA-256 and the Stage2 statistics file SHA-256. Prediction, accuracy and backtest consumers require these to agree with current configuration/statistics and a completed version-2 model contract. Accuracy additionally checks each label bundle and its source CSV fingerprint against the prediction. Forecast values are in scaled/clipped-return units; accuracy IC/ranking and event-return diagnostics compare against raw unclipped labels. Dividing by scale does not invert clipping or make the model a calibrated raw-return forecast.
