# Data and label contract

This document defines the intended behavior of the cleaned pipeline. It is the reference for new tests and supersedes ambiguous historical comments in individual scripts.

## Session input

Each CSV is one stock/session. Rows must retain source order and timestamps must be finite and nondecreasing. Duplicate timestamps are allowed; sorting or deduplicating inside later stages is not allowed because labels, factors, and row identifiers must remain aligned.

The numeric timestamp format is `HHMMSSmmm`. Stage 2 converts it to seconds and stores `t_sec` as `float64`. Training and inference must subtract the first timestamp of each window in `float64` before casting relative time to `float32`; casting absolute intraday seconds first loses part of the millisecond resolution.

## Mid-price validity

For each row:

1. use `(bid + ask) / 2` when both quotes are finite, strictly positive, and `ask >= bid`;
2. otherwise use the last-trade price; and
3. treat the resulting price as invalid unless it is finite and strictly positive.

V5 computes returns from the unrounded `float64` reference price. It also stores `mid` as `float64`; half-tick midpoints must not be rounded to three decimals before label construction. Raw/scaled return arrays remain `float32`. Raw and final label bundles carry `label_contract_version=2`, and label statistics record this version, `mid_price_rounding=none`, and `mid_price_storage=float64`.

Earlier Stage2 code rounded midpoints, so its labels must be rebuilt from CSV along with downstream packs, models and predictions. `--only-scale` rejects raw bundles that lack the V5 precision contract. This does not automatically migrate historical models or certify manually edited artifacts.

An invalid current price cannot produce a valid return label. Invalid future prices are excluded from the future aggregate.

## Future-return label

For event row (i), let

\[
F_i = \{j: t_i + h_{min} \le t_j \le t_i + h_{max}\}.
\]

The current target is

\[
r_i = \frac{\operatorname{mean}\{m_j: j\in F_i,\ m_j\text{ valid}\}}{m_i} - 1.
\]

This is an **event-weighted arithmetic mean**: every observed row in the future interval has equal weight. It is not a time-weighted price and is not the price at one fixed future instant.

A label is valid only when all configured conditions hold:

- the current mid-price is valid;
- at least `label.min_future_observations` valid future prices exist;
- the computed return is finite; and
- when `label.require_full_horizon=true`, the session timeline reaches `t_i + h_max`.

Invalid labels remain `NaN`; an empty future interval is never converted to a zero return. Stage 2 stores a bit mask describing invalidity reasons and reports aggregate counts.

Scaling uses the configured positive fixed multiplier. When `label.clip_mode=q99_abs_train_only`, the scaled target is clipped to the absolute 99th-percentile threshold estimated from training sessions only. Raw returns remain unclipped.

## Historical input window

A model window ends at a row with positive finite sample weight and contains exactly `features.window_W` rows from the same session. It must also satisfy:

- finite, nondecreasing event times;
- total elapsed time no greater than `data.session_rules.max_history_span_seconds` when positive; and
- no adjacent event gap greater than `data.session_rules.max_inter_event_gap_seconds` when positive.

The gap constraint is disabled by setting it to `0.0`. The history-span constraint is separate from future-label coverage; neither substitutes for the other.

## Split and leakage boundary

Date splits are chronological and frozen in Stage 1. Label quantiles and feature statistics use training data only. Trading thresholds use an explicitly documented, disjoint validation calibration split and remain fixed on test. No window crosses a session or trading day.

Prediction bundles carry every window's original CSV endpoint row and absolute `float64` timestamp. Evaluation must index labels by those row IDs and verify timestamp equality; implicit offsets and silent length truncation are prohibited. Missing endpoint ranges created by history-time constraints are treated as discontinuities in event analysis and backtesting.

## V6 target provenance

All current main-pipeline artifacts carry a target-contract fingerprint and exact label-statistics file fingerprint. Raw labels also identify their full label definition and source CSV. This binds actual time bounds, aggregation/price/validity rules, scaling, training-calibrated clipping and calibration settings even when the horizon name is unchanged. Model-contract and prediction-format versions are now 2. Historical and V5 artifacts must be rebuilt from CSV; they cannot be upgraded by editing metadata. See [V6 migration](v6_release_notes.zh-CN.md). These hashes detect stale/mixed artifacts, not deliberate tampering.
