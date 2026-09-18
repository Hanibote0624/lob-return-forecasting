# Changelog

## V7 portfolio documentation

- Rewrite English and Chinese READMEs around the research task, design decisions and evidence.
- Add an architecture explanation, Chinese project walkthrough, evidence guide and private-data GPU runbook.
- Add an original event-time/target schematic with explicit non-empirical labeling.
- Preserve V6 runtime code, tests, configuration, dependencies and numerical verification records.
- Keep GPU experiments, real-data results and hosted CI explicitly pending.


## V6 technical candidate

- Bind Stage2 definitions, scaling and training statistics through packs, models, predictions and evaluation; reject legacy or mismatched artifacts.
- Preserve float32 arithmetic before mixed-policy standardization and elapsed-time encoding.
- Fix weighted/low-variance metric edge cases; restrict batch CCC to one replica.
- Verify model components, weighted gradients and save/load for four losses under two precision policies.
- Add a separate single-GPU synthetic acceptance entry point; GPU execution remains pending.


## V5 — Synthetic preparation integration

- Execute Stage1–5 from generated CSVs through the public runner, with independent reference calculations for every label and eligible window.
- Verify training-statistics invariance when only held-out prices/factors change; exercise shard rollover, session boundaries, time constraints and partial evaluation batches.
- Remove Stage2's implicit three-decimal midpoint rounding, retain float64 midpoints, and version the label precision contract. Reject rescaling older raw labels as the new format; rebuild downstream artifacts.
- Check Stage3/4 manifest structure and sampled timestamps in Stage5. Report detected failures in both strict and diagnostic modes.
- Add eight integration/failure tests and one midpoint regression test (73 tests total), a retained-output verification command and a sanitized evidence report.

See [V5 release notes](docs/v5_release_notes.zh-CN.md) and [verification scope](docs/synthetic_verification.md).

## V4 — Engineering and configuration

- Add a shared configuration loader and semantic preflight checks.
- Resolve project-relative paths against the checkout, independent of caller directory.
- Add Python stage orchestration, a dependency-free dry run, stage selection, and early GPU preflight; retain `run_all.sh` as a wrapper.
- Read label scale from JSON instead of silently overriding it with 200.
- Correct Stage1's `stage0.enabled` configuration key.
- Separate tested development dependencies from unverified GPU/experimental candidates.
- Add local correctness checks, a GitHub workflow, environment inventory and development documentation.
- Complete V4 configuration guards for boolean switches, required timestamp output, output path collisions, supported optimizer/label/weight settings and malformed backtest thresholds.
- Remove the unused optimizer `eps` key from the template while preserving the effective `epsilon`; reject the ambiguous legacy key during preflight.
- Extend configuration/runner regression coverage from 13 to 21 tests (64 total); reject overflowing JSON floating-point literals.

See [V4 release notes](docs/v4_release_notes.zh-CN.md) for evidence and migration details.

## V3 — Training/inference/evaluation consistency

Correct batch coverage, share preprocessing, bind model/prediction artifacts to explicit contracts, align predictions by original rows, and calibrate backtest thresholds on an earlier split. See [V3 release notes](docs/v3_release_notes.zh-CN.md).

## V2 — Data contract

Clarify future-event labels, invalid observations, full-horizon validity, timestamp precision, train-only clipping and history-window constraints. See [data contract](docs/data_contract.md).

## V1 — Public repository boundary

Separate public source/example configuration from private data, historical artifacts and machine-local settings.
