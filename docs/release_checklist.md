# Public release checklist

Use this checklist before making the GitHub repository public.

## Ownership and confidentiality

- [ ] Confirm in writing that publishing the source does not violate an internship, employer, client, exchange-data, or nondisclosure agreement.
- [ ] Confirm that the implementation is yours to publish and does not contain proprietary code copied from another repository.
- [ ] Remove real market data, non-public factor definitions, internal hostnames, usernames, and organization names.
- [ ] Decide whether public visibility without an open-source license matches the intended portfolio use.

## Git boundary

- [ ] Review `git status --short --ignored` before the first commit.
- [ ] Confirm every `*.local.json` file is ignored.
- [ ] Confirm `data/`, `results/`, `predictions/`, `backtest/`, checkpoints, TensorBoard logs, and model files are ignored.
- [ ] Search the complete Git history—not only the working tree—for secrets and sensitive paths.
- [ ] Check large files before pushing and avoid adding raw artifacts through `git add -f`.

## Technical evidence

- [ ] Resolve the correctness items recorded in the private audit before claiming reproducibility.
- [x] Add synthetic unit and targeted integration tests for timestamps, labels, window loading, inference alignment, backtest rules and pipeline configuration.
- [x] Add local development checks and a matching lightweight CI workflow. GitHub-hosted lightweight CI [passed for commit `57b1ead`](https://github.com/Hanibote0624/lob-return-forecasting/actions/runs/35343126972) on 2026-09-18. This run does not cover TensorFlow model tests, GPU training or Windows execution.
- [x] Run small TensorFlow component and serialization checks on CPU; record versions and source hashes.
- [x] Bind target definitions and scaling through training/prediction/evaluation artifacts.
- [ ] Verify the separate GPU acceptance command and pin a tested CUDA environment.
- [x] Run the complete synthetic Stage1–5 preparation fixture with independent label/window references and fault-injection tests. See [V5 evidence](synthetic_verification.md); model execution is not included.
- [ ] Re-run at least one GPU experiment from a clean clone and retain the exact commit hash and configuration snapshot.
- [ ] Publish only sanitized, provenance-linked metrics and clearly distinguish validation from test performance.

## Presentation

- [ ] Replace the repository-status warning in `README.md` only after a clean rerun succeeds.
- [x] Add the model architecture and an explicitly conceptual event-time/target figure.
- [x] Add bilingual READMEs, a project walkthrough and a claim-to-evidence guide.
- [ ] Add a real-data result figure only after its source experiment and publication scope are verified.
- [ ] Explain negative or inconclusive results instead of selecting only favorable metrics.
- [x] Explain execution assumptions, external-factor causality, temporal generalization and missing GPU/result evidence.
- [x] Verify the public configuration dry-run and document the scope of recorded checks.
- [ ] Verify research execution in a clean GPU environment.
