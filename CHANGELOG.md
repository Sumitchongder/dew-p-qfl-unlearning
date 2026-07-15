# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-07-14

### Added
- Initial public release of the `src/qflewp/` package: data re-uploading
  variational quantum circuit, FedAvg federated trainer with exact
  parameter-shift gradients, parameter-shift diagonal QFIM estimator,
  per-gate von Neumann entanglement weighting, EWP + 5 baseline unlearning
  methods, and a full evaluation suite (utility, membership-inference
  forgetting, retrain distance).
- End-to-end reproduction pipeline (`scripts/run_full_pipeline.sh`) and
  matching SLURM batch chain (`hpc/slurm/`).
- 10 CSV result tables and 16 publication figures (PNG + PDF) under
  `results/`.
- Test suite (`tests/test_qflewp.py`) covering circuit unitarity, QFIM
  client-dependence, entanglement-weight differentiation, and pruning
  correctness.
- CI workflow running the test suite on Python 3.10-3.12.

### Notes on prior internal iterations
Earlier internal drafts of this codebase used placeholder QFIM/entanglement
estimators that did not depend on client data, and an evaluation harness
that scored synthetic random predictions rather than the trained model's
actual outputs. The `src/qflewp/` package in this release replaces that
scaffolding entirely with data-dependent, parameter-shift-exact
implementations; see `docs/METHODOLOGY.md` for the mathematical
definitions used.
