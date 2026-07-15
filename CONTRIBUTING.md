# Contributing

This repository accompanies a research paper and is primarily maintained
for reproducibility. External contributions are still welcome, particularly:

- Bug reports for numerical/statistical issues (please include the exact
  `results/json/*.json` config and a minimal reproducing command).
- Extensions to additional datasets, ansatz architectures, or unlearning
  baselines.
- Performance improvements to the statevector simulation in
  `src/qflewp/circuit.py` (e.g. sparse gate application, GPU backends).

## Development setup

```bash
git clone https://github.com/<your-username>/dew-p-qfl-unlearning.git
cd dew-p-qfl-unlearning
python3 -m venv venv && source venv/bin/activate
make install
make test
```

## Before opening a pull request

1. Run `make test` — all existing tests must pass.
2. Run `make lint` and address any new warnings you introduced.
3. If you change any scientific component (`src/qflewp/circuit.py`,
   `qfim.py`, `entanglement.py`, `pruning.py`, `evaluate.py`), add or update
   a corresponding test in `tests/test_qflewp.py` that would have caught
   the change.
4. If your change affects reported numbers, regenerate
   `results/` via `bash scripts/run_full_pipeline.sh` and note the new
   values in your PR description rather than committing stale figures.

## Code style

Plain, explicit numpy/scipy — no hidden magic. Prefer readability over
cleverness; this codebase is read by reviewers checking scientific claims,
not just by other engineers.
