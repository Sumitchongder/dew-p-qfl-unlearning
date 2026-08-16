# Paper Results

This directory contains the exact experimental artifacts associated with
the manuscript *"Dynamic Entanglement-Weighted Pruning for Quantum
Federated Unlearning in Supply-Chain Risk Prediction"* submitted to
arXiv (accuracy 0.837 ± 0.029, Tables 3-9, Figures 4-16).

These results are intentionally frozen and should **not** be overwritten
when modifying the implementation, e.g. by re-running

```
python scripts/run_main_experiment.py
python scripts/run_sweeps.py
python scripts/run_reconstruction.py
python scripts/generate_deliverables.py
```

against the current source tree.

Subsequent code corrections (the QFIM `/4`-factor fix, removal of the
NumPy simulation backend so the circuit is simulated in Qiskit only, the
default switch from von Neumann entropy to concurrence in the
entanglement weight, the Appendix-C-exact dataset generator, the
logistic-regression shadow-model membership-inference attacker matching
Appendix D, and the addition of a separate `forgetting_output_divergence`
metric implementing Section 5.4's output-distribution-divergence wording
literally alongside the `forgetting_score` metric that actually produced
the numbers below) are maintained in `src/qflewp/` for future experiments
and extensions, and are documented in `docs/METHODOLOGY.md`. The exact
configuration that produced the numbers in this directory is recorded in
`configs/paper_v1.yaml`.

`configs/paper_v1.yaml` is a documentation-only record of those settings
(`scripts/run_main_experiment.py` takes them as individual CLI flags, e.g.
`--seeds 0 1 2 --n-rounds 6 --local-maxiter 16`, not as a config file); it
exists so the historical operating point is recorded in one place rather
than only implicitly in Table 1/Table 2.
