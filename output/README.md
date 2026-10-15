# Nutrition QAOA result figures

These figures describe one completed 16-qubit CPU statevector study:
4 QAOA depths x 2 optimizers x 3 independent seeds x 3 restarts = 72
optimizations, each capped at 400 objective evaluations. They describe
simulator results, not quantum hardware performance or clinical dietary advice.

| Figure | Interpretation |
| --- | --- |
| [depth_tradeoffs.png](depth_tradeoffs.png) | Expected true cost, probability of measuring a menu in the true top 1%, and optimizer convergence within the evaluation budget. Error bars show the standard deviation across three seed means; each seed mean averages three restarts. |
| [paired_seed_variability.png](paired_seed_variability.png) | Paired comparisons of p=2 and p=4 for each seed. Seed 1 contributes strongly to the lower average cost at p=2; this does not establish that depth 2 is inherently better. |
| [sampling_vs_uniform.png](sampling_vs_uniform.png) | Fraction of runs observing at least one exact optimum in 1,024, 4,096, or 16,384 simulated measurements. The orange line is the theoretical probability for independent uniform sampling with one exact optimum among 65,536 menus. Shot counts share prefixes of the same random stream and are not independent trials. |
| [surrogate_quality.png](surrogate_quality.png) | High global correlation of the QUBO surrogate can hide poor ordering near the optimum. The weighted fits are diagnostics and were not used to optimize this completed QAOA run. |

[analysis_data.json](analysis_data.json) contains the aggregate metrics, seed
identifiers, and sampling counts used to draw the figures. The anonymized
[executed notebook](../nutrition_qaoa_result_public/start_here.ipynb) and
[complete run outputs](../nutrition_qaoa_result_public/outputs/runs/20261014T103117_956650Z/)
are published separately. Observing an optimum means finding it after scoring
multiple measured samples with the true objective. This does not establish a
high single-measurement optimum probability or a runtime advantage over
exhaustive search.

To redraw the committed figures, install NumPy and Matplotlib and run from the
repository root:

```bash
python output/build_analysis.py
```

To refresh the aggregate data from another completed run before redrawing:

```bash
python output/build_analysis.py --run-dir <completed-run-directory> --exact-optima 1
```

`--exact-optima` must match the number of exact minimizers of that run's true
objective. The published run was checked with a strict numerical tolerance and
has one exact minimizer; do not assume the same count for another dataset.
