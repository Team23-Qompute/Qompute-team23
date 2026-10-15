# Nutrition QAOA: revised experiment

This branch, **`qaoa_new`**, contains the revised experiment and its completed
results. The repository's default `main` branch still shows the earlier work.

## Start here

Download these **two files** into the same Jupyter directory:

1. [start_here.ipynb](start_here.ipynb) - the current runnable notebook.
2. [nutrition_qaoa_extended.tar.gz](nutrition_qaoa_extended.tar.gz) - its matching
   model, experiment code, tests, configuration, and food data.

Open `start_here.ipynb` with a **Python 3.11 or 3.12 kernel** and select **Run
All**. Request **4 CPUs and 16 GB RAM** for the default CPU experiment. The
notebook installs its packages and unpacks the data. A GPU is optional.

The notebook is the experiment launcher: the cost function is implemented in
[model.py](nutrition_qaoa_result_public/model.py), its settings are in
[config.json](nutrition_qaoa_result_public/config.json), and the QAOA evaluation
loop is in [run_experiments.py](nutrition_qaoa_result_public/run_experiments.py).
These files are included in the archive and extracted automatically. Their
implementation is not embedded as one large notebook cell.

## What changed from the earlier notebook

- No fixed total-serving target or total-serving penalty. Each of eight foods
  is encoded with two bits for 0-3 portions, giving one full 16-qubit problem.
- Diversity rewards the number of different foods selected. Repeated portions
  of the same food do not earn additional diversity reward.
- Nutrient penalties use the configured targets, ranges, and one-sided
  references. Sodium has a softly weighted lower planning reference and an
  upper reference; potassium has a lower reference only. The configuration is
  illustrative and documented in the [model guide](nutrition_qaoa_result_public/README.md).
- Brute force evaluates all **65,536 menus** using the true cost. QAOA optimizes
  a quadratic surrogate; its error near the optimum is reported separately.
- The default comparison covers **p=1,2,3,4**, **seeds 1,7,21**, **3 restarts**,
  and **COBYLA / Nelder-Mead**: 72 optimizations with the same 400-evaluation
  cap per optimization. Measurement counts are 1,024, 4,096, and 16,384.
- Progress logs, a status file, heartbeats, model tests, surrogate diagnostics,
  and sensitivity analysis support inspection of completed or interrupted runs.

## Completed run and figures

- [Executed notebook with saved outputs](nutrition_qaoa_result_public/start_here.ipynb)
- [Complete output folder](nutrition_qaoa_result_public/outputs/runs/20261014T103117_956650Z/)
- [Figure guide and analysis](output/README.md)
- [Model, assumptions, and data-source links](nutrition_qaoa_result_public/README.md)
- [Publication and anonymization notes](nutrition_qaoa_result_public/RESULTS_README.md)

The completed CPU run includes 72 QAOA optimizations. Its results do not
establish a quantum runtime advantage. The root runnable notebook has cleared
outputs so a new run is clearly distinct from the archived executed notebook.
The root `nutrition_qaoa.ipynb` and `Track2_challenge_2_tutorial.ipynb` are earlier
notebooks, not the revised entry point.

To rebuild the two root upload files after changing the published source:

```bash
python output/build_upload.py
```
