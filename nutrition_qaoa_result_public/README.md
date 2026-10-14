# Nutrition QAOA: nutrition ranges and food variety

This folder contains the revised, standalone experiment. It solves the full
eight-food menu in one 16-qubit QAOA circuit and compares the samples against
an exhaustive classical answer. No total-serving target or serving-range
penalty is used. The only per-food encoding limit is 0–3 portions.

## Cost function

For food `i`, `n_i` is its number of portions. The nutrient total is
`N_k = sum_i A[i,k] * n_i`. The true policy cost is

```text
nutrition cost = energy target deviation
               + protein/carbohydrate/fat out-of-range costs
               + potassium shortfall cost
               + sodium below-AI and above-CDRR costs

diversity reward = -rho * (number of selected food types / number of foods)

true total cost = nutrition cost + diversity reward
```

Each squared deviation is normalized by its target or boundary. For a range
`[L,U]`, the cost is zero inside it and rises as
`max(0,(L-N)/L)^2 + max(0,(N-U)/U)^2` outside it. Potassium has only a lower
reference; the model does not invent an upper limit. The sodium AI is a soft
planning reference, not a strict safety threshold, so its low-side weight is
0.25 versus 1.0 on the high side in the demonstration configuration.

The reward counts a food once when its portion count is positive. Moving from
one to two or three portions does not change the diversity term. For the
two-bit encoding `n_i = x_i0 + 2*x_i1`, selection is exactly
`x_i0 OR x_i1 = x_i0 + x_i1 - x_i0*x_i1`, so diversity enters the QUBO exactly
without another qubit.

The **nutrient range penalties** are piecewise quadratic, so a 16-bit quadratic
surrogate is fitted to them. This approximation is validated against the true
cost before QAOA is run. Every QAOA sample is ranked using the **true** cost.
The run also reports error and rank agreement within 0.001 of the true optimum.
`surrogate_fit_comparison.csv` compares the global fit with two low-cost-weighted
fits. Weighted fits are **diagnostics only**: smaller local error can still
produce a worse QUBO optimum, so the notebook does not switch objectives
silently. The diversity reward itself remains exact in every fit.
The fit currently evaluates every menu, which is practical for 8 foods but
would need another strategy for a larger food set. It is a modelling study,
not a speed advantage claim.

Qiskit's [quadratic program guide](https://qiskit-community.github.io/qiskit-optimization/tutorials/01_quadratic_program.html)
documents the QUBO representation, and its
[Ising translator](https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.translators.to_ising.html)
maps binary variables to Pauli terms. The QAOA circuit then optimizes the
expected Ising energy.

With the checked-in demonstration settings, the exact optimizer chooses 17
portions across all 8 food types. The quadratic surrogate's best menu ranks
28th under the true objective. These figures are produced by
`--analysis-only` and will change when the food data or configuration changes.

## How the classical answer changes

Brute force now scores **all** `4^8 = 65,536` menus. It no longer filters to
menus with 14 portions. It minimizes the same true policy cost used to judge
QAOA samples, and reports the naturally selected total portions and number of
distinct foods. `diversity_sweep.csv` shows how changing the reward changes
these quantities.

## Data and assumptions

- [USDA FoodData Central](https://fdc.nal.usda.gov/) supplies the food nutrient
  records. Each row in [`data/foods.csv`](data/foods.csv) links to its FDC ID.
  The table begins with the original project's hand-entered, rounded
  per-portion values. It is a **cached project dataset**, not a fresh API
  extraction. The optional downloader below creates candidate USDA values for
  auditing; inspect food descriptions, energy definitions, serving sizes and
  differences before replacing the cached table.
- [USDA API guide](https://fdc.nal.usda.gov/api-guide/) documents the food
  detail endpoint and API key. [USDA downloadable datasets](https://fdc.nal.usda.gov/download-datasets/)
  are an alternative to the API for larger projects.
- [NIH potassium fact sheet](https://ods.od.nih.gov/factsheets/potassium-healthprofessional/)
  explains age/sex-specific adequate intakes and why no general dietary upper
  level was set for healthy people with normal kidney function.
- [National Academies sodium and potassium DRI report](https://www.nationalacademies.org/read/25353/chapter/2)
  describes the adult sodium AI of 1,500 mg/day and CDRR reference of
  2,300 mg/day. These are distinct concepts, not a proven athlete-specific
  optimal range.
- [Dietary Guidelines for Americans](https://www.dietaryguidelines.gov/)
  gives broader dietary context. It does not validate the illustrative
  squash-player targets in this project.

Values in [`config.json`](config.json) for calories, protein, carbohydrate,
fat, and the diversity reward are **illustrative experiment choices**. The
3,000 mg potassium reference follows the original project and is not a
universal adult target. Edit this configuration to match the population being
studied before interpreting menus as real dietary plans. Food variety is a
model preference, not a claim that every one of these eight foods is required.

## Run from Jupyter on a server

Upload the **two individual files** from
[`nutrition_qaoa_upload`](../nutrition_qaoa_upload) to the same directory in
Jupyter: `start_here.ipynb` and `nutrition_qaoa_extended.tar.gz`. That local
upload directory contains only these two files. You do not need to upload a
folder. The first notebook cell validates the archive,
unpacks it into a versioned project directory, and checks that `data/foods.csv`
and the other required inputs are present before installing anything. Always
upload the current notebook and archive together after updating the project.

Use a Python 3.11 or 3.12 **Jupyter kernel**. In the notebook, use **Kernel →
Change Kernel** to select it; a Python 3.11 installation elsewhere on the
server does not automatically change the active kernel. The first cell prints
the kernel's actual Python version and executable path. Python 3.14 is not a
supported GPU environment for the pinned Aer wheel. Request **4 vCPUs and 16 GB RAM**
for the default notebook run, with **one available compatible NVIDIA GPU** if
you want GPU simulation. The code uses only the first GPU visible to the job.
The default scientific comparison runs 4 depths × 3 independent QAOA seeds ×
3 random starts × 2 optimizers = **72 optimizations**, each capped at 400
objective evaluations. Plan a **4-hour Jupyter allocation** for a first CPU
run; actual time depends on the server. A GPU may or may not help this 16-qubit
case. For a fast smoke check, change the notebook settings first, but do not
present that shortened run as the full comparison.
Do not request a particular card model solely for
this 16-qubit experiment: A100, L40S, A40, V100 and P100 all have far more
VRAM than the statevector itself needs. The 16-qubit experiment is run
sequentially across depth/optimizer/restart settings, so requesting many more
CPUs will not give a proportional speed-up. For several notebooks in parallel
or a larger problem, start with 8 vCPUs and 32 GB RAM. GPU acceleration is
optional; the ordinary requirements file installs CPU-compatible packages.

In your server's Jupyter file browser, open `start_here.ipynb` and click
**Run All**. It installs the
packages from [`requirements.txt`](requirements.txt) into that kernel's Python
environment, runs the tests, the exact brute-force comparison, and QAOA, then
displays the results. The first installation requires internet access. The
food table is inside the archive, so no API key or extra data download is needed.
Edit the notebook's experiment settings cell if you want a different experiment
or an analysis-only run. `DEVICE = 'AUTO'` checks the CUDA driver in
the current Jupyter job (not only the presence of `nvidia-smi`). If it sees a
usable GPU, it installs the Aer 0.17 CUDA 11 GPU wheel and executes a tiny
smoke test; otherwise it logs the reason and continues on CPU. Set
`DEVICE = 'GPU'` to require a GPU and stop rather than fall back; set
`DEVICE = 'CPU'` to force a CPU run. This code cannot assign a GPU to an
already-running CPU-only Jupyter job.

The CUDA version shown by `nvidia-smi` is the driver's supported maximum,
not necessarily the CUDA runtime used by Aer. The CUDA 11 wheel may work with
a newer driver; only a passing GPU smoke test confirms this on your node.
The [CUDA 11 Aer 0.17 wheels](https://pypi.org/project/qiskit-aer-gpu-cu11/0.17.2/#files)
are available on PyPI, while the
[CUDA 12 package](https://pypi.org/project/qiskit-aer-gpu/#files) does not
provide the pinned Aer 0.17 version there.

Every **Run All** creates a separate `outputs/runs/<run_id>/` directory. The
notebook streams package installation, GPU checks, tests and experiment output
to the screen and to `notebook.log`. Both the notebook and experiment print a
heartbeat about every 30 seconds during a silent step. If a cell fails or the
kernel restarts, run the notebook's last **inspect diagnostics** cell by itself;
it can find the latest saved run without relying on notebook variables.

| Diagnostic file | What to check |
|---|---|
| `notebook.log` | Package installation, GPU smoke test, test output, full Python traceback, child exit code |
| `status.json` | Current state and phase, last operation, time since meaningful progress, process RAM usage on Linux, error traceback on failure |
| `progress.jsonl` | Timestamped history, optimizer evaluation start/done, periodic heartbeats |
| `environment.json`, `packages.json`, `gpu_probe.json`, `requested_settings.json` | Reproduce Python, package, CUDA driver, GPU allocation and experiment settings |
| `gpu_queue_estimates.json` (when available) | Raw `sbatch --test-only` responses for several GPU card constraints; not an allocation |

A recent heartbeat means the process was alive when it was written; it does
not prove that the current estimator call is making progress. If the last
event is `estimator_eval_start` and no matching completion follows, that call
was the last observed step. If heartbeats also stop, investigate the job or
node logs for a killed process, time limit or out-of-memory event. The
`qaoa_runs.csv`, `qaoa_seed_summary.csv`, `qaoa_summary.csv`, and shot-sweep
files are checkpointed after each completed depth/optimizer/seed/restart
combination, so earlier results survive a
later failure.

The prebuilt Qiskit Aer GPU packages target x86-64 Linux. The CUDA version and
driver must be compatible; see the
[Qiskit Aer installation guidance](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html)
and [NVIDIA's CUDA architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).
The notebook checks the actual environment instead of excluding older cards
based on VRAM or bfloat16 capability. GPU simulation only accelerates the
quantum statevector calculations; the brute-force baseline and optimizer
control remain on CPU. On a 16-qubit case, setup overhead can offset GPU
acceleration, so compare measured run times.

If you use Slurm, set `SLURM_ACCOUNT` in the experiment settings cell to your
allocation account if it is not already present in `SLURM_JOB_ACCOUNT` or
`SBATCH_ACCOUNT`. When no GPU is visible, the notebook runs your
`sbatch --test-only` pattern for A100 80 GB, A100, A40, L40S, and V100 with
`--ntasks=1` and a
four-hour request. It records the **raw** estimated-start responses, then
continues on CPU in `AUTO` mode. The estimate is for a *new* GPU job; it does
not submit one or give this Jupyter kernel access to that GPU. An estimate
may be unavailable or change as the queue changes. See
[Slurm's scheduling note](https://slurm.schedmd.com/SLUG23/Field-Notes-7.pdf).

### Optional USDA download

The checked-in `data/foods.csv` is sufficient for all experiments; no download
is required. To obtain the source records, get a key through the
[USDA API guide](https://fdc.nal.usda.gov/api-guide/), set `FDC_API_KEY` in the
Jupyter server environment, and change `DOWNLOAD_USDA` to `True` in the
notebook. Do not save the key in the notebook. The same optional step can be
run from a terminal:

```bash
export FDC_API_KEY='your-key'          # Linux/macOS
python fetch_usda.py
```

PowerShell: `$env:FDC_API_KEY = 'your-key'`. The script reads
[`data/food_manifest.csv`](data/food_manifest.csv), writes raw responses and a
candidate CSV to `data/usda_download/`, and never overwrites `data/foods.csv`.
Keep the API key outside the repository.

## Command-line alternative

Quick checks and baseline, without Qiskit execution:

```bash
python test_model.py
python run_experiments.py --analysis-only
```

QAOA on the complete 16-qubit problem (same design as the notebook):

```bash
python run_experiments.py --depths 1 2 3 4 --qaoa-seeds 1 7 21 --restarts 3 --optimizers COBYLA Nelder-Mead --max-evaluations 400 --shots 4096 --shot-counts 1024 4096 16384
```

Optional larger study (requires substantially more time):

```bash
python run_experiments.py \
  --depths 1 2 3 4 5 \
  --qaoa-seeds 1 7 21 42 100 \
  --restarts 5 \
  --optimizers COBYLA Powell Nelder-Mead \
  --max-evaluations 600 \
  --shots 4096 --shot-counts 1024 4096 16384 \
  --device GPU
```

The command-line program writes to `outputs/` by default, whereas the notebook
passes its unique run directory through `--output`. The program also accepts
`--config` and `--data` paths. CLI output is JSON and progress lines; the
durable result files are:

| File | Contents |
|---|---|
| `baseline_summary.json` | Exact true and QUBO-best menus, natural portions, nutrient totals |
| `surrogate_validation.json` | Approximation error, rank agreement, best-menu mismatch |
| `surrogate_fit_comparison.csv` | Global versus low-cost-weighted QUBO fits; near-optimum error and true ranks |
| `diversity_sweep.csv` | Exact best menu at several variety weights |
| `sensitivity.csv` | Exact optimum under ±10% target/bound perturbations |
| `baseline_analysis.png` | Model and sensitivity plots |
| `qaoa_runs.csv` | Every depth, optimizer, independent seed, and restart; exact distribution metrics |
| `qaoa_seed_summary.csv` | Within-seed means, standard deviations, and best starts |
| `qaoa_summary.csv` | Means and sample standard deviations **across seed means**; blank SD until two seeds complete |
| `qaoa_shot_sweep.csv`, `qaoa_shot_sweep_summary.csv` | Nested 1,024/4,096/16,384-shot samples and observed-best ranks |
| `qaoa_summary.png` | Probability and diversity summaries |

QAOA metrics include probability of the true optimum, probability of the true
top 1%, probability of landing within 0.001 of the true optimum, probability
of the QUBO optimum, both QUBO and true expected cost, expected portion count, expected
number of distinct foods, and best rank from finite shots. The QUBO surrogate
is assessed separately, so a mismatch in modelling is not attributed to QAOA.
The exact-optimum mask uses zero relative tolerance; the previous near-optimum
overcount is fixed. The best finite-shot menu is selected afterward with the
true objective, so it is an oracle-assisted diagnostic, not the optimizer's
direct output. Policy-sensitivity repetitions are not substitutes for QAOA
seeds: `SEED` controls that separate analysis, and `QAOA_SEEDS` control
independent QAOA initializations and shot draws.

The default experiment uses exact statevector probabilities and then draws
finite samples. It does not simulate hardware noise. No speed advantage is
claimed from classical Qiskit simulation.
