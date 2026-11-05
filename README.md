# Incentivized Federated K-means: Valuing Clients with Shapley Values

CS6007 Multi-Agent Machine Learning project, IIT Bombay.
Team: Aryan Mahajan (23B1076), Yougank Kumar (23B0919).

We run federated k-means (LocalKMeans) on MNIST with 10 clients and give each client a
Shapley value: its fair share of the credit for the final clustering. We then check whether
these values catch free-riders, whether removing low-value clients helps, and how uneven
data across clients changes who gets the credit.

## Files

| File | What it does |
|---|---|
| `fed_kmeans_shapley.py` | Core code: data loading and Dirichlet split, federated k-means, utility U(S), exact and Monte Carlo Shapley values. Also a quick demo. |
| `run_experiments.py` | Runs every experiment in the report on MNIST and saves all numbers to `results/results.json`. |
| `make_figures.py` | Rebuilds every report figure from `results/results.json` into `figures/`. |
| `make_slide_figures.py` | Rebuilds the presentation figures into `slide_figures/`. |
| `format_extras.py` | Builds the extra presentation tables and figures (federated vs centralized by alpha, client-by-digit heatmap, MNIST PCA plot); needs MNIST and about 1 minute. |
| `prepare_mnist.py` | Optional: builds the MNIST file by hand if the automatic download fails. |
| `results/results.json` | Saved results from our run (all numbers in the report). |
| `requirements.txt` | Python packages. |

## Setup

Python 3.9 or newer. CPU only, no GPU needed.

```bash
pip install -r requirements.txt
```

## How to run

**1. Quick check (a few seconds).** Runs the whole pipeline on the small sklearn digits dataset:

```bash
python fed_kmeans_shapley.py --data digits     # figures -> demo_figures/
```

**2. Full experiments on MNIST (about 25 minutes on a 2-core CPU).**

```bash
python run_experiments.py
```

It prints a line as each experiment finishes (e.g. `done B_0`) and ends with `all done`.
All numbers are saved to `results/results.json`.

**3. Figures.**

```bash
python make_figures.py          # report figures   -> figures/
python make_slide_figures.py    # slide figures    -> slide_figures/
python format_extras.py         # extra slide tables and figures (needs MNIST)
```

`results/results.json` from our run is already included, so step 3 works without step 2.
All random seeds are fixed, so re-running step 2 gives the same numbers.

## Dataset

MNIST is **not** included in this zip. On the first run, `fed_kmeans_shapley.py` downloads
the 60,000 MNIST training images automatically from OpenML (`mnist_784`):
https://www.openml.org/d/554

If that download is blocked on your network:

1. Download `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` from
   https://github.com/fgnt/mnist into a folder `data/` next to the scripts.
2. Run `python prepare_mnist.py`. This creates `data/mnist.npz`, which the code then uses.

## Experiments (what `run_experiments.py` does)

| Key in results.json | Experiment |
|---|---|
| `stability` | Utility of all clients under 3 different starting centres |
| `reproduction` | Federated vs centralized vs local-only k-means; effect of local steps L |
| `A_0..A_2` | Data size: clients with 1x, 2x and 4x data |
| `B_0..B_2` | Free-riders: 6 honest, 2 scrambled, 1 copy, 1 noise client |
| `C_<alpha>_<seed>` | Uneven data: Dirichlet alpha in {0.05, 0.1, 0.5, 1, 10} |

Each Shapley experiment computes exact values over all 2^10 = 1024 client groups and is
repeated on 3 random data splits.

## GitHub

No public repository; all code is in this folder.
