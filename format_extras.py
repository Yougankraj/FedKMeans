"""Extra tables/figures for the presentation (previous-year format)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fed_kmeans_shapley import load_data, dirichlet_split, fed_kmeans, evaluate, init_centres, K
from sklearn.decomposition import PCA

OUT = "slide_figures"; os.makedirs(OUT, exist_ok=True)
Xtr, ytr, Xval, yval, Xte, yte = load_data("mnist")
C0 = init_centres(Xval, seed=0)
ALPHAS = (0.05, 0.1, 0.5, 1.0, 10.0); SEEDS = (0, 1, 2)

# 1. table: test NMI by alpha for fed (L=3, L=1), centralized, local-only
cent = evaluate(fed_kmeans([Xtr], C0, 10, 3), Xte, yte)
rows = {}
for a in ALPHAS:
    f3, f1, loc = [], [], []
    for s in SEEDS:
        idx = dirichlet_split(ytr, 10, a, np.random.default_rng(300 + s))   # same splits as Experiment C
        cl = [Xtr[i] for i in idx]
        f3.append(evaluate(fed_kmeans(cl, C0, 10, 3), Xte, yte))
        f1.append(evaluate(fed_kmeans(cl, C0, 30, 1), Xte, yte))
        loc.append(np.mean([evaluate(fed_kmeans([X], C0, 10, 3), Xte, yte) for X in cl]))
    rows[a] = dict(fed_L3=float(np.mean(f3)), fed_L3_sd=float(np.std(f3)), fed_L1=float(np.mean(f1)),
                   local=float(np.mean(loc)), central=float(cent))
    print(a, {k: round(v, 3) for k, v in rows[a].items()})

# 2. per-client table (split seed 0 of Experiment C)
clients = {}
for a in (0.1, 1.0, 10.0):
    idx = dirichlet_split(ytr, 10, a, np.random.default_rng(300))
    counts = np.array([np.bincount(ytr[i], minlength=K) for i in idx])
    clients[a] = {"counts": counts.tolist(),
                  "n_classes": [int((c / c.sum() >= 0.05).sum()) for c in counts],
                  "size": [int(c.sum()) for c in counts]}
    print(a, clients[a]["n_classes"], clients[a]["size"])
json.dump({"by_alpha": {str(k): v for k, v in rows.items()},
           "clients": {str(k): v for k, v in clients.items()}}, open("results/format_extras.json", "w"))

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

# 3. MNIST in 2-D (PCA of a 5,000-image sample)
here = os.path.dirname(os.path.abspath(__file__))
npz = next((c for c in [os.path.join(here, "data", "mnist.npz"), os.path.join(here, "..", "data", "mnist.npz")]
            if os.path.exists(c)), None)
if npz:
    d = np.load(npz); Xtr_raw, ytr_raw = d["X"], d["y"]
else:
    from sklearn.datasets import fetch_openml
    Xtr_raw, ytr_raw = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    Xtr_raw, ytr_raw = Xtr_raw[:60000], ytr_raw[:60000].astype(int)
X, y = Xtr_raw[:5000] / 255.0, ytr_raw[:5000]
Z = PCA(2, random_state=0).fit_transform(X)
fig, ax = plt.subplots(figsize=(7.5, 5))
sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5, alpha=0.7)
cb = plt.colorbar(sc, ax=ax, ticks=range(10)); cb.set_label("digit")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title("MNIST (5,000 images, first two principal components)", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/s_mnist_pca.png", bbox_inches="tight"); plt.close()

# 4. client x digit heatmaps (sequential, one hue)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
vmax = max(np.max(clients[a]["counts"]) for a in clients)
for ax, a in zip(axes, (0.1, 1.0, 10.0)):
    im = ax.imshow(np.array(clients[a]["counts"]), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(rf"$\alpha$ = {a:g}", fontsize=14)
    ax.set_xticks(range(10)); ax.set_yticks(range(10), [f"C{i+1}" for i in range(10)])
    ax.set_xlabel("digit"); ax.spines[:].set_visible(False)
axes[0].set_ylabel("client")
cb = fig.colorbar(im, ax=axes, shrink=0.9); cb.set_label("images")
plt.savefig(f"{OUT}/s_client_heatmap.png", bbox_inches="tight"); plt.close()
print("done")
