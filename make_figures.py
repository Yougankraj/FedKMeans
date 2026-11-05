"""Builds report figures from results/results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
OUT = "figures"; os.makedirs(OUT, exist_ok=True)
SEEDS = (0, 1, 2); ALPHAS = (0.05, 0.1, 0.5, 1.0, 10.0)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
})


def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------- Fig 1
rep = R["reproduction"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cols = [INK2, BLUE, ORANGE, YELLOW]
for (k, v), c in zip(rep["curves"].items(), cols):
    ax[0].plot(range(1, len(v) + 1), v, color=c, label=k, ls="--" if k == "Centralized" else "-")
ax[0].set_xlabel("Communication round (3 Lloyd steps each)"); ax[0].set_ylabel("NMI (validation)")
ax[0].set_title("(a) Federated vs centralized vs local", loc="left", fontsize=10)
ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
for k, v in rep["L_curves"].items():
    L = int(k.split("L=")[1]); even = k.startswith("even")
    ax[1].plot(range(1, len(v) + 1), v, color=BLUE if even else ORANGE,
               ls={1: "-", 3: "--", 10: ":"}[L], label=k)
ax[1].set_xlabel("Total Lloyd iterations"); ax[1].set_ylabel("NMI (validation)")
ax[1].set_title("(b) Effect of local steps L (sync every L iterations)", loc="left", fontsize=10)
ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
save("fig1_reproduction")

# ---------------------------------------------------------------- Fig 2 (free-riders)
B = [R[f"B_{s}"] for s in SEEDS]
types = B[0]["types"]
phi = np.array([b["phi"] for b in B])
tcol = {"honest": BLUE, "duplicate": AQUA, "shuffled": ORANGE, "noise": YELLOW}
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.bar(range(10), phi.mean(0), yerr=phi.std(0), color=[tcol[t] for t in types],
       width=0.7, capsize=3, error_kw={"ecolor": INK2, "lw": 1})
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(10), [f"C{i+1}" for i in range(10)])
ax.set_ylabel("Shapley value (NMI points)")
for t, c in tcol.items():
    ax.bar(0, 0, color=c, label=t)
ax.legend(ncol=4, fontsize=9, loc="upper right")
save("fig2_freeriders")

# ---------------------------------------------------------------- Fig 3 (filtering)
keys = [("all", "All 10 clients", INK2), ("shapley_filtered", "Shapley-filtered", BLUE),
        ("oracle_honest", "Honest only (oracle)", AQUA)]
vals = np.array([[b["test_nmi"][k] for k, _, _ in keys] for b in B])
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.bar(range(3), vals.mean(0), yerr=vals.std(0), color=[c for *_, c in keys], width=0.6, capsize=4,
       error_kw={"ecolor": INK2, "lw": 1})
for i, v in enumerate(vals.mean(0)):
    ax.text(i, v + vals.std(0)[i] + 0.005, f"{v:.3f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(3), [l for _, l, _ in keys])
ax.set_ylabel("NMI on held-out test set"); ax.set_ylim(0, vals.max() * 1.18)
save("fig3_filtered_retraining")

# ---------------------------------------------------------------- Fig 4 (data size)
A = [R[f"A_{s}"] for s in SEEDS]
mult = np.array([1] * 6 + [2] * 2 + [4] * 2)
pa = np.array([a["phi"] for a in A])
grp = {m: pa[:, mult == m].mean(1) for m in (1, 2, 4)}
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.bar(range(3), [grp[m].mean() for m in (1, 2, 4)], yerr=[grp[m].std() for m in (1, 2, 4)],
       color=BLUE, width=0.55, capsize=4, error_kw={"ecolor": INK2, "lw": 1})
base = grp[1].mean()
ax.plot(range(3), [base, 2 * base, 4 * base], color=INK2, ls="--", lw=1.2, marker="o", ms=4,
        label="if value grew in proportion to data")
ax.set_xticks(range(3), ["1x data", "2x data", "4x data"])
ax.set_ylabel("Mean Shapley value per client"); ax.legend(fontsize=8, loc="upper left")
save("fig4_data_size")

# ---------------------------------------------------------------- Fig 5 (skew)
G = np.array([[R[f"C_{a}_{s}"]["gini"] for s in SEEDS] for a in ALPHAS])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].errorbar(ALPHAS, G.mean(1), yerr=G.std(1), color=BLUE, marker="o", ms=6, capsize=4)
ax[0].set_xscale("log"); ax[0].set_xlabel("Dirichlet alpha (small = more uneven data)")
ax[0].set_ylabel("Gini of Shapley values"); ax[0].set_title("(a) Inequality of credit", loc="left", fontsize=10)
for a, c, lab in [(0.1, ORANGE, "alpha = 0.1"), (10.0, BLUE, "alpha = 10")]:
    xs, ys = [], []
    for s in SEEDS:
        r = R[f"C_{a}_{s}"]; xs += r["n_classes"]; ys += r["phi"]
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(xs))
    ax[1].scatter(np.array(xs) + jit, ys, color=c, s=28, alpha=0.85, edgecolor="white", lw=0.8, label=lab)
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].set_xlabel("Number of digit classes a client holds"); ax[1].set_ylabel("Shapley value")
ax[1].set_title("(b) Credit vs client diversity", loc="left", fontsize=10); ax[1].legend(fontsize=8)
save("fig5_skew")

# ---------------------------------------------------------------- Fig 6 (TMC vs exact)
err = R["A_0"]["tmc_err"]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(range(1, len(err) + 1), err, color=BLUE)
ax.set_yscale("log"); ax.set_xlabel("Number of sampled permutations")
ax.set_ylabel("max |Monte Carlo - exact|")
save("fig6_tmc_vs_exact")
print("figures written")
