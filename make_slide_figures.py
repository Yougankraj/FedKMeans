"""Builds large single-panel figures for the presentation from results/results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
OUT = "slide_figures"; os.makedirs(OUT, exist_ok=True)
SEEDS = (0, 1, 2); ALPHAS = (0.05, 0.1, 0.5, 1.0, 10.0)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2.5, "savefig.dpi": 200,
})


def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close()


# 1. federated vs centralized vs local
rep = R["reproduction"]["curves"]
fig, ax = plt.subplots(figsize=(8, 4.6))
style = {"Centralized": (INK2, "--"), "Federated, even data": (BLUE, "-"),
         "Federated, uneven data (a=0.1)": (ORANGE, "-"), "Local only (avg of clients)": (YELLOW, "-")}
label = {"Centralized": "Centralized", "Federated, even data": "Federated, even data",
         "Federated, uneven data (a=0.1)": "Federated, uneven data",
         "Local only (avg of clients)": "Each client alone"}
for k, v in rep.items():
    c, ls = style[k]
    ax.plot(range(1, len(v) + 1), v, color=c, ls=ls)
    dy = {"Centralized": 0.009, "Federated, even data": -0.009}.get(k, 0)
    ax.text(len(v) + 0.4, v[-1] + dy, label[k], color=c, va="center", fontsize=13)
ax.set_xlim(1, 29); ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel("Communication round"); ax.set_ylabel("Clustering quality (NMI)")
save("s_reproduction")

# 2. free-riders
B = [R[f"B_{s}"] for s in SEEDS]
types = B[0]["types"]
phi = np.array([b["phi"] for b in B])
tcol = {"honest": BLUE, "duplicate": AQUA, "shuffled": ORANGE, "noise": YELLOW}
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.bar(range(10), phi.mean(0), yerr=phi.std(0), color=[tcol[t] for t in types], width=0.7,
       capsize=4, error_kw={"ecolor": INK2, "lw": 1.2})
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(10), [f"C{i+1}" for i in range(10)])
ax.set_ylabel("Shapley value (credit)")
for t, c in [("honest", BLUE), ("copy of C1", AQUA), ("scrambled", ORANGE), ("noise", YELLOW)]:
    ax.bar(0, 0, color=c, label=t)
ax.legend(ncol=4, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.13))
save("s_freeriders")

# 3. filtering
keys = [("all", "All 10 clients", INK2), ("shapley_filtered", "Remove low-credit\nclients", BLUE),
        ("oracle_honest", "Honest only\n(oracle)", AQUA)]
vals = np.array([[b["test_nmi"][k] for k, _, _ in keys] for b in B])
fig, ax = plt.subplots(figsize=(7.2, 3.6))
m, sd = vals.mean(0), vals.std(0)
for i, ((_, lab, c), v, e) in enumerate(zip(keys, m, sd)):
    y = 2 - i
    ax.errorbar(v, y, xerr=e, fmt="o", color=c, ms=13, capsize=5, elinewidth=1.5)
    ax.text(v + 0.0012, y + 0.18, f"{v:.3f}", fontsize=14, color=INK)
ax.set_yticks([2, 1, 0], [l.replace("\n", " ") for _, l, _ in keys])
ax.set_ylim(-0.6, 2.6); ax.set_xlim(0.465, 0.487)
ax.set_xlabel("Test NMI (higher is better)"); ax.grid(axis="y", visible=False)
save("s_filtering")

# 4. uneven data -> concentrated credit
G = np.array([[R[f"C_{a}_{s}"]["gini"] for s in SEEDS] for a in ALPHAS])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.errorbar(ALPHAS, G.mean(1), yerr=G.std(1), color=BLUE, marker="o", ms=8, capsize=5)
ax.set_xscale("log"); ax.set_xticks(ALPHAS, ["0.05", "0.1", "0.5", "1", "10"])
ax.set_xlabel(r"Dirichlet $\alpha$   (more uneven $\leftarrow$  $\rightarrow$ more even)")
ax.set_ylabel("Inequality of credit (Gini)")
save("s_skew")
print("slide figures written")

# 5. effect of local steps L (uneven vs even)
Lc = R["reproduction"]["L_curves"]
fig, ax = plt.subplots(figsize=(8, 4.4))
for k, v in Lc.items():
    L = int(k.split("L=")[1]); even = k.startswith("even")
    ax.plot(range(1, len(v) + 1), v, color=BLUE if even else ORANGE, ls={1: "-", 3: "--", 10: ":"}[L])
for k, v in Lc.items():
    if k.startswith("uneven") and not k.endswith("L=1"):
        ax.text(len(v) + 0.5, v[-1], f"uneven, L={k.split('L=')[1]}", color=ORANGE, va="center", fontsize=12)
ax.text(30.5, 0.487, "L=1 (both),\neven data (any L)", color=INK2, va="center", fontsize=12)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color=INK2, ls=l) for l in ("-", "--", ":")], ["L = 1", "L = 3", "L = 10"],
          fontsize=11, loc="center", bbox_to_anchor=(0.55, 0.47), ncol=3)
ax.set_xlim(1, 39); ax.set_xticks([1, 10, 20, 30])
ax.set_xlabel("Total Lloyd iterations (sync every L)"); ax.set_ylabel("NMI (validation)")
save("s_local_steps")

# 6. data size
A = [R[f"A_{s}"] for s in SEEDS]
mult = np.array([1] * 6 + [2] * 2 + [4] * 2)
pa = np.array([a["phi"] for a in A])
grp = {m: pa[:, mult == m].mean(1) for m in (1, 2, 4)}
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.bar(range(3), [grp[m].mean() for m in (1, 2, 4)], yerr=[grp[m].std() for m in (1, 2, 4)],
       color=BLUE, width=0.55, capsize=5, error_kw={"ecolor": INK2, "lw": 1.2})
base = grp[1].mean()
ax.plot(range(3), [base, 2 * base, 4 * base], color=INK2, ls="--", lw=1.5, marker="o", ms=6)
ax.text(2.05, 4 * base, "if credit were\nproportional to data", fontsize=12, color=INK2, va="center")
ax.set_xlim(-0.5, 3.3)
ax.set_xticks(range(3), ["1x data", "2x data", "4x data"]); ax.set_ylabel("Mean Shapley value")
save("s_data_size")

# 7. Monte Carlo vs exact
err = R["A_0"]["tmc_err"]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(range(1, len(err) + 1), err, color=BLUE)
ax.set_yscale("log"); ax.set_xlabel("Number of sampled orders M")
ax.set_ylabel(r"max$_i$ |MC estimate $-$ exact|")
save("s_mc_vs_exact")

# 8. credit vs number of classes held
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for a, c, lab in [(0.1, ORANGE, r"$\alpha$ = 0.1 (uneven)"), (10.0, BLUE, r"$\alpha$ = 10 (even)")]:
    xs, ys = [], []
    for s in SEEDS:
        r = R[f"C_{a}_{s}"]; xs += r["n_classes"]; ys += r["phi"]
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(xs))
    ax.scatter(np.array(xs) + jit, ys, color=c, s=45, alpha=0.9, edgecolor="white", lw=1, label=lab)
ax.axhline(0, color=INK2, lw=1)
ax.set_xlabel("Digit classes a client holds"); ax.set_ylabel("Shapley value"); ax.legend(fontsize=12)
save("s_credit_vs_classes")
print("extra slide figures written")
