"""
Incentivized Federated K-means with Shapley Values
Core algorithms: data split, federated k-means, utility, exact and Monte Carlo Shapley.
Quick demo (small sklearn digits dataset, a few seconds):
      python fed_kmeans_shapley.py --data digits
Full MNIST experiments for the report: see run_experiments.py and README.md
"""
import argparse, itertools, math, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import normalized_mutual_info_score as nmi, roc_auc_score
from sklearn.model_selection import train_test_split

K = 10
OUT = "demo_figures"   # quick-demo output; report figures go to figures/


# ---------------------------------------------------------------- Step 1: data
def load_data(name, seed=0):
    if name == "digits":
        X, y = load_digits(return_X_y=True)
    else:  # MNIST train set (60k): local npz cache if present, else download from OpenML
        here = os.path.dirname(os.path.abspath(__file__))
        cands = [os.path.join(here, "data", "mnist.npz"), os.path.join(here, "..", "data", "mnist.npz")]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path:
            d = np.load(path); X, y = d["X"], d["y"]
        else:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
            X, y = X[:60000], y[:60000].astype(int)   # the standard training set
    X = X.astype(float) / X.max()
    # three-way split: clients' data / validation (for Shapley) / test (final claims)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
    pca = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit(Xtr)
    f = lambda A: pca.transform(A).astype(np.float32)
    return f(Xtr), ytr, f(Xval), yval, f(Xte), yte


def dirichlet_split(y, n_clients, alpha, rng, min_size=20):
    """Label-skewed split: small alpha -> each client holds few classes."""
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            ids = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(ids, cuts)):
                parts[i] += chunk.tolist()
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]


# ------------------------------------------------- Step 2: federated k-means
def assign(X, C):
    # argmin_k ||x - c_k||^2 = argmin_k (||c_k||^2 - 2 x.c_k); ||x||^2 is constant per row
    return ((C ** 2).sum(1)[None] - 2 * X @ C.T).argmin(1)


def local_lloyd(X, C, L):
    """L Lloyd steps on one client. Empty cluster -> keep received centre, count 0."""
    C = C.copy()
    for _ in range(L):
        z = assign(X, C)
        counts = np.bincount(z, minlength=K)
        onehot = np.zeros((len(X), K), dtype=X.dtype)
        onehot[np.arange(len(X)), z] = 1
        sums = onehot.T @ X
        nz = counts > 0
        C[nz] = sums[nz] / counts[nz, None]
    return C, counts


def fed_kmeans(clients, C0, rounds=10, L=3, Xeval=None, yeval=None):
    """LocalKMeans: L local steps, then count-weighted averaging on the server."""
    C, hist = C0.copy(), []
    for _ in range(rounds):
        if not clients:
            break
        num, den = np.zeros_like(C), np.zeros(K)
        for X in clients:
            Ci, ni = local_lloyd(X, C, L)
            num += ni[:, None] * Ci
            den += ni
        C[den > 0] = num[den > 0] / den[den > 0, None]
        if Xeval is not None:
            hist.append(nmi(yeval, assign(Xeval, C)))
    return (C, hist) if Xeval is not None else C


def evaluate(C, X, y):
    return nmi(y, assign(X, C))


# ------------------------------------------------- Step 3: utility + Shapley
class Game:
    """U(S) = NMI on validation set of fed k-means trained on clients in S (cached)."""
    def __init__(self, clients, C0, Xval, yval, rounds=10, L=3):
        self.clients, self.C0, self.Xval, self.yval = clients, C0, Xval, yval
        self.rounds, self.L, self.cache = rounds, L, {}

    def U(self, S):
        key = frozenset(S)
        if key not in self.cache:          # U(empty) = quality of the initial centres, NOT 0
            C = fed_kmeans([self.clients[i] for i in sorted(key)], self.C0, self.rounds, self.L)
            self.cache[key] = evaluate(C, self.Xval, self.yval)
        return self.cache[key]


def exact_shapley(g, n):
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                phi[i] += w * (g.U(S + (i,)) - g.U(S))
    return phi


def tmc_shapley(g, n, n_perms, rng, tol=0.0):
    """Truncated Monte Carlo (Ghorbani & Zou 2019). Returns final phi and running history."""
    phi, hist, full = np.zeros(n), [], g.U(range(n))
    for m in range(1, n_perms + 1):
        S, prev = [], g.U(())
        for i in rng.permutation(n):
            S.append(i)
            cur = prev if abs(full - prev) < tol else g.U(S)
            phi[i] += cur - prev
            prev = cur
        hist.append(phi / m)
    return phi / n_perms, np.array(hist)


def gini(x):
    x = np.sort(np.clip(x, 0, None))
    return (2 * np.arange(1, len(x) + 1) - len(x) - 1) @ x / (len(x) * x.sum())


# ------------------------------------------------------------ helpers
def init_centres(Xval, seed):
    """Shared k-means++ init from a small public (unlabelled) sample."""
    rng = np.random.default_rng(seed)
    sample = Xval[rng.choice(len(Xval), min(500, len(Xval)), replace=False)]
    C0, _ = kmeans_plusplus(sample, K, random_state=seed)
    return C0.astype(np.float32)


def savefig(name):
    os.makedirs(OUT, exist_ok=True)
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150); plt.close()


# ------------------------------------------------------------ experiments
def step0_stability(Xtr, ytr, Xval, yval, seeds=(0, 1, 2)):
    rng = np.random.default_rng(0)
    clients = [Xtr[i] for i in dirichlet_split(ytr, 10, 100.0, rng)]
    vals = [evaluate(fed_kmeans(clients, init_centres(Xval, s)), Xval, yval) for s in seeds]
    print(f"[Step 0] U(all) across init seeds: {np.round(vals, 3)}  std={np.std(vals):.4f}")
    print("         If std > ~0.02, raise rounds or fix one seed before trusting Shapley.")


def fig1_reproduction(Xtr, ytr, Xval, yval, C0, rounds=20):
    rng = np.random.default_rng(0)
    curves = {}
    _, curves["Centralized (L=1)"] = fed_kmeans([Xtr], C0, rounds, 1, Xval, yval)
    for label, a in [("Federated IID", 100.0), ("Federated skew a=0.1", 0.1)]:
        cl = [Xtr[i] for i in dirichlet_split(ytr, 10, a, rng)]
        _, curves[label] = fed_kmeans(cl, C0, rounds, 3, Xval, yval)
    cl = [Xtr[i] for i in dirichlet_split(ytr, 10, 0.1, rng)]
    local = [fed_kmeans([X], C0, rounds, 3, Xval, yval)[1] for X in cl]
    curves["Local only (avg)"] = [float(v) for v in np.mean(local, 0)]
    for k, v in curves.items():
        plt.plot(range(1, len(v) + 1), v, label=k)
    plt.xlabel("Round"); plt.ylabel("NMI (validation)"); plt.legend()
    savefig("fig1_reproduction")
    print("[Fig 1] final NMI:", {k: round(v[-1], 3) for k, v in curves.items()})


def expA_size(Xtr, ytr, Xval, yval, C0):
    """10 IID clients; clients 6-7 get 2x data, 8-9 get 4x."""
    rng = np.random.default_rng(1)
    w = np.array([1] * 6 + [2] * 2 + [4] * 2, float)
    ids = rng.permutation(len(Xtr))
    cuts = (np.cumsum(w / w.sum()) * len(ids)).astype(int)[:-1]
    clients = [Xtr[p] for p in np.split(ids, cuts)]
    g = Game(clients, C0, Xval, yval)
    phi = exact_shapley(g, 10)
    print(f"[Exp A] efficiency: sum(phi)={phi.sum():.4f}  U(all)-U(0)={g.U(range(10)) - g.U(()):.4f}")
    plt.bar(range(10), phi, color=["C0"] * 6 + ["C1"] * 2 + ["C2"] * 2)
    plt.xlabel("Client (blue 1x, orange 2x, green 4x data)"); plt.ylabel("Shapley value")
    savefig("expA_size")
    # TMC vs exact (reuses cached utilities, so almost free)
    _, hist = tmc_shapley(g, 10, 200, np.random.default_rng(0))
    err = np.abs(hist - phi).max(1)
    plt.plot(range(1, len(err) + 1), err); plt.xlabel("Permutations"); plt.ylabel("max |TMC - exact|")
    savefig("fig5_tmc_vs_exact")


def expB_freeriders(Xtr, ytr, Xval, yval, Xte, yte, C0):
    """6 honest, 2 feature-shuffled, 1 duplicate, 1 noise."""
    rng = np.random.default_rng(2)
    parts = [Xtr[p] for p in dirichlet_split(ytr, 9, 100.0, rng)]
    honest = parts[:6]
    shuffled = [rng.permuted(parts[6], axis=1), rng.permuted(parts[7], axis=1)]
    dup = [honest[0].copy()]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    noise = [rng.normal(mu, sd, size=parts[8].shape)]
    clients = honest + shuffled + dup + noise
    types = ["honest"] * 6 + ["shuffled"] * 2 + ["duplicate"] + ["noise"]
    g = Game(clients, C0, Xval, yval)
    phi = exact_shapley(g, 10)

    colors = {"honest": "C2", "shuffled": "C3", "duplicate": "C1", "noise": "C7"}
    plt.bar(range(10), phi, color=[colors[t] for t in types])
    plt.axhline(0, color="k", lw=0.5)
    for t, c in colors.items():
        plt.bar(0, 0, color=c, label=t)
    plt.legend(); plt.xlabel("Client"); plt.ylabel("Shapley value")
    savefig("fig2_freeriders")

    garbage = np.array([t in ("shuffled", "noise") for t in types])
    auroc = roc_auc_score(~garbage, phi)
    print(f"[Exp B] phi = {np.round(phi, 4)}")
    print(f"[Exp B] AUROC honest vs garbage (duplicate counted as honest-looking): {auroc:.3f}")
    print(f"[Exp B] duplicate phi={phi[8]:.4f} vs its original phi={phi[0]:.4f} (symmetry)")

    # Filtered retraining, judged on the held-out TEST set (not validation)
    keep = [i for i in range(10) if phi[i] > 0.1 * phi.max()]
    all_nmi = evaluate(fed_kmeans(clients, C0), Xte, yte)
    kept_nmi = evaluate(fed_kmeans([clients[i] for i in keep], C0), Xte, yte)
    print(f"[Exp B] test NMI: all clients={all_nmi:.3f}  phi-filtered {keep}={kept_nmi:.3f}")
    plt.bar(["All clients", "Shapley-filtered"], [all_nmi, kept_nmi], color=["C7", "C2"])
    plt.ylabel("NMI (held-out test)")
    savefig("fig3_filtered_retraining")


def expC_skew(Xtr, ytr, Xval, yval, C0, alphas=(0.05, 0.1, 0.5, 1, 10)):
    ginis, n_cls_all, phi_all = [], [], []
    for a in alphas:
        rng = np.random.default_rng(3)
        idx = dirichlet_split(ytr, 10, a, rng)
        g = Game([Xtr[i] for i in idx], C0, Xval, yval)
        phi = exact_shapley(g, 10)
        n_cls = [np.sum(np.bincount(ytr[i], minlength=K) / len(i) >= 0.05) for i in idx]
        ginis.append(gini(phi)); n_cls_all += n_cls; phi_all += list(phi)
        print(f"[Exp C] alpha={a:<5} U(all)={g.U(range(10)):.3f}  Gini(phi)={ginis[-1]:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(alphas, ginis, "o-"); ax[0].set_xscale("log")
    ax[0].set_xlabel("Dirichlet alpha"); ax[0].set_ylabel("Gini of Shapley values")
    ax[1].scatter(n_cls_all, phi_all, s=12)
    ax[1].set_xlabel("# classes held by client"); ax[1].set_ylabel("Shapley value")
    savefig("fig4_skew")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="digits", choices=["digits", "mnist"])
    args = ap.parse_args()
    Xtr, ytr, Xval, yval, Xte, yte = load_data(args.data)
    C0 = init_centres(Xval, seed=0)
    step0_stability(Xtr, ytr, Xval, yval)
    fig1_reproduction(Xtr, ytr, Xval, yval, C0)
    expA_size(Xtr, ytr, Xval, yval, C0)
    expB_freeriders(Xtr, ytr, Xval, yval, Xte, yte, C0)
    expC_skew(Xtr, ytr, Xval, yval, C0)
    print(f"Figures saved in ./{OUT}/")
