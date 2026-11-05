"""
Runs every experiment for the report on MNIST and saves numbers + figures.
    python run_experiments.py            -> results/results.json, figures/*.png
Uses the building blocks in fed_kmeans_shapley.py.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from fed_kmeans_shapley import (K, load_data, dirichlet_split, fed_kmeans, evaluate, assign,
                                Game, exact_shapley, tmc_shapley, gini, init_centres)
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = sys.argv[1] if len(sys.argv) > 1 else "mnist"
SEEDS = (0, 1, 2)
ALPHAS = (0.05, 0.1, 0.5, 1.0, 10.0)
os.makedirs("results", exist_ok=True)

Xtr, ytr, Xval, yval, Xte, yte = load_data(DATA)
C0 = init_centres(Xval, seed=0)          # one shared server initialisation for everything


def hungarian_acc(C, X, y):
    cm = confusion_matrix(y, assign(X, C), labels=range(K))
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / len(y)


# ------------------------------------------------------------------ jobs
def job_stability(_):
    out = {}
    for name, a in [("iid", 100.0), ("skew_0.1", 0.1)]:
        cl = [Xtr[i] for i in dirichlet_split(ytr, 10, a, np.random.default_rng(0))]
        out[name] = [evaluate(fed_kmeans(cl, init_centres(Xval, s)), Xval, yval) for s in SEEDS]
    return "stability", out


def job_reproduction(_):
    R, out = 20, {}
    rng = np.random.default_rng(0)
    iid = [Xtr[i] for i in dirichlet_split(ytr, 10, 100.0, rng)]
    skew = [Xtr[i] for i in dirichlet_split(ytr, 10, 0.1, rng)]
    C, h = fed_kmeans([Xtr], C0, R, 3, Xval, yval);   out["Centralized"] = (h, C)  # same 3 Lloyd steps per round
    C, h = fed_kmeans(iid, C0, R, 3, Xval, yval);     out["Federated, even data"] = (h, C)
    C, h = fed_kmeans(skew, C0, R, 3, Xval, yval);    out["Federated, uneven data (a=0.1)"] = (h, C)
    loc = [fed_kmeans([X], C0, R, 3, Xval, yval) for X in skew]
    out["Local only (avg of clients)"] = ([float(v) for v in np.mean([h for _, h in loc], 0)], None)
    curves = {k: [float(v) for v in h] for k, (h, _) in out.items()}
    final = {}
    for k, (h, C) in out.items():
        if C is not None:
            final[k] = {"nmi_test": evaluate(C, Xte, yte), "acc_test": hungarian_acc(C, Xte, yte)}
        else:
            nm = [evaluate(Cl, Xte, yte) for Cl, _ in loc]; ac = [hungarian_acc(Cl, Xte, yte) for Cl, _ in loc]
            final[k] = {"nmi_test": float(np.mean(nm)), "acc_test": float(np.mean(ac))}
    # effect of L (local steps per sync) at a fixed budget of 30 Lloyd iterations
    Lcurves = {}
    for name, cl in [("even", iid), ("uneven", skew)]:
        for L in (1, 3, 10):
            C = C0.copy(); hist = []
            for _ in range(30 // L):
                C = fed_kmeans(cl, C, 1, L)
                hist += [evaluate(C, Xval, yval)] * L     # value held between syncs
            Lcurves[f"{name}, L={L}"] = [float(v) for v in hist]
    return "reproduction", {"curves": curves, "final": final, "L_curves": Lcurves}


def job_A(seed):
    rng = np.random.default_rng(100 + seed)
    w = np.array([1] * 6 + [2] * 2 + [4] * 2, float)
    ids = rng.permutation(len(Xtr))
    cuts = (np.cumsum(w / w.sum()) * len(ids)).astype(int)[:-1]
    clients = [Xtr[p] for p in np.split(ids, cuts)]
    g = Game(clients, C0, Xval, yval)
    phi = exact_shapley(g, 10)
    res = {"phi": phi.tolist(), "size": [len(c) for c in clients],
           "U_all": g.U(range(10)), "U_empty": g.U(())}
    if seed == 0:
        _, hist = tmc_shapley(g, 10, 300, np.random.default_rng(0))
        res["tmc_err"] = np.abs(hist - phi).max(1).tolist()
    return f"A_{seed}", res


def job_B(seed):
    rng = np.random.default_rng(200 + seed)
    parts = [Xtr[p] for p in dirichlet_split(ytr, 9, 100.0, rng)]
    honest = parts[:6]
    shuffled = [rng.permuted(parts[6], axis=1), rng.permuted(parts[7], axis=1)]
    dup = [honest[0].copy()]
    noise = [rng.normal(Xtr.mean(0), Xtr.std(0), size=parts[8].shape).astype(np.float32)]
    clients = honest + shuffled + dup + noise
    types = ["honest"] * 6 + ["shuffled"] * 2 + ["duplicate"] + ["noise"]
    g = Game(clients, C0, Xval, yval)
    phi = exact_shapley(g, 10)
    N = range(10)
    loo = [g.U(N) - g.U([j for j in N if j != i]) for i in N]
    garbage = np.array([t in ("shuffled", "noise") for t in types])
    keep = [i for i in N if phi[i] > 0.1 * phi.max()]
    honest_ids = [i for i in N if types[i] in ("honest", "duplicate")]
    test = {
        "all": evaluate(fed_kmeans(clients, C0), Xte, yte),
        "shapley_filtered": evaluate(fed_kmeans([clients[i] for i in keep], C0), Xte, yte),
        "oracle_honest": evaluate(fed_kmeans([clients[i] for i in honest_ids], C0), Xte, yte),
    }
    return f"B_{seed}", {"phi": phi.tolist(), "loo": loo, "types": types, "kept": keep,
                         "auroc": roc_auc_score(~garbage, phi),
                         "auroc_loo": roc_auc_score(~garbage, loo),
                         "U_all": g.U(N), "U_empty": g.U(()), "test_nmi": test}


def job_C(args):
    a, seed = args
    idx = dirichlet_split(ytr, 10, a, np.random.default_rng(300 + seed))
    g = Game([Xtr[i] for i in idx], C0, Xval, yval)
    phi = exact_shapley(g, 10)
    n_cls = [int(np.sum(np.bincount(ytr[i], minlength=K) / len(i) >= 0.05)) for i in idx]
    return f"C_{a}_{seed}", {"phi": phi.tolist(), "n_classes": n_cls, "size": [len(i) for i in idx],
                             "gini": float(gini(phi)), "U_all": g.U(range(10)), "U_empty": g.U(())}


def run(job):
    f, arg = job
    return f(arg)


if __name__ == "__main__":
    jobs = ([(job_stability, None), (job_reproduction, None)]
            + [(job_A, s) for s in SEEDS] + [(job_B, s) for s in SEEDS]
            + [(job_C, (a, s)) for a in ALPHAS for s in SEEDS])
    t0, results = time.time(), {}
    with Pool(2) as pool:
        for key, val in pool.imap_unordered(run, jobs):
            results[key] = val
            print(f"[{time.time() - t0:6.0f}s] done {key}", flush=True)
            json.dump(results, open("results/results.json", "w"), default=float)
    print("all done")
