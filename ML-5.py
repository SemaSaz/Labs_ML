import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, completeness_score,
                             homogeneity_score, normalized_mutual_info_score,
                             silhouette_score, v_measure_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "mixoutALL_shifted.mat"
FIG_DIR = "figures"
SEED = 0
N_CLASSES = 20
RESAMPLE_LEN = 30


# --------------------------------------------------------------------------- данные
def load_trajectories(path=DATA_PATH):
    d = sio.loadmat(path)
    mix = d["mixout"][0]
    consts = d["consts"][0, 0]
    labels = consts["charlabels"].ravel()
    key = [str(k[0]) if isinstance(k, np.ndarray) else str(k) for k in consts["key"][0]]
    trajs = [np.asarray(m, dtype=float) for m in mix]
    print(f"Загружено траекторий: {len(trajs)}, классов: {len(key)}")
    print(f"Символы: {''.join(key)}")
    lengths = [t.shape[1] for t in trajs]
    print(f"Длина траекторий: от {min(lengths)} до {max(lengths)} точек, "
          f"в среднем {np.mean(lengths):.0f}")
    return trajs, labels, key


def resample(traj, length):
    t_old = np.linspace(0, 1, traj.shape[1])
    t_new = np.linspace(0, 1, length)
    return np.column_stack([np.interp(t_new, t_old, traj[i]) for i in range(traj.shape[0])]).T


def build_features(trajs, length):
    return np.array([resample(t, length).ravel() for t in trajs])


def cluster_quality(labels_true, labels_pred, X=None):
    out = {
        "ARI": adjusted_rand_score(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "homogeneity": homogeneity_score(labels_true, labels_pred),
        "completeness": completeness_score(labels_true, labels_pred),
        "V_measure": v_measure_score(labels_true, labels_pred),
        "n_clusters": len(set(labels_pred)),
    }
    if X is not None and out["n_clusters"] > 1:
        out["silhouette"] = silhouette_score(X, labels_pred, sample_size=2000)
    else:
        out["silhouette"] = float("nan")
    return out


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  график сохранён: {path}")


# --------------------------------------------------------------------------- main
def main():
    trajs, y, key = load_trajectories()

    # ---------------------------------------------------------------- пункт 1
    X_raw = build_features(trajs, RESAMPLE_LEN)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    print(f"\n[1] признаки: {X_raw.shape[1]} (3 канала x {RESAMPLE_LEN} точек), "
          f"масштабированы (mean={X.mean():+.3f}, std={X.std():.3f})")

    # ---------------------------------------------------------------- пункт 2
    print("\n[2] кластеризация")
    models = {
        "KMeans": KMeans(n_clusters=N_CLASSES, n_init=10, random_state=SEED).fit(X),
        "Agglomerative(ward)": AgglomerativeClustering(n_clusters=N_CLASSES).fit(X),
        "GaussianMixture": GaussianMixture(n_components=N_CLASSES, random_state=SEED,
                                           covariance_type="diag").fit(X),
    }
    results = {}
    for name, m in models.items():
        pred = m.labels_ if hasattr(m, "labels_") else m.predict(X)
        results[name] = cluster_quality(y, pred, X)
        r = results[name]
        print(f"  {name:<20} ARI {r['ARI']:.3f}  NMI {r['NMI']:.3f}  "
              f"V {r['V_measure']:.3f}  silhouette {r['silhouette']:.3f}")

    # ---------------------------------------------------------------- эксперименты
    print("\n[3.1] влияние длины ресамплинга (KMeans, k=20)")
    L_sweep = []
    for L in (10, 20, 30, 40, 60):
        Xl = StandardScaler().fit_transform(build_features(trajs, L))
        m = KMeans(n_clusters=N_CLASSES, n_init=10, random_state=SEED).fit(Xl)
        L_sweep.append({"L": L, **cluster_quality(y, m.labels_, Xl)})
        print(f"  L={L:<3} признаков {L * 3:<4} NMI {L_sweep[-1]['NMI']:.3f}  "
              f"ARI {L_sweep[-1]['ARI']:.3f}")

    print("\n[3.2] выбор числа кластеров k (KMeans)")
    k_sweep = []
    for k in range(15, 26):
        m = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        k_sweep.append({"k": k, **cluster_quality(y, m.labels_, X)})
        print(f"  k={k:<3} NMI {k_sweep[-1]['NMI']:.3f}  ARI {k_sweep[-1]['ARI']:.3f}  "
              f"silhouette {k_sweep[-1]['silhouette']:.3f}")

    print("\n[3.3] linkage для AgglomerativeClustering")
    linkage_sweep = []
    for link in ("ward", "complete", "average", "single"):
        m = AgglomerativeClustering(n_clusters=N_CLASSES, linkage=link).fit(X)
        linkage_sweep.append({"linkage": link, **cluster_quality(y, m.labels_, X)})
        print(f"  {link:<10} NMI {linkage_sweep[-1]['NMI']:.3f}  "
              f"ARI {linkage_sweep[-1]['ARI']:.3f}")

    print("\n[3.4] параметры DBSCAN")
    dbscan_sweep = []
    for eps in (3.0, 5.0, 8.0, 12.0, 18.0):
        m = DBSCAN(eps=eps, min_samples=5).fit(X)
        q = cluster_quality(y, m.labels_, X)
        dbscan_sweep.append({"eps": eps, "min_samples": 5, **q})
        print(f"  eps={eps:<5} кластеров {q['n_clusters']:<4} NMI {q['NMI']:.3f}  "
              f"ARI {q['ARI']:.3f}")

    # ---------------------------------------------------------------- визуализация
    print("\n[4] t-SNE")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(X)
    best_kmeans = KMeans(n_clusters=N_CLASSES, n_init=10, random_state=SEED).fit(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(tsne[:, 0], tsne[:, 1], c=y, cmap="tab20", s=8)
    ax.set_title("t-SNE, цвет — истинный класс")
    fig.colorbar(sc, ax=ax)
    savefig(fig, "fig1_tsne_true.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(tsne[:, 0], tsne[:, 1], c=best_kmeans.labels_, cmap="tab20", s=8)
    ax.set_title("t-SNE, цвет — кластер KMeans")
    fig.colorbar(sc, ax=ax)
    savefig(fig, "fig2_tsne_kmeans.png")

    alg_names = list(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(alg_names))
    ax.bar(x - 0.2, [results[n]["NMI"] for n in alg_names], 0.4, label="NMI")
    ax.bar(x + 0.2, [results[n]["ARI"] for n in alg_names], 0.4, label="ARI")
    ax.set_xticks(x)
    ax.set_xticklabels(alg_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("метрика")
    ax.set_title("Сравнение алгоритмов (против истинных меток)")
    ax.legend()
    savefig(fig, "fig3_compare.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["k"] for r in k_sweep], [r["NMI"] for r in k_sweep], "o-")
    ax.set_xlabel("k")
    ax.set_ylabel("NMI")
    ax.set_title("NMI от числа кластеров (KMeans)")
    ax.grid(alpha=0.3)
    savefig(fig, "fig4_k.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["L"] for r in L_sweep], [r["NMI"] for r in L_sweep], "o-")
    ax.set_xlabel("длина ресамплинга L")
    ax.set_ylabel("NMI")
    ax.set_title("NMI от длины ресамплинга (KMeans)")
    ax.grid(alpha=0.3)
    savefig(fig, "fig5_L.png")

    # ---------------------------------------------------------------- итог
    best = max(results.items(), key=lambda kv: kv[1]["NMI"])
    print(f"\nИТОГ: лучший алгоритм — {best[0]}, NMI = {best[1]['NMI']:.3f}, "
          f"ARI = {best[1]['ARI']:.3f}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"n": len(trajs), "n_classes": N_CLASSES, "features": X_raw.shape[1],
                     "key": key, "resample_len": RESAMPLE_LEN},
            "algorithms": results,
            "L_sweep": L_sweep,
            "k_sweep": k_sweep,
            "linkage_sweep": linkage_sweep,
            "dbscan_sweep": dbscan_sweep,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()