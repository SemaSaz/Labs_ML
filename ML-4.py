import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "EEG Eye State.arff"
FIG_DIR = "figures"
SEED = 0
PARTS = (0.7, 0.15, 0.15)
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
            "P8", "T8", "FC6", "F4", "F8", "AF4"]
CLASS_NAMES = ["глаза открыты (0)", "глаза закрыты (1)"]


# --------------------------------------------------------------------------- данные
def load_arff(path=DATA_PATH):
    rows, in_data = [], False
    with open(path, encoding="latin-1") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("@data"):
                in_data = True
                continue
            if not in_data or not s:
                continue
            parts = s.split(",")
            if len(parts) == len(CHANNELS) + 1:
                rows.append([float(v) for v in parts])
    A = np.array(rows)
    X, y = A[:, :-1], A[:, -1].astype(int)
    print(f"Прочитано записей: {A.shape[0]}, признаков: {X.shape[1]}")
    print(f"Классы: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def split_stratified(y, parts=PARTS, seed=SEED):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(n * parts[0]), int(n * (parts[0] + parts[1]))
        tr += list(idx[:a])
        va += list(idx[a:b])
        te += list(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


def scale(Xtr, Xva, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)


# --------------------------------------------------------------------------- метрики
def evaluate(model, Xtr, ytr, Xva, yva, Xte, yte):
    ptr, pva, pte = model.predict(Xtr), model.predict(Xva), model.predict(Xte)
    return {
        "acc_train": accuracy_score(ytr, ptr),
        "acc_val": accuracy_score(yva, pva),
        "acc_test": accuracy_score(yte, pte),
        "f1_val": f1_score(yva, pva, average="macro"),
    }


# --------------------------------------------------------------------------- эксперименты
def sweep_learning_rate(Xtr, ytr, Xva, yva, Xte, yte, rates):
    rows = []
    for r in rates:
        perc = Perceptron(eta0=r, alpha=1e-4, max_iter=300, tol=1e-4,
                          random_state=SEED, shuffle=True).fit(Xtr, ytr)
        mlp = MLPClassifier(hidden_layer_sizes=(100,), solver="sgd", learning_rate_init=r,
                            alpha=1e-4, max_iter=200, random_state=SEED).fit(Xtr, ytr)
        rows.append({"rate": r,
                     "perc": evaluate(perc, Xtr, ytr, Xva, yva, Xte, yte),
                     "mlp": evaluate(mlp, Xtr, ytr, Xva, yva, Xte, yte)})
    return rows


def sweep_alpha(Xtr, ytr, Xva, yva, Xte, yte, alphas):
    rows = []
    for a in alphas:
        perc = Perceptron(eta0=0.01, alpha=a, max_iter=300, tol=1e-4,
                          random_state=SEED, shuffle=True).fit(Xtr, ytr)
        mlp = MLPClassifier(hidden_layer_sizes=(100,), solver="adam", alpha=a,
                            learning_rate_init=0.001, max_iter=200,
                            random_state=SEED).fit(Xtr, ytr)
        rows.append({"alpha": a,
                     "perc": evaluate(perc, Xtr, ytr, Xva, yva, Xte, yte),
                     "mlp": evaluate(mlp, Xtr, ytr, Xva, yva, Xte, yte)})
    return rows


def sweep_solver(Xtr, ytr, Xva, yva, Xte, yte, solvers):
    rows = []
    for s in solvers:
        mlp = MLPClassifier(hidden_layer_sizes=(100,), solver=s, alpha=1e-4,
                            learning_rate_init=0.01, max_iter=200,
                            random_state=SEED).fit(Xtr, ytr)
        rows.append({"solver": s, "result": evaluate(mlp, Xtr, ytr, Xva, yva, Xte, yte)})
    return rows


def sweep_hidden(Xtr, ytr, Xva, yva, Xte, yte, sizes):
    rows = []
    for h in sizes:
        mlp = MLPClassifier(hidden_layer_sizes=h, solver="adam", alpha=1e-4,
                            learning_rate_init=0.001, max_iter=200,
                            random_state=SEED).fit(Xtr, ytr)
        rows.append({"hidden": h, "result": evaluate(mlp, Xtr, ytr, Xva, yva, Xte, yte)})
    return rows


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  график сохранён: {path}")


# --------------------------------------------------------------------------- main
def main():
    X, y = load_arff()

    # ---------------------------------------------------------------- пункт 1
    tr, va, te = split_stratified(y)
    print(f"\n[1] обучение {len(tr)}, валидация {len(va)}, тест {len(te)} "
          f"(70/15/15, стратифицированно, seed={SEED})")

    # ---------------------------------------------------------------- пункт 2
    Xtr, Xva, Xte = scale(X[tr], X[va], X[te])
    ytr, yva, yte = y[tr], y[va], y[te]
    print(f"[2] стандартизация: среднее {Xtr.mean():+.3f}, разброс {Xtr.std():.3f}")

    # ---------------------------------------------------------------- пункт 3
    perc = Perceptron(eta0=0.01, alpha=1e-4, max_iter=300, tol=1e-4,
                      random_state=SEED, shuffle=True).fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), solver="adam", alpha=1e-4,
                        learning_rate_init=0.001, max_iter=200,
                        random_state=SEED).fit(Xtr, ytr)
    base = {"perc": evaluate(perc, Xtr, ytr, Xva, yva, Xte, yte),
            "mlp": evaluate(mlp, Xtr, ytr, Xva, yva, Xte, yte)}
    print("\n[3] модели обучены")
    for name, r in (("Perceptron", base["perc"]), ("MLPClassifier", base["mlp"])):
        print(f"  {name:<16} train {r['acc_train']:.4f}  val {r['acc_val']:.4f}  "
              f"test {r['acc_test']:.4f}  F1(val) {r['f1_val']:.4f}")

    # ---------------------------------------------------------------- пункт 4
    print("\n[4] точность на тестовой выборке")
    print(f"  Perceptron:    {base['perc']['acc_test']:.4f}")
    print(f"  MLPClassifier: {base['mlp']['acc_test']:.4f}")
    cm = confusion_matrix(yte, mlp.predict(Xte))
    print("  матрица ошибок MLP:")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<20}" + " ".join(f"{v:>6}" for v in row))

    # ---------------------------------------------------------------- пункт 5
    rates = [0.001, 0.01, 0.1, 1.0]
    lr_rows = sweep_learning_rate(Xtr, ytr, Xva, yva, Xte, yte, rates)
    print("\n[5.1] коэффициент обучения")
    for r in lr_rows:
        print(f"  {r['rate']:<8} Perceptron {r['perc']['acc_val']:.4f}/{r['perc']['acc_test']:.4f}"
              f"   MLP-sgd {r['mlp']['acc_val']:.4f}/{r['mlp']['acc_test']:.4f}")

    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    al_rows = sweep_alpha(Xtr, ytr, Xva, yva, Xte, yte, alphas)
    print("\n[5.2] параметр регуляризации alpha")
    for r in al_rows:
        print(f"  {r['alpha']:<8} Perceptron {r['perc']['acc_val']:.4f}/{r['perc']['acc_test']:.4f}"
              f"   MLP-adam {r['mlp']['acc_val']:.4f}/{r['mlp']['acc_test']:.4f}")

    sol_rows = sweep_solver(Xtr, ytr, Xva, yva, Xte, yte, ["adam", "sgd", "lbfgs"])
    print("\n[5.3] функция оптимизации")
    for r in sol_rows:
        print(f"  {r['solver']:<7} val {r['result']['acc_val']:.4f}  test {r['result']['acc_test']:.4f}")

    hid_rows = sweep_hidden(Xtr, ytr, Xva, yva, Xte, yte, [(10,), (50,), (100,), (100, 50)])
    print("\n[5.4] размер скрытого слоя")
    for r in hid_rows:
        print(f"  {str(r['hidden']):<12} val {r['result']['acc_val']:.4f}  "
              f"test {r['result']['acc_test']:.4f}")

    # ---------------------------------------------------------------- графики
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(rates, [r["perc"]["acc_val"] for r in lr_rows], "o-", label="Perceptron")
    ax.semilogx(rates, [r["mlp"]["acc_val"] for r in lr_rows], "s-", label="MLPClassifier (sgd)")
    ax.set_xlabel("коэффициент обучения")
    ax.set_ylabel("точность на валидации")
    ax.set_title("Влияние коэффициента обучения")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "fig1_learning_rate.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(alphas, [r["perc"]["acc_val"] for r in al_rows], "o-", label="Perceptron")
    ax.semilogx(alphas, [r["mlp"]["acc_val"] for r in al_rows], "s-", label="MLPClassifier (adam)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("точность на валидации")
    ax.set_title("Влияние параметра регуляризации")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "fig2_alpha.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([r["solver"] for r in sol_rows], [r["result"]["acc_test"] for r in sol_rows])
    ax.set_ylabel("точность на тесте")
    ax.set_title("Влияние функции оптимизации")
    savefig(fig, "fig3_solver.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(r["hidden"]) for r in hid_rows], [r["result"]["acc_val"] for r in hid_rows])
    ax.set_ylabel("точность на валидации")
    ax.set_title("Влияние размера скрытого слоя")
    savefig(fig, "fig4_hidden.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2))
    ax.set_xticklabels(["0", "1"])
    ax.set_yticks(range(2))
    ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("прогноз")
    ax.set_ylabel("истина")
    ax.set_title("Матрица ошибок MLP")
    fig.colorbar(im, ax=ax)
    savefig(fig, "fig5_confusion.png")

    # ---------------------------------------------------------------- итог
    print("\nИТОГ:")
    print(f"  лучший Perceptron: test {max(r['perc']['acc_test'] for r in lr_rows):.4f}")
    print(f"  лучший MLP:        test {max([base['mlp']['acc_test']] + [r['mlp']['acc_test'] for r in al_rows] + [r['result']['acc_test'] for r in sol_rows]):.4f}")


if __name__ == "__main__":
    main()