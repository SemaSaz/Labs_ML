import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "cup98LRN.txt"
FIG_DIR = "figures"
SEED = 42
TEST_SIZE = 0.3

FEATS = ["LASTGIFT", "AVGGIFT", "MAXRAMNT", "MINRAMNT", "NGIFTALL", "CARDGIFT",
         "NUMPROM", "CARDPROM", "TIMELAG", "AGE", "INCOME", "WEALTH1", "HIT", "CLUSTER2"]

POLY_FEATS = ["LASTGIFT", "NGIFTALL", "AGE"]


# --------------------------------------------------------------------------- данные
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, usecols=FEATS + ["TARGET_B", "TARGET_D"],
                     encoding="latin-1", low_memory=False)
    print(f"Прочитано записей: {len(df)}")

    df = df[df["TARGET_B"] == 1].reset_index(drop=True)
    print(f"Из них откликнувшихся (TARGET_B = 1): {len(df)}")

    X = df[FEATS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df["TARGET_D"].to_numpy(dtype=float)
    print(f"Целевая переменная TARGET_D: mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.0f}, max={y.max():.0f}")
    return X, y


def split_manual(X, y, test_size=TEST_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - test_size))
    return idx[:cut], idx[cut:]


def quality(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def print_metrics(tag, tr, te):
    print(f"{tag:<44} R2 {tr['R2']:+.3f}/{te['R2']:+.3f}   "
          f"RMSE {tr['RMSE']:7.2f}/{te['RMSE']:8.2f}   MAE {tr['MAE']:6.2f}/{te['MAE']:6.2f}"
          "   (train/test)")


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  график сохранён: {path}")


# --------------------------------------------------------------------------- main
def main():
    X, y = load_data()

    # ---------------------------------------------------------------- пункт 1
    tr, te = split_manual(X, y)
    print(f"\n[1] train: {len(tr)} записей, test: {len(te)} записей "
          f"({100 * (1 - TEST_SIZE):.0f}/{100 * TEST_SIZE:.0f} %), seed={SEED}")

    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])
    ytr, yte = y[tr], y[te]

    # ---------------------------------------------------------------- пункты 2-3
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
    lin_tr = quality(ytr, model.predict(Xtr))
    lin_te = quality(yte, model.predict(Xte))
    print("\n[2-3] линейная регрессия на 14 признаках")
    print_metrics("  LinearRegression", lin_tr, lin_te)

    coefs = pd.Series(model[-1].coef_, index=FEATS).sort_values(key=np.abs, ascending=False)
    print("  коэффициенты (доллары на 1 СКО):")
    for name, value in coefs.items():
        print(f"    {name:<10} {value:+8.3f}")

    base_rmse = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    print(f"  базовый прогноз средним: RMSE {base_rmse:.2f} (R2 = 0)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(yte, model.predict(Xte), s=12, alpha=0.5, edgecolor="none")
    lim = max(yte.max(), model.predict(Xte).max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=1)
    ax.set_xlabel("факт, доллары")
    ax.set_ylabel("прогноз, доллары")
    ax.set_title("Линейная модель: прогноз vs факт (тест)")
    savefig(fig, "fig6_scatter.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(coefs.index[::-1], coefs.values[::-1],
            color=["#1f77b4" if v >= 0 else "#d62728" for v in coefs.values[::-1]])
    ax.set_xlabel("доллары на 1 стандартное отклонение")
    ax.set_title("Стандартизованные коэффициенты линейной модели")
    savefig(fig, "fig5_coef.png")

    # ---------------------------------------------------------------- пункт 4
    print("\n[4] полиномиальные признаки, степень 1..10")
    cols = [FEATS.index(f) for f in POLY_FEATS]
    Ptr, Pte = Xtr[:, cols], Xte[:, cols]

    degrees, poly_rows = list(range(1, 11)), []
    for deg in degrees:
        poly_model = make_pipeline(PolynomialFeatures(deg, include_bias=False),
                                   StandardScaler(),
                                   LinearRegression()).fit(Ptr, ytr)
        row = {"degree": deg,
               "n_features": poly_model[0].n_output_features_,
               "train": quality(ytr, poly_model.predict(Ptr)),
               "test": quality(yte, poly_model.predict(Pte))}
        poly_rows.append(row)
        print(f"  степень {deg:2d} ({row['n_features']:3d} признаков)  "
              f"RMSE {row['train']['RMSE']:9.2f}/{row['test']['RMSE']:12.2f}  "
              f"R2_test {row['test']['R2']:+.3f}")

    best = max(poly_rows, key=lambda r: r["test"]["R2"])
    print(f"  лучшая степень: {best['degree']} (R2 = {best['test']['R2']:+.3f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, [r["train"]["RMSE"] for r in poly_rows], "o-", label="обучающая выборка")
    ax.plot(degrees, [r["test"]["RMSE"] for r in poly_rows], "s-", label="тестовая выборка")
    ax.set_yscale("log")
    ax.set_xticks(degrees)
    ax.set_xlabel("степень полинома")
    ax.set_ylabel("RMSE, доллары (лог. шкала)")
    ax.set_title("Зависимость RMSE от степени полинома")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "fig1_poly_degree.png")

    # ---------------------------------------------------------------- пункт 5
    print("\n[5] регуляризация")

    alphas = np.logspace(-2, 6, 33)
    ridge_rows = []
    for a in alphas:
        m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr)
        ridge_rows.append({"alpha": a, "train": quality(ytr, m.predict(Xtr)),
                           "test": quality(yte, m.predict(Xte))})
    best_ridge = min(ridge_rows, key=lambda r: r["test"]["RMSE"])
    for r in ridge_rows[::4]:
        print(f"  Ridge  alpha={r['alpha']:>10.2f}  R2_test {r['test']['R2']:+.3f}  "
              f"RMSE_test {r['test']['RMSE']:6.2f}")
    print(f"  лучший alpha для Ridge: {best_ridge['alpha']:.3g} "
          f"(R2 = {best_ridge['test']['R2']:+.3f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(alphas, [r["train"]["RMSE"] for r in ridge_rows], "o-", label="обучающая выборка")
    ax.semilogx(alphas, [r["test"]["RMSE"] for r in ridge_rows], "s-", label="тестовая выборка")
    ax.set_xlabel("alpha (лог. шкала)")
    ax.set_ylabel("RMSE, доллары")
    ax.set_title("Ridge на исходных 14 признаках")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "fig2_ridge_raw.png")

    poly5 = make_pipeline(PolynomialFeatures(5, include_bias=False), StandardScaler())
    Ftr, Fte = poly5.fit_transform(Ptr), poly5.transform(Pte)
    alphas2 = np.logspace(-4, 4, 33)
    ridge_poly = []
    for a in alphas2:
        m = Ridge(alpha=a).fit(Ftr, ytr)
        ridge_poly.append({"alpha": a, "train": quality(ytr, m.predict(Ftr)),
                           "test": quality(yte, m.predict(Fte))})
    best_rp = min(ridge_poly, key=lambda r: r["test"]["RMSE"])
    print(f"  Ridge поверх полинома 5-й степени ({Ftr.shape[1]} признаков): "
          f"лучший alpha = {best_rp['alpha']:.3g}, R2 = {best_rp['test']['R2']:+.3f}, "
          f"RMSE = {best_rp['test']['RMSE']:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(alphas2, [r["train"]["RMSE"] for r in ridge_poly], "o-", label="обучающая выборка")
    ax.semilogx(alphas2, [r["test"]["RMSE"] for r in ridge_poly], "s-", label="тестовая выборка")
    ax.set_xlabel("alpha (лог. шкала)")
    ax.set_ylabel("RMSE, доллары")
    ax.set_title("Ridge поверх полинома 5-й степени")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "fig3_ridge_poly.png")

    lasso_alphas = np.logspace(-3, 0.6, 13)
    lasso_rows = []
    for a in lasso_alphas:
        m = make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=5000)).fit(Xtr, ytr)
        nz = int((np.abs(m[-1].coef_) > 1e-9).sum())
        lasso_rows.append({"alpha": a, "nonzero": nz,
                           "train": quality(ytr, m.predict(Xtr)),
                           "test": quality(yte, m.predict(Xte))})
    best_lasso = min(lasso_rows, key=lambda r: r["test"]["RMSE"])
    for r in lasso_rows:
        print(f"  Lasso  alpha={r['alpha']:>7.3f}  ненулевых коэффициентов {r['nonzero']:2d}  "
              f"R2_test {r['test']['R2']:+.3f}")
    print(f"  лучший alpha для Lasso: {best_lasso['alpha']:.3g} "
          f"({best_lasso['nonzero']} из {len(FEATS)} признаков, R2 = {best_lasso['test']['R2']:+.3f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lasso_alphas, [r["train"]["RMSE"] for r in lasso_rows], "o-", label="обучающая выборка")
    ax.semilogx(lasso_alphas, [r["test"]["RMSE"] for r in lasso_rows], "s-", label="тестовая выборка")
    ax.set_xlabel("alpha (лог. шкала)")
    ax.set_ylabel("RMSE, доллары")
    ax.set_title("Lasso на исходных признаках")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "fig4_lasso.png")

    # ---------------------------------------------------------------- итог
    print("\nИТОГ (тестовая выборка):")
    print(f"  базовая линейная модель:      R2 {lin_te['R2']:+.3f}, RMSE {lin_te['RMSE']:.2f}")
    print(f"  полином {best['degree']}-й степени:           R2 {best['test']['R2']:+.3f}, "
          f"RMSE {best['test']['RMSE']:.2f}")
    print(f"  Ridge (14 признаков):         R2 {best_ridge['test']['R2']:+.3f}, "
          f"RMSE {best_ridge['test']['RMSE']:.2f}")
    print(f"  Ridge (полином 5-й степени):  R2 {best_rp['test']['R2']:+.3f}, "
          f"RMSE {best_rp['test']['RMSE']:.2f}")
    print(f"  Lasso:                        R2 {best_lasso['test']['R2']:+.3f}, "
          f"RMSE {best_lasso['test']['RMSE']:.2f}")


if __name__ == "__main__":
    main()