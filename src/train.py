"""모델 비교 -> Optuna 튜닝 -> 앙상블 -> 확률 보정 -> 저장.

사용법: python src/train.py [튜닝_trial_수=25]
"""
import json
import sys
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config import DATA_DIR, MODEL_DIR, MODEL_PATH, N_FOLDS, OUTPUT_DIR, SEED, TARGET
from models import MODEL_NAMES, Model, apply_temperature, blend
from preprocess import Preprocessor, make_groups

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
TUNE_LR = {"lgbm": 0.05, "xgb": 0.05, "cat": 0.08}


def run_cv(name, X, y, folds, params=None, class_weight=None):
    oof = np.zeros((len(y), 3))
    models = []
    for tr, va in folds:
        m = Model(name, params, class_weight).fit(X.iloc[tr], y[tr], X.iloc[va], y[va])
        oof[va] = m.predict_proba(X.iloc[va])
        models.append(m)
    return oof, models


def metrics(y, p):
    pred = p.argmax(1)
    return {
        "logloss": log_loss(y, p),
        "macro_f1": f1_score(y, pred, average="macro"),
        "class2_recall": recall_score(y == 2, pred == 2),
        "auc_ovr": roc_auc_score(y, p, multi_class="ovr"),
    }


def search_space(name, trial):
    if name == "lgbm":
        return dict(learning_rate=TUNE_LR[name],
                    num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if name == "xgb":
        return dict(learning_rate=TUNE_LR[name],
                    max_depth=trial.suggest_int("max_depth", 3, 9),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    return dict(learning_rate=TUNE_LR[name], depth=trial.suggest_int("depth", 4, 8),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10, log=True))


def tune(name, X, y, folds, n_trials):
    def objective(trial):
        oof, _ = run_cv(name, X, y, folds, search_space(name, trial))
        return log_loss(y, oof)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best = {**study.best_params, "learning_rate": TUNE_LR[name]}
    return best, study.best_value


def fit_blend_weights(oofs, y):
    names = list(oofs)
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return log_loss(y, blend(oofs, dict(zip(names, w))))
    r = minimize(loss, np.ones(len(names)) / len(names), method="Nelder-Mead")
    w = np.abs(r.x) / np.abs(r.x).sum()
    return dict(zip(names, w.tolist()))


def fit_temperature(p, y):
    r = minimize(lambda t: log_loss(y, apply_temperature(p, t[0])), [1.0], method="Nelder-Mead")
    return float(r.x[0])


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    train = pd.read_csv(DATA_DIR / "train.csv")
    y = train[TARGET].astype(int).values
    groups = make_groups(train)
    pre = Preprocessor().fit(train)
    X = pre.transform(train)
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y, groups))

    prior = np.bincount(y) / len(y)
    rows = [{"model": "baseline(클래스 비율)", **metrics(y, np.tile(prior, (len(y), 1)))}]

    # 1) 기본 파라미터 모델 비교 + 변형(클래스 가중치, 성별 제외)
    base_oof, base_models = {}, {}
    for n in MODEL_NAMES:
        t = time.time()
        base_oof[n], base_models[n] = run_cv(n, X, y, folds)
        rows.append({"model": f"{n}(default)", **metrics(y, base_oof[n])})
        print(f"[{n}] default logloss={rows[-1]['logloss']:.4f} ({time.time()-t:.0f}s)", flush=True)
    oof, _ = run_cv("lgbm", X, y, folds, class_weight="balanced")
    rows.append({"model": "lgbm(class_weight=balanced)", **metrics(y, oof)})
    Xng = X.drop(columns=["gender"])
    oof, _ = run_cv("lgbm", Xng, y, folds)
    rows.append({"model": "lgbm(성별 제외)", **metrics(y, oof)})
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)

    # 2) 상위 2개 모델 Optuna 튜닝
    ranked = sorted(MODEL_NAMES, key=lambda n: log_loss(y, base_oof[n]))
    tuned_oof, tuned_models, best_params = {}, {}, {}
    for n in ranked[:2]:
        t = time.time()
        # CatBoost 는 1회 학습이 느려(~160s) trial 수를 1/3 로 줄인다
        best_params[n], val = tune(n, X, y, folds, max(3, n_trials // 3) if n == 'cat' else n_trials)
        tuned_oof[n], tuned_models[n] = run_cv(n, X, y, folds, best_params[n])
        rows.append({"model": f"{n}(tuned)", **metrics(y, tuned_oof[n])})
        print(f"[{n}] tuned logloss={rows[-1]['logloss']:.4f} ({time.time()-t:.0f}s) {best_params[n]}", flush=True)

    # 3) 앙상블: 튜닝 모델 + 나머지 기본 모델 중 CV 좋은 쪽을 사용
    members_oof = {n: base_oof[n] for n in MODEL_NAMES if n not in tuned_oof} | tuned_oof
    members_models = {n: base_models[n] for n in MODEL_NAMES if n not in tuned_models} | tuned_models
    weights = fit_blend_weights(members_oof, y)
    ens = blend(members_oof, weights)
    rows.append({"model": "ensemble(가중 평균)", **metrics(y, ens)})

    # 4) 확률 보정 (온도 스케일링). OOF로 맞춘 값이라 다소 낙관적일 수 있음
    T = fit_temperature(ens, y)
    cal = apply_temperature(ens, T)
    rows.append({"model": "ensemble+calibration(최종)", **metrics(y, cal)})
    comp = pd.DataFrame(rows)
    print(comp.round(4).to_string(index=False))
    print("weights:", weights, "T:", round(T, 4))

    # 5) 저장
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    comp.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    bundle = {"preprocessor": pre, "features": X.columns.tolist(), "members": members_models,
              "weights": weights, "temperature": T, "best_params": best_params}
    joblib.dump(bundle, MODEL_PATH)
    oof_df = train.copy()
    oof_df[["p0", "p1", "p2"]] = cal
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    json.dump({"weights": weights, "temperature": T, "best_params": best_params,
               "final": metrics(y, cal)}, open(OUTPUT_DIR / "metrics.json", "w"), indent=2)
    print("저장 완료:", MODEL_PATH)


if __name__ == "__main__":
    main()
