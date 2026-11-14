"""모델 래퍼(LightGBM / XGBoost / CatBoost)와 앙상블 추론."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from config import SEED
from preprocess import CAT_COLS

MODEL_NAMES = ["lgbm", "xgb", "cat"]

DEFAULT_PARAMS = {
    "lgbm": dict(learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=1.0),
    "xgb": dict(learning_rate=0.03, max_depth=5, min_child_weight=3, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0),
    "cat": dict(learning_rate=0.05, depth=6, l2_leaf_reg=3.0),
}


class Model:
    """fit(Xtr, ytr, Xva, yva) 후 predict_proba(X). 모델별 입력 변환을 캡슐화."""

    def __init__(self, name: str, params: dict | None = None, class_weight=None):
        self.name = name
        self.params = {**DEFAULT_PARAMS[name], **(params or {})}
        self.class_weight = class_weight

    def _cat_input(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in CAT_COLS:
            if c in X.columns:
                # 결측/미지 범주(NaN)를 문자열로 바꿔야 CatBoost 가 받는다 (astype(str) 은 NaN 을 유지함)
                s = X[c].astype("object")
                X[c] = s.where(s.notna(), "__unknown__").astype(str)
        return X

    def fit(self, Xtr, ytr, Xva, yva):
        if self.name == "lgbm":
            self.m = lgb.LGBMClassifier(
                objective="multiclass", n_estimators=3000, random_state=SEED, verbose=-1,
                subsample_freq=1, class_weight=self.class_weight, **self.params)
            self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
            self.best_iter = self.m.best_iteration_
        elif self.name == "xgb":
            sw = None
            if self.class_weight == "balanced":
                cnt = np.bincount(ytr)
                sw = (len(ytr) / (len(cnt) * cnt))[ytr]
            self.m = xgb.XGBClassifier(
                objective="multi:softprob", n_estimators=3000, tree_method="hist",
                enable_categorical=True, early_stopping_rounds=100, eval_metric="mlogloss",
                random_state=SEED, **self.params)
            self.m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
            self.best_iter = self.m.best_iteration
        else:
            cats = [c for c in CAT_COLS if c in Xtr.columns]
            self.m = CatBoostClassifier(
                loss_function="MultiClass", iterations=3000, random_seed=SEED, verbose=0,
                early_stopping_rounds=100, cat_features=cats, thread_count=8,
                auto_class_weights="Balanced" if self.class_weight == "balanced" else None,
                **self.params)
            self.m.fit(self._cat_input(Xtr), ytr, eval_set=(self._cat_input(Xva), yva))
            self.best_iter = self.m.get_best_iteration()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.name == "cat":
            X = self._cat_input(X)
        return self.m.predict_proba(X)


def blend(probas: dict, weights: dict) -> np.ndarray:
    return sum(weights[k] * probas[k] for k in weights)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-9, 1)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ensemble_predict(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """원본 형식 DataFrame -> 보정된 앙상블 확률 (n, 3). 앱/일괄 예측 공통 경로."""
    X = bundle["preprocessor"].transform(df)[bundle["features"]]
    probas = {k: np.mean([m.predict_proba(X) for m in ms], axis=0)
              for k, ms in bundle["members"].items()}
    return apply_temperature(blend(probas, bundle["weights"]), bundle["temperature"])
