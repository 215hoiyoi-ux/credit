"""SHAP 기반 설명 (PRD F5). LightGBM 폴드 모델을 사용한다."""
import numpy as np
import pandas as pd
import shap

RISK_CLASS = 2


def _class_values(sv, cls: int = RISK_CLASS) -> np.ndarray:
    """shap 버전별 출력 형태(list / 3D array)를 (n, features) 로 통일."""
    if isinstance(sv, list):
        return np.asarray(sv[cls])
    sv = np.asarray(sv)
    return sv[:, :, cls] if sv.ndim == 3 else sv


def shap_values(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """폴드 모델 평균 SHAP (클래스 2 기준, 양수 = 위험 증가)."""
    vals = [_class_values(shap.TreeExplainer(m.m).shap_values(X)) for m in bundle["members"]["lgbm"]]
    return np.mean(vals, axis=0)


def global_importance(bundle: dict, X: pd.DataFrame, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    Xs = X.sample(min(n, len(X)), random_state=seed)
    sv = shap_values(bundle, Xs)
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def top_risk_factors(bundle: dict, df_raw: pd.DataFrame, top_n: int = 5) -> list[pd.DataFrame]:
    """행별 위험을 높이는 상위 N개 요인. 반환: 행마다 (feature, value, shap) DataFrame."""
    X = bundle["preprocessor"].transform(df_raw)[bundle["features"]]
    sv = shap_values(bundle, X)
    out = []
    for i in range(len(X)):
        order = np.argsort(-sv[i])[:top_n]
        out.append(pd.DataFrame({
            "feature": X.columns[order], "value": X.iloc[i, order].astype(str).values,
            "shap": sv[i, order]}))
    return out
