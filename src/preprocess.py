"""전처리. fit 은 train 에서만, test/앱 추론에서는 transform 만 사용한다."""
import numpy as np
import pandas as pd

from config import GROUP_COLS, ID_COL, TARGET

EMPLOYED_ANOMALY = 365243
CAT_COLS = ["gender", "car", "reality", "income_type", "edu_type",
            "family_type", "house_type", "occyp_type"]


def make_groups(df: pd.DataFrame) -> np.ndarray:
    """동일 고객 레코드가 같은 CV 폴드에 들어가도록 그룹 id 생성."""
    key = pd.util.hash_pandas_object(df[GROUP_COLS].fillna("Unknown"), index=False)
    return pd.factorize(key)[0]


class Preprocessor:
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        raw = self._clean(df)
        # 상수 컬럼 제거 (예: FLAG_MOBIL)
        self.drop_cols_ = [c for c in raw.columns if raw[c].nunique(dropna=False) <= 1]
        self.categories_ = {c: sorted(raw[c].astype(str).unique()) for c in CAT_COLS}
        self.columns_ = self.transform(df).columns.tolist()
        return self

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns]).copy()
        out["occyp_type"] = out["occyp_type"].fillna("Unknown")
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._clean(df)
        anomaly = out["DAYS_EMPLOYED"] == EMPLOYED_ANOMALY
        out["is_unemployed"] = anomaly.astype(int)
        out["age_years"] = -out["DAYS_BIRTH"] / 365
        out["employed_years"] = np.where(anomaly, 0.0, -out["DAYS_EMPLOYED"] / 365)
        out["months_since_issue"] = -out["begin_month"]
        # 파생 변수 (PRD F1)
        out["income_per_person"] = out["income_total"] / out["family_size"].clip(lower=1)
        out["log_income"] = np.log1p(out["income_total"])
        out["child_ratio"] = out["child_num"] / out["family_size"].clip(lower=1)
        out["employed_ratio"] = out["employed_years"] / out["age_years"]
        out = out.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "begin_month"])
        if hasattr(self, "drop_cols_"):
            out = out.drop(columns=[c for c in self.drop_cols_ if c in out.columns])
        for c in CAT_COLS:
            if hasattr(self, "categories_"):
                out[c] = pd.Categorical(out[c].astype(str), categories=self.categories_[c])
            else:
                out[c] = out[c].astype(str).astype("category")
        if hasattr(self, "columns_"):
            out = out[self.columns_]
        return out
