"""일괄/단건 입력 검증. 오류(errors)는 예측을 막고, 경고(warnings)는 안내만 한다."""
import numpy as np
import pandas as pd

NUM_COLS = ["child_num", "income_total", "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL",
            "work_phone", "phone", "email", "family_size", "begin_month"]
# occyp_type 은 학습 데이터에서도 결측이 많아 비어 있어도 정상으로 본다.
CAT_COLS_STRICT = ["gender", "car", "reality", "income_type", "edu_type", "family_type", "house_type"]
MIN_EMPLOY_START_AGE = 14  # 근속 시작 가능 최소 나이


def train_ranges(oof: pd.DataFrame) -> dict:
    """학습 데이터 기준 입력 범위(경고용)."""
    age = -oof["DAYS_BIRTH"] / 365
    return {"age_min": float(age.min()), "age_max": float(age.max()),
            "income_max": float(oof["income_total"].max())}


def range_warnings(age: float, income: float, rng: dict) -> list[str]:
    out = []
    if not rng["age_min"] - 1 <= age <= rng["age_max"] + 1:
        out.append(f"나이 {age:.0f}세는 학습 데이터 범위({rng['age_min']:.0f}~{rng['age_max']:.0f}세) 밖입니다. 예측 신뢰도가 낮습니다.")
    if income > rng["income_max"]:
        out.append(f"연 소득 {income:,.0f}은 학습 데이터 최대값({rng['income_max']:,.0f})을 넘습니다. 예측 신뢰도가 낮습니다.")
    return out


def validate_batch(df: pd.DataFrame, categories: dict, rng: dict):
    """반환: (정제된 df, errors, warnings)."""
    errors, warns = [], []
    df = df.copy()
    for c in NUM_COLS:
        conv = pd.to_numeric(df[c], errors="coerce")
        bad = conv.isna() & df[c].notna()
        blank = df[c].isna()
        if bad.any():
            errors.append(f"`{c}`: 숫자가 아닌 값 {int(bad.sum())}건 (예: {len(df) and df.loc[bad, c].iloc[0]!r}, "
                          f"{int(np.flatnonzero(bad)[0]) + 2}행)")
        if blank.any():
            errors.append(f"`{c}`: 비어 있는 값 {int(blank.sum())}건 ({int(np.flatnonzero(blank)[0]) + 2}행부터)")
        df[c] = conv
    if errors:
        return df, errors, warns

    if (df["DAYS_BIRTH"] >= 0).any():
        errors.append("`DAYS_BIRTH`: 0 이상인 값이 있습니다. 생일은 오늘 기준 과거 일수(음수)여야 합니다.")
    if (df["family_size"] < 1).any():
        errors.append("`family_size`: 1 미만인 값이 있습니다.")
    if errors:
        return df, errors, warns

    age = -df["DAYS_BIRTH"] / 365
    n_age = int((~age.between(rng["age_min"] - 1, rng["age_max"] + 1)).sum())
    n_inc = int((df["income_total"] > rng["income_max"]).sum())
    if n_age:
        warns.append(f"나이가 학습 데이터 범위({rng['age_min']:.0f}~{rng['age_max']:.0f}세) 밖인 행 {n_age}건: 예측 신뢰도가 낮습니다.")
    if n_inc:
        warns.append(f"소득이 학습 데이터 최대값({rng['income_max']:,.0f})을 넘는 행 {n_inc}건: 예측 신뢰도가 낮습니다.")
    emp = (df["DAYS_EMPLOYED"] < 0) & (-df["DAYS_EMPLOYED"] / 365 > age - MIN_EMPLOY_START_AGE)
    if emp.any():
        warns.append(f"근속 기간이 나이와 맞지 않는 행 {int(emp.sum())}건 (나이 - {MIN_EMPLOY_START_AGE}년 초과).")

    for c in CAT_COLS_STRICT:
        blank = df[c].isna()
        unseen = ~df[c].isin(categories[c]) & ~blank
        if blank.any():
            warns.append(f"`{c}`: 비어 있는 값 {int(blank.sum())}건은 '알 수 없음'으로 처리됩니다.")
        if unseen.any():
            vals = ", ".join(map(str, df.loc[unseen, c].unique()[:3]))
            warns.append(f"`{c}`: 학습에 없던 값 {int(unseen.sum())}건({vals})은 '알 수 없음'으로 처리됩니다.")
    unseen = ~df["occyp_type"].isna() & ~df["occyp_type"].isin(categories["occyp_type"])
    if unseen.any():
        warns.append(f"`occyp_type`: 학습에 없던 값 {int(unseen.sum())}건은 '알 수 없음'으로 처리됩니다.")
    return df, errors, warns
