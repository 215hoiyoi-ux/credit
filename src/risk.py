"""위험 점수와 위험군 판정 (PRD F4)."""
import numpy as np
import pandas as pd

HIGH, MID, LOW = "고위험", "중위험", "저위험"
LABELS = [HIGH, MID, LOW]


def risk_score(proba: np.ndarray) -> np.ndarray:
    """위험 점수 = P(credit=2)."""
    return np.asarray(proba)[:, 2]


def classify(score, thr_high: float, thr_mid: float) -> np.ndarray:
    """score >= thr_high 고위험, thr_mid <= score < thr_high 중위험, 나머지 저위험."""
    if thr_mid > thr_high:
        raise ValueError("중위험 임계값은 고위험 임계값보다 클 수 없습니다.")
    score = np.asarray(score)
    return np.where(score >= thr_high, HIGH, np.where(score >= thr_mid, MID, LOW))


def threshold_summary(score, y_true, thr_high: float, thr_mid: float) -> pd.DataFrame:
    """등급별 인원/비율, 그 안의 실제 2등급(고위험) 비율, 전체 2등급 중 포착률."""
    grade = classify(score, thr_high, thr_mid)
    y2 = np.asarray(y_true) == 2
    rows = []
    for lab in LABELS:
        m = grade == lab
        rows.append({
            "위험군": lab, "인원": int(m.sum()), "비율": m.mean(),
            "실제 2등급 비율": y2[m].mean() if m.any() else np.nan,
            "전체 2등급 포착률": (y2 & m).sum() / max(y2.sum(), 1),
        })
    return pd.DataFrame(rows)


def recommended_action(label: str) -> str:
    return {HIGH: "신규 카드 발급 중지 / 기존 회원 한도 조정·집중 관리",
            MID: "관리 프로그램 편입 (알림·상담·결제 안내)",
            LOW: "정상 발급 및 일반 마케팅"}[label]
