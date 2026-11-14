"""test.csv 예측 -> outputs/submission.csv (index,0,1,2)."""
import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, ID_COL, MODEL_PATH, OUTPUT_DIR
from models import ensemble_predict


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    test = pd.read_csv(DATA_DIR / "test.csv")
    proba = ensemble_predict(test, bundle)

    sub = pd.DataFrame(proba, columns=["0", "1", "2"])
    sub.insert(0, ID_COL, test[ID_COL].values)

    template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert len(sub) == len(template) == 10000, "행 수 불일치"
    assert (sub[ID_COL].values == template[ID_COL].values).all(), "index 불일치"
    assert not sub.isna().any().any(), "NaN 존재"
    assert np.allclose(sub[["0", "1", "2"]].sum(axis=1), 1), "확률 합이 1이 아님"

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"저장: {out} shape={sub.shape}")


if __name__ == "__main__":
    main()
