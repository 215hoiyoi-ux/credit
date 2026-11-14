from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "credit_model.joblib"

SEED = 42
N_FOLDS = 5
TARGET = "credit"
ID_COL = "index"
# 동일 고객(복수 카드 레코드)을 식별하는 키. begin_month 만 레코드마다 다르다.
GROUP_COLS = [
    "gender", "car", "reality", "child_num", "income_total", "income_type",
    "edu_type", "family_type", "house_type", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "FLAG_MOBIL", "work_phone", "phone", "email", "occyp_type", "family_size",
]
