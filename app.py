import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))  # 모델 unpickle 시 preprocess 모듈 필요
import preprocess  # noqa: E402,F401
import ui  # noqa: E402
from config import MODEL_PATH, OUTPUT_DIR  # noqa: E402
from explain import global_importance, top_risk_factors  # noqa: E402
from models import ensemble_predict  # noqa: E402
from validate import MIN_EMPLOY_START_AGE, range_warnings, train_ranges, validate_batch  # noqa: E402
from risk import HIGH, LABELS, MID, classify, recommended_action, risk_score, threshold_summary  # noqa: E402

RAW_COLS = ["gender", "car", "reality", "child_num", "income_total", "income_type", "edu_type",
            "family_type", "house_type", "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL", "work_phone",
            "phone", "email", "occyp_type", "family_size", "begin_month"]
SAMPLE_ROW = ["F", "N", "Y", 0, 135000.0, "State servant", "Higher education", "Married",
              "House / apartment", -18964, -8671, 1, 0, 1, 0, "Core staff", 2.0, -36.0]

st.set_page_config(page_title="연체 위험 예측", page_icon="💳", layout="wide")
ui.inject_css()


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_oof():
    return pd.read_csv(OUTPUT_DIR / "oof_predictions.csv")


@st.cache_data
def load_importance(_bundle):
    oof = load_oof()
    X = _bundle["preprocessor"].transform(oof)[_bundle["features"]]
    return global_importance(_bundle, X)


@st.cache_data(show_spinner=False)
def predict_cached(df: pd.DataFrame) -> np.ndarray:
    return ensemble_predict(df, load_bundle())


if not MODEL_PATH.exists() or not (OUTPUT_DIR / "oof_predictions.csv").exists():
    st.error("모델 또는 학습 산출물이 없습니다. `python src/train.py` 를 먼저 실행하세요.")
    st.stop()

bundle = load_bundle()
oof = load_oof()
oof["score"] = oof["p2"]
cats = bundle["preprocessor"].categories_
rng = train_ranges(oof)

# ---------------- 사이드바 ----------------
st.sidebar.title("💳 연체 위험 예측")
page = st.sidebar.radio("메뉴", ["대시보드", "단건 예측", "일괄 예측", "모델 성능"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.subheader("위험군 임계값")
default_high = float(np.quantile(oof["score"], 0.80))
default_mid = float(np.quantile(oof["score"], 0.50))
thr_high = st.sidebar.slider("고위험 기준 (이상)", 0.0, 1.0, round(default_high, 2), 0.01)
thr_mid = st.sidebar.slider("중위험 기준 (이상)", 0.0, 1.0, round(default_mid, 2), 0.01)
st.sidebar.caption("기본값: 학습 데이터 위험 점수 상위 20% / 50% 지점(초기안)")
if thr_mid > thr_high:
    st.sidebar.error("중위험 기준은 고위험 기준보다 클 수 없습니다.")
    ui.page_header(page, "임계값 설정을 확인해 주세요.")
    st.error(f"중위험 기준({thr_mid:.2f})이 고위험 기준({thr_high:.2f})보다 큽니다. "
             "사이드바에서 중위험 기준을 낮추거나 고위험 기준을 높여 주세요.")
    st.stop()
with st.sidebar.expander("사용 안내"):
    st.markdown(
        "- **위험 점수** = 2등급(저신용) 확률\n"
        f"- **{ui.badge(HIGH)}** 신규 발급 중지\n\n"
        f"- **{ui.badge(MID)}** 관리 프로그램\n\n"
        "- 모델은 의사결정 **보조** 용도입니다.", unsafe_allow_html=True)


def to_result(df_raw: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    res = pd.DataFrame(proba, columns=["P(0등급)", "P(1등급)", "P(2등급)"], index=df_raw.index)
    res["위험 점수"] = risk_score(proba)
    res["위험군"] = classify(res["위험 점수"], thr_high, thr_mid)
    res["권고 조치"] = res["위험군"].map(recommended_action)
    return res


# ---------------- 대시보드 ----------------
if page == "대시보드":
    ui.page_header("대시보드", "학습 데이터(교차검증 예측값) 기준 위험군 현황입니다. 사이드바 임계값을 바꾸면 즉시 반영됩니다.")
    summ = threshold_summary(oof["score"], oof["credit"], thr_high, thr_mid)
    share = summ.set_index("위험군")["비율"]
    ui.kpi_row([("전체 고객 수", f"{len(oof):,}명", None),
                ("고위험 비율", f"{share[HIGH]:.1%}", f"{int(summ.set_index('위험군').loc[HIGH, '인원']):,}명"),
                ("중위험 비율", f"{share[MID]:.1%}", f"{int(summ.set_index('위험군').loc[MID, '인원']):,}명"),
                ("평균 위험 점수", f"{oof['score'].mean():.3f}", None)])
    ui.show(ui.grade_count_chart(summ))
    st.subheader("위험군별 요약")
    view = summ.assign(위험군=[f"{ui.GRADE_ICON[g]} {g}" for g in summ["위험군"]])
    st.dataframe(view.style.format({"인원": "{:,}", "비율": "{:.1%}", "실제 2등급 비율": "{:.1%}",
                                    "전체 2등급 포착률": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption("실제 2등급 비율: 그 위험군 안의 실제 2등급 비중 · 포착률: 전체 2등급 중 해당 군이 잡아낸 비중")
    st.subheader("피처별 위험 현황")
    feat = st.selectbox("피처 선택", list(["income_type", "edu_type", "family_type", "house_type", "occyp_type",
                                         "gender", "car", "reality"]), format_func=lambda f: ui.FEATURE_KO[f])
    ui.show(ui.risk_by_feature_chart(oof, feat, float(oof["score"].mean())))

# ---------------- 단건 예측 ----------------
elif page == "단건 예측":
    ui.page_header("단건 예측", "고객 정보를 입력하면 위험 등급과 발급 가능 여부를 확인할 수 있습니다.")
    form_col, result_col = st.columns([5, 4], gap="large")
    with form_col:
        with st.form("single"):
            with st.container(border=True):
                st.markdown("**인적사항**")
                a, b, c = st.columns(3)
                gender = a.selectbox("성별", cats["gender"], format_func=ui.ko)
                age = b.number_input("나이(세)", 18, 100, 40)
                edu = c.selectbox("학력", cats["edu_type"], format_func=ui.ko)
                a, b, c = st.columns(3)
                family_type = a.selectbox("결혼 상태", cats["family_type"], format_func=ui.ko)
                family_size = b.number_input("가족 수", 1, 20, 2)
                child_num = c.number_input("자녀 수", 0, 19, 0)
            with st.container(border=True):
                st.markdown("**소득·직업**")
                income = st.number_input("연 소득", 0.0, 10_000_000.0, 150000.0, 5000.0, format="%.0f")
                a, b = st.columns(2)
                income_type = a.selectbox("소득 유형", cats["income_type"], format_func=ui.ko)
                occyp = b.selectbox("직업", cats["occyp_type"], index=cats["occyp_type"].index("Unknown"),
                                    format_func=ui.ko)
                a, b = st.columns(2)
                no_job = a.checkbox("근속 없음(무직·연금수급 등)")
                emp_years = b.number_input("근속 연수", 0.0, 60.0, 5.0, 0.5, disabled=no_job)
            with st.container(border=True):
                st.markdown("**자산·카드**")
                a, b, c = st.columns(3)
                house = a.selectbox("주거 형태", cats["house_type"], format_func=ui.ko)
                car = b.selectbox("차량 보유", ["N", "Y"], format_func=ui.ko)
                reality = c.selectbox("부동산 보유", ["N", "Y"], format_func=ui.ko)
                months = a.number_input("카드 발급 후 경과 개월", 0, 120, 12)
            with st.container(border=True):
                st.markdown("**연락처**")
                yn = {0: "없음", 1: "있음"}
                a, b, c = st.columns(3)
                work_phone = a.selectbox("직장 전화", [0, 1], format_func=yn.get)
                phone = b.selectbox("집 전화", [0, 1], format_func=yn.get)
                email = c.selectbox("이메일", [0, 1], format_func=yn.get)
            go = st.form_submit_button("예측하기", type="primary", width="stretch")
        if go:
            if family_size < child_num + 1:
                st.error("가족 수는 자녀 수보다 커야 합니다. 입력값을 확인해 주세요.")
                st.session_state.pop("single_row", None)
            elif not no_job and emp_years > age - MIN_EMPLOY_START_AGE:
                st.error(f"근속 연수({emp_years:g}년)가 나이({age}세)와 맞지 않습니다. "
                         f"근속 연수는 나이 - {MIN_EMPLOY_START_AGE}년 이하여야 합니다.")
                st.session_state.pop("single_row", None)
            else:
                if income == 0:
                    st.warning("연 소득이 0입니다. 입력이 맞는지 확인해 주세요.")
                for w in range_warnings(age, income, rng):
                    st.warning(w)
                st.session_state["single_row"] = pd.DataFrame([{
                    "gender": gender, "car": car, "reality": reality, "child_num": child_num,
                    "income_total": income, "income_type": income_type, "edu_type": edu,
                    "family_type": family_type, "house_type": house, "DAYS_BIRTH": -int(age * 365),
                    "DAYS_EMPLOYED": 365243 if no_job else -int(emp_years * 365), "FLAG_MOBIL": 1,
                    "work_phone": work_phone, "phone": phone, "email": email,
                    "occyp_type": None if occyp == "Unknown" else occyp,
                    "family_size": float(family_size), "begin_month": -float(months)}])
    with result_col:
        row = st.session_state.get("single_row")
        if row is None:
            st.info("왼쪽 양식을 작성하고 **예측하기**를 누르면 결과가 여기에 표시됩니다.")
        else:
            with st.spinner("예측 중..."):
                r = to_result(row, ensemble_predict(row, bundle)).iloc[0]
                factors = top_risk_factors(bundle, row, 5)[0]
            ui.result_card(r["위험군"], r["위험 점수"], r["권고 조치"])
            if r["위험군"] == HIGH:
                st.error("신규 카드 발급 중지 대상입니다.")
            elif r["위험군"] == MID:
                st.warning("관리 프로그램 편입 대상입니다.")
            else:
                st.success("정상 발급 가능합니다.")
            ui.show(ui.proba_chart([r["P(0등급)"], r["P(1등급)"], r["P(2등급)"]]))
            st.markdown("**위험을 높인 요인 Top 5**")
            factors["feature"] = factors["feature"].map(lambda f: ui.FEATURE_KO.get(f, f))
            factors["value"] = factors["value"].map(ui.ko)
            st.dataframe(factors.rename(columns={"feature": "피처", "value": "값", "shap": "기여도"}),
                         hide_index=True, width="stretch",
                         column_config={"기여도": st.column_config.NumberColumn(format="%.3f", width="small")})
            st.caption("설명은 LightGBM 폴드 모델 기준 근사치이며, 최종 점수는 앙상블 결과입니다.")

# ---------------- 일괄 예측 ----------------
elif page == "일괄 예측":
    ui.page_header("일괄 예측", "여러 고객을 CSV로 한 번에 예측하고, 중·고위험 고객 목록을 내려받을 수 있습니다.")
    with st.expander("CSV 형식 안내", expanded=True):
        st.write("필수 컬럼: " + ", ".join(f"`{c}`" for c in RAW_COLS) + " (`index` 컬럼은 선택)")
        st.download_button("양식 CSV 다운로드", pd.DataFrame([SAMPLE_ROW], columns=RAW_COLS).to_csv(index=False)
                           .encode("utf-8-sig"), "template.csv", "text/csv")
    up = st.file_uploader("CSV 파일을 끌어다 놓거나 선택하세요", type="csv")
    if up is None:
        st.info("파일을 업로드하면 예측 결과가 표시됩니다.")
        st.stop()
    try:
        raw = up.getvalue()
        for enc in ("utf-8-sig", "cp949"):  # 엑셀 저장 CSV(CP949) 대응
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("UTF-8 또는 CP949 인코딩의 CSV만 지원합니다.")
    except pd.errors.EmptyDataError:
        st.error("빈 파일입니다. 헤더와 고객 정보가 있는 CSV를 업로드해 주세요.")
        st.stop()
    except Exception as e:  # 깨진 형식 등
        st.error(f"CSV를 읽을 수 없습니다. 파일 형식을 확인해 주세요. ({e})")
        st.stop()
    if df.empty:
        st.error("데이터 행이 없습니다. 헤더 아래에 고객 정보를 채워 주세요.")
        st.stop()
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        st.error("필수 컬럼이 없습니다: " + ", ".join(missing))
        st.stop()
    df, errs, warns = validate_batch(df, cats, rng)
    if errs:
        bullet_list = "\n".join(f"- {e}" for e in errs)
        st.error(f"입력 데이터를 확인해 주세요:\n\n{bullet_list}")
        st.stop()
    for w in warns:
        st.warning(w)
    try:
        with st.spinner("예측 중..."):
            res = to_result(df, predict_cached(df))
    except Exception as e:  # 예상 못 한 모델 입력 오류도 스택 트레이스 대신 안내
        st.error(f"예측 중 오류가 발생했습니다. 데이터 형식을 확인해 주세요. ({type(e).__name__})")
        st.stop()
    ids = df[["index"]] if "index" in df.columns else pd.DataFrame(index=df.index)
    out = pd.concat([ids, res], axis=1)
    cnt = out["위험군"].value_counts()
    ui.kpi_row([("총 건수", f"{len(out):,}건", None)] +
               [(g, f"{cnt.get(g, 0):,}명", f"{cnt.get(g, 0) / len(out):.1%}") for g in LABELS])
    pick = st.multiselect("위험군 필터", LABELS, default=LABELS)
    view = out[out["위험군"].isin(pick)].sort_values("위험 점수", ascending=False).copy()
    view["위험군"] = [f"{ui.GRADE_ICON[g]} {g}" for g in view["위험군"]]
    if view.empty:
        st.info("선택한 위험군에 해당하는 고객이 없습니다.")
    else:
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "위험 점수": st.column_config.ProgressColumn("위험 점수", min_value=0.0, max_value=1.0, format="%.3f"),
            "P(0등급)": st.column_config.NumberColumn(format="%.3f"),
            "P(1등급)": st.column_config.NumberColumn(format="%.3f"),
            "P(2등급)": st.column_config.NumberColumn(format="%.3f")})
    a, b = st.columns(2)
    a.download_button("전체 결과 CSV", out.to_csv(index=False).encode("utf-8-sig"), "predictions.csv", "text/csv",
                      width="stretch")
    b.download_button("중·고위험 고객 목록 CSV",
                      out[out["위험군"].isin([HIGH, MID])].to_csv(index=False).encode("utf-8-sig"),
                      "risk_customers.csv", "text/csv", width="stretch", type="primary")

# ---------------- 모델 성능 ----------------
else:
    ui.page_header("모델 성능", "5-Fold 그룹 교차검증 기준 성능입니다. 데이터에 연체 신호가 약해 변별력은 제한적입니다.")
    comp = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    fin = comp.iloc[-1]
    ui.kpi_row([("Log Loss (최종)", f"{fin['logloss']:.4f}", None), ("Macro F1", f"{fin['macro_f1']:.3f}", None),
                ("AUC (OvR)", f"{fin['auc_ovr']:.3f}", None), ("2등급 재현율", f"{fin['class2_recall']:.3f}", None)])
    st.subheader("모델 비교")
    st.dataframe(comp.rename(columns={"model": "모델", "logloss": "Log Loss", "macro_f1": "Macro F1",
                                      "class2_recall": "2등급 재현율", "auc_ovr": "AUC"})
                 .style.format({c: "{:.4f}" for c in ["Log Loss", "Macro F1", "2등급 재현율", "AUC"]})
                 .highlight_min(subset=["Log Loss"], color="#cde2fb"), hide_index=True, width="stretch")
    st.subheader("성별 편향 점검")
    g = oof.assign(위험군=classify(oof["score"], thr_high, thr_mid))
    gt = g.groupby("gender").agg(인원=("score", "size"), 평균_위험_점수=("score", "mean"),
                                 실제_2등급_비율=("credit", lambda x: (x == 2).mean()))
    for lab in (HIGH, MID):
        gt[f"{lab}_비율"] = g.groupby("gender")["위험군"].apply(lambda x, lab=lab: (x == lab).mean())
    gt.index = gt.index.map(ui.ko)
    st.dataframe(gt.rename(columns=lambda c: c.replace("_", " ")).style.format(
        {"인원": "{:,}", "평균 위험 점수": "{:.3f}", "실제 2등급 비율": "{:.1%}", "고위험 비율": "{:.1%}", "중위험 비율": "{:.1%}"}),
        width="stretch")
    st.caption("성별에 따른 위험 점수·위험군 비율 차이가 작은지 확인하는 표입니다. 성별 제외 시 Log Loss는 0.8156으로 포함(0.8161)과 사실상 같습니다.")
    left, right = st.columns(2)
    with left:
        pred = pd.Series(oof[["p0", "p1", "p2"]].values.argmax(1), name="예측")
        ui.show(ui.confusion_chart(pd.crosstab(oof["credit"].astype(int).rename("실제"), pred)
                                   .reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)))
    with right:
        with st.spinner("SHAP 계산 중..."):
            imp = load_importance(bundle)
        ui.show(ui.importance_chart(imp))
