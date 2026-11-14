"""Streamlit UI 헬퍼: 스타일, 위험 등급 배지, Plotly 차트.

색 규칙: 위험 등급(고/중/저)은 상태 색 + 아이콘 + 라벨을 함께 쓰고, 그 외 단일 계열은 파랑 하나만 쓴다.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk import HIGH, LABELS, LOW, MID

BLUE = "#2a78d6"
NAVY = "#184f95"
GRADE_COLOR = {HIGH: "#b3261e", MID: "#b25e09", LOW: "#1e7a3c"}
GRADE_ICON = {HIGH: "▲", MID: "◆", LOW: "●"}
BLUE_SCALE = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]

FEATURE_KO = {
    "gender": "성별", "car": "차량 보유", "reality": "부동산 보유", "child_num": "자녀 수",
    "income_total": "연 소득", "income_type": "소득 유형", "edu_type": "학력", "family_type": "결혼 상태",
    "house_type": "주거 형태", "work_phone": "직장 전화", "phone": "집 전화", "email": "이메일",
    "occyp_type": "직업", "family_size": "가족 수", "is_unemployed": "근속 없음 여부",
    "age_years": "나이", "employed_years": "근속 연수", "months_since_issue": "카드 발급 후 경과 개월",
    "income_per_person": "1인당 소득", "log_income": "소득(로그)", "child_ratio": "자녀 비율",
    "employed_ratio": "근속/나이 비율",
}

VALUE_KO = {
    "F": "여성", "M": "남성", "Y": "있음", "N": "없음",
    "Commercial associate": "상업 종사자", "Pensioner": "연금 수급자", "State servant": "공무원",
    "Student": "학생", "Working": "일반 근로자",
    "Academic degree": "석·박사", "Higher education": "대졸", "Incomplete higher": "대학 중퇴",
    "Lower secondary": "중졸 이하", "Secondary / secondary special": "고졸·중등 전문",
    "Civil marriage": "사실혼", "Married": "기혼", "Separated": "별거", "Single / not married": "미혼",
    "Widow": "사별",
    "Co-op apartment": "조합주택", "House / apartment": "주택·아파트", "Municipal apartment": "공공주택",
    "Office apartment": "사무실 겸용", "Rented apartment": "임대주택", "With parents": "부모와 동거",
    "Accountants": "회계사", "Cleaning staff": "청소 인력", "Cooking staff": "조리 인력",
    "Core staff": "핵심 인력", "Drivers": "운전기사", "HR staff": "인사 담당", "High skill tech staff": "고숙련 기술직",
    "IT staff": "IT 인력", "Laborers": "노동자", "Low-skill Laborers": "저숙련 노동자", "Managers": "관리자",
    "Medicine staff": "의료 인력", "Private service staff": "개인 서비스", "Realty agents": "부동산 중개",
    "Sales staff": "영업직", "Secretaries": "비서", "Security staff": "보안 인력",
    "Waiters/barmen staff": "웨이터·바텐더", "Unknown": "미상",
}


def ko(v):
    """범주 값을 한글로 (없으면 원문)."""
    return VALUE_KO.get(str(v), str(v))


CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1 {font-size: 1.9rem !important; margin-bottom: .2rem;}
h2, h3 {margin-top: 1.4rem; font-size: 1.2rem !important;}
.kpi-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 14px;}
.kpi {background: var(--secondary-background-color); border-radius: 10px; padding: 12px 16px;
      border-left: 5px solid #184f95;}
.kpi .l {font-size: .85rem; opacity: .75;}
.kpi .v {font-size: 1.6rem; font-weight: 600; line-height: 1.3;}
.kpi .s {font-size: .8rem; opacity: .65;}
.badge {display:inline-block; padding: 6px 16px; border-radius: 999px; color:#fff;
        font-weight: 700; font-size: 1.15rem; letter-spacing: .02em;}
.result-card {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 18px 20px;
              background: var(--secondary-background-color);}
.result-card .score {font-size: 2.2rem; font-weight: 700; margin: 6px 0 2px;}
.result-card .action {margin-top: 8px;}
.caption-muted {opacity: .7; font-size: .85rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, desc: str) -> None:
    st.title(title)
    st.markdown(f"<div class='caption-muted' style='margin-bottom:1rem'>{desc}</div>", unsafe_allow_html=True)


def badge(label: str) -> str:
    return f"<span class='badge' style='background:{GRADE_COLOR[label]}'>{GRADE_ICON[label]} {label}</span>"


def result_card(label: str, score: float, action: str) -> None:
    st.markdown(
        f"<div class='result-card'>{badge(label)}<div class='score'>{score:.3f}</div>"
        f"<div class='caption-muted'>위험 점수 P(2등급)</div><div class='action'>{action}</div></div>",
        unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    cards = "".join(
        f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div>"
        f"<div class='s'>{sub or '&nbsp;'}</div></div>" for label, value, sub in items)
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)


def _layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16)), height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.2)")
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def grade_count_chart(summ: pd.DataFrame) -> go.Figure:
    d = summ.set_index("위험군").loc[LABELS].reset_index()
    fig = go.Figure(go.Bar(
        x=[f"{GRADE_ICON[g]} {g}" for g in d["위험군"]], y=d["인원"],
        marker=dict(color=[GRADE_COLOR[g] for g in d["위험군"]], cornerradius=4),
        text=[f"{n:,}명 ({r:.0%})" for n, r in zip(d["인원"], d["비율"])], textposition="outside",
        hovertemplate="%{x}<br>%{y:,}명<extra></extra>"))
    fig.update_yaxes(title="인원(명)", rangemode="tozero")
    fig.update_xaxes(title="위험군")
    return _layout(fig, "위험군별 인원")


def risk_by_feature_chart(oof: pd.DataFrame, feat: str, overall: float) -> go.Figure:
    g = (oof.assign(**{feat: oof[feat].fillna("Unknown").astype(str)}).groupby(feat)["score"]
         .agg(["mean", "size"]).sort_values("mean"))
    fig = go.Figure(go.Bar(
        x=g["mean"], y=[ko(i) for i in g.index], orientation="h", marker=dict(color=BLUE, cornerradius=4),
        text=[f"{v:.3f}" for v in g["mean"]], textposition="outside", customdata=g["size"],
        hovertemplate="%{y}<br>평균 위험 점수 %{x:.3f}<br>고객 %{customdata:,}명<extra></extra>"))
    fig.add_vline(x=overall, line_dash="dash", line_color="gray",
                  annotation_text=f"전체 평균 {overall:.3f}", annotation_position="top")
    lo = max(0.0, g["mean"].min() - 0.05)
    fig.update_xaxes(title="평균 위험 점수 P(2등급) · 축은 일부 구간만 표시", range=[lo, g["mean"].max() + 0.04])
    fig.update_yaxes(title="")
    return _layout(fig, f"{FEATURE_KO.get(feat, feat)}별 평균 위험 점수", height=max(300, 55 * len(g) + 120))


def proba_chart(p: list[float]) -> go.Figure:
    names = ["0등급", "1등급", "2등급"]
    fig = go.Figure(go.Bar(
        x=p, y=names, orientation="h", marker=dict(color=BLUE, cornerradius=4),
        text=[f"{v:.1%}" for v in p], textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(title="확률", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "등급별 확률", height=220)


def confusion_chart(cm: pd.DataFrame) -> go.Figure:
    z = cm.values
    pct = z / z.sum(axis=1, keepdims=True)
    fig = go.Figure(go.Heatmap(
        z=pct, x=[f"{c}등급" for c in cm.columns], y=[f"{i}등급" for i in cm.index], colorscale=BLUE_SCALE,
        zmin=0, zmax=1, showscale=False,
        text=[[f"{z[i, j]:,}<br>({pct[i, j]:.0%})" for j in range(z.shape[1])] for i in range(z.shape[0])],
        texttemplate="%{text}", hovertemplate="실제 %{y} → 예측 %{x}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="예측", side="bottom")
    fig.update_yaxes(title="실제", autorange="reversed")
    return _layout(fig, "혼동행렬 (행 기준 비율)", height=380)


def importance_chart(imp: pd.DataFrame, top: int = 15) -> go.Figure:
    d = imp.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["mean_abs_shap"], y=[FEATURE_KO.get(f, f) for f in d["feature"]], orientation="h",
        marker=dict(color=BLUE, cornerradius=4), hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="평균 |SHAP| (2등급 기준)")
    return _layout(fig, f"피처 중요도 상위 {top}", height=max(320, 28 * len(d) + 100))
