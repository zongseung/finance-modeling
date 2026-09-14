"""제출 보고서 생성: data/model 산출물 → 그림(PNG) → HTML → PDF (LibreOffice).

실행: uv run python scripts/make_report.py
필요: decide.py, retro_validation.py, operating_characteristics.py, check_share_bias.py 산출물
산출: reports/CREDO_report.pdf, reports/figures/*.png
"""

import hashlib
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image

from decide import EXT, KEY, NEAR, ROOT

MODEL = ROOT / "data" / "model"
OUT = ROOT / "reports"
FIG = OUT / "figures"
KOR = {"H": "한식", "8005": "중국음식", "8004": "일식회집", "8006": "서양음식", "8021": "스넥", "8301": "제과점",
       "4010": "편의점", "4020": "슈퍼마켓", "4004": "대형할인점"}
ORDER = ["H", "8005", "8004", "8006", "8021", "8301", "4010", "4020"]
# dataviz 기본 팔레트 앞 두 칸 + 강조 해제용 회색 (validate_palette.js 통과, 청록은 대비 부족으로 제외)
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#a3a29d", "#0b0b0b", "#52514e", "#e6e5e1"
S = {"b": pl.Utf8}

plt.rcParams.update({"font.family": "NanumGothic", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.unicode_minus": False, "axes.titlesize": 10, "axes.titlecolor": INK, "font.size": 8.5, "savefig.dpi": 220})


def load() -> dict:
    t = pl.read_csv(MODEL / "decision_table.csv", schema_overrides=S)
    diag = pl.read_csv(MODEL / "diagnostics.csv")
    log = (MODEL / "fit.log").read_text()
    cells = re.search(r"셀 ([\d,]+) \(검열 ([\d,]+)\)", log)
    return {
        "table": t,
        "cal": pl.read_csv(MODEL / "calibration.csv", schema_overrides=S),
        "cov": pl.read_csv(MODEL / "coverage_curve.csv", schema_overrides=S),
        "sens": pl.read_csv(MODEL / "sensitivity.csv", schema_overrides=S),
        "pilot_sens": pl.read_csv(MODEL / "pilot_sensitivity.csv", schema_overrides=S),
        "oc": pl.read_csv(MODEL / "operating_characteristics.csv", schema_overrides=S),
        "bias": pl.read_csv(MODEL / "share_bias_check.csv", schema_overrides=S),
        "retro": pl.read_csv(MODEL / "retro_validation.csv"),
        "draws": np.load(MODEL / "r_rel_draws.npz"),
        "rhat": float(diag["r_hat"].max()), "ess_bulk": float(diag["ess_bulk"].min()), "ess_tail": float(diag["ess_tail"].min()),
        "divergences": int(re.findall(r"'divergences': (\d+)", log)[-1]),
        "cells": cells.group(1), "censored": cells.group(2),
        "centroids": pl.read_csv(EXT / "sgg_centroids_202606.csv").select(*KEY, "LON", "LAT"),
        "hash": hashlib.sha256((MODEL / "decision_table.csv").read_bytes()).hexdigest(),
    }


def savefig(fig, name: str) -> str:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def fig_map(d: dict) -> str:
    t = d["table"].join(d["centroids"], on=KEY, how="left")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4))
    for ax, b in zip(axes, ["H", "4010"]):
        g = t.filter(pl.col("b") == b)
        for action, color, size, label in (("hold", GRAY, 9, "보류"), ("pilot", ORANGE, 30, "4주 검증"), ("immediate", BLUE, 30, "개설 후보")):
            s = g.filter(pl.col("action") == action)
            ax.scatter(s["LON"], s["LAT"], s=size, c=color, edgecolors="white", linewidths=0.6, label=f"{label} ({s.height})", zorder=2 if action == "hold" else 3)
        for r in g.filter(pl.col("action") == "immediate").sort("R_rel_mean", descending=True).head(3).iter_rows(named=True):
            ax.annotate(r["CCG_NM"], (r["LON"], r["LAT"]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK)
        ax.set_title(f"{KOR[b]}: 지역별 판단 (점 = 시군구 상권 중심점)")
        ax.set_aspect(1.2)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    return savefig(fig, "fig1_decision_map.png")


def fig_coverage(d: dict) -> str:
    c = d["cov"]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9), sharey=True)
    for ax, b in zip(axes.flat, ORDER):
        lam = 0.0 if b in NEAR else 10.0
        g = c.filter((pl.col("b") == b) & (pl.col("lambda_km") == lam)).sort("K")
        ax.fill_between(g["K"], g["cover_share_q05"] * 100, g["cover_share_q95"] * 100, color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(g["K"], g["cover_share_mean"] * 100, color=BLUE, linewidth=1.6)
        k10 = g.filter(pl.col("K") == 10)
        if k10.height:
            ax.plot(10, k10["cover_share_mean"][0] * 100, "o", color=BLUE, markersize=4, markeredgecolor="white")
            ax.annotate(f"K=10: {k10['cover_share_mean'][0] * 100:.0f}%", (10, k10["cover_share_mean"][0] * 100), xytext=(5, -10), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(f"{KOR[b]} ({'자기 지역' if lam == 0 else 'λ=10km'})", fontsize=8.5)
        ax.set_ylim(0, 100)
    for ax in axes[:, 0]:
        ax.set_ylabel("커버 비율 (%)")
    for ax in axes[1]:
        ax.set_xlabel("선정 지역 수 K")
    fig.tight_layout()
    return savefig(fig, "fig3_coverage.png")


def fig_retro(d: dict) -> str:
    r = d["retro"]
    outcomes = [("nts_net_growth", "국세청 1년 순증"), ("entry_202306_202406", "진입률 23→24"), ("entry_202406_202506", "진입률 24→25"),
                ("entry_202506_202606", "진입률 25→26"), ("survival_1y", "신규 점포 1년 생존")]
    preds = [("supply_only", "공급만 (-log S)", GRAY, -0.22), ("naive_G", "점포당 수요 G", ORANGE, 0.0), ("R_rel", r"CREDO $\tilde{R}$", BLUE, 0.22)]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for y, (o, _) in enumerate(outcomes):
        for p, label, color, dy in preds:
            row = r.filter((pl.col("outcome") == o) & (pl.col("predictor") == p)).row(0, named=True)
            ax.plot([row["rho_q05"], row["rho_q95"]], [y + dy] * 2, color=color, linewidth=1.6)
            ax.plot(row["mean_rho"], y + dy, "o", color=color, markersize=5, markeredgecolor="white", label=label if y == 0 else None)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(outcomes)), [o[1] for o in outcomes])
    ax.invert_yaxis()
    ax.set_xlabel("업종 평균 순위상관 ρ (점: 추정값, 선: 지역 군집 부트스트랩 90% 구간)")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    return savefig(fig, "fig4_retro_validation.png")


def fig_calibration(d: dict) -> str:
    cal = d["cal"]
    nominal = [50, 80, 90, 95]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([45, 100], [45, 100], "--", color=MUTED, linewidth=0.8, label="완전 보정")
    for row in cal.iter_rows(named=True):
        ax.plot(nominal, [row[f"cover{n}"] * 100 for n in nominal], color=BLUE, linewidth=1.0, alpha=0.55)
    ax.plot([], [], color=BLUE, linewidth=1.0, label="업종별 실제 포함률 (9개)")
    ax.set_xlabel("명목 예측구간 (%)"), ax.set_ylabel("실제 포함률 (%)")
    ax.set_xlim(45, 100), ax.set_ylim(45, 100)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return savefig(fig, "fig5_calibration.png")


def fig_case(d: dict, case: dict) -> str:
    z = d["draws"]
    regions = [r.split("|") for r in z["regions"]]
    i = regions.index([case["SIDO_NM"], case["CCG_NM"]])
    b = list(z["industries"]).index(case["b"])
    x = z["R_rel"][:, i, b]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.hist(x, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(z["r_gamma"][b], color=INK, linestyle="--", linewidth=1.0)
    ax.annotate(f"상위 25% 기준선 r_γ = {z['r_gamma'][b]:.2f}", (z["r_gamma"][b], ax.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel(r"상대 시장 여유 $\tilde{R}$ (사후 draw 1,000개)")
    ax.set_yticks([])
    ax.grid(False)
    return savefig(fig, "fig2_case_posterior.png")


def equation(name: str, tex: str, size: int = 13) -> str:
    fig = plt.figure(figsize=(8, 0.5))
    fig.text(0.0, 0.5, tex, fontsize=size, va="center", color=INK)
    return savefig(fig, name)


def img(path: str, width: int) -> str:
    w, h = Image.open(path).size
    return f'<p align="center"><img src="{path}" width="{width}" height="{round(width * h / w)}"></p>'


def tbl(headers: list[str], rows: list[list], right: set[int] = frozenset(), width: str = "100%") -> str:
    head = "".join(f'<th bgcolor="#f0efec">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td{" align=\"right\"" if k in right else ""}>{v}</td>' for k, v in enumerate(r)) + "</tr>" for r in rows)
    return f'<table border="1" cellpadding="3" cellspacing="0" width="{width}"><tr>{head}</tr>{body}</table>'


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def build(d: dict) -> str:
    t, cov, retro = d["table"], d["cov"], d["retro"]
    counts = {b: {a: t.filter((pl.col("b") == b) & (pl.col("action") == a)).height for a in ("immediate", "pilot", "hold")} for b in ORDER}
    imm = [counts[b]["immediate"] for b in ORDER]

    def cover_at(b, k):
        lam = 0.0 if b in NEAR else 10.0
        return cov.filter((pl.col("b") == b) & (pl.col("lambda_km") == lam) & (pl.col("K") == k)).row(0, named=True)

    def retro_row(o, p):
        return retro.filter((pl.col("outcome") == o) & (pl.col("predictor") == p)).row(0, named=True)

    surv = retro_row("survival_1y", "R_rel")
    surv_per = {b: surv[f"rho_{b}"] for b in ORDER}
    best_b = max(ORDER, key=lambda b: surv_per[b])
    case = t.filter((pl.col("b") == best_b) & (pl.col("action") == "immediate")).sort("R_rel_mean", descending=True).row(0, named=True)
    pilot = t.filter((pl.col("b") == best_b) & (pl.col("action") == "pilot")).sort("kg_value", descending=True).row(0, named=True)
    h_picks = cov.filter((pl.col("b") == "H") & (pl.col("lambda_km") == 10.0)).sort("K").head(5)
    oc = d["oc"].group_by("rule").agg(pl.col("fdr_mean").min().alias("f0"), pl.col("fdr_mean").max().alias("f1"),
                                      pl.col("tpr_mean").min().alias("t0"), pl.col("tpr_mean").max().alias("t1"))
    ocr = {r["rule"]: r for r in oc.iter_rows(named=True)}
    cal = d["cal"]
    sens = {r["b"]: r for r in d["sens"].iter_rows(named=True)}
    ps = d["pilot_sens"]["jaccard_vs_main"]

    f_map, f_cov, f_retro, f_cal = fig_map(d), fig_coverage(d), fig_retro(d), fig_calibration(d)
    f_case = fig_case(d, case)
    eqs = [
        equation("eq1.png", r"$\mathrm{Pr}(Y_{igbt}=y)=\mathrm{NB}(y\mid\mu_{igbt},\phi_b)\ \ (y\geq C+1),\qquad \mathrm{Pr}(Y_{igbt}\leq C)=\sum_{k=0}^{C}\mathrm{NB}(k\mid\mu_{igbt},\phi_b)\ \ (\mathrm{suppressed})$", 11),
        equation("eq2.png", r"$\log\mu_{igbt}=\log P_{ig}+\log n_t+\alpha_b+\gamma_{bg}+\tau_{bt}+\eta_{kt}+e_b\,\delta_{tk}+x_{it}^{\top}\beta_b+u_i+v_{ib}$", 12),
        equation("eq3.png", r"$\log E[S_{ib}]=a_b+\psi_b\log D^{c}_{ib}+w_i^{\top}\rho_b,\qquad D^{c}_{ib}=\sum_j e^{-d_{ij}/\lambda}D_{jb}\ \ (\lambda=10\mathrm{km})$", 12),
        equation("eq4.png", r"$R_{ib}=\log\hat{S}_{ib}-\log(S_{ib}+0.5),\qquad \tilde{R}_{ib}=R_{ib}-\frac{1}{8}\sum_{b'}R_{ib'}$", 12),
        equation("eq5.png", r"$v_{ib}=\mathrm{Pr}(\tilde{R}_{ib}>r_{\gamma}),\qquad D^{*}=\max\{D:\ \frac{1}{D}\sum_{k\leq D}(1-v_{(k)})\leq 0.10\}$", 12),
        equation("eq6.png", r"$\nu_{ib}=\tilde{\sigma}\,f\left(-\frac{|\mu_{ib}-r_{\gamma}|}{\tilde{\sigma}}\right),\quad \tilde{\sigma}=\frac{\sigma^2}{\sqrt{\sigma^2+\lambda}},\quad f(z)=z\Phi(z)+\varphi(z)$", 12),
        equation("eq7.png", r"$\max_{|X|=K}\ E\left[\sum_j \tilde{R}^{+}_{jb}\,\max_{i\in X}c_{ij}\right],\qquad c_{ij}=\mathbf{1}[i=j]\ \mathrm{(near)}\ \ \mathrm{or}\ \ e^{-d_{ij}/10\mathrm{km}}\ \mathrm{(destination)}$", 11),
    ]
    bias_row = d["bias_summary"]

    decisions_rows = [[KOR[b], counts[b]["immediate"], counts[b]["pilot"], counts[b]["hold"], pct(cover_at(b, 10)["cover_share_mean"]),
                       pct(cover_at(b, 20)["cover_share_mean"]), f"{surv_per[b]:+.2f}"] for b in ORDER]
    retro_rows = []
    for o, label in (("nts_net_growth", "국세청 1년 순증 (2025-06→2026-06)"), ("entry_202506_202606", "상가정보 진입률 (2025-06→2026-06)"), ("survival_1y", "신규 점포 1년 생존율 (2024~25 개업 → 2026-06)")):
        cells = [label]
        for p in ("supply_only", "naive_G", "R_rel"):
            r = retro_row(o, p)
            cells.append(f"{r['mean_rho']:+.3f}<br>[{r['rho_q05']:+.3f}, {r['rho_q95']:+.3f}]")
        r = retro_row(o, "R_rel")
        cells.append(f"[{r['delta_vs_supply_q05']:+.3f}, {r['delta_vs_supply_q95']:+.3f}]")
        retro_rows.append(cells)
    cal_rows = [[KOR[r["b"]], pct(r["censored_share"]), pct(r["pred_P_le_upper"]), pct(r["cover50"], 0), pct(r["cover80"], 0), pct(r["cover90"], 0), pct(r["cover95"], 0)]
                for r in cal.iter_rows(named=True)]
    sens_rows = [[KOR[b], f"{sens[b]['spearman']:.2f}", f"{sens[b]['jaccard_top25']:.2f}", f"{sens[b]['jaccard_fdr_set']:.2f}"] for b in ("H", "8006", "8021", "4020")]
    oc_rows = [[label, f"{pct(ocr[k]['f0'])} ~ {pct(ocr[k]['f1'])}", f"{pct(ocr[k]['t0'])} ~ {pct(ocr[k]['t1'])}"]
               for k, label in (("credo_fdr", "CREDO FDR 목록"), ("rrel_topn", "R̃ 사후평균 상위 n"), ("naive_G_rel", "점포당 수요 G (지역 안 비교)"), ("naive_G", "점포당 수요 G"))]
    pick_txt = " → ".join(f"{r['CCG_NM']}({pct(r['cover_share_mean'], 0)})" for r in h_picks.iter_rows(named=True))
    card = (f"<b>{case['SIDO_NM']} {case['CCG_NM']} · {KOR[case['b']]}</b> — 판단: <b>개설 후보</b><br>"
            f"비슷한 수요·임대료·유입 조건의 지역과 비교해 이 지역은 다른 업종보다 {KOR[case['b']]} 점포가 적습니다. "
            f"상대 시장 여유 {case['R_rel_mean']:+.2f} (90% 구간 {case['R_rel_q05']:+.2f} ~ {case['R_rel_q95']:+.2f}), "
            f"업종 상위 25%일 확률 {case['v_top25']:.0%}, 전국 순위 90% 구간 {case['rank_q05']:.0f}~{case['rank_q95']:.0f}위.")
    pilot_card = (f"<b>{pilot['SIDO_NM']} {pilot['CCG_NM']} · {KOR[pilot['b']]}</b> — 판단: <b>4주 검증</b><br>"
                  f"상위 25%일 확률 {pilot['v_top25']:.0%}로 기준선 부근에 있고 불확실성이 커서, 팝업·단기 임대로 4주간 실제 매출을 확인하는 정보가치(ν = {pilot['kg_value']:.4f})가 같은 업종에서 가장 큽니다.")

    css = ("body{font-family:'NanumGothic';font-size:9.5pt;line-height:1.5;color:#0b0b0b;} h1,h2,h3{font-family:'NanumGothic';} h1{font-size:17pt;margin-top:4pt;} "
           "h2{font-size:13pt;color:#1c5cab;margin-top:10pt;} h3{font-size:10.5pt;margin-top:8pt;} td,th{font-size:8.5pt;} .note{color:#52514e;font-size:8.5pt;}")
    br = '<h2 style="page-break-before:always">'
    html = f"""<html><head><meta charset="utf-8"><title>CREDO 보고서</title><style>{css}</style></head><body>
<h1>CREDO: 불확실성 인지형 창업 입지 판단 모델</h1>
<p><b>BC카드 소비 데이터로 "어느 지역에 어떤 가게가 부족한가"를, 틀릴 확률까지 함께 알려주는 서비스</b></p>
<p>예비창업자와 소상공인 지원기관은 "이 지역에서 이 업종을 열면 살아남을까"를 묻습니다. CREDO는 전국 255개 시군구 × 8개 업종마다
BC카드 결제로 드러난 수요와 국세청 사업자 수로 드러난 공급을 비교해, 비슷한 조건의 지역보다 점포가 적은 곳을 찾습니다.
결과는 순위표가 아니라 <b>개설 후보 / 4주 검증 / 보류</b>의 세 가지 판단으로 주어지며, 각 판단에는 불확실성 구간이 붙습니다.</p>
{tbl(["핵심 결과", "수치"], [
    ["사라진(억제된) 거래 셀을 모형에 복원", f"{d['censored']}셀 복원 · 전체 {d['cells']}셀 적합, 예측 90% 구간 실제 포함률 {pct(cal['cover90'].min(), 0)}~{pct(cal['cover90'].max(), 0)}"],
    ["업종별 개설 후보 (사후기대 오탐률 ≤ 10%)", f"{min(imm)}~{max(imm)}곳 · 업종마다 4주 검증 후보 10곳"],
    ["신규 점포 1년 생존과의 연관 (과거 자료 사후 점검)", f"순위상관 {surv['mean_rho']:+.3f} [90% {surv['rho_q05']:+.3f}, {surv['rho_q95']:+.3f}] — 경쟁만 보는 기준 대비 [{surv['delta_vs_supply_q05']:+.3f}, {surv['delta_vs_supply_q95']:+.3f}]"],
    ["한식 10곳을 공간적으로 골랐을 때 전국 시장 여유 커버", f"{pct(cover_at('H', 10)['cover_share_mean'])} [{pct(cover_at('H', 10)['cover_share_q05'])}, {pct(cover_at('H', 10)['cover_share_q95'])}]"],
])}
{img(f_map, 560)}
<p class="note">그림 1. 한식·편의점 판단 지도. 파랑 = 개설 후보, 주황 = 4주 검증, 회색 = 보류. 이름은 상대 시장 여유 상위 3곳.</p>

{br}1. 문제: 카드 데이터로 창업 입지를 고를 때 생기는 세 가지 착시</h2>
<h3>(1) 사라진 작은 시장</h3>
<p>제공 데이터 242,574행에서 거래건수 최솟값은 모든 업종·성별·월에서 정확히 11입니다. 10건 이하 셀은 비식별 처리로 삭제되어,
격자의 약 24%가 비어 있습니다. 비어 있는 셀을 0으로 보거나 버리면 작은 시장의 수요가 사라집니다.
CREDO는 이 셀을 "10건 이하였다"는 정보로 우도에 넣습니다.</p>
<h3>(2) 점포당 매출은 부족의 증거가 아니다</h3>
<p>지역 기준은 가맹점 소재지로 확인됩니다(주민 1인당 한식 결제: 부산 중구 19.7건, 화성 동탄구 1.3건). 흔한 지표인 "점포당 수요 G = 수요 ÷ 점포"는
임대료가 비싸 점포가 적은 곳, 유입 인구가 많은 곳, BC카드 점유율이 높은 곳에서 모두 커집니다. 즉 공급 부족이 아니어도 높게 나옵니다.
CREDO는 수요·임대료·근로인구·방문자를 통제한 뒤 남는 점포 부족만 "시장 여유"로 봅니다.</p>
<h3>(3) 순위표는 틀릴 확률을 알려주지 않는다</h3>
<p>상위 N곳 추천은 몇 곳이 우연히 상위에 올랐는지 말해주지 않습니다. CREDO는 목록 전체의 사후기대 오탐률을 10% 이하로 통제하고,
판단이 애매한 곳은 돈을 쓰기 전에 4주 검증으로 보내며, 가까운 지역끼리 중복되지 않게 공간 커버리지로 고릅니다.</p>

<h2>2. 데이터</h2>
<h3>2.1 BC카드 제공 데이터에서 확인한 사실</h3>
{tbl(["항목", "확인 내용", "모형 반영"], [
    ["억제 기준", "cnt 최솟값 11 (업종·성별·월 전부)", "C = min(cnt) − 1 = 10을 데이터에서 유도, 검열 우도"],
    ["지역 기준", "가맹점 소재지 (도심 1인당 결제 과다)", "거주인구는 노출량, 유입은 방문자·근로인구로 통제"],
    ["업종", "한정식 관측률 2.8%, 갈비 14.3%", "일반한식과 합산해 '한식' (국세청 한식음식점 범위와 일치)"],
    ["대형할인점", "57개 군은 전 기간 거래 없음(점포 없음)", "존재 조건부 수요 + 지원금 효과의 대조군"],
    ["코드", "성별·연령 x = 누락(법인 아님), 외국인 3", "국내 개인(성별 1·2, 연령 1~6)만 사용"],
])}
<h3>2.2 결합한 외부 데이터</h3>
{tbl(["데이터", "기준", "역할"], [
    ["행정안전부 주민등록 인구 (성·연령)", "2026-06", "수요 노출량"],
    ["국세청 100대 생활업종 사업자 수", "2026-06, 1년 전", "공급 S"],
    ["한국관광공사 기초지자체 방문자수 (API)", "2026-01~06 일별", "유입 수요(월별 변동은 층 1, 평균은 층 2)"],
    ["국민연금 가입 사업장 가입자수", "2026-07", "주간 근로인구"],
    ["국토교통부 아파트 매매 실거래가 (API)", "2026-01~06", "임대료·비용 대리변수"],
    ["소상공인 상가(상권)정보 (4개 시점)", "2023-06~2026-06", "상권 중심점, 사후 점검(진입·생존)"],
    ["고유가 피해지원금 지역 등급", "2026-04~08", "5~6월 소비 교란 통제"],
])}
<h3>2.3 결합 과정에서 바로잡은 오류</h3>
<p>2026년 7월 행정구역 개편(전남광주통합특별시, 인천 제물포·영종·서해·검단구)으로 외부 자료의 명칭이 공모전 데이터와 달랐습니다.
그대로 결합하면 30개 시군구가 빠지고, 인천 서구는 잔여 행(사업자 20곳)에 붙어 실제 38,693곳이 20곳으로 잡힙니다.
CREDO는 법정동 코드로 개편을 되돌리고 제물포구를 옛 동구 법정동 비율(0.406)로 나눈 뒤, 사업자 합계 보존을 검사합니다.</p>

{br}3. 방법: 세 층으로 나눈 판단</h2>
<h3>층 1. 억제를 복원하는 수요모형 (계층 베이지안 음이항)</h3>
{img(eqs[0], 560)}{img(eqs[1], 520)}
<p>i = 시군구, g = 성·연령, b = 업종, t = 월. P = 인구, n = 월 일수, η·δ = 지원금 등급 × 월 효과(대형할인점이 대조군), x = 방문자의 월별 변동,
u·v = 지역·지역×업종 효과(합-0 제약). 16만 셀을 GPU(numpyro) NUTS로 적합하고, 사후 수요 D의 draw 1,000개를 다음 층으로 넘깁니다.</p>
<h3>층 2. 시장 여유: 비슷한 조건의 지역보다 점포가 얼마나 적은가</h3>
{img(eqs[2], 520)}{img(eqs[3], 470)}
<p>S = 국세청 사업자 수, w = 아파트 ㎡가·국민연금 가입자·평균 방문자. 사업자 수는 계수 자료라 과산포 보정 포아송 회귀를 쓰고,
한식·중식·일식·서양 같은 목적형 업종은 10km 거리 감쇠로 주변 지역 수요까지 합칩니다(편의점·슈퍼·제과·스넥은 자기 지역).
지역 안 업종 평균을 빼는 R̃은 지역마다 다른 BC카드 점유율의 영향을 상쇄합니다. 흔한 "점포당 수요" 지표는 ψ = 1, 비용·유입 통제 없음인 특수한 경우입니다.</p>
<h3>층 3. 판단: 오탐률 통제, 정보가치, 공간 커버리지</h3>
{img(eqs[4], 520)}
<p><b>개설 후보.</b> 업종별 사후 분포를 모두 합친 75% 분위를 기준선 r_γ로 두고(Shen &amp; Louis 1998), 기준선을 넘을 확률 v가 높은 순으로 목록의 평균 오탐 확률이 10%를 넘기 직전까지 담습니다(Müller 외 2004).</p>
{img(eqs[5], 470)}
<p><b>4주 검증.</b> 개설 후보 밖에서, 팝업 한 번으로 판단이 바뀔 가능성(지식 기울기, Frazier 외 2008)이 큰 10곳을 고릅니다. 기준선 근처이면서 불확실성이 큰 곳이 선택됩니다.</p>
{img(eqs[6], 560)}
<p><b>공간 커버리지.</b> 지원기관이 K곳을 고를 때 인접 지역이 겹치지 않게, 기대 커버 여유를 최대화하는 곳을 차례로 더합니다(부분모듈 탐욕법, (1 − 1/e) 근사 보장). 커버 비율은 사후 draw마다 계산해 구간으로 보고합니다.</p>

<h2>4. 결과</h2>
<h3>4.1 업종별 판단</h3>
{tbl(["업종", "개설 후보", "4주 검증", "보류", "커버 K=10", "커버 K=20", "생존 사후 점검 ρ"], decisions_rows, right={1, 2, 3, 4, 5, 6})}
<p class="note">커버 = FDR 개설 후보 중 공간 커버리지로 K곳을 골랐을 때 전국 양(+)의 시장 여유를 덮는 사후평균 비율. 생존 ρ = 5.2절의 업종별 순위상관.</p>
<h3>4.2 판단 카드 예시 (서비스 화면에 그대로 표시되는 문장)</h3>
{tbl(["판단 카드"], [[card], [pilot_card]])}
{img(f_case, 330)}
<p class="note">그림 2. {case['CCG_NM']} {KOR[case['b']]}의 상대 시장 여유 사후분포. 점선 오른쪽 면적이 개설 판단의 근거인 확률 v입니다.</p>
<h3>4.3 공간 커버리지</h3>
{img(f_cov, 560)}
<p class="note">그림 3. 선정 지역 수 K에 따른 커버 비율(선 = 사후평균, 띠 = 90% 구간). 한식 선택 순서: {pick_txt}.</p>

<h2>5. 검증</h2>
<h3>5.1 모형이 데이터를 제대로 설명하는가</h3>
<p>4체인 × 4,000회 NUTS에서 최대 R-hat {d['rhat']:.3f}, 최소 유효표본 bulk {d['ess_bulk']:.0f} / tail {d['ess_tail']:.0f}, 발산 {d['divergences']}회로 수렴 기준(R-hat &lt; 1.01, ESS ≥ 400, 발산 0)을 통과했습니다.
관측 셀이 사후예측 구간에 실제로 들어간 비율은 명목값과 거의 같고, 억제 셀 비율도 모형이 재현합니다.</p>
{tbl(["업종", "실제 억제 비율", "예측 억제 확률", "50% 구간", "80% 구간", "90% 구간", "95% 구간"], cal_rows, right={1, 2, 3, 4, 5, 6})}
{img(f_cal, 290)}
<p class="note">그림 5. 사후예측 구간 보정. 점선에 가까울수록 불확실성 표시가 정확합니다.</p>
<h3>5.2 과거 점포 변화로 한 사후 점검</h3>
{img(f_retro, 540)}
{tbl(["결과변수", "공급만 (−log S)", "점포당 수요 G", "CREDO R̃", "R̃ − 공급만 차이 90% 구간"], retro_rows, right={1, 2, 3, 4})}
<p>세 가지를 정직하게 구분합니다.</p>
<ul>
<li><b>국세청 순증</b>과의 상관은 대부분 평균회귀입니다. 수요 정보가 없는 "공급만" 기준도 비슷한 값을 냅니다.</li>
<li><b>진입률</b>은 어느 기간에서도 예측하지 못했습니다. CREDO는 진입 예측기가 아닙니다.</li>
<li><b>신규 점포 1년 생존</b>은 수요 기간과 생존 기간이 겹쳐 시간 순서가 맞는 유일한 검증입니다.
CREDO R̃만 구간이 0보다 크고 경쟁만 보는 기준보다도 유의하게 높습니다(가장 강한 업종: {KOR[best_b]} ρ {surv_per[best_b]:+.2f}). 효과 크기는 작으므로 "생존 관점의 위험 선별"로 제시합니다.
층 2 모형(포아송 + 목적형 상권수요)은 이 생존 점검 전에 정했습니다. 국세청 순증으로 변형을 비교한 뒤였지만 점수 1위가 아니라 계수 자료·공간 가정(결정 9)과의 일관성으로 골랐고, 지역 절반 분할 200회에서 점수로 고른 변형과 표본 밖 차이가 없었습니다(−0.002, 90% 구간 −0.029 ~ +0.016).</li>
</ul>
<h3>5.3 판단 규칙의 운영특성 (같은 목록 크기, 사후분포 참값 기준)</h3>
{tbl(["규칙", "오탐률", "적중률"], oc_rows, right={1, 2}, width="80%")}
<p class="note">사후분포를 참값으로 쓴 표본 내 비교이므로 CREDO에 유리합니다. CREDO와 "R̃ 상위 n"이 같다는 것은 개선이 순위 규칙이 아니라 추정 대상(R̃)에서 나온다는 뜻입니다.</p>
<h3>5.4 강건성</h3>
{tbl(["업종", "대응표 확장 시 순위상관", "상위 25% 겹침", "개설 목록 겹침"], sens_rows, right={1, 2, 3}, width="80%")}
<p>국세청 업종 대응을 넓혀도(한식 + 기타음식점, 서양 + 패스트푸드·커피 등) 한식·스넥은 안정적이고, 서양음식·슈퍼마켓은 해석에 주의가 필요합니다.
4주 검증 후보는 팝업 관측오차 가정을 1/4배~4배로 바꿔도 겹침 {ps.min():.2f}~{ps.max():.2f}입니다.
고령 BC카드 보유자 구성 편향 점검에서는 계수 κ = {bias_row['kappa']:+.2f} (결합 90% 구간 {bias_row['lo']:+.2f} ~ {bias_row['hi']:+.2f})로 가설 방향의 연관이 있지만
설명력이 {pct(bias_row['r2'])}로 작습니다(기준 5%). 이 연관을 제거해도 업종별 상위 25% 목록 겹침은 {d['bias']['jaccard_top25_sensitivity'].min():.2f}~{d['bias']['jaccard_top25_sensitivity'].max():.2f}로 대부분 유지됩니다.</p>

<h2>6. 서비스 설계</h2>
<h3>6.1 예비창업자 화면</h3>
<ol>
<li><b>업종 선택</b> → 전국 지도에 개설 후보·4주 검증·보류가 표시됩니다(그림 1).</li>
<li><b>지역 선택</b> → 판단 카드(4.2절)가 근거 문장, 확률, 순위 구간과 함께 나타납니다. 문장은 결과표에서 템플릿으로 자동 생성되며, 원자료를 외부 AI에 넣지 않습니다.</li>
<li><b>4주 검증 안내</b> → 검증 후보 지역은 팝업 매장·공유주방·단기 임대로 4주 매출을 확인한 뒤 결정하도록 안내합니다. 검증 결과가 들어오면 같은 업종·인접 지역의 판단도 갱신됩니다.</li>
</ol>
<h3>6.2 지원기관 화면</h3>
<p>예산으로 지원할 지역 수 K를 입력하면, 공간 커버리지로 서로 겹치지 않는 K곳과 기대 커버 비율(구간 포함)을 보여줍니다(그림 3). 창업 교육·상권 컨설팅·임차료 지원의 대상 지역 선정에 쓸 수 있습니다.</p>
<h3>6.3 운영</h3>
<p>BC카드 월별 집계, 국세청(월간)·상가정보(분기) 갱신에 맞춰 층 1~3을 다시 실행합니다. 층 2·3은 수 분 이내이고 층 1은 GPU로 수 시간입니다.
재현 코드: github.com/zongseung/finance-modeling</p>

<h2>7. 기대효과와 사전 등록 검증</h2>
<ul>
<li><b>잘못된 신호에 따른 진입 방지:</b> 임대료·유입을 통제하지 않은 점포당 수요 G로 같은 크기 목록을 만들면 시장 여유 기준 상위와 {pct(ocr['naive_G']['f0'], 0)}~{pct(ocr['naive_G']['f1'], 0)}가 어긋납니다(5.3절, 표본 내 비교). 비싼 임대료나 관광 유입 때문에 점포당 매출이 높아 보이는 곳을 부족 지역으로 오인하지 않게 합니다.</li>
<li><b>검증 후 투자:</b> 애매한 곳을 4주 검증으로 돌려 초기 자본 투입 전 정보를 얻습니다.</li>
<li><b>사전 등록:</b> 이 보고서의 판단표(decision_table.csv, SHA-256 {d['hash'][:16]}…)를 고정하고, 국세청 2026-07·08 자료와 상가정보 2026-09 자료로 개설 후보 지역의 신규 점포 생존·진입을 비후보와 비교하겠습니다.</li>
</ul>

<h2>8. 한계</h2>
<ul>
<li>BC카드 관측 수요이며 전체 매출이 아닙니다. 기간 중 우리카드 독자망 전환으로 BC망 점유율이 떨어졌습니다. 지역별 차이는 R̃로 상쇄하지만 업종별 점유율 차이는 남습니다.</li>
<li>지역 기준과 억제 기준은 공식 코드북이 아닌 데이터로 추정했습니다.</li>
<li>사후 점검은 과거 1년 자료이며 생존 연관의 효과 크기는 작습니다. 진입은 예측하지 못했습니다.</li>
<li>층 2는 6개월 횡단면 2단계 추정이며, 점포 수의 내생성이 남습니다.</li>
<li>서양음식·슈퍼마켓은 업종 대응에 따라 순위가 흔들립니다.</li>
</ul>

{br}부록</h2>
<h3>A. 주요 설계 결정</h3>
{tbl(["결정", "근거"], [
    ["억제 기준을 데이터에서 유도하고 검사로 고정", "업종·성별·월 최솟값이 모두 11"],
    ["한정식·갈비는 한식으로 합산, 대형할인점은 지원금 대조군", "관측률 2.8%/14.3%, 마트 없는 57개 군"],
    ["공급은 국세청 업종 이름 기준 1:1, 넓힌 정의는 민감도", "규모 통제 편상관 1위와 일치"],
    ["결정 지표는 지역 안 업종 간 상대값 R̃", "R 분산의 45%가 지역 공통(점유율 편향과 구분 불가)"],
    ["시도 가구소득 제외", "시군구 소비율 분산 설명력 0.1~5.6%, 실제 소득연도 2024"],
    ["근로인구 = 국민연금 가입자", "SGIS 사업체통계는 2019년까지"],
    ["합-0 centered 효과 + GPU NUTS", "non-centered는 R-hat 1.07로 실패"],
    ["지역 고정 공변량은 층 2로", "층 1에서는 지역효과와 식별 불가(샘플러 궤적 2배)"],
    ["근린형은 자기 지역, 목적형은 10km 감쇠", "편의점 상권은 시군구보다 작음"],
    ["층 2 포아송 + 목적형 상권수요 (생존 점검 전 확정)", "계수 자료·결정 9와 일관, 분할 표본에서 선택 낙관 없음"],
])}
<h3>B. 데이터 출처</h3>
<p class="note">BC카드 AI금융빅데이터플랫폼 공모전 제공 데이터(재배포 금지) · 행정안전부 주민등록 인구통계(jumin.mois.go.kr) · 공공데이터포털: 국세청 100대 생활업종(15061118),
소상공인시장진흥공단 상가(상권)정보(15083033), 국민연금공단 가입 사업장 내역(15083277), 국토교통부 전국 법정동(15063424)·아파트 매매 실거래가 상세,
한국관광공사 빅데이터 지역별 방문자수(15101972) · 행정안전부 인구감소지역·고유가 피해지원금 안내</p>
<h3>C. 참고문헌</h3>
<p class="note">Müller, Parmigiani, Robert, Rousseau (2004) JASA 99:990–1001 · Shen &amp; Louis (1998) JRSS-B 60:455–471 · Lin, Louis, Paddock, Ridgeway (2006) Bayesian Analysis 1:915–946 ·
Frazier, Powell, Dayanik (2008) SIAM J. Control Optim. 47:2410–2439 · Berry &amp; Waldfogel (1999) RAND J. Econ. 30:397–420 · Schaumans &amp; Verboven (2015) REStat 97:195–209 ·
Carree &amp; Dejardin (2007) Small Business Economics 29:203–212</p>
</body></html>"""
    return html


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    out = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "check_share_bias.py")], capture_output=True, text=True, check=True).stdout
    m = re.search(r"kappa 사후평균=([-\d.]+) 결합 90% 구간=\[([-\d.]+), ([-\d.]+)\] 부분 R² 평균=([\d.]+)", out)
    d["bias_summary"] = {"kappa": float(m.group(1)), "lo": float(m.group(2)), "hi": float(m.group(3)), "r2": float(m.group(4))}
    html_path = OUT / "CREDO_report.html"
    html_path.write_text(build(d), encoding="utf-8")
    subprocess.run(["soffice", "--headless", "--infilter=HTML (StarWriter)", "--convert-to", "pdf", "--outdir", str(OUT), str(html_path)],
                   check=True, capture_output=True, timeout=300)
    print(f"보고서: {OUT / 'CREDO_report.pdf'}")


if __name__ == "__main__":
    main()
