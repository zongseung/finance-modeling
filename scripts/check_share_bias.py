"""진단: BC 점유율 연령 구성 편향 (methodology_credo_model.md 상위 한식(H) 목록이 고령 카드보유자 과대표의
인공물인지 점검). 가설: 고령 BC 카드 보유자가 인구 대비 많은 지역에서는 고령 고객 비중이 큰 업종의 BC
거래가 부풀어 그 업종의 R̃이 커진다(κ>0).

한계:
- o_i(고령 과대표 지수)는 노령화 정도, 고령층의 현금성 소비 습관, 가맹점 소재지와 거주지 불일치,
  인구 분모에 포함된 아동·청소년 등 여러 요인과 뒤섞여 있다(confounded). κ가 0과 다르지 않게
  나온다고 해서 이 경로의 편향을 배제할 수 있는 것은 아니며, 아래 판정은 어디까지나 "이 지수로
  검출되는 신호"에 대한 것이다.
- share_bias_check.csv의 jaccard_top25_sensitivity·n_depop_top25_sensitivity 열은 편향을
  "보정"한 결과가 아니라, 적합된 κ 연관성을 제거했을 때 상위 목록이 얼마나 바뀌는지 보는
  민감도 점검(sensitivity check against the fitted association)이다.

실행: .venv/bin/python scripts/check_share_bias.py   (data/model/r_rel_draws.npz 필요)
산출: data/model/share_bias_check.csv
"""

import numpy as np
import polars as pl

from decide import EXT, KEY, ROOT

HANSIK = ["8001", "8002", "8003"]  # H = 일반한식 + 갈비 + 한정식 (fit_demand.py 결정 2와 동일)
TOP_FRAC = 4  # 상위 1/4: 255 // 4 = 63곳 (operating_characteristics.py와 동일 관행)
DEPOP_SUBSIDY = 20  # SUBSIDY_10K_KRW >= 20 을 인구감소지역으로 취급
Z90 = 1.645  # 표준정규 90% 양측 임계값


def load_bc(industries: list[str]) -> pl.DataFrame:
    """국내 개인(GENDER_CD 1/2, AGE_CD 1~6) BC 원자료를 지역×연령×업종으로 즉시 집계(집계값만 반환, 원시 행 없음).
    8001·8002·8003 → H, npz의 8개 업종만 남긴다(대형할인점 4004 제외)."""
    raw = pl.read_csv(ROOT / "ABP_CONTEST_DATA.csv", infer_schema_length=0).with_columns(pl.col("cnt").cast(pl.Int64))
    return (
        raw.filter(pl.col("GENDER_CD").is_in(["1", "2"]) & pl.col("AGE_CD").is_in([str(a) for a in range(1, 7)]))
        .with_columns(pl.when(pl.col("TP_BUZ_NO").is_in(HANSIK)).then(pl.lit("H")).otherwise("TP_BUZ_NO").alias("b"))
        .filter(pl.col("b").is_in(industries))
        .group_by(*KEY, "AGE_CD", "b")
        .agg(pl.col("cnt").sum())
    )


def over_index(bc: pl.DataFrame, regions: pl.DataFrame) -> np.ndarray:
    """지역 i의 고령 과대표 지수 o_i = log(BC 거래 중 AGE_CD 6 비중 / 인구 중 AGE_CD 6 비중), 8개 업종 합산 cnt 기준."""
    bc_age6 = bc.filter(pl.col("AGE_CD") == "6").group_by(KEY).agg(pl.col("cnt").sum().alias("age6"))
    bc_total = bc.group_by(KEY).agg(pl.col("cnt").sum().alias("total"))
    pop = pl.read_csv(EXT / "population_sgg_age_sex_202606.csv")
    pop_age6 = pop.filter(pl.col("AGE_CD") == 6).group_by(KEY).agg(pl.col("POPULATION").sum().alias("pop_age6"))
    pop_total = pop.group_by(KEY).agg(pl.col("POPULATION").sum().alias("pop_total"))
    j = regions.join(bc_age6, on=KEY, how="left", maintain_order="left").join(bc_total, on=KEY, how="left", maintain_order="left") \
        .join(pop_age6, on=KEY, how="left", maintain_order="left").join(pop_total, on=KEY, how="left", maintain_order="left")
    assert j.null_count().sum_horizontal().item() == 0, "지역 매칭 누락"
    bc_share = j["age6"].to_numpy() / j["total"].to_numpy()
    pop_share = j["pop_age6"].to_numpy() / j["pop_total"].to_numpy()
    return np.log(bc_share / pop_share)


def age_slope(bc: pl.DataFrame, industries: list[str]) -> np.ndarray:
    """업종 b의 고령 기울기 t_b = 전국 업종 b 거래 중 AGE_CD 6 비중 - 전국 8개 업종 합산 거래 중 AGE_CD 6 비중."""
    nat = bc.group_by("b").agg(pl.col("cnt").sum().alias("total"), pl.col("cnt").filter(pl.col("AGE_CD") == "6").sum().alias("age6"))
    combined_share = nat["age6"].sum() / nat["total"].sum()
    share = dict(zip(nat["b"], nat["age6"] / nat["total"]))
    return np.array([share[b] - combined_share for b in industries])


def fit_kappa(R_rel: np.ndarray, x_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """draw마다 R̃(I×B)을 [업종 더미, x̃]에 OLS: 더미 잔차화(업종별 지역 평균 제거)로 계산한 닫힌 형태.
    κ = Σ r·x_resid / Σ x_resid², 부분 R² = 1 - RSS_full/RSS_dummies. 반환 (kappa, partial_r2), draw별 (S,)."""
    r = R_rel - R_rel.mean(axis=1, keepdims=True)  # 업종별 지역 평균 제거 (더미 잔차)
    x_resid = x_tilde - x_tilde.mean(axis=0, keepdims=True)
    denom = (x_resid ** 2).sum()
    kappa = (r * x_resid[None, :, :]).sum(axis=(1, 2)) / denom
    rss_dummies = (r ** 2).sum(axis=(1, 2))
    rss_full = ((r - kappa[:, None, None] * x_resid[None, :, :]) ** 2).sum(axis=(1, 2))
    return kappa, 1 - rss_full / rss_dummies


def selftest() -> None:
    rng = np.random.default_rng(0)
    I, B = 255, 8
    o, t = rng.normal(size=I) - 0.5, rng.normal(size=B)  # o 평균을 -0.5로 이동: 업종 평균 제거 누락을 검출
    x = o[:, None] * t[None, :]
    x_tilde = x - x.mean(1, keepdims=True)  # 지역 안 업종 평균 중심화
    industry_effect = rng.normal(size=B) * 10  # 업종 효과를 크게 키워 잔차화 누락 시 κ가 크게 어긋나게 함
    R = 2 * x_tilde + industry_effect[None, :] + rng.normal(0, 0.1, (I, B))  # eps ~ N(0, 0.01)
    kappa, r2 = fit_kappa(R[None, :, :], x_tilde)
    assert abs(kappa[0] - 2.0) <= 0.05, kappa  # 추정 κ가 2 ± 0.05
    assert r2[0] > 0.9, r2

    # 독립 검산: [업종 더미 8개, x̃] 설계행렬에 대한 일반 OLS(np.linalg.lstsq)와 1e-8까지 일치해야 한다
    dummies = np.eye(B)[np.tile(np.arange(B), I)]
    X = np.column_stack([dummies, x_tilde.reshape(-1)])
    beta, *_ = np.linalg.lstsq(X, R.reshape(-1), rcond=None)
    assert abs(beta[-1] - kappa[0]) < 1e-8, (beta[-1], kappa[0])


def main() -> None:
    selftest()
    out = ROOT / "data" / "model"
    z = np.load(out / "r_rel_draws.npz")
    R_rel, industries = z["R_rel"].astype(float), list(z["industries"])
    regions = pl.DataFrame([r.split("|") for r in z["regions"]], schema=KEY, orient="row")
    n_top = regions.height // TOP_FRAC

    bc = load_bc(industries)
    o = over_index(bc, regions)
    t = age_slope(bc, industries)
    x = o[:, None] * t[None, :]
    x_tilde = x - x.mean(1, keepdims=True)

    kappa, partial_r2 = fit_kappa(R_rel, x_tilde)
    k_mean = kappa.mean()
    r2_mean = partial_r2.mean()

    # Important 1 fix: draw 간 분산만으로는 지역 안 8개 업종이 o_i를 공유해서 생기는 회귀
    # 표집오차(지역 군집오차)를 놓친다. R_mean 수준에서 군집-강건 SE를 더해 결합 구간을 낸다.
    xr = x_tilde - x_tilde.mean(0, keepdims=True)  # fit_kappa와 같은 더미 잔차화(업종별 지역 평균 제거)
    R_mean = R_rel.mean(0)  # (255, 8) 사후평균
    r_resid = R_mean - R_mean.mean(0, keepdims=True)
    e = r_resid - k_mean * xr
    g = (xr * e).sum(axis=1)  # 지역별 추정방정식 기여도(업종 합산): o_i가 지역 안에서 공유되므로 지역 단위로 군집화
    se_cluster = np.sqrt((g ** 2).sum()) / (xr ** 2).sum()
    combined_sd = np.sqrt(kappa.var() + se_cluster ** 2)
    k_lo, k_hi = k_mean - Z90 * combined_sd, k_mean + Z90 * combined_sd
    print(f"kappa 사후평균={k_mean:.4f} 결합 90% 구간=[{k_lo:.4f}, {k_hi:.4f}] 부분 R² 평균={r2_mean:.4f}")

    R_sens = R_mean - k_mean * x_tilde  # 민감도 점검: 적합된 κ 연관성을 제거한 가상의 R̃ (편향 보정이 아님)

    depop = regions.join(pl.read_csv(EXT / "subsidy_tier_sgg_2026.csv").select(*KEY, "SUBSIDY_10K_KRW"), on=KEY, how="left", maintain_order="left")
    assert depop["SUBSIDY_10K_KRW"].null_count() == 0, "지역 매칭 누락"
    depop = (depop["SUBSIDY_10K_KRW"] >= DEPOP_SUBSIDY).to_numpy()

    rows = []
    for b, name in enumerate(industries):
        before, after = np.zeros(regions.height, bool), np.zeros(regions.height, bool)
        before[np.argsort(-R_mean[:, b])[:n_top]] = True
        after[np.argsort(-R_sens[:, b])[:n_top]] = True
        jaccard = (before & after).sum() / (before | after).sum()
        rows.append((name, float(t[b]), jaccard, int((before & depop).sum()), int((after & depop).sum())))

    table = pl.DataFrame(rows, schema=["b", "t_b", "jaccard_top25_sensitivity", "n_depop_top25_before", "n_depop_top25_sensitivity"], orient="row")
    table.write_csv(out / "share_bias_check.csv")
    print(table)

    if k_lo > 0 and r2_mean >= 0.05:
        verdict = "가설 방향(κ>0) 편향 신호 있음"
    elif k_hi < 0 and r2_mean >= 0.05:
        verdict = "가설과 반대 방향 연관(κ<0)"
    else:
        verdict = "이 지수로는 편향 신호가 검출되지 않음"
    print(verdict)


if __name__ == "__main__":
    main()
