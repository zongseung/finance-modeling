"""층 2·3: 시장 여유 R → 상대 시장 여유 R̃ → 사후기대 FDR 즉시 목록, 지식 기울기, 공간 커버리지.
층 2(결정 10): 사업자 수 포아송 회귀, 목적형 업종(한식·중식·일식·서양)은 거리 감쇠 상권수요(λ=10km).

실행: uv run python scripts/decide.py            (data/model/log_demand.npz 필요, 층 1 진단 통과본만)
산출: data/model/decision_table.csv, sensitivity.csv, coverage_curve.csv, pilot_sensitivity.csv, r_rel_draws.npz, lambda_sensitivity.csv
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "processed"
KEY = ["SIDO_NM", "CCG_NM"]
# 결정 3: 주 분석 1:1 대응, 넓힌 정의는 민감도
SUPPLY = {"H": ["한식음식점"], "8005": ["중식음식점"], "8004": ["일식음식점"], "8006": ["기타외국식음식점"],
          "8021": ["분식점"], "8301": ["제과점"], "4010": ["편의점"], "4020": ["슈퍼마켓"]}
BROAD = {"H": ["기타음식점"], "8006": ["패스트푸드점", "커피음료점"], "8021": ["패스트푸드점"], "4020": ["식료품가게"]}
GAMMA, ALPHA = 0.75, 0.10  # 상위 25% 기준, 사후기대 FDR 목표
NEAR = {"4010", "4020", "8301", "8021"}  # 결정 9: 근린형(편의점·슈퍼·제과·스넥)은 자기 지역만 커버
LAMBDAS = (10.0, 5.0, 20.0)  # 목적형 거리 감쇠 λ km (주, 민감도)
PILOTS = 10  # 업종당 4주 검증 예산
LAMBDA_SCALES = (1.0, 0.25, 4.0)  # 지식 기울기 λ(팝업 관측오차) scale (주, 민감도)


def supply(regions: pl.DataFrame, industries, broad: bool, col: str = "NTS_CURRENT_COUNT") -> np.ndarray:
    """국세청 사업자 수 (255, B). col: NTS_CURRENT_COUNT(2026-06) 또는 NTS_YEAR_AGO_COUNT(2025-06)."""
    nts = pl.read_csv(EXT / "nts_100_living_industries_sgg_202606.csv")
    out = np.zeros((regions.height, len(industries)))
    for n, b in enumerate(industries):
        names = SUPPLY[b] + (BROAD.get(b, []) if broad else [])
        s = nts.filter(pl.col("NTS_INDUSTRY").is_in(names)).group_by(KEY).agg(pl.col(col).sum())
        out[:, n] = regions.join(s, on=KEY, how="left", maintain_order="left")[col].fill_null(0).to_numpy()  # 행 없음 = 사업자 0
    return out


def covariates(regions: pl.DataFrame) -> np.ndarray:
    """결정 8: 지역 고정 공변량은 층 2의 w_i (아파트 ㎡가, 국민연금 가입자, 평균 외지인·외국인 방문자)."""
    w = (
        regions.join(pl.read_csv(EXT / "apt_price_sgg_2026h1.csv").select(*KEY, "MEDIAN_PRICE_PER_M2_10K_KRW"), on=KEY, how="left", maintain_order="left")
        .join(pl.read_csv(EXT / "workers_sgg_202607.csv").select(*KEY, "NPS_WORKERS"), on=KEY, how="left", maintain_order="left")
        .join(pl.read_csv(EXT / "visitors_sgg_month_2026h1.csv").group_by(KEY).agg(pl.col("OUTSIDE_VISITORS", "FOREIGN_VISITORS").mean()), on=KEY, how="left", maintain_order="left")
        .with_columns(pl.col("MEDIAN_PRICE_PER_M2_10K_KRW").fill_null(pl.col("MEDIAN_PRICE_PER_M2_10K_KRW").median().over("SIDO_NM")))  # 옹진
        .select(pl.all().exclude(KEY).log())
        .to_numpy()
    )
    assert not np.isnan(w).any()
    return (w - w.mean(0)) / w.std(0)


def catchment(log_d: np.ndarray, dist: np.ndarray, dest: np.ndarray, lam: float = LAMBDAS[0]) -> np.ndarray:
    """결정 10: 목적형 업종 수요는 상권수요 log Σ_j exp(−d_ij/λ) D_j (λ 기본 LAMBDAS[0], 민감도는 인자로 교체), 근린형은 자기 지역. log_d (I, B)."""
    shared = logsumexp(-dist[:, :, None] / lam + log_d[None, :, :], axis=1)
    return np.where(dest, shared, log_d)


def poisson_fit(X: np.ndarray, s: np.ndarray, beta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """log E[S] = Xβ 포아송 회귀 IRLS. 반환: β̂, 근사 사후공분산 φ·(XᵀWX)⁻¹.
    점포 수는 과산포라 피어슨 산포 φ = Σ(S−μ)²/μ / (n−p) (≥1)로 공분산을 키운다(준포아송)."""
    if beta is None:
        beta = np.linalg.lstsq(X, np.log(s + 0.5), rcond=None)[0]
    for _ in range(100):
        mu = np.exp(X @ beta)
        step = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (s - mu))
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    mu = np.exp(X @ beta)
    phi = max(((s - mu) ** 2 / mu).sum() / (len(s) - X.shape[1]), 1.0)
    return beta, phi * np.linalg.inv(X.T @ (mu[:, None] * X))


def market_room(log_d: np.ndarray, s: np.ndarray, w: np.ndarray, dist: np.ndarray, dest: np.ndarray, rng, lam: float = LAMBDAS[0]) -> np.ndarray:
    """결정 10: draw·업종마다 log E[S] = a + ψ·log D(목적형은 상권수요, λ 기본 LAMBDAS[0]) + w·ρ 포아송 회귀.
    β는 근사 사후분포에서 한 번 추출(rng=None이면 β̂ 점추정). R = Xβ − log(S+0.5). shape (draws, I, B)."""
    S, I, B = log_d.shape
    R = np.empty_like(log_d)
    warm = [None] * B  # 이전 draw 해로 IRLS 시작
    for t in range(S):
        eff = catchment(log_d[t], dist, dest, lam)
        for b in range(B):
            X = np.column_stack([np.ones(I), eff[:, b], w])
            warm[b], cov = poisson_fit(X, s[:, b], warm[b])
            beta = warm[b] if rng is None else rng.multivariate_normal(warm[b], cov)
            R[t, :, b] = X @ beta - np.log(s[:, b] + 0.5)
    return R


def decide(R_rel: np.ndarray) -> dict:
    """Shen–Louis 앙상블 분위수 기준 초과확률 v, 업종별 사후기대 FDR ≤ ALPHA 최대 목록."""
    r_gamma = np.quantile(R_rel, GAMMA, axis=(0, 1))  # 업종별, 모든 draw·지역 풀링
    v = (R_rel > r_gamma).mean(0)  # (I, B)
    in_set = np.zeros_like(v, dtype=bool)
    for b in range(v.shape[1]):
        order = np.argsort(-v[:, b])
        fdr = np.cumsum(1 - v[order, b]) / np.arange(1, len(order) + 1)
        n_keep = int(np.max(np.nonzero(fdr <= ALPHA)[0], initial=-1)) + 1
        in_set[order[:n_keep], b] = True
    ranks = (-R_rel).argsort(1).argsort(1) + 1  # draw별 업종 내 순위
    return {"v": v, "in_set": in_set, "rank_lo": np.quantile(ranks, 0.05, 0), "rank_hi": np.quantile(ranks, 0.95, 0), "r_gamma": r_gamma}


def overlap_metrics(m: np.ndarray, m_alt: np.ndarray, in_set: np.ndarray, in_set_alt: np.ndarray) -> tuple[float, float, float]:
    """주 분석 대비 대안(넓은 공급 정의 또는 다른 λ)의 사후평균 R̃ 일치도: Spearman, 상위 25% Jaccard, FDR 즉시 목록 Jaccard."""
    top = lambda x: set(np.argsort(-x)[: len(x) // 4])
    return (stats.spearmanr(m, m_alt).statistic, len(top(m) & top(m_alt)) / len(top(m) | top(m_alt)),
            (in_set & in_set_alt).sum() / max((in_set | in_set_alt).sum(), 1))


def kg_value(mu: np.ndarray, sd: np.ndarray, threshold, lam) -> np.ndarray:
    """지식 기울기 ν (Frazier–Powell–Dayanik 2008, 5.3절): 4주 검증 한 곳의 기대 정보가치. sd=0이면 ν=0."""
    var = sd ** 2
    tilde = var / np.sqrt(var + lam)
    denom = np.where(sd == 0, 1.0, tilde)  # 0 나눔 회피 (sd=0인 자리는 tilde=0이라 아래서 결국 0)
    z = -np.abs(mu - threshold) / denom
    nu = tilde * (z * stats.norm.cdf(z) + stats.norm.pdf(z))
    return np.where(sd == 0, 0.0, nu)


def pilot_set(nu: np.ndarray, in_set: np.ndarray) -> np.ndarray:
    """즉시 목록 밖에서 ν>0인 지역 중 ν 상위 PILOTS개 인덱스."""
    cand = np.where(~in_set & (nu > 0))[0]
    return cand[np.argsort(-nu[cand])[:PILOTS]]


def distance_km(regions: pl.DataFrame) -> np.ndarray:
    c = regions.join(pl.read_csv(EXT / "sgg_centroids_202606.csv").select(*KEY, "LON", "LAT"), on=KEY, how="left", maintain_order="left")
    lat, lon = np.radians(c["LAT"].to_numpy()), np.radians(c["LON"].to_numpy())
    h = np.sin((lat[:, None] - lat) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin((lon[:, None] - lon) / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(h))


def coverage(room: np.ndarray, candidates: np.ndarray, cover: np.ndarray) -> tuple[list[int], np.ndarray]:
    """점진 커버링 탐욕법: FDR 즉시 목록 후보 중 기대 커버 여유를 가장 늘리는 지역을 차례로 추가.
    room (S, I): 양의 상대 시장 여유 draw, cover (I, I): c_ij. 반환: 선택 순서, draw별 커버 비율 (K, S).
    ponytail: 기대값 기준 탐욕(1-1/e 근사 보장). 하방위험(CVaR) 최적화가 필요하면 draw별 목적으로 MILP."""
    mean, cov, chosen, shares = room.mean(0), np.zeros(room.shape[1]), [], []
    for _ in range(int(candidates.sum())):
        gain = (np.maximum(cover, cov) - cov) @ mean
        gain[~candidates] = -np.inf
        gain[chosen] = -np.inf
        chosen.append(int(np.argmax(gain)))
        cov = np.maximum(cov, cover[chosen[-1]])
        shares.append(room @ cov / room.sum(1))
    return chosen, np.array(shares)


def selftest() -> None:
    D = np.array([[0, 1, 100], [1, 0, 100], [100, 100, 0]], float)  # 가까운 두 곳 + 먼 한 곳
    chosen, shares = coverage(np.tile([1.0, 1.0, 0.9], (10, 1)), np.ones(3, bool), np.exp(-D / 10))
    assert chosen[:2] == [0, 2] and shares[-1].min() > 0.999, chosen  # 쌍둥이보다 먼 곳을 먼저 커버
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.1, (400, 50, 2))
    R[:, :5, 0] += 3.0  # 뚜렷한 상위 5곳
    d = decide(R)
    assert d["in_set"][:5, 0].all() and d["in_set"][:, 0].sum() <= 12, d["in_set"][:, 0].sum()
    fdr = (1 - d["v"][d["in_set"][:, 0], 0]).mean()
    assert fdr <= ALPHA + 1e-9

    X = np.column_stack([np.ones(255), rng.normal(size=(255, 2))])  # 포아송 회귀 계수 복원
    beta_hat, cov = poisson_fit(X, rng.poisson(np.exp(X @ np.array([1.5, 0.8, -0.3]))).astype(float))
    assert np.allclose(beta_hat, [1.5, 0.8, -0.3], atol=0.1) and np.all(np.diag(cov) > 0), beta_hat
    far = np.full((3, 3), 1e6) - np.diag(np.full(3, 1e6))  # 서로 멀면 상권수요 = 자기 수요
    ld = rng.normal(size=(3, 2))
    assert np.allclose(catchment(ld, far, np.array([True, False])), ld)
    mid = np.array([[0, 10, 40], [10, 0, 40], [40, 40, 0]], float)  # 중간 거리: λ 5 vs 20이 값을 바꾸는 규모
    c5, c20 = catchment(ld, mid, np.array([True, False]), lam=5.0), catchment(ld, mid, np.array([True, False]), lam=20.0)
    assert not np.allclose(c5[:, 0], c20[:, 0])  # 목적형 열: λ가 실제로 쓰인다
    assert np.allclose(c5[:, 1], c20[:, 1])  # 근린형 열: λ와 무관(자기 수요 그대로)

    with np.errstate(all="raise"):  # 0 나눔 등 경고를 에러로 승격해 검증
        assert kg_value(0.0, 0.0, 5.0, 1.0) == 0.0  # sd=0 → ν=0
        close, far = kg_value(0.0, 0.5, 0.1, 1.0), kg_value(0.0, 0.5, 3.0, 1.0)
        assert close > far > 0, (close, far)  # 같은 sd에서 threshold에 가까울수록 ν가 크다
        small_sd, big_sd = kg_value(0.0, 0.5, 2.0, 1.0), kg_value(0.0, 2.0, 2.0, 1.0)
        assert big_sd > small_sd > 0, (small_sd, big_sd)  # 같은 |mu-threshold|에서 sd가 클수록 ν가 크다


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand", type=Path, default=ROOT / "data" / "model" / "log_demand.npz")
    ap.add_argument("--allow-unconverged", action="store_true", help="코드 점검용. 결과를 의사결정에 쓰지 말 것")
    args = ap.parse_args()
    selftest()
    z = np.load(args.demand)
    assert bool(z["passed"]) or args.allow_unconverged, "층 1 진단 실패본: 층 2·3에 사용 금지"
    log_d, industries = z["log_demand"].astype(float), list(z["industries"])
    regions = pl.DataFrame([r.split("|") for r in z["regions"]], schema=KEY, orient="row")
    rng = np.random.default_rng(20260914)
    w, dist = covariates(regions), distance_km(regions)
    dest = np.array([b not in NEAR for b in industries])

    results = {}
    for label, broad in (("main", False), ("broad", True)):
        R = market_room(log_d, supply(regions, industries, broad), w, dist, dest, rng)
        R_rel = R - R.mean(2, keepdims=True)  # 결정 4: 지역 안 업종 간 비교
        results[label] = (R, R_rel, decide(R_rel))

    R, R_rel, d = results["main"]
    mu, sd = R_rel.mean(0), R_rel.std(0)  # (255, 8) 사후평균·표준편차, 지식 기울기 입력
    # scale별 λ(팝업 관측오차 분산 근사)·ν·검증 후보 집합을 한 번씩만 계산 (주 분석 scale=1.0 재사용, 수식 중복 제거)
    kg_lam = {scale: np.array([scale * np.median(sd[:, b] ** 2) for b in range(len(industries))]) for scale in LAMBDA_SCALES}
    nu_by_scale = {scale: np.column_stack([kg_value(mu[:, b], sd[:, b], d["r_gamma"][b], kg_lam[scale][b])
                                            for b in range(len(industries))]) for scale in LAMBDA_SCALES}
    pilot_idx_by_scale = {scale: [pilot_set(nu_by_scale[scale][:, b], d["in_set"][:, b]) for b in range(len(industries))]
                           for scale in LAMBDA_SCALES}
    nu = nu_by_scale[LAMBDA_SCALES[0]]
    pilot = np.zeros_like(d["in_set"])
    for b in range(len(industries)):
        pilot[pilot_idx_by_scale[LAMBDA_SCALES[0]][b], b] = True
    action = np.where(d["in_set"], "immediate", np.where(pilot, "pilot", "hold"))

    rows = []
    for i, (sido, ccg) in enumerate(regions.iter_rows()):
        for b, name in enumerate(industries):
            rows.append((sido, ccg, name, R_rel[:, i, b].mean(), *np.quantile(R_rel[:, i, b], [0.05, 0.95]), d["v"][i, b],
                         bool(d["in_set"][i, b]), d["rank_lo"][i, b], d["rank_hi"][i, b], R[:, i, b].mean(),
                         float(nu[i, b]), str(action[i, b])))
    table = pl.DataFrame(rows, schema=[*KEY, "b", "R_rel_mean", "R_rel_q05", "R_rel_q95", "v_top25", "immediate_fdr10",
                                       "rank_q05", "rank_q95", "R_abs_mean_reference", "kg_value", "action"], orient="row")
    out = args.demand.parent
    table.sort("b", "v_top25", descending=[False, True]).write_csv(out / "decision_table.csv")

    pilot_sens = []  # 검증 후보 집합의 λ 민감도: scale 0.25 / 4.0 vs 주 분석(scale 1.0)
    for b, name in enumerate(industries):
        main_set = set(pilot_idx_by_scale[LAMBDA_SCALES[0]][b].tolist())
        for scale in LAMBDA_SCALES[1:]:
            alt_set = set(pilot_idx_by_scale[scale][b].tolist())
            pilot_sens.append((name, scale, len(main_set & alt_set) / max(len(main_set | alt_set), 1)))
    pl.DataFrame(pilot_sens, schema=["b", "lambda_scale", "jaccard_vs_main"], orient="row").write_csv(out / "pilot_sensitivity.csv")

    np.savez(out / "r_rel_draws.npz", R_rel=R_rel.astype(np.float32), regions=z["regions"], industries=z["industries"],
              in_set=d["in_set"], v=d["v"], r_gamma=d["r_gamma"])

    _, Rb_rel, db = results["broad"]
    sens = []
    for b, name in enumerate(industries):
        if name not in BROAD:
            continue
        sens.append((name, *overlap_metrics(R_rel[:, :, b].mean(0), Rb_rel[:, :, b].mean(0), d["in_set"][:, b], db["in_set"][:, b])))
    pl.DataFrame(sens, schema=["b", "spearman", "jaccard_top25", "jaccard_fdr_set"], orient="row").write_csv(out / "sensitivity.csv")

    lam_sens = []  # 상권수요 λ 민감도(목적형 업종만): 주 분석(λ=10) 대비, 주 공급으로 λ마다 새 rng
    for lam in LAMBDAS[1:]:
        R_lam = market_room(log_d, supply(regions, industries, False), w, dist, dest, np.random.default_rng(20260914), lam=lam)
        R_rel_lam = R_lam - R_lam.mean(2, keepdims=True)
        d_lam = decide(R_rel_lam)
        for b, name in enumerate(industries):
            if not dest[b]:
                continue
            spearman, jt25, jfdr = overlap_metrics(R_rel[:, :, b].mean(0), R_rel_lam[:, :, b].mean(0), d["in_set"][:, b], d_lam["in_set"][:, b])
            lam_sens.append((name, lam, spearman, jt25, jfdr, int(d_lam["in_set"][:, b].sum())))
    lam_df = pl.DataFrame(lam_sens, schema=["b", "lambda_km", "spearman", "jaccard_top25", "jaccard_fdr_set", "n_immediate"], orient="row")
    lam_df.write_csv(out / "lambda_sensitivity.csv")

    D, curves = dist, []
    for b, name in enumerate(industries):
        room = np.maximum(R_rel[:, :, b], 0)
        for lam in (0.0,) if name in NEAR else LAMBDAS:
            chosen, shares = coverage(room, d["in_set"][:, b], np.eye(len(D)) if name in NEAR else np.exp(-D / lam))
            curves += [(name, lam, k, *regions.row(i), sh.mean(), *np.quantile(sh, [0.05, 0.95]))
                       for k, (i, sh) in enumerate(zip(chosen, shares), 1)]
    curve = pl.DataFrame(curves, schema=["b", "lambda_km", "K", *KEY, "cover_share_mean", "cover_share_q05", "cover_share_q95"], orient="row")
    curve.write_csv(out / "coverage_curve.csv")
    print(table.group_by("b").agg(pl.col("immediate_fdr10").sum().alias("n_immediate")).sort("b"))
    print(curve.filter(pl.col("lambda_km").is_in([0.0, 10.0]) & pl.col("K").is_in([5, 10, 20])).select("b", "K", "cover_share_mean", "cover_share_q05", "cover_share_q95").sort("b", "K"))
    print(pl.read_csv(out / "sensitivity.csv"))
    print(lam_df)
    print(table.group_by("b").agg((pl.col("action") == "immediate").sum().alias("immediate"),
                                   (pl.col("action") == "pilot").sum().alias("pilot"),
                                   (pl.col("action") == "hold").sum().alias("hold")).sort("b"))


if __name__ == "__main__":
    main()
