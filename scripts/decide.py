"""층 2·3: 시장 여유 R → 상대 시장 여유 R̃ → 사후기대 FDR 즉시 목록 (methodology_credo_model.md 4~5절).

실행: uv run python scripts/decide.py            (data/model/log_demand.npz 필요, 층 1 진단 통과본만)
산출: data/model/decision_table.csv, sensitivity.csv, coverage_curve.csv, pilot_sensitivity.csv, r_rel_draws.npz
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

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


def supply(regions: pl.DataFrame, industries, broad: bool) -> np.ndarray:
    nts = pl.read_csv(EXT / "nts_100_living_industries_sgg_202606.csv")
    out = np.zeros((regions.height, len(industries)))
    for n, b in enumerate(industries):
        names = SUPPLY[b] + (BROAD.get(b, []) if broad else [])
        s = nts.filter(pl.col("NTS_INDUSTRY").is_in(names)).group_by(KEY).agg(pl.col("NTS_CURRENT_COUNT").sum())
        out[:, n] = regions.join(s, on=KEY, how="left")["NTS_CURRENT_COUNT"].fill_null(0).to_numpy()  # 행 없음 = 사업자 0
    return out


def covariates(regions: pl.DataFrame) -> np.ndarray:
    """결정 8: 지역 고정 공변량은 층 2의 w_i (아파트 ㎡가, 국민연금 가입자, 평균 외지인·외국인 방문자)."""
    w = (
        regions.join(pl.read_csv(EXT / "apt_price_sgg_2026h1.csv").select(*KEY, "MEDIAN_PRICE_PER_M2_10K_KRW"), on=KEY, how="left")
        .join(pl.read_csv(EXT / "workers_sgg_202607.csv").select(*KEY, "NPS_WORKERS"), on=KEY, how="left")
        .join(pl.read_csv(EXT / "visitors_sgg_month_2026h1.csv").group_by(KEY).agg(pl.col("OUTSIDE_VISITORS", "FOREIGN_VISITORS").mean()), on=KEY, how="left")
        .with_columns(pl.col("MEDIAN_PRICE_PER_M2_10K_KRW").fill_null(pl.col("MEDIAN_PRICE_PER_M2_10K_KRW").median().over("SIDO_NM")))  # 옹진
        .select(pl.all().exclude(KEY).log())
        .to_numpy()
    )
    assert not np.isnan(w).any()
    return (w - w.mean(0)) / w.std(0)


def market_room(log_d: np.ndarray, log_s: np.ndarray, w: np.ndarray, rng) -> np.ndarray:
    """draw·업종마다 log S ~ 1 + log D + w 를 평탄 사전분포 켤레 회귀로 한 번 추출. R = 적합값 − log S. shape (S, I, B)."""
    S, I, B = log_d.shape
    R = np.empty_like(log_d)
    for s in range(S):
        for b in range(B):
            X = np.column_stack([np.ones(I), log_d[s, :, b], w])
            y = log_s[:, b]
            beta_hat, rss, *_ = np.linalg.lstsq(X, y, rcond=None)
            dof = I - X.shape[1]
            sigma2 = float(rss[0]) / rng.chisquare(dof)
            beta = rng.multivariate_normal(beta_hat, sigma2 * np.linalg.inv(X.T @ X))
            R[s, :, b] = X @ beta - y
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
    c = regions.join(pl.read_csv(EXT / "sgg_centroids_202606.csv").select(*KEY, "LON", "LAT"), on=KEY, how="left")
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
    w = covariates(regions)

    results = {}
    for label, broad in (("main", False), ("broad", True)):
        R = market_room(log_d, np.log1p(supply(regions, industries, broad)), w, rng)
        R_rel = R - R.mean(2, keepdims=True)  # 결정 4: 지역 안 업종 간 비교
        results[label] = (R, R_rel, decide(R_rel))

    R, R_rel, d = results["main"]
    mu, sd = R_rel.mean(0), R_rel.std(0)  # (255, 8) 사후평균·표준편차, 지식 기울기 입력
    lam = np.array([LAMBDA_SCALES[0] * np.median(sd[:, b] ** 2) for b in range(len(industries))])
    nu = np.column_stack([kg_value(mu[:, b], sd[:, b], d["r_gamma"][b], lam[b]) for b in range(len(industries))])
    pilot = np.zeros_like(d["in_set"])
    for b in range(len(industries)):
        pilot[pilot_set(nu[:, b], d["in_set"][:, b]), b] = True
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
        main_set = set(np.nonzero(pilot[:, b])[0].tolist())
        for scale in LAMBDA_SCALES[1:]:
            nu_s = kg_value(mu[:, b], sd[:, b], d["r_gamma"][b], scale * np.median(sd[:, b] ** 2))
            alt_set = set(pilot_set(nu_s, d["in_set"][:, b]).tolist())
            pilot_sens.append((name, scale, len(main_set & alt_set) / max(len(main_set | alt_set), 1)))
    pl.DataFrame(pilot_sens, schema=["b", "lambda_scale", "jaccard_vs_main"], orient="row").write_csv(out / "pilot_sensitivity.csv")

    np.savez(out / "r_rel_draws.npz", R_rel=R_rel.astype(np.float32), regions=z["regions"], industries=z["industries"],
              in_set=d["in_set"], v=d["v"], r_gamma=d["r_gamma"])

    _, Rb_rel, db = results["broad"]
    sens = []
    for b, name in enumerate(industries):
        if name not in BROAD:
            continue
        top = lambda x: set(np.argsort(-x)[: len(x) // 4])
        m, mb = R_rel[:, :, b].mean(0), Rb_rel[:, :, b].mean(0)
        a, c = d["in_set"][:, b], db["in_set"][:, b]
        sens.append((name, stats.spearmanr(m, mb).statistic, len(top(m) & top(mb)) / len(top(m) | top(mb)),
                     (a & c).sum() / max((a | c).sum(), 1)))
    pl.DataFrame(sens, schema=["b", "spearman", "jaccard_top25", "jaccard_fdr_set"], orient="row").write_csv(out / "sensitivity.csv")
    D, curves = distance_km(regions), []
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
    print(table.group_by("b").agg((pl.col("action") == "immediate").sum().alias("immediate"),
                                   (pl.col("action") == "pilot").sum().alias("pilot"),
                                   (pl.col("action") == "hold").sum().alias("hold")).sort("b"))


if __name__ == "__main__":
    main()
