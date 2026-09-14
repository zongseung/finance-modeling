"""사후 점검: 시장 여유가 이후 점포 변화(진입·생존)와 연관되는가.

예측변수 (모두 지역 안 업종 상대값, 기준 공급 S₀ = 국세청 1년 전 2025-06 사업자 수, D = 층 1 사후평균):
- supply_only : −log(S₀+1)                       (수요 정보 없음, 평균회귀·경쟁 효과 기준선)
- naive_G     : log D − log(S₀+1)                 (점포당 수요)
- R_rel       : decide.market_room (결정 10: 포아송 + 목적형 상권수요), β̂ 점추정

결과변수:
1. nts_net_growth : 국세청 1년 순증 log(S₁+1) − log(S₀+1). S₀를 예측변수와 공유해 평균회귀가 섞이므로
   supply_only와 함께 읽는다.
2. entry_YYYYMM   : 상가정보 진입률 = (새 번호 − 상호·도로명주소가 같은 번호 변경) / (기초 점포 + 1).
3. survival_1y    : 2024-06~2025-06 신규 점포가 2026-06에 남아 있는 비율(번호 또는 상호+주소), 코호트 ≥ 5.
   생존 기간(2025-06~2026-06)과 수요 기간(2026-01~06)이 겹쳐 시간 순서가 맞는 유일한 검증.
지표: 업종 평균 Spearman ρ, 지역 군집 부트스트랩 90% 구간, supply_only 대비 차이의 90% 구간.

한계: 과거 자료 사후 점검(사전 검증 아님), 1년 기간, 상가정보 등록 잡음(번호 변경 약 14.5%),
과거 파일에 구가 없는 화성·부천 등은 해당 기간에서 제외.

실행: uv run python scripts/retro_validation.py
필요: data/model/log_demand.npz, data/external/raw/sbiz_stores_{20230630,20240630,20250630,20260630}.zip
산출: data/model/retro_validation.csv
"""

import zipfile

import numpy as np
import polars as pl
from scipy import stats

from decide import EXT, KEY, NEAR, ROOT, covariates, distance_km, market_room, supply
from fetch_external import RAW, query_map, to_regions

# 결정 3의 국세청 대응과 같은 범위의 상가정보 분류
SBIZ = {"H": ["I201"], "8005": ["I202"], "8004": ["I203"], "8006": ["I204", "I205", "I206"],
        "8021": ["I21007"], "8301": ["I21001"], "4010": ["G20405"], "4020": ["G20404"]}
DATES = ["20230630", "20240630", "20250630", "20260630"]
SIDO_RENAMED = {"강원도": "강원특별자치도", "전라북도": "전북특별자치도"}
COLS = ["상가업소번호", "상호명", "상권업종중분류코드", "상권업종소분류코드", "시도명", "시군구명", "시군구코드", "법정동명", "도로명주소"]
SAME_STORE = ["상호명", "도로명주소"]
MIN_COHORT, N_BOOT = 5, 500


def load_stores(date: str, regions: pl.DataFrame) -> pl.DataFrame:
    """상가정보 한 시점을 8개 업종·공모전 지역(LAWD_CD)으로. 2026-06은 개편 후 코드를 되돌리고, 과거는 명칭으로 맞춘다."""
    with zipfile.ZipFile(RAW / f"sbiz_stores_{date}.zip") as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "상가업소번호" in z.open(n).read(300).decode("utf-8", "ignore")]
        df = pl.concat([pl.read_csv(z.open(n), columns=COLS, schema_overrides={c: pl.Utf8 for c in COLS}) for n in names])
    b = pl.lit(None, pl.Utf8)
    for code, prefixes in SBIZ.items():
        b = pl.when(pl.col("상권업종중분류코드").is_in(prefixes) | pl.col("상권업종소분류코드").is_in(prefixes)).then(pl.lit(code)).otherwise(b)
    df = df.with_columns(b.alias("b")).filter(pl.col("b").is_not_null())
    keep = ["상가업소번호", *SAME_STORE, "b", "LAWD_CD"]
    if date == DATES[-1]:
        return to_regions(query_map(regions), df.rename({"시군구코드": "QUERY_CD", "법정동명": "umdNm"})).select(keep)
    df = df.with_columns(
        pl.when(pl.col("시군구명") == "군위군").then(pl.lit("대구광역시")).otherwise(pl.col("시도명").replace(SIDO_RENAMED)).alias("SIDO_NM"),
        pl.col("시군구명").alias("CCG_NM"),
    )
    return df.join(regions, on=KEY, how="inner").select(keep)


def new_stores(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    """after에만 있는 번호 중, 사라진 번호와 상호·도로명주소가 같은 것(번호만 바뀐 점포)은 제외."""
    gone = before.join(after, on="상가업소번호", how="anti")
    fresh = after.join(before, on="상가업소번호", how="anti")
    return fresh.join(gone.select(SAME_STORE).unique(), on=SAME_STORE, how="anti")


def survived(cohort: pl.DataFrame, later: pl.DataFrame) -> pl.DataFrame:
    """later 시점에 번호가 남아 있거나, 같은 상호·도로명주소로 남아 있는 코호트 점포."""
    by_id = cohort.join(later, on="상가업소번호", how="semi")
    rest = cohort.join(by_id, on="상가업소번호", how="anti").join(later.select(SAME_STORE).unique(), on=SAME_STORE, how="semi")
    return pl.concat([by_id, rest])


def grid(df: pl.DataFrame, regions: pl.DataFrame, industries: list[str]) -> np.ndarray:
    counts = df.group_by("LAWD_CD", "b").len()
    out = np.zeros((regions.height, len(industries)))
    for n, b in enumerate(industries):
        out[:, n] = regions.join(counts.filter(pl.col("b") == b), on="LAWD_CD", how="left", maintain_order="left")["len"].fill_null(0).to_numpy()
    return out


def rel(x: np.ndarray) -> np.ndarray:
    """지역 안 업종 평균 제거(결측 셀은 제외하고 평균)."""
    with np.errstate(invalid="ignore"):
        return x - np.nanmean(np.where(np.isnan(x), np.nan, x), axis=1, keepdims=True)


def mean_rho(x: np.ndarray, y: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> float:
    return float(np.mean([stats.spearmanr(x[idx, n][mask[idx, n]], y[idx, n][mask[idx, n]]).statistic for n in range(x.shape[1])]))


def selftest() -> None:
    frame = lambda rows: pl.DataFrame(rows, schema=["상가업소번호", "상호명", "도로명주소", "b", "LAWD_CD"], orient="row")
    before = frame([("1", "가", "주소1", "H", "1"), ("2", "나", "주소2", "H", "1")])
    after = frame([("1", "가", "주소1", "H", "1"), ("9", "나", "주소2", "H", "1"), ("3", "다", "주소3", "H", "1")])
    assert new_stores(before, after)["상가업소번호"].to_list() == ["3"]  # 번호만 바뀐 "나"는 신규 아님
    later = frame([("7", "다", "주소3", "H", "1")])
    assert survived(new_stores(before, after), later)["상가업소번호"].to_list() == ["3"]  # 번호가 바뀌어도 생존
    x = np.array([[1.0, 3.0], [np.nan, 2.0]])
    assert np.allclose(rel(x)[0], [-1.0, 1.0]) and rel(x)[1, 1] == 0.0


def main() -> None:
    selftest()
    z = np.load(ROOT / "data" / "model" / "log_demand.npz")
    industries = list(z["industries"])
    regions = pl.DataFrame([r.split("|") for r in z["regions"]], schema=KEY, orient="row").join(
        pl.read_csv(EXT / "subsidy_tier_sgg_2026.csv").select(*KEY, pl.col("LAWD_CD").cast(pl.Utf8)), on=KEY, how="left", maintain_order="left")
    log_d = z["log_demand"].astype(float).mean(0)
    s0 = supply(regions, industries, False, "NTS_YEAR_AGO_COUNT")
    s1 = supply(regions, industries, False, "NTS_CURRENT_COUNT")
    dest = np.array([b not in NEAR for b in industries])
    predictors = {
        "supply_only": rel(-np.log1p(s0)),
        "naive_G": rel(log_d - np.log1p(s0)),
        "R_rel": rel(market_room(log_d[None], s0, covariates(regions), distance_km(regions), dest, rng=None)[0]),
    }

    stores = {d: load_stores(d, regions) for d in DATES}
    covered = {d: regions["LAWD_CD"].is_in(stores[d]["LAWD_CD"].unique().implode()).to_numpy() for d in DATES}
    all_rows = np.ones((regions.height, len(industries)), bool)
    outcomes = {"nts_net_growth": (rel(np.log1p(s1) - np.log1p(s0)), all_rows)}
    for d0, d1 in zip(DATES, DATES[1:]):
        rate = rel(np.log1p(grid(new_stores(stores[d0], stores[d1]), regions, industries) / (grid(stores[d0], regions, industries) + 1)))
        outcomes[f"entry_{d0[:6]}_{d1[:6]}"] = (rate, np.repeat((covered[d0] & covered[d1])[:, None], len(industries), 1))
    cohort = new_stores(stores["20240630"], stores["20250630"])
    n_cohort, n_alive = grid(cohort, regions, industries), grid(survived(cohort, stores["20260630"]), regions, industries)
    surv = rel(np.where(n_cohort >= MIN_COHORT, n_alive / np.maximum(n_cohort, 1), np.nan))
    outcomes["survival_1y"] = (surv, (covered["20240630"] & covered["20250630"])[:, None] & ~np.isnan(surv))
    print(f"신규 점포 코호트 {cohort.height:,}곳 · 2026-06 생존 {int(n_alive.sum()):,} ({n_alive.sum() / n_cohort.sum():.1%})")

    rng = np.random.default_rng(20260914)
    boots = [rng.integers(0, regions.height, regions.height) for _ in range(N_BOOT)]
    full = np.arange(regions.height)
    rows = []
    for oname, (y, mask) in outcomes.items():
        base = np.array([mean_rho(predictors["supply_only"], y, mask, i) for i in boots])
        for pname, x in predictors.items():
            bs = np.array([mean_rho(x, y, mask, i) for i in boots])
            per = [stats.spearmanr(x[mask[:, n], n], y[mask[:, n], n]).statistic for n in range(len(industries))]
            rows.append((oname, pname, int(mask.any(1).sum()), mean_rho(x, y, mask, full), *np.quantile(bs, [0.05, 0.95]),
                         *np.quantile(bs - base, [0.05, 0.95]), *per))
    out = pl.DataFrame(rows, schema=["outcome", "predictor", "n_regions", "mean_rho", "rho_q05", "rho_q95",
                                     "delta_vs_supply_q05", "delta_vs_supply_q95", *[f"rho_{b}" for b in industries]], orient="row")
    out.write_csv(ROOT / "data" / "model" / "retro_validation.csv")
    with pl.Config(tbl_rows=30, tbl_cols=9):
        print(out.select("outcome", "predictor", "n_regions", *[pl.col(c).round(3) for c in ("mean_rho", "rho_q05", "rho_q95", "delta_vs_supply_q05", "delta_vs_supply_q95")]))


if __name__ == "__main__":
    main()
