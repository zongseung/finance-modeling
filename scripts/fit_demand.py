"""층 1: 검열 음이항 수요모형 (claudedocs/methodology_credo_model.md 2~3절).

실행: uv run python scripts/fit_demand.py --chains 4 --tune 2000 --draws 2000
산출: data/model/ (posterior.npz, log_demand.npz, diagnostics.csv, calibration.csv)
"""

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "processed"
OUT = ROOT / "data" / "model"
HANSIK = ["8001", "8002", "8003"]  # 결정 2: 한식 = 일반한식 + 갈비 + 한정식
MART = "4004"  # 결정 2: 존재 조건부 + 지원금 대조군
INDUSTRIES = ["H", "8005", "8004", "8006", "8021", "8301", "4010", "4020", MART]
MONTHS = ["202601", "202602", "202603", "202604", "202605", "202606"]
DAYS = np.array([31, 28, 31, 30, 31, 30])
SUBSIDY_MONTHS = 2  # 5~6월
TIERS = [10, 15, 20, 25]
KEY = ["SIDO_NM", "CCG_NM"]


def build_frame() -> tuple[pl.DataFrame, int]:
    raw = pl.read_csv(ROOT / "ABP_CONTEST_DATA.csv", infer_schema_length=0).with_columns(pl.col("cnt").cast(pl.Int64))
    C = int(raw["cnt"].min()) - 1  # 결정 1: 데이터에서 유도
    for axis in ("TP_BUZ_NO", "GENDER_CD", "STRD_YYMM"):  # 조합 단위로는 작은 셀이 우연히 없을 수 있어 축별로 검사
        mins = raw.group_by(axis).agg(pl.col("cnt").min())["cnt"]
        assert (mins == C + 1).all(), f"억제 기준(최소 cnt)이 {axis}마다 다르다"

    bc = (
        raw.filter(pl.col("GENDER_CD").is_in(["1", "2"]) & pl.col("AGE_CD").is_in([str(a) for a in range(1, 7)]))
        .with_columns(pl.when(pl.col("TP_BUZ_NO").is_in(HANSIK)).then(pl.lit("H")).otherwise("TP_BUZ_NO").alias("b"))
        .group_by(*KEY, "GENDER_CD", "AGE_CD", "STRD_YYMM", "b")
        .agg(pl.col("cnt").sum().alias("y"), (pl.col("TP_BUZ_NO") == "8001").any().alias("has8001"))
    )
    assert bc.filter(pl.col("b") == "H")["has8001"].all(), "갈비·한정식이 일반한식 없이 관측된 셀이 있다(상한 3C 전제 위반)"

    regions = pl.read_csv(EXT / "subsidy_tier_sgg_2026.csv").select(*KEY, "SUBSIDY_10K_KRW")
    mart_present = bc.filter(pl.col("b") == MART).select(KEY).unique().with_columns(pl.lit(True).alias("mart"))
    grid = (
        regions.join(mart_present, on=KEY, how="left")
        .join(pl.DataFrame({"GENDER_CD": ["1", "2"]}), how="cross")
        .join(pl.DataFrame({"AGE_CD": [str(a) for a in range(1, 7)]}), how="cross")
        .join(pl.DataFrame({"STRD_YYMM": MONTHS}), how="cross")
        .join(pl.DataFrame({"b": INDUSTRIES}), how="cross")
        .filter((pl.col("b") != MART) | pl.col("mart").fill_null(False))  # 마트 없는 지역은 우도에서 제외
        .join(bc.drop("has8001"), on=[*KEY, "GENDER_CD", "AGE_CD", "STRD_YYMM", "b"], how="left")
        .with_columns(
            pl.col("y").is_null().alias("censored"),
            pl.when(pl.col("b") == "H").then(3 * C).otherwise(C).alias("upper"),
        )
    )

    pop = pl.read_csv(EXT / "population_sgg_age_sex_202606.csv", schema_overrides={"GENDER_CD": pl.Utf8, "AGE_CD": pl.Utf8})
    vis = pl.read_csv(EXT / "visitors_sgg_month_2026h1.csv", schema_overrides={"STRD_YYMM": pl.Utf8})
    vis = (  # 화성 4개 구 1월(신설 전)은 해당 구 2~6월 평균
        regions.select(KEY).join(pl.DataFrame({"STRD_YYMM": MONTHS}), how="cross")
        .join(vis.select(*KEY, "STRD_YYMM", "OUTSIDE_VISITORS", "FOREIGN_VISITORS"), on=[*KEY, "STRD_YYMM"], how="left")
        .with_columns(pl.col("OUTSIDE_VISITORS", "FOREIGN_VISITORS").fill_null(pl.col("OUTSIDE_VISITORS", "FOREIGN_VISITORS").mean().over(KEY)))
    )
    # 지역 고정 공변량은 u_i·v_ib와 사전분포로만 구분돼 식별되지 않는다(능선 → NUTS 궤적 폭증).
    # 층 1에는 방문자의 지역 내 월별 편차만 넣고, 지역 수준(평균 방문자·근로인구)은 층 2의 w_i로 쓴다.
    dev = lambda c: pl.col(c).log() - pl.col(c).log().mean().over(KEY)
    vis = vis.with_columns(dev("OUTSIDE_VISITORS").alias("x_out"), dev("FOREIGN_VISITORS").alias("x_for"))
    vis = vis.with_columns((pl.col(c) / pl.col(c).std()) for c in ("x_out", "x_for"))

    frame = (
        grid.join(pop.select(*KEY, "GENDER_CD", "AGE_CD", "POPULATION"), on=[*KEY, "GENDER_CD", "AGE_CD"], how="left")
        .join(vis.select(*KEY, "STRD_YYMM", "x_out", "x_for"), on=[*KEY, "STRD_YYMM"], how="left")
        .sort(*KEY, "b", "GENDER_CD", "AGE_CD", "STRD_YYMM")
    )
    assert frame.select(pl.col("POPULATION", "x_out", "x_for").null_count()).sum_horizontal().item() == 0
    assert frame["POPULATION"].min() > 0
    return frame, C


def indices(frame: pl.DataFrame) -> dict:
    regions = frame.select(KEY).unique(maintain_order=True)
    rid = {r: n for n, r in enumerate(regions.iter_rows())}
    return {
        "regions": regions,
        "i": np.array([rid[r] for r in frame.select(KEY).iter_rows()]),
        "b": frame["b"].replace_strict({b: n for n, b in enumerate(INDUSTRIES)}, return_dtype=pl.Int64).to_numpy(),
        "g": ((frame["GENDER_CD"].cast(pl.Int64) - 1) * 6 + frame["AGE_CD"].cast(pl.Int64) - 1).to_numpy(),
        "t": frame["STRD_YYMM"].replace_strict({m: n for n, m in enumerate(MONTHS)}, return_dtype=pl.Int64).to_numpy(),
        "k": frame["SUBSIDY_10K_KRW"].replace_strict({v: n for n, v in enumerate(TIERS)}, return_dtype=pl.Int64).to_numpy(),
        "X": frame.select("x_out", "x_for").to_numpy(),
        "offset": np.log(frame["POPULATION"].to_numpy()) + np.log(DAYS[frame["STRD_YYMM"].replace_strict({m: n for n, m in enumerate(MONTHS)}, return_dtype=pl.Int64).to_numpy()]),
    }


def build_model(frame: pl.DataFrame, C: int, ix: dict) -> pm.Model:
    obs = ~frame["censored"].to_numpy()
    y = frame["y"].fill_null(0).to_numpy()
    b = ix["b"]
    rate0 = [np.log(y[obs & (b == n)].sum() / np.exp(ix["offset"][obs & (b == n)]).sum()) for n in range(len(INDUSTRIES))]
    eligible = np.array([bb != MART for bb in INDUSTRIES], dtype=float)
    coords = {"b": INDUSTRIES, "g": [f"{s}{a}" for s in "MF" for a in range(1, 7)], "t": MONTHS, "t_sub": MONTHS[-SUBSIDY_MONTHS:],
              "k": TIERS, "x": ["dlog_outside", "dlog_foreign"],
              "i": [f"{s} {c}" for s, c in ix["regions"].iter_rows()]}

    with pm.Model(coords=coords) as model:
        # 결정 7: 데이터가 강하므로 centered 합-0 효과
        alpha = pm.Normal("alpha", mu=np.array(rate0), sigma=1.0, dims="b")
        gamma = pm.ZeroSumNormal("gamma", sigma=pm.HalfNormal("sigma_gamma", 0.5), dims=("b", "g"))
        tau = pm.ZeroSumNormal("tau", sigma=pm.HalfNormal("sigma_tau", 0.2), dims=("b", "t"))
        eta = pm.ZeroSumNormal("eta", sigma=pm.HalfNormal("sigma_eta", 0.1), dims=("k", "t"), n_zerosum_axes=2)
        delta = pm.ZeroSumNormal("delta", sigma=0.1, dims=("t_sub", "k"))  # 등급 간 차이만 식별(마트 대조)
        beta = pm.Normal("beta", 0.0, 0.5, dims=("b", "x"))
        u = pm.ZeroSumNormal("u", sigma=pm.HalfNormal("sigma_u", 1.0), dims="i")
        v = pm.ZeroSumNormal("v", sigma=pm.HalfNormal("sigma_v", 1.0), dims=("i", "b"), n_zerosum_axes=2)
        inv_sqrt_phi = pm.HalfNormal("inv_sqrt_phi", 1.0, dims="b")
        phi = 1.0 / inv_sqrt_phi**2

        delta_full = pt.concatenate([pt.zeros((len(MONTHS) - SUBSIDY_MONTHS, len(TIERS))), delta], axis=0)
        log_mu = (
            ix["offset"] + alpha[b] + gamma[b, ix["g"]] + tau[b, ix["t"]] + eta[ix["k"], ix["t"]]
            + eligible[b] * delta_full[ix["t"], ix["k"]] + (beta[b] * ix["X"]).sum(axis=1) + u[ix["i"]] + v[ix["i"], b]
        )
        mu = pt.exp(log_mu)
        pm.NegativeBinomial("y_obs", mu=mu[obs], alpha=phi[b[obs]], observed=y[obs])
        # 검열 셀: 0..upper 확률의 logsumexp (C가 작아 정확, JAX에서 betainc 기울기 회피)
        cen = ~obs
        ks = np.arange(3 * C + 1)
        lp = pm.logp(pm.NegativeBinomial.dist(mu=mu[cen][:, None], alpha=phi[b[cen]][:, None]), ks[None, :])
        lp = pt.where(ks[None, :] <= frame["upper"].to_numpy()[cen][:, None], lp, -np.inf)
        pm.Potential("censored", pm.math.logsumexp(lp, axis=1).sum())
    return model


def param_draws(idata, names):
    post = idata.posterior
    return {n: post[n].stack(sample=("chain", "draw")).transpose("sample", ...).values for n in names}


def log_demand(idata, frame: pl.DataFrame, ix: dict, n_draws: int, rng) -> np.ndarray:
    """D_ib = Σ_{g,t} P_ig n_t exp(η) 의 로그, 지원금 차이 δ 제외. shape (draws, 255, 8), 마트 제외."""
    p = param_draws(idata, ["alpha", "gamma", "tau", "eta", "beta", "u", "v"])
    S = p["alpha"].shape[0]
    pick = rng.choice(S, min(n_draws, S), replace=False)
    sel = frame["b"] != MART
    f = frame.filter(sel)
    b, i, g, t, k = (ix[n][sel.to_numpy()] for n in ("b", "i", "g", "t", "k"))
    X, off = ix["X"][sel.to_numpy()], ix["offset"][sel.to_numpy()]
    cell = i * (len(INDUSTRIES) - 1) + b  # (region, industry) 칸
    out = np.empty((len(pick), 255, len(INDUSTRIES) - 1), dtype=np.float32)
    for n, s in enumerate(pick):
        lm = (off + p["alpha"][s][b] + p["gamma"][s][b, g] + p["tau"][s][b, t] + p["eta"][s][k, t]
              + (p["beta"][s][b] * X).sum(1) + p["u"][s][i] + p["v"][s][i, b])
        m = np.max(lm)
        sums = np.bincount(cell, weights=np.exp(lm - m), minlength=255 * (len(INDUSTRIES) - 1))
        out[n] = (np.log(sums) + m).reshape(255, len(INDUSTRIES) - 1)
    return out


def calibration(idata, frame: pl.DataFrame, C: int, ix: dict, rng, n_draws: int = 200) -> pl.DataFrame:
    """업종별 점검: 검열 비율 vs 예측 P(Y ≤ upper), 관측 셀의 사후예측 구간 포함률(50/80/90/95%), log y 분위수."""
    p = param_draws(idata, ["alpha", "gamma", "tau", "eta", "delta", "beta", "u", "v", "inv_sqrt_phi"])
    b, i, g, t, k = (ix[n] for n in ("b", "i", "g", "t", "k"))
    eligible = np.array([bb != MART for bb in INDUSTRIES], dtype=float)
    upper = frame["upper"].to_numpy()
    obs = ~frame["censored"].to_numpy()
    y = frame["y"].fill_null(0).to_numpy()
    p_cen = np.zeros(len(frame))
    reps = np.empty((n_draws, obs.sum()), dtype=np.int64)
    for n, s in enumerate(rng.choice(p["alpha"].shape[0], n_draws, replace=False)):
        dfull = np.vstack([np.zeros((len(MONTHS) - SUBSIDY_MONTHS, len(TIERS))), p["delta"][s]])
        lm = (ix["offset"] + p["alpha"][s][b] + p["gamma"][s][b, g] + p["tau"][s][b, t] + p["eta"][s][k, t]
              + eligible[b] * dfull[t, k] + (p["beta"][s][b] * ix["X"]).sum(1) + p["u"][s][i] + p["v"][s][i, b])
        mu, phi = np.exp(lm), 1.0 / p["inv_sqrt_phi"][s][b] ** 2
        p_cen += stats.nbinom.cdf(upper, phi, phi / (phi + mu)) / n_draws
        reps[n] = rng.negative_binomial(phi[obs], phi[obs] / (phi[obs] + mu[obs]))
    levels = (0.5, 0.8, 0.9, 0.95)
    inside = {lv: (np.quantile(reps, (1 - lv) / 2, axis=0) <= y[obs]) & (y[obs] <= np.quantile(reps, (1 + lv) / 2, axis=0)) for lv in levels}
    bo = b[obs]
    rows = []
    for n, name in enumerate(INDUSTRIES):
        m = bo == n
        oq = np.quantile(np.log(y[obs][m] + 1), [0.1, 0.5, 0.9])
        rq = np.quantile(np.log(reps[:, m] + 1), [0.1, 0.5, 0.9])
        rows.append((name, int((b == n).sum()), (~obs[b == n]).mean(), p_cen[b == n].mean(),
                     *[inside[lv][m].mean() for lv in levels], *oq, *rq))
    return pl.DataFrame(rows, schema=["b", "cells", "censored_share", "pred_P_le_upper", "cover50", "cover80", "cover90", "cover95",
                                      "obs_q10", "obs_q50", "obs_q90", "rep_q10", "rep_q50", "rep_q90"], orient="row")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--demand-draws", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260914)

    frame, C = build_frame()
    ix = indices(frame)
    print(f"C={C} · 셀 {frame.height:,} (검열 {frame['censored'].sum():,})")
    with build_model(frame, C, ix):
        idata = pm.sample(chains=args.chains, tune=args.tune, draws=args.draws, target_accept=0.9,
                          nuts_sampler="numpyro", nuts={"chain_method": "vectorized"}, random_seed=20260914)
    np.savez_compressed(OUT / "posterior.npz", **{k: v.values for k, v in idata.posterior.data_vars.items()})
    st = idata.sample_stats
    print({"step_size": float(st["step_size"].mean()), "mean_n_steps": float(st["n_steps"].mean()), "max_tree_depth_share": float((st["tree_depth"] >= 10).mean())})

    summary = az.summary(idata, kind="diagnostics", var_names=["~censored"], filter_vars=None)
    divergences = int(idata.sample_stats["diverging"].sum().item())
    worst = {"max_r_hat": float(summary["r_hat"].max()), "min_ess_bulk": float(summary["ess_bulk"].min()),
             "min_ess_tail": float(summary["ess_tail"].min()), "divergences": divergences}
    summary.to_csv(OUT / "diagnostics.csv")
    print(worst)
    passed = worst["max_r_hat"] < 1.01 and min(worst["min_ess_bulk"], worst["min_ess_tail"]) >= 400 and divergences == 0
    print("진단 통과" if passed else "진단 실패: 층 2·3에 사용 금지")

    cal = calibration(idata, frame, C, ix, rng)
    cal.write_csv(OUT / "calibration.csv")
    print(cal)
    np.savez_compressed(OUT / "log_demand.npz", log_demand=log_demand(idata, frame, ix, args.demand_draws, rng),
                        regions=np.array([f"{s}|{c}" for s, c in ix["regions"].iter_rows()]), industries=np.array(INDUSTRIES[:-1]),
                        passed=passed)


if __name__ == "__main__":
    main()
