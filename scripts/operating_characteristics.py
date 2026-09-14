"""운영특성 비교: naive 순위 규칙 vs CREDO FDR 즉시 목록 (methodology_credo_model.md 4~5절).

사후분포를 참값으로 쓰는 방식(posterior-as-truth): R_rel 사후 draw 전체를 참값으로 취급해,
각 규칙이 뽑은 고정 목록 L이 draw별로 지역·업종 풀링 분위수 r_γ를 넘는 집합 T_b(t)와 얼마나
겹치는지 FDR·TPR로 요약한다. T_b(t) = {i : R̃_ib^(t) > r_γ,b}는 "진짜 상위 25% 집합"이 아니라
주 분석에서 쓴 pooled 임계값 r_γ의 초과 집합이며, draw마다 지역 수가 35~102개로 변동한다.

한계:
- credo_fdr의 fdr_mean ≤ ALPHA는 decide.py가 스스로 강제한 목표(사후기대 FDR ≤ 10%)를 이
  참값 정의로 다시 확인하는 것일 뿐, credo_fdr에 대해서는 독립적인 운영특성이 아니다.
- 참값 자체가 CREDO와 같은 모형의 사후 draw이고 credo_fdr 목록도 그 사후분포로 만들었으므로,
  이 평가는 표본 내(in-sample) 평가이며 credo_fdr에 낙관적이다(외부 검증이 아니다).
- naive_G는 원래 R_rel(수요 대비 시장 여유)이 아니라 수요-공급 격차 G를 겨냥한 규칙인데,
  여기서는 R_rel 사후분포를 참값으로 채점하므로 애초에 다른 목표를 추정하는 규칙을 CREDO의
  참값 기준으로 채점하는 셈이다. naive_G_rel(지역 안 업종 평균을 뺀 G)과 naive_G의 차이는
  이 지역 안 중심화 효과만을 분리해서 보여준다.
- 모든 규칙이 CREDO 즉시 목록과 같은 크기 n(업종별 in_set 개수)을 써서 같은 예산으로 비교한다
  (methodology_credo_model.md 5.4절).

실행: .venv/bin/python scripts/operating_characteristics.py
      (data/model/r_rel_draws.npz, data/model/log_demand.npz 필요, scripts/decide.py의 supply() 재사용)
산출: data/model/operating_characteristics.csv
"""

import numpy as np
import polars as pl

from decide import KEY, ROOT, supply


def fdr_tpr(L: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """L, T: 같은 길이의 bool 마스크(한 draw). FDR = |L\\T|/|L|, TPR = |L∩T|/|T|."""
    fdr = (L & ~T).sum() / L.sum()
    tpr = (L & T).sum() / T.sum()
    return fdr, tpr


def top_mask(score: np.ndarray, n: int) -> np.ndarray:
    """score 상위 n개를 True로 하는 bool 마스크."""
    mask = np.zeros(score.shape, dtype=bool)
    mask[np.argsort(-score)[:n]] = True
    return mask


def selftest() -> None:
    T = np.array([True, True, False, False])
    fdr, tpr = fdr_tpr(T.copy(), T)
    assert (fdr, tpr) == (0.0, 1.0), (fdr, tpr)  # L = T → 거짓양성 없음, 재현율 100%
    L = np.array([False, False, True, True])
    fdr, tpr = fdr_tpr(L, T)
    assert (fdr, tpr) == (1.0, 0.0), (fdr, tpr)  # L, T 서로소 → 전부 거짓양성, 재현율 0

    # |L| != |T|: fdr/tpr 분모가 뒤바뀌면(각각 |L|, |T|가 아니라 |T|, |L|로) 이 값에서 어긋난다
    T3 = np.array([True, True, True, True, False])
    L3 = np.array([True, False, False, False, True])
    fdr3, tpr3 = fdr_tpr(L3, T3)
    assert (fdr3, tpr3) == (0.5, 0.25), (fdr3, tpr3)

    s = np.array([3.0, 1.0, 2.0, 0.0])
    assert top_mask(s, 2).tolist() == [True, False, True, False]


def main() -> None:
    selftest()
    out = ROOT / "data" / "model"
    zr = np.load(out / "r_rel_draws.npz")
    zd = np.load(out / "log_demand.npz")
    assert (zr["regions"] == zd["regions"]).all() and (zr["industries"] == zd["industries"]).all()

    R_rel, industries = zr["R_rel"].astype(float), list(zr["industries"])
    regions = pl.DataFrame([r.split("|") for r in zr["regions"]], schema=KEY, orient="row")

    T = R_rel > zr["r_gamma"][None, None, :]  # (S, 255, 8) draw별 pooled r_γ 초과 집합 T_b(t)

    S = supply(regions, industries, broad=False)  # (255, 8) 국세청 공급, 주 분석 1:1 대응표
    G = zd["log_demand"].astype(float).mean(0) - np.log1p(S)  # naive 점수: 로그수요 평균 - 로그공급
    G_rel = G - G.mean(1, keepdims=True)  # 지역 안 업종 평균 중심화 (R̃과 같은 변환)
    R_mean = R_rel.mean(0)  # 사후평균 (255, 8)

    rows = []
    for b, name in enumerate(industries):
        n = int(zr["in_set"][:, b].sum())  # CREDO 즉시 목록과 같은 크기(같은 예산)로 맞춤
        rules = {
            "credo_fdr": zr["in_set"][:, b],
            "rrel_topn": top_mask(R_mean[:, b], n),
            "naive_G": top_mask(G[:, b], n),
            "naive_G_rel": top_mask(G_rel[:, b], n),
        }
        for rule, L in rules.items():
            fdr, tpr = np.array([fdr_tpr(L, T[t, :, b]) for t in range(T.shape[0])]).T
            rows.append((name, rule, int(L.sum()), fdr.mean(), *np.quantile(fdr, [0.05, 0.95]), tpr.mean()))

    table = pl.DataFrame(rows, schema=["b", "rule", "list_size", "fdr_mean", "fdr_q05", "fdr_q95", "tpr_mean"], orient="row")
    table.write_csv(out / "operating_characteristics.csv")
    print(table)


if __name__ == "__main__":
    main()
