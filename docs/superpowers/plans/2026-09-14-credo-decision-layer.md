# CREDO 결정 층 보강: 4주 검증 규칙 · 운영특성 · BC 점유율 편향 점검

**Spec:** `claudedocs/methodology_credo_model.md` (§5.3 지식 기울기, §5.4 운영특성, §7 BC 점유율 한계). 확정 결정 1~9는 이 문서의 결정 로그를 따른다.

**현재 상태:** 층 1(`scripts/fit_demand.py`)은 진단을 통과했고 `data/model/log_demand.npz`(draw 1000 × 지역 255 × 업종 8)가 있다. 층 2·3(`scripts/decide.py`)은 `data/model/decision_table.csv`, `sensitivity.csv`, `coverage_curve.csv`를 만든다.

## Global Constraints

- 실행은 `.venv/bin/python scripts/<file>.py` (저장소 루트 `/home/user/finance-modeling`). 새 의존성 금지: numpy, polars, scipy, 표준 라이브러리만.
- 기존 코드 관례를 따른다: 한국어 주석·docstring, `KEY = ["SIDO_NM", "CCG_NM"]` 복합키, polars 입출력, 모듈 수준 상수, 클래스·추상화 없음, 그림 없음. 가장 짧은 올바른 코드(ponytail).
- 비자명한 로직마다 스크립트 시작 시 실행되는 `assert` 기반 `selftest()` 하나(기존 `decide.py` 패턴). 별도 테스트 프레임워크 금지.
- 수정 금지: `scripts/fit_demand.py`, `scripts/fetch_external.py`, `data/model/posterior.npz`, `data/model/log_demand.npz`, `data/external/**`.
- 산출물은 `data/model/`에 쓴다(gitignore 대상, 커밋 금지).
- `ABP_CONTEST_DATA.csv`는 대회 보안서약 대상: 집계값만 쓰고 원시 행을 출력하거나 파일로 복사하지 않는다.
- 공유 인터페이스 `data/model/r_rel_draws.npz` (Task 1이 생성, Task 2·3이 읽음). 키:
  - `R_rel`: float32 (draws, 255, 8) — 주 분석 대응표의 상대 시장 여유 $\tilde R$ draw
  - `regions`: str 배열 `"SIDO_NM|CCG_NM"` (log_demand.npz와 같은 순서)
  - `industries`: str 배열 8개 (log_demand.npz와 같은 순서: H, 8005, 8004, 8006, 8021, 8301, 4010, 4020)
  - `in_set`: bool (255, 8) — FDR 즉시 목록
  - `v`: float (255, 8) — 상위 25% 초과확률
  - `r_gamma`: float (8,) — 업종별 앙상블 분위수 기준선
- 커밋 메시지 끝에 다음 두 줄을 붙인다:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NhMojtcrXQSj9EqAv6JPvu
  ```
  커밋은 `git -c user.name=zongseung -c user.email=energyax123@gmail.com commit ...`로 한다. 브랜치 `feature/credo-decision-layer`.

## Task 1: 지식 기울기 4주 검증 후보와 r_rel_draws.npz

**파일:** `scripts/decide.py` 수정만.

**요구사항:**
1. 모듈 상수 추가: `PILOTS = 10` (업종당 검증 예산), `LAMBDA_SCALES = (1.0, 0.25, 4.0)` (주, 민감도).
2. 함수 `kg_value(mu, sd, threshold, lam)` 추가 (numpy 배열 연산, 브로드캐스팅):
   - $\tilde\sigma = \mathrm{sd}^2 / \sqrt{\mathrm{sd}^2 + \lambda}$
   - $z = -|\mu - \text{threshold}| / \tilde\sigma$
   - $\nu = \tilde\sigma\,(z\,\Phi(z) + \varphi(z))$ (`scipy.stats.norm.cdf/pdf`)
   - `sd == 0`인 원소는 $\nu = 0$ (0 나누기 경고 없이).
3. 주 분석(`results["main"]`)에서 업종 b마다:
   - `mu = R_rel.mean(0)`, `sd = R_rel.std(0)` (shape (255, 8))
   - $\lambda_b$ = scale × `median(sd[:, b]**2)` (4주 검증 한 번이 현재 전형적 사후분산만큼의 정보를 준다고 가정)
   - 검증 후보 = 즉시 목록(`d["in_set"]`)에 **없는** 지역 중 $\nu$ 상위 `PILOTS`곳 ($\nu > 0$인 곳만).
4. `decision_table.csv`에 열 두 개 추가 (기존 열·정렬 유지): `kg_value` (scale 1.0), `action` ∈ {`"immediate"`, `"pilot"`, `"hold"`} (즉시 목록 → immediate, scale 1.0 검증 후보 → pilot, 나머지 → hold).
5. `data/model/pilot_sensitivity.csv` 생성: 열 `b, lambda_scale, jaccard_vs_main` — scale 0.25·4.0 각각의 검증 후보 집합과 scale 1.0 집합의 Jaccard.
6. `data/model/r_rel_draws.npz` 저장 (Global Constraints의 키·dtype 그대로).
7. `selftest()`에 assert 추가:
   - sd=0 → kg_value = 0
   - 같은 sd에서 threshold에 가까울수록 kg_value가 크다
   - 같은 |mu − threshold|에서 sd가 클수록 kg_value가 크다
8. 실행 끝에 업종별 action 개수(immediate/pilot/hold)를 출력한다.
9. docstring 산출물 줄에 `pilot_sensitivity.csv, r_rel_draws.npz`를 추가한다.

**검증:** `.venv/bin/python scripts/decide.py`가 오류 없이 끝나고, 모든 업종에서 pilot = 10(또는 ν>0 후보가 그보다 적으면 그 수), immediate 개수는 기존과 동일(53/65/59/58/62/63/63/60, 업종 순서 4010/4020/8004/8005/8006/8021/8301/H)이며, `r_rel_draws.npz`의 키·shape가 명세와 같다.

## Task 2: 운영특성 비교 (naive 순위 vs CREDO)

**파일:** 새 파일 `scripts/operating_characteristics.py`.

**요구사항:**
1. 입력: `data/model/r_rel_draws.npz`(Task 1), `data/model/log_demand.npz`, 국세청 공급은 `from decide import supply` 재사용 (스크립트 폴더가 sys.path에 있으므로 import 가능). 주 분석 대응표(`broad=False`).
2. 사후분포를 참값으로 쓰는 방식(posterior-as-truth): `rng = np.random.default_rng(20260914)`로 draw 200개를 뽑는다. 각 참값 draw t에서 업종 b의 참 상위 집합 $T_b(t) = \{i : \tilde R^{(t)}_{ib} > r_{\gamma,b}\}$.
3. 비교 규칙 3개 (업종마다 목록 L):
   - `credo_fdr`: npz의 `in_set`
   - `rrel_top25`: 사후평균 $\tilde R$ 상위 63곳 (255 // 4)
   - `naive_G`: $G = \overline{\log D} - \log(S+1)$ (log_demand draw 평균) 상위 63곳
4. 지표 (업종·규칙마다, draw t에 대해 계산 후 요약):
   - `list_size` = |L|
   - `fdr` = |L \ T(t)| / |L| → `fdr_mean`, `fdr_q05`, `fdr_q95`
   - `tpr` = |L ∩ T(t)| / |T(t)| → `tpr_mean`
5. 산출 `data/model/operating_characteristics.csv`, 열 `b, rule, list_size, fdr_mean, fdr_q05, fdr_q95, tpr_mean`. 실행 끝에 표 출력.
6. `selftest()`: 지표 함수에 대해 (a) L = T이면 fdr 0, tpr 1, (b) L과 T가 서로소이면 fdr 1, tpr 0.
7. docstring에 한계 명시: 사후분포 참값·표본 내 평가라 CREDO에 낙관적이며, naive_G는 다른 추정 대상(G)을 $\tilde R$ 참값으로 채점한 것.

**검증:** 실행이 오류 없이 끝나고, `credo_fdr`의 `fdr_mean`이 모든 업종에서 0.12 이하(사후기대 FDR 10% 목표의 표본 근사)이며, CSV 행 수는 8 업종 × 3 규칙 = 24.

## Task 3: BC 점유율 연령 구성 편향 점검

**파일:** 새 파일 `scripts/check_share_bias.py`.

**가설:** 고령 BC 카드 보유자가 인구 대비 많은 지역(예: 농협 비중이 높은 군)에서는, 고령 고객 비중이 큰 업종의 BC 거래가 부풀어 그 업종의 $\tilde R$이 커진다.

**요구사항:**
1. 입력: `ABP_CONTEST_DATA.csv`, `data/external/processed/population_sgg_age_sex_202606.csv`, `data/external/processed/subsidy_tier_sgg_2026.csv`, `data/model/r_rel_draws.npz`.
2. BC 원자료는 국내 개인(`GENDER_CD` ∈ {"1","2"}, `AGE_CD` ∈ "1"~"6")만. 업종 코드 8001·8002·8003은 `"H"`로 합치고, npz의 8개 업종만 쓴다(대형할인점 4004 제외).
3. 지역 i의 고령 과대표 지수 $o_i = \log\big(\frac{\text{BC 거래 중 AGE\_CD 6 비중}}{\text{인구 중 AGE\_CD 6 비중}}\big)$ (8개 업종 합산 cnt 기준).
4. 업종 b의 고령 기울기 $t_b$ = (전국 업종 b 거래 중 AGE_CD 6 비중) − (전국 8개 업종 합산 거래 중 AGE_CD 6 비중).
5. 설명변수 $x_{ib} = o_i\,t_b$를 지역 안 업종 평균으로 중심화: $\tilde x_{ib} = x_{ib} - \frac18\sum_b x_{ib}$ (R̃이 지역 안 평균 0이므로 같은 변환).
6. draw마다 $\tilde R^{(s)}_{ib}$ (255×8 벡터)를 [업종 더미 8개, $\tilde x$]에 OLS → 계수 $\kappa^{(s)}$, 부분 $R^2 = 1 - \mathrm{RSS}_{full}/\mathrm{RSS}_{dummies}$.
7. 요약 출력: $\kappa$ 사후평균과 90% 구간, 부분 $R^2$ 평균.
8. 조정: $\tilde R^{adj} = \overline{\tilde R} - \bar\kappa\,\tilde x$. 업종별로 사후평균 $\tilde R$ 상위 63곳 대비 조정 후 상위 63곳의 Jaccard, 인구감소지역(`SUBSIDY_10K_KRW` ≥ 20) 수 전후 비교.
9. 산출 `data/model/share_bias_check.csv`, 열 `b, t_b, jaccard_top25_after_adjust, n_depop_top25_before, n_depop_top25_after`. 첫 줄 print에 kappa 요약.
10. 판정 출력: $\kappa$ 90% 구간이 0을 포함하지 않고 부분 $R^2 \ge 0.05$이면 `"점유율 구성 편향 신호 있음"`, 아니면 `"점유율 구성 편향 신호 없음"`.
11. `selftest()`: 합성 데이터 $R = 2\tilde x + \text{업종효과} + \epsilon$ (ε ~ N(0, 0.01))에서 추정 κ가 2 ± 0.05.

**검증:** 실행이 오류 없이 끝나고 CSV가 8행이며, 원시 행은 출력하지 않는다.
