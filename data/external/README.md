# 외부 데이터: 수집·결합 상태

2026년 6월 시점의 원본은 `raw/`에 보관하고, 재현 가능한 정규화 결과는 `processed/`에 둔다. 모든 변환은 [02_external_data_acquisition_and_preprocessing.ipynb](../../notebooks/02_external_data_acquisition_and_preprocessing.ipynb)에서 수행한다.

| 상태 | 파일 | 행 수 | 역할 |
| --- | --- | ---: | --- |
| 수집·정규화 완료 | `processed/population_sgg_age_sex_202606.csv` | 3,060 | 255개 시군구 × 내국인 성별 2 × 연령 6의 인구 노출량 후보 |
| 수집·정규화 완료 | `processed/kosis_household_income_sido_2025.csv` | 17 | 시도 간 가구소득 차이를 통제하는 구매력 보조변수 |
| 수집·정규화 완료 | `processed/nts_100_living_industries_sgg_202606.csv` | 24,600 | 국세청 사업체 수의 독립 교차검증 자료 |
| 수집·카탈로그화 완료 | `processed/sbiz_industry_catalog_202606.csv` | 247 | 소상공인 상가정보 업종 교차표 검토용 |
| 수집·정규화 완료 | `processed/subsidy_tier_sgg_2026.csv` | 255 | 고유가 피해지원금 지역 등급(수도권 10 / 비수도권 15 / 인구감소 우대 20 / 특별 25만원) |
| 수집·정규화 완료 | `processed/apt_price_sgg_2026h1.csv` | 255 | 2026-01~06 아파트 매매 ㎡당 중앙가(해제 거래 제외). 임대료·비용 대리변수. 옹진군은 거래 없음, 20곳은 30건 미만 |
| 수집·정규화 완료 | `processed/visitors_sgg_month_2026h1.csv` | 1,526 | 월별 현지인·외지인·외국인 방문자(인·일 합계). 화성시 4개 구는 2026-02 신설로 1월 없음 |
| 대기 | `processed/sbiz_crosswalk_approved.csv` | - | BC카드 11개 업종과 상가정보 소분류의 승인된 매핑 |

## 원본과 출처

| 원본 | 기준일 | 출처 | 사용상 제약 |
| --- | --- | --- | --- |
| `raw/mois_age_population_sgg_202606.csv` | 2026-06 | [행정안전부 주민등록 인구통계](https://jumin.mois.go.kr/ageStatMonth.do) | 거주인구이므로 방문수요의 대리변수다. 외국인·법인 분모로 쓰지 않는다. |
| `raw/kosis_household_income_sido_2025.json` | 2025 | [KOSIS e-지방지표 가구소득](https://kosis.kr/visual/eRegionIndex/eRegionWhole.do) | 시도 단위 가구소득이다. 시군구별·개인별 소득으로 해석하지 않는다. |
| `raw/nts_100_living_industries_20260630.csv` | 2026-06-30 | [국세청 100대 생활업종](https://www.data.go.kr/data/15061118/fileData.do) | 3개 미만 셀은 0으로 제공될 수 있고, 사업자 0인 업종은 행이 없다. 원본은 2026-07 개편 명칭(전남광주통합특별시·인천 제물포/영종/서해/검단구)이라 노트북 02가 공모전 지역으로 되돌린다. 제물포구는 상가정보 점포 법정동 비율(옛 동구 0.406)로 나누고, 화성시 잔여 행(당월 0)은 제외한다. |
| `raw/molit_apt_trade_dev_202601_202606.csv` | 2026-01~06 계약 | 국토교통부 아파트 매매 실거래가 상세(`RTMSDataSvcAptTradeDev`, data.go.kr), `scripts/fetch_external.py` | 2026-07 개편으로 광주·전남은 `12xxx`, 인천 중·동·서구는 새 코드로 조회 후 되돌림. 옛 동구는 법정동명으로 제물포구에서 분리 |
| `raw/visitkorea_locgo_visitors_20260101_20260630.csv` | 2026-01-01~06-30 일별 | [한국관광공사 빅데이터 지역별 방문자수](https://www.data.go.kr/data/15101972/openapi.do) `locgoRegnVisitrDDList` | KT·SKT 이동통신 추정. 동일인 일자별 중복 집계. 시 합계와 일반구는 생활권 기준이 달라 합산·배분하지 않는다 |
| `raw/molit_legal_dong_20260729.csv` | 2026-07-29 | [국토교통부 전국 법정동](https://www.data.go.kr/data/15063424/fileData.do) | 옛 코드 → 새 코드 대응용 |
| 인구감소지역 목록(스크립트 내 상수) | 2026-04-17 | [행정안전부](https://www.mois.go.kr/frt/sub/a06/b06/populationDecline/screen.do), [110 특별·우대 구분](https://www.110.go.kr/data/counselView.do?num=B05_720999) | 지원금은 거주 광역 내 사용 가능하므로 시군구 등급만으로 노출량이 결정되지 않는다 |
| `raw/sbiz_stores_20260630.zip` | 2026-06-30 | [소상공인시장진흥공단 상가(상권)정보](https://www.data.go.kr/data/15083033/fileData.do) | 10/75/247 표준산업분류 기반 체계이며 BC카드 업종과 범위가 다르다. |

## 결합 규칙

1. `SIDO_NM`, `CCG_NM` 복합키만 사용한다. `CCG_NM` 단독 결합은 금지한다.
2. 인구 자료의 세종 광역 집계(`3600000000`)와 시군구 집계(`3611000000`)는 같은 값으로 중복되어 있다. 노트북은 후자만 남기고 값 동일성을 검증한다.
3. `AGE_CD=1`은 설명문의 `20대 이하`와 `AGE_CD=2`의 `20대`가 겹치므로, 현재는 **20세 미만**으로 임시 해석했다. 플랫폼 코드북 확인 전에는 이 구간의 실질적 해석을 확정하지 않는다.
4. 가구소득은 시도 공통 보조변수로만 결합한다. 동일 시도 내 시군구의 소득 차이를 만들거나 추정값으로 채우지 않는다.
5. 소상공인 분류와 BC카드 업종의 교차표에 `mapping_rationale`을 남겨 검토한 뒤에만 사업체 수를 집계한다. 승인 파일이 없으면 공급지표·MCMC 의사결정 모델을 실행하지 않는다.

외부 데이터는 BC카드 결제 실적을 대체하지 않는다. 공급 보정 점수는 검증된 동시점 분모가 있을 때만 계산한다.

## 추가 공변량 (scripts/fetch_external.py)

| 파일 | 행 수 | 역할 | 주의 |
| --- | ---: | --- | --- |
| `processed/workers_sgg_202607.csv` | 255 | 국민연금 가입 사업장(2026-07) 가입자 합계 = 주간 근로인구 | 원본 `raw/nps_workplaces_202607.csv`([공공데이터포털](https://www.data.go.kr/data/15083277/fileData.do), CP949). 법정동코드로 합산하고 개편 코드는 되돌린다. 구 없는 옛 코드 잔여 0.03%는 버린다. `NPS_WORKERS_EXCL_1000`은 본사 일괄 신고 민감도용 |
