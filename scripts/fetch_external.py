"""외부 데이터 수집 (2026-01~06): 지원금 등급표, 국토부 아파트 매매 실거래가(상세), 관광공사 기초지자체 방문자수, 국민연금 가입자.

실행: uv run python scripts/fetch_external.py [subsidy trades visitors workers centroids]   (.env 의 api_key 사용, 생략 시 전부)
"""

import asyncio
import csv
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "raw"
PROCESSED = ROOT / "data" / "external" / "processed"
TRADE_API = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
VISITOR_API = "https://apis.data.go.kr/B551011/DataLabService/locgoRegnVisitrDDList"
MONTHS = [f"2026{m:02d}" for m in range(1, 7)]
DAYS = [(date(2026, 1, 1) + timedelta(d)).strftime("%Y%m%d") for d in range(181)]
PAGE = 1000

# 행정안전부 인구감소지역 89곳 https://www.mois.go.kr/frt/sub/a06/b06/populationDecline/screen.do
DEPOPULATION = {
    "부산광역시": "동구 서구 영도구",
    "대구광역시": "남구 서구 군위군",
    "인천광역시": "강화군 옹진군",
    "경기도": "가평군 연천군",
    "강원특별자치도": "고성군 삼척시 양구군 양양군 영월군 정선군 철원군 태백시 평창군 홍천군 화천군 횡성군",
    "충청북도": "괴산군 단양군 보은군 영동군 옥천군 제천시",
    "충청남도": "공주시 금산군 논산시 보령시 부여군 서천군 예산군 청양군 태안군",
    "전북특별자치도": "고창군 김제시 남원시 무주군 부안군 순창군 임실군 장수군 정읍시 진안군",
    "전라남도": "강진군 고흥군 곡성군 구례군 담양군 보성군 신안군 영광군 영암군 완도군 장성군 장흥군 진도군 함평군 해남군 화순군",
    "경상북도": "고령군 문경시 봉화군 상주시 성주군 안동시 영덕군 영양군 영주시 영천시 울릉군 울진군 의성군 청도군 청송군",
    "경상남도": "거창군 고성군 남해군 밀양시 산청군 의령군 창녕군 하동군 함안군 함양군 합천군",
}
# 고유가 피해지원금 특별지원 40곳 (행정안전부 답변, 2026-04-17)
# https://www.110.go.kr/data/counselView.do?num=B05_720999
SPECIAL = {
    "강원특별자치도": "양구군 화천군",
    "충청북도": "괴산군 단양군 보은군 영동군",
    "충청남도": "부여군 서천군 청양군",
    "전북특별자치도": "고창군 무주군 부안군 순창군 임실군 장수군 진안군",
    "전라남도": "강진군 고흥군 곡성군 구례군 보성군 신안군 완도군 장성군 장흥군 함평군 해남군",
    "경상북도": "봉화군 상주시 영덕군 영양군 의성군 청도군 청송군",
    "경상남도": "고성군 남해군 의령군 하동군 함양군 합천군",
}
CAPITAL = {"서울특별시", "인천광역시", "경기도"}

# 2026-07 개편 후 API는 옛 코드로 조회되지 않는다 → 새 코드로 조회해 공모전 지역에 되돌린다
LEGAL_DONG = RAW / "molit_legal_dong_20260729.csv"  # https://www.data.go.kr/data/15063424/fileData.do
INCHEON = {"28110": ["28125", "28155"], "28140": ["28125"], "28260": ["28275", "28290"]}  # 중구, 동구, 서구
DONGGU = {"만석동", "화수동", "송현동", "화평동", "창영동", "금곡동", "송림동"}  # 옛 동구 → 제물포구(28125) 편입분


def legal_dong() -> pl.DataFrame:
    return pl.read_csv(LEGAL_DONG, infer_schema_length=0).rename(lambda c: c.lstrip("﻿"))


def to_regions(qmap: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """새 코드(QUERY_CD) 행을 공모전 지역(LAWD_CD)으로 되돌린다. 제물포구는 법정동명(umdNm)으로 옛 동구·중구를 나눈다."""
    return qmap.join(df, on="QUERY_CD").filter(
        (pl.col("QUERY_CD") != "28125") | ((pl.col("LAWD_CD") == "28140") == pl.col("umdNm").is_in(DONGGU))
    )


def query_map(regions: pl.DataFrame) -> pl.DataFrame:
    ld = legal_dong()
    merged = dict(
        ld.filter((pl.col("시도명") == "전남광주통합특별시") & pl.col("읍면동명").is_null() & pl.col("시군구명").is_not_null())
        .select("시군구명", pl.col("법정동코드").str.slice(0, 5))
        .iter_rows()
    )
    pairs = [
        (lawd, q)
        for sido, ccg, lawd in regions.select("SIDO_NM", "CCG_NM", "LAWD_CD").iter_rows()
        for q in ([merged[ccg]] if sido in ("광주광역시", "전라남도") else INCHEON.get(lawd, [lawd]))
    ]
    return pl.DataFrame(pairs, schema=["LAWD_CD", "QUERY_CD"], orient="row")


def subsidy_table(regions: pl.DataFrame) -> pl.DataFrame:
    depop = {(s, c) for s, names in DEPOPULATION.items() for c in names.split()}
    special = {(s, c) for s, names in SPECIAL.items() for c in names.split()}
    assert len(depop) == 89 and len(special) == 40 and special <= depop
    assert depop <= set(regions.select("SIDO_NM", "CCG_NM").iter_rows())
    tiers = pl.DataFrame(
        [(s, c, "특별" if (s, c) in special else "우대") for s, c in depop],
        schema=["SIDO_NM", "CCG_NM", "DEPOPULATION_TIER"], orient="row",
    )
    tier = pl.col("DEPOPULATION_TIER")
    return (
        regions.join(tiers, on=["SIDO_NM", "CCG_NM"], how="left")
        .with_columns(
            pl.when(tier == "특별").then(25).when(tier == "우대").then(20)
            .when(pl.col("SIDO_NM").is_in(CAPITAL)).then(10).otherwise(15)
            .alias("SUBSIDY_10K_KRW")
        )
        .sort("SIDO_NM", "CCG_NM")
    )


def service_key() -> str:
    lines = (ROOT / ".env").read_text().splitlines()
    env = {k.strip(): v.strip().strip("\"'") for k, v in (line.split("=", 1) for line in lines if "=" in line)}
    return urllib.parse.unquote(env["api_key"])  # 인코딩/디코딩 키 모두 허용


def parse(xml: str) -> tuple[list[dict], int]:
    root = ET.fromstring(xml)
    code = root.findtext("header/resultCode") or root.findtext("cmmMsgHeader/returnReasonCode")
    if code not in ("000", "0000"):  # 실거래가 000, 관광공사 0000
        raise RuntimeError(f"API {code}: {root.findtext('header/resultMsg') or root.findtext('cmmMsgHeader/errMsg')}")
    items = [{c.tag: (c.text or "").strip() for c in item} for item in root.iter("item")]
    return items, int(root.findtext("body/totalCount") or 0)


def get(url: str, params: dict) -> str:
    try:
        with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=30) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        return e.read().decode(errors="replace")  # 인증 오류 XML이 403 본문으로 온다


async def call(url: str, params: dict, sem: asyncio.Semaphore) -> tuple[list[dict], int]:
    async with sem:  # ponytail: 동시 5건 × 호출 후 0.5초 → 초당 ~10건 이하, TPS 한도 확인되면 조정
        for attempt in range(5):
            try:
                result = parse(await asyncio.to_thread(get, url, params))
                await asyncio.sleep(0.5)
                return result
            except Exception:
                if attempt == 4:
                    raise
                await asyncio.sleep(2**attempt)


async def fetch_pages(url: str, params: dict, sem: asyncio.Semaphore) -> list[dict]:
    rows, page = [], 1
    while True:
        items, total = await call(url, params | {"pageNo": page, "numOfRows": PAGE}, sem)
        rows += items
        if page * PAGE >= total:
            return rows
        page += 1


def price_per_m2(trades: pl.DataFrame) -> pl.DataFrame:
    return (
        trades.filter(pl.col("cdealType") != "O")  # 해제 거래 제외
        .with_columns(
            (pl.col("dealAmount").str.replace_all(",", "").cast(pl.Float64) / pl.col("excluUseAr").cast(pl.Float64))
            .alias("p")
        )
        .group_by("LAWD_CD")
        .agg(pl.len().alias("N_TRADES"), pl.col("p").median().alias("MEDIAN_PRICE_PER_M2_10K_KRW"))
    )


def selftest() -> None:
    xml = (
        "<response><header><resultCode>000</resultCode></header><body><items>"
        "<item><dealAmount>100,000</dealAmount><excluUseAr>50</excluUseAr><cdealType> </cdealType></item>"
        "<item><dealAmount>300,000</dealAmount><excluUseAr>50</excluUseAr><cdealType>O</cdealType></item>"
        "</items><totalCount>2</totalCount></body></response>"
    )
    items, total = parse(xml)
    out = price_per_m2(pl.DataFrame([i | {"LAWD_CD": "11110"} for i in items]))
    assert total == 2 and out["N_TRADES"][0] == 1 and out["MEDIAN_PRICE_PER_M2_10K_KRW"][0] == 2000
    ok = "<response><header><resultCode>0000</resultCode></header><body><totalCount>0</totalCount></body></response>"
    assert parse(ok) == ([], 0)
    try:
        parse("<OpenAPI_ServiceResponse><cmmMsgHeader><errMsg>X</errMsg>"
              "<returnReasonCode>30</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>")
    except RuntimeError:
        return
    raise AssertionError("auth error not raised")


async def trades(regions: pl.DataFrame, key: str, sem: asyncio.Semaphore) -> None:
    qmap = query_map(regions)
    # ponytail: 한 건이라도 5회 실패하면 전체 중단(재실행 ~1,530콜), 쿼터가 빠듯해지면 (코드,월) 단위 캐시 추가
    async with asyncio.TaskGroup() as tg:
        tasks = [
            (c, tg.create_task(fetch_pages(TRADE_API, {"serviceKey": key, "LAWD_CD": c, "DEAL_YMD": ym}, sem)))
            for c in qmap["QUERY_CD"].unique()
            for ym in MONTHS
        ]
    rows = [row | {"QUERY_CD": c} for c, t in tasks for row in t.result()]

    fields = ["QUERY_CD", "sggCd", "umdNm", "aptSeq", "aptNm", "dealYear", "dealMonth", "dealDay",
              "dealAmount", "excluUseAr", "floor", "buildYear", "cdealType", "dealingGbn"]
    with open(RAW / "molit_apt_trade_dev_202601_202606.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    trades = to_regions(qmap, pl.DataFrame(rows, infer_schema_length=None))
    prices = regions.join(price_per_m2(trades), on="LAWD_CD", how="left").sort("SIDO_NM", "CCG_NM")
    assert prices.filter(pl.col("LAWD_CD").is_in(["28110", "28140"]))["N_TRADES"].min() > 0  # 제물포구 분할 확인
    prices.write_csv(PROCESSED / "apt_price_sgg_2026h1.csv")
    print(f"거래 {len(rows):,}행 · 시군구 {prices.height}")
    print("거래 0건(코드 확인 필요):", prices.filter(pl.col("N_TRADES").is_null()).select("SIDO_NM", "CCG_NM", "LAWD_CD").rows())


async def visitors(regions: pl.DataFrame, key: str, sem: asyncio.Semaphore) -> None:
    # 기초지자체 코드는 공모전 255개(일반구 포함, 개편 전 코드)와 그대로 일치한다. 방문자는 일자별 중복 집계(인·일).
    base = {"serviceKey": key, "MobileOS": "ETC", "MobileApp": "finance-modeling"}
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_pages(VISITOR_API, base | {"startYmd": d, "endYmd": d}, sem)) for d in DAYS]
    raw = pl.DataFrame([row for t in tasks for row in t.result()], infer_schema_length=None)
    raw.write_csv(RAW / "visitkorea_locgo_visitors_20260101_20260630.csv")

    bc = raw.filter(pl.col("signguCode").is_in(regions["LAWD_CD"].implode()))
    days = bc.group_by("signguCode", "touDivCd").agg(pl.col("baseYmd").n_unique().alias("n"))
    # 화성시 4개 구는 2026-02-01 신설이라 1월이 없다. 시 합계는 생활권 기준이 달라 구로 배분하지 않는다.
    short = days.filter(pl.col("n") < len(DAYS))
    assert days.height == 255 * 3 and set(short["signguCode"]) == {"41591", "41593", "41595", "41597"}
    assert short["n"].min() == len(DAYS) - 31, "화성 구 이외에 누락된 일자가 있다"
    monthly = (
        bc.with_columns(pl.col("baseYmd").str.slice(0, 6).alias("STRD_YYMM"), pl.col("touNum").cast(pl.Float64))
        .pivot(on="touDivCd", index=["signguCode", "STRD_YYMM"], values="touNum", aggregate_function="sum")
        .rename({"1": "LOCAL_VISITORS", "2": "OUTSIDE_VISITORS", "3": "FOREIGN_VISITORS"})
    )
    out = regions.join(monthly, left_on="LAWD_CD", right_on="signguCode").sort("SIDO_NM", "CCG_NM", "STRD_YYMM")
    assert out.height == 255 * len(MONTHS) - 4  # 화성 구 1월 없음
    out.write_csv(PROCESSED / "visitors_sgg_month_2026h1.csv")
    print(f"방문자 원본 {raw.height:,}행 · 공모전 지역×월 {out.height}행")


def workers(regions: pl.DataFrame) -> None:
    # 국민연금 가입 사업장(2026-07, 사업장 소재지) 가입자수 = 주간 근로인구 대리변수.
    # https://www.data.go.kr/data/15083277/fileData.do  원본을 raw/nps_workplaces_202607.csv 로 받아 둔다.
    names = legal_dong().select(pl.col("법정동코드").alias("bjd"), pl.col("읍면동명").alias("umdNm"))
    nps = (
        pl.read_csv(RAW / "nps_workplaces_202607.csv", encoding="cp949", infer_schema_length=0,
                    columns=[3, 7, 18], new_columns=["status", "bjd", "workers"])
        .filter(pl.col("status") == "1")  # 1 등록, 2 탈퇴
        .with_columns(pl.col("workers").cast(pl.Int64), pl.col("bjd").str.slice(0, 5).alias("QUERY_CD"))
        .join(names, on="bjd", how="left")
    )
    mapped = to_regions(query_map(regions), nps)
    lost = 1 - mapped["workers"].sum() / nps["workers"].sum()
    assert lost < 0.001, f"공모전 지역에 붙지 않은 가입자 {lost:.2%}"  # 구 없는 옛 코드 잔여분(화성시·부천시 등) 0.03%
    small = pl.col("workers").filter(pl.col("workers") < 1000)  # 본사 일괄 신고 민감도
    out = regions.join(
        mapped.group_by("LAWD_CD").agg(
            pl.col("workers").sum().alias("NPS_WORKERS"), small.sum().alias("NPS_WORKERS_EXCL_1000"), pl.len().alias("N_WORKPLACES")
        ),
        on="LAWD_CD", how="left",
    ).sort("SIDO_NM", "CCG_NM")
    assert out["NPS_WORKERS_EXCL_1000"].null_count() == 0 and out["NPS_WORKERS_EXCL_1000"].min() > 0
    out.write_csv(PROCESSED / "workers_sgg_202607.csv")
    print(f"국민연금 가입자 {out['NPS_WORKERS'].sum():,} (미결합 {lost:.3%}) · 시군구 {out.height}")


def centroids(regions: pl.DataFrame) -> None:
    # 결정 9: 상권 중심점 = 상가정보 점포 좌표 평균 (공간 커버리지 거리용). 원본은 개편 후 코드라 되돌린다.
    with zipfile.ZipFile(RAW / "sbiz_stores_20260630.zip") as z:
        stores = pl.concat([
            pl.read_csv(z.open(n), columns=["시군구코드", "법정동명", "경도", "위도"], schema_overrides={"시군구코드": pl.Utf8})
            for n in z.namelist() if n.endswith(".csv")
        ])
    mapped = to_regions(query_map(regions), stores.rename({"시군구코드": "QUERY_CD", "법정동명": "umdNm"}))
    out = regions.join(
        mapped.group_by("LAWD_CD").agg(pl.col("경도").mean().alias("LON"), pl.col("위도").mean().alias("LAT"), pl.len().alias("N_STORES")),
        on="LAWD_CD", how="left",
    ).sort("SIDO_NM", "CCG_NM")
    assert out["LON"].null_count() == 0, out.filter(pl.col("LON").is_null())
    out.write_csv(PROCESSED / "sgg_centroids_202606.csv")
    print(f"상권 중심점 {out.height} · 점포 {out['N_STORES'].sum():,} / 원본 {stores.height:,}")


async def main(steps: list[str]) -> None:
    selftest()
    regions = (
        pl.read_csv(PROCESSED / "population_sgg_age_sex_202606.csv")
        .select("SIDO_NM", "CCG_NM", (pl.col("ADMIN_REGION_CODE") // 100_000).cast(pl.Utf8).alias("LAWD_CD"))
        .unique()
    )
    assert regions.height == 255 and regions["LAWD_CD"].n_unique() == 255
    key, sem = service_key(), asyncio.Semaphore(5)
    if "subsidy" in steps:
        subsidy_table(regions).write_csv(PROCESSED / "subsidy_tier_sgg_2026.csv")
    if "trades" in steps:
        await trades(regions, key, sem)
    if "visitors" in steps:
        await visitors(regions, key, sem)
    if "workers" in steps:
        workers(regions)
    if "centroids" in steps:
        centroids(regions)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["subsidy", "trades", "visitors", "workers", "centroids"]))
