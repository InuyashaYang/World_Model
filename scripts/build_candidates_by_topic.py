from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = DATA_DIR / "candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_MAILTO = ""  # 建议填邮箱提高配额，例如 "you@domain.com"

MAX_WORKERS = 16
PER_KEYWORD_WORKS = 200  # 每个关键词抓多少篇 works（可加大到 500/1000）
WORKS_PER_PAGE = 200  # OpenAlex 单页最大 200
TOP_N_OUTPUT = 200
TOP_50_OUTPUT = 50

# 国内机构兜底关键词（当 country_code 缺失时用）
CHINA_INST_KEYWORDS = [
    "清华",
    "北京大学",
    "北大",
    "中科院",
    "中国科学院",
    "上海交通",
    "上交",
    "复旦",
    "浙江大学",
    "浙大",
    "南京大学",
    "南大",
    "哈尔滨工业",
    "哈工大",
    "中国科学技术大学",
    "中科大",
    "西安交通",
    "西交",
    "华中科技",
    "北京航空航天",
    "北航",
    "同济",
    "中山大学",
    "南方科技",
    "南科大",
    "西湖大学",
    "上海人工智能实验室",
    "鹏城",
    "中国",
    "Beijing",
    "Shanghai",
    "Shenzhen",
    "Hangzhou",
    "Hong Kong",
    "香港",
]

# 你关注的主题关键词（可以继续扩展/替换）
TOPIC_KEYWORDS = [
    "world model",
    "video world model",
    "generative world model",
    "model-based reinforcement learning",
    "environment model",
    "embodied ai",
    "robot learning",
    "video generation",
    "diffusion transformer",
    "neural simulator",
    "digital twin",
]


def oa_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if OPENALEX_MAILTO:
        params = dict(params)
        params["mailto"] = OPENALEX_MAILTO
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def is_domestic_author_from_author_obj(author_obj: Dict[str, Any]) -> Tuple[bool, str]:
    inst = (author_obj.get("last_known_institution") or {}).get("display_name") or ""
    cc = (author_obj.get("last_known_institution") or {}).get("country_code") or ""
    if cc == "CN":
        return True, "country_code == CN"
    return False, "not CN"


def search_works_by_keyword(keyword: str, total_limit: int) -> List[Dict[str, Any]]:
    """
    OpenAlex /works?search=keyword
    抓取按 cited_by_count 排序的前 N 篇，作为粗筛候选池来源。
    """
    works: List[Dict[str, Any]] = []
    page = 1
    while len(works) < total_limit:
        per_page = min(WORKS_PER_PAGE, total_limit - len(works))
        params = {
            "search": keyword,
            "per-page": per_page,
            "page": page,
            "sort": "cited_by_count:desc",
        }
        obj = oa_get(f"{OPENALEX_BASE}/works", params=params)
        batch = obj.get("results") or []
        if not batch:
            break
        works.extend(batch)
        page += 1
    return works[:total_limit]


def get_author(author_id: str) -> Dict[str, Any]:
    return oa_get(f"{OPENALEX_BASE}/authors/{author_id}", params={})


@dataclass
class AuthorAgg:
    author_id: str
    display_name: str
    institution: Optional[str]
    country_code: Optional[str]

    # 统计（仅基于抓到的相关 works）
    related_works: int
    related_citations_sum: int
    related_recent_3y: int

    # 用于审计的样本
    sample_works: List[Dict[str, Any]]

    # 综合分
    score: float
    domestic_reason: str
    keywords_hit: List[str]


def now_year() -> int:
    return time.gmtime().tm_year


def compute_score(related_works: int, citations_sum: int, recent3: int) -> float:
    # 粗筛用：相关性（作品数）+ 影响力（被引总和）+ 趋势（近3年）
    # 防止单篇超高被引压过一切：log 缩放
    import math

    rel = min(related_works, 30) / 30.0  # 0-1
    imp = math.log1p(citations_sum) / math.log1p(50000)  # ~0-1
    mom = min(recent3, 15) / 15.0  # 0-1
    return 100 * (0.50 * rel + 0.35 * imp + 0.15 * mom)


def main():
    print(
        f"Keywords: {len(TOPIC_KEYWORDS)} | per_keyword_works={PER_KEYWORD_WORKS} | workers={MAX_WORKERS}"
    )

    # 1) 拉取 works（按关键词）
    all_works: List[Tuple[str, Dict[str, Any]]] = []
    for kw in TOPIC_KEYWORDS:
        ws = search_works_by_keyword(kw, PER_KEYWORD_WORKS)
        print(f"  - {kw}: works={len(ws)}")
        for w in ws:
            all_works.append((kw, w))

    # 2) 从 works 抽作者（先只收集 author_id）
    author_hits: Dict[str, Dict[str, Any]] = {}  # author_id -> agg
    current_year = now_year()

    for kw, w in all_works:
        cited = int(w.get("cited_by_count") or 0)
        year = w.get("publication_year")
        # Default: recent 2 years (align with app default)
        is_recent3 = year is not None and (current_year - int(year) <= 2)

        for a in w.get("authorships") or []:
            author = a.get("author") or {}
            author_id = author.get("id")
            author_name = author.get("display_name") or ""
            if not author_id:
                continue

            agg = author_hits.get(author_id)
            if not agg:
                author_hits[author_id] = {
                    "author_id": author_id,
                    "display_name": author_name,
                    "related_works": 0,
                    "related_citations_sum": 0,
                    "related_recent_3y": 0,
                    "sample_works": [],
                    "keywords_hit": set(),
                }
                agg = author_hits[author_id]

            agg["related_works"] += 1
            agg["related_citations_sum"] += cited
            if is_recent3:
                agg["related_recent_3y"] += 1
            agg["keywords_hit"].add(kw)

            # 留 5 条样本 works（用于审计）
            if len(agg["sample_works"]) < 5:
                agg["sample_works"].append(
                    {
                        "title": w.get("title"),
                        "year": year,
                        "cited_by_count": cited,
                        "openalex_work_id": w.get("id"),
                        "keyword": kw,
                    }
                )

    print(f"Authors collected from works: {len(author_hits)}")

    # 3) 拉 author 详情，做“国内=当前在国内”过滤
    def fetch_and_filter(author_id: str) -> Optional[AuthorAgg]:
        base = author_hits[author_id]
        try:
            ao = get_author(author_id)
        except Exception:
            return None

        ok, reason = is_domestic_author_from_author_obj(ao)
        if not ok:
            return None

        inst = (ao.get("last_known_institution") or {}).get("display_name")
        cc = (ao.get("last_known_institution") or {}).get("country_code")

        score = compute_score(
            related_works=base["related_works"],
            citations_sum=base["related_citations_sum"],
            recent3=base["related_recent_3y"],
        )

        return AuthorAgg(
            author_id=author_id,
            display_name=base["display_name"] or (ao.get("display_name") or ""),
            institution=inst,
            country_code=cc,
            related_works=base["related_works"],
            related_citations_sum=base["related_citations_sum"],
            related_recent_3y=base["related_recent_3y"],
            sample_works=base["sample_works"],
            score=score,
            domestic_reason=reason,
            keywords_hit=sorted(list(base["keywords_hit"])),
        )

    domestic: List[AuthorAgg] = []
    ids = list(author_hits.keys())
    done = 0
    started = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_and_filter, aid): aid for aid in ids}
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            res = fut.result()
            if res:
                domestic.append(res)
            if done % 100 == 0:
                print(
                    f"[{done}/{len(ids)}] domestic_found={len(domestic)} elapsed={time.time() - started:.1f}s"
                )

    # 4) 排序输出
    domestic.sort(key=lambda x: x.score, reverse=True)
    top200 = domestic[:TOP_N_OUTPUT]
    top50 = domestic[:TOP_50_OUTPUT]

    top200_path = OUT_DIR / "by_topic_top200.jsonl"
    top50_path = OUT_DIR / "by_topic_top50.jsonl"

    def dump_jsonl(path: Path, items: List[AuthorAgg]):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    dump_jsonl(top200_path, top200)
    dump_jsonl(top50_path, top50)

    print(f"\nDomestic authors found: {len(domestic)}")
    print(f"Wrote: {top200_path}")
    print(f"Wrote: {top50_path}")


if __name__ == "__main__":
    main()
