from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"

WEB_PEOPLE_DIR = DATA_DIR / "web" / "people"
TOPICS_DIR = DATA_DIR / "web" / "topics"
PEOPLE_DIR = DATA_DIR / "people"


OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_AUTHOR_WORKERS = int(os.environ.get("OPENALEX_AUTHOR_WORKERS", "10"))

# OpenAlex /works payload gets huge when per-page=200 and includes authorships.
# Default to 25 to keep latency and memory reasonable; tune up if needed.
OPENALEX_PER_PAGE = int(os.environ.get("OPENALEX_PER_PAGE", "25"))
OPENALEX_MAX_PAGES_PER_KEYWORD = int(
    os.environ.get("OPENALEX_MAX_PAGES_PER_KEYWORD", "5")
)
OPENALEX_MAX_TOTAL_PAGES = int(os.environ.get("OPENALEX_MAX_TOTAL_PAGES", "12"))
OPENALEX_MAX_TOTAL_WORKS = int(os.environ.get("OPENALEX_MAX_TOTAL_WORKS", "1200"))
OPENALEX_EARLY_STOP_CN_MULT = int(os.environ.get("OPENALEX_EARLY_STOP_CN_MULT", "1"))


DEFAULT_SOURCE = os.environ.get("DEFAULT_SOURCE", "openalex").strip().lower()

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
S2_WORKERS = int(os.environ.get("S2_WORKERS", "10"))

ROR_BASE = "https://api.ror.org"


# CN-only fallback when last_known_institution is missing:
# require at least N sampled works where this author has a CN affiliation.
# Default to 1 because many authors only appear once in the sampled top-cited works.
MIN_CN_AFFIL_WORKS_TOPIC = int(os.environ.get("MIN_CN_AFFIL_WORKS_TOPIC", "1"))
MIN_CN_AFFIL_WORKS_PERSON = int(os.environ.get("MIN_CN_AFFIL_WORKS_PERSON", "1"))


DEFAULT_BASE_URL = os.environ.get("DEFAULT_BASE_URL", "http://152.53.52.170:3003/v1")
DEFAULT_MODEL_GPT52 = os.environ.get("DEFAULT_MODEL_GPT52", "gpt-5.2-2025-12-11")
DEFAULT_MODEL_DEEPSEARCH = os.environ.get(
    "DEFAULT_MODEL_DEEPSEARCH", "gemini-2.5-pro-deepsearch-async"
)


app = FastAPI(title="World Model Crawler", version="0.1.0")


DEFAULT_RECENT_YEARS = int(os.environ.get("DEFAULT_RECENT_YEARS", "2"))

DEEPSEARCH_WORKERS = int(os.environ.get("DEEPSEARCH_WORKERS", "5"))
PROFILE_WORKERS = int(os.environ.get("PROFILE_WORKERS", "10"))

DETAILING_DEEPSEARCH_WORKERS = int(os.environ.get("DETAILING_DEEPSEARCH_WORKERS", "2"))
DETAILING_PROFILE_WORKERS = int(os.environ.get("DETAILING_PROFILE_WORKERS", "2"))

DEEPSEARCH_RETRIES = int(os.environ.get("DEEPSEARCH_RETRIES", "3"))
DEEPSEARCH_RETRY_BACKOFF_SEC = float(
    os.environ.get("DEEPSEARCH_RETRY_BACKOFF_SEC", "2")
)
DEEPSEARCH_RETRY_BACKOFF_MULT = float(
    os.environ.get("DEEPSEARCH_RETRY_BACKOFF_MULT", "2")
)
DEEPSEARCH_RETRY_JITTER_SEC = float(
    os.environ.get("DEEPSEARCH_RETRY_JITTER_SEC", "0.5")
)


@app.get("/api/models")
def api_models() -> JSONResponse:
    return JSONResponse(
        {
            "default_base_url": DEFAULT_BASE_URL,
            "default_model_gpt52": DEFAULT_MODEL_GPT52,
            "default_model_deepsearch": DEFAULT_MODEL_DEEPSEARCH,
            "models": MODEL_LIST,
        }
    )


@dataclass
class Job:
    job_id: str
    created_at: float
    status: str  # queued|running|done|error
    stage: str
    kind: str  # person|topic
    query: str
    deepsearch: bool
    result_name: Optional[str] = None
    result_topic_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancelled: bool = False
    fast_profile: bool = False


def _openalex_author_api_id(openalex_id: Any) -> Optional[str]:
    s = str(openalex_id or "").strip()
    if not s:
        return None
    # Common forms:
    # - https://openalex.org/A123...
    # - A123...
    if s.startswith("https://openalex.org/"):
        tail = s.rsplit("/", 1)[-1].strip()
        return tail if tail.startswith("A") else None
    if s.startswith("A"):
        return s
    return None


def _openalex_get_json(
    url: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 60
) -> Dict[str, Any]:
    import requests

    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    obj = r.json()
    return obj if isinstance(obj, dict) else {}


def _emit_openalex_page(
    job_id: Optional[str],
    *,
    endpoint: str,
    keyword: Optional[str],
    page: int,
    per_page: int,
    params: Dict[str, Any],
    http_status: Optional[int],
    elapsed_ms: Optional[int],
    results_count: Optional[int],
    totals: Optional[Dict[str, Any]] = None,
    early_stop: bool = False,
    error: Optional[str] = None,
) -> None:
    if not job_id:
        return
    _emit(
        job_id,
        {
            "type": "artifact",
            "kind": "openalex_page",
            "endpoint": endpoint,
            "keyword": keyword,
            "page": int(page),
            "per_page": int(per_page),
            "params": params,
            "http_status": http_status,
            "elapsed_ms": elapsed_ms,
            "results_count": results_count,
            "totals": totals or {},
            "early_stop": bool(early_stop),
            "error": error,
        },
    )


def _openalex_last_known_institution(author_obj: Dict[str, Any]) -> Dict[str, Any]:
    inst = author_obj.get("last_known_institution") or {}
    if not isinstance(inst, dict):
        inst = {}
    return {
        "id": inst.get("id"),
        "display_name": inst.get("display_name"),
        "country_code": inst.get("country_code"),
    }


def _openalex_is_cn_institution(author_obj: Dict[str, Any]) -> bool:
    inst = author_obj.get("last_known_institution") or {}
    if not isinstance(inst, dict):
        return False
    return str(inst.get("country_code") or "").strip().upper() == "CN"


def _openalex_authorship_cn_affiliation_count(
    work: Dict[str, Any], *, author_openalex_id: str
) -> int:
    """Count CN institutions in this work for the specific author."""
    try:
        # Full work object path.
        for a in work.get("authorships") or []:
            author = a.get("author") or {}
            if str(author.get("id") or "") != str(author_openalex_id):
                continue
            insts = a.get("institutions") or []
            cnt = 0
            for inst in insts:
                if not isinstance(inst, dict):
                    continue
                if str(inst.get("country_code") or "").strip().upper() == "CN":
                    cnt += 1
            return cnt

        # Lightweight sample_works path (used by topic stage).
        if str(work.get("author_openalex_id") or "") == str(author_openalex_id):
            insts2 = work.get("author_institutions") or []
            cnt2 = 0
            for inst in insts2:
                if not isinstance(inst, dict):
                    continue
                if str(inst.get("country_code") or "").strip().upper() == "CN":
                    cnt2 += 1
            return cnt2
    except Exception:
        return 0
    return 0


def _openalex_extract_author_institutions_from_work(
    work: Dict[str, Any], *, author_openalex_id: str
) -> List[Dict[str, Any]]:
    """Extract institutions list for the specific author from a work object."""
    out: List[Dict[str, Any]] = []
    try:
        # Full work object.
        for a in work.get("authorships") or []:
            author = a.get("author") or {}
            if str(author.get("id") or "") != str(author_openalex_id):
                continue
            for inst in a.get("institutions") or []:
                if not isinstance(inst, dict):
                    continue
                out.append(
                    {
                        "id": inst.get("id"),
                        "display_name": inst.get("display_name"),
                        "country_code": inst.get("country_code"),
                        "type": inst.get("type"),
                    }
                )
            break

        # Lightweight sample_works from topic stage: look up by work id.
        wid = work.get("openalex_work_id") or work.get("id")
        if wid and not out:
            obj = _openalex_get_json(
                f"{OPENALEX_BASE}/works/{str(wid).rsplit('/', 1)[-1]}", timeout=30
            )
            for a in obj.get("authorships") or []:
                author = a.get("author") or {}
                if str(author.get("id") or "") != str(author_openalex_id):
                    continue
                for inst in a.get("institutions") or []:
                    if not isinstance(inst, dict):
                        continue
                    out.append(
                        {
                            "id": inst.get("id"),
                            "display_name": inst.get("display_name"),
                            "country_code": inst.get("country_code"),
                            "type": inst.get("type"),
                        }
                    )
                break
    except Exception:
        return []
    return out


def _openalex_is_cn_by_recent_works(
    author_openalex_id: str,
    works: List[Dict[str, Any]],
    *,
    min_works_with_cn: int,
) -> bool:
    """Fallback CN check when author.last_known_institution is missing.

    We accept the author only if at least N of the sampled works show CN affiliation
    for this author in the authorship institutions list.
    """
    if min_works_with_cn <= 0:
        return False
    hit = 0
    for w in works or []:
        if (
            _openalex_authorship_cn_affiliation_count(
                w, author_openalex_id=author_openalex_id
            )
            > 0
        ):
            hit += 1
            if hit >= min_works_with_cn:
                return True
    return False


_jobs_lock = threading.Lock()
_jobs: Dict[str, Job] = {}
_job_events: Dict[str, "queue.Queue[dict]"] = {}

_topic_events_lock = threading.Lock()
_topic_events: Dict[str, "queue.Queue[dict]"] = {}

_topic_detailing_lock = threading.Lock()
_topic_detailing_state: Dict[str, Dict[str, Any]] = {}

_limit_lock = threading.Lock()
_active_jobs = 0
_ip_last_submit: Dict[str, float] = {}

MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
IP_COOLDOWN_SEC = float(os.environ.get("IP_COOLDOWN_SEC", "10"))


def _emit(job_id: str, event: Dict[str, Any]) -> None:
    q = _job_events.get(job_id)
    if not q:
        return
    payload = {
        "ts": time.time(),
        **event,
    }
    try:
        q.put_nowait(payload)
    except Exception:
        pass


def _emit_topic(topic_id: str, event: Dict[str, Any]) -> None:
    with _topic_events_lock:
        q = _topic_events.get(topic_id)
    if not q:
        return
    payload = {
        "ts": time.time(),
        **event,
    }
    try:
        q.put_nowait(payload)
    except Exception:
        pass


def _set_topic_detailing_state(topic_id: str, patch: Dict[str, Any]) -> None:
    with _topic_detailing_lock:
        cur = _topic_detailing_state.get(topic_id) or {}
        _topic_detailing_state[topic_id] = {**cur, **patch}


def _get_topic_detailing_state(topic_id: str) -> Dict[str, Any]:
    with _topic_detailing_lock:
        return dict(_topic_detailing_state.get(topic_id) or {})


def _set_job(job: Job) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job


def _get_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _write_topic_result(topic_id: str, obj: Dict[str, Any]) -> None:
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    p = TOPICS_DIR / f"{topic_id}.topic.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_topic_results() -> List[Dict[str, Any]]:
    if not TOPICS_DIR.exists():
        return []
    items = []
    for p in sorted(
        TOPICS_DIR.glob("*.topic.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "id": obj.get("id") or p.stem.replace(".topic", ""),
                "topic": obj.get("topic"),
                "created_at": obj.get("created_at"),
                "item_count": len(obj.get("items") or []),
            }
        )
    return items


def _default_topic_keywords(topic: str) -> List[str]:
    t = (topic or "").strip()
    if not t:
        return []
    base = t
    # Default option 2: multi-keyword expansion.
    seeds = [
        base,
        f"{base} world model",
        "world model",
        "video world model",
        "generative world model",
        "model-based reinforcement learning",
        "environment model",
        "embodied ai",
        "robot learning",
        "spatial intelligence",
        "neural simulator",
        "digital twin",
    ]
    out = []
    seen = set()
    for s in seeds:
        s2 = " ".join(str(s).split()).strip()
        if not s2 or s2.lower() in seen:
            continue
        seen.add(s2.lower())
        out.append(s2)
    return out


def _read_model_list() -> List[str]:
    p = DATA_DIR / "model_list.txt"
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []
    # File is a single comma-separated line.
    parts = [x.strip() for x in raw.split(",")]
    out: List[str] = []
    seen = set()
    for x in parts:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


MODEL_LIST = _read_model_list()


def _openalex_topic_leaderboard(
    *,
    topic: str,
    keywords: List[str],
    per_keyword_works: int = 200,
    top_n: int = 50,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    import math
    import requests

    # concurrent.futures kept for potential future parallelization
    import concurrent.futures

    base = OPENALEX_BASE
    works_per_page = max(1, min(int(OPENALEX_PER_PAGE), 200))

    select_fields = (
        "id,title,publication_year,cited_by_count,doi,primary_location,authorships"
    )

    def oa_get_paged(
        *, endpoint: str, keyword: str, page: int, per_page: int
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "search": keyword,
            "per-page": per_page,
            "page": page,
            "sort": "cited_by_count:desc",
            "select": select_fields,
        }
        t0 = time.time()
        status: Optional[int] = None
        try:
            r = requests.get(f"{base}{endpoint}", params=params, timeout=60)
            status = r.status_code
            r.raise_for_status()
            obj = r.json()
            if not isinstance(obj, dict):
                obj = {}
            return obj
        except Exception as e:
            _emit_openalex_page(
                job_id,
                endpoint=endpoint,
                keyword=keyword,
                page=page,
                per_page=per_page,
                params=params,
                http_status=status,
                elapsed_ms=int((time.time() - t0) * 1000),
                results_count=None,
                totals=None,
                early_stop=False,
                error=f"{type(e).__name__}: {e}",
            )
            return {}

    # NOTE: successful requests are emitted per page in the caller loop to include
    # aggregated totals and early_stop status.

    def compute_score(related_works: int, citations_sum: int, recent3: int) -> float:
        rel = min(related_works, 30) / 30.0
        imp = math.log1p(citations_sum) / math.log1p(50000)
        mom = min(recent3, 15) / 15.0
        return 100 * (0.50 * rel + 0.35 * imp + 0.15 * mom)

    current_year = time.gmtime().tm_year
    author_hits: Dict[str, Dict[str, Any]] = {}

    # Guardrails: bound exploration
    max_keywords = 6
    keywords = (keywords or [])[:max_keywords]
    target_cn = max(1, int(top_n) * max(1, int(OPENALEX_EARLY_STOP_CN_MULT)))

    total_pages = 0
    total_works = 0
    cn_author_ids: Dict[str, Dict[str, Any]] = {}

    for i, kw in enumerate(keywords, 1):
        if job_id:
            _emit(
                job_id,
                {
                    "type": "log",
                    "message": f"OpenAlex scan keyword: {kw} ({i}/{len(keywords)})",
                },
            )

        page = 1
        while True:
            if total_pages >= OPENALEX_MAX_TOTAL_PAGES:
                break
            if total_works >= OPENALEX_MAX_TOTAL_WORKS:
                break
            if page > OPENALEX_MAX_PAGES_PER_KEYWORD:
                break

            t0 = time.time()
            obj = oa_get_paged(
                endpoint="/works", keyword=kw, page=page, per_page=works_per_page
            )
            batch = obj.get("results") or []
            if not isinstance(batch, list):
                batch = []

            # If request failed (oa_get_paged returns {}), still record a page artifact and stop.
            if not obj:
                _emit_openalex_page(
                    job_id,
                    endpoint="/works",
                    keyword=kw,
                    page=page,
                    per_page=works_per_page,
                    params={
                        "search": kw,
                        "per-page": works_per_page,
                        "page": page,
                        "sort": "cited_by_count:desc",
                        "select": select_fields,
                    },
                    http_status=None,
                    elapsed_ms=int((time.time() - t0) * 1000),
                    results_count=0,
                    totals={
                        "works_fetched": total_works,
                        "authors_seen": len(author_hits),
                        "cn_authors_seen": len(cn_author_ids),
                        "total_pages": total_pages,
                    },
                    early_stop=False,
                    error="openalex request failed",
                )
                break

            total_pages += 1
            total_works += len(batch)

            # Aggregate authors and CN evidence from works' authorship institutions.
            for w in batch:
                if not isinstance(w, dict):
                    continue
                cited = int(w.get("cited_by_count") or 0)
                year = w.get("publication_year")
                is_recent3 = year is not None and (
                    current_year - int(year) <= DEFAULT_RECENT_YEARS
                )

                authorships = w.get("authorships") or []
                if not isinstance(authorships, list):
                    authorships = []

                for a in authorships:
                    if not isinstance(a, dict):
                        continue
                    author = a.get("author") or {}
                    if not isinstance(author, dict):
                        author = {}
                    author_id = author.get("id")
                    author_name = author.get("display_name") or ""
                    if not author_id:
                        continue

                    agg = author_hits.get(author_id)
                    if not agg:
                        author_hits[author_id] = {
                            "openalex_id": author_id,
                            "name": author_name,
                            "related_works": 0,
                            "related_citations_sum": 0,
                            "related_recent_3y": 0,
                            "keywords_hit": set(),
                            "sample_works": [],
                        }
                        agg = author_hits[author_id]

                    agg["related_works"] += 1
                    agg["related_citations_sum"] += cited
                    if is_recent3:
                        agg["related_recent_3y"] += 1
                    agg["keywords_hit"].add(kw)

                    # Keep some sample works (with authorships) for audit.
                    if len(agg["sample_works"]) < 5:
                        lp = (w.get("primary_location") or {}).get("landing_page_url")
                        agg["sample_works"].append(
                            {
                                "title": w.get("title"),
                                "year": year,
                                "cited_by_count": cited,
                                "openalex_work_id": w.get("id"),
                                "landing_page": lp,
                                "doi": w.get("doi"),
                                "keyword": kw,
                                "authorships": authorships,
                            }
                        )

                    # CN evidence for this author (institutions country_code == CN)
                    insts = a.get("institutions") or []
                    if isinstance(insts, list):
                        for inst in insts:
                            if not isinstance(inst, dict):
                                continue
                            if (
                                str(inst.get("country_code") or "").strip().upper()
                                != "CN"
                            ):
                                continue
                            if author_id not in cn_author_ids:
                                cn_author_ids[author_id] = {
                                    "author_id": author_id,
                                    "affiliation": inst.get("display_name"),
                                    "institution": {
                                        "id": inst.get("id"),
                                        "display_name": inst.get("display_name"),
                                        "country_code": inst.get("country_code"),
                                        "type": inst.get("type"),
                                    },
                                }
                            break

            early = len(cn_author_ids) >= target_cn
            _emit_openalex_page(
                job_id,
                endpoint="/works",
                keyword=kw,
                page=page,
                per_page=works_per_page,
                params={
                    "search": kw,
                    "per-page": works_per_page,
                    "page": page,
                    "sort": "cited_by_count:desc",
                    "select": select_fields,
                },
                http_status=200 if obj else None,
                elapsed_ms=int((time.time() - t0) * 1000),
                results_count=len(batch),
                totals={
                    "works_fetched": total_works,
                    "authors_seen": len(author_hits),
                    "cn_authors_seen": len(cn_author_ids),
                    "total_pages": total_pages,
                },
                early_stop=early,
                error=None,
            )

            if early:
                break
            if not batch:
                break
            page += 1

        if len(cn_author_ids) >= target_cn:
            break

    # Light author enrichment (no per-author API calls)
    def _build_base_item(agg: Dict[str, Any]) -> Dict[str, Any]:
        score = compute_score(
            related_works=int(agg["related_works"]),
            citations_sum=int(agg["related_citations_sum"]),
            recent3=int(agg["related_recent_3y"]),
        )
        evidence = [{"label": "OpenAlex Author", "url": agg.get("openalex_id")}]
        for sw in agg.get("sample_works") or []:
            if sw.get("openalex_work_id"):
                evidence.append(
                    {"label": "OpenAlex Work", "url": sw.get("openalex_work_id")}
                )
            if sw.get("doi"):
                evidence.append({"label": "DOI", "url": sw.get("doi")})
            if sw.get("landing_page"):
                evidence.append({"label": "Landing", "url": sw.get("landing_page")})
        # Dedup
        seen = set()
        dedup = []
        for e in evidence:
            u = e.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            dedup.append(e)
            if len(dedup) >= 20:
                break
        return {
            "name": agg.get("name"),
            "openalex_id": agg.get("openalex_id"),
            "affiliation": None,
            "country_code": None,
            "openalex_last_known_institution": None,
            "related_works": agg.get("related_works"),
            "related_citations_sum": agg.get("related_citations_sum"),
            "related_recent_3y": agg.get("related_recent_3y"),
            "score": score,
            # Normalize sample works to also carry CN affiliation evidence (if available)
            # so CN-only filtering can work even when author.last_known_institution is missing.
            "sample_works": [
                {
                    **sw,
                    "author_openalex_id": agg.get("openalex_id"),
                    "author_institutions": _openalex_extract_author_institutions_from_work(
                        sw, author_openalex_id=str(agg.get("openalex_id") or "")
                    ),
                    # Drop full authorships to keep payload smaller.
                    "authorships": None,
                }
                for sw in (agg.get("sample_works") or [])
                if isinstance(sw, dict)
            ],
            "evidence_links": dedup,
            "keywords_hit": sorted(list(agg.get("keywords_hit") or [])),
        }

    base_items: List[Dict[str, Any]] = []
    for _, agg in author_hits.items():
        base_items.append(_build_base_item(agg))

    # Keep only CN-only candidates (based on works affiliation evidence).
    if cn_author_ids:
        base_items = [
            it for it in base_items if str(it.get("openalex_id") or "") in cn_author_ids
        ]

        for it in base_items:
            meta = cn_author_ids.get(str(it.get("openalex_id") or "")) or {}
            inst = meta.get("institution") or {}
            it["country_code"] = "CN"
            it["affiliation"] = meta.get("affiliation")
            it["openalex_last_known_institution"] = inst

    base_items.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    total_candidates = len(base_items)
    items = base_items[:top_n]

    if job_id:
        _emit(
            job_id,
            {
                "type": "log",
                "message": f"OpenAlex CN-only: candidates={total_candidates} returned={len(items)}",
            },
        )

    return {
        "topic": topic,
        "keywords": keywords,
        "items": items,
        "total_candidates": total_candidates,
        "cn_only": True,
    }


def _list_people_files() -> List[Path]:
    # Prefer authoritative merged outputs.
    if PEOPLE_DIR.exists():
        files = sorted(PEOPLE_DIR.glob("*.person.json"))
        if files:
            return files
    return sorted(WEB_PEOPLE_DIR.glob("*.person.json"))


def _load_person(path: Path) -> Optional[Dict[str, Any]]:
    try:
        # Some existing files may be UTF-8 or legacy encodings.
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _index_people() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in _list_people_files():
        obj = _load_person(p)
        if not obj:
            continue
        out.append(obj)
    return out


@app.get("/")
def index():
    return RedirectResponse(url="/static/search.html")


@app.get("/api/people")
def api_people(q: str = "", sort: str = "S_prime_desc") -> JSONResponse:
    people = _index_people()

    qq = (q or "").strip().lower()
    if qq:

        def _hay(p: Dict[str, Any]) -> str:
            prof = p.get("profile") or {}

            def _to_text(x: Any) -> str:
                if x is None:
                    return ""
                if isinstance(x, str):
                    return x
                if isinstance(x, (int, float, bool)):
                    return str(x)
                if isinstance(x, list):
                    # flatten list recursively
                    return " ".join(_to_text(i) for i in x)
                if isinstance(x, dict):
                    # keep it lightweight; join values
                    return " ".join(_to_text(v) for v in x.values())
                return str(x)

            parts = [
                _to_text(p.get("name")),
                _to_text(prof.get("affiliation_current")),
                _to_text(prof.get("position_title")),
                _to_text(prof.get("research_keywords")),  # list or str both ok
            ]
            return " ".join(x for x in parts if x).lower()

        people = [p for p in people if qq in _hay(p)]

    def _score(p: Dict[str, Any], key: str) -> float:
        s = (p.get("score") or {}).get(key)
        if isinstance(s, (int, float)):
            return float(s)
        return -1e9

    if sort == "name_asc":
        people.sort(key=lambda p: str(p.get("name") or ""))
    elif sort == "S_desc":
        people.sort(key=lambda p: _score(p, "S"), reverse=True)
    else:
        people.sort(key=lambda p: _score(p, "S_prime"), reverse=True)

    return JSONResponse({"count": len(people), "items": people})


@app.get("/api/people/{name}")
def api_person(name: str) -> JSONResponse:
    # Try both locations.
    candidates = [
        PEOPLE_DIR / f"{name}.person.json",
        WEB_PEOPLE_DIR / f"{name}.person.json",
    ]
    for p in candidates:
        if p.exists():
            obj = _load_person(p)
            if obj is None:
                raise HTTPException(status_code=500, detail="invalid json")
            return JSONResponse(obj)
    raise HTTPException(status_code=404, detail="person not found")


def _write_person_output(name: str, person: Dict[str, Any]) -> None:
    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
    WEB_PEOPLE_DIR.mkdir(parents=True, exist_ok=True)

    text = json.dumps(person, ensure_ascii=False, indent=2)

    (PEOPLE_DIR / f"{name}.person.json").write_text(text, encoding="utf-8")
    (WEB_PEOPLE_DIR / f"{name}.person.json").write_text(text, encoding="utf-8")


def _read_person_output(name: str) -> Optional[Dict[str, Any]]:
    p = PEOPLE_DIR / f"{name}.person.json"
    if not p.exists():
        p = WEB_PEOPLE_DIR / f"{name}.person.json"
        if not p.exists():
            return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_topic_detailing(
    *,
    topic_id: str,
    topic: str,
    plan: Dict[str, Any],
    items: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    model_deepsearch: str,
    model_gpt: str,
    deepsearch_enabled: bool,
) -> None:
    """Background post-processing:

    - per-person deepresearch (best-effort) + incremental profile update
    - topic-level deepsearch notes (optional)
    - emits events via /api/topics/{topic_id}/events
    """
    import concurrent.futures

    total = len(items or [])
    _set_topic_detailing_state(topic_id, {"status": "running", "total": total})

    _emit_topic(topic_id, {"type": "detail_stage", "stage": "deepresearch_people"})

    done = 0
    updated = 0
    failed = 0

    def _base_materials(it: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "topic": topic,
                "definition": plan.get("definition")
                if isinstance(plan, dict)
                else None,
                "time_range": plan.get("time_range")
                if isinstance(plan, dict)
                else None,
                "plan": plan,
                "candidate": it,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _one(idx0: int, it: Dict[str, Any]) -> Dict[str, Any]:
        name0 = str((it.get("name") or "").strip())
        if not name0:
            return {"ok": False, "name": name0, "error": "missing name"}

        if not deepsearch_enabled:
            return {"ok": False, "name": name0, "error": "deepsearch disabled"}

        base_mat = _base_materials(it)
        dr_full = _run_deepsearch(
            name=name0,
            materials=base_mat,
            api_key=api_key,
            base_url=base_url,
            model_id=model_deepsearch,
            job_id=topic_id,  # not a job; but used for cancellation checks only (no cancel here)
        )
        add = ""
        if dr_full and "[DeepSearch Addendum]" in dr_full:
            add = dr_full.split("[DeepSearch Addendum]", 1)[-1].strip()
        if not add:
            return {"ok": False, "name": name0, "error": "empty addendum"}

        prev = _read_person_output(name0) or {}
        prev_prof = prev.get("profile")
        prev_score = prev.get("score")

        upd = _gpt52_update_profile_and_score(
            name=name0,
            base_materials=base_mat,
            deepresearch_addendum=add,
            previous_profile=prev_prof,
            previous_score=prev_score,
            api_key=api_key,
            base_url=base_url,
            model_id=model_gpt,
        )

        rev = int(((prev.get("sources") or {}).get("profile_revision") or 1)) + 1
        person_obj = {
            "name": name0,
            "profile": upd.get("profile"),
            "score": upd.get("score"),
            "sources": {
                **(
                    (prev.get("sources") or {})
                    if isinstance(prev.get("sources"), dict)
                    else {}
                ),
                "last_updated_at": time.time(),
                "profile_revision": rev,
                "has_deepresearch": True,
            },
        }
        _write_person_output(name0, person_obj)
        return {"ok": True, "name": name0, "revision": rev}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(DETAILING_DEEPSEARCH_WORKERS))
    ) as ex:
        futs = [ex.submit(_one, i, it) for i, it in enumerate(items or [])]
        for fut in concurrent.futures.as_completed(futs):
            res = {}
            try:
                res = fut.result()
            except Exception as e:
                res = {"ok": False, "name": None, "error": f"{type(e).__name__}: {e}"}

            done += 1
            if res.get("ok"):
                updated += 1
                _emit_topic(
                    topic_id,
                    {
                        "type": "detail_person_updated",
                        "name": res.get("name"),
                        "revision": res.get("revision"),
                    },
                )
            else:
                failed += 1

            _set_topic_detailing_state(
                topic_id,
                {"done": done, "updated": updated, "failed": failed},
            )
            _emit_topic(
                topic_id,
                {
                    "type": "detail_progress",
                    "done": done,
                    "total": total,
                    "updated": updated,
                    "failed": failed,
                },
            )

    # Optional topic-level notes
    notes = None
    if deepsearch_enabled:
        _emit_topic(topic_id, {"type": "detail_stage", "stage": "topic_notes"})
        try:
            add = _run_deepsearch(
                name=topic,
                materials=json.dumps(
                    {"topic": topic, "items": items}, ensure_ascii=False, indent=2
                ),
                api_key=api_key,
                base_url=base_url,
                model_id=model_deepsearch,
                job_id=topic_id,
            )
            if add and "[DeepSearch Addendum]" in add:
                notes = add.split("[DeepSearch Addendum]", 1)[-1].strip() or None
        except Exception:
            notes = None

    if notes:
        p = TOPICS_DIR / f"{topic_id}.topic.json"
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj["notes"] = notes
                obj["updated_at"] = time.time()
                p.write_text(
                    json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception:
            pass

    _set_topic_detailing_state(
        topic_id,
        {"status": "done", "finished_at": time.time()},
    )
    _emit_topic(topic_id, {"type": "detail_done", "topic_id": topic_id})
    _emit_topic(topic_id, {"type": "eof"})


def _openalex_materials(name: str) -> Dict[str, Any]:
    # Minimal, no-key OpenAlex fetch for V1.
    import requests

    base = "https://api.openalex.org"
    # Search authors (take multiple candidates; select strict CN institution)
    t0 = time.time()
    params0 = {"search": name, "per-page": 5}
    r = requests.get(f"{base}/authors", params=params0, timeout=30)
    r.raise_for_status()
    # NOTE: in person flow we currently don't pass job_id into _openalex_materials;
    # keep this observable via logs elsewhere if needed.
    authors = r.json().get("results") or []
    if not authors:
        return {
            "author": None,
            "works": [],
            "evidence_links": [],
            "notes": "OpenAlex authors search returned 0 results",
        }

    # We'll pick a CN author either by last_known_institution==CN OR by CN affiliation evidence
    # from their top works (authorship institutions country_code==CN).
    chosen_author_obj: Optional[Dict[str, Any]] = None
    for cand in authors[:5]:
        cand_id = cand.get("id") if isinstance(cand, dict) else None
        aid = _openalex_author_api_id(cand_id)
        if not aid:
            continue

        try:
            ao = _openalex_get_json(f"{base}/authors/{aid}", timeout=30)
        except Exception:
            ao = {}

        author_openalex_id = str(ao.get("id") or cand_id or "")
        works_api = (ao.get("works_api_url") or "") if isinstance(ao, dict) else ""

        works_preview: List[Dict[str, Any]] = []
        if works_api:
            try:
                rw = requests.get(
                    works_api,
                    params={"per-page": 25, "sort": "cited_by_count:desc"},
                    timeout=30,
                )
                rw.raise_for_status()
                works_preview = (rw.json().get("results") or [])[:25]
            except Exception:
                works_preview = []

        if _openalex_is_cn_institution(ao):
            chosen_author_obj = ao
            break

        if (
            author_openalex_id
            and works_preview
            and _openalex_is_cn_by_recent_works(
                author_openalex_id,
                works_preview,
                min_works_with_cn=MIN_CN_AFFIL_WORKS_PERSON,
            )
        ):
            # Accept with CN evidence even if last_known_institution missing.
            if isinstance(ao, dict):
                ao = dict(ao)
            ao["_cn_evidence_from_works"] = True
            chosen_author_obj = ao
            break

    if not chosen_author_obj:
        return {
            "author": None,
            "works": [],
            "evidence_links": [],
            "notes": "No CN-author found (strict CN-only) in OpenAlex top results",
            "cn_only": True,
        }

    author = chosen_author_obj
    author_id = author.get("id")
    evidence = []
    if author_id:
        evidence.append({"label": "OpenAlex Author", "url": author_id})

    # Fetch some works linked from author
    works: List[Dict[str, Any]] = []
    works_api = (author.get("works_api_url") or "") if isinstance(author, dict) else ""
    if works_api:
        rw = requests.get(
            works_api,
            params={"per-page": 25, "sort": "cited_by_count:desc"},
            timeout=30,
        )
        rw.raise_for_status()
        for w in (rw.json().get("results") or [])[:25]:
            works.append(
                {
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "venue": (w.get("primary_location") or {})
                    .get("source", {})
                    .get("display_name"),
                    "cited_by_count": w.get("cited_by_count"),
                    "openalex": w.get("id"),
                    "doi": w.get("doi"),
                    "landing_page": (w.get("primary_location") or {}).get(
                        "landing_page_url"
                    ),
                }
            )
            if w.get("id"):
                evidence.append({"label": "OpenAlex Work", "url": w.get("id")})
            if w.get("doi"):
                evidence.append({"label": "DOI", "url": w.get("doi")})
            lp = (w.get("primary_location") or {}).get("landing_page_url")
            if lp:
                evidence.append({"label": "Landing", "url": lp})

    # Dedup evidence by url
    seen = set()
    dedup = []
    for e in evidence:
        u = e.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        dedup.append({"label": e.get("label") or "link", "url": u})
        if len(dedup) >= 20:
            break

    return {
        "author": {
            "openalex_id": author_id,
            "display_name": author.get("display_name"),
            "last_known_institution": (author.get("last_known_institution") or {}).get(
                "display_name"
            ),
            "last_known_institution_country_code": (
                author.get("last_known_institution") or {}
            ).get("country_code"),
            "cn_evidence_from_works": bool(author.get("_cn_evidence_from_works")),
            "works_count": author.get("works_count"),
            "cited_by_count": author.get("cited_by_count"),
            "homepage_url": (
                author.get("homepage_url") if isinstance(author, dict) else None
            ),
        },
        "works": works,
        "evidence_links": dedup,
        "cn_only": True,
    }


def _materials_to_text(name: str, oa: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Researcher: {name}")
    lines.append("\n[OpenAlex Author]")
    lines.append(json.dumps(oa.get("author"), ensure_ascii=False, indent=2))
    lines.append("\n[Top Works (OpenAlex)]")
    lines.append(json.dumps(oa.get("works") or [], ensure_ascii=False, indent=2))
    lines.append("\n[Evidence Links]")
    lines.append(
        json.dumps(oa.get("evidence_links") or [], ensure_ascii=False, indent=2)
    )
    return "\n".join(lines)


def _gpt52_profile_and_score(
    *,
    name: str,
    materials: str,
    api_key: str,
    base_url: str,
    model_id: str,
) -> Dict[str, Any]:
    # V1: reuse the existing prompts (as text templates) but call directly.
    from core.llm_client import chat

    profile_system = "You are an information extraction system. Output JSON only."
    profile_prompt = (
        "You are a researcher information extraction system.\n"
        "Strictly extract information from the provided materials. Do not fabricate.\n"
        "Language: Output all human-readable text in Simplified Chinese (zh-CN).\n"
        "- Do NOT translate URLs/DOIs/IDs.\n"
        "- Do NOT invent a Chinese name.\n"
        "- For the profile 'name' field: if a verified Chinese name exists in materials, output '中文名（English name）'; otherwise output the English name only.\n"
        "- Always keep the English name available in aliases if it is not already included.\n"
        "Output a single JSON object with keys: name, aliases, affiliation_current, position_title, "
        "location, email_public, homepage, google_scholar, semantic_scholar, openalex, orcid, dblp, github, "
        "research_keywords, representative_works, world_model_relevance, metrics, community_roles, evidence_links, missing_or_uncertain.\n"
        "If unknown, use null or empty arrays.\n\n"
        "MATERIALS:\n<<<MATERIALS>>>\n"
    ).replace("<<<MATERIALS>>>", materials)

    score_system = "You are a rigorous academic evaluator. Output JSON only."
    score_prompt = (
        "You are an academic evaluator for World Models / Embodied AI.\n"
        "Score strictly based on provided materials; missing info must reduce completeness.\n"
        "Language: Output all human-readable text in Simplified Chinese (zh-CN).\n"
        "- Do NOT translate URLs/DOIs/IDs.\n"
        "- Do NOT fabricate evidence.\n"
        "Output a single JSON object containing: name, affiliation, primary_directions, S, S_prime, completeness, "
        "dimensions{impact,field,quality,community,momentum}, dimension_explanations{...}, evidence[], "
        "missing_or_uncertain[], one_line_conclusion.\n"
        "Evidence max 12 items.\n\n"
        "MATERIALS:\n<<<MATERIALS>>>\n"
    ).replace("<<<MATERIALS>>>", materials)

    url = base_url.rstrip("/") + "/chat/completions"

    prof_text = chat(
        profile_prompt,
        api_key=api_key,
        model_id=model_id,
        url=url,
        temperature=0.0,
        timeout=180,
        system=profile_system,
        extra={"response_format": {"type": "json_object"}},
    )

    score_text = chat(
        score_prompt,
        api_key=api_key,
        model_id=model_id,
        url=url,
        temperature=0.0,
        timeout=180,
        system=score_system,
        extra={"response_format": {"type": "json_object"}},
    )

    def _parse(text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        try:
            return json.loads(t)
        except Exception:
            l = t.find("{")
            r = t.rfind("}")
            if l != -1 and r != -1 and r > l:
                return json.loads(t[l : r + 1])
            raise

    profile = _parse(prof_text)
    score = _parse(score_text)

    return {"profile": profile, "score": score}


def _gpt52_update_profile_and_score(
    *,
    name: str,
    base_materials: str,
    deepresearch_addendum: str,
    previous_profile: Any,
    previous_score: Any,
    api_key: str,
    base_url: str,
    model_id: str,
) -> Dict[str, Any]:
    """Incrementally update profile/score using additional evidence.

    The model should preserve schema and only improve/clarify based on new materials.
    """
    from core.llm_client import chat

    url = base_url.rstrip("/") + "/chat/completions"
    system = "You are a meticulous information extraction and evaluation system. Output JSON only."
    prompt = (
        "You will UPDATE an existing researcher profile and score using NEW evidence.\n"
        "Rules:\n"
        "- Do NOT fabricate.\n"
        "- Prefer to fill missing fields and replace uncertain placeholders with supported facts.\n"
        "- Keep the same schema for profile and score as existing outputs.\n"
        "- Language: Output all human-readable text in Simplified Chinese (zh-CN).\n"
        "- Do NOT translate URLs/DOIs/IDs.\n"
        "- Do NOT invent a Chinese name.\n"
        "- For the profile 'name' field: if a verified Chinese name exists in materials, output '中文名（English name）'; otherwise keep the English name only.\n"
        "- Ensure the English name is preserved in aliases if not already included.\n"
        '- Output ONE JSON object: {"profile": <obj>, "score": <obj>}\n\n'
        "[NAME]\n<<<NAME>>>\n\n"
        "[BASE_MATERIALS]\n<<<BASE>>>\n\n"
        "[PREVIOUS_PROFILE]\n<<<PP>>>\n\n"
        "[PREVIOUS_SCORE]\n<<<PS>>>\n\n"
        "[DEEPRESEARCH_ADDENDUM]\n<<<ADD>>>\n"
    )
    prompt = (
        prompt.replace("<<<NAME>>>", name)
        .replace("<<<BASE>>>", base_materials)
        .replace("<<<PP>>>", json.dumps(previous_profile, ensure_ascii=False, indent=2))
        .replace("<<<PS>>>", json.dumps(previous_score, ensure_ascii=False, indent=2))
        .replace("<<<ADD>>>", deepresearch_addendum or "")
    )

    text = chat(
        prompt,
        api_key=api_key,
        model_id=model_id,
        url=url,
        temperature=0.0,
        timeout=180,
        system=system,
        extra={"response_format": {"type": "json_object"}},
    )

    t = (text or "").strip()
    try:
        obj = json.loads(t)
    except Exception:
        l = t.find("{")
        r = t.rfind("}")
        if l != -1 and r != -1 and r > l:
            obj = json.loads(t[l : r + 1])
        else:
            raise
    if not isinstance(obj, dict):
        return {"profile": previous_profile, "score": previous_score}
    return {
        "profile": obj.get("profile")
        if obj.get("profile") is not None
        else previous_profile,
        "score": obj.get("score") if obj.get("score") is not None else previous_score,
    }


def _run_deepsearch(
    *,
    name: str,
    materials: str,
    api_key: str,
    base_url: str,
    model_id: str,
    job_id: str,
) -> str:
    # Uses the deepsearch model via the same OpenAI-compatible gateway.
    from core.llm_client import stream_chat

    system = "You are a meticulous research assistant."
    prompt = (
        "Please do deep research for the researcher and provide additional evidence links.\n"
        "Rules: give clickable links (homepage/scholar/dblp/openreview/arxiv/github/institution); "
        "do not fabricate; mark uncertain information.\n\n"
        f"Researcher: {name}\n\n"
        "Existing materials:\n<<<MATERIALS>>>\n"
        "\nOutput only the addendum content (no surrounding commentary)."
    ).replace("<<<MATERIALS>>>", materials)

    import random
    import requests

    url = base_url.rstrip("/") + "/chat/completions"

    def _should_stop() -> bool:
        # job_id can be a real job (normal path) or a topic_id (detailing path).
        try:
            j = _get_job(job_id)
            return bool(j.cancelled or j.fast_profile)
        except Exception:
            return False

    def _emit_stop(where: str) -> None:
        try:
            j = _get_job(job_id)
        except Exception:
            return
        _emit(
            job_id,
            {
                "type": "cancelled" if j.cancelled else "fast_profile",
                "where": where,
            },
        )

    def _sleep_backoff(sec: float) -> None:
        # Sleep in small steps so cancel/fast_profile can interrupt.
        end = time.time() + max(0.0, sec)
        while time.time() < end:
            if _should_stop():
                return
            time.sleep(min(0.25, end - time.time()))

    attempts = max(1, int(DEEPSEARCH_RETRIES))
    backoff = max(0.0, float(DEEPSEARCH_RETRY_BACKOFF_SEC))
    mult = max(1.0, float(DEEPSEARCH_RETRY_BACKOFF_MULT))
    jitter = max(0.0, float(DEEPSEARCH_RETRY_JITTER_SEC))

    for attempt in range(1, attempts + 1):
        if _should_stop():
            _emit_stop("deepsearch")
            return materials

        chunks: List[str] = []
        last = time.time()
        try:
            _emit(
                job_id,
                {
                    "type": "log",
                    "message": f"Deepsearch start (attempt {attempt}/{attempts}) model={model_id}",
                },
            )

            for c in stream_chat(
                prompt,
                api_key=api_key,
                model_id=model_id,
                url=url,
                temperature=0.2,
                timeout=1200,
                system=system,
                extra=None,
            ):
                if _should_stop():
                    _emit_stop("deepsearch")
                    return materials
                chunks.append(c)
                now = time.time()
                if now - last >= 1.0:
                    _emit(
                        job_id,
                        {
                            "type": "log",
                            "message": f"Deepsearch streaming... ({len(chunks)} chunks)",
                        },
                    )
                    last = now

            add = "".join(chunks).strip()
            if not add:
                _emit(job_id, {"type": "log", "message": "Deepsearch returned empty"})
                return materials

            _emit(job_id, {"type": "log", "message": "Deepsearch completed"})
            return materials + "\n\n[DeepSearch Addendum]\n" + add

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ReadTimeout,
        ) as e:
            msg = f"{type(e).__name__}: {e}"
            if attempt >= attempts:
                _emit(
                    job_id,
                    {
                        "type": "log",
                        "message": f"Deepsearch failed after {attempts} attempts: {msg} (skipped)",
                    },
                )
                return materials

            # Retry with backoff.
            sleep_s = backoff * (mult ** (attempt - 1))
            if jitter:
                sleep_s += random.random() * jitter
            _emit(
                job_id,
                {
                    "type": "log",
                    "message": f"Deepsearch transient error: {msg} (retry in {sleep_s:.1f}s)",
                },
            )
            _sleep_backoff(sleep_s)
            continue

        except Exception as e:
            # Non-retriable unknown error: skip.
            _emit(
                job_id,
                {
                    "type": "log",
                    "message": f"Deepsearch unexpected error: {type(e).__name__}: {e} (skipped)",
                },
            )
            return materials

    return materials


def _gpt52_parse_topic_plan(
    *, text: str, api_key: str, base_url: str, model_id: str
) -> Dict[str, Any]:
    from core.llm_client import chat

    url = base_url.rstrip("/") + "/chat/completions"
    system = "You are a research planning assistant. Output JSON only."

    prompt = (
        "You will parse a user's free-form research topic request into a deterministic search plan.\n"
        "Output ONE JSON object only with the following schema:\n"
        "{\n"
        '  "topic_name": string,\n'
        '  "definition": string|null,\n'
        '  "definition_draft": string,\n'
        '  "time_range": {"recent_years": number|null, "start_year": number|null, "end_year": number|null},\n'
        '  "keywords": [string],\n'
        '  "must_terms": [string],\n'
        '  "exclude_terms": [string],\n'
        '  "subtopics": [string],\n'
        '  "missing_fields": [string],\n'
        '  "questions_for_user": [string]\n'
        "}\n"
        "Constraints:\n"
        "- keywords: 3-10 items, short phrases\n"
        "- if time range not specified, set time_range.recent_years to null (server will default)\n"
        "- definition_draft: ALWAYS provide a 1-3 sentence draft definition/range in zh-CN with include/exclude hints\n"
        "- missing_fields should include 'definition' if definition is missing/unclear\n"
        "- questions_for_user: ask at most 3 questions\n\n"
        "User input:\n<<<TEXT>>>\n"
    ).replace("<<<TEXT>>>", text)

    resp = chat(
        prompt,
        api_key=api_key,
        model_id=model_id,
        url=url,
        temperature=0.0,
        timeout=180,
        system=system,
        extra={"response_format": {"type": "json_object"}},
    )

    t = (resp or "").strip()
    try:
        return json.loads(t)
    except Exception:
        l = t.find("{")
        r = t.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(t[l : r + 1])
        raise


def _run_job(job_id: str, *, name: str, api_key: str, deepsearch: bool) -> None:
    # api_key is intentionally only kept in this stack frame.
    global _active_jobs
    job = _get_job(job_id)
    job.status = "running"
    job.stage = "openalex"
    job.kind = "person"
    job.query = name
    _set_job(job)
    _emit(job_id, {"type": "stage", "stage": job.stage})

    try:
        oa = _openalex_materials(name)
        _emit(job_id, {"type": "log", "message": "OpenAlex fetched"})

        # Strict CN-only guardrail: do not proceed if no CN institution author.
        if not (oa.get("author") and isinstance(oa.get("author"), dict)):
            raise ValueError(
                "CN-only: researcher not found with CN institution in OpenAlex"
            )
        cc = (oa.get("author") or {}).get("last_known_institution_country_code")
        if str(cc or "").strip().upper() != "CN":
            # Allow fallback only if we have strong CN affiliation evidence from works.
            if not bool((oa.get("author") or {}).get("cn_evidence_from_works")):
                raise ValueError("CN-only: last_known_institution.country_code != CN")

        try:
            author = oa.get("author") if isinstance(oa, dict) else None
            works = oa.get("works") if isinstance(oa, dict) else None
            if isinstance(author, dict) or isinstance(works, list):
                _emit(
                    job_id,
                    {
                        "type": "artifact",
                        "kind": "openalex",
                        "name": name,
                        "author": author,
                        "works_preview": (works or [])[:10]
                        if isinstance(works, list)
                        else [],
                        "evidence_links": (oa.get("evidence_links") or [])
                        if isinstance(oa, dict)
                        else [],
                    },
                )
        except Exception:
            pass

        materials = _materials_to_text(name, oa)

        if deepsearch:
            job.stage = "deepsearch"
            _set_job(job)
            _emit(job_id, {"type": "stage", "stage": job.stage})
            materials = _run_deepsearch(
                name=name,
                materials=materials,
                api_key=api_key,
                base_url=DEFAULT_BASE_URL,
                model_id=DEFAULT_MODEL_DEEPSEARCH,
                job_id=job_id,
            )

        job.stage = "gpt52"
        _set_job(job)
        _emit(job_id, {"type": "stage", "stage": job.stage})

        out = _gpt52_profile_and_score(
            name=name,
            materials=materials,
            api_key=api_key,
            base_url=DEFAULT_BASE_URL,
            model_id=DEFAULT_MODEL_GPT52,
        )

        person = {
            "name": name,
            "profile": out.get("profile"),
            "score": out.get("score"),
            "sources": {
                "created_at": time.time(),
                "openalex": oa.get("author", {}).get("openalex_id")
                if isinstance(oa, dict)
                else None,
                "used_deepsearch": bool(deepsearch),
                "base_url": DEFAULT_BASE_URL,
                "model_gpt": DEFAULT_MODEL_GPT52,
                "model_deepsearch": DEFAULT_MODEL_DEEPSEARCH,
            },
        }
        _write_person_output(name, person)

        job.status = "done"
        job.stage = "done"
        job.result_name = name
        _set_job(job)
        _emit(job_id, {"type": "done", "name": name})
    except Exception as e:
        job.status = "error"
        job.stage = "error"
        job.error = f"{type(e).__name__}: {e}"
        _set_job(job)
        _emit(job_id, {"type": "error", "message": job.error})
    finally:
        with _limit_lock:
            _active_jobs = max(0, _active_jobs - 1)
        # Signal end of stream.
        _emit(job_id, {"type": "eof"})


def _run_topic_job(job_id: str, *, topic: str, api_key: str, deepsearch: bool) -> None:
    global _active_jobs
    job = _get_job(job_id)
    job.status = "running"
    job.stage = "openalex"
    _set_job(job)
    _emit(job_id, {"type": "stage", "stage": job.stage})

    try:
        job.kind = "topic"
        job.query = topic
        _set_job(job)

        job.stage = "plan"
        _set_job(job)
        _emit(job_id, {"type": "stage", "stage": job.stage})

        plan = _gpt52_parse_topic_plan(
            text=topic,
            api_key=api_key,
            base_url=DEFAULT_BASE_URL,
            model_id=DEFAULT_MODEL_GPT52,
        )

        try:
            _emit(
                job_id,
                {
                    "type": "artifact",
                    "kind": "plan",
                    "topic": topic,
                    "plan": {
                        "topic_name": plan.get("topic_name")
                        if isinstance(plan, dict)
                        else None,
                        "definition": plan.get("definition")
                        if isinstance(plan, dict)
                        else None,
                        "time_range": plan.get("time_range")
                        if isinstance(plan, dict)
                        else None,
                        "keywords": plan.get("keywords")
                        if isinstance(plan, dict)
                        else None,
                        "must_terms": plan.get("must_terms")
                        if isinstance(plan, dict)
                        else None,
                        "exclude_terms": plan.get("exclude_terms")
                        if isinstance(plan, dict)
                        else None,
                        "subtopics": plan.get("subtopics")
                        if isinstance(plan, dict)
                        else None,
                        "missing_fields": plan.get("missing_fields")
                        if isinstance(plan, dict)
                        else None,
                        "questions_for_user": plan.get("questions_for_user")
                        if isinstance(plan, dict)
                        else None,
                    },
                },
            )
        except Exception:
            pass

        # Default time range if missing
        tr = plan.get("time_range") if isinstance(plan, dict) else None
        if not isinstance(tr, dict):
            tr = {}
            plan["time_range"] = tr
        if (
            tr.get("recent_years") is None
            and tr.get("start_year") is None
            and tr.get("end_year") is None
        ):
            tr["recent_years"] = DEFAULT_RECENT_YEARS

        # If definition missing, ask once (but always provide a draft for confirmation).
        missing = plan.get("missing_fields") if isinstance(plan, dict) else None
        missing = missing if isinstance(missing, list) else []
        definition = plan.get("definition")
        definition_draft = (
            plan.get("definition_draft") if isinstance(plan, dict) else None
        )
        if not isinstance(definition_draft, str) or not definition_draft.strip():
            # Fallback draft if model didn't provide.
            definition_draft = (
                f"【定义草案】这里的“{topic}”指与该主题相关的具身智能/机器人/学习系统研究，"
                "重点关注可验证的学术贡献（论文/项目/代码/数据集）与可追溯链接；"
                "默认时间范围为近几年（可按需要扩展），排除纯商业宣传与缺少证据的描述。"
            )
            if isinstance(plan, dict):
                plan["definition_draft"] = definition_draft

        need_def = (not definition) or (
            "definition" in [str(x).lower() for x in missing]
        )
        asked = (
            (job.meta or {}).get("asked_definition")
            if isinstance(job.meta, dict)
            else False
        )
        if need_def and not asked:
            job.status = "waiting_user"
            job.stage = "need_input"
            job.meta = {
                **(job.meta or {}),
                "plan": plan,
                "asked_definition": True,
                "original_text": topic,
                "definition_draft": definition_draft,
            }
            _set_job(job)
            questions = plan.get("questions_for_user")
            if not isinstance(questions, list) or not questions:
                questions = [
                    "请用 1-3 句话说明你这里的‘世界模型’定义/范围（包含/不包含什么）。"
                ]
            _emit(
                job_id,
                {
                    "type": "need_input",
                    "missing_fields": ["definition"],
                    "questions": questions[:3],
                    "mode": "confirm_definition",
                    "default_text": definition_draft,
                    "plan_preview": {
                        "topic_name": plan.get("topic_name"),
                        "time_range": plan.get("time_range"),
                        "keywords": plan.get("keywords"),
                        "must_terms": plan.get("must_terms"),
                        "exclude_terms": plan.get("exclude_terms"),
                        "definition_draft": definition_draft,
                    },
                },
            )
            return

        job.meta = {**(job.meta or {}), "plan": plan, "original_text": topic}
        _set_job(job)

        keywords = plan.get("keywords") if isinstance(plan, dict) else None
        if not isinstance(keywords, list) or not keywords:
            keywords = _default_topic_keywords(topic)
        else:
            # fallback expansion
            expanded = _default_topic_keywords(topic)
            seen = set([str(x).lower() for x in keywords])
            for x in expanded:
                if x.lower() in seen:
                    continue
                keywords.append(x)
                seen.add(x.lower())

        job.stage = "openalex"
        _set_job(job)
        _emit(job_id, {"type": "stage", "stage": job.stage})

        lb = _openalex_topic_leaderboard(
            topic=topic,
            keywords=keywords,
            per_keyword_works=200,
            top_n=50,
            job_id=job_id,
        )

        # If user cancels after candidates, proceed with what we have (no deepresearch/profile).
        if _get_job(job_id).cancelled:
            topic_id = uuid.uuid4().hex
            out_obj = {
                "id": topic_id,
                "topic": (
                    plan.get("topic_name")
                    if isinstance(plan, dict) and plan.get("topic_name")
                    else topic
                ),
                "created_at": time.time(),
                "sources": {
                    "used_deepsearch": False,
                    "base_url": DEFAULT_BASE_URL,
                    "model_gpt": DEFAULT_MODEL_GPT52,
                    "model_deepsearch": DEFAULT_MODEL_DEEPSEARCH,
                    "cancelled_early": True,
                },
                "input_text": topic,
                "definition": plan.get("definition")
                if isinstance(plan, dict)
                else None,
                "time_range": plan.get("time_range")
                if isinstance(plan, dict)
                else None,
                "plan": {
                    "keywords": plan.get("keywords")
                    if isinstance(plan, dict)
                    else None,
                    "must_terms": plan.get("must_terms")
                    if isinstance(plan, dict)
                    else None,
                    "exclude_terms": plan.get("exclude_terms")
                    if isinstance(plan, dict)
                    else None,
                    "subtopics": plan.get("subtopics")
                    if isinstance(plan, dict)
                    else None,
                },
                "keywords": lb.get("keywords"),
                "total_candidates": lb.get("total_candidates"),
                "items": lb.get("items"),
                "notes": "Cancelled early: returned OpenAlex candidates only (no deepresearch/profile)",
            }
            _write_topic_result(topic_id, out_obj)
            job.status = "done"
            job.stage = "done"
            job.result_topic_id = topic_id
            _set_job(job)
            _emit(job_id, {"type": "done", "topic_id": topic_id, "topic": topic})
            return

        try:
            preview_items = []
            for it in (lb.get("items") or [])[:50]:
                if not isinstance(it, dict):
                    continue
                preview_items.append(
                    {
                        "name": it.get("name"),
                        "openalex_id": it.get("openalex_id"),
                        "score": it.get("score"),
                        "related_works": it.get("related_works"),
                        "related_citations_sum": it.get("related_citations_sum"),
                        "related_recent_3y": it.get("related_recent_3y"),
                        "keywords_hit": it.get("keywords_hit"),
                        "sample_works": (it.get("sample_works") or [])[:2],
                        "evidence_links": (it.get("evidence_links") or [])[:6],
                    }
                )

            _emit(
                job_id,
                {
                    "type": "artifact",
                    "kind": "candidates",
                    "topic": topic,
                    "keywords": lb.get("keywords"),
                    "total_candidates": lb.get("total_candidates"),
                    "items_preview": preview_items,
                },
            )
        except Exception:
            pass

        # Fast path: generate initial profiles first (no deepresearch blocking).
        job.stage = "profile_people"
        _set_job(job)
        _emit(job_id, {"type": "stage", "stage": job.stage})

        import concurrent.futures

        items = lb.get("items") or []
        total_items = len(items)

        # ---- Initial profiles (fast) ----
        def _do_one_profile_fast(it: Dict[str, Any]) -> Dict[str, Any]:
            person_name = (it.get("name") or "").strip()
            base_materials_obj = {
                "topic": topic,
                "definition": plan.get("definition")
                if isinstance(plan, dict)
                else None,
                "time_range": plan.get("time_range")
                if isinstance(plan, dict)
                else None,
                "plan": plan,
                "candidate": it,
            }
            base_materials = json.dumps(
                base_materials_obj, ensure_ascii=False, indent=2
            )
            out = _gpt52_profile_and_score(
                name=person_name,
                materials=base_materials,
                api_key=api_key,
                base_url=DEFAULT_BASE_URL,
                model_id=DEFAULT_MODEL_GPT52,
            )
            person_obj = {
                "name": person_name,
                "profile": out.get("profile"),
                "score": out.get("score"),
                "sources": {
                    "created_at": time.time(),
                    "last_updated_at": time.time(),
                    "profile_revision": 1,
                    "pipeline_topic": topic,
                    "base_url": DEFAULT_BASE_URL,
                    "model_gpt": DEFAULT_MODEL_GPT52,
                    "model_deepsearch": DEFAULT_MODEL_DEEPSEARCH,
                    "has_deepresearch": False,
                },
            }
            _write_person_output(person_name, person_obj)
            it2 = dict(it)
            it2["person_json"] = f"people/{person_name}.person.json"
            return it2

        prof_done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=PROFILE_WORKERS) as ex:
            futs = [
                ex.submit(_do_one_profile_fast, it)
                for it in items
                if (it.get("name") or "").strip()
            ]
            total = len(futs)
            _emit(
                job_id,
                {"type": "progress", "phase": "profile", "done": 0, "total": total},
            )

            enriched2: List[Dict[str, Any]] = []
            for fut in concurrent.futures.as_completed(futs):
                if _get_job(job_id).cancelled:
                    _emit(job_id, {"type": "cancelled", "where": "profile_people"})
                    break
                it2 = fut.result()
                enriched2.append(it2)
                prof_done += 1
                _emit(
                    job_id,
                    {
                        "type": "progress",
                        "phase": "profile",
                        "done": prof_done,
                        "total": total,
                        "name": it2.get("name"),
                    },
                )

        # Keep deterministic ordering by original score then fill by name match.
        by_name = {str(x.get("name")): x for x in enriched2}
        ordered: List[Dict[str, Any]] = []
        for it in items:
            nm = str((it.get("name") or "").strip())
            if nm and nm in by_name:
                ordered.append(by_name[nm])

        lb["items"] = ordered

        try:
            _emit(
                job_id,
                {
                    "type": "artifact",
                    "kind": "profile_done",
                    "topic": topic,
                    "count": len(ordered),
                    "names": [
                        str(x.get("name")) for x in ordered if isinstance(x, dict)
                    ][:50],
                },
            )
        except Exception:
            pass

        # Notes are deferred to detailing.
        notes = None

        topic_id = uuid.uuid4().hex
        out_obj = {
            "id": topic_id,
            "topic": (
                plan.get("topic_name")
                if isinstance(plan, dict) and plan.get("topic_name")
                else topic
            ),
            "created_at": time.time(),
            "sources": {
                "used_deepsearch": bool(deepsearch),
                "base_url": DEFAULT_BASE_URL,
                "model_gpt": DEFAULT_MODEL_GPT52,
                "model_deepsearch": DEFAULT_MODEL_DEEPSEARCH,
            },
            "input_text": topic,
            "definition": plan.get("definition") if isinstance(plan, dict) else None,
            "time_range": plan.get("time_range") if isinstance(plan, dict) else None,
            "plan": {
                "keywords": plan.get("keywords") if isinstance(plan, dict) else None,
                "must_terms": plan.get("must_terms")
                if isinstance(plan, dict)
                else None,
                "exclude_terms": plan.get("exclude_terms")
                if isinstance(plan, dict)
                else None,
                "subtopics": plan.get("subtopics") if isinstance(plan, dict) else None,
            },
            "keywords": lb.get("keywords"),
            "total_candidates": lb.get("total_candidates"),
            "items": lb.get("items"),
            "notes": notes,
        }
        _write_topic_result(topic_id, out_obj)

        # Start detailing after the main job is done (non-blocking).
        _set_topic_detailing_state(
            topic_id,
            {
                "status": "running",
                "done": 0,
                "total": len(lb.get("items") or []),
                "updated": 0,
                "failed": 0,
                "started_at": time.time(),
                "finished_at": None,
            },
        )
        _emit_topic(
            topic_id, {"type": "detail_stage", "stage": "started", "topic_id": topic_id}
        )

        try:
            _emit(
                job_id,
                {
                    "type": "artifact",
                    "kind": "result",
                    "topic": out_obj.get("topic"),
                    "topic_id": topic_id,
                },
            )
        except Exception:
            pass

        job.status = "done"
        job.stage = "done"
        job.result_topic_id = topic_id
        _set_job(job)

        _emit(
            job_id,
            {
                "type": "done",
                "topic_id": topic_id,
                "topic": topic,
                "detailing": True,
            },
        )

        # Background detailing thread.
        def _detail_thread() -> None:
            try:
                _run_topic_detailing(
                    topic_id=topic_id,
                    topic=topic,
                    plan=plan,
                    items=lb.get("items") or [],
                    api_key=api_key,
                    base_url=DEFAULT_BASE_URL,
                    model_deepsearch=DEFAULT_MODEL_DEEPSEARCH,
                    model_gpt=DEFAULT_MODEL_GPT52,
                    deepsearch_enabled=bool(deepsearch),
                )
            except Exception as e:
                _emit_topic(
                    topic_id,
                    {"type": "detail_error", "message": f"{type(e).__name__}: {e}"},
                )

        threading.Thread(
            target=_detail_thread, name=f"detail-{topic_id}", daemon=True
        ).start()
    except Exception as e:
        job.status = "error"
        job.stage = "error"
        job.error = f"{type(e).__name__}: {e}"
        _set_job(job)
        _emit(job_id, {"type": "error", "message": job.error})
    finally:
        with _limit_lock:
            _active_jobs = max(0, _active_jobs - 1)
        _emit(job_id, {"type": "eof"})


@app.post("/api/jobs/investigate")
def create_job(request: Request, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    global _active_jobs
    name = str(payload.get("name") or "").strip()
    api_key = str(payload.get("api_key") or "").strip()
    deepsearch = bool(payload.get("deepsearch") or False)

    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    # Basic guardrails for public service.
    if len(api_key) > 200:
        raise HTTPException(status_code=400, detail="api_key too long")
    if len(name) > 80:
        raise HTTPException(status_code=400, detail="name too long")

    client_ip = request.client.host if request.client else "unknown"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        client_ip = fwd.split(",")[0].strip() or client_ip

    with _limit_lock:
        if _active_jobs >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="too many active jobs")
        last = _ip_last_submit.get(client_ip)
        now = time.time()
        if last and (now - last) < IP_COOLDOWN_SEC:
            raise HTTPException(status_code=429, detail="too many requests")
        _ip_last_submit[client_ip] = now

    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        created_at=time.time(),
        status="queued",
        stage="queued",
        kind="person",
        query=name,
        deepsearch=deepsearch,
    )
    _set_job(job)
    _job_events[job_id] = queue.Queue(maxsize=1000)
    _emit(job_id, {"type": "queued", "name": name})

    with _limit_lock:
        _active_jobs += 1

    t = threading.Thread(
        target=_run_job,
        name=f"job-{job_id}",
        kwargs={
            "job_id": job_id,
            "name": name,
            "api_key": api_key,
            "deepsearch": deepsearch,
        },
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    return JSONResponse(_job_to_public(job))


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    _get_job(job_id)  # validate
    q = _job_events.get(job_id)
    if not q:
        raise HTTPException(status_code=404, detail="events not found")

    def gen() -> Generator[bytes, None, None]:
        # Send a hello event.
        yield b"event: hello\ndata: {}\n\n"

        while True:
            try:
                ev = q.get(timeout=25)
            except Exception:
                # Keep-alive.
                yield b": keep-alive\n\n"
                continue

            typ = ev.get("type")
            data = json.dumps(ev, ensure_ascii=False)
            yield f"event: {typ}\ndata: {data}\n\n".encode("utf-8")

            if typ == "eof":
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    job.cancelled = True
    job.status = "cancelled"
    _set_job(job)
    _emit(job_id, {"type": "cancelled", "job_id": job_id})
    return JSONResponse({"job_id": job_id, "status": job.status, "cancelled": True})


@app.post("/api/jobs/{job_id}/fast_profile")
def fast_profile(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    job.fast_profile = True
    # Keep status running; pipeline will fall through to profile using completed deepresearch items.
    if job.status not in ("done", "error"):
        job.status = "running"
    _set_job(job)
    _emit(job_id, {"type": "fast_profile", "job_id": job_id})
    return JSONResponse({"job_id": job_id, "status": job.status, "fast_profile": True})


@app.post("/api/jobs/topic")
def create_topic_job(
    request: Request, payload: Dict[str, Any] = Body(...)
) -> JSONResponse:
    global _active_jobs
    topic = str(payload.get("topic") or "").strip()
    api_key = str(payload.get("api_key") or "").strip()
    deepsearch = bool(payload.get("deepsearch") or False)

    if not topic:
        raise HTTPException(status_code=400, detail="topic is required")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")
    if len(api_key) > 200:
        raise HTTPException(status_code=400, detail="api_key too long")
    if len(topic) > 120:
        raise HTTPException(status_code=400, detail="topic too long")

    client_ip = request.client.host if request.client else "unknown"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        client_ip = fwd.split(",")[0].strip() or client_ip

    with _limit_lock:
        if _active_jobs >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="too many active jobs")
        last = _ip_last_submit.get(client_ip)
        now = time.time()
        if last and (now - last) < IP_COOLDOWN_SEC:
            raise HTTPException(status_code=429, detail="too many requests")
        _ip_last_submit[client_ip] = now

    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        created_at=time.time(),
        status="queued",
        stage="queued",
        kind="topic",
        query=topic,
        deepsearch=deepsearch,
    )
    _set_job(job)
    _job_events[job_id] = queue.Queue(maxsize=1000)
    _emit(job_id, {"type": "queued", "topic": topic})

    with _limit_lock:
        _active_jobs += 1

    t = threading.Thread(
        target=_run_topic_job,
        name=f"job-{job_id}",
        kwargs={
            "job_id": job_id,
            "topic": topic,
            "api_key": api_key,
            "deepsearch": deepsearch,
        },
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/jobs/{job_id}/continue")
def continue_job(
    job_id: str, request: Request, payload: Dict[str, Any] = Body(...)
) -> JSONResponse:
    global _active_jobs
    job = _get_job(job_id)
    if job.status != "waiting_user":
        raise HTTPException(status_code=400, detail="job not waiting for input")

    add_text = str(payload.get("text") or "").strip()
    api_key = str(payload.get("api_key") or "").strip()
    if not add_text:
        raise HTTPException(status_code=400, detail="text is required")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")
    if len(api_key) > 200:
        raise HTTPException(status_code=400, detail="api_key too long")

    # Continuing an existing job should not be blocked by IP_COOLDOWN_SEC.
    # Still respect global concurrency.
    with _limit_lock:
        if _active_jobs >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="too many active jobs")
        _active_jobs += 1

    original = job.query
    combined = (str(original).strip() + "\n\n" + add_text).strip()

    # If this was a definition confirmation, persist the final definition into plan.
    try:
        if isinstance(job.meta, dict) and job.meta.get("asked_definition"):
            plan = job.meta.get("plan")
            if isinstance(plan, dict):
                plan["definition"] = add_text
                plan["definition_source"] = "user_confirmed"
                job.meta["plan"] = plan
    except Exception:
        pass
    job.status = "queued"
    job.stage = "queued"
    job.query = combined
    # Note: asked_definition already set; second time we will proceed even if definition still missing.
    _set_job(job)
    _emit(job_id, {"type": "queued", "topic": job.query})

    t = threading.Thread(
        target=_run_topic_job,
        name=f"job-{job_id}",
        kwargs={
            "job_id": job_id,
            "topic": combined,
            "api_key": api_key,
            "deepsearch": job.deepsearch,
        },
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/topics")
def api_topics() -> JSONResponse:
    return JSONResponse({"items": _list_topic_results()})


@app.get("/api/topics/{topic_id}")
def api_topic(topic_id: str) -> JSONResponse:
    p = TOPICS_DIR / f"{topic_id}.topic.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="topic not found")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="invalid topic json")
    st = _get_topic_detailing_state(topic_id)
    if st:
        obj = {**obj, "detailing": st}
    return JSONResponse(obj)


@app.get("/api/topics/{topic_id}/events")
def topic_events(topic_id: str) -> StreamingResponse:
    p = TOPICS_DIR / f"{topic_id}.topic.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="topic not found")

    with _topic_events_lock:
        q = _topic_events.get(topic_id)
        if not q:
            q = queue.Queue(maxsize=2000)
            _topic_events[topic_id] = q

    def gen() -> Generator[bytes, None, None]:
        yield b"event: hello\ndata: {}\n\n"
        while True:
            try:
                ev = q.get(timeout=25)
            except Exception:
                yield b": keep-alive\n\n"
                continue

            typ = ev.get("type") or "event"
            data = json.dumps(ev, ensure_ascii=False)
            yield f"event: {typ}\ndata: {data}\n\n".encode("utf-8")
            if typ == "eof":
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


def _job_to_public(job: Job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "created_at": job.created_at,
        "status": job.status,
        "stage": job.stage,
        "kind": job.kind,
        "query": job.query,
        "deepsearch": job.deepsearch,
        "result_name": job.result_name,
        "result_topic_id": job.result_topic_id,
        "meta": job.meta,
        "error": job.error,
        "cancelled": bool(getattr(job, "cancelled", False)),
        "fast_profile": bool(getattr(job, "fast_profile", False)),
    }


# 静态资源统一走 /static，避免影响 /api 和相对路径解析
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
