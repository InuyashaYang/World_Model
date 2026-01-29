from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import concurrent.futures

from core.llm_client import chat


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = DATA_DIR / "profiles"

URL = "http://152.53.52.170:3003/v1/chat/completions"
MODEL_ID = "gpt-5.2-2025-12-11"

MAX_WORKERS = 20
TEMPERATURE = 0.0
TIMEOUT = 180

SYSTEM = "You are an information extraction system. Output JSON only."

PROFILE_PROMPT = r"""
你是一名“研究者资料抽取（information extraction）”系统。下面给你一段关于某位研究者的公开材料（可能混杂计划/思考/引用/链接/不确定项）。
你的任务是：严格基于材料抽取信息，输出为一个结构化 JSON。禁止编造；没有则填 null/空数组。

你必须【只输出一个 JSON 对象】（不要 Markdown、不要代码块、不要额外文字），字段如下（必须全部出现，值可为 null/[]）：

{
  "name": string,
  "aliases": [string],

  "affiliation_current": string|null,
  "position_title": string|null,
  "location": string|null,
  "email_public": string|null,

  "homepage": string|null,
  "google_scholar": string|null,
  "semantic_scholar": string|null,
  "openalex": string|null,
  "orcid": string|null,
  "dblp": string|null,
  "github": string|null,

  "research_keywords": [string],

  "representative_works": [
    {
      "type": "paper"|"project"|"system"|"dataset"|"benchmark"|"other",
      "title": string,
      "year": number|null,
      "venue": string|null,
      "authors": string|null,
      "role": string|null,
      "links": [string],
      "notes": string|null
    }
  ],

  "world_model_relevance": {
    "is_core": "yes"|"no"|"uncertain",
    "subareas": [string],
    "reason": string
  },

  "metrics": {
    "citations": number|null,
    "h_index": number|null,
    "citations_last_3y": number|null,
    "citations_last_5y": number|null
  },

  "community_roles": [
    {
      "role": string,
      "venue_or_org": string|null,
      "year": number|null,
      "source": string|null
    }
  ],

  "evidence_links": [
    {
      "label": string,
      "url": string
    }
  ],

  "missing_or_uncertain": [string]
}

抽取规则：
- URL：尽量输出完整链接（http/https）。
- 指标（citations/h_index 等）：只有材料明确给出数值才填，否则 null。
- 不确定信息：写入 missing_or_uncertain，并在对应字段保持 null 或保守填写（例如 world_model_relevance.is_core 可为 "uncertain"）。
- evidence_links 最多 20 条，优先：主页/Scholar/DBLP/OpenReview/arXiv/GitHub/机构页面。

输入材料如下：
<<<MATERIALS>>>
""".strip()


@dataclass(frozen=True)
class ProfileTask:
    cache_path: Path
    task_id: str
    name: str
    materials: str


def _safe_load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_name(task_id: str) -> str:
    parts = task_id.split("_", 2)
    return parts[-1] if len(parts) >= 3 else task_id


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_model_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l : r + 1])
        raise ValueError("Model output is not valid JSON.")


def _write_result(out_path: Path, obj: Dict[str, Any]) -> None:
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _profile_one(t: ProfileTask) -> Tuple[bool, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h = _hash_text(t.materials)[:12]
    out_path = OUT_DIR / f"{t.name}.{h}.profile.json"
    if out_path.exists():
        return True, f"skip exists: {out_path.name}"

    prompt = PROFILE_PROMPT + "\n" + t.materials + "\n<<<END>>>"

    resp = chat(
        prompt,
        model_id=MODEL_ID,
        url=URL,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        system=SYSTEM,
        extra={"response_format": {"type": "json_object"}},
    )

    obj = _parse_model_json(resp)

    # 基本校验
    if "name" not in obj or "metrics" not in obj or "world_model_relevance" not in obj:
        raise ValueError("Missing required fields in profile JSON output.")

    _write_result(out_path, obj)
    return True, f"written: {out_path.name}"


def build_tasks(limit: int = 50) -> List[ProfileTask]:
    tasks: List[ProfileTask] = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        obj = _safe_load_json(p)
        if not obj or not obj.get("ok"):
            continue
        task_id = obj.get("task_id") or p.stem
        name = _extract_name(task_id)
        materials = obj.get("text") or ""
        if not materials.strip():
            continue

        tasks.append(
            ProfileTask(cache_path=p, task_id=task_id, name=name, materials=materials)
        )
        if len(tasks) >= limit:
            break
    return tasks


def run(limit: int = 50):
    tasks = build_tasks(limit=limit)
    total = len(tasks)
    if total == 0:
        print("No cache JSON to parse.")
        return

    print(f"Parsing profiles for {total} researchers with {MAX_WORKERS} workers...")

    done = 0
    started = time.time()

    def _runner(t: ProfileTask):
        return t, _profile_one(t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_runner, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            t, (ok, msg) = fut.result()
            done += 1
            cost = time.time() - started
            print(f"[{done}/{total}] {t.task_id} -> {msg}  (elapsed={cost:.1f}s)")

    print("All done.")


if __name__ == "__main__":
    run(limit=50)
