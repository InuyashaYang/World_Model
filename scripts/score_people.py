from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import concurrent.futures

from core.llm_client import chat


# ---------- 可配置 ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = DATA_DIR / "scored"

URL = "http://152.53.52.170:3003/v1/chat/completions"
MODEL_ID = "gpt-5.2-2025-12-11"

MAX_WORKERS = 20
TEMPERATURE = 0.0
TIMEOUT = 180  # 秒
# ---------------------------


SYSTEM = "You are a rigorous academic evaluator. Output JSON only."

RUBRIC_PROMPT = r"""
你是一名“世界模型（World Models）/具身智能/空间智能”方向的学术影响力评估分析师。
我会给你某位研究者的公开资料（可能包含：个人主页、Scholar/Semantic Scholar/OpenAlex 信息、代表论文、引用数据、奖项、开源仓库等）。请你严格基于我提供的材料打分，不要臆测；对缺失信息要显式标注并降权处理。

## 目标
输出该研究者的影响力评分（0–100），并给出可解释的多维度分解、证据、以及“数据完备度”提示。评分应尽量可复算：每个分数要能追溯到具体指标或证据。

---

## 评分框架（固定使用）
总分 S∈[0,100]：
S = 0.35*I_impact + 0.25*I_field + 0.20*I_quality + 0.10*I_community + 0.10*I_momentum

并给出“完备度惩罚后分数”：
S' = S * (0.7 + 0.3*completeness)
completeness ∈ [0,1]

---

## 输出格式（必须严格遵守）
你必须【只输出一个 JSON 对象】（不要 Markdown、不要代码块、不要额外文本），并包含以下字段：

{
  "name": string,
  "affiliation": string|null,
  "primary_directions": string|null,

  "S": number,
  "S_prime": number,
  "completeness": number,

  "dimensions": {
    "impact": number,
    "field": number,
    "quality": number,
    "community": number,
    "momentum": number
  },

  "dimension_explanations": {
    "impact": string,
    "field": string,
    "quality": string,
    "community": string,
    "momentum": string
  },

  "evidence": [
    {
      "type": string,
      "fact": string,
      "source": string
    }
  ],

  "missing_or_uncertain": [string],

  "one_line_conclusion": string
}

约束：
- evidence 最多 12 条
- 严格基于输入材料；材料没给的不要当事实
- 允许 affiliation/primary_directions 为 null
- 所有分数保留 0-100；completeness 0-1
- 请在 dimension_explanations 中简述为何给该分数（1-3句）

---

## 输入材料
<<<PASTE_RESEARCH_MATERIALS_HERE>>>
<<<END>>>

现在开始评分。请只输出 JSON。
""".strip()


@dataclass(frozen=True)
class ScoreTask:
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
    # person_004_俞扬 -> 俞扬
    parts = task_id.split("_", 2)
    return parts[-1] if len(parts) >= 3 else task_id


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _build_prompt(materials: str) -> str:
    return RUBRIC_PROMPT.replace("<<<PASTE_RESEARCH_MATERIALS_HERE>>>", materials)


def _parse_model_json(text: str) -> Dict[str, Any]:
    """
    尽量把模型输出解析为 JSON：
    - 如果它不小心输出了前后垃圾文本，尝试截取第一个 { 到最后一个 }。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        return json.loads(text[l : r + 1])

    raise ValueError("Model output is not valid JSON.")


def _write_result(out_path: Path, obj: Dict[str, Any]) -> None:
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _score_one(t: ScoreTask) -> Tuple[bool, str]:
    """
    返回 (ok, message). 结果写入 OUT_DIR。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 用 materials hash 做版本号，避免同名覆盖但材料已更新
    h = _hash_text(t.materials)[:12]
    out_path = OUT_DIR / f"{t.name}.{h}.score.json"

    if out_path.exists():
        return True, f"skip exists: {out_path.name}"

    prompt = _build_prompt(t.materials)

    resp = chat(
        prompt,
        model_id=MODEL_ID,
        url=URL,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        system=SYSTEM,
        extra={
            # 如果你的网关支持 response_format(json_object)，会更稳；不支持也没关系
            "response_format": {"type": "json_object"}
        },
    )

    obj = _parse_model_json(resp)

    # 基本字段校验（保证 json 友好且结构可解析）
    if "dimensions" not in obj or "S" not in obj or "S_prime" not in obj:
        raise ValueError("Missing required fields in JSON output.")
    _write_result(out_path, obj)
    return True, f"written: {out_path.name}"


def build_tasks(limit: int = 50) -> List[ScoreTask]:
    tasks: List[ScoreTask] = []

    for p in sorted(CACHE_DIR.glob("*.json")):
        obj = _safe_load_json(p)
        if not obj:
            continue
        if not obj.get("ok"):
            continue
        task_id = obj.get("task_id") or p.stem
        name = _extract_name(task_id)
        materials = obj.get("text") or ""
        if not materials.strip():
            continue

        tasks.append(
            ScoreTask(
                cache_path=p,
                task_id=task_id,
                name=name,
                materials=materials,
            )
        )
        if len(tasks) >= limit:
            break

    return tasks


def run(limit: int = 50):
    tasks = build_tasks(limit=limit)
    total = len(tasks)
    if total == 0:
        print("No cache JSON to score.")
        return

    print(f"Scoring {total} researchers with {MAX_WORKERS} workers...")

    done = 0
    started = time.time()

    def _runner(t: ScoreTask):
        return t, _score_one(t)

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
