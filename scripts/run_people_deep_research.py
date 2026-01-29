from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

from core.deep_research_pool import DeepResearchPool, DeepResearchTask, StreamSink


# 你的目录结构：World_Model/Crawler
ROOT_DIR = Path(__file__).resolve().parent.parent
WORLD_MODEL_DIR = ROOT_DIR
PEOPLE_MD_DIR = WORLD_MODEL_DIR / "people_md"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

URL = "http://152.53.52.170:3003/v1/chat/completions"
MODEL = "gemini-2.5-pro-deepsearch-async"

MAX_TASKS = 50
MAX_WORKERS = 5
OVERWRITE = False

MARKER_TITLE = "## DeepResearch 补全（自动生成，需人工审计）"

PRINT_LOCK = threading.Lock()


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def make_prompt(person_name: str, current_md: str) -> str:
    return f"""
你是研究助理，请对人物做一次“可审计的深度检索/整理（deep research）”，目标是把下述 Markdown 档案补全。

人物：{person_name}

要求（非常重要）：
1) 你必须尽量给出可点击的证据链接（homepage / scholar / dblp / openreview / arxiv / github / institution page 等）。
2) 任何不确定的信息必须标注“不确定”，不要编造。
3) 输出请用中文，结构化，直接给出“可粘贴进该人物 md 文件”的补全内容。
4) 优先补齐与【世界模型 / 具身智能 / 空间智能 / agent / video world model】相关的信息；但也可补充其高影响工作用于定位影响力。
5) 请至少产出：
   - 基本信息（单位/职务/方向/主页/联系方式如公开）
   - 身份ID（Scholar/S2/OpenAlex/ORCID/DBLP/GitHub）
   - 代表性成果（3-8条，给链接与年份）
   - 世界模型相关论文清单（如没有，明确写“未找到直接相关论文（不确定）”，并给最接近方向的论文）
   - 影响力数据（引用、h-index等：若无法确认请标注“不确定”，并给可查链接）
   - 证据链接列表（逐条）
6) 输出末尾附一个“你本次检索覆盖的来源”清单（如：Google Scholar、DBLP、OpenReview、机构主页等）。

这是该人物当前 md 内容（供你对齐字段/避免重复）：
```md
{current_md}
```

现在开始输出【补全区块】（直接可粘贴到 md 末尾），以标题开头：
{MARKER_TITLE}
""".strip()


class FileStreamSink:
    def __init__(self, tmp_path: Path, final_path: Path, header: str, task_id: str):
        self.tmp_path = tmp_path
        self.final_path = final_path
        self.task_id = task_id
        self._fp = open(self.tmp_path, "w", encoding="utf-8")
        self._fp.write(header)
        self._fp.flush()
        self._count = 0
        self._last_print = time.time()

    def write(self, text: str) -> None:
        self._fp.write(text)
        self._fp.flush()
        self._count += len(text)

        now = time.time()
        if now - self._last_print >= 1.0:
            with PRINT_LOCK:
                print(f"[{self.task_id}] +{self._count} chars")
            self._last_print = now

    def close(self, ok: bool) -> None:
        try:
            self._fp.flush()
        finally:
            self._fp.close()

        if ok:
            if self.final_path.exists():
                self.final_path.unlink()
            self.tmp_path.rename(self.final_path)
        else:
            # 未完成：保留 .partial，下次从零开始（不续写）
            pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(
        [p for p in PEOPLE_MD_DIR.glob("*.md") if p.is_file() and p.name != "大名单.md"]
    )

    target_files = md_files[:MAX_TASKS]
    if not target_files:
        print("No md files found.")
        return

    tasks: List[DeepResearchTask] = []

    for i, p in enumerate(target_files, 1):
        person_name = p.stem
        output_path = OUTPUT_DIR / f"{person_name}.md"
        tmp_path = OUTPUT_DIR / f"{person_name}.md.partial"

        if output_path.exists() and not OVERWRITE:
            print(f"[skip] exists: {output_path.name}")
            continue

        content = read_text(p)
        prompt = make_prompt(person_name, content)

        header = content.rstrip() + "\n\n" + MARKER_TITLE + "\n"

        def make_sink(
            task_id: str, tmp_path=tmp_path, output_path=output_path, header=header
        ):
            return FileStreamSink(tmp_path, output_path, header, task_id)

        tasks.append(
            DeepResearchTask(
                task_id=f"person_{i:03d}_{person_name}",
                prompt=prompt,
                model_id=MODEL,
                url=URL,
                temperature=0.2,
                stream_sink_factory=lambda task_id=f"person_{i:03d}_{person_name}": make_sink(
                    task_id
                ),
            )
        )

    if not tasks:
        print("No tasks to run.")
        return

    pool = DeepResearchPool(
        max_workers=MAX_WORKERS,
        cache_dir=CRAWLER_DIR / "cache",
        cache_ttl_sec=30 * 24 * 3600,
    )

    def progress(done, total, task):
        with PRINT_LOCK:
            print(f"[done {done}/{total}] {task.task_id}")

    results = pool.run_batch(tasks, progress=progress)

    for res in results:
        if not res.ok:
            print(f"[failed] {res.task_id}: {res.error}")

    print("All done.")


if __name__ == "__main__":
    main()
