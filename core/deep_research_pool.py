from __future__ import annotations

import json, time, hashlib, threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Protocol

import concurrent.futures

from core.llm_client import stream_chat


class StreamSink(Protocol):
    def write(self, text: str) -> None: ...
    def close(self, ok: bool) -> None: ...


@dataclass(frozen=True)
class DeepResearchTask:
    task_id: str
    prompt: str
    model_id: Optional[str] = None
    url: Optional[str] = None
    temperature: float = 0.2
    system: str = "You are a helpful assistant."
    extra: Optional[Dict[str, Any]] = None
    stream_sink_factory: Optional[Callable[[], StreamSink]] = None


@dataclass
class DeepResearchResult:
    task_id: str
    ok: bool
    model_id: Optional[str]
    created_at: float
    elapsed_sec: float
    from_cache: bool
    text: str = ""
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class DeepResearchPool:
    def __init__(
        self,
        *,
        max_workers: int = 5,
        cache_dir: str | Path = "cache",
        cache_ttl_sec: Optional[int] = 30 * 24 * 3600,  # 30天
        dedup_inflight: bool = True,
    ):
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_sec = cache_ttl_sec
        self.dedup_inflight = dedup_inflight

        self._lock = threading.Lock()
        self._inflight: Dict[str, concurrent.futures.Future] = {}

    def _stable_json(self, obj: Any) -> str:
        return json.dumps(
            obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def _cache_key(self, task: DeepResearchTask) -> str:
        raw = "|".join(
            [
                str(task.model_id or ""),
                str(task.url or ""),
                task.system or "",
                str(task.temperature),
                self._stable_json(task.extra or {}),
                task.prompt,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[DeepResearchResult]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if self.cache_ttl_sec is not None:
                created_at = float(obj.get("created_at", 0))
                if created_at and (time.time() - created_at > self.cache_ttl_sec):
                    return None
            return DeepResearchResult(**obj)
        except Exception:
            return None

    def _cache_set(self, key: str, result: DeepResearchResult) -> None:
        self._cache_path(key).write_text(
            self._stable_json(asdict(result)), encoding="utf-8"
        )

    def _run_one(self, task: DeepResearchTask) -> DeepResearchResult:
        started = time.time()
        key = self._cache_key(task)

        cached = self._cache_get(key)
        if cached is not None:
            data = asdict(cached)
            data["from_cache"] = True
            data["elapsed_sec"] = 0.0
            return DeepResearchResult(**data)

        sink = task.stream_sink_factory() if task.stream_sink_factory else None

        try:
            chunks = []
            for c in stream_chat(
                prompt=task.prompt,
                model_id=task.model_id,
                url=task.url,
                temperature=task.temperature,
                system=task.system,
                extra=task.extra,
            ):
                if sink:
                    sink.write(c)
                chunks.append(c)

            text = "".join(chunks)
            res = DeepResearchResult(
                task_id=task.task_id,
                ok=True,
                model_id=task.model_id,
                created_at=time.time(),
                elapsed_sec=time.time() - started,
                from_cache=False,
                text=text,
                error=None,
                meta={"cache_key": key},
            )
            self._cache_set(key, res)

            if sink:
                sink.close(ok=True)

            return res
        except Exception as e:
            if sink:
                sink.close(ok=False)
            return DeepResearchResult(
                task_id=task.task_id,
                ok=False,
                model_id=task.model_id,
                created_at=time.time(),
                elapsed_sec=time.time() - started,
                from_cache=False,
                text="",
                error=f"{type(e).__name__}: {e}",
                meta={"cache_key": key},
            )

    def submit(
        self, task: DeepResearchTask, executor: concurrent.futures.Executor
    ) -> concurrent.futures.Future:
        key = self._cache_key(task)

        if self.dedup_inflight:
            with self._lock:
                fut = self._inflight.get(key)
                if fut and not fut.done():
                    return fut

                fut = executor.submit(self._run_one, task)
                self._inflight[key] = fut

                def _cleanup(_f):
                    with self._lock:
                        if self._inflight.get(key) is _f:
                            self._inflight.pop(key, None)

                fut.add_done_callback(_cleanup)
                return fut

        return executor.submit(self._run_one, task)

    def run_batch(
        self,
        tasks: List[DeepResearchTask],
        *,
        progress: Optional[Callable[[int, int, DeepResearchTask], None]] = None,
    ) -> List[DeepResearchResult]:
        total = len(tasks)
        results: List[DeepResearchResult] = [None] * total  # type: ignore

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_idx: Dict[concurrent.futures.Future, int] = {}

            for i, t in enumerate(tasks):
                fut = self.submit(t, ex)
                future_to_idx[fut] = i

            done_cnt = 0
            for fut in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[fut]
                task = tasks[idx]
                res = fut.result()
                results[idx] = res
                done_cnt += 1
                if progress:
                    progress(done_cnt, total, task)

        return results
