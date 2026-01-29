# 架构说明（最新版）

本仓库围绕“研究者/主题检索与评估”提供两条主路径：
1) 一个可直接打开使用的 Web 工作台（静态页面 + FastAPI 后端 + SSE 进度流）。
2) 一套离线脚本流水线（从 `data/cache/` 的材料出发，批量产出 `data/people/*.person.json`）。

---

## 1) 目录与职责

- `app/main.py`
  - FastAPI 单体服务：同时提供 UI 静态资源与 `/api/*` 接口
  - 任务系统（Job）：后台线程执行 person/topic 流水线；通过 SSE 推送进度/中间产出

- `app/static/`
  - `search.html`：工作台（Topic/Person 发起任务、展示 stage/progress、展示 artifacts、管理本地 Session）
  - `index.html`：研究者榜单（读取 `/api/people`）
  - `detail.html`：研究者详情（读取 `/api/people/{name}`）
  - `theme.css`：共享样式

- `core/llm_client.py`
  - OpenAI-compatible `chat.completions` 的最小客户端
  - 支持非流式 `chat()` 与流式 `stream_chat()`
  - 默认读取 `data/config.yaml`（仅支持最简 `key: value` 顶层键）

- `core/deep_research_pool.py`
  - 离线 deepresearch 的并发/缓存执行池（按 prompt+参数哈希缓存到 `cache_dir/*.json`）
  - 主要给离线脚本用；Web 端当前在 `app/main.py` 里直接用线程池并发

- `scripts/`
  - `run_people_deep_research.py`：对 `people_md/*.md` 批量 deepresearch，输出到 `data/output/*.md`
  - `score_people.py`：从 `data/cache/*.json` 读取材料，调用 LLM 打分，输出 `data/scored/*.score.json`
  - `extract_profiles.py`：从 `data/cache/*.json` 抽取结构化 profile，输出 `data/profiles/*.profile.json`
  - `merge_people.py`：合并最新的 score/profile，输出 `data/people/*.person.json`
  - `build_candidates_by_topic.py`：基于 OpenAlex 关键词粗筛候选作者（输出 `data/candidates/*.jsonl`）

- `data/`（运行产物与配置）
  - `data/web/people/*.person.json`：Web 读取用的人物聚合结果（通常与 `data/people/` 同步）
  - `data/web/topics/*.topic.json`：主题榜单结果（topic pipeline 输出）
  - `data/people/*.person.json`：人物聚合结果（profile + score + sources）
  - `data/cache/`：离线 deepresearch 缓存（供脚本流水线）
  - `data/scored/`、`data/profiles/`：离线中间产物
  - `data/model_list.txt`：UI 可选模型列表（单行逗号分隔）
  - `data/config.yaml`：离线/通用 LLM client 配置（不要提交真实 key）

- `pipeline.ps1`
  - Windows PowerShell 一键跑离线流水线（默认：score -> profile -> merge）

---

## 2) Web 服务：路由与运行模型

### 2.1 静态资源

- `/`：重定向到 `/static/search.html`
- `/static/*`：挂载 `app/static/`

### 2.2 API（读取）

- `GET /api/models`
  - 返回默认网关/默认模型 + `data/model_list.txt` 中的模型列表

- `GET /api/people?q=<keyword>&sort=S_prime_desc|S_desc|name_asc`
  - 从 `data/people/*.person.json`（优先）或 `data/web/people/*.person.json` 读取并检索

- `GET /api/people/{name}`
  - 读取单个人物的聚合 JSON

- `GET /api/topics`
  - 列出历史 topic 结果摘要（按文件 mtime 倒序）

- `GET /api/topics/{topic_id}`
  - 读取某次 topic 结果全文

### 2.3 Job（异步任务）

Job 是一次实际执行单元（person/topic），由服务端创建、后台线程执行、并通过 SSE 输出事件流。

- `POST /api/jobs/investigate`
  - body: `{ "name": "...", "api_key": "...", "deepsearch": false }`

- `POST /api/jobs/topic`
  - body: `{ "topic": "...", "api_key": "...", "deepsearch": false }`

- `GET /api/jobs/{job_id}`
  - 返回 job 的轻量状态（stage/status/result 指针等）

- `GET /api/jobs/{job_id}/events`（SSE）
  - 持续推送事件直到 `eof`

- `POST /api/jobs/{job_id}/continue`
  - topic plan 发现“缺少定义/范围”时进入 `waiting_user`，前端提交补充文本继续

- `POST /api/jobs/{job_id}/cancel`
  - best-effort 取消：置 `job.cancelled=True`，流水线在可控点提前收敛

- `POST /api/jobs/{job_id}/fast_profile`
  - topic 的“快速建档”：在 `deepresearch_people` 并发阶段提前收敛，只对已完成 deepresearch 的候选进入 `profile_people`

---

## 3) SSE 事件协议（前后端约定）

服务端通过 `/api/jobs/{job_id}/events` 推送：

- `hello`：连接建立
- `queued`：任务已创建/入队
- `stage`：阶段切换（如 `plan/openalex/deepresearch_people/profile_people/deepsearch/done`）
- `progress`：阶段进度（`{phase, done, total, name?}`）
- `log`：文本日志（可能高频；UI 通常折叠展示）
- `artifact`：结构化中间产出（计划/候选列表预览/openalex 材料预览/结果指针等）
- `need_input`：需要用户补充（含 `questions[]` 与 `plan_preview`）
- `cancelled`：取消信号
- `fast_profile`：快速建档信号
- `done`：任务完成（携带 `name` 或 `topic_id`）
- `error`：失败（携带 message）
- `eof`：事件流结束

前端建议处理策略（避免“刷屏”）：
- `stage`/`progress`：只更新状态卡
- `artifact`：写入可回看的 artifacts 时间线
- `log`：写入折叠日志区
- `done`：跳转或展示结果入口

---

## 4) 服务端流水线（Web Job）

### 4.1 Person：人物调查

典型 stage：
- `openalex`：无 key 拉取 OpenAlex 作者与部分作品作为基础材料
- （可选）`deepsearch`：调用 deepsearch 模型流式补充证据/链接
- `gpt52`：调用 GPT 模型抽取 profile + score（JSON only）

输出：
- `data/people/{name}.person.json`
- `data/web/people/{name}.person.json`

### 4.2 Topic：主题榜单

典型 stage：
- `plan`：用 GPT 将用户输入解析为检索计划（可触发 `need_input`）
- `openalex`：按关键词搜索 works、聚合作者、计算粗分，得到 Top 候选
- `deepresearch_people`：并发对候选做 deepresearch addendum（可中途 `fast_profile` 收敛）
- `profile_people`：并发对已 enriched 的候选生成 profile + score，并写入人物库
- （可选）`deepsearch`：对最终榜单做审计友好总结/补证，写入 `notes`

输出：
- `data/web/topics/{topic_id}.topic.json`

---

## 5) 数据产物（JSON 形状要点）

### 5.1 Person（`data/people/*.person.json` / `data/web/people/*.person.json`）

- `name`: string
- `profile`: object（结构化抽取结果）
- `score`: object（S/S_prime/维度分/evidence 等）
- `sources`: object（created_at、是否使用 deepsearch、模型与网关等元信息）

### 5.2 Topic（`data/web/topics/*.topic.json`）

- `id`, `topic`, `created_at`, `input_text`
- `definition`, `time_range`, `plan{keywords,must_terms,exclude_terms,subtopics}`
- `keywords`, `total_candidates`, `items[]`（候选列表，包含 evidence_links、sample_works 等）
- `notes`（可选：deepsearch 审计总结）

---

## 6) 离线脚本流水线（batch）

典型链路（`pipeline.ps1` 默认）：

1) `scripts/score_people.py`
   - 输入：`data/cache/*.json`（DeepResearchPool 输出，含 `ok/text/task_id`）
   - 输出：`data/scored/{name}.{hash}.score.json`

2) `scripts/extract_profiles.py`
   - 输入：`data/cache/*.json`
   - 输出：`data/profiles/{name}.{hash}.profile.json`

3) `scripts/merge_people.py`
   - 输入：`data/scored/` 与 `data/profiles/`（对同名取 mtime 最新版本）
   - 输出：`data/people/{name}.person.json`

补充：
- `scripts/run_people_deep_research.py` 当前脚本里存在未定义变量（`CRAWLER_DIR`），且在 `pipeline.ps1` 默认注释；如需启用请先修复。

---

## 7) 配置与运行约束

### 7.1 Web 端（`app/main.py`）环境变量

- `DEFAULT_BASE_URL`（默认：`http://152.53.52.170:3003/v1`）
- `DEFAULT_MODEL_GPT52`（默认：`gpt-5.2-2025-12-11`）
- `DEFAULT_MODEL_DEEPSEARCH`（默认：`gemini-2.5-pro-deepsearch-async`）
- `DEFAULT_RECENT_YEARS`（默认：`2`）
- `DEEPSEARCH_WORKERS`（默认：`5`）
- `PROFILE_WORKERS`（默认：`10`）
- `MAX_CONCURRENT_JOBS`（默认：`2`）
- `IP_COOLDOWN_SEC`（默认：`10`）

### 7.2 安全/护栏（Web 端）

- `api_key` 仅随请求传入，后端只在 job 线程栈内使用，不落盘。
- 不允许客户端覆盖 `base_url`（降低 SSRF 风险）：Web 端固定使用服务端 `DEFAULT_BASE_URL`。
- 基础限流：全局并发 `MAX_CONCURRENT_JOBS` + 同 IP 冷却 `IP_COOLDOWN_SEC` + 参数长度限制。

---

## 8) Legacy

- `crawler_legacy/`：历史目录结构保留区（当前 Web/脚本不依赖）。
