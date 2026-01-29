# World_Model (World Model Crawler)

面向“世界模型 / 具身智能 / 机器人学习”等方向的研究者与主题检索工具。

它同时提供：
- 一个可直接打开使用的 Web UI（静态页面 + FastAPI 后端）
- 一套离线脚本流水线（对本地材料进行打分/抽取/合并，产出可供网页与 API 读取的 `*.person.json`）

相关补充文档：
- `ARCHITECTURE.md`：目录分层与产物位置
- `README_DEPLOY.md`：Render 单服务部署与安全注意事项

---

## 1. 快速开始（本地运行 Web UI）

环境要求：Python 3.10+（建议），联网可访问 OpenAlex，以及一个 OpenAI-compatible 网关的 `api_key`。

安装依赖并启动：

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开：
- http://127.0.0.1:8000/

使用方式（UI）：
- 在页面中粘贴你的 `api_key`（只用于本次任务，不会写入服务端文件）
- 选择：
  - `Topic`：输入一个研究主题，得到 Top50 研究者 + 证据链接
  - `Person`：输入研究者姓名，抓取 OpenAlex 材料并生成结构化 profile + score
- 任务进度通过 SSE 流式展示（阶段 stage/log/progress/done）

生成的结果会写入：
- 人物：`data/people/*.person.json` 和 `data/web/people/*.person.json`
- 主题：`data/web/topics/*.topic.json`

---

## 2. Web API（给 AI/程序调用）

后端入口：`app/main.py`

模型信息：
- `GET /api/models`

人物：
- `GET /api/people?q=<keyword>&sort=S_prime_desc|S_desc|name_asc`
- `GET /api/people/{name}`

人物任务（异步 + SSE）：
- `POST /api/jobs/investigate`
  - body: `{ "name": "...", "api_key": "...", "deepsearch": false }`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/events`（SSE）

主题任务（异步 + SSE）：
- `POST /api/jobs/topic`
  - body: `{ "topic": "...", "api_key": "...", "deepsearch": false }`
- `POST /api/jobs/{job_id}/continue`
  - 当 topic 解析计划认为“定义缺失”时会进入 `waiting_user`，用该接口补充定义文本继续

主题结果：
- `GET /api/topics`
- `GET /api/topics/{topic_id}`

SSE 事件类型（常见）：`hello`, `queued`, `stage`, `log`, `progress`, `need_input`, `done`, `error`, `eof`。

---

## 3. 配置（模型、网关、并发）

### 3.1 Web 服务端环境变量（推荐）

`app/main.py` 支持：
- `DEFAULT_BASE_URL`（默认：`http://152.53.52.170:3003/v1`）
- `DEFAULT_MODEL_GPT52`（默认：`gpt-5.2-2025-12-11`）
- `DEFAULT_MODEL_DEEPSEARCH`（默认：`gemini-2.5-pro-deepsearch-async`）
- `DEFAULT_RECENT_YEARS`（默认：`2`）
- `DEEPSEARCH_WORKERS`（默认：`5`）
- `PROFILE_WORKERS`（默认：`10`）
- `MAX_CONCURRENT_JOBS`（默认：`2`）
- `IP_COOLDOWN_SEC`（默认：`10`）

### 3.2 `data/config.yaml`（主要给离线脚本/通用 LLM client）

`core/llm_client.py` 会读取 `data/config.yaml`（极简 YAML，仅支持 `key: value` 顶层键）。

重要：不要把真实 `api_key` 提交到仓库（`data/config.yaml` 默认是空 key）。

### 3.3 `data/model_list.txt`（UI 下拉可选模型）

`GET /api/models` 会把 `data/model_list.txt`（单行、逗号分隔）作为可选模型列表返回。

---

## 4. 数据与产物目录（你最终会用到的文件）

核心产物：
- `data/people/*.person.json`：每人一个最终聚合文件（profile + score + sources）
- `data/web/people/*.person.json`：网页读取用（通常与 `data/people/` 同步）
- `data/web/topics/*.topic.json`：主题榜单结果（含 items 与证据链接）

离线流水线中间产物（默认在 `.gitignore` 中忽略）：
- `data/cache/`：材料缓存（DeepResearch 或其它材料聚合的 JSON）
- `data/scored/`：评分输出（`*.score.json`）
- `data/profiles/`：结构化抽取输出（`*.profile.json`）
- `data/output/`：可选的 Markdown 等输出
- `data/candidates/`：候选人列表（如按主题从 OpenAlex 生成的 JSONL）

---

## 5. 离线脚本流水线（可选）

适合场景：你已经有一批“材料缓存”在 `data/cache/`，希望批量产出 `data/people/*.person.json`。

一键跑（Windows PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File "./pipeline.ps1"
```

`pipeline.ps1` 默认执行：
1) `scripts/score_people.py` -> `data/scored/`
2) `scripts/extract_profiles.py` -> `data/profiles/`
3) `scripts/merge_people.py` -> `data/people/`

脚本说明（摘要）：
- `scripts/score_people.py`：从 `data/cache/*.json` 读取材料，调用 LLM 输出 `*.score.json`
- `scripts/extract_profiles.py`：从 `data/cache/*.json` 读取材料，调用 LLM 抽取结构化 `*.profile.json`
- `scripts/merge_people.py`：按姓名合并最新的 score + profile 为 `*.person.json`
- `scripts/build_candidates_by_topic.py`：从 OpenAlex 按关键词聚合作者，输出 `data/candidates/*.jsonl`

注意：`scripts/run_people_deep_research.py` 目前包含未定义变量（`CRAWLER_DIR`）且在 `pipeline.ps1` 中默认注释掉；如需使用请先修复脚本。

---

## 6. 安全与运行约束（重要）

- 不要在仓库中保存真实 `api_key`：用户 key 应通过 UI/API 每次请求传入；服务端仅在任务线程栈内短暂使用。
- 不允许客户端覆盖 `base_url`（避免 SSRF）：当前实现固定使用服务端 `DEFAULT_BASE_URL`。
- 公共服务护栏：并发限制 `MAX_CONCURRENT_JOBS`，同 IP 冷却 `IP_COOLDOWN_SEC`，请求字段长度限制（见 `app/main.py`）。

---

## 7. 常见问题

- 页面没数据：确认 `data/web/people/` 下存在 `*.person.json`；或先跑一次人物/主题任务生成。
- `429 too many active jobs` / `too many requests`：调大 `MAX_CONCURRENT_JOBS` 或降低请求频率。
- 主题任务提示需要定义：按 UI 提示补充 1-3 句话定义后继续（`/api/jobs/{id}/continue`）。
