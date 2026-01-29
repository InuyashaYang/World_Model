
# Crawler（World Model 人物库流水线）

## ▽ 最重要：如何打开网页（榜单/详情）
1) 确认网页可读取数据：`Crawler/web/people/` 下需要有 `*.person.json`
   - 若 `Crawler/web/people/` 为空，把最终产物复制过去：
   ```powershell
   cd World_Model
   Copy-Item .\Crawler\people\*.person.json .\Crawler\web\people\ -Force
   ```

2) 启动本地静态服务器（不要直接双击 html，`file://` 会被浏览器限制 fetch）：
   ```powershell
   cd World_Model/Crawler/web
   python -m http.server 8000
   ```

3) 浏览器打开：
- http://localhost:8000/index.html

停止服务：终端 `Ctrl + C`

---

## ▽ 产物概览（你最终要看的东西）
- `Crawler/people/`：每人一个最终“大字典” `*.person.json`
  ```json
  { "name": "...", "score": {...}, "profile": {...}, "sources": {...} }
  ```
- `Crawler/web/`：静态网页（榜单+详情），读取 `Crawler/web/people/*.person.json`

---

## ▽ 运行顺序（从原始材料到榜单）
1) DeepResearch 抓取 → `Crawler/cache/`
2) 打分 → `Crawler/scored/`
3) 资料结构化抽取 → `Crawler/profiles/`
4) 合并 → `Crawler/people/`
5) 复制 `Crawler/people/*.person.json` 到 `Crawler/web/people/` 后开网页

---

## 功能目录与文件说明（按重要性排序）

### 数据目录
- `Crawler/cache/`
  DeepResearch 原始材料缓存（`*.json`，包含 `task_id / text / ok / elapsed_sec` 等）。

- `Crawler/scored/`
  评分结果（`*.score.json`），包含：总分 S、惩罚后 S'、完备度、五维度分解、证据等。

- `Crawler/profiles/`
  资料解析结果（`*.profile.json`），把材料抽取为统一字段（homepage/scholar/dblp/works/metrics…）。

- `Crawler/people/`
  最终聚合结果（`*.person.json`），供网页渲染与后续分析直接使用。

### 网页
- `Crawler/web/index.html`：榜单页（单行一个人，搜索/排序）
- `Crawler/web/detail.html`：详情页（五维度条形图 + 结构化渲染资料/证据）
- `Crawler/web/theme.css`：公共样式（亮/暗切换）
- `Crawler/web/people/`：网页读取的 `*.person.json`

### 脚本
- `Crawler/api_requester.py`
  OpenAI-compatible 网关请求封装：`chat()` / `stream_chat()`；默认读取 `Crawler/config.yaml`。

- `Crawler/run_people_deepresearch.py`
  并发拉起 DeepResearch（默认 50 任务、并发 5），材料落到 `Crawler/cache/`。

- `Crawler/score_pool.py`
  从 `Crawler/cache/` 读取材料，调用 `gpt-5.2-2025-12-11` 并发 20 做评分，输出到 `Crawler/scored/`。

- `Crawler/profile_pool.py`
  从 `Crawler/cache/` 读取材料，调用 `gpt-5.2-2025-12-11` 并发 20 做资料字段抽取，输出到 `Crawler/profiles/`。

- `Crawler/merge_people.py`
  合并 `Crawler/scored/` + `Crawler/profiles/`，生成 `Crawler/people/*.person.json`。

---

## 其他
- `pipeline.ps1`（位于 `World_Model/pipeline.ps1`）：可选的一键流水线（按顺序跑完整流程）。
- 常见排查：
  - 网页无数据：检查 `Crawler/web/people/` 是否有 `*.person.json`
  - 端口占用：`python -m http.server 8080` 换端口
