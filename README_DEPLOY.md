# Deploy (Render, Single Service)

This repo can be deployed as a single public web app on Render:
- The same domain serves the web UI (static files) and the backend API.
- Users paste `api_key` per task (not saved on server).

## Local run

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:
- http://127.0.0.1:8000/

## Render setup

Create a **Web Service** on Render and connect your GitHub repo.

Settings:
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Environment variables (optional):
- `DEFAULT_BASE_URL` (default: `http://152.53.52.170:3003/v1`)
- `DEFAULT_MODEL_GPT52` (default: `gpt-5.2-2025-12-11`)
- `DEFAULT_MODEL_DEEPSEARCH` (default: `gemini-2.5-pro-deepsearch-async`)
- `MAX_CONCURRENT_JOBS` (default: `2`)
- `IP_COOLDOWN_SEC` (default: `10`)

## Security notes

- Do **not** commit real API keys. Keep `Crawler/config.yaml` key empty.
- Users provide `api_key` per request; the server uses it in-memory only.
- Do not allow users to override `base_url` (SSRF risk).

## Mermaid: Research / Decision Flow

```mermaid
flowchart TB
  U[User in browser]\nsearch.html -->|enter api_key (sessionStorage)| UI[Search & Investigate UI]

  UI -->|Topic mode: POST /api/jobs/topic| JT[Job: topic]
  UI -->|Person mode: POST /api/jobs/investigate| JP[Job: person]

  subgraph SSE[Progress Streaming]
    JT -->|GET /api/jobs/{id}/events| E1[SSE: stage/log/done]
    JP -->|GET /api/jobs/{id}/events| E2[SSE: stage/log/done]
  end

  subgraph TopicFlow[Topic research pipeline]
    JT --> OA1[OpenAlex works search]\nkeywords expanded (<=6)
    OA1 --> AGG[Aggregate authors]\nrelated_works/citations/recent3y/score
    AGG -->|optional| DS1[DeepSearch model]\nadd notes/evidence
    DS1 --> OUTT[Write topic leaderboard]\nCrawler/web/topics/{topic_id}.topic.json
    AGG --> OUTT
    OUTT --> API_T1[GET /api/topics]\nlist history
    OUTT --> API_T2[GET /api/topics/{topic_id}]\nread one result
  end

  subgraph PersonFlow[Person deep research pipeline]
    JP --> OA2[OpenAlex author+works]\nmaterials
    OA2 -->|optional| DS2[DeepSearch model]\nappend materials
    DS2 --> GPT[gpt-5.2]\nprofile + score (JSON)
    OA2 --> GPT
    GPT --> OUTP[Write person]\nCrawler/people/{name}.person.json\n+copy to Crawler/web/people/
    OUTP --> LB[index.html/detail.html]\nresearcher leaderboard
  end

  UI -->|sidebar link| LB
  UI -->|sidebar: topic history| API_T1
  UI -->|open a topic result| API_T2
  UI -->|click "人物深搜" on a topic item| JP
```
