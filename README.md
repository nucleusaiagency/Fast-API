# Day 1 — Ingestion Engine (Pinecone + OpenAI)

This package sets up the ingestion pipeline for `.docx` transcripts and loads
them into a **Pinecone** index with rich metadata. Day 1 goals:

1. Create virtualenv & install deps
1. Configure `.env`
1. Create/verify Pinecone index
1. Implement:
   - filename→metadata mapper (program-specific fields)
   - DOCX parser & text normalizer
   - chunker (500 words / 100 overlap)
   - deterministic IDs
   - embeddings + batch upserts (100/batch)
   - audit logs (CSV)
1. Run a dry ingest on sample files

## 0) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# fill in OPENAI_API_KEY and PINECONE_API_KEY
```

## 1) Pinecone Index

The code will auto-create the index if it doesn't exist:

- name: `transcripts` (via `PINECONE_INDEX`)
- dim: `3072` (for `text-embedding-3-large`)
- metric: cosine (serverless)

## 2) Program Mapping Rules

- `POD 2025 - Episode 269 - Audio.docx` → Podcast (episode_number=269,
  year=2025)
- `PEA Apr 2025 MMM - Video.docx` → MMM (month=Apr, cohort=PEA, year=2025)
- `MWM December 2023 - Session 01 - Video_Tranacription.docx` → MWM
  (month=December, year=2023, session_name=Session 01)
- `PEA 2024 - Workshop 05 - Session 1 - Transcription.docx` → Workshop
  (cohort=PEA 2024, workshop_number=5, session_number=1)

## 3) Run Ingestion

Put your `.docx` files into `data/docs/`. Then:

```bash
python -m src.ingest.folder --path ./data/docs --batch-size 100 --concurrency 4
```

You can target a single file for testing:

```bash
python -m src.ingest.folder --path "./data/docs/PEA 2024 - Workshop 05 - Session 1 - Transcription.docx"
```

## 4) Outputs

- Pinecone vectors with metadata (universal + program-specific)
- `runs/audit_<timestamp>.csv` — chunk-level audit for traceability
- Idempotent behavior: re-running the same file(s) won’t duplicate vectors

## 5) Next (Day 2)

- Build `/search` API (FastAPI)
- Add CLI search utility
- Create golden-test queries and evaluate Hit@K

## API: example requests

When the FastAPI server is running (for example via:
`uvicorn src.search.api:app --reload --port 8000`) the following example
requests are useful for testing metadata lookups and reloading the in-memory
META index.

Base URL:

```
http://127.0.0.1:8000
```

Note about Authorization: if you set `SEARCH_API_TOKEN` in your `.env`, include
the header `Authorization: Bearer <token>` in requests. Examples below use the
shell variable `TOKEN` or PowerShell `$env:SEARCH_API_TOKEN` as placeholders.

Examples (cURL)

- GET meta debug

```
curl -X GET "http://127.0.0.1:8000/meta/debug"
```

- POST exact workshop lookup

```
curl -X POST "http://127.0.0.1:8000/meta/lookup" \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer $TOKEN" \
   -d '{"program":"Workshop","cohort":"PEP","cohort_year":2025,"workshop_number":4,"session_number":1}'
```

- POST partial workshop lookup (returns list of matches)

```
curl -X POST "http://127.0.0.1:8000/meta/lookup" \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer $TOKEN" \
   -d '{"program":"Workshop","cohort":"PEP","workshop_number":4}'
```

- POST reload META

```
curl -X POST "http://127.0.0.1:8000/meta/reload" \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer $TOKEN" \
   -d '{}'
```

Examples (PowerShell / Invoke-RestMethod)

- GET meta debug

`powershell Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/meta/debug' `

- POST exact workshop lookup

`powershell $body = @{ program = 'Workshop'; cohort = 'PEP'; cohort_year = 2025; workshop_number = 4; session_number = 1 } | ConvertTo-Json Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/meta/lookup' -ContentType 'application/json' -Body $body -Headers @{ Authorization = "Bearer $env:SEARCH_API_TOKEN" } `

- POST partial workshop lookup

`powershell $body = @{ program = 'Workshop'; cohort = 'PEP'; workshop_number = 4 } | ConvertTo-Json Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/meta/lookup' -ContentType 'application/json' -Body $body -Headers @{ Authorization = "Bearer $env:SEARCH_API_TOKEN" } `

- POST reload META

`powershell Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/meta/reload' -ContentType 'application/json' -Body '{}' -Headers @{ Authorization = "Bearer $env:SEARCH_API_TOKEN" } `

If you want I can add these examples to a small `scripts/` folder with
ready-to-run PowerShell and bash files.
