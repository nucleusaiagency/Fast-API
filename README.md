# Day 1 — Ingestion Engine (Pinecone + OpenAI)

This package sets up the ingestion pipeline for `.docx` transcripts and loads them into a **Pinecone** index with rich metadata. Day 1 goals:

1. Create virtualenv & install deps
2. Configure `.env`
3. Create/verify Pinecone index
4. Implement:
   - filename→metadata mapper (program-specific fields)
   - DOCX parser & text normalizer
   - chunker (500 words / 100 overlap)
   - deterministic IDs
   - embeddings + batch upserts (100/batch)
   - audit logs (CSV)
5. Run a dry ingest on sample files

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

- `POD 2025 - Episode 269 - Audio.docx` → Podcast (episode_number=269, year=2025)
- `PEA Apr 2025 MMM - Video.docx` → MMM (month=Apr, cohort=PEA, year=2025)
- `MWM December 2023 - Session 01 - Video_Tranacription.docx` → MWM (month=December, year=2023, session_name=Session 01)
- `PEA 2024 - Workshop 05 - Session 1 - Transcription.docx` → Workshop (cohort=PEA 2024, workshop_number=5, session_number=1)

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
