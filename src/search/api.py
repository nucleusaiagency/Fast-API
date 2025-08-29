from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from openai import OpenAI
from pinecone import Pinecone

# ---- Import the meta index helper (your src/meta/index.py) ----
# Must define class MasterMetaIndex with lookup_* methods
from src.meta.index import MasterMetaIndex


# =========================
# Environment / Clients
# =========================

API_TOKEN = os.getenv("SEARCH_API_TOKEN")  # optional bearer token for /search & /meta/lookup
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "transcripts"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Meta index sources (comma-separated list of files in your repo image)
MASTER_INDEX_PATHS = [
    p.strip()
    for p in os.getenv("MASTER_INDEX_PATHS", "").split(",")
    if p.strip()
]
META = MasterMetaIndex(MASTER_INDEX_PATHS) if MASTER_INDEX_PATHS else None


# =========================
# FastAPI
# =========================

app = FastAPI(
    title="Transcripts Search API",
    version="1.2.0",
    description="""
Search Pinecone for transcript chunks and look up authoritative metadata
(title, speaker, date, etc.) from your master workbook/CSVs.
""",
)

# CORS (so CustomGPT or browsers can call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Schemas
# =========================

class Filters(BaseModel):
    """
    Filters that map to Pinecone chunk metadata.
    Set program and/or raw key:value pairs that your ingestion stored per chunk.
    """
    program: Optional[str] = Field(None, description="Podcast | Workshop | MMM | MWM")
    session_name: Optional[str] = None
    speakers: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    date_from: Optional[str] = None  # ISO or yyyy-mm-dd if you stored date as text
    date_to: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None  # e.g., {\"cohort\":\"PEA\",\"cohort_year\":2024,\"workshop_number\":4,\"session_number\":1}

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    filters: Optional[Filters] = None
    dedupe_by_source: bool = False          # collapse to one chunk per source_file
    group_by_session: bool = False          # group matches by session_name
    max_snippet_chars: int = 320            # shorten text in response
    namespace: Optional[str] = None         # if you later use namespaces

class Match(BaseModel):
    id: str
    score: float
    text: str
    program: Optional[str] = None
    title: Optional[str] = None
    speakers: Optional[List[str]] = None
    date: Optional[str] = None
    session_name: Optional[str] = None
    year: Optional[int] = None
    source_file: Optional[str] = None
    chunk_id: Optional[int] = None

class GroupedResult(BaseModel):
    key: str
    matches: List[Match]

class SearchResponse(BaseModel):
    matches: List[Match] = []
    groups: Optional[List[GroupedResult]] = None
    used_filter: Optional[Dict[str, Any]] = None


# ---- Meta lookup models ----
MONTH3 = {
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "sept": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
}

class MetaLookupRequest(BaseModel):
    """
    Authoritative fact lookup.
    - Workshop requires: cohort ('PEA'|'PEP'), cohort_year (int), workshop_number (int), session_number (int)
    - MMM requires: year (int), mmm_month (3-letter, e.g. 'Apr')
    - MWM requires: year (int), mwm_month (3-letter), session_number (int)
    - Podcast requires: year (int), episode_number (int)
    """
    program: str  # "Workshop" | "MMM" | "MWM" | "Podcast"

    # Workshop
    cohort: Optional[str] = None
    cohort_year: Optional[int] = None
    workshop_number: Optional[int] = None
    session_number: Optional[int] = None

    # MMM
    year: Optional[int] = None
    mmm_month: Optional[str] = None

    # MWM
    mwm_month: Optional[str] = None

    # Podcast
    episode_number: Optional[int] = None

    @validator("program")
    def _program_norm(cls, v: str) -> str:
        if not v:
            raise ValueError("program is required")
        return v.strip().title()

    @validator("cohort")
    def _cohort_norm(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        return v.strip().upper()

    @validator("mmm_month", "mwm_month")
    def _month3_norm(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        key = v.strip().lower()
        return MONTH3.get(key, v.strip().title())

class MetaLookupResponse(BaseModel):
    found: bool
    row: Optional[Dict[str, Any]] = None


# =========================
# Helpers
# =========================

def build_pinecone_filter(f: Optional[Filters]) -> Optional[Dict[str, Any]]:
    if not f:
        return None
    filt: Dict[str, Any] = {}
    if f.program:
        filt["program"] = {"$eq": f.program}
    if f.session_name:
        filt["session_name"] = {"$eq": f.session_name}
    if f.speakers:
        filt["speakers"] = {"$in": f.speakers}
    if f.year_from is not None or f.year_to is not None:
        yr: Dict[str, Any] = {}
        if f.year_from is not None:
            yr["$gte"] = f.year_from
        if f.year_to is not None:
            yr["$lte"] = f.year_to
        filt["year"] = yr
    if f.date_from or f.date_to:
        dr: Dict[str, Any] = {}
        if f.date_from:
            dr["$gte"] = f.date_from
        if f.date_to:
            dr["$lte"] = f.date_to
        filt["date"] = dr
    if f.raw:
        for k, v in f.raw.items():
            filt[k] = {"$eq": v}
    return filt or None

def auth_check(authorization: Optional[str]):
    """Optional bearer auth for /search and /meta/lookup."""
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

def _shorten(txt: str, max_chars: int) -> str:
    t = (txt or "").replace("\r", " ").replace("\n", " ")
    return (t[:max_chars] + " ...") if len(t) > max_chars else t


# =========================
# Routes
# =========================

@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health", "post": "/search", "meta": "/meta/lookup"}

@app.get("/health")
def health():
    return {
        "ok": True,
        "index": os.getenv("PINECONE_INDEX", "transcripts"),
        "meta_loaded": bool(META),
        "meta_sources": MASTER_INDEX_PATHS,
    }

# ---- NEW: meta debug endpoint (peek at what was indexed) ----
@app.get("/meta/debug")
def meta_debug():
    if not META:
        return {"loaded": False, "reason": "No MASTER_INDEX_PATHS"}
    return {
        "loaded": True,
        "sources": MASTER_INDEX_PATHS,
        "workshop_count": len(META.by_workshop),
        "mmm_count": len(META.by_mmm),
        "mwm_count": len(META.by_mwm),
        "pod_count": len(META.by_pod),
        # peek at example keys
        "workshop_sample_keys": list(META.by_workshop.keys())[:10],
        "mmm_sample_keys": list(META.by_mmm.keys())[:10],
        "mwm_sample_keys": list(META.by_mwm.keys())[:10],
        "pod_sample_keys": list(META.by_pod.keys())[:10],
    }


# ---- Authoritative metadata lookup ----
@app.post("/meta/lookup", response_model=MetaLookupResponse, tags=["meta"])
def meta_lookup(req: MetaLookupRequest, authorization: Optional[str] = Header(None)):
    """
    Workshop:
      {"program":"Workshop","cohort":"PEA","cohort_year":2024,"workshop_number":4,"session_number":1}

    MMM:
      {"program":"MMM","year":2025,"mmm_month":"Apr"}

    MWM:
      {"program":"MWM","year":2023,"mwm_month":"Dec","session_number":1}

    Podcast:
      {"program":"Podcast","year":2020,"episode_number":1}
    """
    auth_check(authorization)

    if not META:
        raise HTTPException(status_code=500, detail="Master index not loaded. Set MASTER_INDEX_PATHS env and redeploy.")

    program = req.program

    row = None
    if program == "Workshop":
        need = [req.cohort, req.cohort_year, req.workshop_number, req.session_number]
        if not all(need):
            raise HTTPException(
                status_code=400,
                detail="Workshop lookup needs cohort, cohort_year, workshop_number, session_number",
            )
        row = META.lookup_workshop(req.cohort, req.cohort_year, req.workshop_number, req.session_number)

    elif program == "MMM":
        if not (req.year and req.mmm_month):
            raise HTTPException(status_code=400, detail="MMM lookup needs year and mmm_month (3-letter)")
        row = META.lookup_mmm(req.year, req.mmm_month)

    elif program == "MWM":
        if not (req.year and req.mwm_month and req.session_number):
            raise HTTPException(status_code=400, detail="MWM lookup needs year, mwm_month (3-letter) and session_number")
        row = META.lookup_mwm(req.year, req.mwm_month, req.session_number)

    elif program == "Podcast":
        if not (req.year and req.episode_number):
            raise HTTPException(status_code=400, detail="Podcast lookup needs year and episode_number")
        row = META.lookup_podcast(req.year, req.episode_number)

    else:
        raise HTTPException(status_code=400, detail="program must be one of Workshop | MMM | MWM | Podcast")

    return MetaLookupResponse(found=bool(row), row=row or None)


# ---- Semantic search over transcript chunks ----
@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, authorization: Optional[str] = Header(None)):
    """
    Use filters.raw to lock to a specific session/month/episode, e.g.:

    Workshop (PEA 2024 W04 S01):
    {
      "query":"AI prompts",
      "k":8,
      "filters":{"program":"Workshop","raw":{"cohort":"PEA","cohort_year":2024,"workshop_number":4,"session_number":1}},
      "dedupe_by_source":true
    }

    MMM (Apr 2025):
    {
      "query":"Q&A",
      "k":8,
      "filters":{"program":"MMM","raw":{"year":2025,"mmm_month":"Apr"}},
      "dedupe_by_source":true
    }
    """
    auth_check(authorization)

    if not (req.query and req.query.strip()):
        raise HTTPException(status_code=400, detail="Empty query")

    # Embed the query
    try:
        emb = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=req.query
        ).data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI embedding failed: {e}")

    pc_filter = build_pinecone_filter(req.filters)

    try:
        res = index.query(
            vector=emb,
            top_k=req.k,
            include_metadata=True,
            filter=pc_filter,
            namespace=req.namespace or ""
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone query failed: {e}")

    # Shape matches
    matches: List[Match] = []
    seen_sources = set()

    for m in res.get("matches", []) or []:
        md = m.get("metadata", {}) or {}

        if req.dedupe_by_source:
            src = md.get("source_file")
            if src and src in seen_sources:
                continue
            if src:
                seen_sources.add(src)

        matches.append(Match(
            id=m.get("id"),
            score=float(m.get("score", 0.0)),
            text=_shorten(md.get("text", ""), req.max_snippet_chars),
            program=md.get("program"),
            title=md.get("title"),
            speakers=md.get("speakers"),
            date=md.get("date"),
            session_name=md.get("session_name"),
            year=md.get("year"),
            source_file=md.get("source_file"),
            chunk_id=md.get("chunk_id"),
        ))

    # Optional grouping by session_name
    groups = None
    if req.group_by_session:
        by_key: Dict[str, List[Match]] = {}
        for mm in matches:
            key = (mm.session_name or "Unknown")
            by_key.setdefault(key, []).append(mm)
        groups = [GroupedResult(key=k, matches=v) for k, v in by_key.items()]

    return SearchResponse(
        matches=matches if not req.group_by_session else [],
        groups=groups,
        used_filter=pc_filter
    )
