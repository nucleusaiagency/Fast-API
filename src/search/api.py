# src/search/api.py
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
from pinecone import Pinecone

# ---- meta index (lives in src/search/index.py) ----
from .index import MasterMetaIndex


# =========================
# Environment / Clients
# =========================

API_TOKEN = os.getenv("SEARCH_API_TOKEN")  # optional bearer token
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY if OPENAI_API_KEY else None)

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "transcripts"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# ----- META LOADING -----
MASTER_INDEX_PATHS = [
    p.strip() for p in (os.getenv("MASTER_INDEX_PATHS") or "").split(",") if p.strip()
]
META: Optional[MasterMetaIndex] = None
if MASTER_INDEX_PATHS:
    try:
        META = MasterMetaIndex.load_from_paths(MASTER_INDEX_PATHS)
        print(
            f"[META] loaded: "
            f"wk={len(META.by_workshop)} "
            f"mmm={len(META.by_mmm)} "
            f"mwm={len(META.by_mwm)} "
            f"pod={len(META.by_pod)}"
        )
    except Exception as e:
        print(f"[META] load failed: {e}")

MONTH3_SET = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}
def m3(x: str) -> Optional[str]:
    if not x:
        return None
    s = str(x).strip()[:3].lower()
    return s.title() if s in MONTH3_SET else None


# =========================
# FastAPI
# =========================

app = FastAPI(
    title="Transcripts Search API",
    version="1.3.0",
    description="Semantic search over transcript chunks + authoritative meta lookups."
)

# CORS (CustomGPT / browser-friendly)
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
    """Maps to Pinecone chunk metadata."""
    program: Optional[str] = Field(None, description="Podcast | Workshop | MMM | MWM")
    session_name: Optional[str] = None
    speakers: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None  # e.g. {"cohort":"PEP","cohort_year":2024,"workshop_number":2,"session_number":1}

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    filters: Optional[Filters] = None
    dedupe_by_source: bool = False
    group_by_session: bool = False
    max_snippet_chars: int = 320
    namespace: Optional[str] = None

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


# ---- meta lookup request (compact) ----
class MetaLookup(BaseModel):
    program: str  # "Workshop" | "MMM" | "MWM" | "Podcast"

    # Workshop
    cohort: Optional[str] = None          # PEP / PEA
    cohort_year: Optional[int] = None
    workshop_number: Optional[int] = None
    session_number: Optional[int] = None

    # MMM
    year: Optional[int] = None
    mmm_month: Optional[str] = None       # 3-letter (Apr, Jun, ...)

    # MWM
    mwm_month: Optional[str] = None       # 3-letter
    # session_number reused above

    # Podcast
    episode_number: Optional[int] = None


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

# --- meta debug (peek at keys) ---
@app.get("/meta/debug", summary="Meta Debug")
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
        "workshop_sample_keys": list(META.by_workshop.keys())[:10],
        "mmm_sample_keys": list(META.by_mmm.keys())[:10],
        "mwm_sample_keys": list(META.by_mwm.keys())[:10],
        "pod_sample_keys": list(META.by_pod.keys())[:10],
    }

# --- authoritative meta lookup ---
@app.post("/meta/lookup")
def meta_lookup(req: MetaLookup, authorization: Optional[str] = Header(None)):
    auth_check(authorization)
    if not META:
        return {"found": False, "row": None, "reason": "meta not loaded"}

    p = (req.program or "").strip().lower()

    if p == "workshop":
        cohort = (req.cohort or "").upper()
        key = (cohort, int(req.cohort_year or 0), int(req.workshop_number or 0), int(req.session_number or 0))
        row = META.by_workshop.get(key)
        return {"found": bool(row), "row": row}

    if p == "mmm":
        month = m3(req.mmm_month or "")
        key = (int(req.year or 0), month or "")
        row = META.by_mmm.get(key)
        return {"found": bool(row), "row": row}

    if p == "mwm":
        month = m3(req.mwm_month or "")
        key = (int(req.year or 0), month or "", int(req.session_number or 0))
        row = META.by_mwm.get(key)
        return {"found": bool(row), "row": row}

    if p == "podcast":
        key = (int(req.year or 0), int(req.episode_number or 0))
        row = META.by_pod.get(key)
        return {"found": bool(row), "row": row}

    return {"found": False, "row": None, "reason": "unknown program"}

# --- semantic search ---
@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, authorization: Optional[str] = Header(None)):
    auth_check(authorization)

    if not (req.query and req.query.strip()):
        raise HTTPException(status_code=400, detail="Empty query")

    # Embed the query
    try:
        emb = openai_client.embeddings.create(model=EMBED_MODEL, input=req.query).data[0].embedding
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

    matches: List[Match] = []
    seen_sources = set()
    for m in (res.get("matches", []) or []):
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
