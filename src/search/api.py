from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone

# --- optional bearer auth ---
API_TOKEN = os.getenv("SEARCH_API_TOKEN")  # set in .env to require auth

openai_client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "transcripts"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

app = FastAPI(title="Transcripts Search API", version="1.1.0")

# CORS (so you can call from browsers / CustomGPT tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class Filters(BaseModel):
    program: Optional[str] = Field(None, description="Podcast|Workshop|MMM|MWM")
    session_name: Optional[str] = None
    speakers: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None  # e.g. {"workshop_session_number":1}

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    filters: Optional[Filters] = None
    dedupe_by_source: bool = False           # collapse to one chunk per source_file
    group_by_session: bool = False           # group matches by session_name
    max_snippet_chars: int = 320             # shorten text in response
    namespace: Optional[str] = None          # if you later use namespaces

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

# ---------- helpers ----------
def build_pinecone_filter(f: Optional[Filters]) -> Optional[Dict[str, Any]]:
    if not f:
        return None
    filt: Dict[str, Any] = {}
    if f.program:      filt["program"] = {"$eq": f.program}
    if f.session_name: filt["session_name"] = {"$eq": f.session_name}
    if f.speakers:     filt["speakers"] = {"$in": f.speakers}
    if f.year_from or f.year_to:
        yr: Dict[str, Any] = {}
        if f.year_from is not None: yr["$gte"] = f.year_from
        if f.year_to   is not None: yr["$lte"] = f.year_to
        filt["year"] = yr
    if f.date_from or f.date_to:
        dr: Dict[str, Any] = {}
        if f.date_from: dr["$gte"] = f.date_from
        if f.date_to:   dr["$lte"] = f.date_to
        filt["date"] = dr
    if f.raw:
        for k, v in f.raw.items():
            filt[k] = {"$eq": v}
    return filt or None

def auth_check(authorization: Optional[str]):
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

# ---------- routes ----------
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health", "post": "/search"}

@app.get("/health")
def health():
    return {"ok": True, "index": os.getenv("PINECONE_INDEX", "transcripts")}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, authorization: Optional[str] = Header(None)):
    auth_check(authorization)
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    emb = openai_client.embeddings.create(model=EMBED_MODEL, input=req.query).data[0].embedding
    pc_filter = build_pinecone_filter(req.filters)

    res = index.query(
        vector=emb,
        top_k=req.k,
        include_metadata=True,
        filter=pc_filter,
        namespace=req.namespace or ""
    )

    # shape matches
    matches: List[Match] = []
    seen_sources = set()
    for m in res.get("matches", []):
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

    # optional grouping
    groups = None
    if req.group_by_session:
        by_key: Dict[str, List[Match]] = {}
        for m in matches:
            key = (m.session_name or "Unknown")
            by_key.setdefault(key, []).append(m)
        groups = [GroupedResult(key=k, matches=v) for k, v in by_key.items()]

    return SearchResponse(matches=matches if not req.group_by_session else [],
                          groups=groups,
                          used_filter=pc_filter)
