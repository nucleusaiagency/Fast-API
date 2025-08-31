# src/search/api.py
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI

# ---- meta index (lives in src/search/index.py) ----
from .index import MasterMetaIndex
from .pinecone_client import get_pinecone_index, PineconeConnectionError


# =========================
# Environment / Clients
# =========================

API_TOKEN = os.getenv("SEARCH_API_TOKEN")  # optional bearer token
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY if OPENAI_API_KEY else None)

# Initialize Pinecone index (backwards-compatible helper)
try:
    index = get_pinecone_index(index_name=os.getenv("PINECONE_INDEX", "transcripts"), create_if_missing=True, dim=int(os.getenv("EMBED_DIM", "3072")))
except Exception as e:
    print(f"Warning: Pinecone initialization failed: {e}")
    index = None

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
    program: str  # "Workshop" | "MMM" | "MWM" | "Podcast" | "speaker"

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
    
    # Speaker matching
    query_string: Optional[str] = None    # For speaker matching


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

@app.get("/ping")
async def ping():
    """Quick ping endpoint that always returns immediately."""
    return {"status": "ok"}

@app.get("/cron")
async def cron():
    """Endpoint for cron services to keep the app alive."""
    try:
        # Quick connection tests
        test_vector = [0.0] * int(os.getenv("EMBED_DIM", "3072"))
        if index:
            await index.query(vector=test_vector, top_k=1)
        await openai_client.embeddings.create(model=EMBED_MODEL, input="test")
        return {"status": "healthy", "timestamp": str(datetime.now())}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/warmup")
async def warmup():
    """Endpoint to warm up the service and keep it alive."""
    global index
    try:
        # Test Pinecone connection with minimal query
        if index:
            test_vector = [0.0] * int(os.getenv("EMBED_DIM", "3072"))
            await index.query(vector=test_vector, top_k=1)
            
        # Test OpenAI connection with minimal request
        if openai_client:
            await openai_client.embeddings.create(
                model=EMBED_MODEL,
                input="test"
            )
            
        return {
            "status": "warmed_up",
            "pinecone": "connected" if index else "not_configured",
            "openai": "connected",
            "meta_loaded": bool(META)
        }
    except Exception as e:
        return {
            "status": "warming_up",
            "error": str(e),
            "retry_after": 30
        }

@app.get("/health")
async def health():
    """Enhanced health check that ensures Pinecone connection"""
    status = {
        "ok": True,
        "index": os.getenv("PINECONE_INDEX", "transcripts"),
        "meta_loaded": bool(META),
        "meta_sources": MASTER_INDEX_PATHS,
    }
    
    # Test Pinecone connection
    if index:
        try:
            # Simple query to test connection
            test_query = [0.0] * int(os.getenv("EMBED_DIM", "3072"))
            index.query(vector=test_query, top_k=1, include_metadata=True)
            status["pinecone"] = "connected"
        except PineconeConnectionError:
            status["pinecone"] = "starting"
            status["retry_after"] = 30
            raise HTTPException(
                status_code=503,
                detail="Search service is starting up (takes ~30s). Please retry shortly."
            )
        except Exception as e:
            status["pinecone"] = f"error: {str(e)}"
            
    return status

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
        # Exact key lookup
        if req.cohort and req.cohort_year and req.workshop_number and req.session_number:
            cohort = (req.cohort or "").upper()
            key = (cohort, int(req.cohort_year), int(req.workshop_number), int(req.session_number))
            row = META.by_workshop.get(key)
            if row:
                return {"found": True, "row": row, "partial": False}

        # Fallback to partial matches (return list)
        partials = META.lookup_workshop_partial(
            cohort=req.cohort,
            cohort_year=req.cohort_year,
            workshop_number=req.workshop_number,
            session_number=req.session_number,
            title=getattr(req, "title", None)
        )
        return {"found": bool(partials), "rows": partials, "partial": True}

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

    # Special case for speaker lookup
    if p == "speaker":
        if not META or not META._speaker_matcher:
            return {"found": False, "reason": "speaker matcher not initialized"}
        query = req.query_string or ""
        if matched := META._speaker_matcher.match(query):
            return {"found": True, "speaker": matched}
        return {"found": False, "reason": "no matching speaker found"}

    return {"found": False, "row": None, "reason": "unknown program"}


@app.post("/meta/reload", summary="Reload meta from MASTER_INDEX_PATHS")
def meta_reload(authorization: Optional[str] = Header(None)):
    auth_check(authorization)
    global META
    if not MASTER_INDEX_PATHS:
        return {"ok": False, "reason": "MASTER_INDEX_PATHS not configured"}
    try:
        META = MasterMetaIndex.load_from_paths(MASTER_INDEX_PATHS)
        return {"ok": True, "wk": len(META.by_workshop), "mmm": len(META.by_mmm), "mwm": len(META.by_mwm), "pod": len(META.by_pod)}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

# --- semantic search ---
@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, authorization: Optional[str] = Header(None)):
    """Non-async search endpoint with simplified error handling"""
    try:
        # 1. Create embedding
        emb = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=req.question
        ).data[0].embedding

        # 2. Query Pinecone
        res = index.query(
            vector=emb,
            top_k=req.k or 5,
            include_metadata=True
        )

        # 3. Format results
        matches = []
        for m in (res.get("matches", []) or []):
            md = m.get("metadata", {}) or {}
            matches.append({
                "id": m.get("id"),
                "score": float(m.get("score", 0.0)),
                "text": md.get("text", ""),
                "metadata": md
            })

        return {"results": matches}

    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or not index:
            raise HTTPException(
                status_code=503,
                detail="Service starting up - please retry in 30 seconds"
            )
        raise HTTPException(status_code=500, detail=str(e))

    if not (req.query and req.query.strip()):
        raise HTTPException(status_code=400, detail="Empty query")

    # First check Pinecone connection
    try:
        # Quick connection test
        test_query = [0.0] * int(os.getenv("EMBED_DIM", "3072"))
        index.query(vector=test_query, top_k=1)
    except PineconeConnectionError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Search service is starting up",
                "message": "The service needs about 30 seconds to wake up. Please retry shortly.",
                "retry_after": 30
            }
        )
    except Exception as e:
        if "Connection refused" in str(e):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service warming up",
                    "message": "The service is starting (takes ~30s). Please retry.",
                    "retry_after": 30
                }
            )
        raise HTTPException(status_code=502, detail=f"Pinecone connection failed: {e}")

    # Now proceed with the actual search
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
    except PineconeConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service restarting",
                "message": "The search service needs to wake up (takes ~30s). Please retry.",
                "retry_after": 30
            }
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e}")

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
