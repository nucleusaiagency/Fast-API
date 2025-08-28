# src/ingest/folder.py
import os
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any
from collections.abc import Mapping, Sequence

from dotenv import load_dotenv
load_dotenv()

from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from tqdm import tqdm

from openai import OpenAI
client = OpenAI()

from .index_init import get_index
from .mapper import infer_program_fields
from .parser import read_docx_text, normalize_text, build_universal_metadata
from ..utils.chunk import chunk_text
from ..utils.hash import sha1_bytes, sha1_text


# ---------- ENV ----------
OPENAI_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))
DOCS_ROOT = os.getenv("DOCS_ROOT", "./data/docs")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))


# ---------- Helpers ----------
def iter_docx_paths(root_path: str):
    """Yield .docx file paths from a file or directory."""
    if os.path.isdir(root_path):
        for dirpath, _, filenames in os.walk(root_path):
            for fn in filenames:
                if fn.lower().endswith(".docx"):
                    yield os.path.join(dirpath, fn)
    else:
        if root_path.lower().endswith(".docx"):
            yield root_path


def deterministic_id(meta: Dict[str, Any], chunk: Dict[str, Any], source_sha: str) -> str:
    """Stable ID so re-runs don't duplicate."""
    program = meta.get("program", "Unknown")
    year = str(meta.get("year") or "unknown")
    session_slug = (meta.get("title") or meta.get("session_name") or meta.get("source_file") or "session")
    # basic slug
    session_slug = "-".join(session_slug.lower().split())
    return f"{program}/{year}/{session_slug}/{source_sha[:8]}:chunk{chunk['chunk_id']:05}-{chunk['chunk_start_word']}-{chunk['chunk_end_word']}-v1"


@retry(wait=wait_exponential_jitter(initial=1, max=20), stop=stop_after_attempt(6))
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings with retries."""
    resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def upsert_batches(index, items: List[Dict[str, Any]], batch_size: int = 100):
    """Upsert to Pinecone in batches."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        vectors = [{"id": it["id"], "values": it["values"], "metadata": it["metadata"]} for it in batch]
        index.upsert(vectors=vectors)


# ---------- Metadata sanitizer (fixes 400 errors) ----------
def _norm_list(x):
    out = []
    for v in x:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def sanitize_metadata(meta: dict) -> dict:
    """
    Pinecone metadata must be: string, number, boolean or list[str].
    - Drop None
    - Flatten nested dicts:  workshop.session_number -> 'workshop_session_number'
    - Coerce lists to list[str]
    - Drop empty 'date'
    """
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue

        if isinstance(v, (str, int, float, bool)):
            clean[k] = v

        elif isinstance(v, Mapping):
            for sk, sv in v.items():
                if sv is None:
                    continue
                key = f"{k}_{sk}"
                if isinstance(sv, (str, int, float, bool)):
                    clean[key] = sv
                elif isinstance(sv, Sequence) and not isinstance(sv, (str, bytes, bytearray)):
                    lst = _norm_list(sv)
                    if lst:
                        clean[key] = lst
                else:
                    clean[key] = str(sv)

        elif isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            lst = _norm_list(v)
            if lst:
                clean[k] = lst

        else:
            clean[k] = str(v)

    if "date" in clean and (clean["date"] is None or str(clean["date"]).strip() == ""):
        clean.pop("date", None)
    return clean


# ---------- Core ingestion ----------
def ingest_file(index, path: str, audit_writer) -> int:
    """Ingest a single .docx into chunks -> embeddings -> Pinecone."""
    # For stable IDs and change detection
    with open(path, "rb") as f:
        file_bytes = f.read()
    source_sha = sha1_bytes(file_bytes)

    raw_text = read_docx_text(path)
    norm_text = normalize_text(raw_text)
    if not norm_text.strip():
        return 0

    base_meta = infer_program_fields(os.path.basename(path))
    uni_meta = build_universal_metadata(
        base_meta,
        source_file=os.path.basename(path),
        source_path=os.path.abspath(path)
    )

    chunks = chunk_text(norm_text, target_words=500, overlap_words=100)
    if not chunks:
        return 0

    # Build payloads and gather texts to embed
    payloads, to_embed = [], []
    doc_sha = sha1_text(norm_text)

    for ch in chunks:
        if not ch["text"].strip():
            continue

        meta_for_id = dict(uni_meta)  # used only to compute stable ID
        pid = deterministic_id(meta_for_id, ch, source_sha)

        meta = dict(uni_meta)
        meta.update({
            "chunk_id": ch["chunk_id"],
            "chunk_start_word": ch["chunk_start_word"],
            "chunk_end_word": ch["chunk_end_word"],
            "chunk_word_count": ch["chunk_word_count"],
            "overlap_words": ch["overlap_words"],
            "text": ch["text"],
            "source_sha": source_sha,
            "doc_sha": doc_sha,
        })
        meta = sanitize_metadata(meta)  # <-- crucial

        payloads.append((pid, meta, ch["text"]))
        to_embed.append(ch["text"])

    if not to_embed:
        return 0

    # Embed in safe mini-batches
    embeddings: List[List[float]] = []
    B = 64
    for i in range(0, len(to_embed), B):
        embeddings.extend(embed_texts(to_embed[i:i + B]))

    # Assemble, upsert, audit
    items = []
    for (pid, meta, _), emb in zip(payloads, embeddings):
        items.append({"id": pid, "values": emb, "metadata": meta})

    t0 = time.time()
    upsert_batches(index, items, batch_size=BATCH_SIZE)
    dt_ms = int((time.time() - t0) * 1000)

    for it in items:
        audit_writer.writerow({
            "run_ts": datetime.utcnow().isoformat(),
            "source_file": os.path.basename(path),
            "id": it["id"],
            "program": it["metadata"].get("program"),
            "chunk_id": it["metadata"].get("chunk_id"),
            "chunk_word_count": it["metadata"].get("chunk_word_count"),
            "upsert_ms": dt_ms // max(1, len(items)),
        })

    return len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DOCS_ROOT, help="Path to a .docx file or a folder containing .docx files")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("MAX_CONCURRENCY", "4")))  # reserved for future async
    args = ap.parse_args()

    index = get_index(create_if_missing=True)

    runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))
    os.makedirs(runs_dir, exist_ok=True)
    audit_path = os.path.join(runs_dir, f"audit_{int(time.time())}.csv")

    total = 0
    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_ts", "source_file", "id", "program", "chunk_id", "chunk_word_count", "upsert_ms"]
        )
        writer.writeheader()

        paths = list(iter_docx_paths(args.path))
        if not paths:
            print(f"No .docx files found at: {args.path}")
            return

        for p in tqdm(paths, desc="Ingesting"):
            try:
                total += ingest_file(index, p, writer)
            except Exception as e:
                # Record the error in the audit for traceability
                writer.writerow({
                    "run_ts": datetime.utcnow().isoformat(),
                    "source_file": os.path.basename(p),
                    "id": "ERROR",
                    "program": "ERROR",
                    "chunk_id": -1,
                    "chunk_word_count": 0,
                    "upsert_ms": 0
                })
                print(f"[ERROR] {p}: {e}")

    print(f"Done. Upserted {total} chunks. Audit: {audit_path}")


if __name__ == "__main__":
    main()
