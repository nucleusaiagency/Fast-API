from typing import Dict, Any
from docx import Document

def read_docx_text(path: str) -> str:
    doc = Document(path)
    # Join paragraphs, preserving line breaks where appropriate
    paras = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paras.append(text)
    return "\n".join(paras)

def normalize_text(text: str) -> str:
    # Minimal normalization to keep speaker labels useful
    s = text.replace("\u00A0", " ")  # non-breaking space to normal space
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Optional: strip long runs of empty lines
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def build_universal_metadata(base_meta: Dict[str, Any], *, 
                             source_file: str, source_path: str, 
                             language: str = "en") -> Dict[str, Any]:
    m = {
        "program": base_meta.get("program", "Unknown"),
        "title": base_meta.get("title"),
        "date": base_meta.get("date"),  # may be filled later
        "speakers": base_meta.get("speakers", []),
        "session_name": base_meta.get("session_name"),
        "source_file": source_file,
        "source_path": source_path,
        "duration_min": base_meta.get("duration_min"),
        "language": language,
        "tags": base_meta.get("tags", []),
        "schema_version": 1
    }
    # namespace program-specific
    for k in ("podcast", "workshop", "mmm", "mwm"):
        if k in base_meta:
            m[k] = base_meta[k]
    if "year" in base_meta:
        m["year"] = base_meta["year"]
    return m
