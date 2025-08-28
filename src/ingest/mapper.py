import re
from typing import Dict

EPISODE_RE = re.compile(r"Episode\s+(\d+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
WORKSHOP_RE = re.compile(r"Workshop\s*(\d+)", re.IGNORECASE)
SESSION_NUM_RE = re.compile(r"Session\s*(\d+)", re.IGNORECASE)
MONTH_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\b", re.IGNORECASE)

def infer_program_fields(filename: str) -> Dict:
    base = filename
    meta: Dict = {}

    # Default universal fields
    meta["program"] = "Unknown"
    meta["title"] = base.rsplit('.', 1)[0]
    meta["session_name"] = None

    # Heuristics by token
    if base.upper().startswith("POD") or "Episode" in base:
        meta["program"] = "Podcast"
        m = EPISODE_RE.search(base)
        if m:
            meta.setdefault("podcast", {})["episode_number"] = int(m.group(1))
    if "MMM" in base:
        meta["program"] = "MMM"
        m = MONTH_RE.search(base)
        if m:
            meta.setdefault("mmm", {})["month"] = m.group(1)
        if "PEA" in base:
            meta.setdefault("mmm", {})["cohort"] = "PEA"
    if base.upper().startswith("MWM") or "MWM" in base:
        meta["program"] = "MWM"
        m = MONTH_RE.search(base)
        if m:
            meta.setdefault("mwm", {})["month"] = m.group(1)
        s = SESSION_NUM_RE.search(base)
        if s:
            meta["session_name"] = f"Session {s.group(1)}"
    if "Workshop" in base:
        meta["program"] = "Workshop"
        w = WORKSHOP_RE.search(base)
        if w:
            meta.setdefault("workshop", {})["workshop_number"] = int(w.group(1))
        s = SESSION_NUM_RE.search(base)
        if s:
            meta.setdefault("workshop", {})["session_number"] = int(s.group(1))
            meta["session_name"] = f"Session {s.group(1)}"
        # cohort (simple heuristic)
        y = YEAR_RE.search(base)
        if y and "PEA" in base:
            meta.setdefault("workshop", {})["cohort"] = f"PEA {y.group(1)}"

    y = YEAR_RE.search(base)
    if y:
        meta["year"] = int(y.group(1))

    return meta
