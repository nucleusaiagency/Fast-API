# src/meta/index.py
import os, re
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

# ---------- small helpers ----------
def _norm_int(x) -> Optional[int]:
    try:
        s = str(x).strip()
        return int(s)
    except Exception:
        return None

def _norm_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None

def _canonical_month(s: Optional[str]) -> Optional[str]:
    """Normalize month-ish inputs to 3-letter lowercase, e.g., 'April' -> 'apr'."""
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    m = s[:3].lower()
    return m

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the exact column name in df that matches any candidate (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

# ---------- stable content_id builders ----------
def make_content_id_from_row(program: str, row: Dict[str, Any]) -> Optional[str]:
    p = (program or "").strip().lower()
    if p == "workshop":
        cohort = (row.get("cohort") or "").upper()
        cohort_year = _norm_int(row.get("cohort_year"))
        w = _norm_int(row.get("workshop_number"))
        s = _norm_int(row.get("session_number"))
        if cohort and cohort_year and w and s:
            return f"workshop/{cohort}/{cohort_year}/{w:02d}/{s}"
    elif p == "mmm":
        year = _norm_int(row.get("year"))
        mon = _canonical_month(row.get("mmm_month") or row.get("starting_month"))
        if year and mon:
            return f"mmm/{year}/{mon}"
    elif p == "mwm":
        year = _norm_int(row.get("year"))
        mon = _canonical_month(row.get("mwm_month") or row.get("starting_month"))
        s = _norm_int(row.get("session_number"))
        if year and mon and s:
            return f"mwm/{year}/{mon}/{s}"
    elif p == "podcast":
        year = _norm_int(row.get("year"))
        ep = _norm_int(row.get("episode_number"))
        if year and ep:
            return f"pod/{year}/{ep:03d}"
    return None

# ---------- Master index loader supporting Workshop / MMM / MWM / Podcast ----------
class MasterMetaIndex:
    """
    Load 1..N CSV/XLSX files and build lookups:
      - Workshop key: (cohort, cohort_year, workshop_number, session_number)
      - MMM key: (year, mmm_month)
      - MWM key: (year, mwm_month, session_number)
      - Podcast key: (year, episode_number)
    Accepts common header variations across your sheets.
    """
    def __init__(self, paths_csv_or_xlsx: List[str]):
        self.rows: List[Dict[str, Any]] = []
        self.by_workshop: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
        self.by_mmm: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.by_mwm: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
        self.by_pod: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for p in (paths_csv_or_xlsx or []):
            p = p.strip()
            if not p or not os.path.exists(p):
                continue
            df = _read_any(p)

            # resolve common/synonym headers (case-insensitive)
            col_programme  = _first_present(df, ["Programme","Program","programme","program"])
            col_cohort     = col_programme  # often PEA/PEP live here
            col_cohort_year= _first_present(df, ["Cohort","Cohort Year","cohort","cohort year"])
            col_year       = _first_present(df, ["Year","year"])
            col_wsnum      = _first_present(df, ["Workshop #","Workshop Number","workshop #","workshop number"])
            col_session    = _first_present(df, ["Session","Session Number","session","session number"])
            col_startmon   = _first_present(df, ["Starting Month","Month","starting month","month"])
            col_title      = _first_present(df, ["Workshop Title","Title","Session Title","workshop title","title","session title"])
            col_heading    = _first_present(df, ["Session Heading","session heading"])
            col_speaker    = _first_present(df, ["Delivered by","Speaker","delivered by","speaker"])
            col_filename   = _first_present(df, ["File Name","Filename","file name","filename"])
            col_episode    = _first_present(df, ["Episode #","Episode Number","episode #","episode number"])

            # normalize into standard columns
            def put(colname: str, series_name: Optional[str], cast="str"):
                if series_name and colname not in df.columns:
                    if cast == "int":
                        df[colname] = pd.to_numeric(df[series_name], errors="coerce").astype("Int64")
                    else:
                        df[colname] = df[series_name].astype(str).str.strip()

            # program & cohort
            if col_programme:
                put("program_raw", col_programme)
            else:
                df["program_raw"] = ""

            # numeric
            put("cohort_year", col_cohort_year, cast="int")
            put("year", col_year, cast="int")
            put("workshop_number", col_wsnum, cast="int")
            put("episode_number", col_episode, cast="int")

            # strings
            put("session", col_session)
            put("starting_month", col_startmon)
            put("title", col_title)
            put("session_heading", col_heading)
            put("delivered_by", col_speaker)
            put("file_name", col_filename)

            # derive cohort from program label (PEA/PEP)
            def infer_cohort(v):
                s = _norm_str(v) or ""
                s = s.upper()
                if s in {"PEA","PEP"}:
                    return s
                return s
            df["cohort"] = df["program_raw"].apply(infer_cohort)

            # canonical program per row
            def canon_prog(v):
                s = (_norm_str(v) or "").lower()
                if s in {"pea","pep"}:  # your sheets: PEA/PEP imply Workshops
                    return "Workshop"
                if s in {"workshop","mmm","mwm","podcast"}:
                    return s.upper() if s in {"mmm","mwm"} else s.capitalize()
                return "Workshop"
            df["program"] = df["program_raw"].apply(canon_prog)

            # session_number from "Session X"
            def to_session_num(v):
                if v is None: return None
                m = re.search(r"(\d+)", str(v))
                return int(m.group(1)) if m else None
            df["session_number"] = df["session"].apply(to_session_num) if "session" in df.columns else None

            # MMM/MWM month normalization
            df["mmm_month"] = df["starting_month"].apply(_canonical_month) if "starting_month" in df.columns else None
            df["mwm_month"] = df["starting_month"].apply(_canonical_month) if "starting_month" in df.columns else None

            # content_id
            def to_cid(r: pd.Series) -> Optional[str]:
                rowd = {
                    "cohort": r.get("cohort"),
                    "cohort_year": r.get("cohort_year"),
                    "workshop_number": r.get("workshop_number"),
                    "session_number": r.get("session_number"),
                    "year": r.get("year"),
                    "mmm_month": r.get("mmm_month"),
                    "mwm_month": r.get("mwm_month"),
                    "episode_number": r.get("episode_number"),
                    "starting_month": r.get("starting_month"),
                }
                return make_content_id_from_row(r.get("program"), rowd)
            df["content_id"] = df.apply(to_cid, axis=1)

            recs = df.fillna("").to_dict(orient="records")
            self.rows.extend(recs)

        # build lookups
        for r in self.rows:
            prog = (r.get("program") or "").strip()
            if prog == "Workshop":
                key = (
                    (r.get("cohort") or "").upper(),
                    _norm_int(r.get("cohort_year")),
                    _norm_int(r.get("workshop_number")),
                    _norm_int(r.get("session_number")),
                )
                if all(key):
                    self.by_workshop[key] = r
            elif prog == "MMM":
                key = (_norm_int(r.get("year")), _canonical_month(r.get("mmm_month") or r.get("starting_month")))
                if all(key):
                    self.by_mmm[key] = r
            elif prog == "MWM":
                key = (_norm_int(r.get("year")),
                       _canonical_month(r.get("mwm_month") or r.get("starting_month")),
                       _norm_int(r.get("session_number")))
                if all(key):
                    self.by_mwm[key] = r
            elif prog == "Podcast":
                key = (_norm_int(r.get("year")), _norm_int(r.get("episode_number")))
                if all(key):
                    self.by_pod[key] = r

    # ----- public lookups -----
    def lookup_workshop(self, cohort: str, cohort_year: int, workshop_number: int, session_number: int) -> Optional[Dict[str, Any]]:
        return self.by_workshop.get((cohort.upper(), int(cohort_year), int(workshop_number), int(session_number)))

    def lookup_mmm(self, year: int, month: str) -> Optional[Dict[str, Any]]:
        return self.by_mmm.get((int(year), _canonical_month(month)))

    def lookup_mwm(self, year: int, month: str, session_number: int) -> Optional[Dict[str, Any]]:
        return self.by_mwm.get((int(year), _canonical_month(month), int(session_number)))

    def lookup_podcast(self, year: int, episode_number: int) -> Optional[Dict[str, Any]]:
        return self.by_pod.get((int(year), int(episode_number)))
