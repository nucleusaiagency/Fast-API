from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Iterable

from .cache import cached, meta_cache, clear_caches
from .speaker import SpeakerMatcher

import pandas as pd

MONTH3 = {
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "sept": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
}

def month3_from_any(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, pd.Timestamp):
        return MONTH3.get(x.strftime("%b").lower())
    s = str(x).strip()
    if not s: return None
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if isinstance(ts, pd.Timestamp) and pd.notna(ts):
            return MONTH3.get(ts.strftime("%b").lower())
    except Exception:
        pass
    # try exact/full month name
    s_lower = s.lower()
    for k, v in MONTH3.items():
        # k is short key like 'jan', check both short and full names
        if s_lower.startswith(k) or s_lower.startswith(v.lower()):
            return v
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", s, re.I)
    return MONTH3.get(m.group(1).lower()) if m else None

def to_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    try:
        s = str(x).strip()
        if not s: return None
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else int(s)
    except Exception:
        return None

def norm_str(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip()
    return s or None

def first_of(*vals):
    for v in vals:
        s = norm_str(v)
        if s: return s
    return None

def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    if df is None: return None
    # exact-case-insensitive match first
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower().strip()
        if key in lower_map:
            return lower_map[key]

    # compact mapping: remove spaces and non-alphanumeric to match headers like 'Workshop #' -> 'workshop'
    import re
    compact = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r'[^a-z0-9]', '', str(cand).lower())
        if key in compact:
            return compact[key]
    return None


@dataclass
class MasterMetaIndex:
    by_workshop: Dict[Tuple[str, int, int, int], Dict[str, Any]] = field(default_factory=dict)
    by_mmm: Dict[Tuple[int, str], Dict[str, Any]] = field(default_factory=dict)
    by_mwm: Dict[Tuple[int, str, int], Dict[str, Any]] = field(default_factory=dict)
    by_pod: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)
    
    # Speaker matching for fuzzy name lookups
    _speaker_matcher: Optional["SpeakerMatcher"] = None

    # ---------- entry ----------
    @classmethod
    def load_from_paths(cls, paths: Iterable[str]) -> "MasterMetaIndex":
        idx = cls()
        clear_caches()  # Clear caches before reload
        # Support two modes:
        # 1) Exact paths provided in MASTER_INDEX_PATHS (preferred)
        # 2) If a configured path is missing, try to find a file with the same basename under data/master/
        master_dir = os.path.join(os.getcwd(), "data", "master")
        available_master_files = {}
        if os.path.isdir(master_dir):
            for f in os.listdir(master_dir):
                available_master_files[os.path.basename(f).lower()] = os.path.join(master_dir, f)

        for raw in paths:
            path = (raw or "").strip()
            if not path:
                continue
            if not os.path.exists(path):
                # try fallback by basename in data/master
                base = os.path.basename(path).lower()
                if base in available_master_files:
                    path = available_master_files[base]
                else:
                    print(f"[META] warn: path not found -> {path}")
                    continue
            try:
                if path.lower().endswith(".xlsx"):
                    idx._load_workbook(path)
                elif path.lower().endswith(".csv"):
                    idx._load_csv(path)
                else:
                    print(f"[META] skip (unsupported ext): {path}")
            except Exception as e:
                print(f"[META] error loading {path}: {e}")
        
        # Initialize speaker matcher with all known speakers
        idx._speaker_matcher = SpeakerMatcher()
        for row in idx.by_workshop.values():
            if speaker := row.get("speakers"):
                idx._speaker_matcher.add_speaker(speaker)
        for row in idx.by_mmm.values():
            if host := row.get("host"):
                idx._speaker_matcher.add_speaker(host)
        for row in idx.by_mwm.values():
            if host := row.get("host"):
                idx._speaker_matcher.add_speaker(host)
        
        print("[META] loaded: ", end="")
        print(f"wk={len(idx.by_workshop)} ", end="")
        print(f"mmm={len(idx.by_mmm)} ", end="")
        print(f"mwm={len(idx.by_mwm)} ", end="")
        print(f"pod={len(idx.by_pod)}")
        n_speakers = len({speaker for row in idx.by_workshop.values() if (speaker := row.get("speakers"))})
        print(f"[META] indexed {n_speakers} unique speakers")
        return idx

    # ---------- workbook (workshops often live here) ----------
    def _load_workbook(self, path: str):
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        for sheet_name, df in (xls or {}).items():
            if df is None or df.empty: continue
            self._try_ingest_workshops_df(df, source=os.path.basename(path), sheet=sheet_name)

    # ---------- CSVs ----------
    def _load_csv(self, path: str):
        df = pd.read_csv(path)
        if df is None or df.empty: return
        source = os.path.basename(path)

        # 1) Try WORKSHOPS first (CSV export of the workshop sheet)
        if self._try_ingest_workshops_df(df, source=source, sheet=None):
            # still fall through to allow mixed CSVs (rare), but typically workshops CSV is dedicated
            pass

        # 2) MMM
        c_year = find_col(df, "Year")
        c_date = find_col(df, "Date")
        c_file = find_col(df, "File Name", "Filename")
        c_host = find_col(df, "Host", "Delivered by", "Speaker")
        if c_year and (c_date or c_file):
            for _, r in df.iterrows():
                year = to_int(r.get(c_year))
                if not year: continue
                mmm_month = month3_from_any(r.get(c_date)) or month3_from_any(r.get(c_file))
                if not mmm_month: continue
                self.by_mmm[(year, mmm_month)] = {
                    "program": "MMM",
                    "year": year,
                    "mmm_month": mmm_month,
                    "host": norm_str(r.get(c_host)),
                    "file_name": norm_str(r.get(c_file)),
                    "source": source,
                }

        # 3) MWM (derive year/month from Date or File Name; no strict Year column requirement)
        c_session = find_col(df, "Session #", "Session", "Session Number")
        if (c_date or c_file) and c_session:
            for _, r in df.iterrows():
                sess = to_int(r.get(c_session))
                if not sess: continue
                year = None
                month = None
                if c_date:
                    dt = r.get(c_date)
                    month = month3_from_any(dt)
                    try:
                        y = pd.to_datetime(dt, errors="coerce")
                        if pd.notna(y): year = int(y.year)
                    except Exception:
                        pass
                if not year or not month:
                    fname = norm_str(r.get(c_file))
                    if fname:
                        m_y = re.search(r"\b(19|20)\d{2}\b", fname)
                        if m_y: year = int(m_y.group(0))
                        m_m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", fname, re.I)
                        if m_m: month = MONTH3[m_m.group(1).lower()]
                if not (year and month): continue
                self.by_mwm[(year, month, sess)] = {
                    "program": "MWM",
                    "year": year,
                    "mwm_month": month,
                    "session_number": sess,
                    "host": norm_str(r.get(c_host)),
                    "file_name": norm_str(r.get(c_file)),
                    "source": source,
                }

        # 4) Podcast
        c_date = find_col(df, "Date of Podcast", "Date")
        c_file = find_col(df, "File Name", "Filename")
        if c_file:
            for _, r in df.iterrows():
                fname = norm_str(r.get(c_file))
                if not fname: continue
                year = None
                if c_date and r.get(c_date) is not None:
                    try:
                        dt = pd.to_datetime(r.get(c_date), errors="coerce")
                        if pd.notna(dt): year = int(dt.year)
                    except Exception:
                        pass
                if not year:
                    m = re.search(r"\b(19|20)\d{2}\b", fname)
                    if m: year = int(m.group(0))
                m_ep = re.search(r"episode\s*0*([0-9]+)", fname, re.I)
                ep = int(m_ep.group(1)) if m_ep else None
                if not (year and ep): continue
                self.by_pod[(year, ep)] = {
                    "program": "Podcast",
                    "year": year,
                    "episode_number": ep,
                    "file_name": fname,
                    "source": source,
                }

    # ---------- shared workshop ingestor (for both XLSX + CSV) ----------
    def _try_ingest_workshops_df(self, df: pd.DataFrame, *, source: str, sheet: Optional[str]) -> bool:
        c_prog     = find_col(df, "Programme", "Program", "Programme ")
        c_year     = find_col(df, "Cohort Year", "Year")
        c_workshop = find_col(df, "Workshop")
        c_session  = find_col(df, "Session", "Session #")
        if not all([c_prog, c_year, c_workshop, c_session]):
            return False  # not a workshop table

        c_title       = find_col(df, "Workshop Title", "Session Heading", "Topic", "Title")
        c_delivered   = find_col(df, "Delivered by", "Delivered By", "Host", "Speaker", "Speakers")
        c_type        = find_col(df, "File Type", "Type")
        c_start_month = find_col(df, "Starting Month", "Start Month", "Month")
        c_file        = find_col(df, "File Name", "Filename")

        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            programme = norm_str(r.get(c_prog))
            cohort = programme.strip().split()[0].upper() if programme else None  # "PEP 2024" -> "PEP"
            cohort_year = to_int(r.get(c_year))
            wk_no       = to_int(r.get(c_workshop))
            sess_no     = to_int(r.get(c_session))
            if not (cohort and cohort_year and wk_no and sess_no):
                continue

            rows.append({
                "program": "Workshop",
                "cohort": cohort,
                "cohort_year": cohort_year,
                "workshop_number": wk_no,
                "session_number": sess_no,
                "title": first_of(r.get(c_title)),
                "speakers": first_of(r.get(c_delivered)),
                "file_type": (norm_str(r.get(c_type)) or "").lower(),
                "file_name": norm_str(r.get(c_file)) if c_file else None,
                "start_month": month3_from_any(r.get(c_start_month)),
                "sheet": sheet,
                "source": source,
            })

        self._ingest_workshops(rows)
        return True

    # ---------- finalize / prefer transcripts but don't drop others ----------
    def _ingest_workshops(self, rows: List[Dict[str, Any]]):
        for row in rows:
            cohort = (row.get("cohort") or "").upper()
            cohort_year = to_int(row.get("cohort_year"))
            wk_no = to_int(row.get("workshop_number"))
            sess_no = to_int(row.get("session_number"))
            if not (cohort and cohort_year and wk_no and sess_no):
                continue

            key = (cohort, cohort_year, wk_no, sess_no)
            existing = self.by_workshop.get(key)
            if existing:
                old_type = (existing.get("file_type") or "").lower()
                new_type = (row.get("file_type") or "").lower()
                if old_type not in {"transcript", "transcription"} and new_type in {"transcript", "transcription"}:
                    self.by_workshop[key] = row
            else:
                self.by_workshop[key] = row

    # ---------- lookups ----------
    @cached(meta_cache)
    def lookup_workshop(self, cohort: str, cohort_year: int, workshop_number: int, session_number: int) -> Optional[Dict[str, Any]]:
        key = (cohort.upper(), int(cohort_year), int(workshop_number), int(session_number))
        return self.by_workshop.get(key)

    @cached(meta_cache)
    def lookup_workshop_partial(self, cohort: Optional[str] = None, cohort_year: Optional[int] = None,
                              workshop_number: Optional[int] = None, session_number: Optional[int] = None,
                              title: Optional[str] = None, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of workshop rows that match all provided non-empty criteria.

        Criteria are treated as AND. If no criteria provided, returns empty list.
        Speaker matching is fuzzy - will match partial names and common variations.
        """
        results: List[Dict[str, Any]] = []
        # normalize cohort
        cohort_n = str(cohort).upper() if cohort else None
        
        # If speaker provided, try to match to known speaker
        matched_speaker = None
        if speaker and self._speaker_matcher:
            matched_speaker = self._speaker_matcher.match(speaker)
        
        for key, row in self.by_workshop.items():
            k_cohort, k_year, k_wk, k_sess = key
            ok = True
            if cohort_n and k_cohort != cohort_n:
                ok = False
            if cohort_year is not None and k_year != int(cohort_year):
                ok = False
            if workshop_number is not None and k_wk != int(workshop_number):
                ok = False
            if session_number is not None and k_sess != int(session_number):
                ok = False
            if title and row.get("title"):
                if title.lower() not in str(row.get("title")).lower():
                    ok = False
            if matched_speaker and row.get("speakers"):
                if matched_speaker != row["speakers"]:
                    ok = False
            if ok:
                results.append(row)
        return results

    def lookup_mmm(self, year: int, mmm_month: str) -> Optional[Dict[str, Any]]:
        m = month3_from_any(mmm_month)
        return self.by_mmm.get((int(year), m)) if m else None

    def lookup_mwm(self, year: int, mwm_month: str, session_number: int) -> Optional[Dict[str, Any]]:
        m = month3_from_any(mwm_month)
        return self.by_mwm.get((int(year), m, int(session_number))) if m else None

    def lookup_podcast(self, year: int, episode_number: int) -> Optional[Dict[str, Any]]:
        return self.by_pod.get((int(year), int(episode_number)))
