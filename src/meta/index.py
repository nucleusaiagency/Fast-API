# src/meta/index.py
from __future__ import annotations

import os
import math
import pandas as pd
from typing import Dict, Any, Iterable, Optional, Tuple, List

# --------- helpers ---------

MONTH3 = {
    "jan": "Jan", "january": "Jan",
    "feb": "Feb", "february": "Feb",
    "mar": "Mar", "march": "Mar",
    "apr": "Apr", "april": "Apr",
    "may": "May",
    "jun": "Jun", "june": "Jun",
    "jul": "Jul", "july": "Jul",
    "aug": "Aug", "august": "Aug",
    "sep": "Sep", "sept": "Sep", "september": "Sep",
    "oct": "Oct", "october": "Oct",
    "nov": "Nov", "november": "Nov",
    "dec": "Dec", "december": "Dec",
}

def _is_na(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and not x.strip())

def _to_int(x) -> Optional[int]:
    if _is_na(x):
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def _norm_month3_from_str(s: str) -> Optional[str]:
    if _is_na(s):
        return None
    s = str(s).strip()
    # try pandas parser first
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            m = dt.month_name()
            return MONTH3.get(m.lower(), m[:3])
    except Exception:
        pass

    # fallback: scan tokens for a month word
    tokens = [t.strip(",.- ").lower() for t in s.split()]
    for t in tokens:
        if t in MONTH3:
            return MONTH3[t]
    # nothing found
    return None

def _norm_month3_from_date(dt) -> Optional[str]:
    if pd.isna(dt):
        return None
    try:
        if not isinstance(dt, pd.Timestamp):
            dt = pd.to_datetime(dt, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            m = dt.month_name()
            return MONTH3.get(m.lower(), m[:3])
    except Exception:
        return None
    return None

def _safe_str(x) -> Optional[str]:
    if _is_na(x):
        return None
    return str(x).strip()

def _first_token_upper(s: Optional[str]) -> Optional[str]:
    """For Programme like 'PEP 2024', returns 'PEP'."""
    if not s:
        return None
    return str(s).strip().split()[0].upper()

def _year_from_date(x) -> Optional[int]:
    try:
        dt = pd.to_datetime(x, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return int(dt.year)
    except Exception:
        return None
    return None

# --------- MasterMetaIndex ---------

class MasterMetaIndex:
    """
    Loads the master XLSX/CSV sources and indexes four lookups:

    - Workshop: (cohort, cohort_year, workshop_number, session_number)
    - MMM:      (year, mmm_month)
    - MWM:      (year, mwm_month, session_number)
    - Podcast:  (year, episode_number)

    Value is a dict of authoritative fields (title/speaker/date/file_name/etc).
    """

    def __init__(self, paths: Iterable[str]):
        self.paths: List[str] = list(paths or [])
        self.by_workshop: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
        self.by_mmm: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.by_mwm: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
        self.by_pod: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for p in self.paths:
            self._load_one(p)

    # --------------- loaders ---------------

    def _load_one(self, path: str):
        if not os.path.exists(path):
            return  # silently skip; /meta/debug will show sources anyway

        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            # read all sheets and try to classify each
            try:
                sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
            except TypeError:
                sheets = pd.read_excel(path, sheet_name=None)  # fallback, engine autodetect
            for name, df in (sheets or {}).items():
                self._classify_and_ingest(df, source=f"{path}::{name}")
        elif ext == ".csv":
            df = pd.read_csv(path)
            self._classify_and_ingest(df, source=path)
        else:
            # unsupported
            return

    def _classify_and_ingest(self, df: pd.DataFrame, source: str):
        if df is None or df.empty:
            return

        # normalize column names (lower & strip)
        colmap = {c: c for c in df.columns}
        low = [str(c).strip().lower() for c in df.columns]
        df.columns = low

        # Heuristics to detect table type
        cols = set(df.columns)

        # Podcast sheet
        if {"podcast #", "date of podcast"}.issubset(cols):
            self._ingest_podcast(df, source)
            return

        # MWM sheet (has "session #" and "topic")
        if {"session #", "date"}.issubset(cols) and ("topic" in cols or "host" in cols):
            self._ingest_mwm(df, source)
            return

        # MMM sheet (has "session" column whose values are 'Midmonth Mentoring (MMM)', plus "year" and "date")
        if "session" in cols and "year" in cols and "date" in cols:
            # Extra guard: if it also has 'workshop title' then it's workshop; otherwise MMM
            if "workshop title" not in cols and "workshop" not in cols:
                self._ingest_mmm(df, source)
                return

        # Workshop sheet (has 'workshop title' and 'session')
        if "workshop title" in cols and "session" in cols:
            self._ingest_workshop(df, source)
            return

        # If we get here, nothing matched — ignore silently.

    def _ingest_workshop(self, df: pd.DataFrame, source: str):
        """
        Expected columns (case-insensitive):
          programme | cohort year | workshop | session | session heading | delivered by | workshop title | file name | file type
        We only index PEA/PEP rows. We ignore SUPEREVENT rows.
        """
        prog_col = self._find_col(df, ["programme", "program"])
        cohort_year_col = self._find_col(df, ["cohort year", "cohort_year", "cohortyear"])
        wk_col = self._find_col(df, ["workshop"])
        sess_col = self._find_col(df, ["session"])
        delivered_col = self._find_col(df, ["delivered by", "delivered_by", "host"])
        title_col = self._find_col(df, ["workshop title", "title"])
        sess_head_col = self._find_col(df, ["session heading", "sessionheading"])
        file_col = self._find_col(df, ["file name", "filename"])
        ftype_col = self._find_col(df, ["file type", "filetype"])
        start_month_col = self._find_col(df, ["starting month", "startingmonth"])

        for _, r in df.iterrows():
            prog_raw = _safe_str(r.get(prog_col))
            cohort_token = _first_token_upper(prog_raw)  # 'PEP', 'PEA', 'SUPEREVENT', etc.
            if cohort_token not in {"PEP", "PEA"}:
                continue  # skip non-core rows

            cohort_year = _to_int(r.get(cohort_year_col))
            wk = _to_int(r.get(wk_col))
            ses = _to_int(r.get(sess_col))
            if not all([cohort_year, wk, ses]):
                continue

            row: Dict[str, Any] = {
                "program": "Workshop",
                "cohort": cohort_token,
                "cohort_year": cohort_year,
                "workshop_number": wk,
                "session_number": ses,
                "session_heading": _safe_str(r.get(sess_head_col)),
                "title": _safe_str(r.get(title_col)),
                "speaker": _safe_str(r.get(delivered_col)),
                "file_name": _safe_str(r.get(file_col)),
                "file_type": _safe_str(r.get(ftype_col)),
                "starting_month": _safe_str(r.get(start_month_col)),
                "source": source,
            }
            key = (cohort_token, cohort_year, wk, ses)
            self.by_workshop[key] = row

    def _ingest_mmm(self, df: pd.DataFrame, source: str):
        """
        Expected columns:
          programme | session | year | date | host | file name | file type
        We use (year, month3) as the key.
        """
        year_col = self._find_col(df, ["year"])
        date_col = self._find_col(df, ["date"])
        host_col = self._find_col(df, ["host"])
        file_col = self._find_col(df, ["file name", "filename"])
        ftype_col = self._find_col(df, ["file type", "filetype"])
        prog_col = self._find_col(df, ["programme", "program"])
        title_col = self._find_col(df, ["workshop title", "topic", "session heading", "session title"])  # may not exist

        for _, r in df.iterrows():
            year = _to_int(r.get(year_col))
            if not year:
                # Try derive from date if missing
                d = r.get(date_col)
                yy = _year_from_date(d)
                if yy:
                    year = yy
            if not year:
                continue

            # Month: prefer parsed date, fallback to scanning text/date string
            d = r.get(date_col)
            month = _norm_month3_from_date(d) or _norm_month3_from_str(_safe_str(d))
            if not month:
                # last fallback: from file name like "PEP Apr 2025 MMM - ..."
                fn = _safe_str(r.get(file_col))
                month = _norm_month3_from_str(fn)
            if not month:
                continue

            row = {
                "program": "MMM",
                "year": year,
                "mmm_month": month,
                "host": _safe_str(r.get(host_col)),
                "file_name": _safe_str(r.get(file_col)),
                "file_type": _safe_str(r.get(ftype_col)),
                "programme_raw": _safe_str(r.get(prog_col)),
                "title": _safe_str(r.get(title_col)),
                "date": _safe_str(r.get(date_col)),
                "source": source,
            }
            key = (year, month)
            self.by_mmm[key] = row

    def _ingest_mwm(self, df: pd.DataFrame, source: str):
        """
        Expected columns:
          programme | workshop | session # | date | host | topic | file name | file type
        We use (year, month3, session_number) as the key.
        """
        date_col = self._find_col(df, ["date"])
        host_col = self._find_col(df, ["host", "delivered by", "delivered_by"])
        topic_col = self._find_col(df, ["topic", "session heading", "title"])
        sessnum_col = self._find_col(df, ["session #", "session#", "session no", "session number"])
        file_col = self._find_col(df, ["file name", "filename"])
        ftype_col = self._find_col(df, ["file type", "filetype"])
        prog_col = self._find_col(df, ["programme", "program"])

        for _, r in df.iterrows():
            d = r.get(date_col)
            year = _year_from_date(d)
            if not year:
                continue
            month = _norm_month3_from_date(d) or _norm_month3_from_str(_safe_str(d))
            if not month:
                # fallback from file name
                month = _norm_month3_from_str(_safe_str(r.get(file_col)))
            ses = _to_int(r.get(sessnum_col))
            if not (month and ses):
                continue

            row = {
                "program": "MWM",
                "year": year,
                "mwm_month": month,
                "session_number": ses,
                "host": _safe_str(r.get(host_col)),
                "title": _safe_str(r.get(topic_col)),
                "file_name": _safe_str(r.get(file_col)),
                "file_type": _safe_str(r.get(ftype_col)),
                "programme_raw": _safe_str(r.get(prog_col)),
                "date": _safe_str(d),
                "source": source,
            }
            key = (year, month, ses)
            self.by_mwm[key] = row

    def _ingest_podcast(self, df: pd.DataFrame, source: str):
        """
        Expected columns:
          podcast # | type | podcast title | guest(s) | date of podcast | file name | file type
        We use (year, episode_number) as the key.
        """
        ep_col = self._find_col(df, ["podcast #", "podcast no", "episode", "episode #"])
        date_col = self._find_col(df, ["date of podcast", "date"])
        type_col = self._find_col(df, ["type"])
        title_col = self._find_col(df, ["podcast title", "title"])
        guests_col = self._find_col(df, ["guest(s)", "guests", "guest"])
        file_col = self._find_col(df, ["file name", "filename"])
        ftype_col = self._find_col(df, ["file type", "filetype"])

        for _, r in df.iterrows():
            ep = _to_int(r.get(ep_col))
            if not ep:
                continue
            d = r.get(date_col)
            year = _year_from_date(d)
            if not year:
                continue

            row = {
                "program": "Podcast",
                "year": year,
                "episode_number": ep,
                "title": _safe_str(r.get(title_col)),
                "guests": _safe_str(r.get(guests_col)),
                "file_name": _safe_str(r.get(file_col)),
                "file_type": _safe_str(r.get(ftype_col)),
                "date": _safe_str(d),
                "type": _safe_str(r.get(type_col)),
                "source": source,
            }
            key = (year, ep)
            self.by_pod[key] = row

    # --------------- utilities ---------------

    def _find_col(self, df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        cols = set(df.columns)
        for c in candidates:
            cl = c.strip().lower()
            if cl in cols:
                return cl
        return None

    # --------------- lookups ---------------

    def lookup_workshop(
        self,
        cohort: str,
        cohort_year: int,
        workshop_number: int,
        session_number: int,
    ) -> Optional[Dict[str, Any]]:
        key = (str(cohort).upper(), int(cohort_year), int(workshop_number), int(session_number))
        return self.by_workshop.get(key)

    def lookup_mmm(self, year: int, mmm_month: str) -> Optional[Dict[str, Any]]:
        month = MONTH3.get(str(mmm_month).strip().lower(), str(mmm_month).strip().title())
        key = (int(year), month)
        return self.by_mmm.get(key)

    def lookup_mwm(self, year: int, mwm_month: str, session_number: int) -> Optional[Dict[str, Any]]:
        month = MONTH3.get(str(mwm_month).strip().lower(), str(mwm_month).strip().title())
        key = (int(year), month, int(session_number))
        return self.by_mwm.get(key)

    def lookup_podcast(self, year: int, episode_number: int) -> Optional[Dict[str, Any]]:
        key = (int(year), int(episode_number))
        return self.by_pod.get(key)
