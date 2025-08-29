# src/search/index.py
from __future__ import annotations
import os, re, pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional

MONTH3 = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}
EP_PAT = re.compile(r"episode\s*0*([0-9]+)", re.I)
COHORT_PAT = re.compile(r"(PE[AP])\s*(\d{4})", re.I)

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # strip whitespace in all string cells
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def _month3_from_any(val: Any, fallback_from_filename: Optional[str]=None) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        if fallback_from_filename:
            m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", fallback_from_filename, re.I)
            if m: return m.group(1).title()[:3]
        return None
    if isinstance(val, str):
        s = val.strip()
        # date-like? try pandas parse
        try:
            dt = pd.to_datetime(s, errors="raise", dayfirst=True)
            return dt.strftime("%b")
        except Exception:
            pass
        # 3-letter month embedded
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", s, re.I)
        if m: return m.group(1).title()[:3]
    # datetime
    try:
        dt = pd.to_datetime(val, errors="raise")
        return dt.strftime("%b")
    except Exception:
        return None

def _to_int(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    if isinstance(v, (int,)): return int(v)
    s = str(v).strip()
    if s == "": return None
    s = re.sub(r"[^\d\-]", "", s)
    try:
        return int(s)
    except Exception:
        return None

def _episode_from_row(row: dict) -> Optional[int]:
    # Try common columns then File Name pattern
    for k in ["episode #","episode","podcast #","podcast number","episode number"]:
        if k in row and _to_int(row[k]) is not None:
            return _to_int(row[k])
    fname = row.get("file name") or row.get("filename") or ""
    m = EP_PAT.search(str(fname))
    if m:
        return _to_int(m.group(1))
    return None

def _cohort_year_from_programme(s: str) -> Tuple[Optional[str], Optional[int]]:
    if not s: return (None, None)
    m = COHORT_PAT.search(s)
    if not m: return (None, None)
    cohort = m.group(1).upper()
    year = int(m.group(2))
    return (cohort, year)

@dataclass
class MasterMetaIndex:
    by_workshop: Dict[Tuple[str,int,int,int], dict] = field(default_factory=dict)
    by_mmm: Dict[Tuple[int,str], dict] = field(default_factory=dict)
    by_mwm: Dict[Tuple[int,str,int], dict] = field(default_factory=dict)
    by_pod: Dict[Tuple[int,int], dict] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)

    # --- ingestors ----------------------------------------------------------
    def ingest_workshops(self, df: pd.DataFrame):
        if df is None or df.empty: return
        df = _clean_cols(df)
        # expected columns (variants)
        prog_col = "programme"
        wk_col   = "workshop"
        ses_col  = "session"
        head_col = "session heading"
        title_col= "workshop title"
        by_col   = "delivered by"
        file_col = "file name"
        date_col = "starting month"

        for _, r in df.iterrows():
            programme = str(r.get(prog_col) or "").strip()
            cohort, year = _cohort_year_from_programme(programme)
            wk = _to_int(r.get(wk_col))
            ses = _to_int(r.get(ses_col))
            if not (cohort and year and wk and ses): 
                continue
            payload = {
                "program": "Workshop",
                "cohort": cohort,
                "cohort_year": year,
                "workshop_number": wk,
                "session_number": ses,
                "workshop_title": r.get(title_col),
                "session_title": r.get(head_col),
                "speaker": r.get(by_col),
                "start_month": r.get(date_col),
                "file_name": r.get(file_col),
            }
            self.by_workshop[(cohort, year, wk, ses)] = payload

    def ingest_mmm(self, df: pd.DataFrame):
        if df is None or df.empty: return
        df = _clean_cols(df)
        prog_col = "programme"
        date_col = "date"
        host_col = "host"
        file_col = "file name"
        year_col = "year"

        for _, r in df.iterrows():
            year = _to_int(r.get(year_col))
            month3 = _month3_from_any(r.get(date_col), r.get(file_col))
            if not (year and month3): 
                continue
            payload = {
                "program": "MMM",
                "year": year,
                "mmm_month": month3,
                "host": r.get(host_col),
                "file_name": r.get(file_col),
                "programme": r.get(prog_col),
                "date": r.get(date_col),
            }
            self.by_mmm[(year, month3)] = payload

    def ingest_mwm(self, df: pd.DataFrame):
        if df is None or df.empty: return
        df = _clean_cols(df)
        prog_col = "programme"
        date_col = "date"
        host_col = "host"
        topic_col= "topic"
        file_col = "file name"
        ses_col  = "session #"

        for _, r in df.iterrows():
            programme = str(r.get(prog_col) or "")
            cohort, year = _cohort_year_from_programme(programme)
            month3 = _month3_from_any(r.get(date_col), r.get(file_col))
            ses = _to_int(r.get(ses_col))
            if not (year and month3 and ses):
                continue
            payload = {
                "program": "MWM",
                "year": year,
                "mwm_month": month3,
                "session_number": ses,
                "host": r.get(host_col),
                "topic": r.get(topic_col),
                "file_name": r.get(file_col),
                "programme": programme,
            }
            self.by_mwm[(year, month3, ses)] = payload

    def ingest_podcasts(self, df: pd.DataFrame):
        if df is None or df.empty: return
        df = _clean_cols(df)
        date_col = "date of podcast"
        file_col = "file name"
        title_col= "podcast title"
        guests  = "guest(s)"

        for _, r in df.iterrows():
            year = None
            if r.get(date_col) is not None:
                try:
                    year = int(pd.to_datetime(r.get(date_col)).year)
                except Exception:
                    year = _to_int(r.get("year"))
            if not year:
                continue
            ep = _episode_from_row(r.to_dict())
            if not ep:
                continue
            payload = {
                "program": "Podcast",
                "year": year,
                "episode_number": ep,
                "title": r.get(title_col),
                "guests": r.get(guests),
                "file_name": r.get(file_col),
                "date": r.get(date_col),
            }
            self.by_pod[(year, ep)] = payload

    # --- load entry ----------------------------------------------------------
    @classmethod
    def load_from_paths(cls, paths: List[str]) -> "MasterMetaIndex":
        idx = cls()
        idx.sources = paths[:]
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                if p.lower().endswith(".csv"):
                    df = pd.read_csv(p)
                    # choose by a sniff of columns
                    cols = {c.lower() for c in df.columns}
                    if "podcast title" in cols or "date of podcast" in cols:
                        idx.ingest_podcasts(df)
                    elif "midmonth mentoring (mmm)" in cols or "mmm" in cols or "date" in cols and "host" in cols:
                        # MMM CSVs you sent contain those columns
                        idx.ingest_mmm(df)
                    elif "session #" in cols and "topic" in cols:
                        idx.ingest_mwm(df)
                    else:
                        # fallback ignore
                        pass
                else:
                    # Excel: pick tabs by name if present, else by columns
                    xls = pd.ExcelFile(p)
                    for sn in xls.sheet_names:
                        df = xls.parse(sn)
                        lsn = sn.lower()
                        cols = {c.lower() for c in df.columns}
                        if "workshop title" in cols and "session" in cols and "workshop" in cols:
                            idx.ingest_workshops(df)
                        elif "midmonth mentoring" in lsn or ("host" in cols and "file name" in cols and "date" in cols and "year" in cols):
                            idx.ingest_mmm(df)
                        elif "midweek mentoring" in lsn or ("session #" in cols and "topic" in cols and "file name" in cols):
                            idx.ingest_mwm(df)
                        elif "podcast title" in cols or "date of podcast" in cols:
                            idx.ingest_podcasts(df)
            except Exception as e:
                # continue on errors so one bad file doesn't kill whole load
                print(f"[META] Skipped {p}: {e}")
        return idx
