import pandas as pd
import pytest

from src.search.index import find_col, month3_from_any, MasterMetaIndex


def test_find_col_matches_variants():
    df = pd.DataFrame(columns=["Workshop #", "Cohort Year", "Programme ", "Session #"])
    assert find_col(df, "Workshop") == "Workshop #"
    assert find_col(df, "Cohort Year", "Year") == "Cohort Year"
    assert find_col(df, "Program", "Programme") == "Programme "


def test_month3_from_any_variants():
    assert month3_from_any("January") == "Jan"
    assert month3_from_any("jan") == "Jan"
    assert month3_from_any("2024-04-15") == "Apr"
    assert month3_from_any(pd.Timestamp("2025-06-01")) == "Jun"


def test_workshop_lookup_and_partial(tmp_path):
    # build a small CSV mimicking workshop file
    csv = tmp_path / "wk.csv"
    csv.write_text("Programme,Cohort Year,Workshop,Session,Workshop Title,Delivered by,File Type,File Name\nPEP 2025,2025,4,1,Strategy Day,Rachel Davis,Transcript,PEP 2025 - Workshop 04 - Session 1 - Transcript\nPEP 2025,2025,4,2,Prime Time,Adam Goff,Video,PEP 2025 - Workshop 04 - Session 2 - Video\n")

    idx = MasterMetaIndex.load_from_paths([str(csv)])
    # exact lookup
    row = idx.lookup_workshop("PEP", 2025, 4, 1)
    assert row is not None
    assert row.get("speakers") in ("Rachel Davis", "Rachel Davis")

    # partial lookup by cohort+workshop
    partials = idx.lookup_workshop_partial(cohort="PEP", cohort_year=2025, workshop_number=4)
    assert len(partials) == 2
