from __future__ import annotations

import pandas as pd

from comfortwx.validation.real_world import compare_expected_label
from comfortwx.validation.real_world import format_mismatch_audit_table


def test_compare_expected_label_exact_match() -> None:
    assert compare_expected_label("pleasant", "pleasant") == "match"


def test_compare_expected_label_near_match() -> None:
    assert compare_expected_label("pleasant", "ideal") == "near match"


def test_compare_expected_label_mismatch() -> None:
    assert compare_expected_label("poor", "ideal") == "mismatch"


def test_format_mismatch_audit_table_filters_to_mismatches() -> None:
    summary = pd.DataFrame(
        [
            {"case_name": "A", "expected_label": "fair", "actual_label": "fair", "comparison": "match", "dominant_limiting_factor": "temperature", "top_3_reasons": "x"},
            {"case_name": "B", "expected_label": "fair", "actual_label": "ideal", "comparison": "mismatch", "dominant_limiting_factor": "humidity", "top_3_reasons": "y"},
        ]
    )
    output = format_mismatch_audit_table(summary)
    assert "B" in output
    assert "A" not in output
