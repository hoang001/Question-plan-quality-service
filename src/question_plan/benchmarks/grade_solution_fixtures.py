"""Load the grade 2-12 solution fixtures and their inline oracles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GRADE_RANGE = range(2, 13)


def load_grade_solution_fixtures(root: Path) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for grade in GRADE_RANGE:
        path = root / "tests" / f"grade_{grade}_solution_test_cases.json"
        fixtures.extend(json.loads(path.read_text(encoding="utf-8")))
    return fixtures


def load_grade_solution_expected_cases(root: Path) -> list[dict[str, Any]]:
    expected_cases: list[dict[str, Any]] = []
    for grade in GRADE_RANGE:
        path = (
            root
            / "tests"
            / "expected"
            / f"grade_{grade}_solution_test_cases_expected.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_cases.extend(payload["cases"])
    return expected_cases
