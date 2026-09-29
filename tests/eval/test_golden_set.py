"""Fast, no-network schema check for eval/golden_qa.yaml — the real evaluation run
(scripts/run_eval.py) needs a live Qdrant + Postgres + LLM API key, so it is a manual/CI-
optional tool, not a unit test. This just guards against a malformed golden-set entry
breaking `scripts.run_eval` at parse time.
"""

from scripts.run_eval import GOLDEN_SET_PATH, load_golden_set

REQUIRED_FIELDS = {"id", "area_of_law", "language", "question", "expected_citations"}


def test_golden_set_file_exists():
    assert GOLDEN_SET_PATH.exists()


def test_golden_set_is_non_empty():
    cases = load_golden_set()
    assert len(cases) > 0


def test_every_case_has_required_fields():
    for case in load_golden_set():
        missing = REQUIRED_FIELDS - case.keys()
        assert not missing, f"case {case.get('id')!r} missing fields: {missing}"


def test_every_case_has_at_least_one_expected_citation():
    for case in load_golden_set():
        assert case["expected_citations"], f"case {case['id']!r} has no expected_citations"
        for citation in case["expected_citations"]:
            assert "systematic_number" in citation
            assert "article" in citation


def test_case_ids_are_unique():
    ids = [case["id"] for case in load_golden_set()]
    assert len(ids) == len(set(ids))
