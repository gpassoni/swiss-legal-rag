from src.config import Settings, label_for_systematic_number


def test_fedlex_systematic_prefix_list_splits_and_strips():
    settings = Settings(_env_file=None, fedlex_systematic_prefixes=" 640, 642 ,210")
    assert settings.fedlex_systematic_prefix_list == ["640", "642", "210"]


def test_fedlex_systematic_prefix_returns_first_entry():
    settings = Settings(_env_file=None, fedlex_systematic_prefixes="642,210")
    assert settings.fedlex_systematic_prefix == "642"


def test_label_for_systematic_number_matches_known_prefixes():
    assert label_for_systematic_number("642.11") == "tributario"
    assert label_for_systematic_number("210") == "civile"
    assert label_for_systematic_number("311.0") == "penale"
    assert label_for_systematic_number("312.057") == "procedura_penale"
    assert label_for_systematic_number("272.1") == "procedura_civile"
    assert label_for_systematic_number("101") == "costituzionale"


def test_label_for_systematic_number_unknown_prefix_returns_none():
    assert label_for_systematic_number("999.99") is None


def test_label_for_systematic_number_handles_none_and_empty():
    assert label_for_systematic_number(None) is None
    assert label_for_systematic_number("") is None
