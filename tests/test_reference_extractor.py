from src.processing.reference_extractor import extract_references


def test_internal_reference_no_abbreviation():
    text = "Il presente articolo richiama l'art. 21 e l'art. 22 per chiarimenti"
    refs = extract_references(
        text, from_systematic_number="220", from_article="97", language="it", source_url="http://x"
    )
    targets = {(r.to_systematic_number, r.to_article) for r in refs}
    assert targets == {("220", "21"), ("220", "22")}


def test_resolves_known_external_abbreviation():
    text = "gli articoli280 e 281 CPC e altre disposizioni"
    refs = extract_references(
        text, from_systematic_number="220", from_article="97", language="it", source_url="http://x"
    )
    assert len(refs) == 1
    assert refs[0].to_systematic_number == "272"
    assert refs[0].to_article == "280"


def test_unknown_abbreviation_shape_is_stored_unresolved():
    text = "si veda l'art. 5 XYZ non riconosciuto"
    refs = extract_references(
        text, from_systematic_number="220", from_article="97", language="it", source_url="http://x"
    )
    assert len(refs) == 1
    assert refs[0].to_systematic_number is None
    assert refs[0].to_article == "5"
    assert "XYZ" in refs[0].raw_reference_text


def test_self_reference_is_filtered_out():
    text = "Art. 5\n1\nI Cantoni sono autorizzati ad emanare"
    refs = extract_references(
        text, from_systematic_number="210", from_article="5", language="it", source_url="http://x"
    )
    assert refs == []


def test_german_reference_pattern():
    text = "gemäss Art. 97 OR ist der Schuldner haftbar"
    refs = extract_references(
        text, from_systematic_number="311.0", from_article="1", language="de", source_url="http://x"
    )
    assert len(refs) == 1
    assert refs[0].to_systematic_number == "220"
    assert refs[0].to_article == "97"


def test_french_reference_pattern():
    text = "conformément à l'art. 97 CO, le débiteur"
    refs = extract_references(
        text, from_systematic_number="311.0", from_article="1", language="fr", source_url="http://x"
    )
    assert len(refs) == 1
    assert refs[0].to_systematic_number == "220"
    assert refs[0].to_article == "97"


def test_unsupported_language_returns_empty():
    assert extract_references("art. 5", "210", "1", "rm", "http://x") == []


def test_duplicate_references_are_deduplicated():
    text = "vedi art. 21 e ancora art. 21 per conferma"
    refs = extract_references(
        text, from_systematic_number="220", from_article="97", language="it", source_url="http://x"
    )
    assert len(refs) == 1
