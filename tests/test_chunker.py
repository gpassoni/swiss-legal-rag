from src.processing.chunker import chunk_act_text, chunk_business_item


def test_splits_by_article_header():
    text = """Loi sur les impôts

Art. 1 But
La présente loi règle la perception de l'impôt.

Art. 2 Champ d'application
Elle s'applique à toutes les personnes physiques.
"""
    chunks = chunk_act_text(text)

    assert [c.article for c in chunks] == [None, "1", "2"]
    assert "But" in chunks[1].text
    assert "Champ d'application" in chunks[2].text


def test_handles_article_with_letter_suffix():
    text = "Art. 21 Foo\ntext one\n\nArt. 21a Bar\ntext two\n"
    chunks = chunk_act_text(text)

    articles = [c.article for c in chunks]
    assert articles == ["21", "21a"]


def test_no_preamble_chunk_when_text_starts_with_article():
    text = "Art. 1 Foo\nSome content.\n"
    chunks = chunk_act_text(text)

    assert len(chunks) == 1
    assert chunks[0].article == "1"


def test_long_article_is_split_into_paragraph_chunks():
    paragraph = "word " * 900  # ~900 tokens, over half the 1500-token limit
    text = f"Art. 5 Title\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n"

    chunks = chunk_act_text(text, max_chunk_tokens=1500)

    long_article_chunks = [c for c in chunks if c.article == "5"]
    assert len(long_article_chunks) > 1
    assert all(c.paragraph_index is not None for c in long_article_chunks)


def test_empty_text_returns_no_chunks():
    assert chunk_act_text("") == []


def test_no_article_headers_returns_single_preamble_chunk():
    text = "Just some free text without article markers."
    chunks = chunk_act_text(text)

    assert len(chunks) == 1
    assert chunks[0].article is None
    assert chunks[0].text == text


def test_chunk_business_item_combines_fields_into_one_chunk():
    chunk = chunk_business_item(
        short_number="26.046",
        title="Legge sulle epizoozie. Modifica",
        status="Nella Commissione del Consiglio degli Stati",
        summary="Il Consiglio federale adotta il messaggio.",
    )

    assert chunk.article == "26.046"
    assert "26.046" in chunk.text
    assert "Legge sulle epizoozie" in chunk.text
    assert "Nella Commissione" in chunk.text
    assert "adotta il messaggio" in chunk.text


def test_chunk_business_item_omits_missing_optional_fields():
    chunk = chunk_business_item(short_number=None, title="Some title", status=None, summary=None)

    assert chunk.article is None
    assert chunk.text == "Some title"
