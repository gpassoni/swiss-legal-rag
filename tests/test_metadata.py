from src.processing.chunker import Chunk
from src.processing.metadata import build_metadata, content_hash


def test_build_metadata_populates_area_of_law_from_systematic_number():
    chunk = Chunk(article="21", text="Art. 21 ...")
    meta = build_metadata(
        chunk,
        source="fedlex",
        language="it",
        source_url="https://example.test/act",
        systematic_number="642.11",
    )
    assert meta.area_of_law == "tributario"
    assert meta.article == "21"
    assert meta.systematic_number == "642.11"


def test_build_metadata_area_of_law_none_when_systematic_number_missing():
    chunk = Chunk(article=None, text="preamble")
    meta = build_metadata(
        chunk,
        source="curia_vista",
        language="it",
        source_url="https://example.test/item",
    )
    assert meta.area_of_law is None
    assert meta.systematic_number is None


def test_build_metadata_computes_content_hash_deterministically():
    chunk = Chunk(article="5", text="Art. 5 some text")
    meta1 = build_metadata(chunk, source="fedlex", language="it", source_url="https://x")
    meta2 = build_metadata(chunk, source="fedlex", language="it", source_url="https://x")
    assert meta1.content_hash == meta2.content_hash == content_hash(chunk.text)
