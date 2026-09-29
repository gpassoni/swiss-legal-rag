"""Minimal Streamlit UI to manually test the RAG pipeline: hybrid search +
rerank + LLM call + citation verification, with a rough token counter.

Run with:
    uv run streamlit run scripts/streamlit_app.py

Uses ANTHROPIC_API_KEY and LLM_MODEL from .env (see src/config.py) — no separate
configuration needed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.embedding.embed_service import get_embed_service  # noqa: E402
from src.logging_config import configure_logging  # noqa: E402
from src.orchestration.query_engine import QueryEngine, build_llm_client_from_settings  # noqa: E402
from src.retrieval.reranker import get_reranker  # noqa: E402
from src.storage.postgres_store import PostgresStore  # noqa: E402
from src.storage.qdrant_store import QdrantStore  # noqa: E402

configure_logging(json_output=False)

st.set_page_config(page_title="Swiss Legal AI — test RAG", page_icon="⚖️")


@st.cache_resource(show_spinner="Caricamento modelli (embedding + reranker)...")
def get_pipeline_components():
    embed_service = get_embed_service()
    store = QdrantStore()
    reranker = get_reranker()
    return embed_service, store, reranker


def run_async(coro):
    return asyncio.run(coro)


settings = get_settings()

st.title("⚖️ Swiss Legal AI — test RAG pipeline")
st.caption("Fase 3 — interfaccia minima per test locale. Non è consulenza legale o fiscale.")

with st.sidebar:
    st.subheader("Configurazione attiva")
    st.markdown(f"**Provider LLM:** `{settings.llm_provider}`")
    st.markdown(f"**Modello LLM:** `{settings.llm_model}`")
    st.markdown(f"**Modello embedding:** `{settings.embedding_model}`")
    st.caption("Modificabili in .env (LLM_MODEL, EMBEDDING_MODEL) senza toccare il codice.")

question = st.text_area(
    "Domanda sul diritto federale svizzero, sull'attività parlamentare o sulle statistiche SNB:",
    height=100,
    placeholder="Es: Cosa disciplina l'ordinanza sull'imposizione degli utili di liquidazione?",
)

col1, col2 = st.columns(2)
with col1:
    source_filter = st.selectbox(
        "Filtra per fonte (opzionale)", ["(tutte)", "fedlex", "curia_vista"]
    )
with col2:
    top_k = st.slider("Chunk usati per la risposta", min_value=3, max_value=15, value=8)

ask_clicked = st.button("Chiedi", type="primary")

if ask_clicked and not question.strip():
    st.warning("Scrivi prima una domanda.")

if ask_clicked and question.strip():
    try:
        llm_client = build_llm_client_from_settings(settings)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    embed_service, store, reranker = get_pipeline_components()
    engine = QueryEngine(embed_service, store, reranker, llm_client, PostgresStore())

    filters = {"source": source_filter} if source_filter != "(tutte)" else None

    with st.spinner("Ricerca ibrida + reranking + generazione risposta..."):
        result = run_async(engine.answer(question, filters=filters, top_k=top_k))

    st.subheader("Risposta")
    st.markdown(result.answer)

    st.subheader(f"Fonti citate e verificate ({len(result.verified_sources)})")
    if result.verified_sources:
        for c in result.verified_sources:
            st.markdown(f"- Art. {c.article} — [{c.source_url}]({c.source_url})")
    else:
        st.info("Nessuna citazione verificata nella risposta.")

    if result.unverified_citation_count > 0:
        st.warning(
            f"⚠️ Il citation verifier ha scartato {result.unverified_citation_count} "
            "citazione/i che non corrispondono a nessun chunk effettivamente recuperato "
            "— non mostrate come fonti sopra."
        )

    with st.expander(f"Chunk recuperati e usati come contesto ({result.retrieved_chunk_count})"):
        for i, chunk in enumerate(result.chunks, start=1):
            meta = chunk.metadata
            law = meta.get("law_short_name") or meta.get("source")
            st.markdown(
                f"**[{i}] {law}** — art. {meta.get('article') or 'n/a'} "
                f"({meta.get('language')}) — [{meta.get('source_url')}]({meta.get('source_url')})"
            )
            st.text(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))

    if result.expanded_chunks:
        with st.expander(
            f"Contesto di supporto — articoli richiamati per rimando ({len(result.expanded_chunks)})"
        ):
            for i, chunk in enumerate(result.expanded_chunks, start=1):
                meta = chunk.metadata
                law = meta.get("law_short_name") or meta.get("source")
                st.markdown(
                    f"**[{i}] {law}** — art. {meta.get('article') or 'n/a'} "
                    f"({meta.get('language')}) — [{meta.get('source_url')}]({meta.get('source_url')})"
                )
                st.text(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))

    st.divider()
    usage = getattr(llm_client, "last_usage", None)
    if usage:
        total = usage["input_tokens"] + usage["output_tokens"]
        st.caption(
            f"Token usati nell'ultima chiamata ({settings.llm_model}): "
            f"{usage['input_tokens']} input + {usage['output_tokens']} output = "
            f"**{total} totali**"
        )
    else:
        st.caption("Conteggio token non disponibile per questo provider/chiamata.")
