import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api import main as main_module
from src.ingestion.fedlex_client import ActDocument, ActSummary


def _act_summary(uri: str) -> ActSummary:
    return ActSummary(
        act_uri=uri,
        systematic_number="642.11",
        title="Test Act",
        language="it",
        valid_from="2020-01-01",
        valid_to=None,
    )


def _act_document(uri: str, raw_text: str) -> ActDocument:
    return ActDocument(
        act_uri=uri,
        systematic_number="642.11",
        title="Test Act",
        language="it",
        valid_from="2020-01-01",
        valid_to=None,
        source_url=f"https://fedlex.example/{uri}",
        raw_text=raw_text,
    )


def _make_fake_pg_store():
    pg_store = MagicMock()
    pg_store.init_schema = AsyncMock()
    pg_store.update_job_details = AsyncMock()
    pg_store.upsert_article_references = AsyncMock(return_value=0)
    pg_store.finish_job = AsyncMock()
    return pg_store


def _make_fake_embed_service():
    embed_service = MagicMock()
    embed_service.embed_batch.side_effect = lambda texts: [
        MagicMock(dense=[0.1, 0.2]) for _ in texts
    ]
    return embed_service


def _make_fake_store():
    store = MagicMock()
    store.upsert_chunks.side_effect = lambda records: len(records)
    return store


@pytest.mark.asyncio
async def test_bulk_job_embeds_and_upserts_per_act_not_once_at_the_end():
    # Regression for P3-12: the whole job used to accumulate every prefix's chunks in
    # memory for a single embed_batch/upsert_chunks call at the very end. This asserts
    # embedding/upserting happens once per act (streamed), not once total.
    acts = [_act_summary("uri-1"), _act_summary("uri-2")]
    docs = {
        "uri-1": _act_document("uri-1", "Art. 1\nFirst article text."),
        "uri-2": _act_document("uri-2", "Art. 2\nSecond article text."),
    }

    fake_client = MagicMock()
    fake_client.list_consolidated_acts_by_prefix_preferred.return_value = acts
    fake_client.fetch_act_preferred.side_effect = lambda act_uri, include_text: docs[act_uri]

    fake_embed_service = _make_fake_embed_service()
    fake_store = _make_fake_store()
    fake_pg_store = _make_fake_pg_store()

    with (
        patch.object(main_module, "FedlexClient", return_value=fake_client),
        patch.object(main_module, "get_embed_service", return_value=fake_embed_service),
        patch.object(main_module, "_store", return_value=fake_store),
        patch.object(main_module, "PostgresStore", return_value=fake_pg_store),
    ):
        await main_module._run_fedlex_bulk_job(job_id=1, prefixes=["642.11"], max_acts_per_prefix=5)

    assert fake_embed_service.embed_batch.call_count == 2
    assert fake_store.upsert_chunks.call_count == 2
    fake_pg_store.finish_job.assert_awaited_once()
    assert fake_pg_store.finish_job.await_args.args[1] == "completed"


@pytest.mark.asyncio
async def test_bulk_job_checkpoints_progress_incrementally():
    acts = [_act_summary("uri-1"), _act_summary("uri-2")]
    docs = {
        "uri-1": _act_document("uri-1", "Art. 1\nFirst article text."),
        "uri-2": _act_document("uri-2", "Art. 2\nSecond article text."),
    }

    fake_client = MagicMock()
    fake_client.list_consolidated_acts_by_prefix_preferred.return_value = acts
    fake_client.fetch_act_preferred.side_effect = lambda act_uri, include_text: docs[act_uri]

    fake_embed_service = _make_fake_embed_service()
    fake_store = _make_fake_store()
    fake_pg_store = _make_fake_pg_store()

    # `details` is a single mutable dict passed by reference to every update_job_details
    # call — inspecting call_args_list after the fact would just show the final state
    # repeated (all calls alias the same, now-fully-mutated object), so snapshot a deep
    # copy at call time instead to actually observe the incremental sequence.
    snapshots: list[dict] = []

    async def _capture(job_id, details):
        snapshots.append(copy.deepcopy(details))

    fake_pg_store.update_job_details = AsyncMock(side_effect=_capture)

    with (
        patch.object(main_module, "FedlexClient", return_value=fake_client),
        patch.object(main_module, "get_embed_service", return_value=fake_embed_service),
        patch.object(main_module, "_store", return_value=fake_store),
        patch.object(main_module, "PostgresStore", return_value=fake_pg_store),
    ):
        await main_module._run_fedlex_bulk_job(job_id=1, prefixes=["642.11"], max_acts_per_prefix=5)

    # At least one intermediate update_job_details call reported acts=1 (checkpoint after
    # the first act, before the second act had even been fetched) — proof progress is
    # recorded incrementally, not only once at the very end of the whole job.
    seen_acts_counts = [
        snapshot["prefixes"]["642.11"].get("acts")
        for snapshot in snapshots
        if "acts" in snapshot["prefixes"].get("642.11", {})
    ]
    assert 1 in seen_acts_counts
    assert 2 in seen_acts_counts


@pytest.mark.asyncio
async def test_bulk_job_retains_progress_from_acts_completed_before_a_later_failure():
    # Regression for P3-12's resilience goal: a failure partway through a prefix must not
    # lose the record (or the underlying persisted data) of acts already fully processed.
    acts = [_act_summary("uri-1"), _act_summary("uri-2")]

    def fetch_side_effect(act_uri, include_text):
        if act_uri == "uri-2":
            raise RuntimeError("simulated fetch failure")
        return _act_document(act_uri, "Art. 1\nFirst article text.")

    fake_client = MagicMock()
    fake_client.list_consolidated_acts_by_prefix_preferred.return_value = acts
    fake_client.fetch_act_preferred.side_effect = fetch_side_effect

    fake_embed_service = _make_fake_embed_service()
    fake_store = _make_fake_store()
    fake_pg_store = _make_fake_pg_store()

    with (
        patch.object(main_module, "FedlexClient", return_value=fake_client),
        patch.object(main_module, "get_embed_service", return_value=fake_embed_service),
        patch.object(main_module, "_store", return_value=fake_store),
        patch.object(main_module, "PostgresStore", return_value=fake_pg_store),
    ):
        await main_module._run_fedlex_bulk_job(job_id=1, prefixes=["642.11"], max_acts_per_prefix=5)

    # Act 1's chunks were embedded/upserted before act 2 failed — that work isn't undone.
    assert fake_embed_service.embed_batch.call_count == 1
    assert fake_store.upsert_chunks.call_count == 1

    # The job as a whole still completes (the failure is scoped to one prefix), and the
    # failed prefix's status retains the partial progress instead of discarding it.
    fake_pg_store.finish_job.assert_awaited_once()
    completed_details = fake_pg_store.finish_job.await_args.args[2]
    prefix_status = completed_details["prefixes"]["642.11"]
    assert prefix_status["status"] == "failed"
    assert prefix_status["acts"] == 1
    assert prefix_status["chunks"] == 1
    assert completed_details["chunks_upserted"] == 1
