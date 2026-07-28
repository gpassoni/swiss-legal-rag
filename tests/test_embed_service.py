import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import numpy as np

from src.embedding.embed_service import (
    EmbedService,
    _approx_token_len,
    _hash_token,
    _length_bucketed_batches,
)

# Regression test for the P0 bug: `_hash_token` used to be `hash(token) % dim`, and
# Python randomizes `hash()` on `str` per process (PYTHONHASHSEED) unless the process
# pins it — so the same token mapped to a different sparse index in every fresh
# interpreter. Sparse vectors are produced by a different process at ingestion time than
# at query time (and a restarted API process), so that bug silently broke the sparse half
# of hybrid search. This test proves determinism holds *without* pinning PYTHONHASHSEED
# (unset here, and explicitly randomized in one of the two subprocesses) — the whole
# point is that the fix must not depend on the environment doing that for us.
_HASH_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {src_root!r})
    from src.embedding.embed_service import _hash_token
    print(_hash_token("obbligazioni", 2**18))
    print(_hash_token("Art. 41 CO", 2**18))
    """
)


def _run_hash_in_subprocess(src_root: str, hash_seed: str) -> list[str]:
    env = {"PYTHONHASHSEED": hash_seed}
    # Subprocess needs a minimal but functional environment (PATH for the interpreter
    # itself on Windows, SYSTEMROOT for CPython's own startup on Windows).
    import os

    env["PATH"] = os.environ.get("PATH", "")
    env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    result = subprocess.run(
        [sys.executable, "-c", _HASH_SUBPROCESS_SCRIPT.format(src_root=src_root)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip().splitlines()


def test_hash_token_is_deterministic_across_processes_with_different_hash_seeds():
    import os

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    out_seed_0 = _run_hash_in_subprocess(src_root, hash_seed="0")
    out_seed_42 = _run_hash_in_subprocess(src_root, hash_seed="42")
    out_seed_random = _run_hash_in_subprocess(src_root, hash_seed="random")

    assert out_seed_0 == out_seed_42 == out_seed_random


def test_hash_token_matches_in_process_value():
    # Sanity check: the subprocess-computed hash must also match what the current
    # (in-process) implementation produces, not just agree with itself across processes.
    import os

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = _run_hash_in_subprocess(src_root, hash_seed="random")
    assert int(out[0]) == _hash_token("obbligazioni", 2**18)
    assert int(out[1]) == _hash_token("Art. 41 CO", 2**18)


def test_length_bucketed_batches_respects_item_count_cap():
    lengths = [10, 20, 30, 40, 50]
    batches = _length_bucketed_batches(lengths, batch_size=2, max_tokens_per_batch=None)
    assert all(len(b) <= 2 for b in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))


def test_length_bucketed_batches_respects_token_budget():
    # Five items of length 100 with a 250-token budget should never share a batch of
    # more than 2, even though the item-count cap alone would allow up to 5.
    lengths = [100, 100, 100, 100, 100]
    batches = _length_bucketed_batches(lengths, batch_size=5, max_tokens_per_batch=250)
    assert all(sum(lengths[i] for i in b) <= 250 for b in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))


def test_length_bucketed_batches_isolates_long_outlier():
    # A single very long outlier among many short texts should end up in its own
    # (small) batch rather than forcing padding across a batch of unrelated short texts.
    lengths = [5, 5, 5, 5, 2000]
    batches = _length_bucketed_batches(lengths, batch_size=16, max_tokens_per_batch=100)
    outlier_batch = next(b for b in batches if 4 in b)
    assert len(outlier_batch) == 1


def test_approx_token_len_never_zero_for_nonempty_text():
    assert _approx_token_len("word") == 1
    assert _approx_token_len("") == 1  # avoids a zero-length item breaking the budget math


def _make_service_with_mocked_model():
    service = EmbedService(model_name="BAAI/bge-m3", device="cpu")
    mock_model = MagicMock()

    def fake_encode(texts, **kwargs):
        return np.array([[float(len(t))] for t in texts])

    mock_model.encode.side_effect = fake_encode
    service._model = mock_model
    return service, mock_model


def test_embed_dense_preserves_original_order_across_buckets():
    service, mock_model = _make_service_with_mocked_model()
    texts = ["a" * 5, "b" * 500, "c" * 1, "d" * 300, "e" * 200]

    with patch("src.embedding.embed_service.get_settings") as mock_settings:
        mock_settings.return_value.embedding_batch_size = 2
        mock_settings.return_value.embedding_max_tokens_per_batch = 100

        result = service.embed_dense(texts)

    assert result == [[float(len(t))] for t in texts]
    # More than one encode() call proves the token budget actually split the batches.
    assert mock_model.encode.call_count > 1


def test_embed_dense_e5_prefixing_still_applied_with_bucketing():
    service = EmbedService(model_name="intfloat/multilingual-e5-base", device="cpu")
    mock_model = MagicMock()
    seen_texts: list[str] = []

    def fake_encode(texts, **kwargs):
        seen_texts.extend(texts)
        return np.array([[0.0] for _ in texts])

    mock_model.encode.side_effect = fake_encode
    service._model = mock_model

    with patch("src.embedding.embed_service.get_settings") as mock_settings:
        mock_settings.return_value.embedding_batch_size = 16
        mock_settings.return_value.embedding_max_tokens_per_batch = None
        service.embed_dense(["hello"], task="retrieval.query")

    assert seen_texts == ["query: hello"]
