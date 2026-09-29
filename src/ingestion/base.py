"""Shared rate-limiting, retry, and logging helpers for ingestion clients."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("swiss_legal_ai.ingestion")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RateLimiter:
    """A simple async token-bucket-ish limiter: at most `rate` calls per `period` seconds.

    Sources like SNB explicitly ask integrators not to hammer their endpoints, so every
    client in this project routes outbound calls through one of these.
    """

    def __init__(self, rate: int, period: float = 1.0) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate
        self._period = period
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            window_start = now - self._period
            self._timestamps = [t for t in self._timestamps if t >= window_start]
            if len(self._timestamps) >= self._rate:
                sleep_for = self._timestamps[0] + self._period - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            self._timestamps.append(time.monotonic())


def is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def default_retry():
    """Retry decorator: exponential backoff, up to 5 attempts, only on retryable errors."""
    return retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )


class PoliteAsyncClient:
    """httpx.AsyncClient wrapper that applies rate limiting + retry + logging uniformly."""

    def __init__(
        self,
        base_url: str = "",
        rate_limit_per_sec: int = 5,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        default_headers = {
            "User-Agent": "swiss-legal-ai-phase1/0.1 (research/non-commercial testing)"
        }
        if headers:
            default_headers.update(headers)
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, headers=default_headers, follow_redirects=True
        )
        self._limiter = RateLimiter(rate=rate_limit_per_sec, period=1.0)

    async def __aenter__(self) -> PoliteAsyncClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @default_retry()
    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        await self._limiter.acquire()
        logger.info("GET %s", url)
        response = await self._client.get(url, **kwargs)
        if response.status_code != 304:
            response.raise_for_status()
        return response

    @default_retry()
    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        await self._limiter.acquire()
        logger.info("POST %s", url)
        response = await self._client.post(url, **kwargs)
        response.raise_for_status()
        return response
