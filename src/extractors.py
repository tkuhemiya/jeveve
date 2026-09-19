from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from schemas import (
    MODELS,
    ExtractEntitiesRequest,
    ModelName,
    extract_call,
    hub_id_for,
    is_model_name,
)
from timing import SLOW_LOAD_S, ExtractEnvelope, ExtractTiming, LoadStats, log_start

type ExtractorFactory = Callable[[str], Any]
type ModelReadiness = Literal["ready", "unloaded"]


class ExtractorLoadError(Exception):
    """Raised when a model fails to load. Failures are not cached."""

    def __init__(self, model: str, *, retry_after: int = 10) -> None:
        super().__init__(f"model {model!r} failed to load")
        self.model = model
        self.retry_after = retry_after


def _default_factory(hub_id: str) -> Any:
    from gliner2 import AutoExtractor  # ty: ignore[unresolved-import]

    try:
        import torch  # ty: ignore[unresolved-import]

        # Checkpoints are CPU fp32. Pin the default so a half-precision race
        # cannot leak into a later sequential load in the same process.
        torch.set_default_dtype(torch.float32)
    except ImportError:
        pass
    return AutoExtractor.from_pretrained(hub_id)


class ExtractorCache:
    """In-process extractor cache.

    Hugging Face / PyTorch `from_pretrained` is not thread-safe. Concurrent
    first loads of different checkpoints corrupt global dtype state and leave
    models broken until process restart. This cache:

    * serializes all loads on one process-wide lock
    * uses a per-model lock so a burst for the same name waits on one load
    * caches successful extractors only (a failed load is retried next call)
    """

    def __init__(self, factory: ExtractorFactory | None = None) -> None:
        self._factory = factory or _default_factory
        self._ready: dict[str, Any] = {}
        self._stats: dict[str, LoadStats] = {}
        self._meta_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}

    def _lock_for(self, name: str) -> threading.Lock:
        with self._meta_lock:
            lock = self._model_locks.get(name)
            if lock is None:
                lock = threading.Lock()
                self._model_locks[name] = lock
            return lock

    def get(self, name: str) -> Any:
        if not is_model_name(name):
            raise ValueError(f"unknown model {name!r}")
        ready = self._ready.get(name)
        if ready is not None:
            return ready
        with self._lock_for(name):
            ready = self._ready.get(name)
            if ready is not None:
                return ready
            with self._load_lock:
                ready = self._ready.get(name)
                if ready is not None:
                    return ready
                try:
                    started = time.perf_counter()
                    extractor = self._factory(hub_id_for(name))
                    load_s = time.perf_counter() - started
                except ExtractorLoadError:
                    raise
                except Exception as exc:
                    raise ExtractorLoadError(name) from exc
                self._ready[name] = extractor
                self._stats[name] = LoadStats(
                    model=name, load_s=load_s, loaded_at=time.time()
                )
                log_start(
                    "extractor_load",
                    model=name,
                    load_s=round(load_s, 3),
                    slow=load_s >= SLOW_LOAD_S,
                )
                return extractor

    def status(self) -> dict[ModelName, ModelReadiness]:
        return {name: ("ready" if name in self._ready else "unloaded") for name in MODELS}

    def load_stats(self, name: str) -> LoadStats | None:
        return self._stats.get(name)


_CACHE = ExtractorCache()


def load_extractor(name: str, *, cache: ExtractorCache | None = None) -> Any:
    return (cache or _CACHE).get(name)


def load_stats(name: str, *, cache: ExtractorCache | None = None) -> LoadStats | None:
    return (cache or _CACHE).load_stats(name)


def model_status(*, cache: ExtractorCache | None = None) -> dict[ModelName, ModelReadiness]:
    return (cache or _CACHE).status()


def run_timed_extract(
    body: ExtractEntitiesRequest,
    *,
    cache: ExtractorCache | None = None,
    extracts_before: int = 0,
) -> ExtractEnvelope:
    cache = cache or _CACHE
    extractor = load_extractor(body.model, cache=cache)
    stats = cache.load_stats(body.model)
    started = time.perf_counter()
    result = extractor.extract_entities(
        body.text,
        body.labels,
        **extract_call(body),
    )
    infer_s = time.perf_counter() - started
    load_s = stats.load_s if stats is not None else 0.0
    extracts = extracts_before + 1
    timing = ExtractTiming(
        model=body.model,
        load_s=load_s,
        infer_s=infer_s,
        wait_s=0.0,
        cold=extracts_before == 0,
        extracts=extracts,
    )
    log_start(
        "extractor_infer",
        model=body.model,
        load_s=round(load_s, 3),
        infer_s=round(infer_s, 3),
        cold=timing.cold,
        slow=timing.slow,
        extracts=extracts,
    )
    return ExtractEnvelope(result=result, timing=timing)


def is_extractor_startup_failure(exc: BaseException) -> bool:
    """True when a remote extract failed because the model never finished loading."""
    message = str(exc).lower()
    return any(needle in message for needle in ("startup", "@modal.enter", "failed to load"))
