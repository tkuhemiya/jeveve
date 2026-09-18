from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Literal

from schemas import MODELS, ModelName, hub_id_for, is_model_name

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
                    extractor = self._factory(hub_id_for(name))
                except ExtractorLoadError:
                    raise
                except Exception as exc:
                    raise ExtractorLoadError(name) from exc
                self._ready[name] = extractor
                return extractor

    def status(self) -> dict[ModelName, ModelReadiness]:
        return {name: ("ready" if name in self._ready else "unloaded") for name in MODELS}


_CACHE = ExtractorCache()


def load_extractor(name: str, *, cache: ExtractorCache | None = None) -> Any:
    return (cache or _CACHE).get(name)


def model_status(*, cache: ExtractorCache | None = None) -> dict[ModelName, ModelReadiness]:
    return (cache or _CACHE).status()


def is_extractor_startup_failure(exc: BaseException) -> bool:
    """True when a remote extract failed because the model never finished loading."""
    message = str(exc).lower()
    return any(needle in message for needle in ("startup", "@modal.enter", "failed to load"))
