import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from extractors import (
    ExtractorCache,
    ExtractorLoadError,
    is_extractor_startup_failure,
    load_extractor,
)
from schemas import MODELS


class FakeExtractor:
    def __init__(self, hub_id: str) -> None:
        self.hub_id = hub_id


def test_unknown_model_is_rejected() -> None:
    cache = ExtractorCache(factory=lambda hub_id: FakeExtractor(hub_id))
    with pytest.raises(ValueError, match="unknown model"):
        cache.get("large")


def test_successful_load_is_cached() -> None:
    calls: list[str] = []

    def factory(hub_id: str) -> FakeExtractor:
        calls.append(hub_id)
        return FakeExtractor(hub_id)

    cache = ExtractorCache(factory=factory)
    first = cache.get("small")
    second = cache.get("small")
    assert first is second
    assert calls == [MODELS["small"]]
    assert cache.status()["small"] == "ready"
    assert cache.status()["base"] == "unloaded"


def test_failed_load_is_not_cached_and_can_retry() -> None:
    calls = 0

    def factory(hub_id: str) -> FakeExtractor:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("mat1 and mat2 must have the same dtype, but got Float and Half")
        return FakeExtractor(hub_id)

    cache = ExtractorCache(factory=factory)
    with pytest.raises(ExtractorLoadError) as exc_info:
        cache.get("base")
    assert exc_info.value.model == "base"
    assert cache.status()["base"] == "unloaded"

    recovered = cache.get("base")
    assert recovered.hub_id == MODELS["base"]
    assert calls == 2
    assert cache.status()["base"] == "ready"


def test_concurrent_first_loads_of_different_models_are_serialized() -> None:
    in_flight = 0
    peak = 0
    calls: dict[str, int] = {name: 0 for name in MODELS}
    guard = threading.Lock()
    go = threading.Event()

    def factory(hub_id: str) -> FakeExtractor:
        nonlocal in_flight, peak
        name = next(key for key, value in MODELS.items() if value == hub_id)
        with guard:
            calls[name] += 1
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with guard:
            in_flight -= 1
        return FakeExtractor(hub_id)

    cache = ExtractorCache(factory=factory)

    def load(name: str) -> FakeExtractor:
        go.wait()
        return cache.get(name)

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = [pool.submit(load, name) for name in MODELS]
        go.set()
        extractors = [future.result() for future in futures]

    assert peak == 1
    assert calls == {name: 1 for name in MODELS}
    assert len({id(item) for item in extractors}) == len(MODELS)
    assert cache.status() == {name: "ready" for name in MODELS}


def test_concurrent_first_loads_of_the_same_model_run_once() -> None:
    calls = 0
    go = threading.Event()

    def factory(hub_id: str) -> FakeExtractor:
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return FakeExtractor(hub_id)

    cache = ExtractorCache(factory=factory)

    def load(_: int) -> FakeExtractor:
        go.wait()
        return cache.get("multi")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(load, i) for i in range(8)]
        go.set()
        results = [future.result() for future in futures]

    assert calls == 1
    assert all(item is results[0] for item in results)


def test_module_load_extractor_uses_injected_cache() -> None:
    cache = ExtractorCache(factory=lambda hub_id: FakeExtractor(hub_id))
    loaded = load_extractor("small", cache=cache)
    assert loaded.hub_id == MODELS["small"]


def test_startup_failure_detection() -> None:
    class RemoteError(Exception):
        pass

    assert is_extractor_startup_failure(RemoteError("container startup failed"))
    assert is_extractor_startup_failure(RuntimeError("model 'small' failed to load"))
    assert not is_extractor_startup_failure(RemoteError("out of memory"))
    assert not is_extractor_startup_failure(ValueError("bad labels"))
