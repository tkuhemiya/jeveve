from extractors import ExtractorCache, run_timed_extract
from schemas import ExtractEntitiesRequest
from timing import (
    HEADER_COLD,
    HEADER_LOAD_S,
    HEADER_SLOW,
    HEADER_WAIT_S,
    SLOW_LOAD_S,
    SLOW_WAIT_S,
    ExtractEnvelope,
    ExtractTiming,
    StartLog,
    as_envelope,
    timing_headers,
)


def _timing(
    *,
    model: str = "small",
    load_s: float = 1.0,
    infer_s: float = 0.2,
    wait_s: float = 1.5,
    cold: bool = True,
    extracts: int = 1,
) -> ExtractTiming:
    return ExtractTiming(
        model=model,  # ty: ignore[invalid-argument-type]
        load_s=load_s,
        infer_s=infer_s,
        wait_s=wait_s,
        cold=cold,
        extracts=extracts,
    )


def test_slow_when_load_or_wait_crosses_the_line() -> None:
    assert not _timing().slow
    assert _timing(load_s=SLOW_LOAD_S).slow
    assert _timing(wait_s=SLOW_WAIT_S, load_s=0.1, cold=False).slow
    assert not _timing(load_s=SLOW_LOAD_S - 0.01, wait_s=SLOW_WAIT_S - 0.01).slow


def test_with_wait_replaces_only_wait() -> None:
    updated = _timing(load_s=12.0, infer_s=0.4, wait_s=0.0).with_wait(88.0)
    assert updated.load_s == 12.0
    assert updated.infer_s == 0.4
    assert updated.wait_s == 88.0
    assert updated.slow


def test_as_envelope_fills_wait_on_wrapped_result() -> None:
    inner = ExtractEnvelope(
        result={"entities": {}},
        timing=_timing(wait_s=0.0, load_s=41.0),
    )
    wrapped = as_envelope(inner, model="small", wait_s=55.2)
    assert wrapped.result == {"entities": {}}
    assert wrapped.timing.wait_s == 55.2
    assert wrapped.timing.load_s == 41.0
    assert wrapped.timing.slow


def test_as_envelope_wraps_bare_library_objects() -> None:
    wrapped = as_envelope({"ok": True}, model="multi", wait_s=0.05)
    assert wrapped.result == {"ok": True}
    assert wrapped.timing.model == "multi"
    assert wrapped.timing.load_s == 0.0
    assert wrapped.timing.wait_s == 0.05
    assert not wrapped.timing.slow


def test_timing_headers_are_explicit() -> None:
    headers = timing_headers(_timing(load_s=41.2346, infer_s=0.4, wait_s=70.0))
    assert headers[HEADER_LOAD_S] == "41.235"
    assert headers[HEADER_WAIT_S] == "70.000"
    assert headers[HEADER_COLD] == "true"
    assert headers[HEADER_SLOW] == "true"


def test_start_log_marks_health_degraded() -> None:
    log = StartLog()
    assert not log.degraded()
    assert log.snapshot() == {}
    log.record(_timing(load_s=41.0, wait_s=44.0))
    assert log.degraded()
    row = log.snapshot()["small"]
    assert row["slow"] is True
    assert row["load_s"] == 41.0
    assert row["cold"] is True
    assert "at" in row


def test_cache_records_load_seconds() -> None:
    def factory(hub_id: str) -> object:
        return type("Ext", (), {"hub_id": hub_id})()

    cache = ExtractorCache(factory=factory)
    cache.get("base")
    stats = cache.load_stats("base")
    assert stats is not None
    assert stats.model == "base"
    assert stats.load_s >= 0.0
    again = cache.get("base")
    assert cache.load_stats("base") is stats
    assert again.hub_id.endswith("base-v1")


class _FakeNer:
    def extract_entities(self, text: str, labels: object, **_: object) -> dict[str, object]:
        return {"text": text, "labels": labels}


def test_timed_extract_marks_first_call_cold() -> None:
    cache = ExtractorCache(factory=lambda hub_id: _FakeNer())
    body = ExtractEntitiesRequest(text="Apple", labels=["company"])
    first = run_timed_extract(body, cache=cache, extracts_before=0)
    second = run_timed_extract(body, cache=cache, extracts_before=first.timing.extracts)
    assert first.timing.cold is True
    assert first.timing.extracts == 1
    assert second.timing.cold is False
    assert second.timing.extracts == 2
    assert isinstance(first.result, dict)
    assert first.result["text"] == "Apple"
