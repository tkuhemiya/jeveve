from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from schemas import ModelName, StartHealth

# Load >= 30s or total wait >= 60s is degraded. Minutes of cold start is
# not a normal operating mode; these cutoffs are how we notice.
SLOW_LOAD_S = 30.0
SLOW_WAIT_S = 60.0

HEADER_MODEL = "x-gliner-model"
HEADER_LOAD_S = "x-gliner-load-s"
HEADER_INFER_S = "x-gliner-infer-s"
HEADER_WAIT_S = "x-gliner-wait-s"
HEADER_COLD = "x-gliner-cold"
HEADER_SLOW = "x-gliner-slow"
HEADER_EXTRACTS = "x-gliner-extracts"


@dataclass(frozen=True, slots=True)
class LoadStats:
    model: str
    load_s: float
    loaded_at: float


@dataclass(frozen=True, slots=True)
class ExtractTiming:
    model: ModelName
    load_s: float
    infer_s: float
    wait_s: float
    cold: bool
    extracts: int

    @property
    def slow(self) -> bool:
        return self.load_s >= SLOW_LOAD_S or self.wait_s >= SLOW_WAIT_S

    def with_wait(self, wait_s: float) -> ExtractTiming:
        return ExtractTiming(
            model=self.model,
            load_s=self.load_s,
            infer_s=self.infer_s,
            wait_s=wait_s,
            cold=self.cold,
            extracts=self.extracts,
        )

    def as_health(self, *, at: str) -> StartHealth:
        return {
            "load_s": round(self.load_s, 3),
            "infer_s": round(self.infer_s, 3),
            "wait_s": round(self.wait_s, 3),
            "cold": self.cold,
            "slow": self.slow,
            "extracts": self.extracts,
            "at": at,
        }


@dataclass(frozen=True, slots=True)
class ExtractEnvelope:
    """Library extract body plus timing. HTTP still returns `result` only."""

    result: object
    timing: ExtractTiming


def log_start(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def as_envelope(raw: object, *, model: ModelName, wait_s: float) -> ExtractEnvelope:
    if isinstance(raw, ExtractEnvelope):
        return ExtractEnvelope(result=raw.result, timing=raw.timing.with_wait(wait_s))
    return ExtractEnvelope(
        result=raw,
        timing=ExtractTiming(
            model=model,
            load_s=0.0,
            infer_s=wait_s,
            wait_s=wait_s,
            cold=False,
            extracts=0,
        ),
    )


def timing_headers(timing: ExtractTiming) -> dict[str, str]:
    return {
        HEADER_MODEL: timing.model,
        HEADER_LOAD_S: f"{timing.load_s:.3f}",
        HEADER_INFER_S: f"{timing.infer_s:.3f}",
        HEADER_WAIT_S: f"{timing.wait_s:.3f}",
        HEADER_COLD: "true" if timing.cold else "false",
        HEADER_SLOW: "true" if timing.slow else "false",
        HEADER_EXTRACTS: str(timing.extracts),
    }


class StartLog:
    """Last extract timing per model, in this process only.

    Web health must not poke extractor pools (that would boot them).
    """

    def __init__(self) -> None:
        self._rows: dict[ModelName, StartHealth] = {}
        self._lock = threading.Lock()

    def record(self, timing: ExtractTiming) -> None:
        row = timing.as_health(at=datetime.now(UTC).isoformat())
        with self._lock:
            self._rows[timing.model] = row
        log_start(
            "extract_wait",
            model=timing.model,
            load_s=round(timing.load_s, 3),
            infer_s=round(timing.infer_s, 3),
            wait_s=round(timing.wait_s, 3),
            cold=timing.cold,
            slow=timing.slow,
            extracts=timing.extracts,
        )
        if timing.slow:
            log_start(
                "slow_start",
                model=timing.model,
                load_s=round(timing.load_s, 3),
                wait_s=round(timing.wait_s, 3),
                slow_load_s=SLOW_LOAD_S,
                slow_wait_s=SLOW_WAIT_S,
            )

    def snapshot(self) -> dict[ModelName, StartHealth]:
        with self._lock:
            return dict(self._rows)

    def degraded(self) -> bool:
        with self._lock:
            return any(row["slow"] for row in self._rows.values())
