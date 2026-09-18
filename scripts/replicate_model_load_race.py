"""Cold-burst harness for the in-process extractor cache.

Uses a fake factory so it does not download weights. Expected after the fix:

    uv run python scripts/replicate_model_load_race.py
    CONFIRMED FIXED: concurrent loads stayed serialized; no poisoned cache.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extractors import ExtractorCache, ExtractorLoadError
from schemas import MODELS


class FakeExtractor:
    def __init__(self, hub_id: str) -> None:
        self.hub_id = hub_id


def main() -> None:
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
        for future in futures:
            future.result()

    poisoned = False
    try:
        cache.get("small")
    except ExtractorLoadError:
        poisoned = True

    if peak == 1 and all(count == 1 for count in calls.values()) and not poisoned:
        print("CONFIRMED FIXED: concurrent loads stayed serialized; no poisoned cache.")
        print(f"peak in-flight loads={peak} calls={calls} status={cache.status()}")
        return
    print("CONFIRMED: model stays broken until server restart.")
    print(f"peak in-flight loads={peak} calls={calls} poisoned={poisoned}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
