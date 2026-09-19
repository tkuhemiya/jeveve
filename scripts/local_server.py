"""In-process FastAPI server for local testing (no Modal).

Loads GLiNER checkpoints in one process. Model loads are serialized so a
cold burst of `small` / `base` / `multi` cannot race PyTorch state.

    uv pip install "gliner2[local]==2.0.0"
    export ADMIN_TOKEN="local-admin-token-for-testing"
    uv run python scripts/local_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extractors import model_status, run_timed_extract
from keys import FileKeyStore
from schemas import ExtractEntitiesRequest
from web import admin_token_from_env, create_web_app

DATA_DIR = ROOT / ".local-data"
HOST = "127.0.0.1"
PORT = 8765


_EXTRACTS: dict[str, int] = {}


def extract(body: ExtractEntitiesRequest) -> object:
    envelope = run_timed_extract(body, extracts_before=_EXTRACTS.get(body.model, 0))
    _EXTRACTS[body.model] = envelope.timing.extracts
    return envelope


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = FileKeyStore(DATA_DIR / "keys.json")
    app = create_web_app(
        key_store=store,
        admin_token=admin_token_from_env(),
        extract=extract,
        model_status=model_status,
    )
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
