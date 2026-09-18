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

from extractors import load_extractor, model_status
from keys import FileKeyStore
from schemas import ExtractEntitiesRequest, extract_call
from web import admin_token_from_env, create_web_app

DATA_DIR = ROOT / ".local-data"
HOST = "127.0.0.1"
PORT = 8765


def extract(body: ExtractEntitiesRequest) -> object:
    extractor = load_extractor(body.model)
    return extractor.extract_entities(
        body.text,
        body.labels,
        **extract_call(body),
    )


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
