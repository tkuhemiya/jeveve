from __future__ import annotations

from pathlib import Path

import modal

from keys import FileKeyStore
from schemas import MODELS, ExtractEntitiesRequest, ExtractResult, extract_call, hub_id_for
from web import admin_token_from_env, create_web_app

app = modal.App("gliner")

keys_vol = modal.Volume.from_name("gliner-keys", create_if_missing=True)
admin = modal.Secret.from_name("gliner-admin")

KEYS_PATH = Path("/keys/keys.json")
SRC_DIR = Path(__file__).resolve().parent
_PRELOAD = ";".join(f"A.from_pretrained({hub_id!r})" for hub_id in MODELS.values())

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.141.1",
        "pydantic==2.13.5",
    )
    .add_local_dir(str(SRC_DIR), remote_path="/pkg")
    .env({"PYTHONPATH": "/pkg"})
)

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gliner2[local]==2.0.0", "pydantic==2.13.5")
    .add_local_dir(str(SRC_DIR), remote_path="/pkg")
    .env({"HF_HOME": "/root/.cache/huggingface", "PYTHONPATH": "/pkg"})
    .run_commands(f'python -c "from gliner2 import AutoExtractor as A; {_PRELOAD}"')
)


@app.cls(
    image=infer_image,
    cpu=2,
    memory=8192,
    min_containers=0,
    scaledown_window=60,
    timeout=300,
    startup_timeout=600,
    max_inputs=1,
)
class Extractor:
    name: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from gliner2 import AutoExtractor  # ty: ignore[unresolved-import]

        self.extractor = AutoExtractor.from_pretrained(hub_id_for(self.name))

    @modal.method()
    def extract(self, body: ExtractEntitiesRequest) -> ExtractResult:
        return self.extractor.extract_entities(
            body.text,
            body.labels,
            **extract_call(body),
        )


def _dispatch(body: ExtractEntitiesRequest) -> ExtractResult:
    extractor = Extractor(name=body.model)  # ty: ignore[unknown-argument]
    return extractor.extract.remote(body)


@app.function(
    image=web_image,
    cpu=1,
    memory=1024,
    min_containers=0,
    scaledown_window=60,
    secrets=[admin],
    volumes={"/keys": keys_vol},
)
@modal.asgi_app()
def web():
    store = FileKeyStore(KEYS_PATH, reload=keys_vol.reload, commit=keys_vol.commit)
    return create_web_app(
        key_store=store,
        admin_token=admin_token_from_env(),
        extract=_dispatch,
    )
