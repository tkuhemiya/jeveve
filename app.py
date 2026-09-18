from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from keys import FileKeyStore
from schemas import MODELS, ExtractEntitiesRequest
from web import admin_token_from_env, create_web_app

app = modal.App("gliner")

keys_vol = modal.Volume.from_name("gliner-keys", create_if_missing=True)
admin = modal.Secret.from_name("gliner-admin")

KEYS_PATH = Path("/keys/keys.json")

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.141.1",
    "pydantic==2.13.5",
)

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gliner2[local]==2.0.0", "pydantic==2.13.5")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .run_commands(
        "python -c \""
        "from gliner2 import AutoExtractor as A;"
        "A.from_pretrained('fastino/gliner2.5-small-v1');"
        "A.from_pretrained('fastino/gliner2.5-base-v1');"
        "A.from_pretrained('fastino/gliner2.5-multi-v1')\""
    )
)


def _extract_kwargs(body: ExtractEntitiesRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "include_confidence": body.include_confidence,
        "include_spans": body.include_spans,
        "format_results": body.format_results,
    }
    if body.threshold is not None:
        kwargs["threshold"] = body.threshold
    if body.overlap_policy is not None:
        kwargs["overlap_policy"] = body.overlap_policy
    return kwargs


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
        from gliner2 import AutoExtractor

        if self.name not in MODELS:
            raise ValueError(f"unknown model {self.name!r}")
        self.extractor = AutoExtractor.from_pretrained(MODELS[self.name])

    @modal.method()
    def extract(self, body: ExtractEntitiesRequest) -> Any:
        return self.extractor.extract_entities(
            body.text,
            body.labels,
            **_extract_kwargs(body),
        )


def _dispatch(body: ExtractEntitiesRequest) -> Any:
    return Extractor(name=body.model).extract.remote(body)


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
