from pathlib import Path

import modal

from extractors import (
    ExtractorLoadError,
    is_extractor_startup_failure,
    load_extractor,
    run_timed_extract,
)
from keys import FileKeyStore
from schemas import MODELS, ExtractEntitiesRequest
from timing import ExtractEnvelope, as_envelope
from web import admin_token_from_env, create_web_app

app = modal.App("gliner")

# Extractor @modal.enter loads a 74-287M checkpoint. Cold start is minutes.
# The web function blocks on extract.remote(), so its timeout must outlive
# extractor startup plus one inference. Modal's HTTP layer 303s every 150s;
# that is not a substitute for a long enough function timeout (default 300s).
EXTRACTOR_TIMEOUT = 300
EXTRACTOR_STARTUP_TIMEOUT = 600
WEB_TIMEOUT = EXTRACTOR_STARTUP_TIMEOUT + EXTRACTOR_TIMEOUT
WEB_STARTUP_TIMEOUT = 60

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
    .env({"PYTHONPATH": "/pkg"})
    .add_local_dir(str(SRC_DIR), remote_path="/pkg")
)

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gliner2[local]==2.0.0", "pydantic==2.13.5")
    # Checkpoints ship tokenizer_config extra_special_tokens as a list (transformers 5).
    # gliner2[local] 2.0.0 still caps transformers<5, so lift it after the extra install.
    .run_commands("python -m pip install --upgrade transformers==5.16.1")
    .env({"HF_HOME": "/root/.cache/huggingface", "PYTHONPATH": "/pkg"})
    .run_commands(f'python -c "from gliner2 import AutoExtractor as A; {_PRELOAD}"')
    .add_local_dir(str(SRC_DIR), remote_path="/pkg")
)


@app.cls(
    image=infer_image,
    cpu=2,
    memory=8192,
    min_containers=0,
    scaledown_window=60,
    timeout=EXTRACTOR_TIMEOUT,
    startup_timeout=EXTRACTOR_STARTUP_TIMEOUT,
)
class Extractor:
    name: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        # Sole warm-load path. load_extractor serializes from_pretrained so a
        # cold burst cannot race two checkpoints in one process. Timing is
        # recorded on the cache and reported on the first extract.
        self.extractor = load_extractor(self.name)
        self.extracts = 0

    @modal.method()
    def extract(self, body: ExtractEntitiesRequest) -> ExtractEnvelope:
        if getattr(self, "extractor", None) is None:
            raise ExtractorLoadError(self.name)
        envelope = run_timed_extract(
            body, extracts_before=getattr(self, "extracts", 0)
        )
        self.extracts = envelope.timing.extracts
        return envelope


def _dispatch(body: ExtractEntitiesRequest) -> ExtractEnvelope:
    extractor = Extractor(name=body.model)  # ty: ignore[unknown-argument]
    try:
        raw = extractor.extract.remote(body)
    except ExtractorLoadError:
        raise
    except Exception as exc:
        if is_extractor_startup_failure(exc):
            raise ExtractorLoadError(body.model) from exc
        raise
    if isinstance(raw, ExtractEnvelope):
        return raw
    return as_envelope(raw, model=body.model, wait_s=0.0)


@app.function(
    image=web_image,
    cpu=1,
    memory=1024,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=WEB_TIMEOUT,
    startup_timeout=WEB_STARTUP_TIMEOUT,
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
