# Plan: GLiNER2.5 on Modal

One Modal app. `POST /v1/extract_entities` is `extract_entities` plus a `model` field. Bearer keys. Scale to zero.

Out of scope: `classify_text`, relations, JointIE, records, `extract_entities_long`.

## Models

Load with `AutoExtractor.from_pretrained`. `GLiNER2.from_pretrained` is the span loader and will fail on these checkpoints.

| `model` | Hub id | Params | Encoder | Language |
| --- | --- | --- | --- | --- |
| `small` | `fastino/gliner2.5-small-v1` | 74M | DeBERTa-v3-xsmall | English |
| `base` | `fastino/gliner2.5-base-v1` | 194M | DeBERTa-v3-base | English |
| `multi` | `fastino/gliner2.5-multi-v1` | 287M | mDeBERTa-v3-base | Multilingual |

Default `model` is `small`. Use `multi` for non-English. Apache 2.0.

One process loads one checkpoint. A parametrized Modal class (`name: ModelName = modal.parameter()`) gives a separate scale-to-zero pool per `small` / `base` / `multi`.

## Pricing

[modal.com/pricing](https://modal.com/pricing), billed per second while the container exists.

| Resource | Rate |
| --- | --- |
| CPU (physical core = 2 vCPU) | $0.0000131 / core / s |
| Memory | $0.00000222 / GiB / s |
| Volumes | $0.09 / GiB / month, 1 TiB free |
| T4 | $0.000164 / s (~$0.59 / hr) |

Starter: $0/month, $30 compute credit, 100 containers.

Extractor spec is 2 CPU cores, 8 GiB (needed for `multi`, extra for `small`). That is ~$0.000044 / s (~$0.16 / hr). A T4 is ~$0.59 / hr for the same wall time. Stay on CPU until `base` or `multi` latency is actually too high.

`min_containers=0`. `scaledown_window=60`. Idle after that is unpaid. A 24/7 warm extractor is ~$115/month and eats the Starter credit. `small` traffic does not keep a `multi` pool up.

Billable time includes boot plus the scaledown window.

| Pattern | Extractor cost |
| --- | --- |
| 10k `small` req/month × ~0.5 s | ~$0.22 |
| 10k `multi` req/month × ~2 s | ~$0.88 |
| 1 warm extractor, whole month | ~$115 |

## Layout

`src/app.py` is the infra. Images, CPU, memory, `scaledown_window`, Volume, Secret refs, and the ASGI app are declared in Python. `modal deploy src/app.py` is apply. No dashboard click-ops, no Terraform. Modal has no TF provider; the SDK *is* the IaC.

Pin package versions in `pip_install`. Rebuilds should not float.

Web function (`@modal.asgi_app`). FastAPI. Auth, key CRUD. No PyTorch. ~1 GiB CPU. Public URL.

Extractor class. `name: ModelName = modal.parameter()`. `@modal.enter` loads that Hub id. `@modal.method` takes the same Pydantic body the HTTP route already validated. 2 CPU, 8 GiB, `max_inputs=1`.

```python
keys_vol = modal.Volume.from_name("gliner-keys", create_if_missing=True)
admin = modal.Secret.from_name("gliner-admin")

Extractor(name=body.model).extract.remote(body)
```

`web_image`: Debian slim, pinned `fastapi[standard]` (pulls Pydantic v2).

`infer_image`: Debian slim, pinned `gliner2[local]` and `pydantic`. Hub downloads in the image build so cold start does not hit the network. Pydantic is on this image because `.remote` pickle-unpickles the request model here.

```python
MODELS: dict[ModelName, str] = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gliner2[local]==X", "pydantic==2.Y.Z")
    .run_commands(
        "python -c \""
        "from gliner2 import AutoExtractor as A;"
        "A.from_pretrained('fastino/gliner2.5-small-v1');"
        "A.from_pretrained('fastino/gliner2.5-base-v1');"
        "A.from_pretrained('fastino/gliner2.5-multi-v1')\""
    )
)
```

Unused weights stay on disk. One checkpoint in RAM. CPU, fp32.

Do not store keys in a Modal Dict. Entries expire after 7 days with no reads. SHA-256 hashes go in `/keys.json` on Volume `gliner-keys`, mounted on the web function.

## Types

Modal's own FastAPI example is a Pydantic body. `.remote(*args)` cloudpickles; Pydantic models pickle. One `BaseModel` is the HTTP schema *and* the extractor argument. No `dict`, no `Any`.

Treat the boundary like serde: `extra="forbid"`, `Literal` unions, `Field` constraints. FastAPI turns validation errors into 422.

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

type ModelName = Literal["small", "base", "multi"]
type OverlapPolicy = Literal["allow", "nested", "flat", "disallow", "longest"]
type Labels = list[str] | dict[str, str]

class ExtractEntitiesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelName = "small"
    text: str = Field(min_length=1, max_length=50_000)
    labels: Labels
    include_confidence: bool = False
    include_spans: bool = False
    threshold: float | None = None
    overlap_policy: OverlapPolicy | None = None
    format_results: bool = True
```

`MODELS: dict[ModelName, str]` so a bad name is a type error, not a KeyError in prod.

Extract response stays the library object. Do not invent a response model until GLiNER2.5's return shape is pinned.

Key routes get the same treatment (`extra="forbid"`, typed create body and list rows). Hash is `str`, not a loose dict value.

## HTTP

`modal deploy` URL. Extract requires `Authorization: Bearer`.

`POST /v1/extract_entities` body is `ExtractEntitiesRequest`. Unknown fields 422. Return the library object unchanged.

```bash
curl -sS -X POST "$URL/v1/extract_entities" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "multi",
    "text": "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
    "labels": ["company", "person", "product", "location"],
    "include_confidence": true,
    "include_spans": true
  }'
```

`GET /health`. No auth. Means the web process is up, not that an extractor pool is warm.

## Keys

Minting a key must not boot PyTorch. Admin routes live on the web function.

`modal secret create` is the one CLI exception. The token must not live in git. Code only `from_name`s it.

```bash
modal secret create gliner-admin ADMIN_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

`keys.json` is a Pydantic list of `{id, name, hash, created_at}`. Hash is SHA-256 of the raw key. Raw key is not stored.

`POST /v1/keys` with `Authorization: Bearer $ADMIN_TOKEN`, optional `{"name": "prod-bot"}`.

```json
{"id": "k_...", "name": "prod-bot", "key": "jv_..."}
```

`key` is returned once. After that, `Authorization: Bearer jv_...` on extract. One key can call any model.

`GET /v1/keys` (admin): id, name, created_at.

`DELETE /v1/keys/{id}` (admin).

Extract hashes the bearer token and looks it up. Miss or mismatch is 401.

## Run

Cold extract: web boot if needed, then extractor boot for that `model`. Weights come from `infer_image`.

```bash
uv sync --all-groups && uv run modal setup
uv run modal serve src/app.py    # mint a key, POST once per model
uv run modal deploy src/app.py
```
