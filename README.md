# jeveve

GLiNER2.5 entity extraction on Modal. One public URL, bearer keys, scale to zero.

## What's live

Production app `gliner`, web function:

https://tkuhemiya--gliner-web.modal.run

`GET /health` is public. It only means the web process is up and `keys.json` is readable, not that an extractor is warm.

| `model` | Checkpoint | Use |
| --- | --- | --- |
| `small` (default) | `fastino/gliner2.5-small-v1` | English, cheapest/fastest |
| `base` | `fastino/gliner2.5-base-v1` | English, better quality |
| `multi` | `fastino/gliner2.5-multi-v1` | Non-English |

All three are baked into the image. Request one per call with `"model"`. Idle pools scale to zero after 60s. Traffic to `small` does not keep `base` or `multi` warm.

## How to call it

You need two secrets. `ADMIN_TOKEN` lives in Modal secret `gliner-admin` (not git, not GitHub). Use it only to mint and revoke API keys. `API_KEY` is a `jv_...` value returned once from `POST /v1/keys`. That is what extract uses.

```bash
export URL="https://tkuhemiya--gliner-web.modal.run"
export ADMIN_TOKEN="..."   # from Modal secret gliner-admin
```

```bash
curl -sS "$URL/health"
```

Mint a key:

```bash
curl -sS -X POST "$URL/v1/keys" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "prod-bot"}'
```

```json
{"id": "k_...", "name": "prod-bot", "created_at": "...", "key": "jv_..."}
```

Put `key` in `API_KEY`. One key works for every model.

```bash
export API_KEY="jv_..."

curl -sS -X POST "$URL/v1/extract_entities" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "small",
    "text": "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
    "labels": ["company", "person", "product", "location"],
    "include_confidence": true,
    "include_spans": true
  }'
```

Use `"model": "multi"` for non-English text. Labels can be a list or a map of name → description (`{"person": "A named human"}`). Unknown JSON fields are 422. Missing or wrong bearer is 401.

First extract after idle boots the web function, then that model's pool. Key create and delete run on a single web container so concurrent mints cannot overwrite `keys.json`.

List or delete keys with the same admin bearer:

```bash
curl -sS "$URL/v1/keys" -H "Authorization: Bearer $ADMIN_TOKEN"
curl -sS -X DELETE "$URL/v1/keys/$KEY_ID" -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A [Modal](https://modal.com) account

## 1. Install

```bash
uv sync --all-groups
uv run modal setup
```

`modal setup` logs you in. It writes tokens to your machine, not this repo.

## 2. Create the admin secret

Once per Modal workspace. The token must not live in git.

```bash
uv run modal secret create gliner-admin ADMIN_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Save that `ADMIN_TOKEN`. You need it to mint API keys. Code only references the secret by name (`gliner-admin`).

## 3. Serve (dev) or deploy (prod)

```bash
uv run modal serve src/app.py
```

Live-reloads. Prints a public `*.modal.run` URL. Ctrl-C tears it down.

```bash
uv run modal deploy src/app.py
```

Persistent URL. First image build downloads all three checkpoints; later deploys reuse the image.

Pushes to `main` also deploy via `.github/workflows/ci.yml`. Add repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (Modal → Settings → Tokens). The FastAPI `ADMIN_TOKEN` stays in the Modal secret `gliner-admin`. Do not put it in GitHub. Do not use the Modal token id (`ak-...`) as that value.

## Key store recovery

`keys.json` lives on Modal volume `gliner-keys` (mounted at `/keys`). Writes are atomic (temp file + replace). If the file is corrupt, `/health`, key admin routes, and extract auth return 503. There is no API rewrite path. Restore the volume file.

Copy the current file off the volume before you replace it:

```bash
uv run modal volume get gliner-keys keys.json ./keys.json.corrupt
# restore a known-good backup, or write a valid list (`[]` means no API keys)
uv run modal volume put --force gliner-keys ./keys.json keys.json
```

## Local checks

These do not start Modal or download weights:

```bash
uv run pytest
uv run ruff check .
uv run ty check src tests
```

## Demo

Interactive [marimo](https://docs.marimo.io/) notebook that hits the live extract API with `small`, `base`, and `multi` (nested overlap policies, described medical/PII/legal schemas, multilingual + code-switch, a 12k-character earnings call). Dependencies live in the notebook's inline script metadata — uv sandbox, no project install:

```bash
uvx marimo edit --sandbox demo/gliner_limits.py
```

Run it as an app with `uvx marimo run --sandbox demo/gliner_limits.py`. First extract after idle boots the web function, then that model's pool (web timeout is 900s so that wait is allowed). httpx follows Modal's 150s 303 result redirects. Override `GLINER_URL` / `GLINER_API_KEY` if needed. More in [`demo/README.md`](demo/README.md).

## Layout

| Path | What |
| --- | --- |
| `src/app.py` | Modal images, extractor pools, ASGI entry. `modal deploy src/app.py` is apply. |
| `src/web.py` | FastAPI routes and auth |
| `src/keys.py` | SHA-256 key store (`gliner-keys` volume) |
| `src/schemas.py` | Request models |
| `PLAN.md` | Spec (pricing, types, out of scope) |
