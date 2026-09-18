# jeveve

GLiNER2.5 entity extraction on Modal. One public URL, bearer keys, scale to zero.

Models: `small` (default, English), `base` (English), `multi` (non-English).

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

Set the URL:

```bash
export URL="https://<your-app>.modal.run"
export ADMIN_TOKEN="..."   # the value you created above
```

`GET $URL/health` is unauthenticated. It means the web process is up, not that an extractor is warm. `/docs`, `/redoc`, and `/openapi.json` are disabled so the public URL does not advertise admin routes.

## 4. Mint an API key

```bash
curl -sS -X POST "$URL/v1/keys" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "prod-bot"}'
```

```json
{"id": "k_...", "name": "prod-bot", "created_at": "...", "key": "jv_..."}
```

`key` is returned once. Put it in `API_KEY`. One key can call every model. Create and delete run on a single web container so concurrent mints cannot overwrite `keys.json`. Extract still fans out to per-model pools.

```bash
curl -sS "$URL/v1/keys" -H "Authorization: Bearer $ADMIN_TOKEN"
curl -sS -X DELETE "$URL/v1/keys/$KEY_ID" -H "Authorization: Bearer $ADMIN_TOKEN"
```

## 5. Extract

```bash
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

Unknown JSON fields return 422. Missing or wrong bearer returns 401.

Use `"model": "multi"` for non-English text. Labels can also be a map of name → description:

```json
{"person": "A named human", "company": "An organization"}
```

Cold extract boots the web function if needed, then the extractor pool for that `model`. `small` traffic does not keep `multi` warm. Idle containers scale to zero after 60s.

## Local checks

These do not start Modal or download weights:

```bash
uv run pytest
uv run ruff check .
uv run ty check src tests
```

## Layout

| Path | What |
| --- | --- |
| `src/app.py` | Modal images, extractor pools, ASGI entry. `modal deploy src/app.py` is apply. |
| `src/web.py` | FastAPI routes and auth |
| `src/keys.py` | SHA-256 key store (`gliner-keys` volume) |
| `src/schemas.py` | Request models |
| `PLAN.md` | Spec (pricing, types, out of scope) |
