# GLiNER2.5 live demo

Marimo notebook against production `POST /v1/extract_entities`. All three checkpoints, limit tests, playground.

Uses uv's marimo integration: the notebook is a Python script with [inline metadata](https://docs.astral.sh/uv/guides/integration/marimo/), and marimo's `--sandbox` flag lets uv install those deps into an isolated env ([marimo + uv](https://docs.marimo.io/guides/package_management/using_uv/)).

## Run

```bash
# editor
uvx marimo edit --sandbox demo/gliner_limits.py

# app (no editor chrome)
uvx marimo run --sandbox demo/gliner_limits.py
```

From this directory:

```bash
uvx marimo edit --sandbox gliner_limits.py
```

Optional overrides:

```bash
export GLINER_URL="https://tkuhemiya--gliner-web.modal.run"
export GLINER_API_KEY="jv_..."
```

The notebook ships a working default key so the catalog runs without extra setup. Rotate it if this file is public.

## What it hammers

| Case | Why it's a limit |
| --- | --- |
| Packed news × `small`/`base`/`multi` | Same English, three encoders in parallel |
| Nested overlap gauntlet | NYPD ⊂ New York City; five `overlap_policy` values |
| Clinical described schema | Invented types (`ejection_fraction`, `route_or_frequency`) |
| Multilingual world tour | 10 languages + ES/EN/ZH/JA code-switch on `multi` |
| PII dump | MRN, SSN, PAN, IP — all synthetic |
| Legal MSA | Parties, fora, GDPR, liability cap |
| 32-label schema blast | Sports + CRISPR + tickers + handles at once |
| Homographs | Apple/Amazon/Jaguar/Turkey |
| ~12k-char earnings call | Encoder window is 4,096 tokens; API cap is 50k chars |
| `small` vs `multi` on Japanese/Chinese/Arabic | Negative control — English `small` should fail |

## Cold start

`GET /health` only means the web process is up. The first extract for a model boots that pool (often >150s). Modal then 303s to a result URL; the client uses `httpx` with `follow_redirects=True` and retries `503` + `Retry-After`. Click **Ignite small + base + multi** once before running the catalog.
