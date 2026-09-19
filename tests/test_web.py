from pathlib import Path

from fastapi.testclient import TestClient

from extractors import ExtractorLoadError
from keys import FileKeyStore, MemoryKeyStore, create_key
from schemas import ExtractEntitiesRequest
from timing import ExtractEnvelope, ExtractTiming
from web import KEY_STORE_UNAVAILABLE, create_web_app

ADMIN = "admin-test-token"


def fake_extract(body: ExtractEntitiesRequest) -> object:
    if not body.format_results:
        return [("Apple", 0.98, 0, 5)]
    return {
        "model": body.model,
        "entities": {"person": ["Tim Cook"], "company": ["Apple"]},
    }


def make_client(store: MemoryKeyStore | None = None) -> tuple[TestClient, MemoryKeyStore]:
    store = store or MemoryKeyStore()
    app = create_web_app(key_store=store, admin_token=ADMIN, extract=fake_extract)
    return TestClient(app), store


def test_health_is_public() -> None:
    client, _ = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_includes_model_readiness_when_provided() -> None:
    store = MemoryKeyStore()
    app = create_web_app(
        key_store=store,
        admin_token=ADMIN,
        extract=fake_extract,
        model_status=lambda: {"small": "ready", "base": "unloaded", "multi": "unloaded"},
    )
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "models": {"small": "ready", "base": "unloaded", "multi": "unloaded"},
    }


def test_extract_load_failure_is_503_with_retry() -> None:
    def broken_extract(body: ExtractEntitiesRequest) -> object:
        raise ExtractorLoadError(body.model, retry_after=7)

    store = MemoryKeyStore()
    created = create_key(store, name="bot")
    app = create_web_app(key_store=store, admin_token=ADMIN, extract=broken_extract)
    response = TestClient(app).post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={"text": "Apple", "labels": ["company"]},
    )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"
    assert "not ready" in response.json()["detail"]
    assert "small" in response.json()["detail"]


def test_extract_requires_bearer() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/extract_entities",
        json={"text": "Apple CEO Tim Cook", "labels": ["person"]},
    )
    assert response.status_code == 401


def test_extract_rejects_unknown_key() -> None:
    client, _ = make_client()
    response = client.post(
        "/v1/extract_entities",
        headers={"Authorization": "Bearer jv_not-a-real-key"},
        json={"text": "Apple CEO Tim Cook", "labels": ["person"]},
    )
    assert response.status_code == 401


def test_extract_rejects_unknown_fields() -> None:
    client, store = make_client()
    created = create_key(store, name="bot")
    response = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={
            "text": "Apple CEO Tim Cook",
            "labels": ["person"],
            "unexpected": True,
        },
    )
    assert response.status_code == 422


def test_extract_defaults_to_small_and_returns_library_object() -> None:
    client, store = make_client()
    created = create_key(store, name="bot")
    response = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={
            "text": "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
            "labels": ["company", "person", "product", "location"],
            "include_confidence": True,
            "include_spans": True,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "model": "small",
        "entities": {"person": ["Tim Cook"], "company": ["Apple"]},
    }


def test_extract_accepts_described_labels_and_model() -> None:
    client, store = make_client()
    created = create_key(store, name=None)
    response = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={
            "model": "multi",
            "text": "Tim Cook",
            "labels": {"person": "A named human"},
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "multi"


def test_empty_labels_are_rejected() -> None:
    client, store = make_client()
    created = create_key(store, name="bot")
    response = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={"text": "hello", "labels": []},
    )
    assert response.status_code == 422


def test_admin_can_create_list_and_delete_keys() -> None:
    client, _ = make_client()
    created = client.post(
        "/v1/keys",
        headers={"Authorization": f"Bearer {ADMIN}"},
        json={"name": "prod-bot"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "prod-bot"
    assert body["id"].startswith("k_")
    assert body["key"].startswith("jv_")

    listed = client.get("/v1/keys", headers={"Authorization": f"Bearer {ADMIN}"})
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["id"] == body["id"]
    assert "hash" not in rows[0]
    assert "key" not in rows[0]

    extract = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {body['key']}"},
        json={"text": "Apple", "labels": ["company"]},
    )
    assert extract.status_code == 200

    deleted = client.delete(
        f"/v1/keys/{body['id']}",
        headers={"Authorization": f"Bearer {ADMIN}"},
    )
    assert deleted.status_code == 204

    extract_again = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {body['key']}"},
        json={"text": "Apple", "labels": ["company"]},
    )
    assert extract_again.status_code == 401


def test_keys_require_admin_token() -> None:
    client, store = make_client()
    api_key = create_key(store, name="bot")
    response = client.get(
        "/v1/keys",
        headers={"Authorization": f"Bearer {api_key.key}"},
    )
    assert response.status_code == 401


def test_corrupt_key_store_returns_503(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text("{not valid json")
    client = TestClient(
        create_web_app(
            key_store=FileKeyStore(path),
            admin_token=ADMIN,
            extract=fake_extract,
        )
    )
    headers = {"Authorization": f"Bearer {ADMIN}"}

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json() == {"detail": KEY_STORE_UNAVAILABLE}

    created = client.post("/v1/keys", headers=headers, json={"name": "x"})
    assert created.status_code == 503
    assert created.json() == {"detail": KEY_STORE_UNAVAILABLE}

    listed = client.get("/v1/keys", headers=headers)
    assert listed.status_code == 503

    deleted = client.delete("/v1/keys/k_missing", headers=headers)
    assert deleted.status_code == 503

    extract = client.post(
        "/v1/extract_entities",
        headers={"Authorization": "Bearer jv_not-a-real-key"},
        json={"text": "Apple", "labels": ["company"]},
    )
    assert extract.status_code == 503


def test_delete_missing_key_is_404() -> None:
    client, _ = make_client()
    response = client.delete(
        "/v1/keys/k_missing",
        headers={"Authorization": f"Bearer {ADMIN}"},
    )
    assert response.status_code == 404


def test_raw_format_results_are_returned_as_json() -> None:
    client, store = make_client()
    created = create_key(store, name="bot")
    response = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={
            "text": "Apple",
            "labels": ["company"],
            "format_results": False,
        },
    )
    assert response.status_code == 200
    assert response.json() == [["Apple", 0.98, 0, 5]]


def test_extract_timing_headers_and_health_starts() -> None:
    def timed_extract(body: ExtractEntitiesRequest) -> object:
        return ExtractEnvelope(
            result={"model": body.model, "entities": {"company": ["Apple"]}},
            timing=ExtractTiming(
                model=body.model,
                load_s=41.2,
                infer_s=0.4,
                wait_s=0.0,
                cold=True,
                extracts=1,
            ),
        )

    store = MemoryKeyStore()
    created = create_key(store, name="bot")
    client = TestClient(
        create_web_app(key_store=store, admin_token=ADMIN, extract=timed_extract)
    )
    response = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={"text": "Apple", "labels": ["company"]},
    )
    assert response.status_code == 200
    assert response.json() == {"model": "small", "entities": {"company": ["Apple"]}}
    assert response.headers["x-gliner-model"] == "small"
    assert response.headers["x-gliner-load-s"] == "41.200"
    assert response.headers["x-gliner-infer-s"] == "0.400"
    assert response.headers["x-gliner-cold"] == "true"
    assert response.headers["x-gliner-slow"] == "true"
    assert float(response.headers["x-gliner-wait-s"]) >= 0.0

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "degraded"
    assert body["starts"]["small"]["load_s"] == 41.2
    assert body["starts"]["small"]["slow"] is True
    assert body["starts"]["small"]["cold"] is True


def test_fast_extract_keeps_health_ok() -> None:
    client, store = make_client()
    created = create_key(store, name="bot")
    extract = client.post(
        "/v1/extract_entities",
        headers={"Authorization": f"Bearer {created.key}"},
        json={"text": "Apple", "labels": ["company"]},
    )
    assert extract.status_code == 200
    assert extract.headers["x-gliner-slow"] == "false"
    health = client.get("/health")
    assert health.json()["status"] == "ok"
    assert health.json()["starts"]["small"]["slow"] is False
