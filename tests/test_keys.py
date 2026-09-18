from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from keys import (
    FileKeyStore,
    KeyStoreCorrupt,
    MemoryKeyStore,
    create_key,
    delete_key,
    find_by_bearer,
    hash_secret,
    list_keys,
)
from schemas import StoredKey


def test_file_store_roundtrip(tmp_path: Path) -> None:
    store = FileKeyStore(tmp_path / "keys.json")
    created = create_key(store, name="prod-bot")
    assert created.key.startswith("jv_")
    assert hash_secret(created.key) != created.key

    rows = list_keys(store)
    assert len(rows) == 1
    assert rows[0].id == created.id
    assert rows[0].name == "prod-bot"
    assert "hash" not in rows[0].model_dump()
    assert "key" not in rows[0].model_dump()

    reloaded = FileKeyStore(tmp_path / "keys.json")
    found = find_by_bearer(reloaded.load(), created.key)
    assert found is not None
    assert found.id == created.id
    assert found.hash == hash_secret(created.key)
    assert created.key not in (tmp_path / "keys.json").read_text()

    assert delete_key(reloaded, created.id) == "deleted"
    assert find_by_bearer(FileKeyStore(tmp_path / "keys.json").load(), created.key) is None


def test_delete_unknown_key_is_missing(tmp_path: Path) -> None:
    store = FileKeyStore(tmp_path / "keys.json")
    assert delete_key(store, "k_missing") == "missing"


def test_file_store_rejects_invalid_persisted_rows(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text('[{"id": "nope", "name": null, "hash": "abc", "created_at": "now"}]\n')
    with pytest.raises(KeyStoreCorrupt) as excinfo:
        FileKeyStore(path).load()
    assert excinfo.value.path == path
    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_file_store_rejects_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text("{not valid json")
    with pytest.raises(KeyStoreCorrupt) as excinfo:
        FileKeyStore(path).load()
    assert excinfo.value.path == path
    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_stored_key_cannot_hold_a_raw_secret() -> None:
    with pytest.raises(ValidationError):
        StoredKey(
            id="k_abc",
            name=None,
            hash="jv_this-is-not-a-sha256",
            created_at="2026-01-01T00:00:00+00:00",
        )


def test_concurrent_creates_keep_every_key() -> None:
    store = MemoryKeyStore()
    workers = 32

    def mint(index: int):
        return create_key(store, name=f"bot-{index}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        created = list(pool.map(mint, range(workers)))
    rows = list_keys(store)
    assert len(rows) == workers
    assert {row.id for row in rows} == {item.id for item in created}
