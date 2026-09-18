from pathlib import Path

from keys import FileKeyStore, create_key, delete_key, find_by_bearer, hash_key, list_keys


def test_file_store_roundtrip(tmp_path: Path) -> None:
    store = FileKeyStore(tmp_path / "keys.json")
    created = create_key(store, name="prod-bot")
    assert created.key.startswith("jv_")
    assert hash_key(created.key) != created.key

    rows = list_keys(store)
    assert len(rows) == 1
    assert rows[0].id == created.id
    assert rows[0].name == "prod-bot"

    reloaded = FileKeyStore(tmp_path / "keys.json")
    found = find_by_bearer(reloaded.load(), created.key)
    assert found is not None
    assert found.id == created.id
    assert "hash" in found.model_dump()

    assert delete_key(reloaded, created.id) is True
    assert find_by_bearer(FileKeyStore(tmp_path / "keys.json").load(), created.key) is None
