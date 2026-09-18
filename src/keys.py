from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NewType, Protocol

from pydantic import TypeAdapter

from schemas import CreatedKey, DeleteOutcome, KeyRow, StoredKey

ApiKey = NewType("ApiKey", str)
KeyHash = NewType("KeyHash", str)

KEY_PREFIX = "jv_"
ID_PREFIX = "k_"
STORED_KEYS = TypeAdapter(list[StoredKey])


def hash_secret(raw: str) -> KeyHash:
    return KeyHash(hashlib.sha256(raw.encode("utf-8")).hexdigest())


def secrets_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(hash_secret(left), hash_secret(right))


def mint_api_key() -> ApiKey:
    return ApiKey(KEY_PREFIX + secrets.token_urlsafe(32))


def mint_key_id() -> str:
    return ID_PREFIX + secrets.token_urlsafe(12)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class KeyStore(Protocol):
    def load(self) -> list[StoredKey]: ...

    def save(self, keys: list[StoredKey]) -> None: ...


class FileKeyStore:
    def __init__(
        self,
        path: Path,
        *,
        reload: Callable[[], None] | None = None,
        commit: Callable[[], None] | None = None,
    ) -> None:
        self.path = path
        self._reload = reload
        self._commit = commit

    def load(self) -> list[StoredKey]:
        if self._reload is not None:
            self._reload()
        if not self.path.exists():
            return []
        return STORED_KEYS.validate_json(self.path.read_bytes())

    def save(self, keys: list[StoredKey]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(STORED_KEYS.dump_json(keys, indent=2) + b"\n")
        tmp.replace(self.path)
        if self._commit is not None:
            self._commit()


class MemoryKeyStore:
    def __init__(self, keys: list[StoredKey] | None = None) -> None:
        self._keys = list(keys or [])

    def load(self) -> list[StoredKey]:
        return list(self._keys)

    def save(self, keys: list[StoredKey]) -> None:
        self._keys = list(keys)


def find_by_bearer(keys: list[StoredKey], bearer: str) -> StoredKey | None:
    digest = hash_secret(bearer)
    for key in keys:
        if hmac.compare_digest(key.hash, digest):
            return key
    return None


def create_key(store: KeyStore, *, name: str | None) -> CreatedKey:
    raw = mint_api_key()
    record = StoredKey(
        id=mint_key_id(),
        name=name,
        hash=hash_secret(raw),
        created_at=utc_now(),
    )
    keys = store.load()
    keys.append(record)
    store.save(keys)
    return CreatedKey(
        id=record.id,
        name=record.name,
        created_at=record.created_at,
        key=raw,
    )


def list_keys(store: KeyStore) -> list[KeyRow]:
    return [key.to_row() for key in store.load()]


def delete_key(store: KeyStore, key_id: str) -> DeleteOutcome:
    keys = store.load()
    kept = [key for key in keys if key.id != key_id]
    if len(kept) == len(keys):
        return "missing"
    store.save(kept)
    return "deleted"
