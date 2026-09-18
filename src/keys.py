from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from schemas import CreatedKey, KeyRow, StoredKey

KEY_PREFIX = "jv_"
ID_PREFIX = "k_"


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def mint_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


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
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [StoredKey.model_validate(item) for item in raw]

    def save(self, keys: list[StoredKey]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            [key.model_dump() for key in keys],
            indent=2,
            sort_keys=True,
        )
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
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
    digest = hash_key(bearer)
    for key in keys:
        if hmac.compare_digest(key.hash, digest):
            return key
    return None


def create_key(store: KeyStore, *, name: str | None) -> CreatedKey:
    keys = store.load()
    raw = mint_api_key()
    record = StoredKey(
        id=mint_key_id(),
        name=name,
        hash=hash_key(raw),
        created_at=utc_now(),
    )
    keys.append(record)
    store.save(keys)
    return CreatedKey(
        id=record.id,
        name=record.name,
        created_at=record.created_at,
        key=raw,
    )


def list_keys(store: KeyStore) -> list[KeyRow]:
    return [
        KeyRow(id=key.id, name=key.name, created_at=key.created_at) for key in store.load()
    ]


def delete_key(store: KeyStore, key_id: str) -> bool:
    keys = store.load()
    kept = [key for key in keys if key.id != key_id]
    if len(kept) == len(keys):
        return False
    store.save(kept)
    return True
