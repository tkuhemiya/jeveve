from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from keys import KeyStore, create_key, delete_key, find_by_bearer, list_keys
from schemas import CreatedKey, CreateKeyRequest, ExtractEntitiesRequest, KeyRow

bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "Unauthorized") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_web_app(
    *,
    key_store: KeyStore,
    admin_token: str,
    extract: Callable[[ExtractEntitiesRequest], Any],
) -> FastAPI:
    app = FastAPI(title="GLiNER2.5", version="0.1.0")

    def require_bearer(
        creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> str:
        if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
            raise _unauthorized()
        return creds.credentials

    def require_api_key(token: str = Depends(require_bearer)) -> str:
        if find_by_bearer(key_store.load(), token) is None:
            raise _unauthorized()
        return token

    def require_admin(token: str = Depends(require_bearer)) -> str:
        if not _tokens_match(token, admin_token):
            raise _unauthorized()
        return token

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/extract_entities")
    def extract_entities(
        body: ExtractEntitiesRequest,
        _: str = Depends(require_api_key),
    ) -> Any:
        return extract(body)

    @app.post("/v1/keys", status_code=status.HTTP_201_CREATED)
    def post_key(
        body: CreateKeyRequest = CreateKeyRequest(),
        _: str = Depends(require_admin),
    ) -> CreatedKey:
        return create_key(key_store, name=body.name)

    @app.get("/v1/keys")
    def get_keys(_: str = Depends(require_admin)) -> list[KeyRow]:
        return list_keys(key_store)

    @app.delete("/v1/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_key(key_id: str, _: str = Depends(require_admin)) -> None:
        if not delete_key(key_store, key_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")

    return app


def _tokens_match(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def admin_token_from_env() -> str:
    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        raise RuntimeError("ADMIN_TOKEN is not set")
    return token
