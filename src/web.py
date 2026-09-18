from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import Body, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from keys import KeyStore, create_key, delete_key, find_by_bearer, list_keys, secrets_equal
from schemas import (
    CreatedKey,
    CreateKeyRequest,
    ExtractEntitiesRequest,
    ExtractorOutOfMemory,
    Health,
    KeyRow,
)

bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_web_app(
    *,
    key_store: KeyStore,
    admin_token: str,
    extract: Callable[[ExtractEntitiesRequest], object],
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
        if not secrets_equal(token, admin_token):
            raise _unauthorized()
        return token

    @app.get("/health")
    def health() -> Health:
        return {"status": "ok"}

    @app.post("/v1/extract_entities", response_model=None)
    def extract_entities(
        body: ExtractEntitiesRequest,
        _: str = Depends(require_api_key),
    ) -> object:
        try:
            return extract(body)
        except (MemoryError, ExtractorOutOfMemory):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Extractor ran out of memory",
            ) from None

    @app.post("/v1/keys", status_code=status.HTTP_201_CREATED)
    def post_key(
        body: CreateKeyRequest = Body(default_factory=CreateKeyRequest),
        _: str = Depends(require_admin),
    ) -> CreatedKey:
        return create_key(key_store, name=body.name)

    @app.get("/v1/keys")
    def get_keys(_: str = Depends(require_admin)) -> list[KeyRow]:
        return list_keys(key_store)

    @app.delete("/v1/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_key(key_id: str, _: str = Depends(require_admin)) -> None:
        match delete_key(key_store, key_id):
            case "deleted":
                return
            case "missing":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Key not found",
                )

    return app


def admin_token_from_env() -> str:
    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        raise RuntimeError("ADMIN_TOKEN is not set")
    return token
