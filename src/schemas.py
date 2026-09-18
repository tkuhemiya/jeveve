from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

type ModelName = Literal["small", "base", "multi"]
type OverlapPolicy = Literal["allow", "nested", "flat", "disallow", "longest"]
type Labels = list[str] | dict[str, str]

MODELS: dict[ModelName, str] = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}


class ExtractEntitiesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelName = "small"
    text: str = Field(min_length=1, max_length=50_000)
    labels: Labels
    include_confidence: bool = False
    include_spans: bool = False
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    overlap_policy: OverlapPolicy | None = None
    format_results: bool = True

    @field_validator("labels")
    @classmethod
    def labels_must_be_non_empty(cls, value: Labels) -> Labels:
        if len(value) < 1:
            raise ValueError("labels must not be empty")
        if isinstance(value, list) and any(not label for label in value):
            raise ValueError("label names must be non-empty")
        if isinstance(value, dict) and any(not name for name in value):
            raise ValueError("label names must be non-empty")
        return value


class CreateKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=128)


class StoredKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None
    hash: str
    created_at: str


class KeyRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None
    created_at: str


class CreatedKey(KeyRow):
    key: str
