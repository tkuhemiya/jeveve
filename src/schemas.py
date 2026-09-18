from typing import Annotated, Literal, NotRequired, TypedDict, TypeGuard

from pydantic import BaseModel, ConfigDict, Field

type ModelName = Literal["small", "base", "multi"]
type OverlapPolicy = Literal["allow", "nested", "flat", "disallow", "longest"]
type LabelName = Annotated[str, Field(min_length=1)]
type Labels = list[LabelName] | dict[LabelName, str]
type ExtractResult = object
type DeleteOutcome = Literal["deleted", "missing"]


class ExtractorOutOfMemory(Exception):
    """Inference exceeded available memory."""


MODELS: dict[ModelName, str] = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}

_STRICT = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

# GLiNER2.5 encodes a 4096-token window. 50k-char requests OOM-kill CPU inference
# around 32k-36k characters; 8192 stays well under that cliff for English and dense scripts.
TEXT_MAX_LENGTH = 8_192


def is_model_name(value: str) -> TypeGuard[ModelName]:
    return value in MODELS


def hub_id_for(name: str) -> str:
    if not is_model_name(name):
        raise ValueError(f"unknown model {name!r}")
    return MODELS[name]


class ExtractEntitiesRequest(BaseModel):
    model_config = _STRICT

    model: ModelName = "small"
    text: str = Field(min_length=1, max_length=TEXT_MAX_LENGTH)
    labels: Labels = Field(min_length=1)
    include_confidence: bool = False
    include_spans: bool = False
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    overlap_policy: OverlapPolicy | None = None
    format_results: bool = True


class ExtractCall(TypedDict):
    include_confidence: bool
    include_spans: bool
    format_results: bool
    threshold: NotRequired[float]
    overlap_policy: NotRequired[OverlapPolicy]


def extract_call(body: ExtractEntitiesRequest) -> ExtractCall:
    call: ExtractCall = {
        "include_confidence": body.include_confidence,
        "include_spans": body.include_spans,
        "format_results": body.format_results,
    }
    if body.threshold is not None:
        call["threshold"] = body.threshold
    if body.overlap_policy is not None:
        call["overlap_policy"] = body.overlap_policy
    return call


class CreateKeyRequest(BaseModel):
    model_config = _STRICT

    name: str | None = Field(default=None, min_length=1, max_length=128)


class KeyRow(BaseModel):
    model_config = _STRICT

    id: Annotated[str, Field(pattern=r"^k_")]
    name: str | None
    created_at: str


class StoredKey(BaseModel):
    model_config = _STRICT

    id: Annotated[str, Field(pattern=r"^k_")]
    name: str | None
    hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    created_at: str

    def to_row(self) -> KeyRow:
        return KeyRow(id=self.id, name=self.name, created_at=self.created_at)


class CreatedKey(KeyRow):
    key: Annotated[str, Field(pattern=r"^jv_")]


class Health(TypedDict):
    status: Literal["ok"]
