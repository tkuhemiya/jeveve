import pytest
from pydantic import ValidationError

from schemas import MODELS, ExtractEntitiesRequest


def test_models_map_to_hub_ids() -> None:
    assert MODELS == {
        "small": "fastino/gliner2.5-small-v1",
        "base": "fastino/gliner2.5-base-v1",
        "multi": "fastino/gliner2.5-multi-v1",
    }


def test_default_model_is_small() -> None:
    body = ExtractEntitiesRequest(text="Apple", labels=["company"])
    assert body.model == "small"


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(model="large", text="Apple", labels=["company"])


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest.model_validate(
            {"text": "Apple", "labels": ["company"], "foo": 1}
        )
