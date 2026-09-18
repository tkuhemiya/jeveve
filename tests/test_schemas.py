import pytest
from pydantic import ValidationError

from schemas import (
    MODELS,
    TEXT_MAX_LENGTH,
    ExtractCall,
    ExtractEntitiesRequest,
    extract_call,
    hub_id_for,
    is_model_name,
)


def test_models_map_to_hub_ids() -> None:
    assert MODELS == {
        "small": "fastino/gliner2.5-small-v1",
        "base": "fastino/gliner2.5-base-v1",
        "multi": "fastino/gliner2.5-multi-v1",
    }
    assert all(is_model_name(name) for name in MODELS)
    assert hub_id_for("multi") == MODELS["multi"]


def test_unknown_hub_id_is_a_type_error_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        hub_id_for("large")


def test_default_model_is_small() -> None:
    body = ExtractEntitiesRequest(text="Apple", labels=["company"])
    assert body.model == "small"


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest.model_validate(
            {"model": "large", "text": "Apple", "labels": ["company"]}
        )


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest.model_validate(
            {"text": "Apple", "labels": ["company"], "foo": 1}
        )


def test_empty_and_blank_labels_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text="Apple", labels=[])
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text="Apple", labels=[""])
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text="Apple", labels={"": "a company"})


def test_text_at_max_length_is_accepted() -> None:
    body = ExtractEntitiesRequest(text="x" * TEXT_MAX_LENGTH, labels=["company"])
    assert len(body.text) == TEXT_MAX_LENGTH


def test_text_above_max_length_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text="x" * (TEXT_MAX_LENGTH + 1), labels=["company"])
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text="x" * 36_000, labels=["company"])


def test_extract_call_omits_unset_options() -> None:
    body = ExtractEntitiesRequest(text="Apple", labels=["company"], threshold=0.4)
    call: ExtractCall = extract_call(body)
    assert call == {
        "include_confidence": False,
        "include_spans": False,
        "format_results": True,
        "threshold": 0.4,
    }
    assert "overlap_policy" not in call
