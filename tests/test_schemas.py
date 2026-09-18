import pytest
from pydantic import ValidationError

from schemas import (
    MODELS,
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
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text="Apple", labels=["\u200b"])
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text="Apple", labels={"\ufeff": "a company"})


@pytest.mark.parametrize(
    "text",
    [
        "   ",
        "\t",
        "\n",
        "\u00a0",  # NBSP
        "\u200b",  # ZWSP
        "\ufeff",  # BOM
        "\u200c",  # ZWNJ
        "\u200d",  # ZWJ
        "\u200e",  # LRM
        "\u200f",  # RLM
        "\u2060",  # word joiner
        "\u200b\ufeff \t\u00a0",
    ],
)
def test_invisible_or_blank_text_is_rejected(text: str) -> None:
    with pytest.raises(ValidationError):
        ExtractEntitiesRequest(text=text, labels=["company"])


def test_text_strips_invisible_affixes_and_keeps_visible_content() -> None:
    body = ExtractEntitiesRequest(text="\u200b Apple \ufeff", labels=["company"])
    assert body.text == "Apple"


def test_text_is_nfc_normalized() -> None:
    body = ExtractEntitiesRequest(text="Cafe\u0301", labels=["company"])
    assert body.text == "Caf\u00e9"


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
