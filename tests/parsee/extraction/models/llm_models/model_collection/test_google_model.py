import pytest

from parsee import google_config
from parsee.extraction.models.llm_models.model_collection.google_model import GoogleModel


@pytest.mark.parametrize(
    ("location", "expected_base_url"),
    [
        ("eu", "https://aiplatform.eu.rep.googleapis.com"),
        ("us", "https://aiplatform.us.rep.googleapis.com"),
    ],
)
def test_multi_region_uses_jurisdictional_endpoint(monkeypatch, location, expected_base_url):
    client_options = {}

    def create_client(**kwargs):
        client_options.update(kwargs)
        return object()

    monkeypatch.setattr(
        "parsee.extraction.models.llm_models.model_collection.google_model.tiktoken.get_encoding",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "parsee.extraction.models.llm_models.model_collection.google_model.genai.Client",
        create_client,
    )

    GoogleModel(google_config("gemini-3.7-flash", "project", location))

    assert client_options["location"] == location
    assert client_options["http_options"].base_url == expected_base_url


def test_single_region_uses_default_endpoint(monkeypatch):
    client_options = {}

    def create_client(**kwargs):
        client_options.update(kwargs)
        return object()

    monkeypatch.setattr(
        "parsee.extraction.models.llm_models.model_collection.google_model.tiktoken.get_encoding",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "parsee.extraction.models.llm_models.model_collection.google_model.genai.Client",
        create_client,
    )

    GoogleModel(google_config("gemini-2.5-flash", "project", "europe-west1"))

    assert client_options["location"] == "europe-west1"
    assert client_options["http_options"] is None
