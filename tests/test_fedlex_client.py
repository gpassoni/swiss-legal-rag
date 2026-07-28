from unittest.mock import patch

import pytest

from src.ingestion.fedlex_client import ActSummary, FedlexClient


def _binding(value):
    return {"value": value}


@pytest.fixture
def client():
    return FedlexClient()


def test_list_consolidated_acts_by_prefix_parses_bindings(client):
    fake_bindings = [
        {
            "consolidationAbstract": _binding("https://fedlex.example/eli/cc/1"),
            "srNotation": _binding("640.11"),
            "title": _binding("Bundesgesetz über die direkte Bundessteuer"),
            "dateApplicability": _binding("1995-01-01"),
        }
    ]
    with patch.object(FedlexClient, "_run_query", return_value=fake_bindings) as run_query:
        acts = client.list_consolidated_acts_by_prefix("640", language="de")

    assert run_query.call_count == 1
    assert acts == [
        ActSummary(
            act_uri="https://fedlex.example/eli/cc/1",
            systematic_number="640.11",
            title="Bundesgesetz über die direkte Bundessteuer",
            language="de",
            valid_from="1995-01-01",
            valid_to=None,
        )
    ]


def test_list_consolidated_acts_rejects_unsupported_language(client):
    with pytest.raises(ValueError):
        client.list_consolidated_acts_by_prefix("640", language="en")


def test_fetch_act_returns_document_with_metadata(client):
    fake_bindings = [
        {
            "title": _binding("Loi fédérale sur l'impôt fédéral direct"),
            "srNotation": _binding("642.11"),
            "dateApplicability": _binding("1995-01-01"),
        }
    ]
    with patch.object(FedlexClient, "_run_query", return_value=fake_bindings):
        doc = client.fetch_act("https://fedlex.example/eli/cc/2", language="fr")

    assert doc.systematic_number == "642.11"
    assert doc.title == "Loi fédérale sur l'impôt fédéral direct"
    assert doc.language == "fr"
    assert doc.source_url == "https://fedlex.example/eli/cc/2"
    assert doc.raw_text == ""  # include_text defaults to False


def test_fetch_act_raises_lookup_error_when_no_results(client):
    with patch.object(FedlexClient, "_run_query", return_value=[]):
        with pytest.raises(LookupError):
            client.fetch_act("https://fedlex.example/eli/cc/missing")


def test_fetch_act_with_include_text_calls_html_fetch(client):
    fake_bindings = [
        {"title": _binding("Some Act"), "srNotation": _binding("640.1")}
    ]
    with patch.object(FedlexClient, "_run_query", return_value=fake_bindings), patch.object(
        FedlexClient, "fetch_act_html_text", return_value="Art. 1 Some text"
    ) as fetch_html:
        doc = client.fetch_act("https://fedlex.example/eli/cc/3", include_text=True)

    fetch_html.assert_called_once_with("https://fedlex.example/eli/cc/3", language="de")
    assert doc.raw_text == "Art. 1 Some text"


def test_fetch_act_file_url_resolves_filestore_uri(client):
    fake_bindings = [
        {"fileUrl": _binding("https://fedlex.data.admin.ch/filestore/.../act-de-html.html")}
    ]
    with patch.object(FedlexClient, "_run_query", return_value=fake_bindings) as run_query:
        file_url = client.fetch_act_file_url("https://fedlex.example/eli/cc/4", language="de")

    assert run_query.call_count == 1
    assert file_url == "https://fedlex.data.admin.ch/filestore/.../act-de-html.html"


def test_fetch_act_file_url_raises_lookup_error_when_no_manifestation(client):
    with patch.object(FedlexClient, "_run_query", return_value=[]):
        with pytest.raises(LookupError):
            client.fetch_act_file_url("https://fedlex.example/eli/cc/missing")


def test_fetch_act_html_text_fetches_resolved_file_url_not_act_uri(client):
    """The bug this guards against: GETting `act_uri` directly returns Fedlex's JS-app
    shell ("please enable JavaScript"), not the article text. The file URL resolved via
    SPARQL is a static filestore HTML file and must be what gets GETted instead."""
    import httpx

    act_uri = "https://fedlex.data.admin.ch/eli/cc/1993/2940_2940_2940"
    file_url = "https://fedlex.data.admin.ch/filestore/.../act-de-html.html"
    fake_response = httpx.Response(
        200,
        text="<html><body><article id='art_1'><h6><b>Art. 1</b></h6>"
        "<p>Some article text.</p></article></body></html>",
        request=httpx.Request("GET", file_url),
    )
    with patch.object(
        FedlexClient, "fetch_act_file_url", return_value=file_url
    ) as fetch_url, patch("src.ingestion.fedlex_client.httpx.get", return_value=fake_response) as get:
        text = client.fetch_act_html_text(act_uri, language="de")

    fetch_url.assert_called_once_with(act_uri, language="de")
    assert get.call_args.args[0] == file_url
    assert get.call_args.args[0] != act_uri
    assert "Art. 1" in text
    assert "Some article text." in text
