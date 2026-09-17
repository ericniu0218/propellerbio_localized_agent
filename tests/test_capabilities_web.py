"""web.fetch and web.search: the allowlist, the proxy, and the bytes on disk.

Everything runs against ``httpx.MockTransport``. No test here opens a socket,
and the one test that checks the egress proxy checks how the client is *built*
rather than where it connects - which is the only way to assert it without
standing up a proxy.
"""

import hashlib
from pathlib import Path

import httpx
import pytest

from bedrock.agents.capabilities.web import (
    CAPABILITIES,
    WebConfig,
    WebFetch,
    WebSearch,
)
from bedrock.agents.contract import Risk


@pytest.fixture
def config(tmp_path: Path) -> WebConfig:
    return WebConfig(
        proxy="http://egress.internal:3128",
        allowed_domains=("fda.gov", "clinicaltrials.gov"),
        search_url="https://searx.internal",
        store_dir=tmp_path / "web",
    )


def responder(handler):
    return httpx.MockTransport(handler)


def page(text: str = "Guidance text.", content_type: str = "text/plain"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=text, headers={"content-type": content_type})

    return responder(handler)


# ---- What the roster finds ---------------------------------------------------


def test_both_web_capabilities_are_read_only():
    assert {c.name: c.risk for c in CAPABILITIES} == {
        "web.fetch": Risk.READ_ONLY,
        "web.search": Risk.READ_ONLY,
    }


# ---- The allowlist -----------------------------------------------------------


async def test_a_domain_that_is_not_on_the_list_is_refused(config: WebConfig):
    outcome = await WebFetch(config).run({"url": "https://example.com/guidance"})
    assert "not in the allowed list" in (outcome.failed or "")


async def test_a_lookalike_domain_does_not_pass_the_suffix_check(config: WebConfig):
    outcome = await WebFetch(config).run({"url": "https://notfda.gov/x"})
    assert "not in the allowed list" in (outcome.failed or "")


async def test_a_subdomain_of_an_allowed_domain_is_allowed(config: WebConfig):
    fetch = WebFetch(config, transport=page())
    outcome = await fetch.run({"url": "https://www.fda.gov/guidance"})
    assert outcome.failed is None


async def test_only_http_and_https_are_addresses(config: WebConfig):
    outcome = await WebFetch(config).run({"url": "file:///etc/passwd"})
    assert "not an http or https address" in (outcome.failed or "")


async def test_credentials_in_an_address_are_refused(config: WebConfig):
    outcome = await WebFetch(config).run({"url": "https://user:pw@fda.gov/x"})
    assert "username or password" in (outcome.failed or "")


async def test_an_empty_allowlist_allows_nothing(tmp_path: Path):
    config = WebConfig(proxy="http://egress:3128", store_dir=tmp_path / "web")
    outcome = await WebFetch(config).run({"url": "https://fda.gov/x"})
    assert "No web addresses are allowed yet" in (outcome.failed or "")


# ---- The proxy ---------------------------------------------------------------


async def test_without_a_proxy_nothing_is_fetched_rather_than_fetched_directly(
    tmp_path: Path,
):
    config = WebConfig(allowed_domains=("fda.gov",), store_dir=tmp_path / "web")
    outcome = await WebFetch(config).run({"url": "https://fda.gov/guidance"})
    assert "No egress proxy is configured" in (outcome.failed or "")


def test_the_client_is_built_with_the_configured_proxy(config: WebConfig):
    options = WebFetch(config).client_options(proxy=config.proxy)
    assert options["proxy"] == "http://egress.internal:3128"
    assert options["follow_redirects"] is False


def test_search_does_not_go_through_the_egress_proxy_unless_asked(config: WebConfig):
    # The SearxNG instance is on the inside; sending it out through the egress
    # proxy and back in is not the default.
    assert config.search_via_proxy is False


# ---- Fetching ----------------------------------------------------------------


async def test_a_fetch_stores_the_bytes_hashes_them_and_shows_the_text(config: WebConfig):
    fetch = WebFetch(config, transport=page("Guidance on withdrawal rates."))
    outcome = await fetch.run({"url": "https://fda.gov/guidance/withdrawal"})

    assert outcome.failed is None
    stored = Path(outcome.artifacts[0])
    body = b"Guidance on withdrawal rates."
    assert stored.read_bytes() == body
    assert hashlib.sha256(body).hexdigest() in outcome.completed[0]
    assert "Guidance on withdrawal rates." in outcome.completed[-1]


async def test_a_stored_page_is_private_to_its_owner(config: WebConfig):
    fetch = WebFetch(config, transport=page())
    outcome = await fetch.run({"url": "https://fda.gov/guidance"})
    assert (Path(outcome.artifacts[0]).stat().st_mode & 0o777) == 0o600


async def test_a_redirect_is_reported_rather_than_followed(config: WebConfig):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/elsewhere"})

    outcome = await WebFetch(config, transport=responder(handler)).run(
        {"url": "https://fda.gov/old"}
    )
    assert "redirected to https://example.com/elsewhere" in (outcome.failed or "")
    assert outcome.artifacts == ()


async def test_a_page_larger_than_the_limit_is_not_read(tmp_path: Path):
    config = WebConfig(
        proxy="http://egress:3128",
        allowed_domains=("fda.gov",),
        store_dir=tmp_path / "web",
        max_bytes=100,
    )
    fetch = WebFetch(config, transport=page("y" * 500))
    outcome = await fetch.run({"url": "https://fda.gov/big"})
    assert "larger than" in (outcome.failed or "")
    assert outcome.artifacts == ()


async def test_an_http_error_is_reported_in_plain_words(config: WebConfig):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    outcome = await WebFetch(config, transport=responder(handler)).run(
        {"url": "https://fda.gov/down"}
    )
    assert "answered HTTP 503" in (outcome.failed or "")


async def test_binary_content_is_stored_but_not_pasted_into_the_result(config: WebConfig):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"%PDF-1.7\x00\x01", headers={"content-type": "application/pdf"}
        )

    outcome = await WebFetch(config, transport=responder(handler)).run(
        {"url": "https://fda.gov/guidance.pdf"}
    )
    assert outcome.failed is None
    assert Path(outcome.artifacts[0]).read_bytes() == b"%PDF-1.7\x00\x01"
    assert not any("PDF-1.7" in line for line in outcome.completed[2:])


# ---- Searching ---------------------------------------------------------------


def searx(results: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json={"results": results})

    return httpx.MockTransport(handler)


async def test_search_lists_what_the_company_instance_returned(config: WebConfig):
    transport = searx(
        [
            {
                "title": "Withdrawal rates",
                "url": "https://fda.gov/a",
                "content": "A summary.",
            }
        ]
    )
    outcome = await WebSearch(config, transport=transport).run({"query": "withdrawal"})
    assert outcome.failed is None
    assert "Withdrawal rates - https://fda.gov/a" in outcome.completed[1]


async def test_search_refuses_when_no_company_instance_is_configured(tmp_path: Path):
    config = WebConfig(proxy="http://egress:3128", store_dir=tmp_path / "web")
    outcome = await WebSearch(config).run({"query": "withdrawal"})
    assert "No search service is configured" in (outcome.failed or "")
    assert "commercial search API" in (outcome.failed or "")


async def test_search_needs_a_query(config: WebConfig):
    outcome = await WebSearch(config).run({})
    assert "needs a `query`" in (outcome.failed or "")


async def test_search_explains_a_non_json_answer(config: WebConfig):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    outcome = await WebSearch(config, transport=responder(handler)).run({"query": "x"})
    assert "did not answer with JSON" in (outcome.failed or "")


async def test_search_says_when_nothing_was_found(config: WebConfig):
    outcome = await WebSearch(config, transport=searx([])).run({"query": "unobtainium"})
    assert outcome.failed is None
    assert "Nothing found" in outcome.completed[0]
