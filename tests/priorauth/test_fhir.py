import httpx

from priorauth.fhir import HttpFhirSource, facts_to_fhir, parse_chart


def test_http_source_pages_and_maps_like_inmemory(scenarios):
    s = scenarios[3]; pid = s["request"]["patient_ref"]
    res = facts_to_fhir(pid, s["facts"])
    by_type = {}
    for r in res:
        by_type.setdefault(r["resourceType"], []).append(r)

    def handler(req: httpx.Request):
        path = req.url.path
        if path == f"/Patient/{pid}":
            return httpx.Response(200, json=by_type["Patient"][0])
        if path == "/Patient/missing":
            return httpx.Response(404)
        t = path.strip("/")
        items = by_type.get(t, [])
        assert req.url.params["patient"] == pid
        if t == "Observation" and len(items) > 1 and "page" not in req.url.params:      # force a next-link page
            return httpx.Response(200, json={"entry": [{"resource": items[0]}],
                                             "link": [{"relation": "next", "url": f"/Observation?page=2&patient={pid}"}]})
        if t == "Observation" and "page" in req.url.params:
            return httpx.Response(200, json={"entry": [{"resource": i} for i in items[1:]]})
        return httpx.Response(200, json={"entry": [{"resource": i} for i in items]})

    src = HttpFhirSource("http://fhir.test")
    src._c = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://fhir.test")
    assert parse_chart(pid, src.fetch(pid)) == parse_chart(pid, res)
    assert src.fetch("missing") == []


def test_http_source_is_read_only():
    import inspect
    src = inspect.getsource(HttpFhirSource)
    assert ".post(" not in src and ".put(" not in src and ".delete(" not in src and ".patch(" not in src
