from priorauth.scripts.gen_docs import OUT, render


def test_tool_catalog_doc_is_up_to_date():
    assert OUT.read_text() == render(), "docs/priorauth/MCP_TOOLS.md is stale: run python -m priorauth.scripts.gen_docs"
