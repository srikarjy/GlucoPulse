"""
Unit tests for resource-verified citation grounding: a FHIR citation only
counts if the cited resource was actually returned by a tool this session.
Pure functions -- no MCP server or model needed.
"""
from __future__ import annotations

import json

from agents.citation import ground_finding
from agents.reasoning_agent import retrieved_resource_ids
from agents.schemas import ClinicalFinding

S = "The patient's most recent glucose reading was 143 mg/dL."


def _draft(fhir_id=None, kb=None, missing=()):
    return ClinicalFinding(finding=S, confidence="high", missing_data=list(missing),
                           evidence=[{"claim": S, "fhir_resource_id": fhir_id, "kb_source": kb}])


def test_citation_to_retrieved_resource_survives():
    out, dropped = ground_finding(_draft("Observation/abc"), retrieved_ids={"Observation/abc"})
    assert out.finding == S and not dropped
    assert not any(f.startswith("rejected_") for f in out.red_flags)


def test_citation_to_unretrieved_resource_is_rejected_and_forces_abstention():
    out, dropped = ground_finding(_draft("Observation/fabricated"), retrieved_ids={"Observation/abc"})
    assert out.finding == "" and dropped == [S]
    assert out.evidence == []
    assert "rejected_1_citation(s)_to_unretrieved_resource" in out.red_flags
    assert "no_grounded_finding_and_no_missing_data_declared" in out.red_flags


def test_kb_source_citation_is_not_resource_checked():
    out, _ = ground_finding(_draft(kb="ADA Standards of Care 2025"), retrieved_ids=set())
    assert out.finding == S


def test_no_retrieved_ids_keeps_textual_only_behavior():
    out, _ = ground_finding(_draft("Observation/anything"))
    assert out.finding == S


def test_retrieved_resource_ids_reads_bundles_and_notes_from_tool_results():
    bundle = {"resourceType": "Bundle", "id": "b1",
              "entry": [{"resource": {"resourceType": "Observation", "id": "o1"}},
                        {"resource": {"resourceType": "Observation", "id": "o2"}}]}
    notes = {"notes": [{"note_id": "n1", "quarantined": False, "text": "ok"}]}
    messages = [
        {"role": "user", "content": [{"text": "q"}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "1", "status": "success",
                                                     "content": [{"text": json.dumps(bundle)}]}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "2", "status": "success",
                                                     "content": [{"json": notes}, {"text": "not json"}]}}]},
    ]
    ids = retrieved_resource_ids(messages)
    assert {"Observation/o1", "Observation/o2", "DocumentReference/n1"} <= ids


def test_model_text_cannot_inject_retrieved_ids():
    # Only toolResult blocks count -- an id that appears in the model's own text does not.
    messages = [{"role": "assistant",
                 "content": [{"text": json.dumps({"resourceType": "Observation", "id": "fake"})}]}]
    assert retrieved_resource_ids(messages) == set()
