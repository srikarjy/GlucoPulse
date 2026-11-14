"""
Regression test for ehr/sync.py's forecast-feedback safeguard: a
previously-written forecast Observation (or a retracted/cancelled one)
must never be sent back to GlucoPulse as if it were a real reading.

Exercises the actual sync loop (sync_patient), not just the isolated
fhir_mapping.readings_from_bundle filter -- that filter already passed
while this loop was broken, because the loop never carried status/meta.tag
far enough for readings_from_bundle's own check to ever see them.

Run from repo root inside the venv:
    .venv/bin/pytest tests/healthcare/test_sync.py -q
"""
from datetime import datetime, timezone

import pytest

from ehr import sync
from serving.healthcare import fhir_mapping as fm


@pytest.fixture(autouse=True)
def pseudonym_key(monkeypatch):
    monkeypatch.setenv("GLUCOPULSE_PSEUDONYM_KEY", "k" * 40)


class FakeEhrClient:
    """Duck-types ehr.sync.EhrClient's read side without any network I/O."""

    def __init__(self, entries):
        self._entries = entries
        self.put_calls: list = []

    def cgm_observations(self, patient_id, count=100):
        return self._entries

    def put_all(self, resources):
        self.put_calls.append(resources)
        return {"type": "transaction-response"}


def _real_reading(ts: str) -> dict:
    return {"resource": {
        "resourceType": "Observation",
        "status": "final",
        "code": {"coding": [{"system": fm.LOINC, "code": fm.CGM_LOINC}]},
        "effectiveDateTime": ts,
        "valueQuantity": {"value": 120.0, "code": "mg/dL"},
    }}


def _forecast_reading(ts: str) -> dict:
    """Shape of a forecast Observation as GlucoPulse writes it -- and as the
    EHR then returns on a later search by the same LOINC code, since the
    search has no status/tag filter of its own."""
    target = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return {"resource": fm.forecast_observation(
        "pseudo", datetime(2024, 1, 1, tzinfo=timezone.utc), target, 30, 140.0, 120.0, 160.0, "device-1")}


def _entered_in_error_reading(ts: str) -> dict:
    r = dict(_real_reading(ts)["resource"])
    r["status"] = "entered-in-error"
    return {"resource": r}


def _run_sync(monkeypatch, entries):
    """Stubs the GlucoPulse HTTP call and returns what sync_patient actually
    sent in the outgoing bundle, plus the headers it used."""
    sent: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent["bundle"], sent["headers"] = json, headers

        class Resp:
            status_code = 200

            def json(self) -> dict:
                return {"resourceType": "Bundle", "type": "collection", "entry": []}

        return Resp()

    monkeypatch.setattr(sync.httpx, "post", fake_post)
    ehr = FakeEhrClient(entries)
    sync.sync_patient(ehr, "patient-1", "http://glucopulse.test", "tok")
    return sent, ehr


def test_previously_written_forecast_is_not_fed_back(monkeypatch):
    """The bug this guards: a forecast written back on an earlier sync must
    not be sent to GlucoPulse again as if it were a real reading."""
    entries = [_real_reading("2024-01-15T10:00:00Z"), _forecast_reading("2024-01-15T10:05:00Z")]
    sent, _ = _run_sync(monkeypatch, entries)
    assert len(sent["bundle"]["entry"]) == 1
    assert sent["bundle"]["entry"][0]["resource"]["effectiveDateTime"] == "2024-01-15T10:00:00Z"


def test_entered_in_error_reading_is_excluded(monkeypatch):
    entries = [_real_reading("2024-01-15T10:00:00Z"), _entered_in_error_reading("2024-01-15T10:05:00Z")]
    sent, _ = _run_sync(monkeypatch, entries)
    assert len(sent["bundle"]["entry"]) == 1


def test_purpose_of_use_is_not_treat(monkeypatch):
    """The forecast request asks GlucoPulse to simulate, not to treat --
    sending TREAT would contradict the Device resource's own 'not for
    treatment decisions' note in the audit trail."""
    sent, _ = _run_sync(monkeypatch, [_real_reading("2024-01-15T10:00:00Z")])
    assert sent["headers"].get("X-Purpose-Of-Use") is None


def test_all_forecast_entries_raises_instead_of_silently_forecasting_on_nothing(monkeypatch):
    with pytest.raises(SystemExit):
        _run_sync(monkeypatch, [_forecast_reading("2024-01-15T10:05:00Z")])
