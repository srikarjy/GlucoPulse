"""Alert consumer logic (consumer/alerts_stream.Processor) with an in-memory store and publisher: duplicates, crashes
between save and delivery, and restarts from a checkpoint must never lose or repeat an alert."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("numpy")
from consumer.alerts_stream import Processor, alert_id, alert_message, parse_reading  # noqa: E402

T0 = datetime(2026, 1, 1, 8, 0)
TOPIC = "cgm-raw"


class FakeStore:
    def __init__(self):
        self.alerts: dict[str, dict] = {}            # alert_id -> row incl. published
        self.checkpoints: dict[int, tuple[int, dict]] = {}

    def save_alert(self, a):
        row = self.alerts.setdefault(alert_id(a), {**a, "published": False})
        return row["published"]

    def mark_published(self, key):
        if key not in self.alerts:
            return False
        self.alerts[key]["published"] = True
        return True

    def unpublished(self):
        return [r for r in self.alerts.values() if not r["published"]]

    def load_checkpoint(self, topic, partition):
        cp = self.checkpoints.get(partition)
        return copy.deepcopy(cp) if cp else None

    def save_checkpoints(self, topic, rows):
        for p, off, state in rows:
            self.checkpoints[p] = (off, json.loads(json.dumps(state)))


class FakePublisher:
    """Delivers on flush (like Kafka's callbacks); `lose_pending()` simulates a crash before delivery."""

    def __init__(self):
        self.sent: list[str] = []
        self.pending: list[str] = []
        self.proc = None

    def publish(self, key, value):
        assert value["alert_id"] == key and "disclaimer" in value
        self.pending.append(key)

    fail_next = 0                                   # deliveries to report as failed
    stuck = False                                   # broker unreachable: nothing confirmed within the flush timeout

    def flush(self):
        if self.stuck:
            return len(self.pending)
        for k in self.pending:
            if self.fail_next:
                self.fail_next -= 1
                self.proc.delivered(k, False, "Local: Message timed out")
                continue
            self.sent.append(k)
            self.proc.delivered(k, True)
        self.pending = []
        return 0

    def lose_pending(self):
        self.pending = []


def new_proc(store, pub, **kw):
    p = Processor(store, pub, TOPIC, **kw)
    pub.proc = p
    return p


def msg(pid, t, g):
    return json.dumps({"patient_id": pid, "event_datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "cgm": g}).encode()


def falling(pid="cgmacros-001", n=60):
    """Repeating fall/recovery so several alerts fire."""
    vals = ([130, 122, 114, 106, 98, 90, 82, 76, 74, 76, 85, 100, 120, 135, 140, 140] * 10)[:n]
    return [msg(pid, T0 + timedelta(minutes=5 * k), v) for k, v in enumerate(vals)]


def run(proc, msgs, partition=0, start=0, stop=None):
    for off in range(start, len(msgs) if stop is None else stop):
        proc.handle(partition, off, msgs[off])


def test_parse_reading_accepts_only_usable_readings():
    assert parse_reading(msg("cgmacros-001", T0, 120)) == ("cgmacros-001", T0, 120.0)
    assert parse_reading(b"not json") is None
    assert parse_reading(json.dumps({"patient_id": "x", "event_datetime": "2026-01-01 08:00:00"}).encode()) is None
    assert parse_reading(json.dumps({"patient_id": "x", "event_datetime": "bad", "cgm": 100}).encode()) is None
    assert parse_reading(json.dumps({"patient_id": "x", "event_datetime": "2026-01-01 08:00:00",
                                     "cgm": True}).encode()) is None
    assert parse_reading(msg("x", T0, 30)) is None and parse_reading(msg("x", T0, 450)) is None
    tz = json.dumps({"patient_id": "x", "event_datetime": "2026-01-01T09:00:00+01:00", "cgm": 100}).encode()
    assert parse_reading(tz)[1] == T0                     # converted to naive UTC


def test_alerts_are_saved_published_once_and_marked():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub)
    run(proc, falling())
    proc.checkpoint()
    assert len(store.alerts) >= 3
    assert sorted(pub.sent) == sorted(store.alerts) and all(r["published"] for r in store.alerts.values())


def test_replaying_the_same_messages_sends_nothing_twice():
    store, pub = FakeStore(), FakePublisher()
    msgs = falling()
    proc = new_proc(store, pub)
    run(proc, msgs)
    proc.checkpoint()
    first = list(pub.sent)
    proc2 = new_proc(store, pub)                          # e.g. no checkpoint: everything redelivered
    run(proc2, msgs)
    proc2.checkpoint()
    assert pub.sent == first


def test_crash_after_save_before_delivery_resends_on_replay():
    store, pub = FakeStore(), FakePublisher()
    msgs = falling()
    proc = new_proc(store, pub)
    run(proc, msgs, stop=30)
    proc.checkpoint()
    cp_offset = store.checkpoints[0][0]
    run(proc, msgs, start=30)
    lost = list(pub.pending)
    assert lost                                           # alerts saved after the checkpoint, not yet delivered
    pub.lose_pending()                                    # crash: no delivery, no checkpoint
    assert {alert_id(r) for r in store.unpublished()} == set(lost)
    proc2 = new_proc(store, pub)
    assert proc2.assign(0) == cp_offset
    run(proc2, msgs, start=cp_offset)
    proc2.checkpoint()
    assert not store.unpublished()
    assert sorted(pub.sent) == sorted(store.alerts)       # each alert delivered exactly once here


def test_startup_sweep_resends_unpublished():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub)
    run(proc, falling())
    pub.lose_pending()
    n = len(store.unpublished())
    assert n
    proc2 = new_proc(store, pub)
    assert proc2.resend_unpublished() == n
    proc2.checkpoint()
    assert not store.unpublished()


@pytest.mark.parametrize("cut", [5, 17, 33, 48])
def test_restart_from_checkpoint_matches_an_uninterrupted_run(cut):
    msgs = falling(n=60)
    ref_store, ref_pub = FakeStore(), FakePublisher()
    ref = new_proc(ref_store, ref_pub)
    run(ref, msgs)
    ref.checkpoint()

    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub)
    run(proc, msgs, stop=cut)
    proc.checkpoint()
    run(proc, msgs, start=cut, stop=cut + 4)              # a little work after the checkpoint, then crash
    pub.lose_pending()
    proc2 = new_proc(store, pub)
    run(proc2, msgs, start=proc2.assign(0))
    proc2.checkpoint()
    assert set(store.alerts) == set(ref_store.alerts)
    assert all(r["published"] for r in store.alerts.values())


def test_population_outside_validation_never_gets_alerts():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub)
    run(proc, falling(pid="azt1d-14"))
    assert not store.alerts and proc.stats["skipped_not_validated_population"] == 60


def test_old_readings_are_not_sent_in_live_mode():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub, max_age=timedelta(minutes=15), now=lambda: datetime(2026, 6, 1, tzinfo=timezone.utc))
    run(proc, falling())
    assert not store.alerts and proc.stats["alert_too_old_to_send"] > 0


def test_revoke_checkpoints_and_forgets_the_partitions_people():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub)
    run(proc, falling(pid="cgmacros-001"), partition=1)
    proc.revoke(1)
    assert 1 in store.checkpoints and "cgmacros-001" in store.checkpoints[1][1]
    assert "cgmacros-001" not in proc.stream.people


def test_alert_message_is_self_describing():
    a = {"patient_id": "cgmacros-001", "time": T0, "tier": "loud", "direction": "low", "glucose": 82.0,
         "projected_20min": 60.0, "spacing_min": 5}
    m = alert_message(a)
    assert m["alert_id"] == "cgmacros-001|2026-01-01T08:00:00+00:00|loud"
    assert m["time"].endswith("+00:00") and "meter" in m["disclaimer"]
    assert (m["projected_20min"], m["projection_capped"]) == (60.0, False)


@pytest.mark.parametrize("raw, shown, capped", [(-1.0, 40.0, True), (39.9, 40.0, True), (40.0, 40.0, False),
                                                (255.0, 255.0, False), (400.0, 400.0, False), (451.0, 400.0, True)])
def test_published_projection_is_capped_to_the_sensor_range(raw, shown, capped):
    """Seen in the live demo: a fast fall projected -1 mg/dL. Sensors read 40-400, as the MCP tool already reports."""
    a = {"patient_id": "cgmacros-008", "time": T0, "tier": "loud", "direction": "low" if raw < 70 else "high",
         "glucose": 76.0, "projected_20min": raw, "spacing_min": 5}
    m = alert_message(a)
    assert (m["projected_20min"], m["projection_capped"]) == (shown, capped)
    assert a["projected_20min"] == raw                  # the stored record keeps the raw value


# --- review findings (ecc python-reviewer + silent-failure-hunter, 2026-11-24) -----------------------------------
class FlakyStore(FakeStore):
    """save_alert raises once, like a statement timeout or deadlock on a still-usable connection."""
    fail_saves = 1

    def save_alert(self, a):
        if self.fail_saves:
            self.fail_saves -= 1
            raise RuntimeError("canceling statement due to statement timeout")
        return super().save_alert(a)


def test_failed_save_is_never_checkpointed_past_and_the_alert_comes_back():
    msgs = falling()
    ref_store, ref_pub = FakeStore(), FakePublisher()
    ref = new_proc(ref_store, ref_pub)
    run(ref, msgs)
    ref.checkpoint()

    store, pub = FlakyStore(), FakePublisher()
    store.fail_saves = 0
    proc = new_proc(store, pub)
    run(proc, msgs, stop=3)
    proc.checkpoint()
    store.fail_saves = 1
    with pytest.raises(RuntimeError):
        run(proc, msgs, start=3)                     # dies on the first alert's save
    assert proc.dirty
    assert proc.checkpoint() == [] and proc.stats["checkpoint_refused_dirty"] == 1
    proc.revoke(0)                                   # the rebalance on shutdown must not checkpoint either
    assert store.checkpoints[0][0] == 3
    proc2 = new_proc(store, pub)                     # restart from the last good checkpoint
    run(proc2, msgs, start=proc2.assign(0))
    proc2.checkpoint()
    assert set(store.alerts) == set(ref_store.alerts)
    assert all(r["published"] for r in store.alerts.values())


def test_failed_delivery_is_resent_at_the_next_checkpoint_without_a_restart():
    store, pub = FakeStore(), FakePublisher()
    pub.fail_next = 2
    proc = new_proc(store, pub)
    run(proc, falling())
    proc.checkpoint()
    assert proc.stats["delivery_failed"] == 2 and proc.stats["resent_after_failure"] == 2
    assert not store.unpublished()


def test_checkpoint_waits_while_deliveries_are_still_pending():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub)
    run(proc, falling())
    pub.stuck = True
    assert proc.checkpoint() == [] and proc.stats["checkpoint_skipped_pending"] == 1
    assert 0 not in store.checkpoints
    pub.stuck = False
    assert proc.checkpoint() and not store.unpublished()


def test_mark_published_error_does_not_escape_the_delivery_callback():
    class BrokenMark(FakeStore):
        def mark_published(self, key):
            raise RuntimeError("connection reset")
    store, pub = BrokenMark(), FakePublisher()
    proc = new_proc(store, pub)
    run(proc, falling())
    proc.checkpoint()                                # must not raise
    assert proc.stats["mark_published_failed"] > 0


def test_mark_published_matching_no_row_is_not_counted_as_delivered():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub, log=lambda s: None)
    proc.delivered("cgmacros-001|2026-01-01T08:00:00+00:00|loud", True)
    assert proc.stats["delivered"] == 0 and proc.stats["mark_published_no_row"] == 1


def test_a_future_timestamp_cannot_silence_a_person():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub, now=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc), log=lambda s: None)
    proc.handle(0, 0, msg("cgmacros-001", datetime(2099, 1, 1), 120))     # wrong sensor clock
    assert proc.stats["skipped_future_timestamp"] == 1
    run(proc, falling(), start=0)
    assert store.alerts                              # real readings still alert


def test_skip_reasons_are_counted_separately():
    store, pub = FakeStore(), FakePublisher()
    proc = new_proc(store, pub, log=lambda s: None)
    proc.handle(0, 0, b"{broken")
    proc.handle(0, 1, json.dumps({"patient_id": "cgmacros-001", "event_datetime": "0001-01-01T00:00:00+05:00",
                                  "cgm": 100}).encode())
    proc.handle(0, 2, msg("cgmacros-001", T0, 20))
    assert (proc.stats["skipped_malformed"], proc.stats["skipped_bad_timestamp"],
            proc.stats["skipped_implausible"]) == (1, 1, 1)
    assert proc.next_offset[0] == 3                  # skipped messages still advance the offset


def test_startup_resend_respects_the_live_age_limit():
    store, pub = FakeStore(), FakePublisher()
    run(new_proc(store, pub), falling())
    pub.lose_pending()
    n = len(store.unpublished())
    late = new_proc(store, pub, max_age=timedelta(minutes=15), now=lambda: datetime(2026, 6, 1, tzinfo=timezone.utc),
                    log=lambda s: None)
    assert late.resend_unpublished() == 0 and late.stats["unpublished_too_old_to_send"] == n
