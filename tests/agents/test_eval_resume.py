"""The live triage eval must survive being killed: finished cases are checkpointed and skipped on restart."""
from __future__ import annotations

import json

from agents.triage.eval import append_checkpoint, load_checkpoint, summarize


def _row(cid, level="monitor", llm=True):
    stats = {"calls": 1, "seconds": 0.5, "parse_errors": 0, "claims_kept": 1, "claims_dropped": 0,
             "raised_above_rule": 0}
    return {"id": cid, "rule": level, "single": level, "panel": level,
            "llm": {"single": stats, "panel": stats} if llm else None}


def test_torn_line_is_skipped_and_does_not_eat_the_next_row(tmp_path):
    ckpt = tmp_path / "eval_x.cases.jsonl"
    append_checkpoint(ckpt, _row("case-000"))
    with ckpt.open("a") as f:                      # process killed mid-write
        f.write('{"id": "case-0')
    append_checkpoint(ckpt, _row("case-001"))
    assert set(load_checkpoint(ckpt)) == {"case-000", "case-001"}


def test_missing_checkpoint_means_nothing_done(tmp_path):
    assert load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_summarize_counts_llm_stats_only_where_present():
    labels = {"a": {"label": "monitor"}, "b": {"label": "urgent"}, "_meta": {}}
    res = summarize([_row("a"), _row("b", llm=False)], labels, "m")
    assert res["cases"] == 2
    assert res["llm_totals_cases"] == 1
    assert res["llm_totals"]["panel"]["calls"] == 1
    assert res["methods"]["rule"]["accuracy"] == 0.5
    json.dumps(res)
