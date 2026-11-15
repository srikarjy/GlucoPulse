"""Deterministic Strands model double: plays back a fixed list of turns
(("tool", name, input) or ("text", reply)). Same shape as agents/eval/run_eval.py's."""
from __future__ import annotations

import json

from strands.models import Model


class ScriptedModel(Model):
    def __init__(self, turns):
        self.turns, self.i = list(turns), 0

    def update_config(self, **kw): ...
    def get_config(self): return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kw):  # pragma: no cover
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        kind, *rest = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        yield {"messageStart": {"role": "assistant"}}
        if kind == "text":
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": rest[0]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            name, inp = rest
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.i}", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(inp)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                            "metrics": {"latencyMs": 1}}}
