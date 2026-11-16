"""Zero-shot Chronos forecasts for the forecast lab. Runs in its own venv (torch/transformers would clash with
pytorch_forecasting in .venv), and imports nothing from the project:

    .venv-chronos/bin/pip install "chronos-forecasting==2.0.1" pandas
    .venv/bin/python -m model.cgmacros.lab --export-context
    .venv-chronos/bin/python model/cgmacros/chronos_run.py chronos_bolt_small [context_len]
    .venv-chronos/bin/python model/cgmacros/chronos_run.py chronos2 [context_len]

Reads data/cgmacros/triage/lab_dev_context.npy (n x 512 glucose, NaN in gaps, last value = as_of) and writes
lab_pred_<name>.npy (n x 12 x 3: p10, p50, p90). Pretrained weights come from Hugging Face; no data leaves the machine.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[2] / "data" / "cgmacros" / "triage"
MODELS = {"chronos_bolt_small": "amazon/chronos-bolt-small", "chronos2": "amazon/chronos-2"}
H, Q = 12, [0.1, 0.5, 0.9]


def main() -> int:
    name = sys.argv[1]
    ctx_len = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODELS[name], device_map="cpu", torch_dtype=torch.float32)
    ctx = np.load(OUT / "lab_dev_context.npy")[:, -ctx_len:]
    out = np.zeros((len(ctx), H, 3), np.float32)
    t0, bs = time.time(), 256
    for s in range(0, len(ctx), bs):
        batch = torch.tensor(ctx[s:s + bs])
        if name == "chronos2":
            q, _ = pipe.predict_quantiles([b[None, :] for b in batch], prediction_length=H, quantile_levels=Q)
            q = torch.stack([x[0] for x in q])                  # (b, H, 3)
        else:
            q, _ = pipe.predict_quantiles(batch, prediction_length=H, quantile_levels=Q)
        out[s:s + bs] = q.numpy()
        if s % (bs * 20) == 0:
            print(f"  {s + len(batch)}/{len(ctx)} in {time.time() - t0:.0f}s", flush=True)
    np.save(OUT / f"lab_pred_{name}.npy", out)
    print(f"{name} ctx={ctx_len}: wrote {out.shape} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
