"""Train the CGMacros forecaster variants for the ablation.

    python -m model.cgmacros.train glucose        # A: glucose history only
    python -m model.cgmacros.train fitbit         # B: + Fitbit HR / METs / calories (past-observed)
    python -m model.cgmacros.train meals          # C: B + meal macros as KNOWN-FUTURE inputs
    python -m model.cgmacros.train meals_activity # D: C + METs as KNOWN-FUTURE (planned activity)
    python -m model.cgmacros.train all

Known-future means: at forecast time the decoder is told what will be eaten (or how
active the person will be) over the next hour. That is what makes "what if she eats
60 g of carbs at 12:30" answerable: the scenario supplies those future values. In
training they come from the real logged meals, i.e. the model assumes the meal plan
is known. Outputs go to model/artifacts/cgmacros/ (gitignored; reproducible here).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from model.cgmacros.dataset import build_grid
from model.cgmacros.splits import SCHEME, split_participants

ART = Path(__file__).resolve().parents[1] / "artifacts" / ("cgmacros" if SCHEME == "primary" else f"cgmacros_{SCHEME}")
ENCODER, MIN_ENCODER, HORIZON = 24, 6, 12           # 2 h history, >= 30 min, 60 min ahead (5-min steps)
CALENDAR = ["hour", "dow"]
MEAL = ["carbs", "protein", "fat", "fiber"]
FITBIT = ["hr", "mets", "cal_act"]
VARIANTS = {
    "glucose": {"known": CALENDAR, "unknown": ["glucose", "delta"]},
    "fitbit": {"known": CALENDAR, "unknown": ["glucose", "delta"] + FITBIT},
    "meals": {"known": CALENDAR + MEAL, "unknown": ["glucose", "delta"] + FITBIT},
    "meals_activity": {"known": CALENDAR + MEAL + ["mets"], "unknown": ["glucose", "delta", "hr", "cal_act"]},
}


class MemSafeTFT(TemporalFusionTransformer):
    """Same fix as model/train.py: pytorch_forecasting 1.1.1 keeps every step's
    autograd graph in the per-epoch outputs list; store a detached copy instead."""

    def training_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        self.training_step_outputs.append({**log, "loss": log["loss"].detach()})
        return log

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        log.update(self.create_log(x, y, out, batch_idx))
        self.validation_step_outputs.append({**log, "loss": log["loss"].detach()})
        return log


def datasets(variant: str, grid=None):
    grid = build_grid() if grid is None else grid
    train_ids, val_ids, test_ids = split_participants(sorted(grid.participant.unique()))
    spec = VARIANTS[variant]
    training = TimeSeriesDataSet(
        grid[grid.participant.isin(train_ids)],
        time_idx="time_idx", target="glucose", group_ids=["participant"],
        min_encoder_length=MIN_ENCODER, max_encoder_length=ENCODER, max_prediction_length=HORIZON,
        # No participant embedding: every test participant is unseen, so it would map to an
        # embedding that is never trained. Per-window EncoderNormalizer carries the personal level.
        time_varying_known_reals=spec["known"], time_varying_unknown_reals=spec["unknown"],
        # EncoderNormalizer scales each window by its own history, so the model forecasts relative to the
        # current level. With no target normalizer the first run collapsed to predicting the population mean
        # (flat ~130 mg/dL for every input; RMSE 31 vs 19.5 for persistence).
        allow_missing_timesteps=True, target_normalizer=EncoderNormalizer(),
    )
    validation = TimeSeriesDataSet.from_dataset(training, grid[grid.participant.isin(val_ids)],
                                                predict=False, stop_randomization=True)
    return training, validation, (train_ids, val_ids, test_ids)


def train(variant: str, max_epochs: int = 12, batches_per_epoch: int = 400) -> Path:
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ART.mkdir(parents=True, exist_ok=True)
    training, validation, _ = datasets(variant)
    tft = MemSafeTFT.from_dataset(training, learning_rate=0.03, hidden_size=16, attention_head_size=1,
                                  dropout=0.1, hidden_continuous_size=8, loss=QuantileLoss(), log_interval=0)
    ckpt = ModelCheckpoint(dirpath=ART / "checkpoints", filename=f"{variant}-best", monitor="val_loss",
                           mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", gradient_clip_val=0.1,
                         limit_train_batches=batches_per_epoch, enable_progress_bar=False, logger=False,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min"), ckpt])
    trainer.fit(tft, train_dataloaders=training.to_dataloader(train=True, batch_size=64, num_workers=0),
                val_dataloaders=validation.to_dataloader(train=False, batch_size=256, num_workers=0))
    best = Path(ckpt.best_model_path)
    target = ART / f"{variant}.ckpt"
    target.write_bytes(best.read_bytes())
    with open(ART / f"{variant}.dataset.pkl", "wb") as fh:
        pickle.dump(training, fh)
    print(f"[{variant}] best val_loss={float(ckpt.best_model_score):.3f} -> {target}", flush=True)
    return target


if __name__ == "__main__":
    names = list(VARIANTS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    unknown = [n for n in names if n not in VARIANTS]
    if not names or unknown:
        sys.exit(f"usage: python -m model.cgmacros.train {{{'|'.join(VARIANTS)}|all}}")
    import os
    for name in names:
        train(name, max_epochs=int(os.getenv("EPOCHS", "12")), batches_per_epoch=int(os.getenv("BATCHES", "400")))
