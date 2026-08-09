"""
Train a Temporal Fusion Transformer on cgm_features to forecast glucose
12 steps ahead (60 min on the 5-min grid) -- T+30 and T+60 are read off
that sequence at step 6 and step 12, evaluated separately in evaluate.py.

Held out by patient, not by time (docs/STATUS.md) -- TRAIN_PATIENTS/
VAL_PATIENTS only, never TEST_PATIENTS, so the final held-out evaluation
means what it claims to mean.

Feature channels (see README's TFT justification):
  - known future (calendar, always knowable ahead of time): hour of day,
    day of week
  - past observed (only known up to "now"): glucose_value, delta,
    rolling_mean/std_15/60, time_since_last_bolus/carb_min
  - static: patient_id (categorical, per-patient embedding)
Bolus/carb are sparse relative to the 5-min grid -- has_prior_bolus/carb
plus the existing NaN cold-start convention (docs/QUESTIONS.md) is what
resolves that sparsity, not a new imputation scheme.
"""

import pickle
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss


class MemSafeTFT(TemporalFusionTransformer):
    """pytorch_forecasting 1.1.1's BaseModel.step() returns a loss tensor that
    still carries its full autograd graph, and training_step/validation_step
    append that same dict to a list only cleared at epoch end -- with ~5800
    batches/epoch that retains every batch's backward graph simultaneously,
    growing memory linearly across the epoch until Docker OOM-kills the
    container (observed: 800MB -> 4.5GB in 26 min). Detaching the stored copy
    (not the one returned for backprop) fixes it without touching the actual
    training math."""

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

from db import load_features
from splits import TRAIN_PATIENTS, VAL_PATIENTS

MAX_ENCODER_LENGTH = 24  # 2h of history
MAX_PREDICTION_LENGTH = 12  # 60 min ahead, 5-min steps
MODEL_DIR = Path(__file__).parent / "artifacts"
CHECKPOINT_PATH = MODEL_DIR / "tft.ckpt"


def build_dataframe(patients=None) -> pd.DataFrame:
    if patients is None:
        patients = TRAIN_PATIENTS + VAL_PATIENTS
    df = load_features()
    df = df[df["patient_id"].isin(patients)].copy()

    # Integer time index per patient, gaps allowed (rows with a dropped
    # glucose_value were never written -- see spark/feature_job.py).
    df = df.sort_values(["patient_id", "time"])
    df["time_idx"] = (
        df.groupby("patient_id")["time"].transform(lambda s: ((s - s.min()) / pd.Timedelta(minutes=5)).round().astype(int))
    )

    df["hour"] = df["time"].dt.hour.astype(float)
    df["day_of_week"] = df["time"].dt.dayofweek.astype(float)

    # Cold-start channels: has_prior_* start as real booleans/NaN (design in
    # docs/QUESTIONS.md); TFT needs numeric, non-null past-observed reals.
    df["has_prior_bolus"] = df["has_prior_bolus"].fillna(False).astype(float)
    df["has_prior_carb"] = df["has_prior_carb"].fillna(False).astype(float)
    df["time_since_last_bolus_min"] = df["time_since_last_bolus_min"].fillna(-1.0)
    df["time_since_last_carb_min"] = df["time_since_last_carb_min"].fillna(-1.0)
    df["delta"] = df["delta"].fillna(0.0)
    df["rolling_std_15"] = df["rolling_std_15"].fillna(0.0)
    df["rolling_std_60"] = df["rolling_std_60"].fillna(0.0)
    df["rolling_mean_15"] = df["rolling_mean_15"].fillna(df["glucose_value"])
    df["rolling_mean_60"] = df["rolling_mean_60"].fillna(df["glucose_value"])

    return df


def make_datasets(df: pd.DataFrame):
    train_df = df[df["patient_id"].isin(TRAIN_PATIENTS)]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="glucose_value",
        group_ids=["patient_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["patient_id"],
        time_varying_known_reals=["hour", "day_of_week"],
        time_varying_unknown_reals=[
            "glucose_value",
            "delta",
            "rolling_mean_15",
            "rolling_std_15",
            "rolling_mean_60",
            "rolling_std_60",
            "has_prior_bolus",
            "time_since_last_bolus_min",
            "has_prior_carb",
            "time_since_last_carb_min",
        ],
        allow_missing_timesteps=True,
        target_normalizer=None,
        categorical_encoders={"patient_id": NaNLabelEncoder(add_nan=True)},
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=False, stop_randomization=True)
    return training, validation


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = build_dataframe()
    training, validation = make_datasets(df)

    train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=32, num_workers=0)

    tft = MemSafeTFT.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=0,
    )
    print(f"Model params: {sum(p.numel() for p in tft.parameters())}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=MODEL_DIR / "checkpoints", filename="best", monitor="val_loss", mode="min", save_top_k=1
    )
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min"), checkpoint_cb],
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_cb.best_model_path
    print(f"Best checkpoint: {best_path} (val_loss={checkpoint_cb.best_model_score})")

    import shutil

    shutil.copy(best_path, CHECKPOINT_PATH)
    with open(MODEL_DIR / "training_dataset.pkl", "wb") as f:
        pickle.dump(training, f)
    print(f"Saved best checkpoint + dataset spec to {MODEL_DIR}")


if __name__ == "__main__":
    main()
