"""
FastAPI inference endpoint for GlucoPulse TFT model.

Two serving paths, chosen by how much history the request provides:

- Full 24-step (2h) history -> the ONNX (INT8) model. Fast, light runtime
  dependency (onnxruntime only).
- Shorter history -> the PyTorch checkpoint directly. pytorch_forecasting's
  TFT uses torch.nn.utils.rnn.pack_padded_sequence internally for variable-
  length encoder sequences, which does not survive ONNX export (Python-level
  branches on tensor values get baked in as trace-time constants -- this is
  a library/ONNX incompatibility, not something a local patch fixes safely).
  The checkpoint path handles variable encoder_lengths natively, so it's the
  only correct way to serve genuinely-short history rather than the
  ONNX path's zero-padding-read-as-real-history degradation.

Both paths return T+30 and T+60 minute predictions with 80% prediction
intervals.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pytorch_forecasting import TemporalFusionTransformer

app = FastAPI(title="GlucoPulse Inference API", version="1.0.0")

MODEL_PATH = Path(__file__).parent.parent / "model" / "artifacts" / "tft.onnx"
CHECKPOINT_PATH = Path(__file__).parent.parent / "model" / "artifacts" / "tft.ckpt"
SCALERS_PATH = Path(__file__).parent.parent / "model" / "artifacts" / "scalers.json"
FULL_HISTORY_STEPS = 24  # 2h at 5-min intervals
DECODER_STEPS = 12  # 60 min ahead, 5-min steps -- also train.py's MAX_PREDICTION_LENGTH

# Feature columns in the order expected by the model (matches
# training_dataset.pkl's `reals`: pytorch_forecasting auto-prepends
# "encoder_length" whenever min_encoder_length < max_encoder_length -- see
# train.py's MIN_ENCODER_LENGTH -- then known_reals, then unknown_reals).
# encoder_length isn't a per-timestep reading; it's a constant per-request
# value (same in every encoder and decoder row) so the model can condition
# on how much real history it was given -- see _encoder_length_feature.
ENCODER_CONT_FEATURES = [
    "encoder_length", "hour", "day_of_week", "glucose_value", "delta",
    "rolling_mean_15", "rolling_std_15", "rolling_mean_60", "rolling_std_60",
    "has_prior_bolus", "time_since_last_bolus_min",
    "has_prior_carb", "time_since_last_carb_min"
]

with open(SCALERS_PATH) as f:
    SCALERS = json.load(f)


def scale_feature(name: str, value: float) -> float:
    """Apply the same per-feature StandardScaler the TimeSeriesDataSet used
    during training ((value - mean) / scale). Every continuous feature is
    scaled this way except glucose_value -- it's the target and training
    used target_normalizer=None, so it stays raw mg/dL. Feeding raw values
    for the other 11 features (the bug this fixes) silently mismatches
    what the model was trained on -- not caught by ONNX parity checks since
    those replay real, already-scaled training samples."""
    scaler = SCALERS.get(name)
    if scaler and scaler.get("mean") and scaler.get("scale"):
        mean, scale = scaler["mean"][0], scaler["scale"][0]
        if scale != 0:
            return (value - mean) / scale
    return value

# Patient ID encoding (from training)
PATIENT_ID_MAP = {
    '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '18': 9, '19': 10,
    '2': 11, '20': 12, '22': 13, '23': 14, '24': 15, '3': 16, '6': 17, '7': 18
}


class GlucoseReading(BaseModel):
    """Single glucose reading with covariates."""
    timestamp: str = Field(..., description="ISO format timestamp (e.g., '2024-01-15T10:00:00Z')")
    glucose_value: float = Field(..., ge=40, le=400, description="CGM glucose in mg/dL")
    delta: Optional[float] = Field(None, description="Glucose change from previous reading")
    rolling_mean_15: Optional[float] = None
    rolling_std_15: Optional[float] = None
    rolling_mean_60: Optional[float] = None
    rolling_std_60: Optional[float] = None
    has_prior_bolus: Optional[bool] = False
    time_since_last_bolus_min: Optional[float] = -1.0
    has_prior_carb: Optional[bool] = False
    time_since_last_carb_min: Optional[float] = -1.0


class InferenceRequest(BaseModel):
    """Request for glucose prediction."""
    patient_id: str = Field(..., description="Patient ID (string)")
    readings: list[GlucoseReading] = Field(
        ..., min_length=6, max_length=24,
        description=(
            "Recent glucose readings, oldest first (6-24 = 30 min to 2h at "
            "5-min intervals). 6 is the model's trained minimum "
            "(train.py's MIN_ENCODER_LENGTH) -- fewer would be extrapolating "
            "into a regime the model never saw rather than a genuine prediction."
        ),
    )


class PredictionResponse(BaseModel):
    """Response with glucose predictions."""
    patient_id: str
    prediction_time: str
    t30_min: dict
    t60_min: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


_session: Optional[ort.InferenceSession] = None
_torch_model: Optional[TemporalFusionTransformer] = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Run export first.")
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def get_torch_model() -> TemporalFusionTransformer:
    """Loaded lazily -- only requests with < 24 readings need it, so the
    common case (full history, ONNX) never pays this cost."""
    global _torch_model
    if _torch_model is None:
        if not CHECKPOINT_PATH.exists():
            raise HTTPException(status_code=503, detail="Checkpoint not found for short-history fallback.")
        _torch_model = TemporalFusionTransformer.load_from_checkpoint(str(CHECKPOINT_PATH), map_location="cpu")
        _torch_model.eval()
    return _torch_model


def encode_patient_id(patient_id: str) -> int:
    """Encode patient ID to categorical index."""
    return PATIENT_ID_MAP.get(patient_id, 0)  # 0 = unknown/NaN


def _encoder_length_feature(n_readings: int) -> float:
    """pytorch_forecasting's auto-added "encoder_length" real (see
    ENCODER_CONT_FEATURES) -- confirmed empirically against real training
    batches across encoder_lengths 15-24: value = n_readings/DECODER_STEPS
    - 1 (its StandardScaler is identity, mean=0/scale=1, fit on data already
    in this form -- not a scalers.json lookup). Constant across every
    timestep in both encoder_cont and decoder_cont for a given request."""
    return n_readings / DECODER_STEPS - 1.0


def _encode_reading(reading: GlucoseReading, encoder_length_feature: float) -> list:
    """One encoder timestep's 13 features, in ENCODER_CONT_FEATURES order.
    Every feature except glucose_value must be scaled -- see scale_feature.
    Shared by both serving paths so a fix here can't drift between them the
    way the ONNX and PyTorch paths did before."""
    ts = pd.Timestamp(reading.timestamp)
    return [
        encoder_length_feature,
        scale_feature("hour", float(ts.hour)),
        scale_feature("day_of_week", float(ts.dayofweek)),
        reading.glucose_value,
        scale_feature("delta", reading.delta if reading.delta is not None else 0.0),
        scale_feature("rolling_mean_15", reading.rolling_mean_15 if reading.rolling_mean_15 is not None else reading.glucose_value),
        scale_feature("rolling_std_15", reading.rolling_std_15 if reading.rolling_std_15 is not None else 0.0),
        scale_feature("rolling_mean_60", reading.rolling_mean_60 if reading.rolling_mean_60 is not None else reading.glucose_value),
        scale_feature("rolling_std_60", reading.rolling_std_60 if reading.rolling_std_60 is not None else 0.0),
        scale_feature("has_prior_bolus", 1.0 if reading.has_prior_bolus else 0.0),
        scale_feature("time_since_last_bolus_min", reading.time_since_last_bolus_min if reading.time_since_last_bolus_min is not None else -1.0),
        scale_feature("has_prior_carb", 1.0 if reading.has_prior_carb else 0.0),
        scale_feature("time_since_last_carb_min", reading.time_since_last_carb_min if reading.time_since_last_carb_min is not None else -1.0),
    ]


def _decoder_row(future_time: pd.Timestamp, encoder_length_feature: float) -> list:
    """One decoder timestep: encoder_length (index 0) is constant same as
    the encoder rows; hour/day_of_week (indices 1-2) are known ahead of
    time; the rest stay 0 (the model's decoder pathway only attends to the
    known-reals columns regardless of what's there)."""
    row = [0.0] * len(ENCODER_CONT_FEATURES)
    row[0] = encoder_length_feature
    row[1] = scale_feature("hour", float(future_time.hour))
    row[2] = scale_feature("day_of_week", float(future_time.dayofweek))
    return row


def prepare_inputs_onnx(request: InferenceRequest):
    """Full 24-step history only -- the ONNX graph was traced with
    encoder_lengths=24 baked in as a constant (pytorch_forecasting's
    attention masking reads it as a Python int during tracing), so any
    other length crashes ONNX Runtime's attention broadcast. Callers must
    check n_readings == FULL_HISTORY_STEPS before using this path."""
    n_features = len(ENCODER_CONT_FEATURES)
    elf = _encoder_length_feature(FULL_HISTORY_STEPS)

    encoder_cont = np.zeros((1, FULL_HISTORY_STEPS, n_features), dtype=np.float32)
    encoder_cat = np.zeros((1, FULL_HISTORY_STEPS, 1), dtype=np.int64)
    patient_cat = encode_patient_id(request.patient_id)
    encoder_cat[0, :, 0] = patient_cat

    for i, reading in enumerate(request.readings):
        encoder_cont[0, i, :] = _encode_reading(reading, elf)

    decoder_cont = np.zeros((1, DECODER_STEPS, n_features), dtype=np.float32)
    decoder_cat = np.zeros((1, DECODER_STEPS, 1), dtype=np.int64)
    decoder_cat[0, :, 0] = patient_cat
    last_time = pd.Timestamp(request.readings[-1].timestamp)
    for i in range(DECODER_STEPS):
        decoder_cont[0, i, :] = _decoder_row(last_time + pd.Timedelta(minutes=5 * (i + 1)), elf)

    return {
        "encoder_lengths": np.array([FULL_HISTORY_STEPS], dtype=np.int64),
        "decoder_lengths": np.array([DECODER_STEPS], dtype=np.int64),
        "encoder_cont": encoder_cont,
        "encoder_cat": encoder_cat,
        "decoder_cont": decoder_cont,
        "decoder_cat": decoder_cat,
        # [center, scale] (pytorch_forecasting's actual order). training
        # used target_normalizer=None -> identity: center=0, scale=1.
        "target_scale": np.array([[0.0, 1.0]], dtype=np.float32),
    }


def prepare_inputs_torch(request: InferenceRequest) -> dict:
    """Fewer than 24 readings -- served by the PyTorch checkpoint directly.
    No padding: pytorch_forecasting's own collate_fn (torch.nn.utils.rnn.
    pad_sequence, batch_first=True) only pads when batching multiple
    variable-length samples together; for a single request the encoder
    tensor is exactly as long as the real history, matching what the
    (retrained-for-this) model actually saw during training."""
    n_readings = len(request.readings)
    elf = _encoder_length_feature(n_readings)
    patient_cat = encode_patient_id(request.patient_id)

    encoder_cont = torch.tensor(
        [[_encode_reading(r, elf) for r in request.readings]], dtype=torch.float32
    )
    encoder_cat = torch.full((1, n_readings, 1), patient_cat, dtype=torch.int64)

    last_time = pd.Timestamp(request.readings[-1].timestamp)
    decoder_cont = torch.tensor(
        [[_decoder_row(last_time + pd.Timedelta(minutes=5 * (i + 1)), elf) for i in range(DECODER_STEPS)]],
        dtype=torch.float32,
    )
    decoder_cat = torch.full((1, DECODER_STEPS, 1), patient_cat, dtype=torch.int64)

    return {
        "encoder_lengths": torch.tensor([n_readings], dtype=torch.int64),
        "decoder_lengths": torch.tensor([DECODER_STEPS], dtype=torch.int64),
        "encoder_cont": encoder_cont,
        "encoder_cat": encoder_cat,
        "decoder_cont": decoder_cont,
        "decoder_cat": decoder_cat,
        "target_scale": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        session = get_session()
        return HealthResponse(status="healthy", model_loaded=session is not None)
    except Exception as e:
        return HealthResponse(status="unhealthy", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: InferenceRequest):
    if len(request.readings) == FULL_HISTORY_STEPS:
        session = get_session()
        inputs = prepare_inputs_onnx(request)
        outputs = session.run(None, inputs)
        prediction = outputs[0]  # Shape: (1, 12, 7) - 7 quantiles
    else:
        model = get_torch_model()
        inputs = prepare_inputs_torch(request)
        with torch.no_grad():
            prediction = model(inputs).prediction.numpy()  # Shape: (1, 12, 7)

    # Extract median (quantile 0.5 is at index 3 for 7 quantiles: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
    median_pred = prediction[0, :, 3]  # (12,)
    q10 = prediction[0, :, 1]  # 10th percentile
    q90 = prediction[0, :, 5]  # 90th percentile

    # T+30 = step 5 (0-indexed), T+60 = step 11
    t30_idx = 5
    t60_idx = 11

    last_time = np.datetime64(request.readings[-1].timestamp.replace('Z', '+00:00'))
    pred_time_30 = last_time + np.timedelta64(30, 'm')
    pred_time_60 = last_time + np.timedelta64(60, 'm')

    return PredictionResponse(
        patient_id=request.patient_id,
        prediction_time=str(last_time),
        t30_min={
            "predicted_glucose": float(median_pred[t30_idx]),
            "prediction_interval_80pct": [float(q10[t30_idx]), float(q90[t30_idx])],
            "prediction_time": str(pred_time_30),
        },
        t60_min={
            "predicted_glucose": float(median_pred[t60_idx]),
            "prediction_interval_80pct": [float(q10[t60_idx]), float(q90[t60_idx])],
            "prediction_time": str(pred_time_60),
        },
    )


@app.get("/")
async def root():
    return {
        "service": "GlucoPulse Inference API",
        "version": "1.0.0",
        "model": "Temporal Fusion Transformer (ONNX for 24-step history, PyTorch checkpoint fallback for shorter)",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)