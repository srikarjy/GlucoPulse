"""
FastAPI inference endpoint for GlucoPulse TFT model.

Serves the ONNX-exported TFT model via REST. Accepts a patient's recent
glucose history + covariates and returns T+30 and T+60 minute predictions
with 80% prediction intervals.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="GlucoPulse Inference API", version="1.0.0")

MODEL_PATH = Path(__file__).parent.parent / "model" / "artifacts" / "tft.onnx"
SCALERS_PATH = Path(__file__).parent.parent / "model" / "artifacts" / "scalers.json"

# Feature columns in the order expected by the model (matches
# training_dataset.pkl's `reals`: known_reals then unknown_reals)
ENCODER_CONT_FEATURES = [
    "hour", "day_of_week", "glucose_value", "delta",
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
        ..., min_length=1, max_length=24,
        description="Recent glucose readings (up to 24 = 2 hours at 5-min intervals)"
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


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Run export first.")
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def encode_patient_id(patient_id: str) -> int:
    """Encode patient ID to categorical index."""
    return PATIENT_ID_MAP.get(patient_id, 0)  # 0 = unknown/NaN


def prepare_inputs(request: InferenceRequest):
    """Prepare ONNX inputs from request."""
    n_readings = len(request.readings)
    encoder_steps = 24
    decoder_steps = 12

    # Pad encoder to 24 steps
    encoder_cont = np.zeros((1, encoder_steps, 12), dtype=np.float32)
    encoder_cat = np.zeros((1, encoder_steps, 1), dtype=np.int64)
    
    patient_cat = encode_patient_id(request.patient_id)
    encoder_cat[0, :, 0] = patient_cat

    for i, reading in enumerate(request.readings):
        if i >= encoder_steps:
            break
        # pd.Timestamp.dayofweek (Monday=0) matches train.py's
        # df["time"].dt.dayofweek exactly -- a raw numpy epoch-day%7 formula
        # does not (it's offset by a constant 4, since epoch day 0 was a
        # Thursday), which would silently feed the model a rotated,
        # never-seen day-of-week signal.
        ts = pd.Timestamp(reading.timestamp)
        hour = ts.hour
        day_of_week = ts.dayofweek

        # Fill encoder features (last n_readings steps). Every feature except
        # glucose_value (index 2) must be scaled -- see scale_feature.
        idx = encoder_steps - n_readings + i
        encoder_cont[0, idx, 0] = scale_feature("hour", float(hour))
        encoder_cont[0, idx, 1] = scale_feature("day_of_week", float(day_of_week))
        encoder_cont[0, idx, 2] = reading.glucose_value
        encoder_cont[0, idx, 3] = scale_feature("delta", reading.delta if reading.delta is not None else 0.0)
        encoder_cont[0, idx, 4] = scale_feature("rolling_mean_15", reading.rolling_mean_15 if reading.rolling_mean_15 is not None else reading.glucose_value)
        encoder_cont[0, idx, 5] = scale_feature("rolling_std_15", reading.rolling_std_15 if reading.rolling_std_15 is not None else 0.0)
        encoder_cont[0, idx, 6] = scale_feature("rolling_mean_60", reading.rolling_mean_60 if reading.rolling_mean_60 is not None else reading.glucose_value)
        encoder_cont[0, idx, 7] = scale_feature("rolling_std_60", reading.rolling_std_60 if reading.rolling_std_60 is not None else 0.0)
        encoder_cont[0, idx, 8] = scale_feature("has_prior_bolus", 1.0 if reading.has_prior_bolus else 0.0)
        encoder_cont[0, idx, 9] = scale_feature("time_since_last_bolus_min", reading.time_since_last_bolus_min if reading.time_since_last_bolus_min is not None else -1.0)
        encoder_cont[0, idx, 10] = scale_feature("has_prior_carb", 1.0 if reading.has_prior_carb else 0.0)
        encoder_cont[0, idx, 11] = scale_feature("time_since_last_carb_min", reading.time_since_last_carb_min if reading.time_since_last_carb_min is not None else -1.0)

    # Decoder: future time steps. decoder_cont must have the same 12 columns
    # as encoder_cont (the model's unified `reals` ordering -- known_reals
    # then unknown_reals, see training_dataset.pkl); only the known-future
    # columns (hour, day_of_week; indices 0-1) are populated and scaled the
    # same way as the encoder, the rest stay zero since those values aren't
    # knowable ahead of time (and the model's decoder pathway only attends
    # to the known-reals columns regardless of what's in the others).
    decoder_cont = np.zeros((1, decoder_steps, 12), dtype=np.float32)
    decoder_cat = np.zeros((1, decoder_steps, 1), dtype=np.int64)
    decoder_cat[0, :, 0] = patient_cat

    last_time = pd.Timestamp(request.readings[-1].timestamp)
    for i in range(decoder_steps):
        future_time = last_time + pd.Timedelta(minutes=5 * (i + 1))
        hour = future_time.hour
        day_of_week = future_time.dayofweek
        decoder_cont[0, i, 0] = scale_feature("hour", float(hour))
        decoder_cont[0, i, 1] = scale_feature("day_of_week", float(day_of_week))

    # Lengths. The exported ONNX graph was traced with encoder_lengths=24
    # baked in as a constant (pytorch_forecasting's attention masking reads
    # encoder_lengths as a Python int during tracing -- see the
    # TracerWarnings in export_onnx.py's export step) -- passing any other
    # value crashes ONNX Runtime's attention broadcast at inference. So this
    # is always 24 regardless of n_readings, which means requests with
    # fewer than 24 readings get zero-padded history that the model treats
    # as real, not masked-out, history. Known limitation: send full 24-step
    # (2h) history for reliable predictions; fewer readings degrade quality
    # rather than erroring.
    encoder_lengths = np.array([encoder_steps], dtype=np.int64)
    decoder_lengths = np.array([decoder_steps], dtype=np.int64)

    # Target scale (identity normalizer from training)
    # [center, scale] (pytorch_forecasting's actual order -- confirmed against
    # a real training sample). target_normalizer=None in training means
    # identity: center=0, scale=1.
    target_scale = np.array([[0.0, 1.0]], dtype=np.float32)

    return {
        "encoder_lengths": encoder_lengths,
        "decoder_lengths": decoder_lengths,
        "encoder_cont": encoder_cont,
        "encoder_cat": encoder_cat,
        "decoder_cont": decoder_cont,
        "decoder_cat": decoder_cat,
        "target_scale": target_scale,
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
    session = get_session()
    inputs = prepare_inputs(request)

    # Run inference
    outputs = session.run(None, inputs)
    prediction = outputs[0]  # Shape: (1, 12, 7) - 7 quantiles

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
        "model": "Temporal Fusion Transformer (ONNX)",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)