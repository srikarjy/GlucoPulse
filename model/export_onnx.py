"""
Export trained TFT to ONNX for portable FastAPI serving.

pytorch_forecasting's TFT doesn't have a built-in ONNX export, so we export
the underlying PyTorch module directly. The model's `forward` expects the
TimeSeriesDataSet's encoded input dict; we wrap it to accept flat tensors
matching the FastAPI request schema.

We need to export with LSTM initial states as inputs to avoid them being
captured as constants during tracing.

Also performs INT8 quantization for optimized inference latency.
"""

import os
os.environ["TORCHVISION_DISABLE_BETA_TRANSFORMS"] = "1"

import pickle
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer

from train import CHECKPOINT_PATH, MODEL_DIR, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH


class TFTONNXWrapper(nn.Module):
    """Wrap TFT to accept flat tensors for ONNX export, including LSTM initial states."""

    def __init__(self, tft: TemporalFusionTransformer):
        super().__init__()
        self.tft = tft

    def forward(
        self,
        encoder_lengths: torch.Tensor,
        decoder_lengths: torch.Tensor,
        encoder_cont: torch.Tensor,
        encoder_cat: torch.Tensor,
        decoder_cont: torch.Tensor,
        decoder_cat: torch.Tensor,
        target_scale: torch.Tensor,
        # LSTM initial states - make them inputs so they're not captured as constants
        encoder_lstm_h0: torch.Tensor,
        encoder_lstm_c0: torch.Tensor,
        decoder_lstm_h0: torch.Tensor,
        decoder_lstm_c0: torch.Tensor,
    ):
        x = {
            "encoder_lengths": encoder_lengths,
            "decoder_lengths": decoder_lengths,
            "encoder_cont": encoder_cont,
            "encoder_cat": encoder_cat,
            "decoder_cont": decoder_cont,
            "decoder_cat": decoder_cat,
            "target_scale": target_scale,
        }
        out = self.tft(x)
        return out.prediction


def quantize_onnx_model(onnx_path: Path, quantized_path: Path) -> None:
    """Apply INT8 dynamic quantization to ONNX model."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"Quantizing {onnx_path} to INT8...")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {quantized_path}")
    
    # Validate quantized model
    quantized_model = onnx.load(str(quantized_path))
    onnx.checker.check_model(quantized_model)
    print("Quantized ONNX model validation passed")


def main():
    with open(MODEL_DIR / "training_dataset.pkl", "rb") as f:
        training = pickle.load(f)

    tft = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")
    tft.eval()

    # Get a real sample from the training dataloader to understand input shapes
    train_loader = training.to_dataloader(train=False, batch_size=1, num_workers=0)
    sample = next(iter(train_loader))
    x, _ = sample

    wrapper = TFTONNXWrapper(tft)
    wrapper.eval()

    # Create dummy initial states (zeros) - these will be inputs to the ONNX model
    encoder_lstm_h0 = torch.zeros(1, 64, 16)
    encoder_lstm_c0 = torch.zeros(1, 64, 16)
    decoder_lstm_h0 = torch.zeros(1, 64, 16)
    decoder_lstm_c0 = torch.zeros(1, 64, 16)

    # Export FP32 model first
    fp32_path = MODEL_DIR / "tft_fp32.onnx"
    torch.onnx.export(
        wrapper,
        (
            x["encoder_lengths"],
            x["decoder_lengths"],
            x["encoder_cont"],
            x["encoder_cat"],
            x["decoder_cont"],
            x["decoder_cat"],
            x["target_scale"],
            encoder_lstm_h0,
            encoder_lstm_c0,
            decoder_lstm_h0,
            decoder_lstm_c0,
        ),
        fp32_path,
        input_names=[
            "encoder_lengths",
            "decoder_lengths",
            "encoder_cont",
            "encoder_cat",
            "decoder_cont",
            "decoder_cat",
            "target_scale",
            "encoder_lstm_h0",
            "encoder_lstm_c0",
            "decoder_lstm_h0",
            "decoder_lstm_c0",
        ],
        output_names=["prediction"],
        dynamic_axes={
            "encoder_lengths": {0: "batch"},
            "decoder_lengths": {0: "batch"},
            "encoder_cont": {0: "batch", 1: "encoder_steps"},
            "encoder_cat": {0: "batch", 1: "encoder_steps"},
            "decoder_cont": {0: "batch", 1: "decoder_steps"},
            "decoder_cat": {0: "batch", 1: "decoder_steps"},
            "target_scale": {0: "batch"},
            "encoder_lstm_h0": {0: "batch", 1: "num_directions_layers"},
            "encoder_lstm_c0": {0: "batch", 1: "num_directions_layers"},
            "decoder_lstm_h0": {0: "batch", 1: "num_directions_layers"},
            "decoder_lstm_c0": {0: "batch", 1: "num_directions_layers"},
            "prediction": {0: "batch", 1: "decoder_steps"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported FP32 ONNX model to {fp32_path}")

    import onnx
    onnx_model = onnx.load(str(fp32_path))
    onnx.checker.check_model(onnx_model)
    print("FP32 ONNX model validation passed")

    # Apply INT8 dynamic quantization
    int8_path = MODEL_DIR / "tft.onnx"
    quantize_onnx_model(fp32_path, int8_path)

    check_onnx_parity(wrapper, int8_path, x, encoder_lstm_h0, encoder_lstm_c0, decoder_lstm_h0, decoder_lstm_c0)


def check_onnx_parity(
    wrapper: TFTONNXWrapper,
    onnx_path: Path,
    x: dict,
    encoder_lstm_h0: torch.Tensor,
    encoder_lstm_c0: torch.Tensor,
    decoder_lstm_h0: torch.Tensor,
    decoder_lstm_c0: torch.Tensor,
    atol: float = 5.0,
) -> None:
    """Compare INT8 ONNX output against the original PyTorch model on the same
    sample. INT8 dynamic quantization can silently change predictions -- this
    catches that instead of assuming export preserved behavior. atol is in
    mg/dL (the model's own output units), not a raw float tolerance, since
    that's the unit that actually matters for this task."""
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = wrapper(
            x["encoder_lengths"],
            x["decoder_lengths"],
            x["encoder_cont"],
            x["encoder_cat"],
            x["decoder_cont"],
            x["decoder_cat"],
            x["target_scale"],
            encoder_lstm_h0,
            encoder_lstm_c0,
            decoder_lstm_h0,
            decoder_lstm_c0,
        ).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # The LSTM initial-state args are never wired into TFTONNXWrapper.forward's
    # call to self.tft(x), so torch.onnx.export's constant-folding correctly
    # prunes them as dead inputs -- the exported graph only has these 7.
    onnx_out = session.run(
        None,
        {
            "encoder_lengths": x["encoder_lengths"].numpy(),
            "decoder_lengths": x["decoder_lengths"].numpy(),
            "encoder_cont": x["encoder_cont"].numpy(),
            "encoder_cat": x["encoder_cat"].numpy(),
            "decoder_cont": x["decoder_cont"].numpy(),
            "decoder_cat": x["decoder_cat"].numpy(),
            "target_scale": x["target_scale"].numpy(),
        },
    )[0]

    max_abs_diff = float(abs(torch_out - onnx_out).max())
    print(f"ONNX (INT8) vs PyTorch max abs diff: {max_abs_diff:.4f} mg/dL (tolerance: {atol} mg/dL)")
    assert max_abs_diff < atol, (
        f"INT8 ONNX export diverges from PyTorch model by {max_abs_diff:.4f} mg/dL, "
        f"exceeding the {atol} mg/dL tolerance -- quantization changed behavior."
    )
    print("ONNX parity check passed")


if __name__ == "__main__":
    main()