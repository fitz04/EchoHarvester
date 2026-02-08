"""Export trained model to ONNX or TorchScript format."""

import logging
from pathlib import Path

import torch
import torch.nn as nn

from echoharvester.training.config import TrainingConfig
from echoharvester.training.model import create_model
from echoharvester.training.tokenizer import CharTokenizer
from echoharvester.training.utils import resolve_device

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export trained Conformer CTC model."""

    def __init__(self, config: TrainingConfig, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback

    def _emit(self, event: dict):
        """Emit a progress event if callback is set."""
        if self.progress_callback:
            self.progress_callback(event)

    def export(
        self,
        checkpoint_path: str,
        output_path: str | None = None,
        format: str = "onnx",
    ) -> Path:
        """Export model to the specified format.

        Args:
            checkpoint_path: Path to model checkpoint.
            output_path: Output file path (auto-generated if None).
            format: 'onnx' or 'torchscript'.

        Returns:
            Path to exported model file.
        """
        # Load tokenizer
        tokens_path = Path(self.config.data_dir) / "lang_char" / "tokens.txt"
        tokenizer = CharTokenizer(tokens_path)

        # Load model
        device = torch.device("cpu")  # Export on CPU
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        model = create_model(
            num_classes=tokenizer.vocab_size,
            input_dim=self.config.features.num_mel_bins,
            config=self.config.model,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        epoch = checkpoint.get("epoch", "unknown")

        # Default output path
        if output_path is None:
            exp_dir = Path(self.config.exp_dir)
            ext = "onnx" if format == "onnx" else "pt"
            output_path = str(exp_dir / f"model-epoch{epoch}.{ext}")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        self._emit({"event": "export_start", "format": format})

        if format == "onnx":
            self._export_onnx(model, output, tokenizer)
        elif format == "torchscript":
            self._export_torchscript(model, output)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'onnx' or 'torchscript'.")

        file_size_mb = output.stat().st_size / (1024 * 1024)
        logger.info(f"Model exported to {output} (format={format})")

        self._emit({
            "event": "export_complete",
            "output_path": str(output),
            "size_mb": round(file_size_mb, 1),
        })
        return output

    def _export_onnx(
        self, model: nn.Module, output: Path, tokenizer: CharTokenizer
    ) -> None:
        """Export to ONNX format.

        Requires: pip install onnxscript onnx
        """
        try:
            import onnxscript  # noqa: F401
        except ImportError:
            raise ImportError(
                "ONNX export requires 'onnxscript' package. "
                "Install it with: pip install onnxscript onnx"
            )

        # Create dummy input
        batch_size = 1
        max_time = 400  # ~4 seconds at 100fps
        input_dim = self.config.features.num_mel_bins

        dummy_features = torch.randn(batch_size, max_time, input_dim)
        dummy_lens = torch.tensor([max_time], dtype=torch.long)

        torch.onnx.export(
            model,
            (dummy_features, dummy_lens),
            str(output),
            input_names=["features", "feature_lens"],
            output_names=["log_probs", "out_lens"],
            dynamic_axes={
                "features": {0: "batch", 1: "time"},
                "feature_lens": {0: "batch"},
                "log_probs": {0: "batch", 1: "time"},
                "out_lens": {0: "batch"},
            },
            opset_version=17,
        )

        file_size_mb = output.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model exported: {output} ({file_size_mb:.1f} MB)")

    def _export_torchscript(self, model: nn.Module, output: Path) -> None:
        """Export to TorchScript format."""
        batch_size = 1
        max_time = 400
        input_dim = self.config.features.num_mel_bins

        dummy_features = torch.randn(batch_size, max_time, input_dim)
        dummy_lens = torch.tensor([max_time], dtype=torch.long)

        scripted = torch.jit.trace(
            model, (dummy_features, dummy_lens), check_trace=False
        )
        scripted.save(str(output))

        file_size_mb = output.stat().st_size / (1024 * 1024)
        logger.info(f"TorchScript model exported: {output} ({file_size_mb:.1f} MB)")
