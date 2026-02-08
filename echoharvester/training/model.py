"""ASR CTC models for Korean ASR training.

Supports two architectures:
1. Conformer CTC (standalone, 32.9M params default)
2. Zipformer2 CTC (vendored from k2-fsa/icefall)

Both share the same interface: (features, feature_lens) -> (log_probs, out_lens)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from echoharvester.training.config import (
    ConformerModelConfig,
    ZipformerModelConfig,
)


# ---------------------------------------------------------------------------
# Conformer CTC model (existing)
# ---------------------------------------------------------------------------


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (reduces time by factor of 4)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After two stride-2 convolutions: time -> time//4, freq -> freq//4
        self.linear = nn.Linear(output_dim * (input_dim // 4), output_dim)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, input_dim)
            x_lens: (batch,) original lengths

        Returns:
            out: (batch, time//4, output_dim)
            out_lens: (batch,) subsampled lengths
        """
        # (batch, time, freq) -> (batch, 1, time, freq)
        x = x.unsqueeze(1)
        x = self.conv(x)
        # (batch, channels, time//4, freq//4)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.linear(x)

        # Update lengths
        out_lens = ((x_lens - 1) // 2 + 1)
        out_lens = ((out_lens - 1) // 2 + 1)

        return x, out_lens


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def _extend_pe(self, length: int) -> None:
        """Extend positional encoding buffer if needed."""
        if length <= self.pe.size(1):
            return
        d_model = self.pe.size(2)
        pe = torch.zeros(length, d_model, device=self.pe.device)
        position = torch.arange(0, length, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=self.pe.device)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, time, d_model)"""
        self._extend_pe(x.size(1))
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvModule(nn.Module):
    """Conformer convolution module.

    Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, d_model)"""
        x = self.layer_norm(x)
        # (batch, time, d_model) -> (batch, d_model, time) for Conv1d
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        # (batch, d_model, time) -> (batch, time, d_model)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    """Feed-forward module with expansion factor."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class ConformerEncoderLayer(nn.Module):
    """Single Conformer encoder layer (Macaron-net structure).

    FF(1/2) -> MHSA -> ConvModule -> FF(1/2) -> LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConvModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, time, d_model)
            key_padding_mask: (batch, time), True for padded positions
        """
        # Macaron FF (1/2)
        x = x + 0.5 * self.ff1(x)

        # Multi-Head Self-Attention
        residual = x
        x_norm = self.self_attn_layer_norm(x)
        x_attn, _ = self.self_attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask,
        )
        x = residual + self.self_attn_dropout(x_attn)

        # Convolution Module
        x = x + self.conv_module(x)

        # Macaron FF (1/2)
        x = x + 0.5 * self.ff2(x)

        # Final LayerNorm
        x = self.final_layer_norm(x)

        return x


class ConformerCtcModel(nn.Module):
    """Conformer CTC model for ASR.

    Args:
        num_classes: Vocabulary size (including blank).
        input_dim: Input feature dimension (num_mel_bins).
        config: Model architecture configuration.
    """

    def __init__(self, num_classes: int, input_dim: int, config: ConformerModelConfig):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        d_model = config.attention_dim
        num_layers = config.num_encoder_layers
        num_heads = config.num_attention_heads
        d_ff = config.feedforward_dim
        kernel_size = config.depthwise_conv_kernel_size
        dropout = config.dropout

        self.subsampling = Conv2dSubsampling(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        self.encoder_layers = nn.ModuleList([
            ConformerEncoderLayer(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, time, input_dim) - Fbank features
            feature_lens: (batch,) - lengths before padding

        Returns:
            log_probs: (batch, time//4, num_classes) - log softmax output
            out_lens: (batch,) - output sequence lengths
        """
        x, out_lens = self.subsampling(features, feature_lens)
        x = self.pos_enc(x)

        # Create padding mask
        max_len = x.size(1)
        key_padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_lens.unsqueeze(1)

        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        logits = self.output_proj(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, out_lens

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Zipformer2 CTC model (vendored from icefall)
# ---------------------------------------------------------------------------


class ZipformerCtcModel(nn.Module):
    """Zipformer2 CTC model for ASR.

    Wraps the vendored Zipformer2 encoder with Conv2dSubsampling frontend.
    Provides the same interface as ConformerCtcModel:
        (features, feature_lens) -> (log_probs, out_lens)

    Args:
        num_classes: Vocabulary size (including blank).
        input_dim: Input feature dimension (num_mel_bins).
        config: Zipformer model configuration.
    """

    def __init__(self, num_classes: int, input_dim: int, config: ZipformerModelConfig):
        super().__init__()
        from echoharvester.training.zipformer.subsampling import (
            Conv2dSubsampling as ZipformerSubsampling,
        )
        from echoharvester.training.zipformer.zipformer import Zipformer2

        self.num_classes = num_classes
        self.input_dim = input_dim

        encoder_dims = config.encoder_dims
        output_dim = max(encoder_dims)

        # Conv2dSubsampling: (batch, T, input_dim) -> (batch, T', encoder_dims[0])
        # T' = (T - 7) // 2
        self.encoder_embed = ZipformerSubsampling(
            in_channels=input_dim,
            out_channels=encoder_dims[0],
            dropout=config.dropout,
        )

        # Zipformer2 encoder: (T', batch, encoder_dims[0]) -> (T'', batch, output_dim)
        self.encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=tuple(config.downsampling_factors),
            encoder_dim=tuple(encoder_dims),
            num_encoder_layers=tuple(config.encoder_layers),
            encoder_unmasked_dim=tuple(
                min(d, 256) for d in encoder_dims
            ),
            num_heads=tuple(config.heads),
            feedforward_dim=tuple(config.ff_dims),
            cnn_module_kernel=tuple(config.cnn_kernels),
            dropout=config.dropout,
        )

        self.output_proj = nn.Linear(output_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, time, input_dim) - Fbank features
            feature_lens: (batch,) - lengths before padding

        Returns:
            log_probs: (batch, time', num_classes) - log softmax output
            out_lens: (batch,) - output sequence lengths
        """
        # Subsampling: (batch, T, input_dim) -> (batch, T', encoder_dim[0])
        x, out_lens = self.encoder_embed(features, feature_lens)

        # Zipformer2 expects (seq_len, batch, dim) format
        x = x.permute(1, 0, 2)  # (batch, T', dim) -> (T', batch, dim)

        # Padding mask for Zipformer2: (batch, seq_len), True = masked
        src_key_padding_mask = torch.arange(
            x.size(0), device=x.device
        ).unsqueeze(0) >= out_lens.unsqueeze(1)

        # Encoder: (T', batch, dim) -> (T'', batch, output_dim)
        x, out_lens = self.encoder(x, out_lens, src_key_padding_mask)

        # Back to batch-first: (T'', batch, dim) -> (batch, T'', dim)
        x = x.permute(1, 0, 2)

        logits = self.output_proj(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, out_lens

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def create_model(
    num_classes: int,
    input_dim: int,
    config: ConformerModelConfig | ZipformerModelConfig,
) -> nn.Module:
    """Create an ASR model based on config type.

    Args:
        num_classes: Vocabulary size (including blank).
        input_dim: Input feature dimension (num_mel_bins).
        config: Model configuration (discriminated union on 'type' field).

    Returns:
        Model instance with interface: (features, feature_lens) -> (log_probs, out_lens)
    """
    if config.type == "conformer_ctc":
        return ConformerCtcModel(num_classes, input_dim, config)
    elif config.type == "zipformer_ctc":
        return ZipformerCtcModel(num_classes, input_dim, config)
    else:
        raise ValueError(f"Unknown model type: {config.type}")
