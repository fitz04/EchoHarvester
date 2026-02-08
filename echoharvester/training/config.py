"""Training configuration models."""

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    """Train/val/test split configuration."""

    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    type: Literal["char"] = "char"


class FeatureConfig(BaseModel):
    """Audio feature configuration."""

    num_mel_bins: int = 80


class ConformerModelConfig(BaseModel):
    """Conformer CTC model configuration."""

    type: Literal["conformer_ctc"] = "conformer_ctc"
    attention_dim: int = 256
    num_encoder_layers: int = 12
    num_attention_heads: int = 4
    feedforward_dim: int = 2048
    depthwise_conv_kernel_size: int = 31
    dropout: float = 0.1


class ZipformerModelConfig(BaseModel):
    """Zipformer2 CTC model configuration.

    Multi-stack architecture: each parameter is a comma-separated string
    representing per-stack values (e.g. 6 stacks).
    """

    type: Literal["zipformer_ctc"] = "zipformer_ctc"
    encoder_dim: str = "192,256,384,512,384,256"
    num_encoder_layers: str = "2,2,3,4,3,2"
    num_heads: str = "4,4,4,8,4,4"
    feedforward_dim: str = "512,768,1024,1536,1024,768"
    cnn_module_kernel: str = "31,31,15,15,15,31"
    downsampling_factor: str = "1,2,4,8,4,2"
    dropout: float = 0.1

    def _parse_ints(self, s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",")]

    @property
    def encoder_dims(self) -> list[int]:
        return self._parse_ints(self.encoder_dim)

    @property
    def encoder_layers(self) -> list[int]:
        return self._parse_ints(self.num_encoder_layers)

    @property
    def heads(self) -> list[int]:
        return self._parse_ints(self.num_heads)

    @property
    def ff_dims(self) -> list[int]:
        return self._parse_ints(self.feedforward_dim)

    @property
    def cnn_kernels(self) -> list[int]:
        return self._parse_ints(self.cnn_module_kernel)

    @property
    def downsampling_factors(self) -> list[int]:
        return self._parse_ints(self.downsampling_factor)

    @property
    def output_dim(self) -> int:
        """Final encoder output dimension (last stack)."""
        return self.encoder_dims[-1]

    @property
    def max_encoder_dim(self) -> int:
        """Max encoder dim across stacks (for NoamScheduler d_model)."""
        return max(self.encoder_dims)


# Discriminated union on 'type' field
ModelConfig = Annotated[
    Union[ConformerModelConfig, ZipformerModelConfig],
    Field(discriminator="type"),
]


class TrainingParamsConfig(BaseModel):
    """Training hyperparameters."""

    num_epochs: int = 50
    max_duration: float = 200.0
    lr_factor: float = 2.5
    warm_step: int = 5000
    weight_decay: float = 1e-6
    clip_grad_norm: float = 5.0
    log_interval: int = 50
    valid_interval: int = 1
    keep_last_n: int = 5


class TrainingConfig(BaseModel):
    """Top-level training configuration."""

    shar_sources: list[str] = Field(default_factory=lambda: ["./output/shar"])
    data_dir: Path = Path("./training_data")
    exp_dir: Path = Path("./exp")
    split: SplitConfig = Field(default_factory=SplitConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ConformerModelConfig)
    training_params: TrainingParamsConfig = Field(default_factory=TrainingParamsConfig)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    def ensure_dirs(self):
        """Create training directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
