"""Typed configuration for CDD-AVDL.

The reference implementation hard-codes every architectural constant
(1190, 7161, 1192, 73, 147, 296, 512 ...).  Those numbers are not independent
hyper-parameters: they are all derived from a handful of physical quantities --
feature dimensions, sequence lengths and patch sizes.  This module keeps only
the physical quantities; every other number is computed at build time by
cdd_avdl.models.fusion.

Consequently the same code runs unchanged on any audio-visual classification
task, as long as the two feature extractors and the sequence lengths are
declared here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

__all__ = ["ModelConfig", "DataConfig", "TrainConfig", "Config"]


def _as_tuple(value: Any) -> Any:
    """YAML gives lists; the model wants hashable tuples."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


@dataclass
class ModelConfig:
    """Architecture hyper-parameters (Sec. III of the paper).

    Every field is a physical quantity.  Derived quantities -- patch counts,
    concatenated channel widths, the flattened fusion dimension -- are never
    configured, they are computed.
    """

    # ---- audio branch: wav2vec2 -> Conv1D -> AST -------------------------
    audio_in_dim: int = 768           # wav2vec2 hidden size
    audio_conv_out: int = 64          # Conv1D out channels (paper: 64)
    audio_conv_kernel: int = 2        # Conv1D kernel size (paper: 2)
    ast_out_dim: int = 8              # AST label_dim -> 8 audio features
    ast_model_size: str = "small224"  # tiny224 | small224 | base224 | base384
    ast_fstride: int = 10             # patch stride on the frequency axis
    ast_tstride: int = 10             # patch stride on the time axis
    ast_imagenet_pretrain: bool = True
    ast_audioset_pretrain: bool = False
    ast_pretrained_dir: str = "pretrained_models"

    # ---- visual branch: OpenGraphAU -> MLSTM-FCN -------------------------
    visual_in_dim: int = 41           # number of AUs fed to MLSTM-FCN
    mlstm_out_dim: int = 4            # MLSTM-FCN num_classes -> 4 visual features
    mlstm_conv_channels: Tuple[int, int, int] = (128, 64, 32)
    mlstm_conv_kernels: Tuple[int, int, int] = (8, 5, 3)
    mlstm_lstm_hidden: int = 16
    mlstm_lstm_layers: int = 5
    mlstm_dropout: float = 0.7
    mlstm_se_reduction: int = 16

    # ---- feature dimension reduction: sliding window (Eq. 2) -------------
    window_size: int = 10             # N in Eq. (2)
    window_stride: int = 10           # M in Eq. (2)

    # ---- convolution-layer fusion (TABLE II) -----------------------------
    fusion_patch_sizes: Tuple[int, int, int] = (32, 32, 8)
    fusion_final_channels: int = 8    # channels after the last fusion conv
    fusion_weight: float = 0.5        # weight of X_C in Eq. (4)

    # ---- classification head --------------------------------------------
    num_classes: int = 2              # DPs vs. NDPs

    def __post_init__(self) -> None:
        for name in ("mlstm_conv_channels", "mlstm_conv_kernels", "fusion_patch_sizes"):
            setattr(self, name, _as_tuple(getattr(self, name)))

    # ---- derived quantities ---------------------------------------------
    @property
    def audio_conv_out_len(self) -> int:
        """Length removed by the Conv1D of Sec. III-B."""
        return self.audio_conv_kernel - 1

    def validate(self, audio_len: int, visual_len: int) -> None:
        """Fail early with an actionable message instead of a shape error."""
        if audio_len - self.audio_conv_out_len <= 0:
            raise ValueError(
                "audio_len=%d is too short for a Conv1D with kernel=%d"
                % (audio_len, self.audio_conv_kernel)
            )
        if visual_len <= max(self.mlstm_conv_kernels):
            raise ValueError(
                "visual_len=%d is too short for MLSTM-FCN kernels %s"
                % (visual_len, self.mlstm_conv_kernels)
            )
        if visual_len < self.window_size:
            raise ValueError(
                "visual_len=%d is shorter than window_size=%d"
                % (visual_len, self.window_size)
            )
        widths = (self.audio_conv_out,) + tuple(self.mlstm_conv_channels)
        for patch in self.fusion_patch_sizes:
            for width in widths:
                if width % patch != 0:
                    raise ValueError(
                        "channel count %d is not divisible by fusion patch size %d; "
                        "patch sizes must divide every branch width" % (width, patch)
                    )


@dataclass
class DataConfig:
    """Where the pre-extracted features live and how they are padded."""

    root: str = "data"
    audio_dir: str = "audio"          # <root>/audio/<id>.npy  -> (T_a, 768)
    visual_dir: str = "visual"        # <root>/visual/<id>.npy -> (T_v, 41)
    label_file: str = "labels.csv"    # columns: id,label
    audio_len: int = 1193             # raw audio frames; Conv1D removes 1
    visual_len: int = 7168            # padded visual frames
    num_folds: int = 5                # K-fold cross-validation (paper: K=5)
    fold: int = 0
    num_workers: int = 4
    pin_memory: bool = True

    @property
    def audio_path(self) -> Path:
        return Path(self.root) / self.audio_dir

    @property
    def visual_path(self) -> Path:
        return Path(self.root) / self.visual_dir

    @property
    def label_path(self) -> Path:
        return Path(self.root) / self.label_file


@dataclass
class TrainConfig:
    """Optimisation settings (Sec. IV-B of the paper)."""

    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4                  # paper: 1e-4
    momentum: float = 0.9             # paper: SGD, momentum=0.9
    weight_decay: float = 0.0
    optimizer: str = "sgd"            # sgd | adamw
    scheduler: str = "cosine"         # none | cosine | step
    warmup_epochs: int = 0
    grad_clip: float = 0.0
    amp: bool = True
    seed: int = 42
    device: str = "cuda"              # cuda | cpu | mps
    output_dir: str = "runs"
    early_stop_patience: int = 0      # 0 disables early stopping
    monitor: str = "loss"             # loss | accuracy | f1
    log_interval: int = 10


@dataclass
class Config:
    """Top-level configuration: model + data + training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        raw = dict(raw or {})
        return cls(
            model=ModelConfig(**(raw.get("model") or {})),
            data=DataConfig(**(raw.get("data") or {})),
            train=TrainConfig(**(raw.get("train") or {})),
        )

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)

    def validate(self) -> None:
        self.model.validate(self.data.audio_len, self.data.visual_len)
