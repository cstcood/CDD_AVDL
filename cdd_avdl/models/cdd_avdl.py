"""The CDD-AVDL network (Sec. III of the paper).

Three branches are combined:

1. audio   -- wav2vec2 features -> Conv1D -> AST -> 8 audio features
2. visual  -- OpenGraphAU action units -> MLSTM-FCN -> 4 visual features
3. fusion  -- the three MLSTM-FCN convolution blocks are fused with the audio
              sequence in three stages (TABLE II) and classified

The two classification heads are averaged in the log-probability domain,
which is the code-level form of Eq. (4):

    F = LogSoftmax( Linear( Norm( Concat(X_M, X_A) + X_C / 2 ) ) )

Every intermediate width is derived from the configuration; the reference
implementation hard-coded 1190, 7161, 1192, 73, 147, 296 and 512.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from ..config import Config
from .ast import ASTModel
from .fusion import ConvLayerFusion
from .mlstm_fcn import MLSTMFCN

__all__ = ["CDDAVDL"]


class CDDAVDL(nn.Module):
    """Audio-visual depression classifier.

    Parameters
    ----------
    config:
        A fully populated Config.  The sequence lengths in config.data decide
        every derived width, so the same class works for any dataset.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        model_cfg = config.model
        data_cfg = config.data

        # ---- derived geometry --------------------------------------------
        self.audio_len = data_cfg.audio_len
        self.visual_len = data_cfg.visual_len
        self.audio_conv_len = self.audio_len - (model_cfg.audio_conv_kernel - 1)

        self.mlstm = MLSTMFCN(
            num_classes=model_cfg.mlstm_out_dim,
            num_features=model_cfg.visual_in_dim,
            num_lstm_out=model_cfg.mlstm_lstm_hidden,
            num_lstm_layers=model_cfg.mlstm_lstm_layers,
            conv_channels=model_cfg.mlstm_conv_channels,
            conv_kernels=model_cfg.mlstm_conv_kernels,
            lstm_drop_p=model_cfg.mlstm_dropout,
            fc_drop_p=model_cfg.mlstm_dropout,
            se_reduction=model_cfg.mlstm_se_reduction,
            window_size=model_cfg.window_size,
            window_stride=model_cfg.window_stride,
        )
        self.visual_block_lengths = self.mlstm.out_lengths(self.visual_len)

        # ---- audio branch -------------------------------------------------
        self.conv1d = nn.Conv1d(
            model_cfg.audio_in_dim,
            model_cfg.audio_conv_out,
            model_cfg.audio_conv_kernel,
        )
        self.norm_audio = nn.LayerNorm(model_cfg.audio_conv_out)
        self.ast = ASTModel(
            label_dim=model_cfg.ast_out_dim,
            fstride=model_cfg.ast_fstride,
            tstride=model_cfg.ast_tstride,
            input_fdim=model_cfg.audio_conv_out,
            input_tdim=self.audio_conv_len,
            imagenet_pretrain=model_cfg.ast_imagenet_pretrain,
            audioset_pretrain=model_cfg.ast_audioset_pretrain,
            model_size=model_cfg.ast_model_size,
        )

        # ---- convolution-layer fusion -------------------------------------
        self.fusion = ConvLayerFusion(
            visual_channels=model_cfg.mlstm_conv_channels,
            visual_lengths=self.visual_block_lengths,
            audio_channels=model_cfg.audio_conv_out,
            audio_length=self.audio_conv_len,
            patch_sizes=model_cfg.fusion_patch_sizes,
            final_channels=model_cfg.fusion_final_channels,
        )
        self.classifier_fusion = nn.Linear(self.fusion.out_dim, model_cfg.num_classes)

        # ---- early fusion head --------------------------------------------
        self.feature_dim = model_cfg.ast_out_dim + model_cfg.mlstm_out_dim
        self.norm_features = nn.LayerNorm(self.feature_dim)
        self.classifier_features = nn.Linear(self.feature_dim, model_cfg.num_classes)

        self.fusion_weight = model_cfg.fusion_weight

    # ---- introspection ---------------------------------------------------
    def describe(self) -> Dict[str, object]:
        """Every derived width, for logging and for the architecture docs."""
        return {
            "audio_len": self.audio_len,
            "audio_conv_len": self.audio_conv_len,
            "visual_len": self.visual_len,
            "visual_block_lengths": tuple(self.visual_block_lengths),
            "fusion_stage_channels": (
                self.fusion.stage1.out_channels,
                self.fusion.stage2.out_channels,
                self.fusion.stage3.out_channels,
            ),
            "fusion_out_dim": self.fusion.out_dim,
            "feature_dim": self.feature_dim,
            "num_parameters": self.num_parameters(),
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )

    # ---- forward ---------------------------------------------------------
    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        visual_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify a batch.

        Parameters
        ----------
        audio:
            wav2vec2 features of shape (B, audio_len, 768).
        visual:
            OpenGraphAU features of shape (B, visual_len, 41).
        visual_lengths:
            Optional true lengths of shape (B,).

        Returns
        -------
        Log-probabilities of shape (B, num_classes).
        """
        if audio.dim() != 3 or visual.dim() != 3:
            raise ValueError("audio and visual must both be (B, T, F) tensors")

        # ---- audio branch -------------------------------------------------
        x_audio = self.conv1d(audio.float().transpose(1, 2)).transpose(1, 2)
        x_audio = self.norm_audio(x_audio)
        audio_features = self.ast(x_audio)

        # ---- visual branch ------------------------------------------------
        visual_features, visual_blocks = self.mlstm(visual, visual_lengths)

        # ---- convolution-layer fusion -------------------------------------
        # The fusion patchifier expects channel-major sequences (B, C, L),
        # while the audio branch above works time-major (B, T, F).
        fused = self.fusion(visual_blocks, x_audio.transpose(1, 2))
        logits_fusion = self.classifier_fusion(fused)

        # ---- early fusion (Eq. 4) -----------------------------------------
        combined = torch.cat((audio_features, visual_features), dim=1)
        logits_features = self.classifier_features(self.norm_features(combined))

        weight = self.fusion_weight
        return nn.functional.log_softmax(
            logits_fusion * weight + logits_features * (1.0 - weight), dim=1
        )

    # ---- convenience -----------------------------------------------------
    @classmethod
    def from_example(
        cls,
        audio_len: int = 1193,
        visual_len: int = 7168,
        **model_kwargs,
    ) -> "CDDAVDL":
        """Build the paper configuration for a given pair of sequence lengths."""
        from ..config import DataConfig, ModelConfig

        config = Config(
            model=ModelConfig(**model_kwargs),
            data=DataConfig(audio_len=audio_len, visual_len=visual_len),
        )
        return cls(config)
