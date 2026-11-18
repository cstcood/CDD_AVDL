"""Audio Spectrogram Transformer branch (Sec. III-C of the paper).

The audio sequence produced by wav2vec2 and the Conv1D of Sec. III-B is fed to
a DeiT-style transformer whose patch embedding is re-shaped for a
single-channel, non-square input.  The classifier head is replaced by a small
linear layer that emits a handful of audio features (8 in the paper).

Compared with the reference implementation this version

* does not assert a specific timm version and does not monkey-patch
  timm.models.vision_transformer globally -- the patch embedding module is
  replaced on the instance instead, so several AST models can coexist;
* resolves the DeiT checkpoint name across timm 0.4.x and 1.x;
* drops the @autocast() decorator, leaving mixed precision to the training
  loop where it belongs;
* derives the number of patches from the actual input geometry.
"""

from __future__ import annotations

from typing import Optional, Tuple

import timm
import torch
from torch import nn

try:  # timm >= 0.9
    from timm.layers import to_2tuple, trunc_normal_
except ImportError:  # pragma: no cover - timm 0.4.x
    from timm.models.layers import to_2tuple, trunc_normal_

__all__ = ["PatchEmbed", "ASTModel"]

#: DeiT backbones, newest timm name first.
_DEIT_NAMES = {
    "tiny224": ("deit_tiny_distilled_patch16_224", "vit_deit_tiny_distilled_patch16_224"),
    "small224": ("deit_small_distilled_patch16_224", "vit_deit_small_distilled_patch16_224"),
    "base224": ("deit_base_distilled_patch16_224", "vit_deit_base_distilled_patch16_224"),
    "base384": ("deit_base_distilled_patch16_384", "vit_deit_base_distilled_patch16_384"),
}


class PatchEmbed(nn.Module):
    """Single-channel patch embedding without timm's image-size restriction."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def _create_deit(model_size: str, pretrained: bool) -> nn.Module:
    if model_size not in _DEIT_NAMES:
        raise ValueError(
            "model_size must be one of %s, got %r" % (sorted(_DEIT_NAMES), model_size)
        )
    last_error: Optional[Exception] = None
    for name in _DEIT_NAMES[model_size]:
        try:
            return timm.create_model(name, pretrained=pretrained)
        except Exception as error:  # noqa: BLE001 - name may not exist in this timm
            last_error = error
    raise RuntimeError(
        "could not create a DeiT backbone for model_size=%r with timm %s"
        % (model_size, getattr(timm, "__version__", "unknown"))
    ) from last_error


class ASTModel(nn.Module):
    """Audio Spectrogram Transformer with a configurable feature head.

    Parameters
    ----------
    label_dim:
        Number of output features (8 in the paper).
    fstride, tstride:
        Patch stride along the frequency and time axes.  A stride smaller than
        the 16x16 patch size produces overlapping patches.
    input_fdim:
        Number of frequency bins, i.e. the channel count of the audio sequence.
    input_tdim:
        Number of time frames of the audio sequence.
    """

    def __init__(
        self,
        label_dim: int = 8,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 64,
        input_tdim: int = 1190,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = False,
        model_size: str = "small224",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if audioset_pretrain:
            raise NotImplementedError(
                "AudioSet pretraining is not wired into this refactor; "
                "load the checkpoint manually and call load_state_dict."
            )

        self.label_dim = label_dim
        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim

        self.v = _create_deit(model_size, pretrained=imagenet_pretrain)
        self.original_num_patches = self.v.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim),
        )

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        self.f_dim = f_dim
        self.t_dim = t_dim
        num_patches = f_dim * t_dim

        # ---- replace the patch embedding on this instance only -----------
        new_proj = nn.Conv2d(
            1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride)
        )
        if imagenet_pretrain:
            with torch.no_grad():
                new_proj.weight.copy_(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias.copy_(self.v.patch_embed.proj.bias)

        patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim),
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=self.original_embedding_dim,
        )
        patch_embed.num_patches = num_patches
        patch_embed.proj = new_proj
        self.v.patch_embed = patch_embed

        # ---- positional embedding ----------------------------------------
        if imagenet_pretrain:
            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, self.original_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
            )
            if t_dim <= self.original_hw:
                start = int(self.original_hw / 2) - int(t_dim / 2)
                new_pos_embed = new_pos_embed[:, :, :, start : start + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(self.original_hw, t_dim), mode="bilinear"
                )
            if f_dim <= self.original_hw:
                start = int(self.original_hw / 2) - int(f_dim / 2)
                new_pos_embed = new_pos_embed[:, :, start : start + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                )
            new_pos_embed = new_pos_embed.reshape(
                1, self.original_embedding_dim, num_patches
            ).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )
        else:
            new_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 2, self.original_embedding_dim)
            )
            trunc_normal_(new_pos_embed, std=0.02)
            self.v.pos_embed = new_pos_embed

        if verbose:
            print(
                "AST: fstride=%d tstride=%d patches=%d (f=%d, t=%d)"
                % (fstride, tstride, num_patches, f_dim, t_dim)
            )

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def get_shape(
        fstride: int, tstride: int, input_fdim: int = 128, input_tdim: int = 1024
    ) -> Tuple[int, int]:
        """Number of patches along the frequency and time axes."""
        f_dim = (input_fdim - 16) // fstride + 1
        t_dim = (input_tdim - 16) // tstride + 1
        if f_dim <= 0 or t_dim <= 0:
            raise ValueError(
                "input_fdim=%d / input_tdim=%d are too small for 16x16 patches"
                % (input_fdim, input_tdim)
            )
        return f_dim, t_dim

    @property
    def num_patches(self) -> int:
        return self.f_dim * self.t_dim

    # ---- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the audio branch.

        Parameters
        ----------
        x:
            Audio sequence of shape (B, T, F) where F is the number of
            frequency bins (the Conv1D output channels).

        Returns
        -------
        Tensor of shape (B, label_dim).
        """
        if x.dim() != 3:
            raise ValueError("ASTModel expects (B, T, F), got %d-D" % x.dim())
        if x.shape[2] != self.input_fdim:
            raise ValueError(
                "expected %d frequency bins, got %d" % (self.input_fdim, x.shape[2])
            )

        x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, F, T)
        batch = x.shape[0]

        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(batch, -1, -1)
        dist_token = self.v.dist_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for block in self.v.blocks:
            x = block(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        return self.mlp_head(x)
