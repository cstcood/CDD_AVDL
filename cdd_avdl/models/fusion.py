"""Convolution-layer fusion (Sec. III-D, TABLE II of the paper).

The reference implementation exposes three helpers -- patch_split, patch_conv
and patch_concat -- whose img_size arguments mix up (channels, length) and
(length, channels) and whose output widths are hard-coded at the call site
(73, 147, 296, 512).  This module keeps the same computation but derives every
number from the branch widths and the patch sizes.

Two patchification modes exist:

sequence  input (B, C, L)  ->  (B, L // C, ps, ps)
          The sequence is cut into floor(L / C) consecutive blocks of C
          samples; each block is a C x C tile that a strided convolution
          reduces to ps x ps.  The convolution also maps the C input channels
          onto the L // C output channels, so the patch count becomes the new
          channel dimension.

map       input (B, C, S, S)  ->  (B, C, ps, ps)
          An already patchified feature map is reduced spatially with a
          strided convolution that keeps the channel count.

A fusion stage concatenates the two patchified tensors along the channel axis,
and a 3x3 convolution then halves the channel count.  Everything is derived;
nothing is hard-coded.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = ["Patchify", "FusionStage", "ConvLayerFusion"]


class Patchify(nn.Module):
    """Turn a sequence or a feature map into a stack of square patches.

    Parameters
    ----------
    channels:
        Number of channels of the input tensor.
    patch_size:
        Side length of the produced patches.
    length:
        Length of the 1-D sequence.  Mutually exclusive with size.
    size:
        Spatial side length of the 2-D feature map.  Mutually exclusive with
        length.
    """

    def __init__(
        self,
        channels: int,
        patch_size: int,
        length: Optional[int] = None,
        size: Optional[int] = None,
        project: bool = True,
    ) -> None:
        super().__init__()
        if (length is None) == (size is None):
            raise ValueError("Patchify needs exactly one of length= or size=")
        self.channels = channels
        self.patch_size = patch_size
        self.mode = "sequence" if length is not None else "map"

        if self.mode == "sequence":
            if channels % patch_size != 0:
                raise ValueError(
                    "channels=%d is not divisible by patch_size=%d"
                    % (channels, patch_size)
                )
            if length < channels:
                raise ValueError(
                    "sequence length %d is shorter than the channel count %d; "
                    "the sequence patchifier needs L >= C" % (length, channels)
                )
            self.num_patches = length // channels
            self.length = length
            self.size = None
            self.project = True
            self.kernel = channels // patch_size
            # The patch count becomes the channel axis, so the strided
            # convolution maps num_patches -> num_patches.
            self.conv = nn.Conv2d(
                self.num_patches, self.num_patches, self.kernel, self.kernel
            )
            self.out_channels = self.num_patches
        else:
            if size % patch_size != 0:
                raise ValueError(
                    "map size %d is not divisible by patch_size=%d" % (size, patch_size)
                )
            self.num_patches = channels
            self.length = None
            self.size = size
            self.kernel = size // patch_size
            self.conv = nn.Conv2d(channels, channels, self.kernel, self.kernel)
            self.out_channels = channels
            # The reference patch_conv only runs its convolution when the
            # patchification actually shrinks the spatial size; a 1x1
            # projection is skipped.  project=False reproduces that.
            self.project = project

    def extra_repr(self) -> str:
        if self.mode == "sequence":
            return "sequence channels=%d length=%d -> %d patches of %dx%d" % (
                self.channels,
                self.length,
                self.num_patches,
                self.patch_size,
                self.patch_size,
            )
        return "map channels=%d size=%d -> %dx%d" % (
            self.channels,
            self.size,
            self.patch_size,
            self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "sequence":
            batch, channels, length = x.shape
            if channels != self.channels:
                raise ValueError(
                    "expected %d channels, got %d" % (self.channels, channels)
                )
            if length < self.num_patches * channels:
                raise ValueError(
                    "sequence length %d is too short for %d patches of %d samples"
                    % (length, self.num_patches, channels)
                )
            # (B, C, L) -> (B, C, N, C) -> (B, N, C, C); the tail beyond
            # N * C is discarded, exactly like the reference patch_split.
            x = x[:, :, : self.num_patches * channels]
            x = x.reshape(batch, channels, self.num_patches, channels)
            x = x.transpose(1, 2)
        else:
            if x.dim() != 4:
                raise ValueError("map patchifier expects a 4-D tensor, got %d-D" % x.dim())
            if x.shape[1] != self.channels:
                raise ValueError(
                    "expected %d channels, got %d" % (self.channels, x.shape[1])
                )
        if not self.project:
            return x
        return self.conv(x)


class FusionStage(nn.Module):
    """Patchify two tensors and concatenate them along the channel axis."""

    def __init__(self, left: Patchify, right: Patchify, patch_size: int) -> None:
        super().__init__()
        if left.patch_size != patch_size or right.patch_size != patch_size:
            raise ValueError("both patchifiers must produce %d x %d patches" % (patch_size, patch_size))
        self.left = left
        self.right = right
        self.patch_size = patch_size
        self.out_channels = left.out_channels + right.out_channels

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        y_left = self.left(x_left)
        y_right = self.right(x_right)
        if y_left.shape[2:] != y_right.shape[2:]:
            raise ValueError(
                "patch shapes differ: %s vs %s" % (tuple(y_left.shape[2:]), tuple(y_right.shape[2:]))
            )
        return torch.cat([y_left, y_right], dim=1)


class ConvLayerFusion(nn.Module):
    """The three-stage convolution-layer fusion of TABLE II.

    Stage k concatenates the k-th visual convolution block with either the
    audio sequence (k = 1) or the output of the previous stage (k > 1), then
    halves the channel count with a 3x3 convolution.  The last convolution
    projects to fusion_final_channels and the result is flattened.
    """

    def __init__(
        self,
        visual_channels: Sequence[int],
        visual_lengths: Sequence[int],
        audio_channels: int,
        audio_length: int,
        patch_sizes: Sequence[int] = (32, 32, 8),
        final_channels: int = 8,
    ) -> None:
        super().__init__()
        if not (len(visual_channels) == len(visual_lengths) == len(patch_sizes) == 3):
            raise ValueError("expected exactly three fusion stages")

        self.patch_sizes = tuple(patch_sizes)
        self.final_channels = final_channels

        ps1, ps2, ps3 = self.patch_sizes

        # ---- stage 1: visual conv block 1 + audio sequence ---------------
        self.stage1 = FusionStage(
            Patchify(visual_channels[0], ps1, length=visual_lengths[0]),
            Patchify(audio_channels, ps1, length=audio_length),
            ps1,
        )
        width1 = self.stage1.out_channels
        self.conv1 = nn.Conv2d(width1, width1 // 2, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)

        # ---- stage 2: visual conv block 2 + stage-1 feature map ----------
        self.stage2 = FusionStage(
            Patchify(visual_channels[1], ps2, length=visual_lengths[1]),
            Patchify(width1 // 2, ps2, size=ps1, project=ps1 > ps2),
            ps2,
        )
        width2 = self.stage2.out_channels
        self.conv2 = nn.Conv2d(width2, width2 // 2, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)

        # ---- stage 3: visual conv block 3 + stage-2 feature map ----------
        self.stage3 = FusionStage(
            Patchify(visual_channels[2], ps3, length=visual_lengths[2]),
            Patchify(width2 // 2, ps3, size=ps2, project=ps2 > ps3),
            ps3,
        )
        width3 = self.stage3.out_channels
        self.conv3 = nn.Conv2d(width3, width3 // 2, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(width3 // 2, final_channels, kernel_size=3, padding=1, stride=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.out_dim = final_channels * ps3 * ps3

    def extra_repr(self) -> str:
        return "patch_sizes=%s final_channels=%d out_dim=%d" % (
            self.patch_sizes,
            self.final_channels,
            self.out_dim,
        )

    def forward(
        self,
        visual_blocks: Sequence[torch.Tensor],
        audio_sequence: torch.Tensor,
    ) -> torch.Tensor:
        conv1, conv2, conv3 = visual_blocks

        y = self.stage1(conv1, audio_sequence)
        y = self.relu1(self.conv1(y))

        y = self.stage2(conv2, y)
        y = self.relu2(self.conv2(y))

        y = self.stage3(conv3, y)
        y = self.relu3(self.conv3(y))
        y = self.relu4(self.conv4(y))

        return torch.flatten(y, 1)
