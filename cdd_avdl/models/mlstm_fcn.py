"""MLSTM-FCN visual branch (Sec. III-C of the paper).

A three-block fully convolutional network runs in parallel with an LSTM over
window-averaged inputs; the two representations are concatenated and projected
to a small number of visual features (4 in the paper).

Compared with the reference implementation this version

* takes the sliding-window size and stride from the configuration instead of
  hard-coding unfold(1, 10, 10);
* exposes out_lengths() so the fusion stage can be built without guessing;
* makes the sequence lengths optional, so the module also works on padded
  batches without a length vector.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = ["SELayer", "MLSTMFCN"]


class SELayer(nn.Module):
    """Squeeze-and-excitation over the channel axis of a 1-D feature map."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.size()
        scale = self.avg_pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1)
        return x * scale.expand_as(x)


class MLSTMFCN(nn.Module):
    """Multivariate LSTM-FCN for the OpenGraphAU action-unit sequence.

    Parameters
    ----------
    num_classes:
        Number of output features (4 in the paper).
    num_features:
        Number of input action units (41 in the paper).
    conv_channels / conv_kernels:
        Widths and kernel sizes of the three convolutional blocks.
    window_size / window_stride:
        Sliding window used to average the input before the LSTM (Eq. 2).
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_features: int = 41,
        num_lstm_out: int = 16,
        num_lstm_layers: int = 5,
        conv_channels: Sequence[int] = (128, 64, 32),
        conv_kernels: Sequence[int] = (8, 5, 3),
        lstm_drop_p: float = 0.7,
        fc_drop_p: float = 0.7,
        se_reduction: int = 16,
        window_size: int = 10,
        window_stride: int = 10,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 3 or len(conv_kernels) != 3:
            raise ValueError("MLSTM-FCN expects exactly three convolutional blocks")

        self.num_classes = num_classes
        self.num_features = num_features
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers
        self.conv_channels = tuple(conv_channels)
        self.conv_kernels = tuple(conv_kernels)
        self.window_size = window_size
        self.window_stride = window_stride

        c1, c2, c3 = self.conv_channels
        k1, k2, k3 = self.conv_kernels

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=num_lstm_out,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.conv1 = nn.Conv1d(num_features, c1, k1)
        self.conv2 = nn.Conv1d(c1, c2, k2)
        self.conv3 = nn.Conv1d(c2, c3, k3)

        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)

        self.se1 = SELayer(c1, se_reduction)
        self.se2 = SELayer(c2, se_reduction)

        self.relu = nn.ReLU(inplace=True)
        self.lstm_drop = nn.Dropout(lstm_drop_p)
        self.conv_drop = nn.Dropout(fc_drop_p)

        self.fc = nn.Linear(c3 + num_lstm_out, num_classes)

    # ---- shape helpers ---------------------------------------------------
    def out_lengths(self, length: int) -> Tuple[int, int, int]:
        """Lengths of the three convolutional blocks for an input of length L."""
        k1, k2, k3 = self.conv_kernels
        l1 = length - (k1 - 1)
        l2 = l1 - (k2 - 1)
        l3 = l2 - (k3 - 1)
        if l3 <= 0:
            raise ValueError(
                "input length %d is too short for kernels %s" % (length, self.conv_kernels)
            )
        return l1, l2, l3

    def num_windows(self, length: int) -> int:
        """Number of sliding windows produced by the LSTM branch."""
        if length < self.window_size:
            raise ValueError(
                "input length %d is shorter than window_size=%d" % (length, self.window_size)
            )
        return (length - self.window_size) // self.window_stride + 1

    # ---- forward ---------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the visual branch.

        Parameters
        ----------
        x:
            Action-unit sequence of shape (B, T, F).
        seq_lens:
            Optional true lengths of shape (B,).  When given, the LSTM branch
            is packed so that padding does not influence the final state.

        Returns
        -------
        (logits, [conv1, conv2, conv3]) where the three feature maps have
        shapes (B, C1, L1), (B, C2, L2), (B, C3, L3).
        """
        if x.dim() != 3:
            raise ValueError("MLSTM-FCN expects (B, T, F), got %d-D" % x.dim())

        # ---- LSTM branch over window-averaged inputs ---------------------
        windows = x.unfold(1, self.window_size, self.window_stride)
        mean_windows = torch.mean(windows, dim=3)

        if seq_lens is None:
            lstm_out, _ = self.lstm(mean_windows)
        else:
            lengths = torch.clamp(
                seq_lens.to("cpu") // self.window_stride, min=1, max=mean_windows.shape[1]
            )
            packed = nn.utils.rnn.pack_padded_sequence(
                mean_windows, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, padding_value=0.0
            )
        lstm_feat = self.lstm_drop(lstm_out[:, -1, :])

        # ---- fully convolutional branch ----------------------------------
        y = x.transpose(2, 1)
        y = self.conv_drop(self.relu(self.bn1(self.conv1(y))))
        conv1 = self.se1(y)
        y = self.conv_drop(self.relu(self.bn2(self.conv2(conv1))))
        conv2 = self.se2(y)
        conv3 = self.conv_drop(self.relu(self.bn3(self.conv3(conv2))))
        conv_feat = torch.mean(conv3, 2)

        logits = self.fc(torch.cat((lstm_feat, conv_feat), dim=1))
        return logits, [conv1, conv2, conv3]
