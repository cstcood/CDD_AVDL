"""Shape and derivation tests for CDD-AVDL.

Every width in the network is derived from the two sequence lengths, so the
same assertions must hold for the paper geometry and for other corpora.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdd_avdl.config import Config, DataConfig, ModelConfig
from cdd_avdl.models.cdd_avdl import CDDAVDL
from cdd_avdl.models.fusion import ConvLayerFusion, Patchify
from cdd_avdl.models.mlstm_fcn import MLSTMFCN


def make_config(audio_len: int, visual_len: int, **model_kwargs) -> Config:
    model_kwargs.setdefault("ast_imagenet_pretrain", False)
    return Config(
        model=ModelConfig(**model_kwargs),
        data=DataConfig(audio_len=audio_len, visual_len=visual_len),
    )


def test_paper_geometry():
    """The numbers reported in the paper must fall out of the configuration."""
    config = make_config(1193, 7168)
    model = CDDAVDL(config)
    info = model.describe()

    assert info["audio_conv_len"] == 1192
    assert info["visual_block_lengths"] == (7161, 7157, 7155)
    assert info["fusion_stage_channels"] == (73, 147, 296)
    assert info["fusion_out_dim"] == 512
    assert info["feature_dim"] == 12


def test_forward_shape():
    config = make_config(1193, 7168)
    model = CDDAVDL(config).eval()
    audio = torch.randn(2, 1193, 768)
    visual = torch.randn(2, 7168, 41)
    with torch.no_grad():
        output = model(audio, visual)
    assert output.shape == (2, 2)
    # the model returns log-probabilities
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize(
    "audio_len,visual_len",
    [(1193, 7168), (1200, 7200), (900, 6000)],
)
def test_other_geometries(audio_len: int, visual_len: int):
    config = make_config(audio_len, visual_len)
    model = CDDAVDL(config).eval()
    audio = torch.randn(1, audio_len, 768)
    visual = torch.randn(1, visual_len, 41)
    with torch.no_grad():
        output = model(audio, visual)
    assert output.shape == (1, 2)


def test_mlstm_out_lengths():
    mlstm = MLSTMFCN()
    assert mlstm.out_lengths(7168) == (7161, 7157, 7155)
    assert mlstm.out_lengths(100) == (93, 89, 87)


def test_patchify_sequence():
    patchify = Patchify(channels=64, patch_size=32, length=1192)
    assert patchify.num_patches == 18
    assert patchify.kernel == 2
    output = patchify(torch.randn(3, 64, 1192))
    assert output.shape == (3, 18, 32, 32)


def test_patchify_map_skips_identity_projection():
    """A 1x1 projection is skipped, matching the reference patch_conv."""
    patchify = Patchify(channels=36, patch_size=32, size=32, project=False)
    assert patchify.kernel == 1
    source = torch.randn(2, 36, 32, 32)
    assert torch.equal(patchify(source), source)


def test_fusion_out_dim():
    fusion = ConvLayerFusion(
        visual_channels=(128, 64, 32),
        visual_lengths=(7161, 7157, 7155),
        audio_channels=64,
        audio_length=1192,
    )
    assert fusion.out_dim == 512
    assert fusion.stage1.out_channels == 73
    assert fusion.stage2.out_channels == 147
    assert fusion.stage3.out_channels == 296


def test_parameter_count_is_stable():
    model = CDDAVDL(make_config(1193, 7168))
    assert model.num_parameters() == 24106680
