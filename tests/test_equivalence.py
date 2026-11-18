"""Numerical equivalence with the reference implementation.

tests/legacy holds the released net/ sources with three mechanical patches:
the timm version assertion is dropped, the autocast decorator is replaced by an
identity decorator, and the timm 0.4.5 module paths are updated.  Nothing that
affects the computation is changed.

The test transplants the reference weights into the refactored model and
asserts that both produce the same log-probabilities.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "tests" / "legacy"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEGACY))

from cdd_avdl.config import Config, DataConfig, ModelConfig
from cdd_avdl.models.cdd_avdl import CDDAVDL

WEIGHT_MAP = [
    ("Conv1d.", "conv1d."),
    ("layerNorma1.", "norm_audio."),
    ("layerNorma2.", "norm_features."),
    ("FN.", "classifier_features."),
    ("FN_pc.", "classifier_fusion."),
    ("astmodel.", "ast."),
    ("mlstmfun.", "mlstm."),
    ("pc1.pc1.conv.", "fusion.stage1.left.conv."),
    ("pc1.pc2.conv.", "fusion.stage1.right.conv."),
    ("pc_cov1.", "fusion.conv1."),
    ("pc2.pc1.conv.", "fusion.stage2.left.conv."),
    ("pc2.pc2.conv.", "fusion.stage2.right.conv."),
    ("pc_cov2.", "fusion.conv2."),
    ("pc3.pc1.conv.", "fusion.stage3.left.conv."),
    ("pc3.pc2.conv.", "fusion.stage3.right.conv."),
    ("pc_cov3.", "fusion.conv3."),
    ("pc_cov4.", "fusion.conv4."),
]


def translate(key: str):
    for old, new in WEIGHT_MAP:
        if key.startswith(old):
            return new + key[len(old):]
    return None


def load_legacy_model():
    spec = importlib.util.spec_from_file_location(
        "legacy_cdd_avdl", LEGACY / "CDD-AVDL.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CDD_AVDL()


def test_state_dict_keys_are_fully_mapped():
    torch.manual_seed(0)
    legacy = load_legacy_model()
    config = Config(
        model=ModelConfig(ast_imagenet_pretrain=False),
        data=DataConfig(audio_len=1193, visual_len=7168),
    )
    model = CDDAVDL(config)

    legacy_keys = set(legacy.state_dict())
    translated = {}
    unmapped = []
    for key, value in legacy.state_dict().items():
        new_key = translate(key)
        if new_key is None:
            unmapped.append(key)
        else:
            translated[new_key] = value

    assert unmapped == []
    assert set(translated) == set(model.state_dict())
    for key, value in translated.items():
        assert value.shape == model.state_dict()[key].shape, key


def test_forward_matches_reference():
    torch.manual_seed(0)
    legacy = load_legacy_model()
    config = Config(
        model=ModelConfig(ast_imagenet_pretrain=False),
        data=DataConfig(audio_len=1193, visual_len=7168),
    )
    model = CDDAVDL(config)

    translated = {
        translate(key): value for key, value in legacy.state_dict().items()
    }
    model.load_state_dict(translated, strict=True)

    # timm 1.x validates the patch-embed image size that the reference bypassed
    # with a global monkey patch, so restore the reference patch embed.
    import AST as legacy_ast

    restored = legacy_ast.PatchEmbed(
        img_size=(64, 1192), patch_size=16, in_chans=1, embed_dim=384
    )
    restored.proj = legacy.astmodel.v.patch_embed.proj
    restored.num_patches = 590
    legacy.astmodel.v.patch_embed = restored

    torch.manual_seed(123)
    audio = torch.randn(2, 1193, 768)
    visual = torch.randn(2, 7168, 41)
    lengths = torch.tensor([7168, 7000])

    legacy.eval()
    model.eval()
    with torch.no_grad():
        reference = legacy(audio, visual, lengths)
        candidate = model(audio, visual, lengths)

    assert torch.allclose(reference, candidate, atol=1e-5), (
        (reference - candidate).abs().max().item()
    )
