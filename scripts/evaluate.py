"""Evaluate a trained CDD-AVDL checkpoint.

    python scripts/evaluate.py --config configs/default.yaml --checkpoint runs/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdd_avdl.config import Config
from cdd_avdl.engine.train import evaluate
from cdd_avdl.factory import build_dataloaders, build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate CDD-AVDL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.from_yaml(args.config)
    if args.device is not None:
        config.train.device = args.device
    if args.fold is not None:
        config.data.fold = args.fold
    config.validate()

    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")
    model = build_model(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    print("loaded checkpoint from epoch %s" % checkpoint.get("epoch", "?"))

    _, val_loader = build_dataloaders(config)
    result = evaluate(model, val_loader, nn.NLLLoss(), device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
