"""Train CDD-AVDL.

Examples
--------
Single fold:

    python scripts/train.py --config configs/default.yaml --fold 0

Five-fold cross validation:

    python scripts/train.py --config configs/default.yaml --cross-validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdd_avdl.config import Config
from cdd_avdl.engine.train import cross_validate, fit
from cdd_avdl.factory import build_fold_loaders, build_model
from cdd_avdl.utils.seed import set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CDD-AVDL")
    parser.add_argument("--config", required=True, help="path to a yaml config")
    parser.add_argument("--fold", type=int, default=None, help="fold index, 0-based")
    parser.add_argument("--cross-validate", action="store_true", help="run every fold")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--device", default=None, help="override train.device")
    parser.add_argument("--output-dir", default=None, help="override train.output_dir")
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = Config.from_yaml(args.config)

    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.device is not None:
        config.train.device = args.device
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.seed is not None:
        config.train.seed = args.seed
    if args.fold is not None:
        config.data.fold = args.fold
    config.validate()

    set_seed(config.train.seed)
    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")
    print("device: %s" % device)

    model = build_model(config)
    print(json.dumps(model.describe(), indent=2, default=str))

    loaders = build_fold_loaders(config)

    if args.cross_validate:
        report = cross_validate(
            loaders,
            lambda: build_model(config),
            config,
            device,
            Path(config.train.output_dir),
        )
        print(json.dumps(report["summary"], indent=2))
        return 0

    index = config.data.fold
    if not 0 <= index < len(loaders):
        raise SystemExit("fold %d is out of range for %d folds" % (index, len(loaders)))
    train_loader, val_loader = loaders[index]
    model = model.to(device)
    result = fit(model, train_loader, val_loader, config, device, Path(config.train.output_dir))
    print(json.dumps(result["best_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
