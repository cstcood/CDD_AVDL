"""Build models and dataloaders from a Config."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import FeatureDataset, read_label_file, split_folds
from .models.cdd_avdl import CDDAVDL
from .utils.seed import worker_init_fn

__all__ = ["build_model", "build_dataset", "build_dataloaders", "build_fold_loaders"]


def build_model(config: Config) -> CDDAVDL:
    """Instantiate CDD-AVDL for the sequence lengths in config.data."""
    return CDDAVDL(config)


def build_dataset(
    config: Config,
    rows: Sequence[dict],
    in_memory: bool = False,
) -> FeatureDataset:
    """Wrap a list of label rows in a FeatureDataset."""
    data_cfg = config.data
    return FeatureDataset(
        root=Path(data_cfg.root),
        rows=rows,
        audio_dir=data_cfg.audio_dir,
        visual_dir=data_cfg.visual_dir,
        audio_len=data_cfg.audio_len,
        visual_len=data_cfg.visual_len,
        in_memory=in_memory,
    )


def _loader(
    dataset: FeatureDataset,
    config: Config,
    shuffle: bool,
) -> DataLoader:
    data_cfg = config.data
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=bool(data_cfg.pin_memory),
        drop_last=False,
        worker_init_fn=worker_init_fn if data_cfg.num_workers else None,
    )


def build_fold_loaders(config: Config) -> List[Tuple[DataLoader, DataLoader]]:
    """Return one (train, validation) loader pair per cross-validation fold."""
    data_cfg = config.data
    rows = read_label_file(Path(data_cfg.root) / data_cfg.label_file)
    folds = split_folds(rows, data_cfg.num_folds, config.train.seed)

    pairs: List[Tuple[DataLoader, DataLoader]] = []
    for index in range(data_cfg.num_folds):
        validation_rows = folds[index]
        train_rows = [row for position, fold in enumerate(folds) if position != index for row in fold]
        if not train_rows or not validation_rows:
            raise ValueError("fold %d is empty; check the label file" % (index + 1))
        train_dataset = build_dataset(config, train_rows)
        validation_dataset = build_dataset(config, validation_rows)
        pairs.append((_loader(train_dataset, config, True), _loader(validation_dataset, config, False)))
    return pairs


def build_dataloaders(
    config: Config,
    fold: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Return the loaders of one fold (config.data.fold by default)."""
    pairs = build_fold_loaders(config)
    index = config.data.fold if fold is None else fold
    if not 0 <= index < len(pairs):
        raise ValueError(
            "fold %d is out of range for %d folds" % (index, len(pairs))
        )
    return pairs[index]
