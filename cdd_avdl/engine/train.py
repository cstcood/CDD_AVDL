"""Training and evaluation loops.

The paper trains with SGD (momentum 0.9, learning rate 1e-4) for a fixed number
of epochs and reports loss, accuracy, TPR, FPR, precision/recall and F1 under
five-fold cross validation.  Everything here is driven by TrainConfig, so the
same code runs on the paper dataset and on any other corpus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import Config
from .metrics import Metrics, compute_metrics

__all__ = [
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "fit",
    "cross_validate",
]


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Create the optimiser named by config.train.optimizer."""
    train_cfg = config.train
    parameters = [p for p in model.parameters() if p.requires_grad]
    name = train_cfg.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=train_cfg.lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
    raise ValueError("unknown optimizer %r" % train_cfg.optimizer)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create the learning-rate schedule named by config.train.scheduler."""
    train_cfg = config.train
    name = (train_cfg.scheduler or "none").lower()
    if name in ("none", ""):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, train_cfg.epochs)
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
    raise ValueError("unknown scheduler %r" % train_cfg.scheduler)


def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> tuple:
    audio = batch["audio"].to(device, non_blocking=True)
    visual = batch["visual"].to(device, non_blocking=True)
    lengths = batch["visual_len"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return audio, visual, lengths, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """Run one training epoch and return the mean loss and accuracy."""
    model.train()
    train_cfg = config.train
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.time()

    for step, batch in enumerate(loader):
        audio, visual, lengths, labels = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = bool(train_cfg.amp) and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(audio, visual, lengths)
            loss = criterion(logits, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if train_cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if train_cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_samples += batch_size

        if train_cfg.log_interval and (step + 1) % train_cfg.log_interval == 0:
            print(
                "epoch %d step %d/%d loss %.4f"
                % (epoch + 1, step + 1, len(loader), float(loss.detach()))
            )

    elapsed = time.time() - started
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "seconds": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate a model and return the loss plus every reported metric."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions: List[int] = []
    targets: List[int] = []

    for batch in loader:
        audio, visual, lengths, labels = _move(batch, device)
        logits = model(audio, visual, lengths)
        loss = criterion(logits, labels)
        batch_size = labels.numel()
        total_loss += float(loss) * batch_size
        total_samples += batch_size
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    metrics: Metrics = compute_metrics(predictions, targets)
    result: Dict[str, object] = {"loss": total_loss / max(1, total_samples)}
    result.update(metrics.to_dict())
    return result


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Train for config.train.epochs and keep the best checkpoint."""
    train_cfg = config.train
    criterion = nn.NLLLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.amp) and device.type == "cuda")

    output_dir = Path(output_dir) if output_dir is not None else Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("inf")
    best_epoch = -1
    best_metrics: Dict[str, object] = {}
    patience = int(train_cfg.early_stop_patience or 0)
    stale = 0
    history: List[Dict[str, object]] = []

    for epoch in range(train_cfg.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config, scaler, epoch
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["loss"])
            else:
                scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "val_tpr": val_stats["tpr"],
            "val_fpr": val_stats["fpr"],
            "val_f1": val_stats["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            "epoch %d train_loss %.4f val_loss %.4f acc %.4f tpr %.4f fpr %.4f f1 %.4f"
            % (
                epoch + 1,
                record["train_loss"],
                record["val_loss"],
                record["val_accuracy"],
                record["val_tpr"],
                record["val_fpr"],
                record["val_f1"],
            )
        )

        score = val_stats["loss"] if train_cfg.monitor == "loss" else -float(val_stats["f1"])
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = dict(val_stats)
            stale = 0
            torch.save(
                {"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch + 1},
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if patience and stale >= patience:
                print("early stopping at epoch %d" % (epoch + 1))
                break

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "checkpoint": str(output_dir / "best.pt"),
    }


def cross_validate(
    fold_loaders: Sequence[tuple],
    model_factory,
    config: Config,
    device: torch.device,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Run five-fold cross validation and aggregate the per-fold metrics.

    fold_loaders is a sequence of (train_loader, val_loader) pairs and
    model_factory is a zero-argument callable returning a fresh model.
    """
    output_dir = Path(output_dir) if output_dir is not None else Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds: List[Dict[str, object]] = []
    for index, (train_loader, val_loader) in enumerate(fold_loaders):
        print("===== fold %d/%d =====" % (index + 1, len(fold_loaders)))
        model = model_factory().to(device)
        result = fit(
            model,
            train_loader,
            val_loader,
            config,
            device,
            output_dir / ("fold%d" % (index + 1)),
        )
        folds.append({"fold": index + 1, **result})

    keys = ("accuracy", "tpr", "fpr", "precision", "recall", "f1")
    summary = {}
    for key in keys:
        values = [float(fold["best_metrics"][key]) for fold in folds if fold["best_metrics"]]
        if values:
            summary[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    report = {"folds": folds, "summary": summary}
    with (output_dir / "cross_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
