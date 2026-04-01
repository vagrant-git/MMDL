from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from mmdl_baseline.dataset.discovery import SessionRecord, discover_sessions
from mmdl_baseline.dataset.splits import build_session_split
from mmdl_baseline.dataset.windowed_dataset import MultiModalWindowDataset
from mmdl_baseline.models.factory import build_model
from mmdl_baseline.utils.io import ensure_dir, write_json
from mmdl_baseline.utils.metrics import compute_classification_metrics, save_confusion_matrix_figure
from mmdl_baseline.utils.reporting import append_markdown_section
from mmdl_baseline.utils.task import resolve_task


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def build_dataloaders(
    splits: Dict[str, List[SessionRecord]],
    config: Dict[str, object],
    modality: str,
) -> Tuple[Dict[str, MultiModalWindowDataset], Dict[str, DataLoader]]:
    datasets = {
        split_name: MultiModalWindowDataset(split_sessions, config, modality)
        for split_name, split_sessions in splits.items()
    }
    loaders: Dict[str, DataLoader] = {}
    train_dataset = datasets["train"]
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after windowing.")
    sampler = None
    shuffle = True
    if config.get("weighted_sampler"):
        sampler_mode = str(config.get("weighted_sampler_mode", "class_window")).lower()
        label_counts = Counter(idx.label for idx in train_dataset.window_indexes)
        if sampler_mode == "class_window":
            weights = [1.0 / label_counts[idx.label] for idx in train_dataset.window_indexes]
        elif sampler_mode == "class_session":
            session_window_counts = Counter(idx.session_id for idx in train_dataset.window_indexes)
            session_to_label = {idx.session_id: idx.label for idx in train_dataset.window_indexes}
            label_session_counts = Counter(session_to_label.values())
            weights = [
                1.0 / (label_session_counts[idx.label] * session_window_counts[idx.session_id])
                for idx in train_dataset.window_indexes
            ]
        else:
            raise ValueError(f"Unsupported weighted_sampler_mode: {sampler_mode}")
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    for split_name, dataset in datasets.items():
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=shuffle if split_name == "train" and sampler is None else False,
            sampler=sampler if split_name == "train" else None,
            num_workers=int(config["num_workers"]),
        )
    return datasets, loaders


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
    return output


def _resolve_audio_backbone(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "audio_encoder") and hasattr(model.audio_encoder, "encoder"):
        return model.audio_encoder.encoder
    return None


def configure_audio_backbone_freeze(
    model: nn.Module,
    freeze_backbone: bool,
    keep_bn_eval: bool = True,
) -> None:
    backbone = _resolve_audio_backbone(model)
    if backbone is None:
        return
    for param in backbone.parameters():
        param.requires_grad = not freeze_backbone
    if freeze_backbone and keep_bn_eval:
        backbone.eval()
        for module in backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()


def compute_loss_weights(train_sessions: List[SessionRecord], num_classes: int) -> torch.Tensor:
    counts = Counter(session.label for session in train_sessions)
    weights = []
    for label in range(num_classes):
        count = counts.get(label, 1)
        weights.append(1.0 / count)
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(
    config: Dict[str, object],
    class_weights: torch.Tensor | None,
) -> nn.Module:
    loss_name = str(config.get("loss", "cross_entropy")).lower()
    if loss_name == "focal":
        return FocalCrossEntropyLoss(
            gamma=float(config.get("focal_gamma", 2.0)),
            weight=class_weights,
        )
    return nn.CrossEntropyLoss(weight=class_weights)


def build_lr_scheduler(
    config: Dict[str, object],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = str(config.get("lr_scheduler", "none")).lower()
    if scheduler_name in {"", "none", "off", "disabled"}:
        return None
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(config.get("lr_scheduler_factor", 0.5)),
            patience=int(config.get("lr_scheduler_patience", 1)),
            min_lr=float(config.get("min_learning_rate", 1e-6)),
        )
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    grad_clip_norm: float | None = None,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    for batch in loader:
        labels = batch["label"].to(device)
        model_inputs = move_batch_to_device(batch, device)
        logits = model(model_inputs)
        loss = criterion(logits, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        losses.append(float(loss.item()))
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    metrics = compute_classification_metrics(y_true, y_pred, labels=list(range(num_classes)))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, object]]:
    model.eval()
    predictions: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            model_inputs = move_batch_to_device(batch, device)
            logits = model(model_inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            for i in range(labels.shape[0]):
                predictions.append(
                    {
                        "session_id": batch["session_id"][i],
                        "start_sec": float(batch["start_sec"][i].item()),
                        "true_label": int(labels[i].item()),
                        "pred_label": int(preds[i].item()),
                        "logits": [float(x) for x in logits[i].cpu().tolist()],
                        "probabilities": [float(x) for x in probs[i].cpu().tolist()],
                    }
                )
    return predictions


def write_training_log(path: str | Path, rows: List[Dict[str, object]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch",
            "split",
            "loss",
            "learning_rate",
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_session_predictions(predictions: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for item in predictions:
        grouped.setdefault(str(item["session_id"]), []).append(item)
    session_predictions: List[Dict[str, object]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    for session_id, items in sorted(grouped.items()):
        probs = np.asarray([item["probabilities"] for item in items], dtype=np.float32)
        avg_probs = probs.mean(axis=0)
        true_label = int(items[0]["true_label"])
        pred_label = int(np.argmax(avg_probs))
        session_predictions.append(
            {
                "session_id": session_id,
                "num_windows": len(items),
                "true_label": true_label,
                "pred_label": pred_label,
                "probabilities": avg_probs.tolist(),
            }
        )
        y_true.append(true_label)
        y_pred.append(pred_label)
    num_classes = max(max(y_true, default=0), max(y_pred, default=0)) + 1
    metrics = compute_classification_metrics(y_true, y_pred, labels=list(range(num_classes)))
    return session_predictions, metrics


def summarize_split(
    datasets: Dict[str, MultiModalWindowDataset],
    splits: Dict[str, List[SessionRecord]],
) -> Dict[str, Dict[str, object]]:
    return {
        split_name: {
            "num_sessions": len(split_sessions),
            "num_windows": len(datasets[split_name]),
            "label_distribution": dict(sorted(Counter(s.label for s in split_sessions).items())),
        }
        for split_name, split_sessions in splits.items()
    }


def train_and_evaluate_with_splits(
    config: Dict[str, object],
    modality: str,
    run_dir: str | Path,
    splits: Dict[str, List[SessionRecord]],
    experiment_name: str = "baseline",
    append_summary: bool = True,
) -> Dict[str, object]:
    run_dir = ensure_dir(run_dir)
    write_json(run_dir / "session_split.json", {k: [s.to_dict() for s in v] for k, v in splits.items()})

    datasets, loaders = build_dataloaders(splits, config, modality)
    split_summary = summarize_split(datasets, splits)
    write_json(run_dir / "split_summary.json", split_summary)
    num_classes = max(session.label for sessions in splits.values() for session in sessions) + 1
    class_names = config.get("task", {}).get("class_names") or [str(i) for i in range(num_classes)]

    device = choose_device()
    model = build_model(modality, config, num_classes=num_classes).to(device)
    class_weights = None
    if config.get("class_weight"):
        class_weights = compute_loss_weights(splits["train"], num_classes=num_classes).to(device)
    criterion = build_criterion(config, class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_lr_scheduler(config, optimizer)
    grad_clip_norm = float(config.get("grad_clip_norm", 0.0)) or None

    best_state = None
    best_metric = -1.0
    best_epoch = -1
    patience = int(config["early_stop_patience"])
    no_improve_epochs = 0
    training_rows: List[Dict[str, object]] = []
    freeze_audio_backbone_epochs = int(config.get("freeze_audio_backbone_epochs", 0))
    freeze_audio_backbone_bn_eval = bool(config.get("freeze_audio_backbone_bn_eval", True))
    previous_freeze_state: bool | None = None

    for epoch in range(1, int(config["epochs"]) + 1):
        current_freeze_state = epoch <= freeze_audio_backbone_epochs
        if current_freeze_state != previous_freeze_state:
            configure_audio_backbone_freeze(
                model,
                freeze_backbone=current_freeze_state,
                keep_bn_eval=freeze_audio_backbone_bn_eval,
            )
            previous_freeze_state = current_freeze_state
        train_metrics = run_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            num_classes=num_classes,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = (
            run_epoch(model, loaders["val"], None, criterion, device, num_classes=num_classes)
            if len(datasets["val"]) > 0
            else train_metrics
        )
        print(
            f"[{experiment_name}:{modality}] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        for split_name, metrics in [("train", train_metrics), ("val", val_metrics)]:
            training_rows.append(
                {
                    "epoch": epoch,
                    "split": split_name,
                    "loss": metrics["loss"],
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                }
            )
        current_score = float(val_metrics["macro_f1"])
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_score)
            else:
                scheduler.step()
        if current_score > best_metric:
            best_metric = current_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("No model checkpoint was captured during training.")

    torch.save(best_state, run_dir / "best_model.pt")
    write_training_log(run_dir / "training_log.csv", training_rows)

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, loaders["test"], None, criterion, device, num_classes=num_classes)
    print(
        f"[{experiment_name}:{modality}] best_epoch={best_epoch} "
        f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    predictions = collect_predictions(model, loaders["test"], device)
    session_predictions, session_metrics = aggregate_session_predictions(predictions)
    write_json(run_dir / "test_predictions.json", predictions)
    write_json(run_dir / "test_session_predictions.json", session_predictions)
    save_confusion_matrix_figure(
        np.asarray(test_metrics["confusion_matrix"]),
        labels=class_names,
        output_path=run_dir / "confusion_matrix.png",
        title=f"{modality} test confusion matrix",
    )
    save_confusion_matrix_figure(
        np.asarray(session_metrics["confusion_matrix"]),
        labels=class_names,
        output_path=run_dir / "confusion_matrix_session.png",
        title=f"{modality} test session confusion matrix",
    )
    summary = {
        "experiment_name": experiment_name,
        "modality": modality,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "task_info": {"num_classes": num_classes, "class_names": class_names},
        "split_summary": split_summary,
        "test_metrics_window": test_metrics,
        "test_metrics_session": session_metrics,
    }
    write_json(run_dir / "summary.json", summary)

    if append_summary:
        append_markdown_section(
            config["summary_markdown"],
            f"{experiment_name} | {modality}",
            [
                f"- python_env: `dl`",
                f"- model: `{modality}`",
                f"- method: 保持 baseline_5class 的预处理与模型结构不变，按 `{experiment_name}` 设定训练评估。",
                f"- result_window: acc={test_metrics['accuracy']:.4f}, macro-F1={test_metrics['macro_f1']:.4f}, precision={test_metrics['macro_precision']:.4f}, recall={test_metrics['macro_recall']:.4f}",
                f"- result_session: acc={session_metrics['accuracy']:.4f}, macro-F1={session_metrics['macro_f1']:.4f}, precision={session_metrics['macro_precision']:.4f}, recall={session_metrics['macro_recall']:.4f}",
            ],
        )
    return summary


def train_and_evaluate(
    config: Dict[str, object],
    modality: str,
    run_dir: str | Path,
) -> Dict[str, object]:
    sessions = discover_sessions(config["data_root"], config["labels"])
    sessions, task_info = resolve_task(config, sessions)
    config = {**config, "task": {**config.get("task", {}), **task_info}}
    splits = build_session_split(
        sessions=sessions,
        seed=int(config["seed"]),
        test_per_class=int(config["split"]["test_per_class"]),
        val_fraction_of_remaining=float(config["split"]["val_fraction_of_remaining"]),
    )
    return train_and_evaluate_with_splits(
        config=config,
        modality=modality,
        run_dir=run_dir,
        splits=splits,
        experiment_name="baseline_5class",
        append_summary=True,
    )
