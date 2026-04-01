from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

from grouped_cv import build_grouped_cv_splits
from mmdl_baseline.dataset.discovery import SessionRecord, discover_sessions
from mmdl_baseline.train_eval import train_and_evaluate_with_splits
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import ensure_dir, read_json, write_json
from mmdl_baseline.utils.seed import set_seed
from mmdl_baseline.utils.task import resolve_task


CSV_FIELDS = [
    "timestamp",
    "trial_id",
    "phase",
    "status",
    "subset_mask",
    "num_selected",
    "selected_ids",
    "dropped_ids",
    "label_counts",
    "duration_sec",
    "primary_metric_name",
    "primary_metric_mean",
    "primary_metric_std",
    "session_accuracy_mean",
    "session_accuracy_std",
    "session_macro_f1_mean",
    "session_macro_f1_std",
    "window_accuracy_mean",
    "window_macro_f1_mean",
    "score",
    "return_code",
    "error",
    "trial_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search the cleanest 0/2/4 session subset for multimodal training. "
            "The parent process schedules trials and the worker subprocess evaluates one subset at a time."
        )
    )
    parser.add_argument("--config", default="configs/chapter4_024.yaml")
    parser.add_argument("--output-dir", default="outputs/clean_subset_search_024")
    parser.add_argument("--modality", default="multimodal")
    parser.add_argument(
        "--primary-metric",
        default="session_accuracy",
        choices=["session_accuracy", "session_macro_f1", "window_accuracy", "window_macro_f1"],
        help="Metric used inside the composite score.",
    )
    parser.add_argument(
        "--alpha-per-folder",
        type=float,
        default=0.01,
        help="Composite score = metric + alpha_per_folder * num_selected.",
    )
    parser.add_argument("--max-trials", type=int, default=120, help="Number of new trials to run in this invocation. Use 0 for no limit.")
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for faster search.")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--min-per-class", type=int, default=None, help="Defaults to n_splits.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-fold-artifacts", action="store_true", help="Keep per-fold checkpoints/predictions/figures instead of pruning them.")
    parser.add_argument("--initial-temperature", type=float, default=0.05)
    parser.add_argument("--cooling-rate", type=float, default=0.97)
    parser.add_argument("--restart-every", type=int, default=12)
    parser.add_argument("--hidden-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--subset-json", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    parser.add_argument("--trial-dir", help=argparse.SUPPRESS)
    parser.add_argument("--trial-id", help=argparse.SUPPRESS)
    parser.add_argument("--phase", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_search_context(config_path: str | Path, n_splits_override: int | None) -> Dict[str, object]:
    config = load_config(config_path)
    sessions = discover_sessions(config["data_root"], config["labels"])
    sessions, task_info = resolve_task(config, sessions)
    config = {**config, "task": {**config.get("task", {}), **task_info}}
    ordered_sessions = sorted(sessions, key=lambda x: x.session_id)
    ordered_ids = [session.session_id for session in ordered_sessions]
    id_to_session = {session.session_id: session for session in ordered_sessions}
    n_splits = int(n_splits_override or config["grouped_cv"]["n_splits"])
    return {
        "config": config,
        "sessions": ordered_sessions,
        "ordered_ids": ordered_ids,
        "id_to_session": id_to_session,
        "n_splits": n_splits,
        "n_repeats": int(config["grouped_cv"]["n_repeats"]),
        "task_info": task_info,
    }


def subset_mask_from_ids(selected_ids: Sequence[str], ordered_ids: Sequence[str]) -> str:
    selected = set(selected_ids)
    return "".join("1" if session_id in selected else "0" for session_id in ordered_ids)


def selected_ids_from_mask(mask: str, ordered_ids: Sequence[str]) -> List[str]:
    return [session_id for keep, session_id in zip(mask, ordered_ids) if keep == "1"]


def label_counts_for_ids(selected_ids: Sequence[str], id_to_session: Dict[str, SessionRecord]) -> Dict[int, int]:
    counts = Counter(id_to_session[session_id].label for session_id in selected_ids)
    return dict(sorted(counts.items()))


def is_subset_valid(selected_ids: Sequence[str], id_to_session: Dict[str, SessionRecord], min_per_class: int) -> tuple[bool, str]:
    counts = label_counts_for_ids(selected_ids, id_to_session)
    present_labels = {session.label for session in id_to_session.values()}
    missing = [label for label in sorted(present_labels) if counts.get(label, 0) < min_per_class]
    if missing:
        return False, f"class counts below min_per_class={min_per_class}: {counts}"
    return True, ""


def compute_score(metric_value: float, num_selected: int, alpha_per_folder: float) -> float:
    return float(metric_value + alpha_per_folder * num_selected)


def is_better_result(candidate: Dict[str, object] | None, incumbent: Dict[str, object] | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    score_a = float(candidate["score"])
    score_b = float(incumbent["score"])
    if not math.isclose(score_a, score_b, rel_tol=0.0, abs_tol=1e-12):
        return score_a > score_b
    num_a = int(candidate["num_selected"])
    num_b = int(incumbent["num_selected"])
    if num_a != num_b:
        return num_a > num_b
    metric_a = float(candidate["primary_metric_mean"])
    metric_b = float(incumbent["primary_metric_mean"])
    if not math.isclose(metric_a, metric_b, rel_tol=0.0, abs_tol=1e-12):
        return metric_a > metric_b
    return str(candidate["subset_mask"]) < str(incumbent["subset_mask"])


def ensure_csv(path: Path) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    ensure_csv(path)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def parse_float(value: str, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_existing_cache(csv_path: Path, retry_failed: bool) -> Dict[str, Dict[str, object]]:
    cache: Dict[str, Dict[str, object]] = {}
    if not csv_path.exists():
        return cache
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mask = row["subset_mask"]
            status = row["status"]
            if status == "failed" and retry_failed:
                continue
            cache[mask] = {
                "timestamp": row["timestamp"],
                "trial_id": row["trial_id"],
                "phase": row["phase"],
                "status": status,
                "subset_mask": mask,
                "num_selected": parse_int(row["num_selected"]),
                "selected_ids": row["selected_ids"].split(";") if row["selected_ids"] else [],
                "dropped_ids": row["dropped_ids"].split(";") if row["dropped_ids"] else [],
                "label_counts": row["label_counts"],
                "duration_sec": parse_float(row["duration_sec"]),
                "primary_metric_name": row["primary_metric_name"],
                "primary_metric_mean": parse_float(row["primary_metric_mean"]),
                "primary_metric_std": parse_float(row["primary_metric_std"]),
                "session_accuracy_mean": parse_float(row["session_accuracy_mean"]),
                "session_accuracy_std": parse_float(row["session_accuracy_std"]),
                "session_macro_f1_mean": parse_float(row["session_macro_f1_mean"]),
                "session_macro_f1_std": parse_float(row["session_macro_f1_std"]),
                "window_accuracy_mean": parse_float(row["window_accuracy_mean"]),
                "window_macro_f1_mean": parse_float(row["window_macro_f1_mean"]),
                "score": parse_float(row["score"]),
                "return_code": parse_int(row["return_code"]),
                "error": row["error"],
                "trial_dir": row["trial_dir"],
            }
    return cache


def count_existing_trials(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in csv.DictReader(f)), 0)


def build_parent_state(
    ordered_ids: Sequence[str],
    id_to_session: Dict[str, SessionRecord],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    return {
        "version": 1,
        "ordered_ids": list(ordered_ids),
        "labels_by_id": {session_id: int(id_to_session[session_id].label) for session_id in ordered_ids},
        "search_args": {
            "config": args.config,
            "modality": args.modality,
            "primary_metric": args.primary_metric,
            "alpha_per_folder": args.alpha_per_folder,
            "n_splits": args.n_splits,
            "n_repeats": args.n_repeats,
            "window_sec": args.window_sec,
            "epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "min_per_class": args.min_per_class,
        },
        "paths": {
            "csv": str(output_dir / "trials.csv"),
            "best_json": str(output_dir / "best_subset.json"),
        },
    }


def write_state(
    state_path: Path,
    base_state: Dict[str, object],
    cache: Dict[str, Dict[str, object]],
    best_result: Dict[str, object] | None,
    phase: str,
    current_mask: str,
    temperature: float,
    new_trials: int,
) -> None:
    success_results = [row for row in cache.values() if row["status"] == "success"]
    ranked = sorted(success_results, key=lambda row: (float(row["score"]), int(row["num_selected"]), float(row["primary_metric_mean"])), reverse=True)
    payload = {
        **base_state,
        "phase": phase,
        "current_mask": current_mask,
        "temperature": temperature,
        "new_trials_this_run": new_trials,
        "num_unique_masks": len(cache),
        "num_successful_masks": len(success_results),
        "best_result": best_result,
        "top5": ranked[:5],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(state_path, payload)


def write_best_json(path: Path, row: Dict[str, object] | None) -> None:
    if row is None:
        return
    write_json(path, row)


def trial_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def has_budget(max_trials: int, new_trials: int) -> bool:
    return max_trials <= 0 or new_trials < max_trials


def label_counts_to_str(label_counts: Dict[int, int]) -> str:
    return ",".join(f"{label}:{count}" for label, count in sorted(label_counts.items()))


def row_from_result(
    result: Dict[str, object],
    ordered_ids: Sequence[str],
    id_to_session: Dict[str, SessionRecord],
    return_code: int,
) -> Dict[str, object]:
    selected_ids = list(result.get("selected_ids", []))
    dropped_ids = [session_id for session_id in ordered_ids if session_id not in set(selected_ids)]
    label_counts = label_counts_for_ids(selected_ids, id_to_session)
    return {
        "timestamp": trial_timestamp(),
        "trial_id": result.get("trial_id", ""),
        "phase": result.get("phase", ""),
        "status": result.get("status", "failed"),
        "subset_mask": result.get("subset_mask", ""),
        "num_selected": len(selected_ids),
        "selected_ids": ";".join(selected_ids),
        "dropped_ids": ";".join(dropped_ids),
        "label_counts": label_counts_to_str(label_counts),
        "duration_sec": float(result.get("duration_sec", -1.0)),
        "primary_metric_name": result.get("primary_metric_name", ""),
        "primary_metric_mean": float(result.get("primary_metric_mean", -1.0)),
        "primary_metric_std": float(result.get("primary_metric_std", -1.0)),
        "session_accuracy_mean": float(result.get("session_accuracy_mean", -1.0)),
        "session_accuracy_std": float(result.get("session_accuracy_std", -1.0)),
        "session_macro_f1_mean": float(result.get("session_macro_f1_mean", -1.0)),
        "session_macro_f1_std": float(result.get("session_macro_f1_std", -1.0)),
        "window_accuracy_mean": float(result.get("window_accuracy_mean", -1.0)),
        "window_macro_f1_mean": float(result.get("window_macro_f1_mean", -1.0)),
        "score": float(result.get("score", -1.0)),
        "return_code": return_code,
        "error": str(result.get("error", "")),
        "trial_dir": str(result.get("trial_dir", "")),
    }


def build_failure_row(
    trial_id: str,
    phase: str,
    mask: str,
    ordered_ids: Sequence[str],
    id_to_session: Dict[str, SessionRecord],
    duration_sec: float,
    error: str,
    return_code: int,
    primary_metric_name: str,
    trial_dir: Path,
) -> Dict[str, object]:
    selected_ids = selected_ids_from_mask(mask, ordered_ids)
    dropped_ids = [session_id for session_id in ordered_ids if session_id not in set(selected_ids)]
    return {
        "timestamp": trial_timestamp(),
        "trial_id": trial_id,
        "phase": phase,
        "status": "failed",
        "subset_mask": mask,
        "num_selected": len(selected_ids),
        "selected_ids": ";".join(selected_ids),
        "dropped_ids": ";".join(dropped_ids),
        "label_counts": label_counts_to_str(label_counts_for_ids(selected_ids, id_to_session)),
        "duration_sec": duration_sec,
        "primary_metric_name": primary_metric_name,
        "primary_metric_mean": -1.0,
        "primary_metric_std": -1.0,
        "session_accuracy_mean": -1.0,
        "session_accuracy_std": -1.0,
        "session_macro_f1_mean": -1.0,
        "session_macro_f1_std": -1.0,
        "window_accuracy_mean": -1.0,
        "window_macro_f1_mean": -1.0,
        "score": -1.0,
        "return_code": return_code,
        "error": error,
        "trial_dir": str(trial_dir),
    }


def ranked_success_rows(cache: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = [row for row in cache.values() if row["status"] == "success"]
    rows.sort(key=lambda row: (float(row["score"]), int(row["num_selected"]), float(row["primary_metric_mean"])), reverse=True)
    return rows


def maybe_evaluate_mask(
    mask: str,
    phase: str,
    args: argparse.Namespace,
    ordered_ids: Sequence[str],
    id_to_session: Dict[str, SessionRecord],
    min_per_class: int,
    cache: Dict[str, Dict[str, object]],
    csv_path: Path,
    output_dir: Path,
    next_trial_index: int,
) -> tuple[Dict[str, object] | None, int]:
    if mask in cache:
        return cache[mask], 0

    selected_ids = selected_ids_from_mask(mask, ordered_ids)
    is_valid, reason = is_subset_valid(selected_ids, id_to_session, min_per_class)
    trial_id = f"trial_{next_trial_index:06d}"
    trial_dir = ensure_dir(output_dir / "trials" / trial_id)
    if not is_valid:
        row = {
            "timestamp": trial_timestamp(),
            "trial_id": trial_id,
            "phase": phase,
            "status": "invalid",
            "subset_mask": mask,
            "num_selected": len(selected_ids),
            "selected_ids": ";".join(selected_ids),
            "dropped_ids": ";".join([session_id for session_id in ordered_ids if session_id not in set(selected_ids)]),
            "label_counts": label_counts_to_str(label_counts_for_ids(selected_ids, id_to_session)),
            "duration_sec": 0.0,
            "primary_metric_name": args.primary_metric,
            "primary_metric_mean": -1.0,
            "primary_metric_std": -1.0,
            "session_accuracy_mean": -1.0,
            "session_accuracy_std": -1.0,
            "session_macro_f1_mean": -1.0,
            "session_macro_f1_std": -1.0,
            "window_accuracy_mean": -1.0,
            "window_macro_f1_mean": -1.0,
            "score": -1.0,
            "return_code": -1,
            "error": reason,
            "trial_dir": str(trial_dir),
        }
        cache[mask] = row
        append_csv_row(csv_path, row)
        return row, 1

    subset_payload = {"selected_ids": selected_ids}
    subset_json = trial_dir / "subset.json"
    result_json = trial_dir / "worker_result.json"
    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    write_json(subset_json, subset_payload)
    command = [
        args.python_executable,
        str(Path(__file__).resolve()),
        "--hidden-worker",
        "--config",
        args.config,
        "--subset-json",
        str(subset_json),
        "--result-json",
        str(result_json),
        "--trial-dir",
        str(trial_dir),
        "--trial-id",
        trial_id,
        "--phase",
        phase,
        "--modality",
        args.modality,
        "--primary-metric",
        args.primary_metric,
        "--alpha-per-folder",
        str(args.alpha_per_folder),
        "--seed",
        str(args.seed),
    ]
    optional_pairs = [
        ("--n-splits", args.n_splits),
        ("--n-repeats", args.n_repeats),
        ("--window-sec", args.window_sec),
        ("--epochs", args.epochs),
        ("--early-stop-patience", args.early_stop_patience),
        ("--min-per-class", args.min_per_class),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            command.extend([flag, str(value)])
    if args.keep_fold_artifacts:
        command.append("--keep-fold-artifacts")

    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            stdout=stdout_f,
            stderr=stderr_f,
            check=False,
        )
    duration_sec = time.time() - started

    if completed.returncode == 0 and result_json.exists():
        result = read_json(result_json)
        row = row_from_result(result, ordered_ids, id_to_session, completed.returncode)
        cache[mask] = row
        append_csv_row(csv_path, row)
        return row, 1

    error_text = f"worker exited with code {completed.returncode}"
    if result_json.exists():
        try:
            result = read_json(result_json)
            if result.get("error"):
                error_text = str(result["error"])
        except Exception:
            pass
    else:
        try:
            error_text = stderr_path.read_text(encoding="utf-8")[-4000:] or error_text
        except OSError:
            pass
    row = build_failure_row(
        trial_id=trial_id,
        phase=phase,
        mask=mask,
        ordered_ids=ordered_ids,
        id_to_session=id_to_session,
        duration_sec=duration_sec,
        error=error_text.strip(),
        return_code=completed.returncode,
        primary_metric_name=args.primary_metric,
        trial_dir=trial_dir,
    )
    cache[mask] = row
    append_csv_row(csv_path, row)
    return row, 1


def remove_one_neighbors(mask: str, ordered_ids: Sequence[str], id_to_session: Dict[str, SessionRecord], min_per_class: int) -> List[str]:
    selected_ids = selected_ids_from_mask(mask, ordered_ids)
    counts = label_counts_for_ids(selected_ids, id_to_session)
    neighbors: List[str] = []
    for idx, keep in enumerate(mask):
        if keep != "1":
            continue
        session_id = ordered_ids[idx]
        label = id_to_session[session_id].label
        if counts.get(label, 0) <= min_per_class:
            continue
        next_mask = mask[:idx] + "0" + mask[idx + 1 :]
        neighbors.append(next_mask)
    return neighbors


def add_one_neighbors(mask: str) -> List[str]:
    neighbors: List[str] = []
    for idx, keep in enumerate(mask):
        if keep == "0":
            neighbors.append(mask[:idx] + "1" + mask[idx + 1 :])
    return neighbors


def propose_neighbor(
    base_mask: str,
    ordered_ids: Sequence[str],
    id_to_session: Dict[str, SessionRecord],
    min_per_class: int,
    rng: random.Random,
) -> str:
    removable = remove_one_neighbors(base_mask, ordered_ids, id_to_session, min_per_class)
    addable = add_one_neighbors(base_mask)
    move = rng.random()
    if removable and (move < 0.55 or not addable):
        return rng.choice(removable)
    if addable and (move < 0.85 or not removable):
        return rng.choice(addable)
    if removable and addable:
        removed = rng.choice(removable)
        new_addable = add_one_neighbors(removed)
        if new_addable:
            return rng.choice(new_addable)
    return base_mask


def prune_fold_artifacts(run_dir: Path) -> None:
    for filename in [
        "best_model.pt",
        "test_predictions.json",
        "test_session_predictions.json",
        "confusion_matrix.png",
        "confusion_matrix_session.png",
    ]:
        path = run_dir / filename
        if path.exists():
            path.unlink()


def summarize_metric(values: Iterable[float]) -> tuple[float, float]:
    values_list = [float(v) for v in values]
    if not values_list:
        return -1.0, 0.0
    return float(mean(values_list)), float(pstdev(values_list))


def run_worker(args: argparse.Namespace) -> int:
    started = time.time()
    try:
        config = load_config(args.config)
        if args.window_sec is not None:
            config["window_sec"] = float(args.window_sec)
        if args.epochs is not None:
            config["epochs"] = int(args.epochs)
        if args.early_stop_patience is not None:
            config["early_stop_patience"] = int(args.early_stop_patience)
        if args.n_splits is not None:
            config["grouped_cv"]["n_splits"] = int(args.n_splits)
        if args.n_repeats is not None:
            config["grouped_cv"]["n_repeats"] = int(args.n_repeats)
        set_seed(int(args.seed))

        sessions = discover_sessions(config["data_root"], config["labels"])
        sessions, task_info = resolve_task(config, sessions)
        config = {**config, "task": {**config.get("task", {}), **task_info}}
        ordered_sessions = sorted(sessions, key=lambda x: x.session_id)
        id_to_session = {session.session_id: session for session in ordered_sessions}

        subset_payload = read_json(args.subset_json)
        selected_ids = list(subset_payload["selected_ids"])
        selected_sessions = [id_to_session[session_id] for session_id in selected_ids]
        n_splits = int(config["grouped_cv"]["n_splits"])
        min_per_class = int(args.min_per_class or n_splits)
        valid, reason = is_subset_valid(selected_ids, id_to_session, min_per_class=min_per_class)
        if not valid:
            raise ValueError(reason)

        split_defs = build_grouped_cv_splits(
            sessions=selected_sessions,
            seed=int(args.seed),
            n_splits=n_splits,
            n_repeats=int(config["grouped_cv"]["n_repeats"]),
            val_fraction=float(config["grouped_cv"]["val_fraction_of_train"]),
        )

        trial_dir = ensure_dir(args.trial_dir)
        fold_root = ensure_dir(trial_dir / "folds")
        fold_results: List[Dict[str, object]] = []
        for split_def in split_defs:
            run_dir = ensure_dir(fold_root / split_def["name"])
            summary = train_and_evaluate_with_splits(
                config=config,
                modality=args.modality,
                run_dir=run_dir,
                splits={
                    "train": split_def["train"],
                    "val": split_def["val"],
                    "test": split_def["test"],
                },
                experiment_name=f"subset_search/{args.trial_id}/{split_def['name']}",
                append_summary=False,
            )
            fold_result = {
                "fold_name": split_def["name"],
                "session_accuracy": float(summary["test_metrics_session"]["accuracy"]),
                "session_macro_f1": float(summary["test_metrics_session"]["macro_f1"]),
                "window_accuracy": float(summary["test_metrics_window"]["accuracy"]),
                "window_macro_f1": float(summary["test_metrics_window"]["macro_f1"]),
                "best_epoch": int(summary["best_epoch"]),
            }
            fold_results.append(fold_result)
            if not args.keep_fold_artifacts:
                prune_fold_artifacts(run_dir)

        session_accuracy_mean, session_accuracy_std = summarize_metric(item["session_accuracy"] for item in fold_results)
        session_macro_f1_mean, session_macro_f1_std = summarize_metric(item["session_macro_f1"] for item in fold_results)
        window_accuracy_mean, window_accuracy_std = summarize_metric(item["window_accuracy"] for item in fold_results)
        window_macro_f1_mean, window_macro_f1_std = summarize_metric(item["window_macro_f1"] for item in fold_results)
        primary_metric_mean = {
            "session_accuracy": session_accuracy_mean,
            "session_macro_f1": session_macro_f1_mean,
            "window_accuracy": window_accuracy_mean,
            "window_macro_f1": window_macro_f1_mean,
        }[args.primary_metric]
        primary_metric_std = {
            "session_accuracy": session_accuracy_std,
            "session_macro_f1": session_macro_f1_std,
            "window_accuracy": window_accuracy_std,
            "window_macro_f1": window_macro_f1_std,
        }[args.primary_metric]
        score = compute_score(primary_metric_mean, len(selected_ids), float(args.alpha_per_folder))

        result = {
            "trial_id": args.trial_id,
            "phase": args.phase,
            "status": "success",
            "trial_dir": str(trial_dir),
            "subset_mask": subset_mask_from_ids(selected_ids, [session.session_id for session in ordered_sessions]),
            "selected_ids": selected_ids,
            "dropped_ids": [session.session_id for session in ordered_sessions if session.session_id not in set(selected_ids)],
            "label_counts": label_counts_for_ids(selected_ids, id_to_session),
            "duration_sec": time.time() - started,
            "primary_metric_name": args.primary_metric,
            "primary_metric_mean": primary_metric_mean,
            "primary_metric_std": primary_metric_std,
            "session_accuracy_mean": session_accuracy_mean,
            "session_accuracy_std": session_accuracy_std,
            "session_macro_f1_mean": session_macro_f1_mean,
            "session_macro_f1_std": session_macro_f1_std,
            "window_accuracy_mean": window_accuracy_mean,
            "window_macro_f1_mean": window_macro_f1_mean,
            "score": score,
            "fold_results": fold_results,
            "config_snapshot": {
                "modality": args.modality,
                "window_sec": config["window_sec"],
                "epochs": config["epochs"],
                "n_splits": config["grouped_cv"]["n_splits"],
                "n_repeats": config["grouped_cv"]["n_repeats"],
            },
        }
        write_json(args.result_json, result)
        return 0
    except Exception as exc:
        failure_payload = {
            "trial_id": args.trial_id,
            "phase": args.phase,
            "status": "failed",
            "trial_dir": str(args.trial_dir or ""),
            "duration_sec": time.time() - started,
            "error": f"{exc}\n\n{traceback.format_exc()}",
        }
        if args.result_json:
            write_json(args.result_json, failure_payload)
        return 1


def run_parent(args: argparse.Namespace) -> int:
    context = load_search_context(args.config, args.n_splits)
    config = context["config"]
    ordered_ids = context["ordered_ids"]
    id_to_session = context["id_to_session"]
    n_splits = int(context["n_splits"])
    if args.n_splits is None:
        args.n_splits = n_splits
    if args.n_repeats is None:
        args.n_repeats = int(context["n_repeats"])
    if args.min_per_class is None:
        args.min_per_class = int(args.n_splits)

    output_dir = ensure_dir(args.output_dir)
    csv_path = output_dir / "trials.csv"
    best_json_path = output_dir / "best_subset.json"
    state_path = output_dir / "search_state.json"
    manifest_path = output_dir / "search_manifest.json"
    write_json(
        manifest_path,
        {
            "config": args.config,
            "task_info": config["task"],
            "ordered_sessions": [id_to_session[session_id].to_dict() for session_id in ordered_ids],
        },
    )

    cache = load_existing_cache(csv_path, retry_failed=args.retry_failed)
    best_result = None
    for row in cache.values():
        if row["status"] == "success" and is_better_result(row, best_result):
            best_result = row
    if best_result is not None:
        write_best_json(best_json_path, best_result)

    base_state = build_parent_state(ordered_ids, id_to_session, args, output_dir)
    full_mask = "1" * len(ordered_ids)
    current_mask = str(best_result["subset_mask"]) if best_result is not None else full_mask
    temperature = float(args.initial_temperature)
    rng = random.Random(args.seed)
    new_trials = 0
    next_trial_index = count_existing_trials(csv_path) + 1

    phase = "backward_elimination"
    while has_budget(args.max_trials, new_trials):
        if phase == "backward_elimination":
            row, delta = maybe_evaluate_mask(
                mask=current_mask,
                phase=phase,
                args=args,
                ordered_ids=ordered_ids,
                id_to_session=id_to_session,
                min_per_class=int(args.min_per_class),
                cache=cache,
                csv_path=csv_path,
                output_dir=output_dir,
                next_trial_index=next_trial_index,
            )
            new_trials += delta
            next_trial_index += delta
            if row is not None and row["status"] == "success" and is_better_result(row, best_result):
                best_result = row
                write_best_json(best_json_path, best_result)

            improved_mask = current_mask
            improved_row = cache.get(current_mask)
            for neighbor in remove_one_neighbors(current_mask, ordered_ids, id_to_session, int(args.min_per_class)):
                if not has_budget(args.max_trials, new_trials):
                    break
                neighbor_row, delta = maybe_evaluate_mask(
                    mask=neighbor,
                    phase=phase,
                    args=args,
                    ordered_ids=ordered_ids,
                    id_to_session=id_to_session,
                    min_per_class=int(args.min_per_class),
                    cache=cache,
                    csv_path=csv_path,
                    output_dir=output_dir,
                    next_trial_index=next_trial_index,
                )
                new_trials += delta
                next_trial_index += delta
                if neighbor_row is not None and neighbor_row["status"] == "success":
                    if is_better_result(neighbor_row, best_result):
                        best_result = neighbor_row
                        write_best_json(best_json_path, best_result)
                    if is_better_result(neighbor_row, improved_row):
                        improved_mask = neighbor
                        improved_row = neighbor_row

            if improved_mask != current_mask:
                current_mask = improved_mask
            else:
                phase = "simulated_annealing"

        else:
            elite = ranked_success_rows(cache)
            candidate_mask = None
            for _ in range(64):
                if elite and new_trials % max(1, args.restart_every) == 0:
                    base_mask = rng.choice(elite[: min(5, len(elite))])["subset_mask"]
                elif best_result is not None:
                    base_mask = str(best_result["subset_mask"])
                else:
                    base_mask = full_mask
                proposed = propose_neighbor(base_mask, ordered_ids, id_to_session, int(args.min_per_class), rng)
                if proposed not in cache:
                    candidate_mask = proposed
                    break
            if candidate_mask is None:
                break
            candidate_row, delta = maybe_evaluate_mask(
                mask=candidate_mask,
                phase=phase,
                args=args,
                ordered_ids=ordered_ids,
                id_to_session=id_to_session,
                min_per_class=int(args.min_per_class),
                cache=cache,
                csv_path=csv_path,
                output_dir=output_dir,
                next_trial_index=next_trial_index,
            )
            new_trials += delta
            next_trial_index += delta
            if candidate_row is not None and candidate_row["status"] == "success":
                if is_better_result(candidate_row, best_result):
                    best_result = candidate_row
                    write_best_json(best_json_path, best_result)
                current_row = cache.get(current_mask)
                if current_row is None or current_row["status"] != "success":
                    current_mask = candidate_mask
                else:
                    delta_score = float(candidate_row["score"]) - float(current_row["score"])
                    accepted = delta_score >= 0.0
                    if not accepted and temperature > 1e-8:
                        accepted = rng.random() < math.exp(delta_score / temperature)
                    if accepted:
                        current_mask = candidate_mask
            temperature = max(1e-4, temperature * float(args.cooling_rate))

        write_state(
            state_path=state_path,
            base_state=base_state,
            cache=cache,
            best_result=best_result,
            phase=phase,
            current_mask=current_mask,
            temperature=temperature,
            new_trials=new_trials,
        )

    return 0


def main() -> int:
    args = parse_args()
    if args.hidden_worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
