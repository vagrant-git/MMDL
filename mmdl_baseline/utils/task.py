from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from mmdl_baseline.dataset.discovery import SessionRecord


def apply_session_filter(config: Dict[str, object], sessions: List[SessionRecord]) -> Tuple[List[SessionRecord], Dict[str, object]]:
    session_filter = config.get("session_filter") or {}
    include_ids = [str(x) for x in session_filter.get("include_session_ids", [])]
    exclude_ids = [str(x) for x in session_filter.get("exclude_session_ids", [])]
    include_set = set(include_ids)
    exclude_set = set(exclude_ids)

    overlap = sorted(include_set & exclude_set)
    if overlap:
        raise ValueError(f"session_filter has overlapping include/exclude ids: {overlap}")

    available_ids = {session.session_id for session in sessions}
    missing_includes = sorted(include_set - available_ids)
    if missing_includes:
        raise ValueError(f"session_filter include_session_ids not found in dataset: {missing_includes}")

    filtered = sessions
    if include_set:
        filtered = [session for session in filtered if session.session_id in include_set]
    if exclude_set:
        filtered = [session for session in filtered if session.session_id not in exclude_set]
    if not filtered:
        raise ValueError("session_filter removed every available session.")

    filter_info = {
        "include_session_ids": sorted(include_set),
        "exclude_session_ids": sorted(exclude_set),
        "num_sessions_before_filter": len(sessions),
        "num_sessions_after_filter": len(filtered),
    }
    return filtered, filter_info


def resolve_task(config: Dict[str, object], sessions: List[SessionRecord]) -> Tuple[List[SessionRecord], Dict[str, object]]:
    sessions, filter_info = apply_session_filter(config, sessions)
    task_cfg = config.get("task", {})
    class_subset = task_cfg.get("class_subset")
    if class_subset:
        class_subset = [int(x) for x in class_subset]
        filtered = [session for session in sessions if session.label in class_subset]
        label_to_index = {label: idx for idx, label in enumerate(class_subset)}
        remapped = [
            replace(session, label=label_to_index[session.label], label_text=f"{session.label_text}|mapped:{label_to_index[session.label]}")
            for session in filtered
        ]
        class_names = task_cfg.get("class_names") or [str(label) for label in class_subset]
        original_labels = class_subset
    else:
        unique_labels = sorted({session.label for session in sessions})
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        remapped = [
            replace(session, label=label_to_index[session.label], label_text=f"{session.label_text}|mapped:{label_to_index[session.label]}")
            for session in sessions
        ]
        class_names = [str(label) for label in unique_labels]
        original_labels = unique_labels

    task_info = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "original_labels": original_labels,
        "label_to_index": label_to_index,
        "session_filter": filter_info,
    }
    return remapped, task_info
