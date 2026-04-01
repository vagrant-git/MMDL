from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from .discovery import SessionRecord


def build_session_split(
    sessions: List[SessionRecord],
    seed: int,
    test_per_class: int,
    val_fraction_of_remaining: float,
) -> Dict[str, List[SessionRecord]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for session in sessions:
        by_label[session.label].append(session)
    train: List[SessionRecord] = []
    val: List[SessionRecord] = []
    test: List[SessionRecord] = []
    for label in sorted(by_label):
        label_sessions = sorted(by_label[label], key=lambda x: x.session_id)
        rng.shuffle(label_sessions)
        label_test = label_sessions[: min(test_per_class, len(label_sessions))]
        remaining = label_sessions[len(label_test) :]
        val_count = int(round(len(remaining) * val_fraction_of_remaining))
        if len(remaining) >= 3:
            val_count = max(1, val_count)
        else:
            val_count = 0
        label_val = remaining[:val_count]
        label_train = remaining[val_count:]
        test.extend(label_test)
        val.extend(label_val)
        train.extend(label_train)
    return {
        "train": sorted(train, key=lambda x: x.session_id),
        "val": sorted(val, key=lambda x: x.session_id),
        "test": sorted(test, key=lambda x: x.session_id),
    }


def build_train_val_split(
    sessions: List[SessionRecord],
    seed: int,
    val_fraction: float,
) -> Dict[str, List[SessionRecord]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for session in sessions:
        by_label[session.label].append(session)
    train: List[SessionRecord] = []
    val: List[SessionRecord] = []
    for label in sorted(by_label):
        label_sessions = sorted(by_label[label], key=lambda x: x.session_id)
        rng.shuffle(label_sessions)
        if len(label_sessions) >= 4:
            val_count = max(1, int(round(len(label_sessions) * val_fraction)))
        else:
            val_count = 0
        label_val = label_sessions[:val_count]
        label_train = label_sessions[val_count:]
        val.extend(label_val)
        train.extend(label_train)
    return {
        "train": sorted(train, key=lambda x: x.session_id),
        "val": sorted(val, key=lambda x: x.session_id),
    }
