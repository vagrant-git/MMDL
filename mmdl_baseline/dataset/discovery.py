from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SessionRecord:
    session_id: str
    session_dir: str
    audio_path: str
    daq_path: str
    metadata_path: str
    label: int
    label_text: str
    duration_sec: float
    audio_sample_rate: float
    sensor_sample_rate: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalize_text(text: str) -> str:
    return re.sub(r"[_\\-]+", " ", text.lower()).strip()


def parse_label(label_text: str, folder_name: str, config_labels: Dict[int, List[str]]) -> Optional[int]:
    search_space = _normalize_text(f"{label_text} {folder_name}")
    numeric_match = re.search(r"([0-4])\s*ml", search_space)
    if numeric_match:
        return int(numeric_match.group(1))
    if "no secretion" in search_space or re.search(r"(?:^|\\s)no(?:\\s|$)", search_space):
        return 0
    for numeric_label, aliases in config_labels.items():
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if alias_norm and alias_norm in search_space:
                return int(numeric_label)
    return None


def discover_sessions(data_root: str | Path, config_labels: Dict[int, List[str]]) -> List[SessionRecord]:
    data_root = Path(data_root)
    sessions: List[SessionRecord] = []
    for directory in sorted(data_root.glob("MMdata_*")):
        if not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        audio_path = directory / "audio.wav"
        daq_path = directory / "daq.csv"
        if not (metadata_path.exists() and audio_path.exists() and daq_path.exists()):
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        label_text = str(metadata.get("label", ""))
        label = parse_label(label_text, directory.name, config_labels)
        if label is None:
            raise ValueError(f"Unable to parse label for session: {directory}")
        sessions.append(
            SessionRecord(
                session_id=directory.name,
                session_dir=str(directory),
                audio_path=str(audio_path),
                daq_path=str(daq_path),
                metadata_path=str(metadata_path),
                label=label,
                label_text=label_text,
                duration_sec=float(metadata.get("duration_sec_captured") or 0.0),
                audio_sample_rate=float(metadata.get("audio", {}).get("sample_rate_hz") or 0.0),
                sensor_sample_rate=float(metadata.get("daq", {}).get("sample_rate_hz") or 0.0),
            )
        )
    if not sessions:
        raise FileNotFoundError(f"No complete sessions found in {data_root}")
    return sessions
