from __future__ import annotations

from pathlib import Path
from typing import Iterable


def append_markdown_section(path: str | Path, title: str, lines: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"## {title}\n\n")
        for line in lines:
            f.write(f"{line}\n")
        f.write("\n")
