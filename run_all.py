from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all baseline experiments.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    print(f"[runtime] python={sys.executable}", flush=True)
    print("[runtime] Tip: use `conda activate dl` before launching this script to enable GPU runs.", flush=True)

    modalities = ["audio_only", "pressure_flow", "multimodal"]
    for modality in modalities:
        subprocess.run(
            [sys.executable, "train.py", "--config", args.config, "--modality", modality],
            check=True,
        )
    subprocess.run([sys.executable, "generate_report.py", "--config", args.config], check=True)


if __name__ == "__main__":
    main()
