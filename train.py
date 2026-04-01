from __future__ import annotations

import argparse
from pathlib import Path

from mmdl_baseline.train_eval import train_and_evaluate
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import ensure_dir, write_json
from mmdl_baseline.utils.runtime import log_runtime_environment
from mmdl_baseline.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal 5-class baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--modality",
        required=True,
        choices=["audio_only", "pressure_flow", "multimodal"],
    )
    args = parser.parse_args()

    log_runtime_environment()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    output_root = ensure_dir(config["output_root"])
    run_dir = output_root / args.modality
    ensure_dir(run_dir)
    write_json(Path(run_dir) / "resolved_config.json", config)
    train_and_evaluate(config, args.modality, run_dir)


if __name__ == "__main__":
    main()
