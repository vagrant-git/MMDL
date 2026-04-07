from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch

from edge_deploy_utils import build_dummy_batch, build_multimodal_model, load_deploy_config

DEFAULT_OUTPUT = Path("deploy/artifacts/hcaf_pcen_dualxattn.onnx")


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        audio: torch.Tensor,
        pressure: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        return self.model({"audio": audio, "pressure": pressure, "flow": flow})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the default multimodal model to ONNX.")
    parser.add_argument(
        "--config",
        default="configs/final_model_unified_evidence.yaml",
        help="Path to the canonical experiment config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "summary-MMmodel/final_model_unified_evidence/runs/"
            "hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold1/best_model.pt"
        ),
        help="Path to a trained multimodal checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output ONNX file path.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for export.")
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="Export a fixed-batch graph instead of keeping the batch axis dynamic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sys.version_info >= (3, 12):
        raise RuntimeError(
            "export_onnx.py requires Python 3.11 or lower when using torch.onnx.dynamo_export. "
            "Please run it in a Python 3.9/3.10/3.11 environment."
        )
    config = load_deploy_config(args.config)
    model = build_multimodal_model(config, args.checkpoint, device="cpu")
    wrapper = OnnxExportWrapper(model).eval()
    batch = build_dummy_batch(config, batch_size=args.batch_size, device="cpu")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_shapes = None if args.no_dynamic_batch else {
        "audio": {0: "batch"},
        "pressure": {0: "batch"},
        "flow": {0: "batch"},
    }
    with torch.inference_mode():
        onnx_program = torch.onnx.export(
            wrapper,
            (batch["audio"], batch["pressure"], batch["flow"]),
            f=None,
            input_names=["audio", "pressure", "flow"],
            output_names=["logits"],
            opset_version=args.opset,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
        )
        if onnx_program is None:
            raise RuntimeError("torch.onnx.export returned no ONNXProgram.")
        onnx_program.save(str(output_path))

    onnx_model = onnx.load(str(output_path))
    input_names = ["audio", "pressure", "flow"]
    for value_info, name in zip(onnx_model.graph.input, input_names):
        value_info.name = name
    if len(onnx_model.graph.output) == 1:
        onnx_model.graph.output[0].name = "logits"
    onnx.save(onnx_model, str(output_path))

    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "onnx_path": str(output_path.resolve()),
        "audio_shape": list(batch["audio"].shape),
        "pressure_shape": list(batch["pressure"].shape),
        "flow_shape": list(batch["flow"].shape),
        "opset": args.opset,
        "dynamic_batch": not args.no_dynamic_batch,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Exported ONNX to {output_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
