from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT.parent / "summary-MMmodel" / "hcaf_confgate_interpretability" / "summary.json"
OUTPUT_PATH = ROOT / "figures" / "图4-7-交叉注意力可视化.png"


def register_font(path: Path) -> str:
    fm.fontManager.addfont(str(path))
    return fm.FontProperties(fname=str(path)).get_name()


def main() -> None:
    cn_font_path = Path("/mnt/c/Windows/Fonts/simsun.ttc")
    en_font_path = Path("/mnt/c/Windows/Fonts/times.ttf")
    if not cn_font_path.exists():
        raise FileNotFoundError(f"Missing Chinese font: {cn_font_path}")
    if not en_font_path.exists():
        raise FileNotFoundError(f"Missing English font: {en_font_path}")

    cn_font_name = register_font(cn_font_path)
    en_font_name = register_font(en_font_path)

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = cn_font_name
    plt.rcParams["font.size"] = 9.5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    cn_fp = fm.FontProperties(family=cn_font_name, size=9.5)
    en_fp = fm.FontProperties(family=en_font_name, size=9.5)
    cn_fp_small = fm.FontProperties(family=cn_font_name, size=8.5)
    en_fp_small = fm.FontProperties(family=en_font_name, size=8.5)

    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    best_examples = payload["best_examples"]
    class_map = {"0": "0 ml", "1": "2 ml", "2": "4 ml"}
    keys = ["0", "1", "2"]

    fig, axes = plt.subplots(3, 2, figsize=(8.2, 8.4))

    for row_idx, key in enumerate(keys):
        example = best_examples[key]
        class_name = class_map[key]
        a2s = np.asarray(example["audio_to_sensor_attn"], dtype=np.float32).mean(axis=0)
        s2a = np.asarray(example["sensor_to_audio_attn"], dtype=np.float32).mean(axis=0)

        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        im_left = ax_left.imshow(a2s, aspect="auto", cmap="magma")
        im_right = ax_right.imshow(s2a, aspect="auto", cmap="viridis")

        ax_left.set_title(
            f"{class_name}：音频到传感器注意力\n起始时间 {example['start_sec']:.1f} s，置信度 {example['final_confidence']:.3f}",
            fontproperties=cn_fp_small,
            pad=4,
        )
        ax_right.set_title(f"{class_name}：传感器到音频注意力", fontproperties=cn_fp, pad=4)

        ax_left.set_xlabel("传感器特征序号", fontproperties=cn_fp_small, labelpad=2)
        ax_left.set_ylabel("音频特征序号", fontproperties=cn_fp_small, labelpad=2)
        ax_right.set_xlabel("音频特征序号", fontproperties=cn_fp_small, labelpad=2)
        ax_right.set_ylabel("传感器特征序号", fontproperties=cn_fp_small, labelpad=2)

        cbar_left = fig.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.03)
        cbar_right = fig.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.03)
        for label in cbar_left.ax.get_yticklabels():
            label.set_fontproperties(en_fp_small)
        for label in cbar_right.ax.get_yticklabels():
            label.set_fontproperties(en_fp_small)

        for label in ax_left.get_xticklabels() + ax_left.get_yticklabels():
            label.set_fontproperties(en_fp_small)
        for label in ax_right.get_xticklabels() + ax_right.get_yticklabels():
            label.set_fontproperties(en_fp_small)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.96, hspace=0.38, wspace=0.18)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
