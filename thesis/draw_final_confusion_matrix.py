from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm


ROOT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "fonts"
OUTPUT_PATH = ROOT / "图4-最终模型混淆矩阵.png"


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
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    en_fp = fm.FontProperties(family=en_font_name, size=10)
    cn_fp = fm.FontProperties(family=cn_font_name, size=10)

    cm = np.array(
        [
            [718, 76, 0],
            [100, 945, 0],
            [3, 39, 933],
        ],
        dtype=int,
    )
    classes = ["0 ml", "2 ml", "4 ml"]

    fig, ax = plt.subplots(figsize=(3.8, 3.25))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

    ax.set_xlabel("预测类别", fontproperties=cn_fp, labelpad=6)
    ax.set_ylabel("真实类别", fontproperties=cn_fp, labelpad=6)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontproperties=en_fp)
    ax.set_yticklabels(classes, fontproperties=en_fp)

    threshold = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontproperties=en_fp,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(axis="both", which="major", pad=4, labelsize=10)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.97)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
