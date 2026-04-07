from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "figures"

CN_FONT_PATH = Path("/mnt/c/Windows/Fonts/simsun.ttc")
EN_FONT_PATH = Path("/mnt/c/Windows/Fonts/times.ttf")


def register_font(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing font: {path}")
    fm.fontManager.addfont(str(path))
    return fm.FontProperties(fname=str(path)).get_name()


def contains_cjk(text: object) -> bool:
    value = str(text)
    return any("\u4e00" <= ch <= "\u9fff" for ch in value)


def setup_rcparams() -> tuple[fm.FontProperties, fm.FontProperties]:
    cn_font_name = register_font(CN_FONT_PATH)
    en_font_name = register_font(EN_FONT_PATH)

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = cn_font_name
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    cn_fp = fm.FontProperties(family=cn_font_name, size=10.5)
    en_fp = fm.FontProperties(family=en_font_name, size=10.5)
    return cn_fp, en_fp


def apply_cell_font(cell, cn_fp: fm.FontProperties, en_fp: fm.FontProperties, text: object, *, size: float | None = None) -> None:
    fp = cn_fp if contains_cjk(text) else en_fp
    if size is not None:
        fp = fp.copy()
        fp.set_size(size)
    cell.get_text().set_fontproperties(fp)


def save_table_figure(
    filename_stem: str,
    title: str,
    columns: list[str],
    rows: list[list[object]],
    *,
    col_widths: list[float],
    figure_size: tuple[float, float],
    cn_fp: fm.FontProperties,
    en_fp: fm.FontProperties,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figure_size)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="top",
        fontproperties=cn_fp,
        fontsize=12,
        transform=ax.transAxes,
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.02, 0.05, 0.96, 0.84],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    n_rows = len(rows)
    n_cols = len(columns)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#303030")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#e9eef6")
            apply_cell_font(cell, cn_fp, en_fp, columns[col_idx], size=10.5)
            cell.set_height(cell.get_height() * 1.10)
        else:
            value = rows[row_idx - 1][col_idx]
            cell.set_facecolor("white" if row_idx % 2 else "#f8f8f8")
            apply_cell_font(cell, cn_fp, en_fp, value, size=10.2)
        if row_idx in (0, n_rows) or col_idx in (0, n_cols - 1):
            cell.set_linewidth(1.0)

    for suffix in (".png", ".svg"):
        fig.savefig(
            OUTPUT_DIR / f"{filename_stem}{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(fig)


def save_confusion_matrix(
    filename_stem: str,
    title: str,
    cm: np.ndarray,
    labels: list[str],
    cn_fp: fm.FontProperties,
    en_fp: fm.FontProperties,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=int(cm.max()))

    ax.set_title(title, fontproperties=cn_fp, pad=8)
    ax.set_xlabel("预测类别", fontproperties=cn_fp, labelpad=6)
    ax.set_ylabel("真实类别", fontproperties=cn_fp, labelpad=6)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontproperties=en_fp)
    ax.set_yticklabels(labels, fontproperties=en_fp)

    threshold = cm.max() * 0.58
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontproperties=en_fp,
                fontsize=10.5,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(axis="both", which="major", pad=4, labelsize=10)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.88)

    for suffix in (".png", ".svg"):
        fig.savefig(
            OUTPUT_DIR / f"{filename_stem}{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    cn_fp, en_fp = setup_rcparams()

    save_table_figure(
        "表4-1-数据集与样本分布",
        "表 4.1 数据集与样本分布",
        ["类别", "session 数", "窗口数"],
        [
            ["0 ml", "7", "847"],
            ["2 ml", "6", "1045"],
            ["4 ml", "6", "975"],
            ["合计", "19", "2867"],
        ],
        col_widths=[0.28, 0.22, 0.22],
        figure_size=(5.0, 2.5),
        cn_fp=cn_fp,
        en_fp=en_fp,
    )

    save_table_figure(
        "表4-2-最终模型整体分类结果",
        "表 4.2 最终模型整体分类结果",
        ["模型", "Macro-F1", "session 级 Macro-F1"],
        [
            ["HCAF-PCEN-DualXAttn", "0.8225 ± 0.1681", "0.8148 ± 0.2619"],
        ],
        col_widths=[0.42, 0.25, 0.29],
        figure_size=(6.0, 2.1),
        cn_fp=cn_fp,
        en_fp=en_fp,
    )

    save_confusion_matrix(
        "图4-6-最终模型三分类混淆矩阵",
        "最终模型三分类混淆矩阵",
        np.asarray(
            [
                [718, 76, 0],
                [100, 945, 0],
                [3, 39, 933],
            ],
            dtype=int,
        ),
        ["0 ml", "2 ml", "4 ml"],
        cn_fp=cn_fp,
        en_fp=en_fp,
    )

    save_table_figure(
        "表4-3-不同模态组合与融合策略的性能比较",
        "表 4.3 不同模态组合与融合策略的性能比较",
        ["模型", "模态组合 / 条件", "Window Acc", "Window Macro-F1", "session 级 Macro-F1"],
        [
            ["Audio-only", "Audio", "0.7170 ± 0.0578", "0.7052 ± 0.0667", "0.8296 ± 0.1362"],
            ["Pressure+Flow-only", "Pressure + Flow", "0.7692 ± 0.2274", "0.7499 ± 0.2513", "0.8519 ± 0.2095"],
            ["HCAF-PCEN-DualXAttn", "Audio + Pressure + Flow", "0.8418 ± 0.1427", "0.8225 ± 0.1681", "0.8148 ± 0.2619"],
            ["去除音频", "Pressure + Flow", "0.7653 ± 0.1514", "0.7578 ± 0.1570", "0.7333 ± 0.1257"],
            ["去除压力", "Audio + Flow", "0.9009 ± 0.0898", "0.8932 ± 0.0968", "0.9407 ± 0.0838"],
            ["去除流量", "Audio + Pressure", "0.9167 ± 0.0360", "0.9116 ± 0.0359", "0.9407 ± 0.0838"],
        ],
        col_widths=[0.24, 0.26, 0.16, 0.17, 0.17],
        figure_size=(10.8, 3.4),
        cn_fp=cn_fp,
        en_fp=en_fp,
    )

    save_table_figure(
        "表4-4-当前最终模型的可复核组件对照",
        "表 4.4 当前最终模型的可复核组件对照",
        ["模型", "Window Acc", "Window Macro-F1", "session 级 Macro-F1"],
        [
            ["HCAF-PCEN-DualXAttn", "0.8418 ± 0.1427", "0.8225 ± 0.1681", "0.8148 ± 0.2619"],
            ["Simple Gate", "0.8321 ± 0.0785", "0.8307 ± 0.0650", "0.8296 ± 0.1362"],
            ["Log-Mel 96", "0.8988 ± 0.0704", "0.8816 ± 0.0664", "0.9407 ± 0.0838"],
        ],
        col_widths=[0.34, 0.20, 0.22, 0.22],
        figure_size=(7.2, 2.7),
        cn_fp=cn_fp,
        en_fp=en_fp,
    )


if __name__ == "__main__":
    main()
