"""Plot Mini-LVSM training curves (train/val MSE and PSNR, val SSIM) from one or more log.jsonl files.
  python tools/plot_lvsm_curves.py <out.png> <run_dir>[:label] ...
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out, runs = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for spec in runs:
        path, _, label = spec.partition(":")
        rows = [json.loads(line) for line in (pathlib.Path(path) / "log.jsonl").read_text().splitlines() if line.strip()]
        label = label or pathlib.Path(path).name
        ep = [r["epoch"] for r in rows]
        (line,) = axes[0].plot(ep, [r["train"]["loss"] for r in rows], label=f"{label} train", alpha=0.8)
        axes[0].plot(ep, [r["val"]["loss"] for r in rows], "--", color=line.get_color(), label=f"{label} val")
        axes[1].plot(ep, [r["train"]["psnr"] for r in rows], color=line.get_color(), alpha=0.8, label=f"{label} train")
        axes[1].plot(ep, [r["val"]["psnr"] for r in rows], "--", color=line.get_color(), label=f"{label} val")
        axes[2].plot(ep, [r["val"]["ssim"] for r in rows], "--", color=line.get_color(), label=f"{label} val")
    for ax, title in zip(axes, ["MSE loss", "PSNR (dB)", "SSIM (val)"]):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
