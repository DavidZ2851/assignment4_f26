"""Bonus plots: Q2.3 (LVSM training view count) and Q3.3 (VGGT number of views)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/screenshots"

# Q2.3: models trained with S=1/2/4, each evaluated at S=1/2/4 (correct rays).
runs = {"train S=1": "outputs/lvsm_s1", "train S=2": "outputs/lvsm", "train S=4": "outputs/lvsm_s4"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for label, d in runs.items():
    m = json.load(open(f"{d}/eval_val/metrics.json"))
    xs = [1, 2, 4]
    for ax, key in zip(axes, ["psnr", "ssim"]):
        ax.plot(xs, [m[f"S={s},conditioning=correct"][key] for s in xs], "o-", label=label)
for ax, name in zip(axes, ["PSNR (dB)", "SSIM"]):
    ax.set_xlabel("source views at evaluation")
    ax.set_xticks([1, 2, 4])
    ax.set_title(f"Val {name}, 100 epochs each")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/lvsm_bonus_views.png", dpi=130)

# Q3.3: VGGT accuracy / runtime / memory vs number of views.
views = [2, 4, 8, 16]
mets = {}
for v in views:
    path = "outputs/vggt_train/metrics.json" if v == 8 else f"outputs/vggt_train/bonus_views/val_v{v}.json"
    mets[v] = json.load(open(path))
bench = json.load(open("outputs/vggt_train/bonus_views/bench.json"))
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].plot(views, [mets[v]["depth_abs_rel"] for v in views], "o-", label="AbsRel (lower better)")
axes[0].plot(views, [mets[v]["depth_delta1"] for v in views], "s-", label="δ1 (higher better)")
axes[0].set_title("Depth vs views")
for k, lab in [("rotation_auc_30", "rot AUC@30"), ("translation_auc_30", "trans AUC@30"), ("pose_auc_30", "pose AUC@30"),
               ("rotation_auc_15", "rot AUC@15")]:
    axes[1].plot(views, [mets[v][k] for v in views], "o-", label=lab)
axes[1].set_title("Pose AUC vs views")
bv = [b["views"] for b in bench]
axes[2].plot(bv, [b["ms_per_forward"] for b in bench], "o-", color="C0", label="ms / forward")
axes[2].set_ylabel("ms per forward (batch 1)")
ax2 = axes[2].twinx()
ax2.plot(bv, [b["peak_mem_gb"] for b in bench], "s--", color="C3", label="peak memory (GB)")
ax2.set_ylabel("peak memory (GB)")
axes[2].set_title("Runtime / memory vs views (RTX 3090, bf16)")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("input views")
    ax.grid(alpha=0.3)
axes[0].legend()
axes[1].legend()
axes[2].legend(loc="upper left")
ax2.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/vggt_bonus_views.png", dpi=130)
print("ok")
