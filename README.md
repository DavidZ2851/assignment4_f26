# 16-825 Assignment 4: Multi-View Transformers (hzhan2)

My submission for Assignment 4: multi-view data preprocessing, novel-view synthesis with Mini-LVSM, and camera + depth prediction with Mini-VGGT, all on the ScanNet++ iPhone data at 224 px.

The write-up with all results, plots and discussion is in **`index.html`** (open it in a browser; the figures are in `images/`).

## What's in here

```
main.py            one entry point that runs every step below
index.html         report webpage
images/            figures used by the report
a4data/            data loading + my Q1 geometry code (geometry.py)
models/lvsm/       Mini-LVSM (Q2): model.py, losses.py, train.py, eval.py
models/vggt/       Mini-VGGT (Q3): model.py, heads.py, losses.py, train.py, eval.py
configs/           training configs (lvsm_long.json = 400-epoch run, lvsm_s1/s4 = Q2.3 bonus)
scripts/           data visualization and checks for Q1
tools/             my helper scripts: plotting, viewer screenshots, VGGT view-count benchmark
docs/              the original assignment handout for each part
```

The code I wrote is in:

- `a4data/geometry.py`: resizing intrinsics, depth unprojection, canonicalization, camera rays and Plücker coordinates (Q1.1–Q1.5).
- `models/lvsm/`: patchify/unpatchify, token building, forward pass, MSE loss and the autocast training step (Q2).
- `models/vggt/`: frame/global alternating attention, camera and depth heads, and the camera/depth/gradient losses (Q3).

## Setup

Install PyTorch for your machine, then from the repository root:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Download the dataset (31.9 GB zipped, 40.7 GB extracted) into `data/scannetpp`:

```bash
pip install huggingface_hub
hf download learning3dvision/ScanNetpp_iphone ScanNetpp_A4.zip --repo-type dataset --local-dir data/scannetpp
unzip data/scannetpp/ScanNetpp_A4.zip -d data/scannetpp
```

## Running

Everything goes through `main.py`. To see every step and the exact command it runs:

```bash
python main.py --list
```

Then run one or more steps by name, for example:

```bash
python main.py q1-check
python main.py q2-train q2-eval
```

Add `--dry-run` to just print the commands.

| Step | What it does | Output |
|---|---|---|
| `q1-unprojection` | viewer of depth unprojected into world space | `outputs/unprojection/index.html` |
| `q1-canonicalization` | viewer of canonicalized cameras and points | `outputs/canonicalization/index.html` |
| `q1-check` | reprojection error and centroid checks | printed to the terminal |
| `q2-overfit` | Q2.1 overfit Mini-LVSM | `outputs/lvsm_overfit/` |
| `q2-train` | Q2.2 train Mini-LVSM for 400 epochs (the model in the report) | `outputs/lvsm_long/` |
| `q2-train-default` | Q2.2 default 100-epoch run, for comparison | `outputs/lvsm/` |
| `q2-eval` | Q2.2 eval at S=1/2/4 with correct, shuffled and no rays | `outputs/lvsm_long/eval_val/` |
| `q2-bonus` | Q2.3 train with 1 and 4 source views, eval each at 1/2/4 | `outputs/lvsm_s1/`, `outputs/lvsm_s4/` |
| `q3-overfit` | Q3.1 overfit Mini-VGGT | `outputs/vggt_overfit/` |
| `q3-train` | Q3.2 train Mini-VGGT for 1000 epochs | `outputs/vggt_train/` |
| `q3-eval` | Q3.2 val (with 3D viewer) and test metrics, 8 views | `outputs/vggt_train/metrics.json`, `test_metrics.json` |
| `q3-bonus` | Q3.3 val metrics at 2/4/16 views + speed/memory benchmark | `outputs/vggt_train/bonus_views/` |

The eval steps need the checkpoint from the matching train step. I trained on single RTX 3090s: Mini-LVSM took about 2.7 h for 400 epochs, and Mini-VGGT about 4.4 h for 1000 epochs.
