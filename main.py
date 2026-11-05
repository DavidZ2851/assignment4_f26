"""Single entry point for Assignment 4. Each step runs the same command listed in docs/part*.md.

  python main.py --list                 # show every step and its command
  python main.py q1-check               # run one step
  python main.py q2-train q2-eval       # run several steps in order
  python main.py q3-eval --dry-run      # print the commands without running them

Run from the repository root, with the dataset in data/scannetpp (see README.md).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable, "-u"]
VIS = ["--num-scenes", "5", "--max-points", "10000"]
VGGT_EVAL = ["--eval-root", "data/scannetpp/prepared_eval", "--eval-num-frames", "32",
             "--image-size", "224", "--seed", "825"]

# step name -> (description, list of commands run in order)
STEPS = {
    # ---- Part 1: data preprocessing ----
    "q1-unprojection": ("Q1 viewer: depth unprojected into world space", [
        PY + ["scripts/visualize_data.py", "--stage", "unprojection", *VIS, "--output", "outputs/unprojection"],
    ]),
    "q1-canonicalization": ("Q1 viewer: canonicalized cameras and points", [
        PY + ["scripts/visualize_data.py", "--stage", "canonicalization", *VIS, "--output", "outputs/canonicalization"],
    ]),
    "q1-check": ("Q1 numeric checks (reprojection error, centroid distance)", [
        PY + ["scripts/check_data.py", "--config", "configs/scannetpp/iphone/data.json",
              "--split", "train", "--max-samples", "4"],
    ]),
    # ---- Part 2: Mini-LVSM ----
    "q2-overfit": ("Q2.1 overfit Mini-LVSM on one batch", [
        PY + ["-m", "models.lvsm.train", "--overfit", "2000", "--output-dir", "outputs/lvsm_overfit"],
    ]),
    "q2-train": ("Q2.2 train Mini-LVSM, 400 epochs (the model in the report)", [
        PY + ["-m", "models.lvsm.train", "--config", "configs/lvsm_long.json"],
    ]),
    "q2-train-default": ("Q2.2 train Mini-LVSM, default 100 epochs (reference run)", [
        PY + ["-m", "models.lvsm.train", "--config", "configs/lvsm.json"],
    ]),
    "q2-eval": ("Q2.2 evaluate S=1/2/4 with correct/shuffled/no rays", [
        PY + ["-m", "models.lvsm.eval", "--checkpoint", "outputs/lvsm_long/best.pt",
              "--num-source-views", "1", "2", "4", "--conditioning", "correct", "shuffled", "none",
              "--output", "outputs/lvsm_long/eval_val"],
    ]),
    "q2-bonus": ("Q2.3 train with S=1 and S=4 source views, evaluate each at S=1/2/4", [
        cmd
        for s in ("1", "4")
        for cmd in (
            PY + ["-m", "models.lvsm.train", "--config", f"configs/lvsm_s{s}.json"],
            PY + ["-m", "models.lvsm.eval", "--checkpoint", f"outputs/lvsm_s{s}/best.pt",
                  "--num-source-views", "1", "2", "4", "--conditioning", "correct",
                  "--output", f"outputs/lvsm_s{s}/eval_val"],
        )
    ]),
    # ---- Part 3: Mini-VGGT ----
    "q3-overfit": ("Q3.1 overfit Mini-VGGT on fixed clips", [
        PY + ["-m", "models.vggt.train", "--config", "configs/vggt.json", "--overfit",
              "--output", "outputs/vggt_overfit"],
    ]),
    "q3-train": ("Q3.2 train Mini-VGGT (1000 epochs)", [
        PY + ["-m", "models.vggt.train", "--config", "configs/vggt.json", "--output", "outputs/vggt_train"],
    ]),
    "q3-eval": ("Q3.2 evaluate on val (with viewer) and test, 8 views", [
        PY + ["-m", "models.vggt.eval", "--checkpoint", "outputs/vggt_train/best.pt", "--split", "val",
              *VGGT_EVAL, "--num-views", "8", "--report", "outputs/vggt_train/metrics.json",
              "--visualization", "outputs/vggt_train/visualization/index.html"],
        PY + ["-m", "models.vggt.eval", "--checkpoint", "outputs/vggt_train/best.pt", "--split", "test",
              *VGGT_EVAL, "--num-views", "8", "--report", "outputs/vggt_train/test_metrics.json"],
    ]),
    "q3-bonus": ("Q3.3 val metrics at 2/4/16 views, then latency/memory benchmark", [
        *(
            PY + ["-m", "models.vggt.eval", "--checkpoint", "outputs/vggt_train/best.pt", "--split", "val",
                  *VGGT_EVAL, "--num-views", v, "--report", f"outputs/vggt_train/bonus_views/val_v{v}.json"]
            for v in ("2", "4", "16")
        ),
        PY + ["tools/vggt_views_bench.py", "outputs/vggt_train/best.pt",
              "outputs/vggt_train/bonus_views/bench.json", "2", "4", "8", "16", "24", "32"],
    ]),
}


def show(cmd):
    return " ".join(["python" if c == sys.executable else c for c in cmd])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", metavar="step", help="steps to run, in order (see --list)")
    parser.add_argument("--list", action="store_true", help="list all steps and their commands")
    parser.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    args = parser.parse_args()
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}; choose from {', '.join(STEPS)}")

    if args.list or not args.steps:
        for name, (desc, cmds) in STEPS.items():
            print(f"{name:20s} {desc}")
            for cmd in cmds:
                print(f"{'':20s}   $ {show(cmd)}")
        return

    # Run from the repo root and make a4data/models importable even without `pip install -e .`.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    for name in args.steps:
        desc, cmds = STEPS[name]
        print(f"==> {name}: {desc}", flush=True)
        for cmd in cmds:
            print(f"$ {show(cmd)}", flush=True)
            if not args.dry_run:
                code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
                if code != 0:
                    sys.exit(f"step {name} failed (exit code {code})")


if __name__ == "__main__":
    main()
