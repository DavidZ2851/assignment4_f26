"""Q3.3 bonus: Mini-VGGT forward latency and peak GPU memory vs. number of input views (batch 1, 224px).
  python tools/vggt_views_bench.py <checkpoint> <out.json> [views ...]
"""

import json
import sys
import time

import torch

from models.vggt.runtime import build_components


def main():
    ckpt_path, out = sys.argv[1], sys.argv[2]
    views = [int(v) for v in sys.argv[3:]] or [2, 4, 8, 16, 24, 32]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model, _, _ = build_components(ckpt["config"], from_checkpoint=True)
    model.load_state_dict(ckpt["model"])
    model = model.cuda().eval()
    size = ckpt["config"]["model"]["image_size"]
    rows = []
    for v in views:
        rgb = torch.rand(1, v, 3, size, size, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):  # warm-up
                model({"rgb": rgb})
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                model({"rgb": rgb})
            torch.cuda.synchronize()
        ms = (time.time() - start) / 10 * 1000
        rows.append({"views": v, "ms_per_forward": ms, "ms_per_view": ms / v,
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
        print(rows[-1], flush=True)
    json.dump(rows, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
