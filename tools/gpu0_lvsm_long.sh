#!/bin/bash
# After gpu0_lvsm_chain.sh: longer LVSM baseline (400 epochs; the 100-epoch default underfits), then the Q2.2 ablation eval.
cd /home/haotian/assignment4_f26
P=/home/haotian/miniconda3/envs/a4/bin/python
export CUDA_VISIBLE_DEVICES=0
log() { echo "[$(date +%m/%d\ %H:%M)] $*"; }
until grep -q "GPU0 CHAIN DONE" outputs/logs/gpu0_chain.log 2>/dev/null; do sleep 30; done
log "train 400-epoch baseline"
$P -u -m models.lvsm.train --config configs/lvsm_long.json > outputs/logs/lvsm_long_train.log 2>&1
log "Q2.2 eval of the long baseline"
$P -u -m models.lvsm.eval --checkpoint outputs/lvsm_long/best.pt --num-source-views 1 2 4 --conditioning correct shuffled none \
  --output outputs/lvsm_long/eval_val > outputs/logs/lvsm_long_eval.log 2>&1
log "LONG DONE"
