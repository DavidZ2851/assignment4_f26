#!/bin/bash
# GPU1 after Mini-VGGT training: Q3.2 val/test eval (8 views of the first 32 frames, 224px, seed 825), then the Q3.3
# bonus (val metrics at 2/4/8/16 views + latency/memory benchmark).
cd /home/haotian/assignment4_f26
P=/home/haotian/miniconda3/envs/a4/bin/python
export CUDA_VISIBLE_DEVICES=1
log() { echo "[$(date +%m/%d\ %H:%M)] $*"; }
while pgrep -f "models[.]vggt[.]train --config" > /dev/null; do sleep 60; done
log "training done; Q3.2 eval"
E="--eval-root data/scannetpp/prepared_eval --eval-num-frames 32 --image-size 224 --seed 825"
$P -u -m models.vggt.eval --checkpoint outputs/vggt_train/best.pt --split val $E --num-views 8 \
  --report outputs/vggt_train/metrics.json --visualization outputs/vggt_train/visualization/index.html > outputs/logs/vggt_eval_val.log 2>&1
log "val exit $?"
$P -u -m models.vggt.eval --checkpoint outputs/vggt_train/best.pt --split test $E --num-views 8 \
  --report outputs/vggt_train/test_metrics.json > outputs/logs/vggt_eval_test.log 2>&1
log "test exit $?"
for V in 2 4 16; do
  $P -u -m models.vggt.eval --checkpoint outputs/vggt_train/best.pt --split val $E --num-views $V \
    --report outputs/vggt_train/bonus_views/val_v$V.json > outputs/logs/vggt_eval_val_v$V.log 2>&1
  log "views $V exit $?"
done
$P -u tools/vggt_views_bench.py outputs/vggt_train/best.pt outputs/vggt_train/bonus_views/bench.json 2 4 8 16 24 32 > outputs/logs/vggt_bench.log 2>&1
log "GPU1 CHAIN DONE"
