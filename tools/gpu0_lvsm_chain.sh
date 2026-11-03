#!/bin/bash
# GPU0 after the baseline LVSM run: Q2.2 ablation eval, then the Q2.3 bonus (train S=1 and S=4, eval each at 1/2/4).
cd /home/haotian/assignment4_f26
P=/home/haotian/miniconda3/envs/a4/bin/python
export CUDA_VISIBLE_DEVICES=0
log() { echo "[$(date +%m/%d\ %H:%M)] $*"; }
while pgrep -f "models[.]lvsm[.]train --config configs/lvsm.json" > /dev/null; do sleep 30; done
log "baseline done; Q2.2 eval"
$P -u -m models.lvsm.eval --checkpoint outputs/lvsm/best.pt --num-source-views 1 2 4 --conditioning correct shuffled none > outputs/logs/lvsm_eval.log 2>&1
log "Q2.2 eval exit $?"
for S in 1 4; do
  log "train S=$S"
  $P -u -m models.lvsm.train --config configs/lvsm_s$S.json > outputs/logs/lvsm_s${S}_train.log 2>&1
  log "eval S=$S model"
  $P -u -m models.lvsm.eval --checkpoint outputs/lvsm_s$S/best.pt --num-source-views 1 2 4 --conditioning correct \
    --output outputs/lvsm_s$S/eval_val > outputs/logs/lvsm_s${S}_eval.log 2>&1
done
log "GPU0 CHAIN DONE"
