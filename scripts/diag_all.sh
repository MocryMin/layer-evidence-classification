#!/usr/bin/env bash
# Sequential driver for EXP-20260729-002 diagnostic phases (python -u for live logs).
# stats already completed; runs the remaining phases one at a time to avoid
# SQLite MLflow write contention and to keep GPU use single-stream.
set -u
cd ~/projects/layer-evidence-classification
source ~/projects/ai-env/bin/activate
export PYTHONPATH=.
LOG=/tmp/diag/diag_all.log
echo "=== driver start $(date +%H:%M:%S) ===" > "$LOG"
for phase in lr_grid ln_ablation optimizer no_prompt ft_backbone; do
  echo ">>> [$phase] start $(date +%H:%M:%S)" >> "$LOG"
  python -u scripts/diag_run.py "$phase" >> "$LOG" 2>&1
  rc=$?
  echo "<<< [$phase] done rc=$rc at $(date +%H:%M:%S)" >> "$LOG"
done
echo "=== driver finished $(date +%H:%M:%S) ===" >> "$LOG"
