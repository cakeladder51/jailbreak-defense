#!/bin/bash
# Phase 1 experiment: does the signal survive a HARD benign set?
#
# Same 70 attack prompts as the 2026-07-27 run, but benign is now 35 Alpaca (easy) +
# 35 XSTest-safe (hard: prompts that look harmful but are not). separation_report()
# breaks the AUC out against each benign source, so the two are directly comparable
# within one run -- that comparison IS the experiment.
#
# SafeDecoding is skipped: it is a baseline for the final table and contributes nothing
# to this question, and it is ~a third of the wall clock.
#
# Serial, one model on the card at a time -- see run_all.sh for why.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export CGP_DEFENSES=none,cgp

run() {
  tag="$1"; model="$2"; quant="$3"
  echo "### START $tag ($model, $quant) $(date +%H:%M:%S)"
  python -u run_cgp.py "$tag" "$model" "$quant" > "run_$tag.log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "### DONE $tag $(date +%H:%M:%S)"
  else
    echo "### FAILED $tag exit=$code $(date +%H:%M:%S) -- last lines:"
    tail -15 "run_$tag.log"
  fi
}

run gemma_hb gemma-3-4b bf16
run qwen_hb  qwen2.5-3b bf16
run phi3_hb  phi-3-mini bf16
echo "### ALL HARD-BENIGN RUNS FINISHED $(date +%H:%M:%S)"
