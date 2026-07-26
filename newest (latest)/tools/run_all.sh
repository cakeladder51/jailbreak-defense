#!/bin/bash
# Sequential GPU runs -- ONE model on the card at a time. Running two concurrently is what
# caused the OOM on 2026-07-26: caching_allocator_warmup asks for a single contiguous block
# (8 GiB for gemma bf16) and Windows WDDM refuses while another process holds VRAM, even
# with ~14.5 GiB nominally free. Keep these serial.
cd "$(dirname "$0")"
# NOTE: torch logs "expandable_segments not supported on this platform" on Windows, so this
# is a no-op here; it is kept only so the same script helps on Linux. Serial execution is
# the actual fix.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# hf_hub tries to symlink into the snapshot dir; without Developer Mode that raises
# WinError 1314 (it killed the first llama prefetch after the weights had downloaded).
export HF_HUB_DISABLE_SYMLINKS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

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

run gemma_bf16 gemma-3-4b bf16
run gemma_4bit gemma-3-4b 4bit
run qwen_bf16  qwen2.5-3b bf16
echo "### ALL GPU RUNS FINISHED $(date +%H:%M:%S)"
