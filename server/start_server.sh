#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Single vLLM server that powers both:
#   - the regular --backend vllm path (just hits MODEL directly), AND
#   - the --backend vllm_dynamic_lora 2-pass path:
#       Pass 1: model=$MODEL                     (base, generates <think>...</think>)
#       Pass 2: model=$LORA_NAME                 (LoRA, generates the action)
#
# Both passes share the same prompt prefix, so vLLM's prefix cache makes
# Pass 2's prefill effectively free.
#
# Usage:
#   ./start_server.sh                                  # base only
#   ./start_server.sh --lora action_lora=/path/to/adapter   # base + named LoRA
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU=0
PORT=8000
MODEL="Qwen/Qwen3-VL-8B-Instruct"
KEY_FILE="${KEY_FILE:-$HOME/.config/vllm/api.key}"

# Optional LoRA adapter(s). Repeat --lora to register more than one.
# Each value is `name=path`, e.g. `action_lora=/homes/orionf/LlamaFactory/saves/qwen3-vl-8b/lora/aitw_reasoning_tf_100`.
LORA_MODULES=()
MAX_LORA_RANK=64        # must be >= the LoRA rank used at training time
MAX_LORAS=1             # how many adapters can be in-flight concurrently

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)            GPU="$2";            shift 2 ;;
    --port)           PORT="$2";           shift 2 ;;
    --model)          MODEL="$2";          shift 2 ;;
    --lora)           LORA_MODULES+=("$2");shift 2 ;;
    --max-lora-rank)  MAX_LORA_RANK="$2";  shift 2 ;;
    --max-loras)      MAX_LORAS="$2";      shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! -f "$KEY_FILE" ]]; then
  echo "API key not found at $KEY_FILE"
  exit 1
fi

echo "Starting vLLM: model=$MODEL  gpu=$GPU  port=$PORT"
if (( ${#LORA_MODULES[@]} > 0 )); then
  echo "  LoRA adapters: ${LORA_MODULES[*]}"
  echo "  max_lora_rank=$MAX_LORA_RANK  max_loras=$MAX_LORAS"
fi

export VLLM_API_KEY="$(cat "$KEY_FILE")"

LORA_ARGS=()
if (( ${#LORA_MODULES[@]} > 0 )); then
  LORA_ARGS+=(--enable-lora)
  LORA_ARGS+=(--max-lora-rank "$MAX_LORA_RANK")
  LORA_ARGS+=(--max-loras "$MAX_LORAS")
  LORA_ARGS+=(--lora-modules "${LORA_MODULES[@]}")
fi

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
    --host "127.0.0.1" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 31972 \
    --enable-prefix-caching \
    --max_num_seqs 32 \
    "${LORA_ARGS[@]}"