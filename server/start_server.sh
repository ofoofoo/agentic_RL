#!/usr/bin/env bash
set -euo pipefail

GPU=0
PORT=8000
MODEL="Qwen/Qwen3-VL-8B-Instruct"
LORA_NAME=""
LORA_PATH=""
MAX_LORA_RANK=""
KEY_FILE="${KEY_FILE:-$HOME/.config/vllm/api.key}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)   GPU="$2";   shift 2 ;;
    --port)  PORT="$2";  shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --lora-name) LORA_NAME="$2"; shift 2 ;;
    --lora-path) LORA_PATH="$2"; shift 2 ;;
    --max-lora-rank) MAX_LORA_RANK="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! -f "$KEY_FILE" ]]; then
  echo "API key not found at $KEY_FILE"
  exit 1
fi

echo "Starting vLLM: model=$MODEL  gpu=$GPU  port=$PORT"

export VLLM_API_KEY="$(cat "$KEY_FILE")"
CMD=(vllm serve "$MODEL"
    --host "127.0.0.1"
    --port "$PORT"
    --dtype bfloat16
    --max-model-len 31972
    --enable-prefix-caching
    --max_num_seqs 32
)

if [[ -n "$LORA_NAME" || -n "$LORA_PATH" ]]; then
  if [[ -z "$LORA_NAME" || -z "$LORA_PATH" ]]; then
    echo "If using LoRA, both --lora-name and --lora-path are required."
    exit 1
  fi
  echo "  LoRA enabled: $LORA_NAME=$LORA_PATH"
  CMD+=(--enable-lora --lora-modules "${LORA_NAME}=${LORA_PATH}")
  if [[ -n "$MAX_LORA_RANK" ]]; then
    echo "  max_lora_rank=$MAX_LORA_RANK"
    CMD+=(--max-lora-rank "$MAX_LORA_RANK")
  fi
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"