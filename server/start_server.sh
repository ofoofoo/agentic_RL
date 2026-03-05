#!/usr/bin/env bash
set -euo pipefail


PORT="${PORT:-8000}"
HOST="127.0.0.1" 
KEY_FILE="${KEY_FILE:-$HOME/.config/vllm/api.key}"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "API key not found at $KEY_FILE"
  echo "You can create one with:"
  echo "  mkdir -p $(dirname "$KEY_FILE") && python -c 'import secrets; print(secrets.token_urlsafe(48))' > \"$KEY_FILE\" && chmod 600 \"$KEY_FILE\""
  exit 1
fi

# Set CUDA_HOME so FlashInfer can find nvcc for JIT compilation
# Adjust this path to match your actual CUDA install (run 'which nvcc' to find it)
export CUDA_HOME="/usr/local/cuda-12.2"
export PATH="$CUDA_HOME/bin:$PATH"
model=Qwen/Qwen3-VL-8B-Instruct
# model=Qwen/Qwen3.5-9B

# ── SGLang (active) ──────────────────────────────────────────────────────────
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
    --model $model \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --context-length 131072 \
    --tensor-parallel-size 4 \
    --api-key "$(cat "$KEY_FILE")"

# vllm
# export VLLM_API_KEY="$(cat "$KEY_FILE")"
# vllm serve $model \
#     --host "$HOST" \
#     --port "$PORT" \
#     --dtype bfloat16 \
#     --max-model-len 128000 \
#     --enable-prefix-caching \
#     --tensor-parallel-size 4 \
#     --max_num_seqs 32