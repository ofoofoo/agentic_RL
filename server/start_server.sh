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

export VLLM_API_KEY="$(cat "$KEY_FILE")"

# model="Qwen/Qwen2.5-VL-32B-Instruct"
#model=Qwen/Qwen3.5-9B
model=Qwen/Qwen3-VL-8B-Instruct
# model="nvidia/Cosmos-Reason1-7B"

 vllm serve $model \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 128000 \
    --enable-prefix-caching \
    --tensor-parallel-size 4 \
    --max_num_seqs 32 \
    #--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'