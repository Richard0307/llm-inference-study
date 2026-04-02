#!/usr/bin/env bash
set -euo pipefail

MODEL="${LOCAL_QWEN_MODEL:-Qwen/Qwen3-8B-AWQ}"
VLLM_HOST="${LOCAL_LLM_HOST:-127.0.0.1}"
VLLM_PORT="${LOCAL_LLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
API_KEY="${LOCAL_LLM_API_KEY:-EMPTY}"

echo "Starting local vLLM server"
echo "Model: ${MODEL}"
echo "Host: ${VLLM_HOST}"
echo "Port: ${VLLM_PORT}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max num seqs: ${MAX_NUM_SEQS}"

cmd=(
  conda run -n llm-infer
  vllm serve "${MODEL}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --api-key "${API_KEY}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

case "${MODEL}" in
  *AWQ*)
    cmd+=(--quantization awq)
    ;;
  *GPTQ*)
    cmd+=(--quantization gptq)
    ;;
esac

exec "${cmd[@]}"
