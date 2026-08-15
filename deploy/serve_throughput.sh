#!/usr/bin/env bash
# Profile A (default): c=8+ @ 256K context, native quality. Champion config.
set -euo pipefail
MODELS="${MODELS:-/home/keyspark/models-local-qwen38}"
docker rm -f qwen38 2>/dev/null || true
docker run -d --restart unless-stopped --name qwen38 --gpus all --ipc=host --network host \
  -v "$MODELS":/models \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  eugr/spark-vllm-b12x:nightly-20260813 \
  vllm serve /models/Qwen3.8-27B-NVFP4 --served-model-name qwen38-nvfp4 \
    --host 0.0.0.0 --port 8078 \
    --max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 \
    --enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
echo "Profile A up: c=8.8 @ 256K, MTP-3, 31.7 tok/s single. http://localhost:8078/v1"
