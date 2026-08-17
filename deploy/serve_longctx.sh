#!/usr/bin/env bash
# Profile B: 1M context via YaRN 4x. ~2 concurrent full-1M (pool 2.31M tokens);
# or c=8 as long as combined context <= 2.31M (~288K avg). YaRN softens quality.
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
    --max-model-len 1048576 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 \
    --enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
echo "Profile B up: 1M context (YaRN), c~2 full-1M. http://localhost:8078/v1"

# --- first-load warmup (see README "first stuck prompt" gotcha) ---------------
# Compile the LARGE-prefill path before any client connects. A cold serve that
# takes a big first prompt (e.g. Hermes injecting a ~20K-token system prompt) can
# stall or return a garbled/"stunned" first reply. A tiny "hello" warm request
# does NOT cover this — it must be a large prompt (> the client's first load).
echo "Warming the large-prefill path (first-load fix)..."
python3 - <<'PY'
import json, time, urllib.request
base = "http://localhost:8078"
for _ in range(180):
    try: urllib.request.urlopen(base + "/v1/models", timeout=3); break
    except Exception: time.sleep(5)
prompt = ("Unified memory bandwidth bounds decode throughput on edge accelerators today. " * 2200) + "\nReply with: OK"
body = json.dumps({"model": "qwen38-nvfp4", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
try:
    urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=300).read()
    print("  warmup ok (~26K-token prefill compiled) — first client prompt will be fast")
except Exception as e:
    print("  WARN warmup request failed (serve may still be compiling):", e)
PY
# --- GB10 vLLM spin-wait fix (see GB10_SPIN_WAIT_PATCH.md) --------------------
# If this script runs a stock vLLM image, the served container will busy-spin CPU
# cores at max clock while waiting on shm_broadcast (busy_loop_s=1s default),
# heating the shared GB10 SoC. Prefer an image built with the patch baked in.
# https://nacyot.github.io/artifacts/vllm-spin-wait-gb10/
