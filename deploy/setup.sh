#!/usr/bin/env bash
# One-time: pull runtime image + model to a single DGX Spark.
set -euo pipefail
docker pull eugr/spark-vllm-b12x:nightly-20260813
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3.8-27B-NVFP4",
                  local_dir="/home/keyspark/models-local-qwen38/Qwen3.8-27B-NVFP4")
print("model ready")
PY
sudo apt-get install -y earlyoom 2>/dev/null && sudo systemctl enable --now earlyoom || true
echo "setup done — run deploy/serve_throughput.sh"
