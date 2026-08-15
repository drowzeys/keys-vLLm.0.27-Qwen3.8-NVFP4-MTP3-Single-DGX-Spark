#!/usr/bin/env bash
# =============================================================================
# ONE-SHOT: Qwen3.8-27B-NVFP4 + MTP-3 on a single DGX Spark (GB10 / sm_121a)
# =============================================================================
# Idempotent and self-checking. Re-run any time; it skips finished steps.
#
#   bash oneshot.sh                 # Profile A: c=8 @ 256K (default, recommended)
#   PROFILE=longctx bash oneshot.sh # Profile B: 1M context via YaRN, c~2 full-1M
#
# Requires: a DGX Spark (GB10), Docker with the NVIDIA runtime, ~60 GB free disk,
# internet for the first pull. NOTHING else — the GB10 vLLM image is pinned and
# mirrored, so official-vLLM's missing sm_121a support can't bite you.
# =============================================================================
set -euo pipefail

# ---- pinned, immutable components (this is the "can't go wrong" part) --------
# Primary = our pinned mirror; fallback = eugr's upstream nightly (in case the
# mirror package is momentarily unreachable). Either is a valid GB10 build.
IMAGE="ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813"   # mirror of eugr's GB10 nightly (credit @eugr)
IMAGE_FALLBACK="eugr/spark-vllm-b12x:nightly-20260813"
MODEL_REPO="unsloth/Qwen3.8-27B-NVFP4"
MODELS_DIR="${MODELS_DIR:-$HOME/models-local-qwen38}"
MODEL_DIR="$MODELS_DIR/Qwen3.8-27B-NVFP4"
PORT="${PORT:-8078}"
PROFILE="${PROFILE:-throughput}"
NAME="qwen38"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- 0. preflight ------------------------------------------------------------
say "0/5 preflight"
command -v docker >/dev/null || die "docker not installed"
docker info >/dev/null 2>&1 || die "docker daemon not reachable (need sudo? add your user to the docker group)"
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L >/dev/null 2>&1 \
  || die "GPU not visible to docker — install the NVIDIA container runtime"
arch="$(docker run --rm --gpus all ubuntu:22.04 sh -c 'nvidia-smi --query-gpu=name --format=csv,noheader' 2>/dev/null || true)"
echo "  GPU: ${arch:-unknown}"
case "$arch" in *GB10*|*Spark*|*Blackwell*) : ;; *) echo "  WARNING: this build targets GB10 (sm_121a); '$arch' may not run the FP4 kernels";; esac
free_gb=$(df -BG --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
[ "${free_gb:-0}" -ge 45 ] 2>/dev/null || echo "  WARNING: <45 GB free at $MODELS_DIR (image ~23G + model ~22G)"

# ---- 1. pull the pinned GB10 image ------------------------------------------
say "1/5 pull runtime image (pinned; official vLLM 0.27 lacks sm_121a — this build has it)"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  image present"
elif docker pull "$IMAGE" 2>/dev/null; then
  echo "  pulled mirror"
elif docker pull "$IMAGE_FALLBACK" 2>/dev/null; then
  IMAGE="$IMAGE_FALLBACK"; echo "  mirror unreachable — pulled upstream fallback ($IMAGE_FALLBACK)"
else
  die "could not pull the GB10 image (mirror + fallback both failed — check internet)"
fi

# ---- 2. fetch the model (resumable) -----------------------------------------
say "2/5 fetch $MODEL_REPO -> $MODEL_DIR"
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  echo "  model present"
else
  mkdir -p "$MODEL_DIR"
  python3 - "$MODEL_REPO" "$MODEL_DIR" <<'PY' || die "model download failed (pip install -U huggingface_hub, then re-run)"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], resume_download=True)
print("  model downloaded")
PY
fi
[ -f "$MODEL_DIR/config.json" ] || die "model incomplete (no config.json)"

# ---- 3. launch the serve -----------------------------------------------------
say "3/5 launch serve (profile: $PROFILE)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
COMMON=(--restart unless-stopped --name "$NAME" --gpus all --ipc=host --network host
  -v "$MODELS_DIR":/models
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "$IMAGE"
  vllm serve /models/Qwen3.8-27B-NVFP4 --served-model-name qwen38-nvfp4
  --host 0.0.0.0 --port "$PORT" --kv-cache-dtype fp8 --enable-flashinfer-autotune
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
if [ "$PROFILE" = "longctx" ]; then
  docker run -d "${COMMON[@]}" --max-model-len 1048576 --gpu-memory-utilization 0.92 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' >/dev/null \
    || die "launch failed"
  echo "  Profile B: 1M context (YaRN), ~2 concurrent full-1M"
else
  docker run -d "${COMMON[@]}" --max-model-len 262144 --gpu-memory-utilization 0.90 >/dev/null \
    || die "launch failed"
  echo "  Profile A: c=8.8 @ 256K, native quality"
fi

# ---- 4. wait for health (real timeout, shows logs on failure) ---------------
say "4/5 wait for serve health (first run compiles FP4 kernels; up to ~12 min)"
for i in $(seq 1 150); do
  curl -sf -m3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "  healthy"; break; }
  [ "$i" = 150 ] && { docker logs --tail 40 "$NAME"; die "serve did not become healthy"; }
  sleep 5
done

# ---- 5. smoke test (confirms MTP + endpoint) --------------------------------
say "5/5 smoke test"
docker logs "$NAME" 2>&1 | grep -iE "MTP model|GPU KV cache size|Maximum concurrency" | tail -3 | sed 's/^/  /'
out=$(curl -s -m 60 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-nvfp4","messages":[{"role":"user","content":"Reply with exactly: READY"}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null || true)
[ -n "$out" ] || die "smoke test got no response"
echo "  model replied: $out"

cat <<DONE

\033[1;32m✅ DONE.\033[0m  Qwen3.8-27B-NVFP4 + MTP-3 serving on:
    http://localhost:$PORT/v1   (model id: qwen38-nvfp4)

  Test:   curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\
            -d '{"model":"qwen38-nvfp4","messages":[{"role":"user","content":"hi"}]}'
  Bench:  bash bench/run_bench.sh http://localhost:$PORT
  Logs:   docker logs -f $NAME
  Stop:   docker rm -f $NAME
DONE
