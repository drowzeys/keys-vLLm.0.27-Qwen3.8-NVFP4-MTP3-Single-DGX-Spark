# keys-vLLm.0.27 — Qwen3.8-27B-NVFP4 + MTP-3 on a Single DGX Spark

Tuned single-node serving of **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)**
on one **DGX Spark (GB10, 121 GB UMA)** using vLLM 0.27 (eugr `spark-vllm-b12x` nightly).
The headline: the NVFP4 checkpoint ships with **MTP (multi-token-prediction) heads baked
in**, and enabling MTP speculative decoding at depth 3 nearly **triples single-stream
throughput** — 11.1 → **31.7 tok/s** — while staying on a native Blackwell FP4 GEMM.

All numbers below are **measured on the hardware**, not estimated.

---

## TL;DR champion config

```
Model:      unsloth/Qwen3.8-27B-NVFP4   (native FP4, MTP head included)
Runtime:    vLLM 0.27 nightly (eugr/spark-vllm-b12x)
Decode:     MTP speculative decoding, num_speculative_tokens=3   ← the whole win
KV cache:   fp8
Speed:      31.7 tok/s single-stream (2.85x the 11.1 baseline)
KV pool:    2.31M tokens  →  8.8x concurrency at 256K context
```

## Measured speed ladder (single-stream decode, 864-token gen)

| Config | tok/s | vs base | Note |
|---|---:|---:|---|
| baseline (no spec) | 11.1 | 1.00x | 61% of the ~18 tok/s GB10 memory-bw ceiling |
| ngram spec | 11.7 | 1.05x | **wash** — reasoning prose isn't n-gram-predictable |
| MTP n=1 | 19.4 | 1.75x | native MTP head |
| MTP n=2 | 26.3 | 2.37x | |
| **MTP n=3** | **31.7** | **2.85x** | **champion** — exceeds naive bw ceiling (verifies >1 tok/load) |
| MTP n=4 / n=5 | — | — | **crash**: model has 1 MTP layer, depth ≥4 emits invalid tokens |

MTP depth is capped at 3 because Qwen3.8-27B has exactly **one** MTP layer
(`mtp_num_hidden_layers=1`); vLLM runs it autoregressively for n≤3, acceptance
collapses beyond.

## Measured concurrency (aggregate tok/s, MTP-3)

| concurrency | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| agg tok/s | 21.6 | 108.9 | 183.9 | 282.6 | 245.8 | **313.0** |

## NVFP4 vs FP8 (both with MTP-3) — NVFP4 wins

| quant | single tok/s | 64-way agg | why |
|---|---:|---:|---|
| **NVFP4** | **31.7** | **313** | ~13.5 GB weights → less memory traffic; decode is memory-bound |
| FP8 | 21.9 | 225 | ~27 GB weights = 2x traffic → slower decode (FP8 only wins on quality) |

The NVFP4 path is a **native `FlashInferCutlassNvFp4LinearKernel` GEMM** on Blackwell,
confirmed in the engine log — not a bf16/fp8 fallback.

---

## 🎛️ Context vs concurrency — the single-Spark tradeoff

The KV cache is a **shared pool of 2,311,633 tokens** (fp8 KV, util 0.90). Any mix of
requests fits as long as **Σ(context lengths) ≤ 2.31M**. So on ONE Spark you choose a
point on this curve — you cannot have both ends at once:

| Per-request context | Max concurrent (full context) |
|---|---:|
| 256K | **8.8x**  ← throughput profile |
| 288K | 8.0x |
| 512K | 4.4x |
| **1M** (needs YaRN) | **2.2x**  ← long-context profile |

> **Straight talk:** *c=8 at a full 1M context is NOT possible on one GB10* — that needs
> 8.4M KV tokens vs the 2.31M pool (3.6x over). Real 1M-at-high-concurrency requires
> **TP=2+ across multiple Sparks**. On a single node you pick throughput **or** ultra-long
> context. Both profiles ship below.

### Profile A — throughput (default): c=8 @ 256K
[`deploy/serve_throughput.sh`](deploy/serve_throughput.sh) — 8+ concurrent full-256K
sessions, native quality (no rope scaling), 313 tok/s aggregate at 64-way. **This is the
recommended production config for almost everyone** (256K covers virtually all real use).

### Profile B — long-context: c=2 @ 1M
[`deploy/serve_longctx.sh`](deploy/serve_longctx.sh) — YaRN 4x rope-scaling extends the
256K native window to 1M; the 2.31M pool serves ~2 concurrent full-1M requests (or c=8 as
long as their *combined* length ≤ 2.31M, e.g. ~288K avg). YaRN slightly softens quality —
use only if you genuinely exceed 256K.

## Quick start

```bash
# one DGX Spark, model at /models/Qwen3.8-27B-NVFP4:
bash deploy/setup.sh                    # pull image + model
bash deploy/serve_throughput.sh         # Profile A: c=8 @ 256K (default)
# or:
bash deploy/serve_longctx.sh            # Profile B: 1M context (YaRN), c≈2 full-1M
bash bench/run_bench.sh http://localhost:8078   # reproduce the tables above
```

## Hardware / software

| | |
|---|---|
| GPU | 1x DGX Spark (GB10, sm_121a, 121 GB LPDDR5X ~273 GB/s) |
| Runtime | `eugr/spark-vllm-b12x:nightly-20260813` (vLLM 0.27, loads the new `Qwen3_5MTP` arch that stable vLLM can't) |
| Env | `FLASHINFER_CUDA_ARCH_LIST=12.1a`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |

## Notes / gotchas (measured)

- **ngram spec decoding is useless here** — and its cudagraph profiler asserts
  `kv_len ≥ spec+1` and crashes at high depth; MTP is the right lever.
- The DSV4F `VLLM_B12X_W4A16_*` GEMM knobs are **no-ops** on the eugr image (they belong
  to the anemll dspark image); eugr's NVFP4 autotuner is already near the bw ceiling.
- vLLM auto-selects `kv_cache_dtype=fp8` under MTP; we set it explicitly.
- GB10 has too few SMs for `max_autotune_gemm` — expected, not an error.

*Sibling repos: [MiniMax-H3 parallel film factory](https://github.com/drowzeys/keys-MiniMax-H3-Nvidia-Sol-Engine-Two-or-More-DGX-Sparks-Parallel-Processing-Enabled) ·
[DSV4F Power Pack](https://github.com/drowzeys/keys-DGX-Sparkticus-Ultimate-Power-Pack-Unleashed-Dual-DGX-Sparks-Needed).*
