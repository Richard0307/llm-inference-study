# W3 Wrap-up Memo: GPU Memory Hierarchy + Roofline + Precision Comparison

**Week:** W3 (2026-04-13 ~ 2026-04-19)
**Hardware:** RTX 4090 Laptop GPU (Ada Lovelace, CC 8.9, 16 GB GDDR6X)
**Model:** GPT-2 124M (HuggingFace checkpoint)
**Status:** Post-illness recovery wrap-up — locking in W3 engineering deliverables, no new experiments.

---

## 0. TL;DR

One sentence: **GPT-2 decode on RTX 4090 Laptop is strangled by HBM bandwidth — true FLOPs utilization is <1%; changing precision (FP16/INT8) does not fix this. You must raise arithmetic intensity via KV cache / batching.**

Three numbers:
- **57.3 FLOPs/byte** — 4090 Laptop's Roofline ridge point (FP32)
- **0.5 FLOPs/byte** — GPT-2 decode arithmetic intensity, 115× below the ridge
- **0.36%** — true decode FLOPs utilization (measured GFLOPs/s ÷ peak compute)

---

## 1. Roofline Analysis Conclusions

### 1.1 Hardware baseline

| Quantity | Value |
|---|---:|
| Peak FP32 compute | 33 TFLOPs |
| HBM bandwidth (GDDR6X) | 576 GB/s |
| **Ridge point** | **57.3 FLOPs/byte** |

Below ridge = memory-bound, above = compute-bound.

### 1.2 Where GPT-2 sits on the Roofline

| Stage | Arithmetic Intensity | vs Ridge | Regime | Roofline ceiling |
|---|---:|:---:|:---:|---:|
| Decode (any prompt length) | **0.5 FLOPs/byte** | 115× below | **memory-bound** | 288 GFLOPs/s |
| Prefill L=32 | 16 | below | memory-bound | 9.2 TFLOPs/s |
| Prefill L=64 | 32 | below | memory-bound | 18.4 TFLOPs/s |
| Prefill L=128 | 64 | **above** | **compute-bound** | 33 TFLOPs/s |
| Prefill L=256 | 128 | above | compute-bound | 33 TFLOPs/s |

Prefill crosses into compute-bound around L=128. Decode lives permanently at the bottom of the memory-bound valley.

### 1.3 Roofline vs W2 measurements

| prompt_len | prefill GFLOPs/s | ceiling | efficiency | decode GFLOPs/s | ceiling | efficiency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32  | 3,167  | 9,216  | 34% | 120 | 288 | 41.5% |
| 64  | 6,049  | 18,432 | 33% | 121 | 288 | 42.0% |
| 128 | 11,123 | 33,000 | 34% | 120 | 288 | 41.5% |
| 256 | 13,875 | 33,000 | 42% | 120 | 288 | 41.5% |

Decode is pinned at ~120 GFLOPs/s (≈ 482 tok/s) regardless of prompt length. Root cause: at bs=1 every step reloads the full 496 MB of weights from HBM but only produces 1 token.

Full derivation: [roofline_notes.md](roofline_notes.md) and [w2_profiling_roofline_interpretation.md](w2_profiling_roofline_interpretation.md).
Chart: [roofline_chart.png](roofline_chart.png)

---

## 2. GPU Memory Hierarchy

Figure: [gpu_memory_hierarchy.png](gpu_memory_hierarchy.png) (source: [gpu_memory_hierarchy_diagram.py](gpu_memory_hierarchy_diagram.py))

Top-down levels:

| Level | Capacity (RTX 4090 Laptop) | Bandwidth (est.) | Latency | Visibility |
|---|---:|---:|:---:|---|
| Register | ~256 KB / SM | — | 1 cycle | single warp |
| L1 / Shared Memory (SRAM) | 128 KB / SM | ~19 TB/s | ~30 cycles | all warps in one SM |
| L2 Cache | 48 MB (chip-wide) | ~5 TB/s | ~200 cycles | all SMs |
| **HBM / GDDR6X (device memory)** | **16 GB** | **576 GB/s** | ~400 cycles | all SMs |
| Host DRAM (via PCIe 4.0 x16) | system RAM | ~32 GB/s | ~ms | CPU-GPU DMA |

**Key insight:** there is a **30×+ bandwidth cliff** between HBM and SRAM. The essence of LLM decode bottleneck is this exact hop — the 500 MB of weights does not fit in SRAM, so every decode step must stream the whole thing from HBM back to registers to run a GEMV. Every classical optimization fights this hop:

- **KV cache**: past-step K/V no longer has to be recomputed from HBM
- **Flash Attention**: keep softmax intermediates in SRAM instead of writing them back to HBM
- **Quantization**: fewer bytes per weight — effectively widens HBM's usable bandwidth

---

## 3. FP32 / FP16 / INT8 Latency Comparison

### 3.1 End-to-end latency and memory (GPT-2 124M, bs=1, seq_len=128, decode 32 steps)

| Precision | Prefill (ms) | Decode (ms/token) | Peak Mem (MB) | vs FP32 |
|---|---:|---:|---:|---|
| **FP32** | 3.29  | 2.58 | 542.4 | baseline |
| **FP16** | 3.05  | 2.64 | 508.5 | prefill 7% faster, decode flat, memory −6% |
| **INT8** (bnb) | 14.64 | 9.94 | 320.4 | prefill 4.4× slower, decode 3.85× slower, memory −41% |

Raw data: [precision_latency.json](precision_latency.json)
Chart: [precision_latency_comparison.png](precision_latency_comparison.png)

### 3.2 Nsight Compute key metrics (decode, single step)

| Precision | Kernels | DRAM avg | **DRAM peak** | Tensor Core HMMA peak | SM active avg |
|---|---:|---:|---:|---:|---:|
| FP32 | 235 | 17.9% | **96.1%** | 4.1% | 15.3% |
| FP16 | 235 | 13.3% | **94.6%** | 4.3% | 16.8% |
| INT8 | 555 | 4.5%  | **94.7%** | 4.3% | 10.8% |

Chart: [ncu_utilization_comparison.png](ncu_utilization_comparison.png)
Raw data: [ncu_decode_fp32.csv](ncu_decode_fp32.csv) / [ncu_decode_fp16.csv](ncu_decode_fp16.csv) / [ncu_decode_int8.csv](ncu_decode_int8.csv)

### 3.3 Three conclusions

**(1) All three precisions hit ≈95% DRAM peak during decode — memory-boundedness is a hardware fact, not a precision problem.**
Hot kernels across all three are `gemv*` variants (General Matrix-Vector): the weight matrix streams in from HBM in full, gets multiplied by a single vector, and arithmetic intensity stays identical.

**(2) Tensor Cores fire only in prefill FP16 — decode never uses them.**
HMMA instructions require both inputs to be matrices. Decode's activation is a vector, so it falls back to CUDA Core FMA. That is exactly why every serving system (vLLM, TGI) pours engineering into batching — turning the batch dimension into the second matrix dimension, so decode GEMV becomes a small GEMM and Tensor Cores can finally engage.

**(3) INT8 via bitsandbytes is a net slowdown on small models like GPT-2.**
- Every matmul is wrapped in quant/dequant kernels → kernel count jumps from 235 to 555 (+136%)
- `gemmSN_kernel_int32` uses an INT32 accumulator — throughput is far below same-size FP16 Tensor Core GEMM
- INT8's real value is "doesn't fit → fits" (e.g. 13B on a 24 GB card). Small models don't need this trade

**Implication for W4/W5:** do not use bnb INT8 as a GPT-2 acceleration knob; the real decode wins come from **KV cache + continuous batching + MQA/GQA** — all of which reduce HBM traffic, aligned with the Roofline diagnosis.

---

## 4. Key Insights

### 4.1 Decode is deeply memory-bound, not compute-bound

- Arithmetic intensity 0.5 FLOPs/byte, 115× below ridge
- DRAM peak ≈95% across all three precisions — HBM is saturated
- Of the ~2 ms/token measured, the theoretical bandwidth floor is 0.86 ms (496 MB / 576 GB/s); the remaining 1.2 ms is kernel launch / Python-CUDA sync / KV read overhead
- Swapping the 4090 for an H100 would only shift decode linearly with bandwidth; more FLOPs buy you nothing here

### 4.2 True FLOPs utilization <1% (and the <5% hypothesis holds)

| Metric | Value | Notes |
|---|---:|---|
| Roofline decode ceiling | 288 GFLOPs/s | = AI × bandwidth = 0.5 × 576 |
| Measured decode throughput | 120 GFLOPs/s | W2 profiling |
| **True FLOPs utilization** | **0.36%** | 120 / 33,000 |
| ncu SM active rate | 10–17% | only means warp resident, not FLOPs utilization |
| ncu HMMA peak (decode) | ~4% | actual Tensor Core utilization |

The original W3 hypothesis — "true FLOPs utilization < 5%" — **holds**, and is in fact much tighter (<1%).

**Gotcha reminder:** `nvidia-smi dmon` shows `sm% = 64.6%` during decode and makes the GPU look busy — but that metric counts memory-stalled warps as resident. The honest answer is the 0.36% from Roofline. For LLM performance diagnosis, trust ncu's `sm__throughput` and `dram__throughput`; ignore `nvidia-smi`'s `sm%`.

### 4.3 Prefill eats compute, decode cannot — diagnose them separately

| Stage | Bottleneck | Right optimizations |
|---|---|---|
| **Prefill** (GEMM) | compute-bound once L ≥ 128 | FlashAttention, larger batch, FP16/BF16 + Tensor Core |
| **Decode** (GEMV) | always memory-bound | KV cache, continuous batching, MQA/GQA, speculative decoding, weight quantization |

"Treating them as one optimization problem" is the most common engineer trap. On the Roofline chart they are not even in the same region.

### 4.4 Every classical optimization = moving the red dot on the Roofline

- **Right** (raise AI): KV cache, batching, speculative decoding
- **Up** (raise the ceiling): quantization (FP16 / INT8 / INT4), Flash Attention (cut HBM round-trips)

Every optimization learned through end of April can be classified as one of these two moves. This is the anchor for everything that follows.

---

## 5. Hypothesis Scorecard

| Hypothesis | Prediction | Measured | Verdict |
|---|---|---|---|
| H1a decode SM util < 30% | 15–25% | 10.8–16.8% (ncu smsp active) | ✅ holds, lower than predicted |
| H1b decode mem util > 60% | > 60% | DRAM peak 94–96% | ✅ strongly holds |
| H3 prefill util > decode util | prefill > 70%, decode < 30% | prefill 66% / decode 15% | ✅ direction right, prefill a bit lower |
| **True FLOPs utilization < 5%** | < 5% | **0.36%** (Roofline) / 4% (HMMA peak) | ✅ far below 5% |
| H2 larger batch improves util | — | not tested this week | ⏳ deferred to W4/W5 |

Primary hypotheses all hold. W3 engineering work done.

---

## 6. W3 Deliverables

- [x] [roofline_notes.md](roofline_notes.md) — Roofline theory + 4090 Laptop ridge derivation + Q&A
- [x] [w2_profiling_roofline_interpretation.md](w2_profiling_roofline_interpretation.md) — Roofline-based reading of W2 data
- [x] [roofline_chart.png](roofline_chart.png) — Roofline chart with GPT-2 prefill/decode points
- [x] [gpu_memory_hierarchy.png](gpu_memory_hierarchy.png) — GPU memory hierarchy diagram
- [x] [precision_latency.json](precision_latency.json) + [precision_latency_comparison.png](precision_latency_comparison.png) — FP32/FP16/INT8 latency comparison
- [x] `ncu_{prefill,decode}_{fp32,fp16,int8}.csv` (6 files) — Nsight Compute raw data
- [x] [ncu_utilization_comparison.png](ncu_utilization_comparison.png) — ncu utilization comparison chart
- [x] [W3_hypothesis.md](W3_hypothesis.md) — W3 hypothesis doc with ncu appendix
- [x] [w3_memo.md](w3_memo.md) — W3 memo (Chinese)
- [x] This file (w3_memo_en.md)

**Not done (rolled into W4):** Docker Compose / FastAPI / Pydantic; 9 unfinished LeetCode problems (Hashmap Day 2–4 + Two Pointer Day 1); this week's Zhihu blog.

---

## 7. W4 Hypothesis (starting 4/20)

> **H4: On RTX 4090 Laptop, with bs=1 and FP32 GPT-2, enabling KV cache yields >2× decode speedup at seq_len ≥ 256 versus the no-cache baseline.**

### Reasoning

- Without cache, decode step *t* recomputes K/V for the previous *t−1* tokens (O(t × d_model²))
- With cache, only the current token's K/V is computed each step (O(d_model²)); savings grow quadratically with seq_len
- At seq_len ≥ 256, both saved FLOPs and saved HBM traffic become dominant (>2× is a reasonable target)
- The mechanism is "fewer repeated HBM reads" — aligned with W3's memory-bound diagnosis

### Acceptance criteria

| seq_len | Expected decode ms/token delta | Expected DRAM peak delta |
|---|---|---|
| 32 | roughly flat or <1.5× | flat |
| 128 | 1.5–2× | slight drop |
| **256** | **>2×** | clear drop |
| 512 | >3× | significant drop |

If DRAM utilization drops after enabling KV cache but latency does not, the bottleneck has shifted to **kernel launch overhead / CPU-GPU sync** — that becomes the next target (CUDA Graph, torch.compile, etc.).

### Observables

- **HMMA utilization:** cache alone should not raise decode HMMA (still a GEMV) unless combined with batching
- **Kernel count:** cache should not blow up kernel count, else launch overhead eats the win
- **Memory growth curve:** `2 × n_layers × n_heads × head_dim × seq_len × batch × bytes` — probe until OOM

Full plan: see W4 section in [roadmap/2026_04_Daily_Plan.md](../roadmap/2026_04_Daily_Plan.md).

---

*Written: 2026-04-19 (post-illness W3 wrap-up)*
