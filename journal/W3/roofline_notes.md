# Roofline Model Notes

**Week:** W3 (2026-04-13 ~ 2026-04-19)
**Goal:** understand compute-bound vs memory-bound, and use Roofline to explain why GPT-2 decode is stuck at ~500 tok/s.

---

## 1. The Roofline Model in One Picture

```
Performance (GFLOPs/s)
    ^
    |                         ___________________  <- Peak compute (ceiling)
    |                        /
    |                       /
    |                      /   COMPUTE-BOUND
    |                     /    (right of ridge)
    |                    /
    |                   /
    |                  / <- slope = peak bandwidth
    |                 /
    |                /  MEMORY-BOUND
    |               /   (left of ridge)
    |              /
    +-------------+------------------------------>
                  ^                    Arithmetic Intensity
              Ridge point              (FLOPs / byte)
```

- **X axis = Arithmetic Intensity (AI)** -- how many FLOPs you do per byte of data loaded from memory.
- **Y axis = achievable performance** (GFLOPs/s).
- **The diagonal line** is the bandwidth limit (slope = peak bandwidth).
- **The horizontal line** is the compute limit (peak FLOPs).
- **The ridge point** is where the two meet. Left of it = memory-bound. Right of it = compute-bound.

---

## 2. Compute-Bound vs Memory-Bound

| | Compute-bound | Memory-bound |
|---|---|---|
| **Bottleneck** | Arithmetic units (FLOPs) | HBM bandwidth (GB/s) |
| **Symptom** | All SMs busy, cache fast enough | SMs wait for data |
| **Arithmetic intensity** | High (above ridge) | Low (below ridge) |
| **Speedup via faster GPU?** | Yes | No (bandwidth is the limit) |
| **Speedup via smaller data?** | No | Yes (quantization, caching) |
| **LLM example** | Prefill with long sequences | Decode, one token at a time |

**Formula**

Achievable performance = min(peak compute, AI * peak bandwidth)

**Intuitively**

> Compute-bound: the chef (SMs) is the bottleneck.
> Memory-bound: the waiter (HBM bandwidth) is the bottleneck.

---

## 3. RTX 4090 Laptop Spec

| Metric | Value |
|--------|-------|
| Peak FP32 compute | 33 TFLOPs (33,000 GFLOPs/s) |
| HBM bandwidth (GDDR6X) | 576 GB/s |
| **Ridge point** | **33,000 / 576 = 57.3 FLOPs/byte** |

Any workload with AI < 57 is memory-bound on this GPU. AI > 57 is compute-bound.

---

## 4. GPT-2 (124M) Arithmetic Intensity Analysis

### 4.1 Decode step (generate 1 token)

- FLOPs per step: 2 * 124M = **248 M FLOPs** (each param contributes 1 MAC = 2 FLOPs)
- Bytes loaded per step: 124M * 4 = **496 MB** (reload all weights in FP32)
- **AI = 248M / 496M = 0.50 FLOPs/byte**

Ridge point is 57 -- decode sits **115x below the ridge**. Deeply memory-bound.

Roofline ceiling at AI=0.5:
- performance = 0.5 * 576 GB/s = **288 GFLOPs/s**
- This is only **0.87% of peak compute**. You could 100x the compute and decode would barely speed up.

Theoretical decode latency (just the data movement):
- 496 MB / 576 GB/s = **0.86 ms per token**

### 4.2 Prefill step (L tokens in parallel)

- FLOPs: 2 * 124M * L
- Bytes loaded: 496 MB (weights loaded once, reused across L)
- **AI = 2 * 124M * L / 496M = L/2**

So AI grows linearly with sequence length:

| L | AI | Ridge compare | Bound | Roofline ceiling |
|---|:---:|:---:|:---:|---|
| 32 | 16 | < 57 | memory | 9,216 GFLOPs/s |
| 64 | 32 | < 57 | memory | 18,432 GFLOPs/s |
| 128 | 64 | > 57 | **compute** | 33,000 GFLOPs/s |
| 256 | 128 | > 57 | **compute** | 33,000 GFLOPs/s |

Prefill only enters the compute-bound region around L=128.

---

## 5. Explaining W2 Profiling Data via Roofline

W2 measured on RTX 4090 Laptop, FP32, GPT-2 124M:

| prompt_len | prefill ms | decode ms | decode% | prefill GFLOPs/s | decode GFLOPs/s |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 2.51 | 134.50 | 98.2% | 3,167 | 120 |
| 64 | 2.62 | 131.47 | 98.0% | 6,049 | 121 |
| 128 | 2.85 | 132.70 | 97.9% | 11,123 | 120 |
| 256 | 4.58 | 132.72 | 96.7% | 13,875 | 120 |

### 5.1 Decode sits at ~120 GFLOPs/s regardless of prompt length

- Roofline ceiling at AI=0.5: **288 GFLOPs/s**
- Measured: **120 GFLOPs/s = 41.5% of ceiling**
- Physical lower bound on time per token: 0.86 ms
- Measured: 2.07 ms/token
- The gap (41.5% vs 100%) comes from kernel launch overhead, Python/CUDA sync, KV cache reads, and tensor core idle time

**Why decode does not scale with prompt length:** at bs=1, each decode step still needs the full 496 MB weight reload. Longer prompt does not change that. The ceiling stays at 288 GFLOPs/s and the measured stays at 120.

### 5.2 Prefill throughput grows with prompt length

| L | measured GFLOPs/s | roofline ceiling | efficiency |
|:---:|:---:|:---:|:---:|
| 32 | 3,167 | 9,216 | 34% |
| 64 | 6,049 | 18,432 | 33% |
| 128 | 11,123 | 33,000 | 34% |
| 256 | 13,875 | 33,000 | 42% |

Prefill efficiency is consistent around 33-42% of the ceiling.

L=32 and L=64 are memory-bound (AI below ridge). L=128 and L=256 are compute-bound. The efficiency stays similar because real workloads rarely hit 100% of theory -- kernel launch overhead, memory divergence, and less-than-perfect tensor core mapping eat the rest.

### 5.3 Why decode_ratio is 96-98% total time

Even at the longest prompt (L=256), prefill takes 4.58 ms while decode takes 132.7 ms. Ratio: **decode / total = 96.7%**.

This is because:
- Prefill: 1 forward pass, **high arithmetic intensity**, compute-bound -> fast.
- Decode: 64 sequential forward passes, **tiny arithmetic intensity**, memory-bound -> slow.

The 64:1 forward pass ratio combined with the 100x lower per-pass efficiency makes decode dominate.

---

## 6. Why Decode GPU Utilization is "Low"

Two different meanings of "utilization" confuse people here:

### 6.1 nvidia-smi sm% (misleading for bandwidth-bound workloads)

`nvidia-smi dmon` reported **sm% = 64.6%** during sustained decode. This is **NOT** the fraction of compute capacity being used. It is the fraction of time at least one warp was active on an SM.

For memory-bound kernels, warps are "active but stalled on memory" -- sm% looks high even though actual FLOPs utilization is near zero.

### 6.2 True compute utilization (from roofline)

True FLOPs utilization = measured / peak_compute = 120 / 33,000 = **0.36%**

This is the honest answer to "how much of the GPU is doing useful work during decode". It is **vanishingly small**. The SMs are not doing arithmetic -- they are waiting for the next slice of weight data.

### 6.3 How to actually measure it

Use `nsys profile` or `ncu` (Nsight Compute) to get:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` -- true compute utilization
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` -- true bandwidth utilization

Prediction:
- sm throughput during decode: < 5%
- dram throughput during decode: > 60%

This would cleanly confirm memory-bound.

---

## 7. How to Move the Decode Point Up/Right on the Roofline

Optimization is literally "move the red dot on the roofline chart":

**Move right (increase AI)**
- **KV Cache**: skip recomputing past K/V. Each step only computes the new row. AI goes from 0.5 to roughly 2-5.
- **Batching**: load weights once, use them for N requests. AI scales by N.
- **Speculative decoding**: one target-model forward pass verifies K tokens. AI scales by K.

**Move up (raise the ceiling)**
- **Quantization**: fewer bytes per weight. FP32 -> FP16 = 2x bandwidth headroom. FP32 -> INT8 = 4x. FP32 -> INT4 = 8x. The slope of the bandwidth line becomes steeper relative to data volume.
- **Flash Attention**: fuses attention kernel so intermediates stay in SRAM instead of going back to HBM.

Every well-known LLM inference optimization is one of these two moves.

---

## 8. Key Numbers to Remember

| Number | Meaning |
|:---:|---|
| **57 FLOPs/byte** | Ridge point on RTX 4090 Laptop (FP32) |
| **0.5 FLOPs/byte** | GPT-2 decode arithmetic intensity |
| **115x** | How far decode sits below the ridge |
| **0.87%** | Decode ceiling as fraction of peak compute |
| **0.36%** | Decode measured as fraction of peak compute |
| **0.86 ms** | Physical minimum time per decode token (just weight reload) |
| **2.07 ms** | Measured time per decode token |

---

## 9. What I Can Now Explain in One Sentence

> Decode is slow because GPT-2 loads its entire 500 MB of weights from HBM for every single generated token, and with only one token of work per load the arithmetic intensity (0.5 FLOPs/byte) is 115x below where RTX 4090 stops being memory-bound -- so 99% of the SMs sit idle waiting for HBM, and the ceiling for decode performance is 0.87% of the GPU's peak compute.

That sentence is the thesis of everything I will study for the rest of April and May.

---

*Sources: W2 profiling data ([gpt2_profiling_results.json](../W2/gpt2_profiling_results.json)), roofline computation ([../W2/roofline_analysis.py](../W2/roofline_analysis.py)), RTX 4090 Laptop spec sheet.*
