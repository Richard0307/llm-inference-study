# Using Roofline to Interpret W2 Profiling Data

**Purpose:** take the raw profiling numbers from W2 and explain every observation with one framework: the Roofline Model.

---

## 1. Hardware setup

| Quantity | Value |
|---|---|
| GPU | RTX 4090 Laptop |
| Peak FP32 compute | 33 TFLOPs |
| HBM bandwidth | 576 GB/s |
| Ridge point | **57.3 FLOPs/byte** |

Any workload below AI = 57 is memory-bound. Above = compute-bound.

---

## 2. W2 raw measurements

From [W2/gpt2_profiling_results.json](../W2/gpt2_profiling_results.json):

| prompt_len | prefill (ms) | decode (ms) | decode% | prefill tok/s | decode tok/s |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 2.51 | 134.50 | 98.2% | 12,771 | 476 |
| 64 | 2.62 | 131.47 | 98.0% | 24,390 | 487 |
| 128 | 2.85 | 132.70 | 97.9% | 44,850 | 482 |
| 256 | 4.58 | 132.72 | 96.7% | 55,947 | 482 |

Five things that need explaining:

1. Why is decode ~97% of total time?
2. Why is decode per-token time ~2 ms regardless of prompt length?
3. Why does prefill throughput (tok/s) scale roughly linearly with prompt length?
4. Why is prefill 50-100x faster than decode in tokens/sec?
5. Why does decode hit a ceiling around 482 tok/s and stop?

All five answers come from the same roofline picture.

---

## 3. Compute arithmetic intensity for every data point

### 3.1 Decode (any prompt length)

```
FLOPs per token        = 2 * 124M             = 248 M
Bytes loaded per token = 124M * 4 (FP32)      = 496 MB
AI                     = 248M / 496M          = 0.50 FLOPs/byte
```

0.50 vs ridge 57 -> 115x below ridge -> deeply memory-bound.

### 3.2 Prefill (per prompt length)

```
FLOPs  = 2 * 124M * L
Bytes  = 496 MB  (weights loaded once, reused across all L tokens)
AI     = L / 2
```

| L | AI | vs Ridge (57) | Bound | Roofline ceiling |
|:---:|:---:|:---:|:---:|:---:|
| 32 | 16 | below | memory | 9.2 TFLOPs |
| 64 | 32 | below | memory | 18.4 TFLOPs |
| 128 | 64 | above | **compute** | 33 TFLOPs |
| 256 | 128 | above | **compute** | 33 TFLOPs |

Prefill crosses into compute-bound territory around L=128.

---

## 4. Answering the five questions

### Q1: Why is decode 97% of total time?

**Short answer:** prefill has high AI so it finishes fast; decode has low AI AND happens 64 times, so it dominates.

**With numbers (L=256, gen=64):**

- Prefill: 1 forward pass at AI=128 (compute-bound, near peak)
  - Measured: 4.58 ms
- Decode: 64 forward passes at AI=0.5 (memory-bound, deep in the valley)
  - Measured: 132.72 ms = 64 * 2.07 ms/token

Decode ratio = 132.72 / (132.72 + 4.58) = **96.7%** ✓

### Q2: Why is decode per-token time ~2 ms regardless of prompt length?

Because the dominant cost is loading all 496 MB of weights, and that happens **once per token** no matter how long the prompt is.

Theoretical lower bound: 496 MB / 576 GB/s = **0.86 ms**
Measured: **2.07 ms**

The gap is kernel launch overhead, Python/CUDA sync, KV cache reads, and imperfect memory access patterns. All of those are roughly constant per step, so per-token time is flat across prompt lengths.

### Q3: Why does prefill throughput (tok/s) scale with prompt length?

Because longer prompts increase AI, which moves prefill up the bandwidth slope until it hits the compute ceiling.

| L | AI | Roofline ceiling | Measured GFLOPs/s | Measured tok/s |
|:---:|:---:|:---:|:---:|:---:|
| 32 | 16 | 9,216 | 3,167 | 12,771 |
| 64 | 32 | 18,432 | 6,049 | 24,390 |
| 128 | 64 | 33,000 | 11,123 | 44,850 |
| 256 | 128 | 33,000 | 13,875 | 55,947 |

From L=32 to L=64, AI doubles -> ceiling doubles -> measured throughput doubles. Clean bandwidth slope behavior.

From L=128 onward the ceiling flattens at 33 TFLOPs, but measured throughput keeps growing slightly because efficiency improves with larger GEMM tiles.

### Q4: Why is prefill 50-100x faster than decode in tokens/sec?

Prefill at L=256: 55,947 tok/s
Decode: 482 tok/s
Ratio: **116x**

Because:
- Prefill amortizes one weight load across 256 tokens -> AI=128
- Decode amortizes one weight load across 1 token -> AI=0.5
- AI ratio: 256x
- Converted to measured performance, the ratio collapses to ~116x because prefill efficiency (42% of 33 TFLOPs) is lower than decode efficiency (41.5% of 288 GFLOPs/s) relative to each side's ceiling, but the huge AI gap dominates.

### Q5: Why does decode hit ~482 tok/s and not go higher?

The roofline ceiling for decode is 288 GFLOPs/s, which translates to **1,161 tok/s theoretical max** (288e9 / 248e6). We measured 482 -> **41.5% efficiency**. The rest is eaten by kernel launch + sync + KV cache fetch overhead.

To go faster than 482 tok/s you have to either:
- raise the ceiling (reduce bytes per weight -> quantization)
- raise the AI (amortize the weight load -> KV cache, batching, speculative decoding)

No amount of CPU/Python tuning will break through that 1,161 tok/s theoretical cap on this GPU in FP32.

---

## 5. GPU utilization: the nvidia-smi trap

When I ran `nvidia-smi dmon` during sustained decode I saw:

```
sm%    = 64.6%
mem%   = 41.9%
power  = 93 W
```

This initially looked like decode was "using 64% of the GPU", which would contradict memory-bound.

**It does not contradict. It is just a different metric.**

`sm%` from `nvidia-smi dmon` = fraction of time at least one warp was resident on an SM. Memory-stalled warps still count as "resident". So for a bandwidth-bound kernel sm% shows up high even though the SMs are not doing arithmetic.

**The honest number** is true FLOPs utilization:

```
FLOPs utilization = measured performance / peak compute
                  = 120 GFLOPs/s / 33,000 GFLOPs/s
                  = 0.36%
```

The GPU's arithmetic units are running at 0.36% of capacity during decode.

To see this directly you would use `nsys profile` / `ncu` and read:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`

Expected: sm throughput < 5%, dram throughput > 60%. That is the honest roofline picture.

**Lesson:** `nvidia-smi sm%` is a weak signal for LLM inference. It can say "the GPU is busy" even when 99% of the arithmetic units are waiting on HBM. Always cross-check with arithmetic-intensity reasoning or ncu metrics.

---

## 6. Summary in one table

| Observation | Roofline explanation |
|---|---|
| Decode is 97% of total time | 64 sequential forward passes at AI=0.5, each bounded by HBM bandwidth |
| Per-token decode ~2 ms regardless of prompt | Weight reload cost (0.86 ms) + overhead, independent of prompt length |
| Prefill throughput grows with L | AI = L/2, rising on the bandwidth slope until it hits compute ceiling at L>=128 |
| Prefill 116x faster than decode | AI ratio 256x dominates efficiency differences |
| Decode capped at ~482 tok/s | Roofline ceiling = 1,161 tok/s, ~41.5% efficiency -> 482 |
| nvidia-smi sm% = 64.6% during decode | Misleading metric; true FLOPs util = 0.36% |

---

## 7. Takeaway

**One sentence:** W2 profiling data is entirely explained by placing GPT-2 decode at AI=0.5 and GPT-2 prefill at AI=L/2 on the RTX 4090 Laptop roofline chart; everything else follows from the arithmetic of min(peak compute, AI * peak bandwidth).

Completion criteria check:

- [x] Can explain compute-bound vs memory-bound (Section 2 of roofline_notes.md)
- [x] Can use Roofline to explain why decode GPU utilization is low (Section 5 here + Section 6 of roofline_notes.md)

---

*Written: 2026-04-14*
