# W3 Hypothesis Document

**Week:** 2026-04-13 ~ 2026-04-19
**Theme:** GPU Memory Hierarchy + Roofline Model + Compute vs Memory Bound
**Hardware:** RTX 4090 Laptop GPU

---

## 1. Primary Hypothesis

**H1: At batch_size=1, GPU SM utilization stays below 30% during GPT-2 decode phase.**

### Rationale

W2 profiling showed decode phase dominates inference time (97%+). The bottleneck is not compute but **HBM bandwidth**: each decode step loads the full ~500MB model weights from HBM to SRAM, while only doing GEMV (vector-matrix multiply) on a single token.

- **GEMV arithmetic intensity** (FLOPs / bytes loaded) is extremely low
- **RTX 4090 Laptop** peak FP32: ~33 TFLOPs, HBM bandwidth: ~576 GB/s
- **Roofline prediction**: at bs=1, we are deep in the memory-bound region
- Therefore: SM compute units should be mostly idle waiting for memory

### Verification Plan

```bash
# Terminal 1: run decode loop
python W2/gpt2_profiling.py  # or a dedicated decode-only loop

# Terminal 2: monitor GPU utilization
nvidia-smi dmon -s pucmt -i 0 -c 60
#           ^        ^ power, util, clocks, mem, temp
#           |        poll every 1s
```

Record:
- `sm` column (SM utilization %)
- `mem` column (memory controller utilization %)
- `pclk` (GPU clock) — does it downclock when idle-waiting?

### Expected Outcome

| Metric | Prediction |
|--------|:---:|
| SM util (sm%) | **< 30%** (likely 15-25%) |
| Memory util (mem%) | **> 60%** (bandwidth-bound) |
| Power | Well below TDP |

### What If Hypothesis Is Wrong?

- If SM util > 30% → decode might be compute-limited for small models on this GPU. Investigate kernel launch overhead vs actual compute.
- If mem util < 60% → maybe kernel launch overhead or Python/CUDA sync is the real bottleneck, not HBM bandwidth.

---

## 2. Secondary Hypotheses

**H2: Increasing batch_size improves GPU utilization but hits a ceiling.**

- At bs=1: memory-bound (bad utilization)
- At bs=8/16/32: moving toward compute-bound (better utilization)
- At bs=64+: might hit OOM or plateau (KV Cache memory grows linearly)

Test: sweep batch_size in {1, 4, 8, 16, 32}, measure SM util and throughput.

**H3: Prefill phase has much higher SM utilization than decode.**

- Prefill = GEMM (matrix-matrix) = compute-bound
- Decode = GEMV (matrix-vector) = memory-bound
- Prediction: prefill SM util > 70%, decode SM util < 30%

Test: run prefill-only and decode-only separately with `nvidia-smi dmon` in parallel.

---

## 3. Tools Prepared for W3

| Tool | Purpose | Command |
|------|---------|---------|
| `nvidia-smi dmon` | Poll GPU metrics per second | `nvidia-smi dmon -s pucmt -i 0` |
| `nvidia-smi pmon` | Per-process GPU metrics | `nvidia-smi pmon -i 0 -c 30` |
| `torch.cuda.Event` | Precise CUDA timing | used in W2 |
| `torch.profiler` | Chrome trace export | used in W2 |
| `nvprof` / `nsys` | Low-level kernel profiling | for deeper analysis |
| Roofline chart | Visualize compute vs memory bound | to be plotted in W3 |

### Quick nvidia-smi dmon reference

```
-s: select metrics
   p = power (W)
   u = util % (sm, mem, enc, dec)
   c = clocks (mclk, pclk)
   m = memory (fb used/free)
   t = temp (gtemp, mtemp)
-i: GPU index (0 for first GPU)
-c: count of samples (omit for continuous)
-d: delay between samples (seconds, default 1)
```

---

## 4. W3 Daily Plan Preview

| Day | Task |
|-----|------|
| Mon 4/13 | Read GPU architecture paper, understand HBM hierarchy |
| Tue 4/14 | Implement Roofline model plotter, identify compute vs memory bound regions |
| Wed 4/15 | Run GPT-2 decode with `nvidia-smi dmon` monitoring — verify H1 |
| Thu 4/16 | Batch size sweep — verify H2 |
| Fri 4/17 | Prefill vs Decode SM util comparison — verify H3 |
| Sat 4/18 | Plot Roofline chart with real measurement points |
| Sun 4/19 | W3 memo + W4 hypothesis (KV Cache acceleration ratio) |

---

## 5. Key Questions to Answer by End of W3

1. **What fraction of peak GPU performance does GPT-2 decode actually achieve?**
2. **Where does GPT-2 sit on the Roofline chart — compute side or memory side?**
3. **How does batch size shift the operating point?**
4. **What is the theoretical decode speed limit on this GPU?**
5. **How much headroom is there for KV Cache optimization (W4)?**

---

## 6. Link to W2 Data

W3 hypotheses are built on W2 profiling findings:

- [W2 decoder profiling memo](decoder_profiling_memo.md)
- [GPT-2 profiling results](gpt2_profiling_results.json)
- Key W2 finding: decode ~97% of total time, 482 tok/s steady state regardless of prompt length

> W3 goal: explain *why* decode is stuck at 482 tok/s, in terms of GPU hardware limits.

---

*Written: 2026-04-11*
