# W2 Mini Memo — Decoder Block + GPT-2 Profiling

**Week:** 2026-04-07 ~ 2026-04-11
**Focus:** GPT-style Decoder Block implementation + GPT-2 inference profiling
**Hypothesis:** Decode phase accounts for >80% of total inference time

---

## 1. Deliverables

| File | Purpose |
|------|---------|
| [LayerNorm.py](LayerNorm.py) | Hand-written LayerNorm, matches `nn.LayerNorm` within 5e-7 |
| [PreNorm.py](PreNorm.py) | Generic Pre-Norm wrapper `x + fn(norm(x))` |
| [DecoderBlock.py](DecoderBlock.py) | CausalSelfAttention + FFN + DecoderBlock, 4-layer stack verified |
| [decoder_block_diagram.py](decoder_block_diagram.py) | Matplotlib script to generate data flow diagram |
| [decoder_block_dataflow.png](decoder_block_dataflow.png) | GPT-style Decoder Block visualization |
| [gpt2_profiling.py](gpt2_profiling.py) | Prefill/Decode latency benchmark + Chrome trace |
| [gpt2_profiling_results.json](gpt2_profiling_results.json) | Raw profiling data |
| [gpt2_profiling_trace.json](gpt2_profiling_trace.json) | Chrome trace for torch.profiler |
| [blog_decoder_only.md](blog_decoder_only.md) | Zhihu-style blog post summarizing findings |

---

## 2. Decoder Block Implementation

### 2.1 Architecture (Pre-Norm)

```
x -> LayerNorm -> CausalSelfAttention -> +x -> LayerNorm -> FFN -> +x -> out
```

Key design decisions:

- **Pre-Norm over Post-Norm**: LayerNorm before the sublayer, more stable for deep stacks (GPT-2/GPT-3 standard)
- **Causal mask**: `torch.triu(..., diagonal=1).bool()` prevents attending to future tokens
- **FFN expansion ratio 4x**: standard GPT-2 choice (512 -> 2048 -> 512)
- **GELU activation**: smoother than ReLU, standard in modern LLMs

### 2.2 Verification

| Check | Result |
|-------|--------|
| LayerNorm vs `nn.LayerNorm` | max diff 4.77e-07 |
| PreNorm vs manual `x + fn(norm(x))` | max diff 7.15e-07 |
| 4-layer Decoder forward shape | `[2,10,512]` -> `[2,10,512]` preserved |
| Causal mask (pos 0 not affected by pos 1+) | diff 7.15e-07 -> confirmed |

### 2.3 Std growth across layers (stability check)

| Layer | mean | std |
|:---:|:---:|:---:|
| Block 1 | 0.0081 | 1.0363 |
| Block 2 | 0.0043 | 1.0819 |
| Block 3 | -0.0063 | 1.1361 |
| Block 4 | -0.0060 | 1.1805 |

Std grows ~0.03 per block -- slow, controlled. Pre-Norm keeps numerics stable for deep stacks.

---

## 3. GPT-2 Profiling Results

### 3.1 Setup

- **Model:** GPT-2 (124M params)
- **Hardware:** RTX 4090 Laptop GPU
- **Precision:** FP32
- **Generation length:** 64 tokens
- **Prompt lengths:** 32 / 64 / 128 / 256 tokens
- **Warmup:** 3 runs, **Measure:** 5 runs avg

### 3.2 Prefill vs Decode Latency

| prompt_len | prefill(ms) | decode(ms) | total(ms) | **decode%** | prefill tok/s | decode tok/s |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 2.51 | 134.50 | 137.01 | **98.2%** | 12,771 | 476 |
| 64 | 2.62 | 131.47 | 134.10 | **98.0%** | 24,390 | 487 |
| 128 | 2.85 | 132.70 | 135.55 | **97.9%** | 44,850 | 482 |
| 256 | 4.58 | 132.72 | 137.30 | **96.7%** | 55,947 | 482 |

**Peak GPU memory:** 588.5 MB

### 3.3 CUDA Operator Breakdown

| Operator | CUDA time % | Phase |
|------|:---:|------|
| `aten::addmm` (GEMM) | 47.5% | Prefill |
| `gemvx` (GEMV) | 43.1% | Decode |
| `scaled_dot_product_attention` | 14.0% | Both |
| `native_layer_norm` | 4.7% | Both |

GEMM + GEMV account for 90.6% of GPU time -- direct evidence of the two distinct compute patterns.

---

## 4. Hypothesis Verification

**Hypothesis:** Decode phase accounts for >80% of total inference time

**Result:** CONFIRMED across all prompt lengths (96.7% ~ 98.2%)

### Why decode dominates despite being "memory-bound":

1. **Number of forward passes**: prefill = 1 pass (all tokens in parallel), decode = 64 passes (sequential)
2. **GPU 4090 is too strong for GPT-2**: prefill GEMM finishes in ~3ms, so prefill cost is negligible
3. **Decode throughput ceiling ~482 tok/s**: bottlenecked by HBM bandwidth (must load full 500MB weights each step)
4. **Prefill throughput scales**: 12k -> 55k tok/s as prompt grows, confirming compute-bound nature

### Key insight

> Prefill is compute-bound. Decode is memory-bound.
> But on a powerful GPU with a small model, decode DOMINATES because of sheer number of sequential steps.

Expected shift at scale: LLaMA-70B on A100 would show prefill/decode ratio closer to 20/80 or 30/70 (prefill catches up as GEMM gets larger).

---

## 5. Why Decoder-only Wins (Takeaway)

1. **KV Cache works only on unidirectional attention** -- Decoder-only can cache past K/V, Encoder cannot
2. **Training objective unified** -- "predict next token" handles all tasks, no paired data needed
3. **Scaling Law validated** -- GPT-2 -> GPT-3 -> GPT-4, Decoder-only is the only architecture proven to scale
4. **Decode is the bottleneck** -- all inference optimizations (KV Cache, Flash Attention, quantization, speculative decoding, vLLM) target decode phase

---

## 6. What's Next (W3 Preview)

Based on this week's findings:

- **W3 hypothesis:** GPU utilization <30% at batch=1 (decode memory-bound)
- **W3 focus:** GPU memory hierarchy, Roofline Model, compute vs memory bound analysis
- **W4 target:** Implement KV Cache -> target >2x decode speedup at seq_len>256

---

## 7. Lessons Learned

- **Start from inference, not training**: running `model.generate()` on day 1 immediately reveals the decode bottleneck. No amount of textbook reading substitutes for real profiling data.
- **Causal mask is non-negotiable**: forgot it initially, then realized self-attention without mask = bidirectional = Encoder. Single `torch.triu` line separates GPT from BERT.
- **Pre-Norm > Post-Norm for deep stacks**: verified by std stability across 4 layers. This is why modern LLMs all use Pre-Norm.
- **HuggingFace download via ModelScope mirror**: when network to HF is slow, `modelscope.snapshot_download('AI-ModelScope/gpt2')` is the escape hatch.

---

*Generated on 2026-04-11*
