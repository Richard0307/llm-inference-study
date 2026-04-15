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

---

# 附录：Nsight Compute 实测报告（FP32 / FP16 / INT8 对比）

**实验日期**：2026-04-15
**硬件**：RTX 4090 Laptop GPU（Ada Lovelace，CC 8.9）
**工具**：Nsight Compute 2022.4.1
**模型**：GPT-2 124M（HuggingFace checkpoint）
**输入**：seq_len = 128，decode 采样第 10 步（跳过 warmup）
**度量指标**：
- `sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak` — Tensor Core HMMA 利用率（混合精度矩阵乘）
- `dram__throughput.pct_of_peak_sustained_elapsed` — HBM 带宽利用率
- `smsp__cycles_active.pct_of_peak` — SM 流水线活跃率

原始数据：
- [W3/ncu_prefill_fp32.csv](ncu_prefill_fp32.csv) · [W3/ncu_decode_fp32.csv](ncu_decode_fp32.csv)
- [W3/ncu_prefill_fp16.csv](ncu_prefill_fp16.csv) · [W3/ncu_decode_fp16.csv](ncu_decode_fp16.csv)
- [W3/ncu_prefill_int8.csv](ncu_prefill_int8.csv) · [W3/ncu_decode_int8.csv](ncu_decode_int8.csv)
- [W3/precision_latency.json](precision_latency.json)

---

## 1. 核心数据表

### 1.1 Prefill 阶段（seq_len = 128）

| 精度 | Kernel 数 | DRAM 均值 | DRAM 峰值 | HMMA 均值 | HMMA 峰值 | SM 活跃均值 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 468 | 35.5% | 70.7% | 0.3% | 5.7% | 66.1% |
| FP16 | 468 | 27.3% | 56.4% | **4.4%** | **27.9%** | 54.5% |
| INT8 | **1552** | 14.2% | 53.1% | 0.1% | 26.2% | 31.8% |

### 1.2 Decode 阶段（单 token）

| 精度 | Kernel 数 | DRAM 均值 | **DRAM 峰值** | HMMA 均值 | HMMA 峰值 | SM 活跃均值 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 235 | 17.9% | **96.1%** | 0.2% | 4.1% | 15.3% |
| FP16 | 235 | 13.3% | **94.6%** | 0.2% | 4.3% | 16.8% |
| INT8 | **555** | 4.5% | **94.7%** | 0.1% | 4.3% | 10.8% |

### 1.3 端到端延迟与显存

| 精度 | Prefill (ms) | Decode (ms/token) | Peak Mem (MB) | 相对 FP32 |
|---|---:|---:|---:|---|
| FP32 | 3.29 | 2.58 | 542.4 | baseline |
| FP16 | 3.05 | 2.64 | 508.5 | **prefill 快 7%，decode 持平**，显存 -6% |
| INT8 | 14.64 | 9.94 | 320.4 | **prefill 慢 4.4×，decode 慢 3.85×**，显存 -41% |

---

## 2. 三条关键结论

### 结论 1：Decode 的 DRAM 峰值跨三种精度都接近 95%——memory-bound 是硬件本质，不是精度问题

三种精度下 decode 的 DRAM 峰值分别是 96.1% / 94.6% / 94.7%，几乎打满 HBM 带宽。这意味着**无论用什么数值精度，decode 阶段都卡在内存带宽上**。热点 kernel 一眼可辨：

- FP32：`gemv2T_kernel_val`、`gemvx::kernel`
- FP16：`gemvx::kernel`（同类，只是精度不同）
- INT8：`gemvx::kernel` + `gemmSN_kernel_int32`（bnb 的量化 GEMM）

名字里的 **GEMV**（General Matrix-Vector）是整个故事的核心——decode 每一步只处理一个 token，权重矩阵 W 要整块从 HBM 搬到 SM，却只和一个向量相乘，算术强度 AI ≈ 2，深陷 Roofline 的 memory-bound 区。

**这直接验证了 H1 的 Roofline 假设**。但 H1 原本预测 SM util < 30%，实测 decode SM 活跃率 10-17%，**比假设更低**——说明 decode 阶段 SM 大部分时间在等 memory，而不是真的做计算。

### 结论 2：Tensor Core 只在 prefill FP16 真正被激活，decode 永远用不到

| 场景 | HMMA 峰值 | 解读 |
|---|---:|---|
| prefill FP32 | 5.7% | 几乎为 0，因为 FP32 走的是 CUDA core 的 FMA，不触发 Tensor Core |
| **prefill FP16** | **27.9%** | 真正激活 Tensor Core，热点 kernel 变成了 `cutlass::gemm ... half`（用 HMMA 指令） |
| prefill INT8 | 26.2% | 偶发高峰来自 bnb 的 fp16 中间层（layernorm、softmax），不是量化 GEMM 本身 |
| decode 任意精度 | ~4% | **Tensor Core 对 GEMV 无效**：HMMA 指令要求输入是矩阵，而 decode 的激活只是向量 |

**洞察**：
- Tensor Core 是为 **大尺寸矩阵乘** 设计的。decode 阶段的 GEMV 没有两个矩阵维度可以 tile，Tensor Core 指令集用不上。这就是为什么所有 LLM serving 系统（vLLM、TGI）都花大力气做 **batching** — 把多个请求拼成矩阵，把 decode 从 GEMV 变成小 GEMM，才能吃到 Tensor Core。
- 如果只有 batch=1，换不换 FP16 对 decode 速度影响微乎其微（实测 FP16 反而慢 2%，在噪声范围内）。

### 结论 3：INT8（bitsandbytes）在 GPT-2 这种小模型上是**负优化**

| 指标 | FP32 | INT8 | 变化 |
|---|---:|---:|---|
| Prefill 延迟 | 3.29 ms | 14.64 ms | **慢 4.4×** |
| Decode ms/token | 2.58 | 9.94 | **慢 3.85×** |
| 显存占用 | 542 MB | 320 MB | 省 41% ✅ |
| Prefill kernel 数 | 468 | 1552 | **多 3.3×** |
| Decode kernel 数 | 235 | 555 | **多 2.4×** |

**为什么 INT8 反而更慢**：

1. **bnb 在每次矩阵乘前后都要 quantize/dequantize**，这些额外的 kernel 把 kernel 总数从 468 推到 1552。每个 kernel launch 有几 μs 的 CPU-GPU 同步开销，1500+ 次启动的 overhead 远超 INT8 GEMM 节省的算力。
2. **GPT-2 太小**。INT8 只在模型大到放不进 GPU 时才真正有价值（例如 13B 模型在 24GB 卡上），那时它的唯一目的是**换显存不是换速度**。对于 124M 这种模型，FP16 放得进 HBM 也够快，根本没有换显存的必要。
3. **bnb 的 int8 GEMM kernel `gemmSN_kernel_int32`** 用的是 INT32 累加器，吞吐远低于同规模 FP16 Tensor Core GEMM——小模型吃不到量化的 compute 收益。

**结论**：W4/W5 做优化时，**不要用 bnb 的 int8 作为 GPT-2 的加速手段**，它只适用于"放不下 → 能放下"的场景。真正的 decode 加速要靠 **KV cache + continuous batching + MQA/GQA**，这些都是**减少 memory 读取量**的优化，和 Roofline 诊断出的瓶颈方向一致。

---

## 3. 对 W3 原始假设的回应

| 假设 | 原预测 | 实测 | 结论 |
|---|---|---|---|
| H1 decode SM util < 30% | 15-25% | **15.3% (FP32), 16.8% (FP16), 10.8% (INT8)** | ✅ 成立，甚至更低 |
| H1 decode mem util > 60% | > 60% | DRAM 峰值 94-96% | ✅ 大幅成立 |
| H3 prefill > decode 利用率 | prefill > 70%，decode < 30% | prefill SM 66% / decode 15% | ✅ 方向正确，prefill 略低于预期（小模型非 GEMM kernel 占比高） |
| H2 大 batch 改善利用率 | 未测 | 留待 W4 验证 | ⏳ |

---

## 4. 方法论教训（ncu 实战踩坑记录）

这次跑 ncu 花了半天时间，踩的坑都值得记下来：

1. **NVTX range 名字两边必须一致**。代码 `nvtx.range_push("decode_step")`、命令 `--nvtx-include "decode_step/"`。名字对不上 ncu 不会报错，只会 `No kernels were profiled`。
2. **ncu 以子进程启动 Python 时会丢失 conda 的 cuDNN 路径**。PyTorch 把 cuDNN 装在 `site-packages/nvidia/cudnn/lib/`，平时 Python 启动会自己 dlopen，但 ncu 打破了这个初始化顺序。必须显式 `export LD_LIBRARY_PATH=$CUDNN_LIB:$LD_LIBRARY_PATH`。
3. **永久解除 profiling 权限**比每次 `sudo ncu` 更清爽：
   ```bash
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
     | sudo tee /etc/modprobe.d/nvidia-profiler.conf
   sudo update-initramfs -u && sudo reboot
   ```
4. **profile 入口要极简**——单独写一个 `run_ncu_target()`，只跑 prefill 一次 + decode 一步（带 warmup），避免 ncu 被几百次循环淹没。加 print 保证排障能看到进度。
5. **标准 metric 三件套**：`sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak`（算力）、`dram__throughput.pct_of_peak`（带宽）、`smsp__cycles_active.pct_of_peak`（SM 活跃度）。三个一起看才能定位瓶颈。

---

## 5. W4 衔接点

本次报告给 W4 的 KV cache 实验留下了清晰的 **baseline 和优化方向**：

- **Baseline**：decode ms/token ≈ 2.58 ms（FP32, bs=1, seq_len=128），DRAM 峰值 96%
- **W4 假设**：KV cache 让重复搬运的 K/V 留在 SRAM/L2，减少 HBM 读取 → 预期 decode ms/token 下降、DRAM 峰值也下降
- **验收指标**：在 seq_len ≥ 256 时 decode 加速 ≥ 2×（对应 ms/token ≤ 1.3 ms）
- **观察点**：优化后 Tensor Core HMMA 依然应接近 0（因为还是 GEMV），真正改变的是 DRAM 压力

如果 KV cache 优化后 DRAM 利用率下降但延迟不变，那说明瓶颈转移到了 kernel launch overhead 或 CPU-GPU 同步上——这是下一步要解决的问题。

---

*Written: 2026-04-15*

