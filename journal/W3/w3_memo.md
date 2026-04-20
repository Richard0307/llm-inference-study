# W3 收尾 Memo：GPU 内存层级 + Roofline + 精度对比

**周次**：W3（2026-04-13 ~ 2026-04-19）
**硬件**：RTX 4090 Laptop GPU（Ada Lovelace，CC 8.9，16 GB GDDR6X）
**模型**：GPT-2 124M（HuggingFace checkpoint）
**状态**：病后恢复版收尾，锁定 W3 工程主体产出，不新增实验。

---

## 0. TL;DR

一句话：**GPT-2 在 RTX 4090 Laptop 上 decode 被 HBM 带宽锁死，真实算力利用率 < 1%；换精度（FP16/INT8）救不了这个问题——必须靠 KV cache / batching 提升算术强度。**

三个数字：
- **57.3 FLOPs/byte** — 4090 Laptop 的 Roofline ridge point（FP32）
- **0.5 FLOPs/byte** — GPT-2 decode 的算术强度，比 ridge 低 115×
- **0.36%** — decode 的真实 FLOPs 利用率（用实测 GFLOPs/s ÷ peak compute）

---

## 1. Roofline 分析结论

### 1.1 硬件基线

| 指标 | 数值 |
|---|---:|
| Peak FP32 算力 | 33 TFLOPs |
| HBM 带宽（GDDR6X） | 576 GB/s |
| **Ridge point** | **57.3 FLOPs/byte** |

Ridge point 以下 = memory-bound，以上 = compute-bound。

### 1.2 GPT-2 在 Roofline 上的位置

| 阶段 | 算术强度 AI | 相对 ridge | 归属 | Roofline 天花板 |
|---|---:|:---:|:---:|---:|
| Decode（任意 prompt 长度） | **0.5 FLOPs/byte** | 115× 低于 ridge | **memory-bound** | 288 GFLOPs/s |
| Prefill L=32 | 16 | 低于 ridge | memory-bound | 9.2 TFLOPs/s |
| Prefill L=64 | 32 | 低于 ridge | memory-bound | 18.4 TFLOPs/s |
| Prefill L=128 | 64 | **高于 ridge** | **compute-bound** | 33 TFLOPs/s |
| Prefill L=256 | 128 | 高于 ridge | compute-bound | 33 TFLOPs/s |

Prefill 大约在 L=128 跨入 compute-bound 区。Decode 永远待在 memory-bound 谷底。

### 1.3 Roofline 与 W2 实测的对照

| prompt_len | prefill 实测 GFLOPs/s | 天花板 | 效率 | decode 实测 GFLOPs/s | 天花板 | 效率 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32  | 3,167  | 9,216  | 34% | 120 | 288 | 41.5% |
| 64  | 6,049  | 18,432 | 33% | 121 | 288 | 42.0% |
| 128 | 11,123 | 33,000 | 34% | 120 | 288 | 41.5% |
| 256 | 13,875 | 33,000 | 42% | 120 | 288 | 41.5% |

Decode 不论 prompt 多长都卡在 ~120 GFLOPs/s，对应 ~482 tok/s。根因：bs=1 时每一步都要把 496 MB 权重从 HBM 整块搬一次，单步只产出 1 个 token。

完整推导见 [roofline_notes.md](roofline_notes.md) 与 [w2_profiling_roofline_interpretation.md](w2_profiling_roofline_interpretation.md)。
Roofline 图：[roofline_chart.png](roofline_chart.png)

---

## 2. GPU 内存层级图

图：[gpu_memory_hierarchy.png](gpu_memory_hierarchy.png)（源码 [gpu_memory_hierarchy_diagram.py](gpu_memory_hierarchy_diagram.py)）

层级自上而下：

| 层级 | 容量（RTX 4090 Laptop） | 带宽（估） | 延迟 | 谁能访问 |
|---|---:|---:|:---:|---|
| Register | ~256 KB / SM | — | 1 cycle | 单个 warp |
| L1 / Shared Memory (SRAM) | 128 KB / SM | ~19 TB/s | ~30 cycle | 单个 SM 内所有 warp |
| L2 Cache | 48 MB（整卡共享） | ~5 TB/s | ~200 cycle | 所有 SM |
| **HBM / GDDR6X（device memory）** | **16 GB** | **576 GB/s** | ~400 cycle | 所有 SM |
| Host DRAM（via PCIe 4.0 x16） | 系统内存 | ~32 GB/s | ~ms 量级 | CPU-GPU DMA |

**关键洞察**：HBM 与 SRAM 之间存在 **30×+ 的带宽落差**。LLM decode 瓶颈的本质是这一跳——权重 500 MB 放不进 SRAM，只能每步从 HBM 重新搬到寄存器做 GEMV。所有经典优化都在对抗这一跳：

- **KV cache**：让过去步骤的 K/V 不必从 HBM 再算一遍
- **Flash Attention**：把 softmax 中间结果留在 SRAM 不写回 HBM
- **量化**：降低每个权重的字节数，变相拓宽 HBM 的有效带宽

---

## 3. FP32 / FP16 / INT8 Latency 对比

### 3.1 端到端延迟与显存（GPT-2 124M, bs=1, seq_len=128, decode 32 步）

| 精度 | Prefill (ms) | Decode (ms/token) | Peak Mem (MB) | 相对 FP32 |
|---|---:|---:|---:|---|
| **FP32** | 3.29  | 2.58 | 542.4 | baseline |
| **FP16** | 3.05  | 2.64 | 508.5 | prefill 快 7%，decode 持平，显存 −6% |
| **INT8**（bnb） | 14.64 | 9.94 | 320.4 | prefill 慢 4.4×，decode 慢 3.85×，显存 −41% |

原始数据：[precision_latency.json](precision_latency.json)，图：[precision_latency_comparison.png](precision_latency_comparison.png)

### 3.2 Nsight Compute 关键指标（decode 单步）

| 精度 | Kernel 数 | DRAM 均值 | **DRAM 峰值** | Tensor Core HMMA 峰值 | SM 活跃均值 |
|---|---:|---:|---:|---:|---:|
| FP32 | 235 | 17.9% | **96.1%** | 4.1% | 15.3% |
| FP16 | 235 | 13.3% | **94.6%** | 4.3% | 16.8% |
| INT8 | 555 | 4.5%  | **94.7%** | 4.3% | 10.8% |

图：[ncu_utilization_comparison.png](ncu_utilization_comparison.png)
原始数据：[ncu_decode_fp32.csv](ncu_decode_fp32.csv) / [ncu_decode_fp16.csv](ncu_decode_fp16.csv) / [ncu_decode_int8.csv](ncu_decode_int8.csv)

### 3.3 三条结论

**① 三种精度 decode 的 DRAM 峰值都 ≈ 95%——memory-bound 是硬件本质，换精度救不了。**
热点 kernel 都叫 `gemv*` 系列（General Matrix-Vector）：权重矩阵整块从 HBM 搬过来，只乘一个向量，算术强度没有任何改变的余地。

**② Tensor Core 只在 prefill FP16 被真正激活，decode 永远用不到。**
HMMA 指令要求两个输入都是矩阵。decode 的激活是向量，只能走 CUDA Core 的 FMA。这就是为什么所有 serving 系统都在做 batching——把 batch 维当作矩阵另一维，把 decode 的 GEMV 重新变成 GEMM，才能吃到 Tensor Core。

**③ INT8（bitsandbytes）在 GPT-2 这种小模型上是负优化。**
- 每次矩阵乘前后都要 quantize/dequantize，kernel 数从 235 涨到 555（+136%）
- `gemmSN_kernel_int32` 用 INT32 累加器，吞吐远低于同规模 FP16 Tensor Core GEMM
- INT8 的价值在"放不下 → 能放下"（例如 13B 在 24 GB 卡上），小模型没有这个必要

**W4/W5 优化方向推论**：不要用 bnb INT8 作为 GPT-2 的加速手段；真正的 decode 加速来自 **KV cache + continuous batching + MQA/GQA**——这些都是减少 HBM 读取量的优化，和 Roofline 诊断出的瓶颈方向一致。

---

## 4. 关键洞察

### 4.1 Decode 是深度 memory-bound，不是 compute-bound

- 算术强度 0.5 FLOPs/byte，比 ridge point 低 115×
- 三种精度下 DRAM 峰值都 ≈ 95%，HBM 被打满
- decode 的 ~2 ms/token 里，理论带宽下限就是 0.86 ms（496 MB / 576 GB/s），剩下 1.2 ms 是 kernel launch / Python-CUDA sync / KV 读取 overhead
- 即便把 RTX 4090 换成 H100，decode 也只会略快一点——带宽提升多少就加速多少，算力堆再多也无效

### 4.2 真实 FLOPs 利用率 < 1%（且 < 5% 这个假设成立）

| 度量 | 值 | 说明 |
|---|---:|---|
| Roofline 理论 decode 天花板 | 288 GFLOPs/s | = AI × 带宽 = 0.5 × 576 |
| 实测 decode 吞吐 | 120 GFLOPs/s | W2 profiling |
| **真实 FLOPs 利用率** | **0.36%** | 120 / 33,000 |
| Nsight Compute SM 活跃率 | 10–17% | 仅表示 warp resident，不等于算力利用率 |
| Nsight Compute HMMA 峰值（decode） | ~4% | Tensor Core 实际算力利用率 |

W3 原假设 "真实 FLOPs 利用率 < 5%" **成立**，而且比预期更极端（< 1%）。

**陷阱提醒**：`nvidia-smi dmon` 的 `sm%` 在 decode 下能显示到 64.6%，看上去 GPU 挺忙——但那是"至少一个 warp resident"的指标，memory-stalled warp 照样算 resident。Roofline 算出来的 0.36% 才是诚实答案。以后排查 LLM 性能，优先用 ncu 的 `sm__throughput` 和 `dram__throughput`，不要相信 `nvidia-smi` 的 `sm%`。

### 4.3 Prefill 能吃到算力，decode 吃不到——两个阶段要分别诊断

| 阶段 | 瓶颈 | 应该优化什么 |
|---|---|---|
| **Prefill**（GEMM） | L ≥ 128 后 compute-bound | FlashAttention、大 batch、FP16/BF16 + Tensor Core |
| **Decode**（GEMV） | 永远 memory-bound | KV cache、continuous batching、MQA/GQA、speculative decoding、权重量化 |

"把这俩当同一回事优化"是工程师最常踩的坑。Roofline 图上它俩根本不在一个区。

### 4.4 一切经典优化 = Roofline 图上移动红点

- **向右**（提升算术强度 AI）：KV cache、batching、speculative decoding
- **向上**（抬高天花板）：量化（FP16/INT8/INT4）、Flash Attention（减少 HBM 往返）

月度目标到 4 月底为止学到的每一个优化，都可以被归入这两种移动之一。这是后续所有实验的锚点。

---

## 5. W3 假设复核

| 假设 | 预测 | 实测 | 结论 |
|---|---|---|---|
| H1a decode SM util < 30% | 15–25% | 10.8–16.8%（ncu smsp 活跃） | ✅ 成立，比预期更低 |
| H1b decode mem util > 60% | > 60% | DRAM 峰值 94–96% | ✅ 大幅成立 |
| H3 prefill > decode 利用率 | prefill > 70%，decode < 30% | prefill SM 66%，decode 15% | ✅ 方向正确，prefill 略低 |
| **真实 FLOPs 利用率 < 5%** | < 5% | **0.36%**（Roofline）/ 4%（HMMA 峰值） | ✅ 远低于 5% |
| H2 大 batch 改善利用率 | — | 本周未测 | ⏳ 推到 W4/W5 |

主假设全部成立，W3 工程主体收工。

---

## 6. W3 交付物清单

- [x] [roofline_notes.md](roofline_notes.md) — Roofline 理论 + 4090 Laptop ridge 计算 + 五问五答
- [x] [w2_profiling_roofline_interpretation.md](w2_profiling_roofline_interpretation.md) — 用 Roofline 解释 W2 实测
- [x] [roofline_chart.png](roofline_chart.png) — Roofline 图（含 GPT-2 prefill/decode 实测点）
- [x] [gpu_memory_hierarchy.png](gpu_memory_hierarchy.png) — GPU 内存层级图
- [x] [precision_latency.json](precision_latency.json) + [precision_latency_comparison.png](precision_latency_comparison.png) — FP32/FP16/INT8 延迟对比
- [x] `ncu_{prefill,decode}_{fp32,fp16,int8}.csv`（共 6 份） — Nsight Compute 原始数据
- [x] [ncu_utilization_comparison.png](ncu_utilization_comparison.png) — ncu 利用率对比图
- [x] [W3_hypothesis.md](W3_hypothesis.md) — W3 原始假设文档 + ncu 附录
- [x] 本文件（w3_memo.md）— W3 综合 memo（中）
- [x] [w3_memo_en.md](w3_memo_en.md) — 英文版

**未做（合并到 W4）**：Docker Compose / FastAPI / Pydantic；哈希表 Day 2–4 + 双指针 Day 1 的 9 道 LeetCode 欠账；本周知乎博客。

---

## 7. W4 假设（4/20 起）

> **H4：在 RTX 4090 Laptop 上，bs=1、FP32 的 GPT-2 decode 启用 KV cache 后，在 seq_len ≥ 256 时相对无 cache 版本获得 >2× 的 decode 加速。**

### 推导

- 无 cache 时，decode 第 t 步需要重新对前 t−1 个 token 做 K/V 投影（O(t × d_model²)）
- 有 cache 时，只对当前第 t 个 token 做 K/V 投影（O(d_model²)），随 seq_len 增长总节省量是二次的
- seq_len ≥ 256 时，cache 节省的 FLOPs 与 HBM 搬运量都显著（>2× 预期合理）
- 但 cache 的收益本质是"减少重复 HBM 读"——与 W3 诊断出的 memory-bound 瓶颈方向一致

### 验收指标

| seq_len | 预期 decode ms/token 变化 | 预期 DRAM 峰值变化 |
|---|---|---|
| 32 | 基本持平或 <1.5× | 持平 |
| 128 | 1.5–2× | 略降 |
| **256** | **>2×** | 明显下降 |
| 512 | >3× | 显著下降 |

若 KV cache 打开后 DRAM 利用率下降但延迟没下降，说明瓶颈转到了 **kernel launch overhead / CPU-GPU sync**——那是下一个要攻克的点（CUDA Graph、torch.compile 等）。

### 观察点

- **HMMA 利用率**：理论上 cache 不会提升 decode 的 HMMA（仍是 GEMV），除非叠加 batching
- **kernel 数**：cache 不应显著增加 kernel 数，否则 launch overhead 会反噬
- **显存增长曲线**：`2 × n_layers × n_heads × head_dim × seq_len × batch × bytes` —— 测量到 OOM 边界

详细计划见 [roadmap/2026_04_Daily_Plan.md](../roadmap/2026_04_Daily_Plan.md) W4 段。

---

*Written: 2026-04-19（病后恢复版 W3 收尾）*
