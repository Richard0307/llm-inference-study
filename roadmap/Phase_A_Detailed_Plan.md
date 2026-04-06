# 阶段 A 详细执行计划：2026.04 - 2026.08（LLM Inference 系统优化版）

> **Phase A 主线：**LLM Inference 系统优化** —— 从 Transformer 底层组件到推理部署全链路。
>
> **一句话定位：** 你要做的是 **hands-on, systems-level, benchmark-grounded LLM Inference optimization research**。也就是不仅理解模型本身，还要理解"它如何在 GPU 上高效运行，瓶颈在哪，怎么优化"。
>
> **你的差异化故事：** 你不是只会调 API 的应用型选手，也不是只会设计 Agent 流程的架构师。你有计算机系统基础，你天然会从 **内存层级、计算瓶颈、量化精度、服务调度、解码策略** 这些维度去理解和优化 LLM 推理系统。这才是真正的技术壁垒和护城河。

---

## 这版 Phase A 到底在做什么

这一版计划统一围绕下面这个问题展开：

**如何从系统层面理解、分析和优化大语言模型的推理效率？**

把它拆开，就是四个子问题：

1. **架构理解问题**
   - Transformer 的 Attention、FFN、Normalization、Positional Encoding 各自如何影响推理效率？
   - 现代架构（LLaMA）的哪些改进（RoPE、RMSNorm、SwiGLU、GQA）专门针对推理优化？

2. **计算瓶颈问题**
   - Prefill 阶段是 compute-bound 还是 memory-bound？Decode 阶段呢？
   - KV Cache 显存随 batch/seq_len 如何增长？何时成为瓶颈？

3. **优化技术问题**
   - Flash Attention、量化（INT8/INT4/GPTQ/AWQ）、投机解码各自优化了什么？
   - 这些技术的精度-速度-显存 tradeoff 如何量化？

4. **系统服务问题**
   - Continuous Batching、PagedAttention 如何提升吞吐量？
   - vLLM 的调度和内存管理机制如何运作？

---

## 你这条线的科研故事

这版 Phase A 的目标，不是把你包装成"会用 vLLM 部署模型的工程师"，而是把你包装成：

**一个做 LLM Inference 系统优化研究的研究者，能从 Transformer 组件级别到推理系统级别做系统性的性能分析和优化实验，并把推理研究做成精度-效率-显存三位一体的量化评测。**

更适合申请时讲成下面这种话：

- 我对 LLM Inference 的系统级优化感兴趣
- 但我不满足于只用现成工具做部署
- 我尤其关注注意力机制的计算优化、KV Cache 管理、量化对模型质量的影响、以及服务系统的调度效率
- 我从组件级手写复现做起，所以我理解每一层优化背后的原理

这会比单纯说"我做 Agent"更有技术深度，也比单纯说"我用过 vLLM"更成体系。

---

## 这版 Phase A 的双主线

### 主线 A：Transformer + 推理核心组件

你这一阶段优先研究三类技术问题：

1. **模型架构与推理特性（Architecture & Inference Characteristics）**
2. **注意力优化与显存管理（Attention Optimization & Memory Management）**
3. **量化与高效推理（Quantization & Efficient Inference）**

这些方向的共同特点是：

- 有清晰的组件边界，适合消融实验
- 有现成 benchmark 支撑（perplexity、tokens/sec、显存占用）
- 更容易切出干净的 research question

### 主线 B：推理系统与部署优化

你不会把推理研究当成纯理论工作，而是把它变成有实验支撑的系统分析：

1. **做 attention mechanism ablation（naive vs Flash vs MQA/GQA）**
2. **做 quantization method comparison（GPTQ vs AWQ vs bitsandbytes）**
3. **做 serving system profiling（vLLM vs HuggingFace pipeline）**
4. **做 decoding strategy comparison（autoregressive vs speculative）**

也就是说：

- 别人只问 "这个模型部署上去了"
- 你会继续问 "瓶颈在哪，换一种方案会快多少，精度损失了多少，显存省了多少"

---

## 适合你发第一篇的贡献形态

这一版 Phase A 默认优先产出下面四类东西：

1. **推理组件消融实验**
   - 例如 naive attention vs Flash Attention vs Flash Attention 2 的显存/速度对比
   - 例如 RoPE vs Learned PE、RMSNorm vs LayerNorm 的推理效率差异

2. **量化方法系统评测**
   - FP16 vs INT8 vs INT4 vs GPTQ vs AWQ 的 perplexity-速度-显存联合分析
   - 不同量化粒度（per-tensor vs per-channel vs per-group）的效果差异

3. **推理服务性能分析**
   - vLLM vs HuggingFace 在不同并发/序列长度下的性能对比
   - Continuous Batching 在不同负载下的吞吐量-延迟 tradeoff

4. **解码策略效率对比**
   - Autoregressive vs Speculative Decoding 的速度-精度分析
   - 不同 draft model 大小和 gamma 值对 acceptance rate 的影响

---

## 整体节奏

| 时间段 | 主题 | 周数 |
|--------|------|------|
| 4月（W1-W4） | Transformer 从零复现 + GPU 计算基础 + 推理 Profiling | 4周 |
| 5月（W5-W8） | GPT-2 推理复现 + KV Cache + Flash Attention | 4周 |
| 6月（W9-W12） | Scaling Laws + 量化全面实战 | 4周 |
| 7月（W13-W16） | LLaMA 架构复现 + vLLM 部署与源码分析 | 4周 |
| 8月（W17-W20） | LoRA/QLoRA + 投机解码 + 技术报告 + Phase B 准备 | 4周 |

**每周固定节奏：**

- Day 1-2：论文精读 + 理论笔记
- Day 3-5：代码复现 + 工程实验（这是重点，至少 3 天在写代码和跑实验）
- Day 6：实验分析 + benchmark + 可视化结果
- Day 7：整理笔记、面试题库、公众号文章、Git push

**每天最低产出：**

- 1 次 GitHub 提交
- 1 份结构化实验记录或论文笔记
- 1 次闭卷主动回忆
- 1 个可执行小结果

**每周额外硬产出：**

- 1 份 mini research memo（1-2 页）
- 1 组可复现图表或表格
- 1 次负结果记录
- 1 套高密度压缩卡片
- 至少 200 行有效 Python 代码（注释和空行不算）

---

## 这版 Phase A 的关键指标

### 性能指标

- `inference latency`（ms/token）
- `throughput`（tokens/sec）
- `peak GPU memory`（GB）
- `KV Cache memory`（GB）
- `prefill time` vs `decode time`
- `FLOPs per forward pass`

### 质量与效率指标

- perplexity（量化前后对比）
- GPU utilization（%）
- memory bandwidth utilization（%）
- arithmetic intensity（FLOPs/byte）
- speedup ratio（优化后 vs baseline）
- quality degradation（量化/投机解码带来的精度损失）

> 这一组指标就是你把 LLM Inference 优化研究真正落实为可测量科研的地方。

---

## Phase A 学习操作系统：三核记忆体系

### 核心原则

1. **高密度压缩**
   - 每学完一个主题，必须压缩成一个可以快速回看的最小知识单元
   - 推荐统一格式：`1页总图 + 3个核心问题 + 10条要点`
   - 压缩对象不是原文摘抄，而是你自己的解释

2. **主动回忆**
   - 所有关键知识必须做闭卷回忆
   - 推荐四个时间点：`当天 30 分钟后 / 24 小时后 / 72 小时后 / 7 天后`
   - 回忆方式固定为三选一：
     - 白纸画架构图 / 数据流图
     - 口头讲 3 分钟
     - 不看资料写伪代码 / 公式推导 / 手算显存

3. **小规模复现**
   - 每个主题必须有一个最小可跑实验
   - 优先小模型、短序列、单变量对照
   - 单次实验尽量控制在 `2 小时内`

### 统一输出格式

每个主题尽量留下这四类材料：

- `压缩卡片`
- `回忆记录`
- `最小复现脚本`
- `mini memo`

### 每日学习模板

1. **输入 30-60 分钟**（论文/教材/博客）
2. **压缩 15-20 分钟**（卡片/总图）
3. **复现 45-120 分钟**（代码/实验）
4. **回忆 10-20 分钟**（闭卷测试）
5. **收口 5-10 分钟**（提交/记录）

### 每周检验标准

- 能不能 5 分钟讲清这周核心概念？
- 能不能画出 Transformer 推理的完整数据流？
- 能不能手算一个模型的显存 / KV Cache / FLOPs？
- 能不能跑出最小 benchmark 对比？
- 能不能同时解释效果和效率？

---

## 4月：Transformer 从零复现 + GPU 计算基础 + 推理 Profiling

### 本月目标

- 搭建本地 LLM Inference 研究实验环境
- 从零手写 Transformer 核心组件（Attention、FFN、PE、LayerNorm）
- 理解 GPU 内存层级和推理瓶颈（compute-bound vs memory-bound）
- 做第一次 model.generate() 全流程 profiling
- 形成第一套推理性能 baseline 数据

### W1（4/1-4/6）：环境搭建 + Attention 机制精读与手写

**本周三核执行法：**

- **高密度压缩：** 写 1 页 Scaled Dot-Product Attention 数学推导 + 1 页 Multi-Head Attention 设计动机卡
- **主动回忆：** 闭卷写出 Attention 的完整计算公式和 Multi-Head 的参数量计算
- **小规模复现：** 手写 ScaledDotProductAttention 和 MultiHeadAttention，用 assert 验证 shape

**实验任务：**

- 配置本地环境：
  - Python 3.10+ / PyTorch 2.x / CUDA
  - `torch.profiler` / `wandb` 基本配置
  - 验证 GPU 可用（`torch.cuda.is_available()`）
- 精读 Attention Is All You Need 的 Section 3.2（Attention）
- 手写 `ScaledDotProductAttention`：
  - 输入：Q, K, V (batch=2, seq_len=10, d_k=64)
  - 输出 shape: (2, 10, 64)
  - 验证：`torch.allclose` 与 `torch.nn.functional.scaled_dot_product_attention`
- 手写 `MultiHeadAttention`：
  - 输入：(batch=2, seq_len=10, d_model=512), num_heads=8
  - 验证输出 shape 和参数量

**产出：**

- `README.md`（环境信息）
- `W1/scaled_dot_product_attention.py`
- `W1/multi_head_attention.py`
- 1 张 Attention 计算流程卡

### W2（4/7-4/13）：Transformer 完整复现 + Causal Mask

**本周三核执行法：**

- **高密度压缩：** 做 1 张 Transformer Encoder-Decoder 完整架构图
- **主动回忆：** 闭卷画出 Transformer 的数据流（从输入 embedding 到输出 logits）
- **小规模复现：** 手写完整 Transformer 并在小型翻译数据集上跑通

**实验任务：**

- 手写 PositionalEncoding（sin/cos）
- 手写 PositionwiseFeedForward
- 手写 EncoderLayer 和 DecoderLayer
- 组装完整 Transformer
- 实现 Causal Mask
- 用 Multi30k (de→en) 或类似小型数据集跑通训练循环，用 wandb 记录 loss
- 对比自己实现的 attention 输出与 `torch.nn.MultiheadAttention` 的输出（`torch.allclose`）

**产出：**

- `W2/positional_encoding.py`
- `W2/transformer_model.py`
- `W2/train_translation.py`
- 1 张 training loss 曲线图

### W3（4/14-4/20）：GPU 计算基础 + 推理瓶颈分析

**本周三核执行法：**

- **高密度压缩：** 做 1 张 GPU 内存层级图（HBM→L2→SRAM→Register）+ 1 页 Roofline Model 解读
- **主动回忆：** 闭卷解释 compute-bound vs memory-bound 的区别，以及 prefill vs decode 各自的瓶颈
- **小规模复现：** 对 GPT-2 做完整推理 profiling，手写显存计算器

**实验任务：**

- 用 `torch.profiler` 对 HuggingFace GPT-2 的 `model.generate()` 做 profiling，导出 Chrome trace
- 用 `torch.cuda.Event` 分别测量 prefill 和 decode 阶段耗时
- 写 `memory_calculator.py`：输入模型参数量、层数、head 数、head_dim、seq_len、batch_size、dtype，输出模型显存 + KV Cache 显存 + 激活显存的估算值
- 用 `nvidia-smi dmon` 监控推理过程 GPU 利用率，画时间-利用率曲线
- 手写 GEMM vs GEMV benchmark（用 `torch.matmul`），验证不同矩阵维度下的吞吐量差异

**产出：**

- `W3/gpt2_profiling.py`
- `W3/memory_calculator.py`
- `W3/gemm_vs_gemv_benchmark.py`
- 1 张 GPU 利用率曲线图
- 1 张 GEMM vs GEMV 吞吐量对比图

### W4（4/21-4/27）：4 月 mini project 收口

**本周三核执行法：**

- **高密度压缩：** 做 1 张 4 月总图：Transformer 组件 → 推理特性 → GPU 瓶颈
- **主动回忆：** 脱稿讲清 Attention 的 O(n²) 复杂度为什么是推理瓶颈，以及 prefill/decode 的不同性能特征
- **小规模复现：** 重跑 4 月最关键 benchmark 结果

**产出：**

- `W4/april_mini_project/`
- `W4/april_report.md`
- 1 张 Transformer 推理性能总表（显存、延迟、GPU 利用率）

**月末检查点：**

- [ ] 你有从零手写的完整 Transformer（不用 nn.Transformer）
- [ ] 你能手算模型显存和 KV Cache 显存
- [ ] 你有 GPT-2 推理的 profiling 数据（prefill/decode 分离）
- [ ] 你能解释 compute-bound vs memory-bound 以及 GEMM vs GEMV

---

## 5月：GPT-2 推理复现 + KV Cache + Flash Attention

### 本月目标

- 从零实现 minimal GPT-2，手动实现 KV Cache
- 深度理解 Flash Attention 原理和 tiling 思想
- 实现 MQA 和 GQA，建立第一套注意力机制消融实验
- 形成 KV Cache 显存边界分析

### W5（4/28-5/4）：GPT-2 从零实现 + KV Cache

**本周三核执行法：**

- **高密度压缩：** 做 1 张 GPT-2 推理数据流图（embedding → blocks → lm_head → sampling）
- **主动回忆：** 闭卷画出 KV Cache 的工作原理（为什么能避免重复计算）
- **小规模复现：** 手写 minimal GPT-2，对比有无 KV Cache 的生成速度

**实验任务：**

- 从零手写 minimal GPT-2（参考 Karpathy nanoGPT）：
  - CausalSelfAttention（含 causal mask）
  - GPT Block（Pre-Norm + Attention + FFN）
  - GPT Model（embedding + blocks + lm_head）
- 加载 HuggingFace GPT-2 预训练权重，验证 `model.generate()` 输出一致
- 实现 greedy decoding 和 top-k sampling 两种生成策略
- 手动实现 KV Cache 版 decode：
  - 对比有无 KV Cache 的逐 token 生成速度（用 `time.perf_counter` 测量）
  - 画出 speedup 曲线
- 分析从 inference 视角看 BERT 双向注意力为什么无法用于自回归生成

**产出：**

- `W5/minimal_gpt2.py`
- `W5/kv_cache_decode.py`
- `W5/kv_cache_benchmark.py`
- 1 张 KV Cache speedup 曲线图

### W6（5/5-5/11）：Flash Attention 原理 + Tiling 实现

**本周三核执行法：**

- **高密度压缩：** 做 1 张 Flash Attention tiling + online softmax 原理图
- **主动回忆：** 闭卷推导 online softmax 的递推公式
- **小规模复现：** 实现 Python 层面的 tiling attention，验证与 naive attention 输出一致

**实验任务：**

- 实现 naive attention（标准 O(n²) 显存）并用 `torch.cuda.max_memory_allocated()` 测量峰值显存
- 实现简化版 Flash Attention（Python 层面的 tiling 模拟）：
  - 将 Q/K/V 按 block_size 分块
  - 逐块计算 attention 并用 online softmax 合并
  - 验证输出与 naive attention 的 `torch.allclose`
- 对比三者的峰值显存和速度：
  - naive attention
  - 自己的 tiling 版
  - `torch.nn.functional.scaled_dot_product_attention`（PyTorch 内置 Flash Attention）
- 画出 seq_len vs 显存占用曲线（seq_len 从 512 到 8192）

**产出：**

- `W6/naive_attention.py`
- `W6/flash_attention_tiling.py`
- `W6/attention_memory_benchmark.py`
- 1 张 seq_len vs 显存占用对比图

### W7（5/12-5/18）：MQA/GQA + KV Cache 显存边界

**本周三核执行法：**

- **高密度压缩：** 做 1 张 MHA → MQA → GQA 演进图，标注每种方式的 KV Cache 参数量公式
- **主动回忆：** 闭卷计算给定配置下 MHA/MQA/GQA 的 KV Cache 大小
- **小规模复现：** 实现三种 attention，用实验验证 KV Cache 大小的理论计算

**实验任务：**

- 实现 MQA（1 个 KV head）和 GQA（num_kv_heads = num_heads // 4）
- 对比 MHA/MQA/GQA 在相同 seq_len 下的 KV Cache 大小
- 写 KV Cache benchmark：
  - 固定模型大小，变化 batch_size 和 seq_len
  - 测量何时 OOM
  - 画出 "batch × seq_len 的显存边界图"

**产出：**

- `W7/mqa_gqa_attention.py`
- `W7/kv_cache_boundary.py`
- 1 张显存边界热力图

### W8（5/19-5/25）：Mini Project 1 收口

**项目主题：**

`Attention Mechanism and KV Cache: A Component-Level Inference Efficiency Analysis`

**实验要求：**

- 1 个 attention 机制消融（naive / Flash / MQA / GQA）
- 1 个 KV Cache 显存分析
- 1 组 prefill vs decode 性能对比

**产出：**

- `W8/attention_kv_project/`
- `W8/attention_kv_report.md`

**月末检查点：**

- [ ] 你有从零手写的 GPT-2（可加载预训练权重）
- [ ] 你能解释 Flash Attention 的 tiling 和 online softmax 原理
- [ ] 你有 MHA/MQA/GQA 的 KV Cache 大小对比数据
- [ ] 你有 KV Cache 显存边界分析图

---

## 6月：Scaling Laws + 量化全面实战

### 本月目标

- 理解 Scaling Laws 和 Chinchilla 对推理成本的影响
- 掌握量化核心原理（对称/非对称、PTQ/QAT）
- 实战 GPTQ、AWQ、bitsandbytes
- 建立量化方法系统评测框架

### W9（5/26-6/1）：Scaling Laws 实验 + FLOPs 计算

**本周三核执行法：**

- **高密度压缩：** 做 1 张 Scaling Laws 三条曲线总图 + Chinchilla 修正要点
- **主动回忆：** 闭卷写出 6ND FLOPs 近似公式和 Chinchilla-optimal 的核心结论
- **小规模复现：** 在 nanoGPT 上跑 mini scaling laws 实验

**实验任务：**

- 在 nanoGPT 上训练 3-5 个不同大小的模型（1M/5M/10M/50M 参数）
- 固定数据集（OpenWebText 子集），用 wandb 画出参数量 vs validation loss 曲线
- 测量不同模型大小的推理延迟（固定 seq_len=256, batch=1）
- 画出参数量 vs 推理延迟和参数量 vs 吞吐量（tokens/sec）曲线
- 计算 Chinchilla-optimal 下训练 7B 模型需要多少 tokens，与 LLaMA 实际训练数据量对比
- 写 `flops_calculator.py`：输入模型配置，输出 forward pass FLOPs 估算（用 6ND 近似 + 逐层计算）
- 分析："同样的训练预算，训一个大模型还是训一个小模型久一点，哪个推理更划算？"

**产出：**

- `W9/mini_scaling_laws.py`
- `W9/flops_calculator.py`
- 1 张参数量 vs loss vs 推理延迟曲线图

### W10（6/2-6/8）：量化基础 + 手写量化函数

**本周三核执行法：**

- **高密度压缩：** 做 1 张浮点数表示对比卡（FP32/FP16/BF16/INT8/INT4/FP8）+ 1 页对称 vs 非对称量化推导
- **主动回忆：** 闭卷写出 quantize/dequantize 的公式和代码
- **小规模复现：** 手写量化函数，用 bitsandbytes 做模型量化对比

**实验任务：**

- 手写 `quantize_tensor(x, bits=8)` 和 `dequantize_tensor(x_q, scale, zero_point)`
- 验证 quantize→dequantize 的重建误差
- 用 `bitsandbytes` 将 GPT-2（或 LLaMA-7B）分别加载为 FP16 / INT8 / INT4（NF4）
- 对比显存占用、perplexity、生成速度
- 可视化权重分布：画出量化前后某一层 weight 的直方图

**产出：**

- `W10/quantization_basics.py`
- `W10/bitsandbytes_comparison.py`
- 1 张量化方法对比表 + 1 张权重分布直方图

### W11（6/9-6/15）：GPTQ + AWQ 实战

**本周三核执行法：**

- **高密度压缩：** 做 1 张 GPTQ（OBQ/Hessian-based）vs AWQ（activation-aware scaling）原理对比卡
- **主动回忆：** 闭卷解释 GPTQ 为什么按列量化、AWQ 为什么关注 salient channels
- **小规模复现：** 用 auto-gptq 和 awq 库分别量化同一模型并对比

**实验任务：**

- 用 `auto-gptq` 对 7B 模型做 GPTQ 4-bit 量化，记录量化前后 perplexity 和推理速度
- 用 `awq` 库做 AWQ 量化，对比 GPTQ vs AWQ 在相同 bit-width 下的质量差异
- 写统一 benchmark 脚本 `quantization_benchmark.py`：
  - 输出表格：方法 × 精度 × 显存 × perplexity × tokens/sec
- 可视化权重分布：画出量化前后某一层 weight 的直方图，观察 outlier 分布

**产出：**

- `W11/gptq_quantization.py`
- `W11/awq_quantization.py`
- `W11/quantization_benchmark.py`
- 1 张方法 × 精度 × 显存 × perplexity × tokens/sec 总表

### W12（6/16-6/22）：Mini Project 2 收口

**项目主题：**

`Quantization Methods for LLM Inference: A Systematic Accuracy-Efficiency Analysis`

**实验要求：**

- 至少 3 种量化方法对比（bitsandbytes / GPTQ / AWQ）
- 精度-速度-显存三维联合分析
- 权重分布可视化

**产出：**

- `W12/quantization_project/`
- `W12/quantization_report.md`

**月末检查点：**

- [ ] 你能解释对称 vs 非对称量化的数学原理
- [ ] 你有 GPTQ/AWQ/bitsandbytes 的系统对比数据
- [ ] 你的 benchmark 有具体数字（显存 X GB、速度 X tok/s、ppl X.XX）

---

## 7月：LLaMA 架构复现 + vLLM 部署与源码分析

### 本月目标

- 从零手写 LLaMA 核心组件（RoPE、RMSNorm、SwiGLU、GQA）
- 理解 vLLM 的 Continuous Batching 和 PagedAttention
- 做 vLLM vs HuggingFace 的服务性能对比
- 形成推理服务系统的完整认知

### W13（6/23-6/29）：LLaMA 组件手写 + 权重加载

**本周三核执行法：**

- **高密度压缩：** 做 1 张 LLaMA 架构改进全景图（RoPE/RMSNorm/SwiGLU/GQA/Pre-Norm）
- **主动回忆：** 闭卷推导 RoPE 的旋转矩阵和相对位置编码性质
- **小规模复现：** 从零手写每个组件，逐个验证与 HuggingFace 输出一致

**实验任务：**

- 从零手写 RMSNorm：`class RMSNorm(nn.Module)`，验证与 HuggingFace LlamaRMSNorm 输出一致
- 从零手写 RoPE：实现 `rotary_embedding(x, seq_len)` 和 `apply_rotary_pos_emb(q, k, cos, sin)`
  - 验证位置旋转后内积只依赖相对位置差
- 从零手写 SwiGLU：`class SwiGLU(nn.Module)`
  - 对比 SwiGLU vs ReLU vs GELU 的前向传播速度和梯度分布
- 组装 minimal LLaMA：用以上组件 + GQA attention 搭建完整 LLaMA-style 模型
- 加载 HuggingFace LLaMA-2-7B 权重（至少加载一层并验证输出一致）
- 对比推理效率：RoPE vs Learned PE、RMSNorm vs LayerNorm、SwiGLU vs GELU（用 `torch.cuda.Event` 测量）

**产出：**

- `W13/rmsnorm.py`
- `W13/rope.py`
- `W13/swiglu.py`
- `W13/minimal_llama.py`
- 1 张组件效率对比表

### W14（6/30-7/6）：vLLM 部署 + 服务性能基线

**本周三核执行法：**

- **高密度压缩：** 做 1 张 Static Batching vs Continuous Batching 对比图 + PagedAttention 虚拟内存类比图
- **主动回忆：** 闭卷解释 TTFT / TPOT / TPS 三个 SLO 指标的定义和意义
- **小规模复现：** 用 vLLM 部署模型，写 benchmark 脚本测试性能

**实验任务：**

- 在 RunPod 或本地用 vLLM 部署 LLaMA-2-7B（或 Qwen-7B），用 OpenAI-compatible API 接口测试
- 写 `serving_benchmark.py`：
  - 不同 concurrency（1/4/8/16/32 并发请求）
  - 不同 max_tokens（64/256/1024）
  - 记录 TTFT、TPOT、吞吐量（tokens/sec）
  - 画出性能曲线
- 对比 vLLM vs HuggingFace `pipeline("text-generation")` 在相同模型、相同请求下的吞吐量差异

**产出：**

- `W14/vllm_deployment/`
- `W14/serving_benchmark.py`
- 1 张并发-吞吐量-延迟性能曲线图

### W15（7/7-7/13）：vLLM 源码分析 + Continuous Batching 模拟

**本周三核执行法：**

- **高密度压缩：** 画 1 张 vLLM 请求完整生命周期流程图
- **主动回忆：** 闭卷解释 Block Manager 的分配/换入/换出逻辑
- **小规模复现：** 实现极简版 continuous batching 模拟器

**实验任务：**

- 阅读 vLLM 源码关键模块：
  - `vllm/core/block_manager.py`：理解 Block 分配和换入换出逻辑
  - `vllm/core/scheduler.py`：理解 continuous batching 的调度决策
- 用自己的话画出 vLLM 处理一个请求的完整生命周期流程图
- 实现极简版 continuous batching 模拟器（纯 Python，不需要 GPU）：
  - 模拟多个请求到达
  - 按 iteration-level 调度
  - 统计平均延迟和吞吐量
- 调整 vLLM 的 `gpu_memory_utilization` 参数（0.5/0.7/0.9），观察 KV Cache 可用 block 数量和最大并发处理能力的变化

**产出：**

- `W15/vllm_lifecycle_diagram.png`
- `W15/continuous_batching_simulator.py`
- 1 份 vLLM 源码分析笔记

### W16（7/14-7/20）：综合项目

**项目主题：**

`LLM Inference Serving: From Component Architecture to System-Level Optimization`

**实验要求：**

- 至少 2 种模型架构组件的效率对比
- vLLM 部署的完整性能分析
- 不同 gpu_memory_utilization 下的吞吐量对比

**产出：**

- `W16/serving_project/`
- `W16/serving_report.md`

**月末检查点：**

- [ ] 你有从零手写的 LLaMA 核心组件（RoPE/RMSNorm/SwiGLU/GQA）
- [ ] 你能画出 vLLM 的请求处理完整流程
- [ ] 你有 vLLM 在不同配置下的性能数据
- [ ] 你能解释 PagedAttention 的虚拟内存思想

---

## 8月：LoRA/QLoRA + 投机解码 + 技术报告 + Phase B 准备

### 本月目标

- 实战 LoRA/QLoRA 微调与推理部署
- 实现 Speculative Decoding 并量化 speedup
- 形成 LLM Inference 优化技术全景总结
- 写综合技术报告，让 GitHub 仓库达到可展示水平

### W17（7/21-7/27）：LoRA/QLoRA 微调 + 推理效率

**本周三核执行法：**

- **高密度压缩：** 做 1 张 LoRA 低秩分解原理图（W = W₀ + BA）+ QLoRA 三技术要点卡
- **主动回忆：** 闭卷推导 LoRA 的 forward：`y = W₀x + BAx`，解释为什么 intrinsic dimension 低
- **小规模复现：** 手写 minimal LoRA 层 + QLoRA 微调实战

**任务：**

- 手写 `class LoRALinear(nn.Module)`：frozen 的 W₀ + trainable 的 A, B（rank=4）
  - 验证 forward 输出为 `W₀x + BAx`
- 用 `peft` + `trl` 对 LLaMA-2-7B 做 QLoRA 微调（Alpaca 指令微调）
- 用 wandb 记录 training loss、eval loss、GPU 显存占用
- 对比不同 rank（r=4/8/16/64）的训练速度和最终效果
- 对比显存：Full Fine-Tuning vs LoRA vs QLoRA（`torch.cuda.max_memory_allocated()`）
- 测试 adapter merge：对比 merge 后推理速度 vs 动态加载 adapter 推理速度
- 将微调后模型部署到 vLLM，验证端到端 微调→部署 流程

### W18（7/28-8/3）：Speculative Decoding 实现

**本周三核执行法：**

- **高密度压缩：** 做 1 张 Speculative Decoding 的 draft-then-verify 流程图 + rejection sampling 数学保证
- **主动回忆：** 闭卷推导 rejection sampling 为什么保证输出分布不变
- **小规模复现：** 实现完整 Speculative Decoding demo

**任务：**

- 实现完整 Speculative Decoding demo：
  - Draft model：GPT-2 small (124M)
  - Target model：GPT-2 large (774M)
  - 实现 `speculative_decode(prompt, draft_model, target_model, gamma=4)`
  - 实现 rejection sampling 验证逻辑
  - 测量 acceptance rate 在不同 gamma（2/4/8）下的变化
  - 对比 speculative decoding vs 普通 autoregressive decoding 的 wall-clock speedup
- 实现简单的 Tensor Parallelism 模拟：将线性层 weight 按列切分到 2 个"虚拟 GPU"
- 写 `inference_optimization_comparison.py`：FP16 baseline / INT8 / INT4 / Speculative Decoding 速度和质量对比
- 画出 LLM Inference 优化技术全景图

### W19（8/4-8/10）：写综合技术报告 + PhD 材料

**任务：**

- 整理所有实验结果，生成完整 benchmark 报告
- 确保 GitHub 仓库 README 完整、代码可运行、有清晰的 reproduction instructions
- 回顾所有代码，补充注释和 docstring
- 写 `EXPERIMENTS.md`：汇总所有 benchmark 数字（显存、速度、perplexity）
- 完成 PhD research statement 初稿（300-500 字英文）

**产出：**

- `tech_report.md` 或 `pre_paper_draft.md`
- `EXPERIMENTS.md`

### W20（8/11-8/17）：Phase B 衔接准备

**任务：**

- 复盘哪一条线最适合继续深挖：
  - Flash Attention 变种与 long-context 推理
  - 量化方法改进（mixed-precision, outlier-aware）
  - Serving 系统优化（prefill-decode disaggregation）
  - Speculative Decoding 改进（Medusa, EAGLE）
  - CUDA kernel 级别优化
- 整理 10-15 位目标导师列表（含研究方向、代表论文、学校）
- 整理完整面试题库（目标 50+ 题）

**产出：**

- Phase B 阅读清单
- 下一阶段问题定义草案
- PhD research statement 完整版 + 15 位导师列表

---

## 完整论文阅读进度表

| 周次 | 论文 | 精读/粗读 | 状态 |
|------|------|-----------|------|
| W1 | Attention Is All You Need | 精读 | |
| W2 | Attention Is All You Need（工程复现周） | 复现 | |
| W3 | GPU 计算基础：Roofline Model / CUDA 内存层级（博客/教程） | 精读 | |
| W5 | Language Models are Unsupervised Multitask Learners (GPT-2) | 精读 | |
| W6 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 精读 | |
| W7 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | 精读 | |
| W9 | Scaling Laws for Neural Language Models (Kaplan et al.) | 精读 | |
| W9 | Training Compute-Optimal Large Language Models (Chinchilla) | 精读 | |
| W10 | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | 精读 | |
| W11 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | 精读 | |
| W11 | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | 精读 | |
| W13 | LLaMA: Open and Efficient Foundation Language Models | 精读 | |
| W14 | Efficient Memory Management for Large Language Model Serving with PagedAttention | 精读 | |
| W15 | Orca: A Distributed Serving System for Transformer-Based Generative Models | 精读 | |
| W17 | LoRA: Low-Rank Adaptation of Large Language Models | 精读 | |
| W17 | QLoRA: Efficient Finetuning of Quantized Language Models | 精读 | |
| W18 | Fast Inference from Transformers via Speculative Decoding | 精读 | |
| W18 | Medusa: Simple LLM Inference Acceleration Framework | 粗读 | |

---

## 阶段 A 交付物 Checklist

- [ ] 一个从零手写的完整 Transformer + minimal GPT-2 + minimal LLaMA
- [ ] 1 个 attention 机制消融项目（naive / Flash / MQA / GQA）
- [ ] 1 个量化方法系统评测项目（FP16 / INT8 / INT4 / GPTQ / AWQ）
- [ ] 1 个 vLLM 部署与性能分析项目
- [ ] 1 个 Speculative Decoding 实现与 benchmark
- [ ] 4-6 份 mini research memo
- [ ] 至少 10 张高密度压缩卡片
- [ ] 至少 1 轮 7 天闭卷回忆记录
- [ ] 1 份综合技术报告或 pre-paper 草稿
- [ ] 1 个可公开展示的 GitHub 仓库（代码可运行、有 reproduction instructions）
- [ ] 1 份完整 `EXPERIMENTS.md`（汇总显存、速度、perplexity 数字）
- [ ] 至少 50 道面试题库
- [ ] 1 份 PhD research statement + 15 位导师列表

---

## 费用预估

| 项目 | 费用 |
|------|------|
| 本地 RTX 4090（自有） | 电费 |
| RunPod/Vast.ai 云 GPU（大模型实验） | 约 $50-150 |
| 可选 API / HuggingFace Pro | 约 $0-30 |
| **合计** | **约 $50-180** |

> 这版计划的核心优势在于：LLM Inference 优化是有明确技术壁垒的方向——它要求你理解 GPU 计算原理、内存管理、数值精度、系统调度，而不是简单的 API 调用和 prompt 工程。这才是真正的护城河。
