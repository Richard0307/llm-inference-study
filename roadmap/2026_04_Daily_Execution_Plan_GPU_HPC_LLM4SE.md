# 2026 年 4 月逐日执行版（LLM Inference 系统优化）

> 适用前提：
> - 本地有一台 4090 笔记本
> - 4 月主线是 `Transformer 从零复现 + GPU 计算基础 + 推理 Profiling + KV Cache 实现`
> - 这份文档服务于新的 `LLM Inference 系统优化` 版 Phase A

---

## 4 月总目标

- 搭建稳定可用的本地 LLM Inference 研究实验环境
- **Day 1 就跑通 GPT-2 推理，带着"它为什么慢"的问题进入后续学习**
- 从零手写 Decoder-only Transformer 核心组件（跳过 Encoder，直接面向推理架构）
- 理解 GPU 内存层级和推理瓶颈（compute-bound vs memory-bound、GEMM vs GEMV）
- 做 GPT-2 全流程 profiling（prefill/decode 分离）
- **手写 minimal GPT-2 + KV Cache 实现，拿到"有无 KV Cache 的速度对比"硬数据**
- 每周一个"假设→验证"闭环，从第一周就培养 research thinking

---

## 本月固定规则

- Day 1-2：论文精读 + 理论笔记
- Day 3-5：代码复现 + 工程实验（重点，至少 3 天写代码跑实验）
- Day 6：实验分析 + benchmark + 可视化结果
- Day 7：整理笔记、面试题库、公众号文章、Git push

**每天最低产出：**

- 1 次 GitHub 提交
- 1 份实验记录或论文笔记
- 1 次闭卷回忆
- 1 个可执行小结果

**每周额外要求（新增）：**

- 1 个明确的假设（hypothesis），在本周内用实验验证或否证

---

## W1：4/1 - 4/6

> **本周假设：** Attention 计算在 GPT-2 推理中占比超过 50%（用 profiler 验证）

### 4 月 1 日（周三）

- 主任务：搭环境 + **第一次推理体验**
- 要做：
  - 整理本地 Python / PyTorch / CUDA 环境信息
  - 验证 GPU 可用：`torch.cuda.is_available()`、显存大小、CUDA 版本
  - 配置 `torch.profiler` 和 `wandb` 基本设置
  - **核心：下载 HuggingFace GPT-2，跑一次 `model.generate()`**
    - 输入 prompt: "The future of AI is"
    - 生成 128 tokens
    - 用 `time.perf_counter()` 记录总耗时
    - 用 `torch.cuda.max_memory_allocated()` 记录峰值显存
    - 记录：总耗时 X ms、吞吐量 X tokens/sec、峰值显存 X MB
  - **写下你的第一个问题：** "这 X ms 花在了哪里？哪个组件最慢？"
- 当天输出：
  - 1 份环境记录（Python/PyTorch/CUDA 版本、GPU 型号和显存）
  - 1 份 GPT-2 推理初体验记录（耗时、显存、吞吐量）
  - 1 个问题："推理为什么慢？我猜是因为 ___"
- 完成标准：
  - 你亲眼看到了 GPT-2 推理的速度和显存，有了直觉

### 4 月 2 日（周四）

- 主任务：精读 Attention Is All You Need — Attention 部分
- 要做：
  - 精读 Section 3.2（Scaled Dot-Product Attention）和 Section 3.2.2（Multi-Head Attention）
  - 推导 Attention 公式：`Attention(Q,K,V) = softmax(QK^T / √d_k)V`
  - 理解为什么要除以 √d_k（softmax 梯度饱和问题）
  - 写 1 页 Attention 数学推导卡
  - **带着 Day 1 的问题读：** Attention 的 O(n²) 复杂度意味着什么？它在推理中有多贵？
- 当天输出：
  - 结构化论文笔记
  - 1 张 Attention 计算流程与数学推导卡
- 完成标准：
  - 能闭卷写出 Attention 的完整公式
  - 能解释 Multi-Head Attention 的设计动机（多子空间 + 参数量不变）
  - 能初步回答"Attention 在推理中为什么贵"

### 4 月 3 日（周五）

- 主任务：手写 ScaledDotProductAttention + MultiHeadAttention
- 要做：
  - 用 PyTorch 手写 `ScaledDotProductAttention`：
    - 输入：Q, K, V (batch=2, seq_len=10, d_k=64)
    - 输出 shape: (2, 10, 64)
    - 验证：`torch.allclose` 与 `F.scaled_dot_product_attention`
  - 用 PyTorch 手写 `MultiHeadAttention`：
    - 输入：(batch=2, seq_len=10, d_model=512), num_heads=8
    - 验证输出 shape: (2, 10, 512)
    - 计算参数量并验证：4 × d_model² = 4 × 512² = 1,048,576
- 当天输出：
  - `W1/scaled_dot_product_attention.py`
  - `W1/multi_head_attention.py`
  - 1 张最小结果表（shape 验证 + torch.allclose 结果）
- 完成标准：
  - 两个模块 assert 全部通过
  - 与 PyTorch 官方实现输出一致

### 4 月 4 日（周六）

- 主任务：推理视角分析 + 手算显存 + 验证本周假设
- 要做：
  - 从推理视角分析 Attention 的计算复杂度：
    - 时间复杂度 O(n²d)，空间复杂度 O(n²)
    - 为什么 seq_len 增长时 Attention 成为瓶颈
  - 手算一个具体例子的 Attention 显存：
    - d_model=512, num_heads=8, seq_len=1024, batch=4, FP16
    - QKV 矩阵 + attention weights 的显存
  - **验证本周假设：** 用 `torch.profiler` 跑一次 GPT-2，看 Attention 相关 op 占总耗时的比例
    - 如果 > 50%，假设成立
    - 如果 < 50%，记录实际占比和真正的瓶颈是什么
- 当天输出：
  - 1 页总结笔记（含推理视角分析）
  - 1 道手算显存题（含完整过程）
  - **本周假设验证结论**（含 profiler 截图或数据）
- 完成标准：
  - 能手算给定配置下 Attention 的显存消耗
  - 本周假设有明确的"成立"或"不成立"结论

### 4 月 5 日（周日）

- 主任务：输出一篇轻量内容
- 要做：
  - 写知乎或公众号文章
  - 主题建议：「从零手写 Attention：理解 Transformer 推理的第一步」
- 当天输出：
  - 1 篇对外内容草稿
- 完成标准：
  - 有 1 份外部可读内容

### 4 月 6 日（周一）

- 主任务：手写 PositionalEncoding + PositionwiseFeedForward
- 要做：
  - 手写 sin/cos Positional Encoding：
    - 输入：(batch=2, seq_len=10, d_model=512)
    - 验证 PE 矩阵的 shape 和数值范围
  - 手写 PositionwiseFeedForward：
    - 输入：(batch=2, seq_len=10, d_model=512), d_ff=2048
    - 验证输出 shape 和参数量
  - 做 1 次 GitHub 提交
- 当天输出：
  - `W1/positional_encoding.py`
  - `W1/feedforward.py`
  - 统一结果记录模板
- 完成标准：
  - PE 和 FFN 模块 assert 全部通过
  - 能解释为什么用 sin/cos 而不是 learned PE（泛化性 + 无参数）

---

## W2：4/7 - 4/13

> **本周假设：** GPT-2 推理中 Decode 阶段占总耗时超过 80%（用 torch.cuda.Event 验证）
>
> **本周关键改动：** 跳过 Encoder-Decoder 架构，直接做 Decoder-only。Encoder 和 Cross-Attention 对 LLM Inference 研究几乎没有价值。

### 4 月 7 日（周二）

- 主任务：手写 LayerNorm + Causal Mask
- 要做：
  - 手写 LayerNorm：验证与 `torch.nn.LayerNorm` 输出一致（`torch.allclose`）
  - 手写 Causal Mask（下三角矩阵）：
    - 验证 mask 后 attention weights 的上三角为 0
    - 理解：Causal Mask 是自回归生成的前提，也是 KV Cache 能工作的前提
  - 思考：没有 Causal Mask 为什么不能用 KV Cache？
    - 答案：因为双向注意力下，每个新 token 会改变所有位置的 attention，KV 不能复用
- 当天输出：
  - `W2/layer_norm.py`
  - `W2/causal_mask.py`
  - 1 段关于 Causal Mask 与 KV Cache 关系的笔记
- 完成标准：
  - LayerNorm `torch.allclose` 通过
  - 能解释 Causal Mask → 单向注意力 → KV 可复用 这条推理链

### 4 月 8 日（周三）

- 主任务：手写 GPT-style Decoder Block（Pre-Norm）
- 要做：
  - 手写 Decoder Block = Pre-Norm + CausalSelfAttention + FFN
    - **注意：这里是 Pre-Norm（先 Norm 再 Attention），不是原始 Transformer 的 Post-Norm**
    - 这就是 GPT-2 / LLaMA 实际使用的结构
  - **不写 Encoder、不写 Cross-Attention、不写 Encoder-Decoder**
    - 原因：现代 LLM 推理全部是 Decoder-only，Encoder 对你的研究方向无用
  - 验证：输入 (batch=2, seq_len=10, d_model=512)，输出 shape 正确
  - 叠加多个 Block，验证 N 层 Decoder 的 forward
- 当天输出：
  - `W2/decoder_block.py`（Pre-Norm 版）
  - 1 张 GPT-style Decoder Block 数据流图
- 完成标准：
  - 单个 Block 和多层 Block forward 跑通
  - 能画出 LN → Attention → Residual → LN → FFN → Residual 的数据流

### 4 月 9 日（周四）

- 主任务：精读 Attention 论文剩余部分 + 为什么 Decoder-only 胜出
- 要做：
  - 精读 Attention Is All You Need 的 Section 5（Training）和 Section 6（Results）
  - 重点理解从推理视角看的架构演化：
    - Encoder-Decoder → Encoder-only (BERT) → **Decoder-only (GPT)**
    - 为什么 Decoder-only 在推理上最高效：
      - 无 Cross-Attention 开销
      - KV Cache 结构更简单
      - 自回归生成天然匹配 Causal Mask
  - 写 1 张高密度压缩卡：Transformer 架构 + 推理视角分析
- 当天输出：
  - 结构化论文笔记
  - 1 张"三种架构的推理特性对比"卡（Enc-Dec / Enc-only / Dec-only）
- 完成标准：
  - 能闭卷解释为什么 Decoder-only 在推理上胜出
    最关键的一点是，Decoder-only的数据获取成本最低

### 4 月 10 日（周五）

- 主任务：**GPT-2 推理 Profiling（提前到 W2 做！）**
- 要做：
  - 用 `torch.profiler` 对 GPT-2 `model.generate()` 做完整 profiling，导出 Chrome trace
  - 用 `torch.cuda.Event` 分别测量 prefill 和 decode 阶段耗时：
    - 输入 prompt 长度: 32/64/128/256 tokens
    - 生成长度: 64 tokens
    - 记录 prefill_time 和 decode_time
  - 做初步分析：
    - prefill 和 decode 的耗时比例
    - **验证本周假设：** Decode 阶段是否占总耗时 > 80%？
- 当天输出：
  - `W2/gpt2_profiling.py`
  - 1 张 profiling 结果表（prompt_len × prefill_time × decode_time × decode_ratio）
  - 本周假设验证结论
- 完成标准：
  - 有具体的 prefill/decode 耗时数据（ms 级别）
  - 假设有明确结论（如果 Decode < 80%，分析原因）

### 4 月 11 日（周六）

- 主任务：闭卷回忆 + 手写显存计算器
- 要做：
  - 闭卷回答：
    - Decoder Block 的数据流？
    - Pre-Norm vs Post-Norm 区别？
    - Causal Mask 和 KV Cache 的关系？
  - **手写 `memory_calculator.py`（提前到 W2 做！）：**
    - 输入：参数量、层数、head 数、head_dim、seq_len、batch_size、dtype
    - 输出：模型权重显存 + KV Cache 显存 + 激活显存
    - KV Cache 公式：`2 × num_layers × batch × seq_len × num_heads × head_dim × dtype_bytes`
  - 用 GPT-2 配置验证计算器估算 vs `torch.cuda.max_memory_allocated()` 实际值
- 当天输出：
  - `W2/memory_calculator.py`
  - 1 份回忆记录
  - 计算器估算 vs 实测对比表
- 完成标准：
  - 计算器估算与实际显存误差 < 20%
  - 能手算 GPT-2 的模型显存 + KV Cache 显存

### 4 月 12 日（周日）

- 主任务：知乎 / 公众号输出
- 要做：
  - 写一篇关于"为什么现代 LLM 都用 Decoder-only：从推理效率说起"的文章
  - 重点讲：Causal Mask → KV Cache → Decoder-only 的推理优势
- 当天输出：
  - 1 篇外部内容
- 完成标准：
  - 内容能让非研究者理解为什么 GPT 架构胜出

### 4 月 13 日（周一）

- 主任务：W2 收口 + 准备进入 GPU 深度分析
- 要做：
  - 整理 W2 mini memo：
    - Decoder Block 实现
    - Profiling 数据
    - 显存计算器验证
  - 写下 W3 假设：
    - 假设：batch_size=1 时 GPU 利用率 < 30%
  - 准备 W3 所需工具：`nvidia-smi dmon`
- 当天输出：
  - `W2/decoder_profiling_memo.md`
  - W3 假设文档
- 完成标准：
  - W2 所有数据整理完毕
  - 你知道 W3 从哪里开始

---

## W3：4/14 - 4/20

> **本周假设：** batch_size=1 时 GPT-2 推理的 GPU 利用率 < 30%，因为 decode 阶段是 GEMV（memory-bound）
>
> **本周定位：** GPU 计算理论 + GEMM/GEMV 实验 + **开始手写 GPT-2**

### 4 月 14 日（周二）

- 主任务：GPU 内存层级 + Roofline Model 理论
- 要做：
  - 学习 GPU 内存层级：HBM → L2 Cache → SRAM (Shared Memory) → Register
    - RTX 4090: HBM 带宽 ~1 TB/s, SRAM ~100 TB/s
  - 理解 Roofline Model：
    - Arithmetic Intensity (AI) = FLOPs / Bytes
    - Compute-bound: AI > 硬件的 ops:byte 比
    - Memory-bound: AI < 硬件的 ops:byte 比
  - 分析 Prefill vs Decode 的瓶颈差异：
    - Prefill: GEMM（大矩阵乘法）→ 倾向 compute-bound
    - Decode: GEMV（矩阵向量乘法）→ 倾向 memory-bound
  - **用 Roofline 框架解释 W2 的 profiling 数据：** 为什么 decode 慢？
- 当天输出：
  - 1 张 GPU 内存层级图
  - 1 页 Roofline Model 笔记（含 Prefill/Decode 分析）
  - 用 Roofline 解释 W2 profiling 数据的笔记
- 完成标准：
  - 能解释 compute-bound vs memory-bound 的区别
  - 能用 Roofline 框架解释为什么 decode 阶段 GPU 利用率低

### 4 月 15 日（周三）

- 主任务：GEMM vs GEMV Benchmark + GPU 利用率监控
- 要做：
  - 手写 GEMM vs GEMV benchmark（用 `torch.matmul`）：
    - GEMM: (M, K) × (K, N)，M=N=K=1024/2048/4096
    - GEMV: (M, K) × (K, 1)，M=K=1024/2048/4096
    - 测量吞吐量（TFLOPS）和耗时
  - 用 `nvidia-smi dmon` 监控 GPT-2 推理过程的 GPU 利用率
  - 画出时间-GPU 利用率曲线
  - **验证本周假设：** batch_size=1 时 GPU 利用率是否 < 30%？
- 当天输出：
  - `W3/gemm_vs_gemv_benchmark.py`
  - 1 张 GEMM vs GEMV 吞吐量对比图
  - 1 张 GPU 利用率时间曲线图
  - 本周假设验证结论
- 完成标准：
  - 能用数字证明 GEMV 比 GEMM 的 GPU 利用率低
  - 假设验证有明确结论

### 4 月 16 日（周四）

- 主任务：精读 GPU 计算材料 + 手算 FLOPs
- 要做：
  - 精读相关博客/教程（推荐：Efficient Transformers Survey, GPU 内存层级讲解）
  - 手算 GPT-2 (124M) 的 forward pass FLOPs：
    - 每层 Attention: `2 × 2 × seq_len × d_model × d_model`（QKV projection + output projection）
    - 每层 Attention score: `2 × seq_len × seq_len × d_model`
    - 每层 FFN: `2 × 2 × seq_len × d_model × d_ff`
    - 总计 ≈ 6ND 近似验证
  - 计算 GPT-2 推理的 Arithmetic Intensity，判断它在 RTX 4090 上是 compute-bound 还是 memory-bound
- 当天输出：
  - 1 份 FLOPs 手算过程
  - 结构化笔记
- 完成标准：
  - 能手算 forward FLOPs
  - 能判断给定配置在给定硬件上的瓶颈类型

### 4 月 17 日（周五）

- 主任务：**开始手写 minimal GPT-2 — CausalSelfAttention**
- 要做：
  - 手写 `CausalSelfAttention`（GPT-2 风格）：
    - 合并 QKV projection 为一个 Linear（高效）
    - 内置 Causal Mask
    - **预留 KV Cache 接口**：`forward(x, past_kv=None)` → `(output, present_kv)`
    - 输入：(batch=2, seq_len=10, d_model=768), num_heads=12
    - 验证输出 shape: (2, 10, 768)
  - 手写 GPT Block = LayerNorm → CausalSelfAttention → Residual → LayerNorm → FFN → Residual
- 当天输出：
  - `W3/causal_self_attention.py`（含 KV Cache 接口）
  - `W3/gpt_block.py`
- 完成标准：
  - CausalSelfAttention 支持 `past_kv` 参数（即使本周还不实现 cache 逻辑）
  - GPT Block forward 跑通

### 4 月 18 日（周六）

- 主任务：闭卷回忆 + 显存手算练习
- 要做：
  - 闭卷回答：
    - GPU 内存层级有哪些？各自带宽量级？
    - Prefill 是 compute-bound 还是 memory-bound？为什么？
    - KV Cache 显存公式是什么？
    - GEMM 和 GEMV 的区别？为什么跟 prefill/decode 相关？
  - 手算题：
    - LLaMA-7B（32 层, d=4096, heads=32, head_dim=128）在 batch=8, seq_len=2048, FP16 下的 KV Cache 显存
  - 写 W3 mini memo
- 当天输出：
  - 1 份闭卷回忆记录
  - 1 道手算题完整过程
  - `W3/gpu_profiling_memo.md`
- 完成标准：
  - 手算 KV Cache 显存答案正确
  - 能说出至少 3 个影响 LLM 推理效率的关键因素

### 4 月 19 日（周日）

- 主任务：写外部输出
- 要做：
  - 写一篇文章，主题建议："LLM 推理为什么慢？从 GPU 内存层级到 Prefill-Decode 瓶颈分析"
  - 重点讲：compute-bound vs memory-bound、GEMM vs GEMV、KV Cache 显存增长
- 当天输出：
  - 1 篇对外内容
- 完成标准：
  - 能让非本领域读者理解 LLM 推理的瓶颈在哪

### 4 月 20 日（周一）

- 主任务：**组装 minimal GPT-2 + 加载预训练权重**
- 要做：
  - 组装完整 GPT-2 模型：token embedding + position embedding + N × GPT Block + LayerNorm + lm_head
  - 加载 HuggingFace GPT-2 (124M) 预训练权重到自己的实现中
    - 需要处理权重名映射
  - 实现 greedy decoding（无 KV Cache 版本）
  - **验证：** 自己实现的 GPT-2 生成输出与 HuggingFace 官方输出完全一致
- 当天输出：
  - `W3/minimal_gpt2.py`
  - 权重加载和输出对比记录
- 完成标准：
  - 加载预训练权重后，生成结果与 HuggingFace 输出一致（`torch.allclose` on logits）

---

## W4：4/21 - 4/27（KV Cache 实现 + 硬数据 + 4 月收口）

> **本周假设：** KV Cache 在 seq_len > 256 时能让 GPT-2 decode 加速 > 2x
>
> **本周定位：** 这是 4 月最关键的一周。你要拿到 KV Cache 的硬 benchmark，而不只是理论理解。

### 4 月 21 日（周二）

- 主任务：**实现 KV Cache decode**
- 要做：
  - 在 `CausalSelfAttention` 中实现完整的 KV Cache 逻辑：
    - Prefill 阶段：处理完整 prompt，生成初始 KV Cache
    - Decode 阶段：每步只输入新 token 的 embedding，拼接到已有 KV Cache
    - `forward(x, past_kv=None)` → `(output, present_kv)`
    - present_kv shape: `(2, batch, num_heads, seq_len, head_dim)`（K 和 V 各一个）
  - 验证：有 KV Cache vs 无 KV Cache 的 decode 输出完全一致（`torch.allclose`）
  - 实现 `generate_with_cache(prompt_ids, max_new_tokens)` 函数
- 当天输出：
  - `W4/kv_cache_attention.py`
  - `W4/generate_with_cache.py`
  - torch.allclose 验证记录
- 完成标准：
  - KV Cache 版和无 Cache 版生成的 token 序列完全一致
  - Cache 版每步只处理 1 个 token 的 Q

### 4 月 22 日（周三）

- 主任务：**KV Cache Benchmark — 拿硬数据**
- 要做：
  - 对比有无 KV Cache 的逐 token 生成速度：
    - 测量生成 64/128/256 tokens 的总耗时
    - 分别记录 prefill_time 和 per_token_decode_time
    - 画出 speedup 曲线（generated_tokens vs speedup_ratio）
  - 不同 seq_len 下的 KV Cache 对比：
    - prompt_len = 32/64/128/256/512
    - 生成 128 tokens
    - 记录有无 Cache 的总耗时
  - 用 `torch.cuda.max_memory_allocated()` 对比显存：
    - 无 Cache：每步重算所有 KV
    - 有 Cache：额外的 Cache 显存 vs 省下的重算开销
  - **验证本周假设：** KV Cache 在 seq_len > 256 时是否加速 > 2x？
- 当天输出：
  - `W4/kv_cache_benchmark.py`
  - 1 张 speedup 曲线图
  - 1 张显存对比表
  - 本周假设验证结论（含具体数字）
- 完成标准：
  - 有具体的 speedup 数字（不是"明显更快"）
  - 假设有明确结论

### 4 月 23 日（周四）

- 主任务：精读 GPT-2 论文 + KV Cache 显存缩放分析
- 要做：
  - 精读 GPT-2 论文 Language Models are Unsupervised Multitask Learners
  - 重点关注从推理视角看的架构选择：Pre-Norm、Decoder-only
  - KV Cache 显存缩放深度分析：
    - 手算：KV Cache 显存 = f(num_layers, batch, seq_len, num_heads, head_dim, dtype)
    - 画出 batch_size × seq_len 的 KV Cache 显存热力图
    - 找到你的 GPU（24GB 4090）在不同模型配置下的 OOM 边界
  - 思考题：KV Cache 显存增长是线性的（seq_len），但 Attention 计算是 O(n²) —— 这两个瓶颈在什么条件下哪个先碰到？
- 当天输出：
  - 结构化论文笔记
  - 1 张 KV Cache 显存热力图
  - 1 份 OOM 边界分析
- 完成标准：
  - 能回答"24GB 显存，GPT-2 最多能 serve 多大 batch × seq_len"

### 4 月 24 日（周五）

- 主任务：4 月综合 Inference Benchmark
- 要做：
  - 在自己手写的 GPT-2 上做综合推理 benchmark：
    - seq_len vs 延迟（128/256/512/1024）
    - batch_size vs 吞吐量（1/2/4/8）
    - With/Without KV Cache 的联合对比
    - 峰值显存记录
  - 所有结果汇总到一个脚本和一张表
  - 写 4 月 mini report 初稿
- 当天输出：
  - `W4/comprehensive_inference_benchmark.py`
  - 1 张综合性能表（seq_len × batch × cache × latency × throughput × memory）
  - `W4/april_report.md` 初稿
- 完成标准：
  - report 里有具体的 benchmark 数字
  - 能用一张表概括 4 月所有关键实验结果

### 4 月 25 日（周六）

- 主任务：闭卷回忆 + 面试题整理
- 要做：
  - 闭卷回答 4 月核心问题（不看任何资料）：
    - Attention 的时间/空间复杂度？
    - Multi-Head Attention 为什么不增加参数量？
    - Causal Mask 和 KV Cache 的关系？
    - Prefill 和 Decode 的瓶颈分别是什么？为什么？
    - KV Cache 加速了什么，代价是什么？
    - GEMM 和 GEMV 的区别？跟 Prefill/Decode 什么关系？
    - 24GB 显存的 GPU 能 serve GPT-2 多大的 batch × seq_len？
  - 整理 4 月面试题库（至少 7 道）
  - 画 1 张 4 月知识总图
- 当天输出：
  - 1 份回忆记录
  - 4 月面试题库（7+ 道，含技术版和直觉版答案）
  - 1 张 4 月知识总图
- 完成标准：
  - 能 5 分钟脱稿讲清 4 月学了什么
  - 面试题答案准确

### 4 月 26 日（周日）

- 主任务：写阶段性复盘 + 外部输出
- 要做：
  - 写一篇阶段复盘，主题："从手写 Attention 到 KV Cache 加速：我的 LLM Inference 学习第一个月"
  - 文中必须包含：
    - 至少 3 个具体的 benchmark 数字
    - KV Cache 的 speedup 实测数据
    - 一个你验证后发现"跟预期不一样"的发现
- 当天输出：
  - 1 篇阶段复盘
- 完成标准：
  - 文中有理论理解、实验数据、和至少一个 surprise

### 4 月 27 日（周一）

- 主任务：4 月 mini project 收口
- 要做：
  - 重跑关键实验，确保结果可复现
  - 完善 `W4/april_report.md`
  - 补 README、代码注释
  - 做本月核心提交
  - 写下 5 月待验证假设：
    - 假设 1：Flash Attention 的 tiling 能在 seq_len > 2048 时把峰值显存从 O(n²) 降到 O(n)
    - 假设 2：GQA（4 个 KV head）的 KV Cache 大小是 MHA（32 个 KV head）的 1/8，但 perplexity 损失 < 1%
- 当天输出：
  - 完善后的 `W4/april_report.md`
  - `2026_05_hypotheses.md`
- 完成标准：
  - 4 月成果完整、代码可运行
  - 你知道 5 月要做什么

---

## W5（4/28 - 4/30，提前进入 5 月主题）

> **本周任务：** 开始 Flash Attention 的对照实验准备

### 4 月 28 日（周二）

- 主任务：实现 naive attention + 显存测量
- 要做：
  - 实现独立的 naive attention 函数：
    - 标准 `softmax(QK^T / √d_k)V`
    - 显式分配 O(n²) 的 attention weight 矩阵
  - 测量不同 seq_len 下的峰值显存：
    - seq_len = 512/1024/2048/4096/8192
    - 用 `torch.cuda.max_memory_allocated()` 和 `torch.cuda.reset_peak_memory_stats()`
    - 记录何时 OOM
  - 这是 Flash Attention 对比的 baseline
- 当天输出：
  - `W5/naive_attention.py`
  - 1 张 seq_len vs 峰值显存曲线（标注 OOM 点）
- 完成标准：
  - 有 naive attention 在不同 seq_len 的显存数据
  - 观察到 O(n²) 显存增长

### 4 月 29 日（周三）

- 主任务：开始 Flash Attention tiling 实现
- 要做：
  - 学习 online softmax 算法：
    - 维护 running max 和 running sum
    - 每个 block 更新后仍然得到正确的 softmax
  - 开始实现简化版 Flash Attention（Python 层面 tiling）：
    - 将 Q 按 block_size 分块
    - 将 K, V 按 block_size 分块
    - 逐块计算并用 online softmax 合并
  - 即使没写完也记录进度
- 当天输出：
  - `W5/online_softmax.py`
  - `W5/flash_attention_tiling.py`（可以是半成品）
  - 1 份 online softmax 推导笔记
- 完成标准：
  - online softmax 函数验证正确（与标准 softmax 输出 torch.allclose）

### 4 月 30 日（周四）

- 主任务：4 月总复盘 + 5 月开题
- 要做：
  - 写 4 月总结：
    - 4 个假设的验证结果汇总
    - 关键 benchmark 数字汇总
    - 最大的 surprise 是什么
  - 明确 5 月目标：Flash Attention 完整实现 + MQA/GQA + KV Cache 显存边界
  - 列出 5 月需要精读的论文：
    - FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（Tri Dao）
    - GQA: Training Generalized Multi-Query Transformer Models（Ainslie et al.）
- 当天输出：
  - `2026_04_monthly_review.md`（含假设验证汇总）
  - `2026_05_reading_list.md`
- 完成标准：
  - 你知道 5 月要验证什么
  - 你有一份明确的 5 月阅读和实验计划

---

## 4 月结束时你应该拿到什么

- **一个从零手写的 Decoder-only GPT-2**（能加载预训练权重、生成与官方一致的输出）
- **一个 KV Cache 实现 + 有无 Cache 的 speedup 实测数据**
- 一个 GPT-2 推理 profiling（prefill/decode 分离计时，具体 ms 数字）
- 一个手写的显存计算器（模型显存 + KV Cache 显存）
- 一组 GEMM vs GEMV 吞吐量对比数据
- 一组 GPU 利用率监控数据
- naive attention 在不同 seq_len 下的显存 baseline（Flash Attention 对比准备）
- **4 个"假设→验证"闭环记录**
- 至少 4 份 mini memo
- 至少 7 道 LLM Inference 面试题
- 1 份 4 月 mini report（有具体 benchmark 数字）

---

## 4 月执行原则

- 每天只追一个最小问题
- 每周至少留下一张图，而不是只留结论
- 每次做对照时，只改一个变量
- 每次结果整理时，都同时写：
  - 这个组件的推理耗时是多少
  - 它消耗了多少显存
  - 瓶颈在哪（compute-bound 还是 memory-bound）
- **推理视角贯穿：** 每天的工作都要回到"这对推理意味着什么"
- **假设驱动：** 每周有一个假设，用实验验证，不做无目的的学习
- **benchmark 数字化：** 所有性能对比必须出具体数字（显存 X GB、速度 X ms、吞吐量 X tokens/sec），不允许用"明显更快""显存更少"等模糊描述
- **不写 Encoder：** 4 月不碰 Encoder、Cross-Attention、Encoder-Decoder 架构、翻译任务。所有精力集中在 Decoder-only 推理链路上
