# 2026 年 4 月逐日执行版（LLM Inference 系统优化）

> 适用前提：
> - 本地有一台 4090 笔记本
> - 4 月主线是 `Transformer 从零复现 + GPU 计算基础 + 推理 Profiling`
> - 这份文档服务于新的 `LLM Inference 系统优化` 版 Phase A

---

## 4 月总目标

- 搭建稳定可用的本地 LLM Inference 研究实验环境
- 从零手写 Transformer 核心组件（ScaledDotProductAttention、MultiHeadAttention、FFN、PE、LayerNorm）
- 理解 GPU 内存层级和推理瓶颈（compute-bound vs memory-bound、GEMM vs GEMV）
- 做第一次 GPT-2 model.generate() 全流程 profiling（prefill/decode 分离）
- 手写显存计算器，能手算模型显存 + KV Cache 显存
- 形成第一套推理性能 baseline 数据

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

---

## W1：4/1 - 4/6

### 4 月 1 日（周三）

- 主任务：搭起 LLM Inference 研究实验地基
- 要做：
  - 整理本地 Python / PyTorch / CUDA 环境信息
  - 验证 GPU 可用：`torch.cuda.is_available()`、显存大小、CUDA 版本
  - 配置 `torch.profiler` 和 `wandb` 基本设置
  - 确定 4 月用的 baseline 模型（HuggingFace GPT-2）
- 当天输出：
  - 1 份环境记录（Python/PyTorch/CUDA 版本、GPU 型号和显存）
  - 1 份 4 月实验计划确认
- 完成标准：
  - 知道自己的 GPU 显存、CUDA 版本、PyTorch 版本
  - torch.profiler 能跑通最小 demo

### 4 月 2 日（周四）

- 主任务：精读 Attention Is All You Need — Attention 部分
- 要做：
  - 精读 Section 3.2（Scaled Dot-Product Attention）和 Section 3.2.2（Multi-Head Attention）
  - 推导 Attention 公式：`Attention(Q,K,V) = softmax(QK^T / √d_k)V`
  - 理解为什么要除以 √d_k（softmax 梯度饱和问题）
  - 写 1 页 Attention 数学推导卡
- 当天输出：
  - 结构化论文笔记
  - 1 张 Attention 计算流程与数学推导卡
- 完成标准：
  - 能闭卷写出 Attention 的完整公式
  - 能解释 Multi-Head Attention 的设计动机（多子空间 + 参数量不变）

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

- 主任务：轻量复盘 + 推理视角思考
- 要做：
  - 整理前 3 天结果
  - 从推理视角分析 Attention 的计算复杂度：
    - 时间复杂度 O(n²d)，空间复杂度 O(n²)
    - 为什么 seq_len 增长时 Attention 成为瓶颈
  - 手算一个具体例子的 Attention 显存：
    - d_model=512, num_heads=8, seq_len=1024, batch=4, FP16
    - QKV 矩阵 + attention weights 的显存
- 当天输出：
  - 1 页总结笔记（含推理视角分析）
  - 1 道手算显存题（含完整过程）
- 完成标准：
  - 能说出 Attention 的 O(n²) 为什么是推理的主要瓶颈
  - 能手算给定配置下 Attention 的显存消耗

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

### 4 月 7 日（周二）

- 主任务：手写 LayerNorm + EncoderLayer
- 要做：
  - 手写 LayerNorm：验证与 `torch.nn.LayerNorm` 输出一致（`torch.allclose`）
  - 手写 EncoderLayer：MultiHeadAttention + FFN + Add&Norm
    - 输入：(batch=2, seq_len=10, d_model=512)
    - 验证输出 shape
  - 理解 Post-Norm（原始 Transformer）的结构
- 当天输出：
  - `W2/layer_norm.py`
  - `W2/encoder_layer.py`
  - 1 张 EncoderLayer 数据流图
- 完成标准：
  - EncoderLayer 跑通，输出 shape 正确
  - 能画出 Attention → Add&Norm → FFN → Add&Norm 的数据流

### 4 月 8 日（周三）

- 主任务：手写 DecoderLayer + Causal Mask
- 要做：
  - 手写 DecoderLayer：Masked Self-Attention + Cross-Attention + FFN
  - 实现 Causal Mask（下三角矩阵）
  - 验证 Causal Mask 后 attention weights 的上三角为 0
  - 思考：Causal Mask 对推理的意义（自回归生成的前提）
- 当天输出：
  - `W2/decoder_layer.py`
  - `W2/causal_mask.py`
  - 1 段观察结论
- 完成标准：
  - DecoderLayer 跑通
  - 能解释"没有 Causal Mask 为什么不能做自回归生成"

### 4 月 9 日（周四）

- 主任务：组装完整 Transformer + 精读剩余部分
- 要做：
  - 把所有组件组装成完整 Transformer（Encoder + Decoder）
  - 精读 Attention Is All You Need 的 Section 5（Training）和 Section 6（Results）
  - 写 1 张高密度压缩卡：Transformer 完整架构
- 当天输出：
  - `W2/transformer_model.py`
  - 结构化论文笔记
  - 1 张 Transformer 完整架构卡
- 完成标准：
  - 完整 Transformer forward 跑通
  - 能闭卷画出 Transformer 的完整数据流

### 4 月 10 日（周五）

- 主任务：小型翻译训练 + 整理 W2 结果
- 要做：
  - 用 Multi30k (de→en) 或类似小型数据集跑通训练循环
  - 用 wandb 记录 training loss 曲线
  - 对比自己实现的 attention 与 `torch.nn.MultiheadAttention` 输出（`torch.allclose`）
  - 写 1 份 mini memo
- 当天输出：
  - `W2/train_translation.py`
  - 1 张 training loss 曲线图
  - `W2/transformer_memo.md`
- 完成标准：
  - 训练循环跑通，loss 有下降趋势
  - memo 里有模型配置、训练结果、与官方实现的对比

### 4 月 11 日（周六）

- 主任务：闭卷回忆 + 推理视角分析
- 要做：
  - 闭卷画出完整 Transformer 架构图
  - 从推理视角分析：
    - Encoder-Decoder 架构中，哪些计算可以复用？
    - 为什么 Decoder-only（GPT）最终成为主流？（推理更简单、KV Cache 更高效）
  - 手算：一个 6 层 Transformer（d_model=512, d_ff=2048）的参数量
- 当天输出：
  - 1 份回忆记录
  - 1 道手算参数量题
- 完成标准：
  - 不看资料能讲 3 分钟 Transformer 架构
  - 手算参数量误差 < 5%

### 4 月 12 日（周日）

- 主任务：知乎 / 公众号输出
- 要做：
  - 写一篇关于"从零手写 Transformer：我学到了什么"的文章
  - 重点讲：哪些组件影响推理效率，为什么 Decoder-only 更适合推理
- 当天输出：
  - 1 篇外部内容
- 完成标准：
  - 内容能让非研究者理解 Transformer 的推理特性

### 4 月 13 日（周一）

- 主任务：准备进入 GPU 计算基础 + 推理 profiling
- 要做：
  - 用 pip 安装 HuggingFace transformers，加载 GPT-2 模型
  - 做一次最简 `model.generate()` 测试，确认推理跑通
  - 整理 GPU profiling 工具清单：
    - `torch.profiler`
    - `torch.cuda.Event`
    - `torch.cuda.max_memory_allocated()`
    - `nvidia-smi dmon`
  - 写下 W3 的 2 个待验证假设：
    - 假设 1：GPT-2 的 decode 阶段是 memory-bound
    - 假设 2：batch_size=1 时 GPU 利用率 < 30%
- 当天输出：
  - GPT-2 推理跑通记录
  - profiling 工具清单
  - W3 假设文档
- 完成标准：
  - GPT-2 model.generate() 跑通
  - 你知道 W3 的实验从哪里开始

---

## W3：4/14 - 4/20

### 4 月 14 日（周二）

- 主任务：GPU 内存层级 + Roofline Model 理论
- 要做：
  - 学习 GPU 内存层级：HBM → L2 Cache → SRAM (Shared Memory) → Register
    - RTX 4090: HBM 带宽 ~1 TB/s, SRAM ~100 TB/s
  - 理解 Roofline Model：
    - Arithmetic Intensity = FLOPs / Bytes
    - Compute-bound: AI > 硬件的 ops:byte 比
    - Memory-bound: AI < 硬件的 ops:byte 比
  - 分析 Prefill vs Decode 的瓶颈差异：
    - Prefill: GEMM（大矩阵乘法）→ 倾向 compute-bound
    - Decode: GEMV（矩阵向量乘法）→ 倾向 memory-bound
- 当天输出：
  - 1 张 GPU 内存层级图
  - 1 页 Roofline Model 笔记（含 Prefill/Decode 分析）
- 完成标准：
  - 能解释 compute-bound vs memory-bound 的区别
  - 能说出 Prefill 和 Decode 各自的瓶颈类型和原因

### 4 月 15 日（周三）

- 主任务：GPT-2 推理 Profiling
- 要做：
  - 用 `torch.profiler` 对 GPT-2 的 `model.generate()` 做完整 profiling
  - 导出 Chrome trace（`.json`），用 Chrome `chrome://tracing` 查看
  - 用 `torch.cuda.Event` 分别测量 prefill 和 decode 阶段耗时：
    - 输入 prompt 长度: 32/64/128/256 tokens
    - 生成长度: 64 tokens
    - 记录 prefill_time 和 decode_time
  - 做初步分析：prefill 和 decode 的耗时比例
- 当天输出：
  - `W3/gpt2_profiling.py`
  - 1 张 profiling 结果表（prompt_len × prefill_time × decode_time）
- 完成标准：
  - Chrome trace 能看到各 op 的耗时分布
  - 有具体的 prefill/decode 耗时数据（ms 级别）

### 4 月 16 日（周四）

- 主任务：手写显存计算器 + 精读 GPU 计算相关材料
- 要做：
  - 精读相关博客/教程（GPU 内存层级、LLM Inference 瓶颈分析）
  - 写 `memory_calculator.py`：
    - 输入：参数量、层数、head 数、head_dim、seq_len、batch_size、dtype
    - 输出：模型权重显存 + KV Cache 显存 + 激活显存
    - KV Cache 公式：`2 × num_layers × batch × seq_len × num_heads × head_dim × dtype_bytes`
  - 用 GPT-2 配置验证计算器的估算 vs `torch.cuda.max_memory_allocated()` 实际值
- 当天输出：
  - `W3/memory_calculator.py`
  - 1 张计算器输出 vs 实测显存对比表
  - 结构化笔记
- 完成标准：
  - 计算器估算与实际显存误差 < 20%
  - 能手算 GPT-2 的模型显存 + KV Cache 显存

### 4 月 17 日（周五）

- 主任务：GEMM vs GEMV Benchmark + GPU 利用率监控
- 要做：
  - 手写 GEMM vs GEMV benchmark（用 `torch.matmul`）：
    - GEMM: (M, K) × (K, N)，M=N=K=1024/2048/4096
    - GEMV: (M, K) × (K, 1)，M=K=1024/2048/4096
    - 测量吞吐量（TFLOPS）和耗时
  - 用 `nvidia-smi dmon` 监控 GPT-2 推理过程的 GPU 利用率
  - 画出时间-GPU利用率曲线
  - 验证 W3 开始时的假设
- 当天输出：
  - `W3/gemm_vs_gemv_benchmark.py`
  - 1 张 GEMM vs GEMV 吞吐量对比图
  - 1 张 GPU 利用率时间曲线图
- 完成标准：
  - 能用数字证明 GEMV 比 GEMM 的 GPU 利用率低
  - 假设验证有明确结论

### 4 月 18 日（周六）

- 主任务：闭卷回忆 + 显存手算练习
- 要做：
  - 闭卷回答以下问题：
    - GPU 内存层级有哪些？各自带宽量级？
    - Prefill 是 compute-bound 还是 memory-bound？为什么？
    - KV Cache 显存公式是什么？
  - 手算题：
    - LLaMA-7B（32 层, d=4096, heads=32, head_dim=128）在 batch=8, seq_len=2048, FP16 下的 KV Cache 显存
  - 总结 W3 所有 profiling 结果
- 当天输出：
  - 1 份闭卷回忆记录
  - 1 道手算题完整过程
  - 1 张 W3 推理 profiling 总结图
- 完成标准：
  - 能说出至少 3 个影响 LLM 推理效率的关键因素
  - 手算 KV Cache 显存答案正确

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

- 主任务：W3 收口 + 综合 benchmark
- 要做：
  - 把 W3 所有实验整合到统一格式：
    - profiling 结果（prefill/decode 耗时）
    - 显存计算器 vs 实测
    - GEMM vs GEMV 吞吐量
    - GPU 利用率
  - 做 1 次完整 GitHub 提交
  - 写 1 份 W3 mini memo
- 当天输出：
  - `W3/gpu_profiling_memo.md`
  - 综合 benchmark 结果表
- 完成标准：
  - W3 所有实验数据整理完毕
  - memo 里有问题、方法、结果、结论

---

## W4：4/21 - 4/27（4 月 Mini Project 收口）

### 4 月 21 日（周二）

- 主任务：Transformer 推理特性综合实验
- 要做：
  - 在自己手写的 Transformer 上做推理 profiling：
    - 测量不同 seq_len（128/256/512/1024）下的推理延迟
    - 测量不同 batch_size（1/2/4/8）下的吞吐量
    - 用 `torch.cuda.max_memory_allocated()` 记录峰值显存
  - 画出 seq_len vs 延迟 和 batch_size vs 吞吐量 曲线
- 当天输出：
  - `W4/transformer_inference_benchmark.py`
  - 2 张性能曲线图
- 完成标准：
  - 能用数字说明 seq_len 翻倍时延迟如何变化（O(n²) 验证）

### 4 月 22 日（周三）

- 主任务：Pre-Norm vs Post-Norm + Decoder-only 推理分析
- 要做：
  - 实现 Pre-Norm 版 Transformer Block（先 Norm 再 Attention/FFN）
  - 对比 Pre-Norm vs Post-Norm 的训练稳定性和推理速度
  - 写一份分析：为什么现代模型（GPT-2、LLaMA）都用 Pre-Norm
  - 分析 Decoder-only 相比 Encoder-Decoder 在推理上的优势
- 当天输出：
  - `W4/pre_norm_block.py`
  - 1 份 Pre-Norm vs Post-Norm 对比分析
- 完成标准：
  - 能解释 Pre-Norm 为什么更容易训练
  - 能解释 Decoder-only 为什么推理更高效

### 4 月 23 日（周四）

- 主任务：精读 GPT-2 论文 + KV Cache 理论准备
- 要做：
  - 精读 GPT-2 论文 Language Models are Unsupervised Multitask Learners
  - 重点关注：
    - Pre-Norm 的使用
    - Zero-shot 能力
    - Decoder-only 架构的推理特性
  - 理论学习 KV Cache 的工作原理：
    - 为什么可以缓存 K 和 V
    - Cache 如何随 token 生成增长
    - Cache 对显存的影响
- 当天输出：
  - 结构化论文笔记
  - 1 张 KV Cache 工作原理图
- 完成标准：
  - 能画出 KV Cache 在逐 token 生成中的作用流程
  - 能说出"有 KV Cache 时，每步只需计算新 token 的 Q，而非全部重算"

### 4 月 24 日（周五）

- 主任务：4 月 mini report 初稿
- 要做：
  - 整理 4 月所有实验结果：
    - Transformer 组件实现（W1-W2）
    - GPU profiling 和显存分析（W3）
    - Transformer 推理 benchmark（W4）
  - 写 4 月 mini report，按以下结构组织：
    - 背景与目标
    - 实验环境
    - 实现的组件
    - 推理性能分析结果
    - 关键发现
    - 下月计划
- 当天输出：
  - `W4/april_report.md` 初稿
- 完成标准：
  - report 里有具体的 benchmark 数字

### 4 月 25 日（周六）

- 主任务：闭卷回忆 + 面试题整理
- 要做：
  - 闭卷回答 4 月核心问题：
    - Attention 的时间/空间复杂度？
    - Multi-Head Attention 为什么不增加参数量？
    - Prefill 和 Decode 的瓶颈分别是什么？
    - KV Cache 为什么能加速推理？
    - GEMM 和 GEMV 的区别？
  - 整理 4 月面试题库（至少 5 道）
  - 画 1 张 4 月知识总图
- 当天输出：
  - 1 份回忆记录
  - 4 月面试题库
  - 1 张 4 月知识总图
- 完成标准：
  - 能 5 分钟脱稿讲清 4 月学了什么

### 4 月 26 日（周日）

- 主任务：写阶段性复盘 + 外部输出
- 要做：
  - 写一篇阶段复盘，主题："从手写 Attention 到理解 GPU 瓶颈：我的 LLM Inference 学习第一个月"
  - 可同步写知乎/公众号文章
- 当天输出：
  - 1 篇阶段复盘
- 完成标准：
  - 文中能同时写出理论理解和实验数据

### 4 月 27 日（周一）

- 主任务：4 月 mini project 收口
- 要做：
  - 重跑关键实验，确保结果可复现
  - 完善 `W4/april_report.md`
  - 补 README、代码注释
  - 做本月核心提交
  - 写下 5 月待验证假设：
    - 假设 1：KV Cache 能让 GPT-2 decode 加速 X 倍
    - 假设 2：Flash Attention 的 tiling 能在 seq_len > 2048 时显著减少峰值显存
- 当天输出：
  - 完善后的 `W4/april_report.md`
  - `2026_05_hypotheses.md`
- 完成标准：
  - 4 月成果完整、代码可运行
  - 你知道 5 月要做什么

---

## W5（4/28 - 4/30，提前进入 5 月主题）

### 4 月 28 日（周二）

- 主任务：开始 GPT-2 从零实现
- 要做：
  - 手写 CausalSelfAttention（含 Causal Mask + 为 KV Cache 预留接口）
  - 手写 GPT Block（Pre-Norm + CausalSelfAttention + FFN）
  - 验证单个 Block 的输出 shape
- 当天输出：
  - `W5/causal_self_attention.py`
  - `W5/gpt_block.py`
  - shape 验证记录
- 完成标准：
  - GPT Block forward 跑通

### 4 月 29 日（周三）

- 主任务：组装 minimal GPT-2 + 权重加载
- 要做：
  - 组装完整 GPT-2 模型（embedding + blocks + lm_head）
  - 尝试加载 HuggingFace GPT-2 预训练权重
  - 实现 greedy decoding，验证输出与官方一致
- 当天输出：
  - `W5/minimal_gpt2.py`
  - 权重加载和输出对比记录
- 完成标准：
  - 自己的 GPT-2 能加载预训练权重并生成文本

### 4 月 30 日（周四）

- 主任务：4 月总复盘 + 5 月开题
- 要做：
  - 写 4 月总结
  - 明确 5 月目标：KV Cache 实现 + Flash Attention 复现 + MQA/GQA
  - 列出 5 月需要精读的论文：
    - FlashAttention（Tri Dao）
    - GQA（Ainslie et al.）
    - GPT-2 论文补充阅读
- 当天输出：
  - `2026_04_monthly_review.md`
  - `2026_05_reading_list.md`
- 完成标准：
  - 你知道 5 月要验证什么
  - 你有一份明确的 5 月阅读和实验计划

---

## 4 月结束时你应该拿到什么

- 一个从零手写的完整 Transformer（ScaledDotProductAttention → MultiHeadAttention → FFN → PE → LayerNorm → EncoderLayer → DecoderLayer → Full Model）
- 一个跑通的 GPT-2 推理 profiling（prefill/decode 分离计时）
- 一个手写的显存计算器（模型显存 + KV Cache 显存 + 激活显存）
- 一组 GEMM vs GEMV 吞吐量对比数据
- 一组 GPU 利用率监控数据
- 一个 minimal GPT-2 的初始实现（W5 开头）
- 至少 4 份像样的实验记录或 mini memo
- 至少 5 道 LLM Inference 面试题
- 1 份 4 月 mini report

---

## 4 月执行原则

- 每天只追一个最小问题
- 每周至少留下一张图，而不是只留结论
- 每次做对照时，只改一个变量
- 每次结果整理时，都同时写：
  - 这个组件的推理耗时是多少
  - 它消耗了多少显存
  - 瓶颈在哪（compute-bound 还是 memory-bound）
- **推理视角贯穿**：即使是模型基础周（W1-W2），每天至少一个任务从 inference 角度思考
- **benchmark 数字化**：所有性能对比必须出具体数字（显存 X GB、速度 X ms、吞吐量 X tokens/sec），不允许用"明显更快""显存更少"等模糊描述
