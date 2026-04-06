# 提示词：生成 LLM Inference 方向 PhD 入门 20 周每日任务（理论 + 工程复现）

将以下提示词完整粘贴到 Claude Code 中使用。

---

```
你是一个 LLM Inference 方向的 PhD 导师，同时也是一个高性能计算的工程导师。请为我在本地仓库的根目录下生成 20 个周任务文件（W1/ 到 W20/，每个文件夹下一个 daily_tasks.md）。

## 我的背景

- CS/AI 本硕（Sheffield），准备申请 2027 Fall CS/AI PhD，研究方向：LLM Inference 系统优化
- 已完成 Transformer 论文（Attention Is All You Need）的深度阅读、19 道面试题整理、Scaled Dot-Product Attention 的 PyTorch 手写实现
- 数学基础一般，但执行力强，能坚持每日输出
- 每天有 4-6 小时学习时间
- 计算资源：RunPod/Vast.ai 租用 RTX 4090（24GB 显存）
- 工作流：VSCode Remote-SSH + tmux + wandb
- 每日在微信公众号/知乎发技术文章，每日更新 GitHub（github.com/Richard0307/llm-inference-study）
- Python 熟练，PyTorch 有基础但不算精通，CUDA 零基础

## 核心原则：理论与工程 1:1 交替

每周 7 天的分配原则：
- Day 1-2：论文精读 + 理论笔记
- Day 3-5：代码复现 + 工程实验（这是重点，至少 3 天在写代码和跑实验）
- Day 6：实验分析 + benchmark + 可视化结果
- Day 7：整理笔记、面试题库、公众号文章、Git push

**代码任务必须具体到函数级别**，不允许出现"实现 XXX"这种模糊描述。必须明确：输入维度、输出 shape、用什么库、验证方法、预期结果。

## 20 周总体规划

每周的主题、理论内容、和对应的工程复现项目如下：

### W1-W2：Transformer 深度阅读与从零复现
- **理论**：Attention Is All You Need 逐 section 精读；Scaled Dot-Product Attention 数学推导；Multi-Head Attention 设计动机；Positional Encoding（sin/cos）；LayerNorm vs BatchNorm；Causal Masking；Encoder-Decoder 架构
- **工程复现**：
  - 从零用 PyTorch 实现完整 Transformer（不用 nn.Transformer）
  - 必须手写的模块：ScaledDotProductAttention、MultiHeadAttention、PositionwiseFeedForward、PositionalEncoding、EncoderLayer、DecoderLayer
  - 每个模块写完后用 `assert` 验证输出 shape，例如：`assert output.shape == (batch=2, seq_len=10, d_model=512)`
  - 用一个小型机器翻译数据集（如 Multi30k de→en）跑通训练循环
  - 用 wandb 记录 training loss 曲线
  - 对比自己实现的 attention 输出与 `torch.nn.MultiheadAttention` 的输出是否一致（`torch.allclose`）

### W3：GPU 计算基础与 LLM 推理瓶颈分析
- **理论**：GPU 内存层级（HBM/L2/SRAM/Register）；Compute-bound vs Memory-bound；Roofline Model；Arithmetic Intensity；Prefill vs Decode 阶段的瓶颈差异；GEMM vs GEMV
- **工程实验**：
  - 用 `torch.profiler` 对一个 HuggingFace GPT-2 模型的 `model.generate()` 做完整 profiling，导出 Chrome trace
  - 分别测量 prefill 阶段和 decode 阶段的耗时，用 `torch.cuda.Event` 精确计时
  - 写一个 `memory_calculator.py`：输入模型参数量、层数、head 数、head_dim、seq_len、batch_size、dtype，输出模型显存 + KV Cache 显存 + 激活显存的估算值
  - 用 `nvidia-smi dmon` 监控推理过程中的 GPU 利用率，画出时间-利用率曲线
  - 手写一个简单的 GEMM vs GEMV benchmark（用 `torch.matmul`），验证不同矩阵维度下的吞吐量差异

### W4-W5：BERT + GPT-1/GPT-2 精读与 GPT-2 推理复现
- **理论**：BERT 的 MLM + NSP 预训练目标；Encoder-only vs Decoder-only 架构分化；GPT-1 的 generative pre-training 范式；GPT-2 的 zero-shot 能力；Pre-Norm vs Post-Norm；为什么 Decoder-only 最终胜出
- **工程复现**：
  - 从零实现一个 minimal GPT-2（参考 Karpathy 的 minGPT/nanoGPT）
  - 手写 CausalSelfAttention（含 causal mask）、GPT Block（Pre-Norm + Attention + FFN）、GPT Model（embedding + blocks + lm_head）
  - 加载 HuggingFace 的 GPT-2 预训练权重到自己的实现中，验证 `model.generate()` 输出与官方一致
  - 实现 greedy decoding 和 top-k sampling 两种生成策略
  - 手动实现 KV Cache 版本的 decode：对比有无 KV Cache 的逐 token 生成速度（用 `time.perf_counter` 测量），画出 speedup 曲线
  - 分析从 inference 视角看 BERT 双向注意力为什么无法用于自回归生成

### W6-W7：KV Cache 深度理解 + Flash Attention 原理与实验
- **理论**：KV Cache 完整工作流程；KV Cache 显存随 batch/seq_len 的增长分析；Flash Attention 的 IO-awareness 和 tiling 思想；Online Softmax 算法；MQA → GQA → MHA 的演进；Flash Attention 2 的改进点
- **工程复现**：
  - 实现 naive attention（标准 O(n²) 显存）并用 `torch.cuda.max_memory_allocated()` 测量峰值显存
  - 实现一个简化版 Flash Attention（Python 层面的 tiling 模拟）：将 Q/K/V 按 block_size 分块，逐块计算 attention 并用 online softmax 合并，验证输出与 naive attention 的 `torch.allclose`
  - 对比三者的峰值显存和速度：naive attention vs 自己的 tiling 版 vs `torch.nn.functional.scaled_dot_product_attention`（PyTorch 内置 Flash Attention）
  - 画出 seq_len vs 显存占用曲线（seq_len 从 512 到 8192）
  - 实现 MQA 和 GQA 的 attention：MQA 用 1 个 KV head，GQA 用 num_kv_heads=num_heads//4，对比三种方式在相同 seq_len 下的 KV Cache 大小
  - 写一个 KV Cache benchmark：固定模型大小，变化 batch_size 和 seq_len，测量何时 OOM，画出 "batch × seq_len 的显存边界图"

### W8-W9：GPT-3 + Scaling Laws + Chinchilla
- **理论**：GPT-3 的 In-Context Learning 机制；few-shot/one-shot/zero-shot 的区别；Scaling Laws（Kaplan）三条核心曲线（参数/数据/计算 vs loss）；Chinchilla 修正；Compute-optimal 训练策略对推理成本的影响
- **工程实验**：
  - 在 nanoGPT 上跑一个 mini scaling laws 实验：训练 3-5 个不同大小的模型（如 1M/5M/10M/50M 参数），固定数据集（如 OpenWebText 子集），用 wandb 画出 参数量 vs validation loss 曲线
  - 测量不同模型大小的推理延迟（固定 seq_len=256, batch=1），画出 参数量 vs 推理延迟 和 参数量 vs 吞吐量（tokens/sec）曲线
  - 计算 Chinchilla-optimal 下训练一个 7B 模型需要多少 tokens，与 LLaMA 的实际训练数据量对比
  - 写一个 `flops_calculator.py`：输入模型配置（layers, hidden_dim, seq_len, batch），输出 forward pass 的 FLOPs 估算（用 6ND 近似公式 + 更精确的逐层计算）
  - 分析："同样的训练预算，训一个大模型还是训一个小模型久一点，哪个推理更划算？"——用数字说话

### W10-W11：量化（Quantization）全面理解与实战
- **理论**：浮点数表示（FP32/FP16/BF16/INT8/INT4/FP8）；对称量化 vs 非对称量化；PTQ vs QAT；LLM.int8() 的混合精度分解和 outlier 检测；GPTQ 的 OBQ/Hessian-based 逐列量化；AWQ 的 salient channels 和 activation-aware 缩放；SmoothQuant 的 per-channel smoothing 数学变换
- **工程实战**：
  - 手写一个最简量化函数：`quantize_tensor(x, bits=8)` 和 `dequantize_tensor(x_q, scale, zero_point)`，验证 quantize→dequantize 的重建误差
  - 用 `bitsandbytes` 将 GPT-2（或 LLaMA-7B）分别加载为 FP16 / INT8 / INT4（NF4），对比显存占用、perplexity、和生成速度
  - 用 `auto-gptq` 对一个 7B 模型做 GPTQ 4-bit 量化，记录量化前后的 perplexity 变化和推理速度
  - 用 `awq` 库做 AWQ 量化，对比 GPTQ vs AWQ 在相同 bit-width 下的质量差异
  - 写一个量化 benchmark 脚本 `quantization_benchmark.py`：统一测试框架，输出一张表格：方法 × 精度 × 显存 × perplexity × tokens/sec
  - 可视化权重分布：画出量化前后某一层 weight 的直方图，观察 outlier 分布

### W12-W13：LLaMA 系列现代架构与组件级复现
- **理论**：LLaMA 1 & 2 的架构改进全景（RoPE、RMSNorm、SwiGLU、GQA、Pre-Norm）；RoPE 的旋转矩阵推导和相对位置编码性质；RMSNorm vs LayerNorm 的计算节省；SwiGLU 的 gating 机制；LLaMA 如何体现 Chinchilla Scaling Laws
- **工程复现**：
  - 从零手写 RMSNorm：`class RMSNorm(nn.Module)`，验证与 HuggingFace LlamaRMSNorm 输出一致
  - 从零手写 RoPE：实现 `rotary_embedding(x, seq_len)` 和 `apply_rotary_pos_emb(q, k, cos, sin)`，验证位置旋转后内积只依赖相对位置差
  - 从零手写 SwiGLU：`class SwiGLU(nn.Module)`，对比 SwiGLU vs ReLU vs GELU 的前向传播速度和梯度分布
  - 组装一个 minimal LLaMA：用以上组件 + GQA attention 搭建一个完整的 LLaMA-style 模型
  - 加载 HuggingFace 的 LLaMA-2-7B 权重到自己实现中（至少加载一层并验证输出一致）
  - 对比 LLaMA 和 GPT-2 的推理效率：相同参数量下，RoPE vs Learned PE、RMSNorm vs LayerNorm、SwiGLU vs GELU 各带来多少推理加速（用 `torch.cuda.Event` 测量）

### W14-W15：推理服务系统——vLLM 部署与源码分析
- **理论**：Static Batching 的问题；Continuous Batching / iteration-level scheduling（Orca）；PagedAttention 的虚拟内存思想；vLLM 的 Block Manager 分配机制；Prefill-Decode Disaggregation 的动机与设计；SLO（TTFT / TPOT / TPS）指标
- **工程实战**：
  - 在 RunPod 上用 vLLM 部署 LLaMA-2-7B（或 Qwen-7B），用 OpenAI-compatible API 接口测试
  - 写一个 `serving_benchmark.py`：用不同的 concurrency（1/4/8/16/32 并发请求）和 max_tokens（64/256/1024）测试，记录 TTFT、TPOT、吞吐量（tokens/sec），画出性能曲线
  - 对比 vLLM vs HuggingFace `pipeline("text-generation")` 在相同模型、相同请求下的吞吐量差异
  - 阅读 vLLM 源码的关键模块：
    - `vllm/core/block_manager.py`：理解 Block 分配和换入换出逻辑
    - `vllm/core/scheduler.py`：理解 continuous batching 的调度决策
    - 用自己的话画出 vLLM 处理一个请求的完整生命周期流程图
  - 实现一个极简版 continuous batching 模拟器（纯 Python，不需要 GPU）：模拟多个请求到达，按 iteration-level 调度，统计平均延迟和吞吐量
  - 调整 vLLM 的 `gpu_memory_utilization` 参数（0.5/0.7/0.9），观察 KV Cache 可用 block 数量和最大并发处理能力的变化

### W16-W17：LoRA + QLoRA 微调实战
- **理论**：LoRA 的低秩分解原理（W = W₀ + BA）；为什么微调时 intrinsic dimension 很低；Rank r 的选择对效果的影响；QLoRA 的三个关键技术（NF4 量化 + Double Quantization + Paged Optimizers）；Adapter merge 机制及推理开销
- **工程实战**：
  - 手写一个 minimal LoRA 层：`class LoRALinear(nn.Module)`，包含 frozen 的 `W₀` 和 trainable 的 `A`、`B`（rank=4），验证 forward 输出为 `W₀x + BAx`
  - 在 RTX 4090 上用 `peft` + `trl` 对 LLaMA-2-7B 做 QLoRA 微调（选一个下游任务如 Alpaca 指令微调或某个分类任务）
  - 用 wandb 记录 training loss、eval loss、GPU 显存占用
  - 对比不同 rank（r=4/8/16/64）的训练速度和最终效果
  - 对比微调方式的显存占用：Full Fine-Tuning vs LoRA vs QLoRA（同一模型，用 `torch.cuda.max_memory_allocated()` 记录）
  - 测试 adapter merge：将 LoRA adapter merge 回主模型，对比 merge 后的推理速度 vs 动态加载 adapter 的推理速度
  - 将微调后的模型部署到 vLLM 上，验证端到端的 微调→部署 流程

### W18-W19：投机解码 + 高级推理优化
- **理论**：Speculative Decoding 的核心思想（draft-then-verify）；rejection sampling 的数学保证（输出分布不变）；Medusa 的多头并行草稿机制；EAGLE 的特征级投机；Tensor Parallelism vs Pipeline Parallelism 的适用场景；Prefill-Decode Disaggregation 的系统设计
- **工程复现**：
  - 实现一个完整的 Speculative Decoding demo：
    - Draft model：GPT-2 small (124M)
    - Target model：GPT-2 large (774M)
    - 实现 `speculative_decode(prompt, draft_model, target_model, gamma=4)` 函数
    - 实现 rejection sampling 验证逻辑
    - 测量 acceptance rate 在不同 gamma（2/4/8）下的变化
    - 对比 speculative decoding vs 普通 autoregressive decoding 的实际 wall-clock speedup
  - 用 vLLM 开启 speculative decoding（如果支持），对比开启前后的吞吐量
  - 实现一个简单的 Tensor Parallelism 模拟：将一个线性层的 weight 按列切分到 2 个"虚拟 GPU"（用两个 tensor），前向传播后 all-gather 合并，验证输出与单 GPU 一致
  - 画出 LLM Inference 优化技术全景图（手绘或用代码生成）：涵盖量化、KV Cache 优化、注意力优化、解码策略、并行策略、serving 系统六大类
  - 写一个 `inference_optimization_comparison.py`：在同一个模型上测试 FP16 baseline / INT8 量化 / INT4 量化 / Speculative Decoding 的速度和质量对比

### W20：综合回顾 + PhD 申请材料 + Phase B 规划
- **理论**：整合 19 周的知识，形成 LLM Inference 知识图谱
- **工程总结**：
  - 整理所有实验结果，生成一份完整的 benchmark 报告（Markdown 或 HTML）
  - 确保 GitHub 仓库的 README 完整、代码可运行、有清晰的 reproduction instructions
  - 回顾所有代码，补充注释和 docstring
  - 写一份 `EXPERIMENTS.md`：汇总所有 benchmark 数字（显存、速度、perplexity）
  - 完成 PhD research statement 初稿（300-500 字英文）
  - 整理 10-15 位目标导师列表（含研究方向、代表论文、学校）
  - 整理完整面试题库（目标 50+ 题）

## 每周文件的格式要求

每个 daily_tasks.md 必须严格按以下结构生成：

```markdown
# W{N} 每日任务：{主题名}

> **周期**：2026/MM/DD — 2026/MM/DD（从 2026/03/31 开始，每周连续）
> **主线**：Line 1（模型基础）/ Line 2（推理优化）/ 综合
> **本周目标**：一句话概括
> **核心产出**：列出本周必须产出的文件名（.md 笔记 + .py 代码）

---

## 核心论文/阅读材料

列出本周需要读的 2-4 篇论文或资料，附 arxiv 链接或博客链接。

## 关键概念与问题清单

列出 5-8 个本周必须能回答的核心问题（可直接用作 PhD 面试题）。

## Day 1（周一）— Day 7（周日）每日任务

**每天的格式**：

### Day N（周X）：{当日主题}

**类型**：📖 理论 / 💻 工程 / 📖+💻 混合 / 📝 整理

**阅读**：
- [ ] 具体读什么（精确到 section 或页码）

**代码任务**（工程日必须有）：
- [ ] 具体写什么代码
  - 文件名：`W{N}/xxx.py`
  - 输入：描述输入维度和类型
  - 输出：描述预期输出和验证方法
  - 验证：`assert` 或 `torch.allclose` 或 benchmark 数字

**思考/笔记**：
- [ ] 当天需要记录或思考的问题

**时间预估**：X 小时

---

**Day 7 固定任务**：
- [ ] 完成本周深度笔记 `2026_MM_DD_TopicName.md`
- [ ] 更新面试题库（本周新增 3-5 道）
- [ ] 写公众号文章（给出建议标题）
- [ ] Git push 所有代码和笔记
- [ ] 检查所有代码可运行，补充 README

## 本周工程产出清单

明确列出本周必须提交到 GitHub 的所有文件：
- `W{N}/xxx.py` — 一句话描述
- `W{N}/yyy.py` — 一句话描述
- `2026_MM_DD_TopicName.md` — 深度笔记

## 本周面试题（3-5 道）

每道题需要：
- **题目**：问题本身
- **技术版答案**：给导师/面试官的精确回答
- **直觉版答案**：用比喻或类比解释给非技术人士

## PhD 关联思考

本周的 PhD 方向性思考题（必须回答并写入笔记）。

## 下周预告

一句话预告下周主题和主线切换。
```

## 额外硬性要求

1. **代码量要求**：每周至少产出 200 行以上的有效 Python 代码（注释和空行不算）。W1-W7 以复现为主（手写模块），W8-W13 以实验脚本为主（benchmark + 可视化），W14-W20 以系统实验为主（部署 + profiling + 对比测试）。

2. **所有实验必须可复现**：每个 .py 文件顶部要有注释说明运行环境（Python 版本、PyTorch 版本、GPU 型号）、安装依赖的命令、和运行命令。

3. **benchmark 数字化**：所有性能对比必须出具体数字（显存 X GB、速度 X tokens/sec、perplexity X.XX），不允许用"明显更快""显存更少"等模糊描述。

4. **可视化输出**：每周至少一张图表（matplotlib/wandb），保存为 PNG 到 `W{N}/figures/` 目录。

5. **推理视角贯穿**：即使是模型基础周（W1-2, W4-5, W8-9, W12-13, W16-17），每天至少一个任务从 inference 角度思考。例如 W4 读 BERT 时要回答"BERT 的双向注意力为什么不适合高效推理？"

6. **显存/FLOPs 手算**：从 W3 开始，每周至少一道手算题。

7. **PhD 申请渐进积累**：
   - W10：写第一版"我为什么对 inference 感兴趣"（200 字中文）
   - W12 开始：每周思考"本周内容有没有 workshop paper 的潜力"
   - W15：列出 5 位目标导师
   - W17：写第一版 research statement 草稿（英文）
   - W20：完成完整 research statement + 15 位导师列表

8. **错误与调试记录**：鼓励在笔记中记录"踩坑日志"——遇到了什么 bug、OOM、数值不一致，如何解决的。这些是真正的工程经验。

9. **语言**：任务描述用中文，代码注释和变量名用英文，技术术语保留英文原文。

请现在开始生成所有 20 个文件夹和对应的 daily_tasks.md。一次性全部生成，不要中途停止。如果上下文不够，告诉我从哪里继续。
```
