# 2026 年 4 月每日执行计划 v3

> 整合 LLM Inference 学习 + MLOps 转型 + 每日 LeetCode
> 基于：
> - W1-W2 LLM Inference 已完成工作
> - Phase A LLM Inference roadmap
> - 2026 04-08 MLOps AI Engineer Study Plan (PDF)
>
> 硬件：RTX 4090 Laptop, Ubuntu, conda env `llm-infer`
> 目标：完成 LLM Inference 基础（Roofline + KV Cache）+ 启动 MLOps 转型（Docker + FastAPI）+ 每日 LeetCode

---

## 三条并行线

| 主线 | 每日时间 | 目标 |
|------|:---:|------|
| **工程主线**（LLM Inference + MLOps） | 4-5h | 构建可部署的 AI 系统能力 |
| **论文副线**（ThinkRouter 实验） | 1-1.5h | Workshop 投稿 |
| **基础副线**（LeetCode + 系统设计） | 1h | 面试准备，8 月底前累计 100+ 题 |

---

## 每日固定产出

- 至少 1 次 GitHub 提交
- 1 份实验记录或论文笔记
- 1 次闭卷回忆
- **每天 2-3 道 LeetCode 题（同一专题，必须有题号 + 笔记）**
- 每周 1 个明确的假设 + 验证

---

## LeetCode 专题刷题策略（v3 修正）

> **核心原则：** 按专题分块，同一天只做同一类型题目，从 Easy 到 Medium 递进
> **每道题节奏：** 独立想 15-20min → 没思路看提示不看代码 → 再想 10min → 看题解逐行理解 → 合上答案自己写一遍 → 笔记：什么模式、时间复杂度、哪里卡住

| 专题 | 天数 | 题数 | 对应日期 |
|------|:---:|:---:|---|
| 哈希表 | 4 天 | 11 题 | 4/14 - 4/17 |
| 双指针 | 4 天 | 11 题 | 4/18 - 4/21 |
| 滑动窗口 | 3 天 | 6 题 | 4/22 - 4/24 |
| 栈 | 2 天 | 6 题 | 4/25 - 4/26 |
| 动态规划入门 | 4 天 | 11 题 | 4/27 - 4/30 |
| **总计** | **17 天** | **45 题** | |

---

## W1（4/1 - 4/5）：Transformer 理论 + GPT-2 首次推理

> **本周假设：** Attention 计算在 GPT-2 推理中占比 > 50%
> **状态：** 已完成（本周尚未启动 LeetCode）

| 日期 | 主任务 | 产出 |
|:---:|---|---|
| 4/1 周三 | 环境搭建，跑 GPT-2 `model.generate()`，记录耗时/显存 | 环境记录、首次推理体验 |
| 4/2 周四 | 精读 Attention Is All You Need (Section 3.2) | 数学推导卡、结构化笔记 |
| 4/3 周五 | 手写 ScaledDotProductAttention + MultiHeadAttention | `W1/scaled_dot_product_attention.py`、`W1/multi_head_attention.py` |
| 4/4 周六 | 精读 Section 3.1, 3.5；位置编码 + Pre/Post Norm | PE / dropout / 归一化笔记 |
| 4/5 周日 | W1 复盘 + 闭卷回忆 + GitHub 推送 | W1 mini memo |

---

## W2（4/6 - 4/12）：Decoder Block + GPT-2 Profiling

> **本周假设：** Decode 阶段占总推理时间 > 80%
> **状态：** 已完成（假设确认，decode 占 96.7%-98.2%）

| 日期 | 主任务 | 产出 |
|:---:|---|---|
| 4/7 周一 | 手写 LayerNorm，与 `nn.LayerNorm` 数值对齐 | `W2/LayerNorm.py` |
| 4/8 周二 | PreNorm 包装器 + GPT-style Decoder Block | `W2/PreNorm.py`、`W2/DecoderBlock.py` |
| 4/9 周三 | 精读：为什么 Decoder-only 胜出（Causal mask、KV Cache 友好） | Decoder-only 笔记 |
| 4/10 周四 | GPT-2 prefill/decode profiling，torch.profiler trace | `W2/gpt2_profiling.py`、`gpt2_profiling_results.json` |
| 4/11 周五 | Memo + W3 假设 + Decoder Block 数据流图 | `W2/decoder_profiling_memo.md`、`W3_hypothesis.md`、`decoder_block_dataflow.png` |
| 4/12 周六 | 知乎博客：为什么 Decoder-only 胜出 | `W2/blog_decoder_only.md` |

---

## W3（4/13 - 4/19）：GPU 内存层级 + Roofline + MLOps/LeetCode 启动

> **本周假设：** GPT-2 decode 是深度 memory-bound，真实 FLOPs 利用率 < 5%
> **状态：** 进行中
> **LeetCode 专题：** 哈希表（4/14-4/17）→ 双指针启动（4/18-4/19）

### 4/13 周一 — Roofline 理论日（纯 LLM Inference）

- **工程（5h）：** 精读 Williams 2008 Roofline 论文。理解 peak compute、peak bandwidth、ridge point。计算 RTX 4090 Laptop 的 ridge point（57 FLOPs/byte）。GPU 内存层级图初稿。
- **论文（1.5h）：** 继续 ThinkRouter MATH-500 实验
- **LeetCode：** 暂未启动
- **MLOps：** 暂未启动

### 4/14 周一 — MLOps + LeetCode 双启动日

- **工程（3h）：** GPU 内存层级图（Register → SRAM → L2 → HBM → DRAM）。读 HBM2 vs GDDR6X 带宽差异。Roofline 笔记打磨。
- **MLOps 启动（2h）：** 安装 Docker，理解 image vs container。`docker run hello-world`。写一个跑 Python 脚本的 Dockerfile。
- **LeetCode（1h）— 哈希表 Day 1：**
  - ✅ 1. Two Sum（Easy）
  - ✅ 121. Best Time to Buy and Sell Stock（Easy）
  - ✅ 217. Contains Duplicate（Easy）
- **论文（1h）：** ThinkRouter MMLU STEM benchmark
- **产出：** `W3/gpu_memory_hierarchy.png`、`W3/roofline_notes.md`、第一个 Dockerfile、LeetCode 笔记

### 4/15 周二 — Roofline 作图 + Docker 进阶

- **工程（3h）：** 用 W2 profiling 数据计算 prefill（每个 L）和 decode 的 arithmetic intensity。在 Roofline 图上标出实测点。
- **MLOps（2h）：** 用 Dockerfile 容器化一个 Python 脚本。理解 layer、缓存、`.dockerignore`。
- **LeetCode（1h）— 哈希表 Day 2：**
  - 242. Valid Anagram（Easy）
  - 383. Ransom Note（Easy）
  - 387. First Unique Character in a String（Easy）
- **产出：** `W3/roofline_chart.png`、`W3/w2_profiling_roofline_interpretation.md`

### 4/16 周三 — Docker Compose + FastAPI 入门

- **工程（2h）：** 用 Roofline 简短解释为什么 decode SM util 看起来高，但真实 FLOPs 利用率很低。
- **MLOps（3h）：** docker-compose 基础（多容器：Python + Redis）。FastAPI hello world，写一个 `/predict` 接口，接收文本返回假数据。
- **LeetCode（1h）— 哈希表 Day 3：**
  - 49. Group Anagrams（Medium）
  - 128. Longest Consecutive Sequence（Medium）

### 4/17 周四 — FastAPI + Pydantic

- **工程（2h）：** Roofline 解释 W2 数据，笔记打磨。
- **MLOps（3h）：** 给请求/响应加 Pydantic models。加错误处理。理解路由、async。用 `curl` 和 `httpie` 测试。
- **LeetCode（1h）— 哈希表 Day 4：**
  - 347. Top K Frequent Elements（Medium）
  - 238. Product of Array Except Self（Medium）

### 4/18 周五 — Docker + FastAPI 集成日

- **工程 + MLOps（4h）：** 把 FastAPI 应用容器化。`docker build`、`docker run -p 8000:8000`。用 docker-compose 启动它。
- **LeetCode（1h）— 双指针 Day 1：**
  - 26. Remove Duplicates from Sorted Array（Easy）
  - 27. Remove Element（Easy）
  - 283. Move Zeroes（Easy）
- **论文（1h）：** ThinkRouter pilot 数据汇总（GSM8K ~2% gap、MMLU 0% gap、MATH L5 ~11% gap）

### 4/19 周日 — 病后恢复 + W3 最小收尾（重做版）

> **情况说明：** 4/15 - 4/18 因身体原因中断。W3 工程主体（Roofline 分析、GPU 内存层级、ncu profiling fp32/fp16/int8、precision latency 对比）在病前已基本完成。
> **今日策略：** 恢复节奏 + 锁定 W3 产出；**不补欠账**，MLOps 与漏做的 LC 题合并到 W4 推进。

- **工程（1.5h）— W3 收尾：**
  - 写 `W3/w3_memo.md`：总结 Roofline 分析结论、GPU 内存层级图、fp32/fp16/int8 latency 对比、关键洞察（decode memory-bound、真实 FLOPs 利用率 < 5%）。
  - 只做总结,不再新增实验。
- **W4 铺垫（0.5h）：** 把 W4 假设写进 `W3/w3_memo.md` 末尾或新建 `W4/W4_hypothesis.md`：**KV Cache 在 seq_len ≥ 256 时给出 >2x decode 加速**。
- **LeetCode（1h）— 双指针 Day 2（跳过欠账题）：**
  - ✅ 125. Valid Palindrome（Easy）
  - ✅ 344. Reverse String（Easy）
  - ✅ 167. Two Sum II - Input Array Is Sorted（Medium）
  - *哈希表 Day 2-4 的 6 题 + 双指针 Day 1 的 3 题欠账不补,继续往前走,避免陷在"补课模式"里。*
- **今日不做（已下调预期）：**
  - ❌ Docker Compose / FastAPI / Pydantic — 合并到 W4 的 LLM 服务化里
  - ❌ 本周知乎/公众号博客 — 推迟到 W4 末
  - ❌ 论文副线（ThinkRouter）— 精力允许再说,不强求
- **产出：** `W3/w3_memo.md`（含 W4 假设）

**节奏调整（后续一周指引）：**
- 原定 4/15 - 4/18 的 MLOps（Docker Compose + FastAPI + Pydantic + 集成）并入 W4:4/22 / 4/25 的 MLOps 时段扩展承接。
- W4 主时间轴（4/20 - 4/26）保持不变,但允许 KV Cache benchmark 结论在 4/27 - 4/28 之间产出,不强求 4/24 完成。
- 月底 checklist 目标不变:KV Cache 验证 + 一个可部署的 FastAPI LLM 服务。

---

## W4（4/20 - 4/26）：KV Cache 实现 + FastAPI + LLM 包装

> **本周假设：** KV Cache 在 seq_len >= 256 时给出 >2x decode 加速
> **状态：** 即将开始
> **LeetCode 专题：** 双指针收尾（4/20-4/21）→ 滑动窗口（4/22-4/24）→ 栈（4/25-4/26）

### 4/20 周日 — KV Cache 理论 + 读 GPT-2 源码

- **工程（3h）：** 读 HuggingFace GPT-2 modeling 源码，找到 `past_key_values` 处理逻辑。理解 KV Cache 如何减少冗余计算。
- **MLOps（2h）：** 用 FastAPI 包装 HuggingFace `text-generation` pipeline。`/generate` 接口接收 prompt，返回生成文本。
- **LeetCode（1h）— 双指针 Day 3：**
  - 11. Container With Most Water（Medium）
  - 15. 3Sum（Medium）

### 4/21 周一 — 从零实现 KV Cache

- **工程（4h）：** 手写最小 GPT-2 decode loop，手动管理 KV cache。验证输出和 HuggingFace 一致。
- **LeetCode（1h）— 双指针 Day 4：**
  - 16. 3Sum Closest（Medium）
  - 75. Sort Colors（Medium）

### 4/22 周二 — KV Cache 加速 Benchmark

- **工程（4h）：** Benchmark 有/无 KV cache 在不同 prompt 长度 {32, 64, 128, 256, 512} 的速度。画出加速曲线。验证 >2x at seq_len >= 256 假设。
- **MLOps（1h）：** 容器化 FastAPI + GPT-2 服务。
- **LeetCode（1h）— 滑动窗口 Day 1：**
  - 209. Minimum Size Subarray Sum（Medium）
  - 904. Fruit Into Baskets（Medium）

### 4/23 周三 — KV Cache 显存增长 + OOM 边界

- **工程（3h）：** 测量 KV cache 显存随 seq_len 增长。找到 OOM 边界。计算 KV cache 大小公式：`2 * num_layers * num_heads * head_dim * seq_len * batch * dtype_bytes`。
- **MLOps（2h）：** 给 FastAPI 加基础日志（`logging` 模块）和请求耗时中间件。
- **LeetCode（1h）— 滑动窗口 Day 2：**
  - 3. Longest Substring Without Repeating Characters（Medium）
  - 424. Longest Repeating Character Replacement（Medium）

### 4/24 周四 — 综合推理 Benchmark

- **工程（4h）：** 把 W2-W4 所有数据汇总成一份完整的推理 benchmark 表（FP32、有/无 KV cache、不同 seq_len）。
- **LeetCode（1h）— 滑动窗口 Day 3：**
  - 567. Permutation in String（Medium）
  - 438. Find All Anagrams in a String（Medium）

### 4/25 周五 — Docker-compose 部署 LLM 服务

- **MLOps（4h）：** docker-compose：FastAPI + 简易 Redis 缓存 + Nginx 反向代理。端到端部署。
- **LeetCode（1h）— 栈 Day 1：**
  - 20. Valid Parentheses（Easy）
  - 155. Min Stack（Medium）
  - 150. Evaluate Reverse Polish Notation（Medium）
- **论文（1h）：** ThinkRouter pilot 数据整理成表格

### 4/26 周六 — W4 Memo + 栈收尾

- **工程（2h）：** W4 memo。写 KV Cache 加速结论，假设验证。
- **MLOps（1h）：** 4 月 mini 报告：技能 checklist（Docker / FastAPI / Profiling）。
- **LeetCode（1h）— 栈 Day 2：**
  - 739. Daily Temperatures（Medium）
  - 496. Next Greater Element I（Easy）
  - 503. Next Greater Element II（Medium）
- **产出：** `W4/w4_memo.md`

---

## W4.5（4/27 - 4/30）：DP 入门 + 4 月收尾

> **LeetCode 专题：** 动态规划入门（4/27-4/30）

### 4/27 周日 — 补漏 + DP 启动

- **工程（3h）：** 补 W3-W4 任何未完成的工作。读 MLflow 文档为 W5 做准备。
- **MLOps（2h）：** 安装 MLflow。运行 `mlflow ui`。验证本地工作。
- **LeetCode（1h）— DP Day 1：**
  - 70. Climbing Stairs（Easy）
  - 746. Min Cost Climbing Stairs（Easy）
  - 198. House Robber（Medium）

### 4/28 周一 — 全部推到 GitHub

- **工程（3h）：** 打磨 W2-W4 代码，加 README，把 `llm-inference-study` 推到 GitHub。
- **MLOps（2h）：** GitHub README，加架构图。
- **LeetCode（1h）— DP Day 2：**
  - 213. House Robber II（Medium）
  - 53. Maximum Subarray（Medium）
  - 152. Maximum Product Subarray（Medium）

### 4/29 周二 — 博客 + DP 继续

- **工程（2h）：** 写一篇知乎/公众号博客：W4 KV Cache 实验。
- **LeetCode（1h）— DP Day 3：**
  - 322. Coin Change（Medium）
  - 300. Longest Increasing Subsequence（Medium）

### 4/30 周三 — 4 月 mini 报告

- **工程（2h）：** 编写 4 月 mini 报告：数据、图表、关键洞察。
- **LeetCode（1h）— DP Day 4：**
  - 62. Unique Paths（Medium）
  - 64. Minimum Path Sum（Medium）
  - 392. Is Subsequence（Easy）
- **产出：** `roadmap/2026_04_mini_report.md`、`April_mini_report.md`、完整 GitHub 推送

---

## 4 月技能 checklist（4/30 前打勾）

### LLM Inference
- [x] 手写 LayerNorm 与 `nn.LayerNorm` 数值一致
- [x] 手写 PreNorm + DecoderBlock 含 causal mask
- [x] GPT-2 prefill/decode profiling 拿到硬数据
- [x] 在 RTX 4090 Laptop 上对 GPT-2 做 Roofline 分析
- [ ] 手写 KV Cache，验证 >2x decode 加速
- [ ] 综合推理 benchmark 表

### MLOps 基础
- [ ] Docker 基础（image、container、Dockerfile、docker-compose）
- [ ] FastAPI 基础（路由、async、Pydantic）
- [ ] 容器化一个 FastAPI 应用
- [ ] FastAPI + HuggingFace text-generation pipeline

### LeetCode（目标：4/30 前 45 题）
- [ ] 哈希表：11 题（4/14-4/17）
- [ ] 双指针：11 题（4/18-4/21）
- [ ] 滑动窗口：6 题（4/22-4/24）
- [ ] 栈：6 题（4/25-4/26）
- [ ] 动态规划入门：11 题（4/27-4/30）

### 论文副线
- [ ] ThinkRouter MATH-500 + MMLU STEM benchmark 完成
- [ ] Pilot 数据整理成表格

---

*最后更新：2026-04-19（病后恢复,4/19 周日重做）*
