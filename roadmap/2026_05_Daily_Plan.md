# 2026 年 5 月每日执行计划 v1

> 承接 `2026_04_Daily_Plan.md` 与 `Phase_A_Detailed_Plan.md`
>
> 硬件：RTX 4090 Laptop 16GB，conda env `llm-infer`
>
> 本月主目标：完成 minimal GPT-2 推理复现、手动 KV Cache、Flash Attention tiling 理解与复现、MHA/MQA/GQA 消融，并收口成第一个 component-level inference efficiency mini project。

---

## 三条并行线

| 主线 | 每日时间 | 目标 |
|------|:---:|------|
| **工程主线**（GPT-2 / KV Cache / Flash Attention / MQA-GQA） | 4-5h | 建立注意力与 KV Cache 的组件级实验能力 |
| **MLOps 支线**（服务化 / 监控 / batching） | 1-2h | 把推理实验逐步接到可观测、可压测的服务系统 |
| **基础副线**（LeetCode + 闭卷回忆） | 1h | 继续按专题刷题，保持面试基本盘 |

---

## 每日固定产出

- 至少 1 次 GitHub 提交
- 1 份实验记录、论文笔记或工程日志
- 1 次闭卷回忆：当天核心概念用白纸画图或口述 3 分钟
- 每天 2 道 LeetCode；工程压力大时至少 1 道 Medium，但必须写题号 + 模式笔记
- 每周 1 个明确假设 + 1 组最小可复现实验验证

---

## 本月硬产出

- `journal/W5/minimal_gpt2.py`
- `journal/W5/kv_cache_decode.py`
- `journal/W5/kv_cache_benchmark.py`
- `journal/W6/naive_attention.py`
- `journal/W6/flash_attention_tiling.py`
- `journal/W6/attention_memory_benchmark.py`
- `journal/W7/mqa_gqa_attention.py`
- `journal/W7/kv_cache_boundary.py`
- `journal/W8/attention_kv_project/`
- `journal/W8/attention_kv_report.md`
- 3 张核心图：KV Cache speedup 曲线、seq_len vs attention 显存曲线、batch x seq_len KV Cache 显存边界热力图

---

## LeetCode 专题策略

| 专题 | 天数 | 题数 | 对应日期 |
|------|:---:|:---:|---|
| 动态规划进阶 | 6 天 | 12 题 | 5/1 - 5/6 |
| 二分查找 | 3 天 | 7 题 | 5/7 - 5/9 |
| 链表 | 4 天 | 9 题 | 5/10 - 5/13 |
| 树 / DFS / BFS | 7 天 | 15 题 | 5/14 - 5/20 |
| 堆 / 区间 / 贪心 | 6 天 | 13 题 | 5/21 - 5/26 |
| 图论入门 + 综合复盘 | 5 天 | 9 题 | 5/27 - 5/31 |
| **总计** | **31 天** | **65 题** | |

---

## W5.5（5/1 - 5/4）：4 月收口 + minimal GPT-2 启动

> **本周假设：** 只要先完成 logits 对齐，后续 KV Cache、sampling、benchmark 都能稳定推进。
>
> **策略：** 不补 4 月所有欠账，只收口会阻塞 5 月实验的部分：KV Cache 基准、FastAPI 服务骨架、README 链接。

### 5/1 周五 — 4 月收口与 W5 实验边界确认

- **工程（3h）：** 检查 W4 KV Cache、FastAPI server、README 中尚未对齐的路径与入口命令。列出 5 月必须保留的最小接口：`generate_no_cache`、`generate_with_cache`、`benchmark_decode`。
- **MLOps（1.5h）：** 确认 FastAPI `/generate` 可跑通；记录一次本地 curl 调用结果。
- **LeetCode（1h）— DP 进阶 Day 1：**
  - 91. Decode Ways（Medium）
  - 139. Word Break（Medium）
- **产出：** `journal/W5/w5_scope.md`、W4/W5 接棒 checklist

### 5/2 周六 — minimal GPT-2 骨架

- **工程（5h）：** 实现 `GPTConfig`、token embedding、position embedding、`CausalSelfAttention`、MLP、Block、GPT model forward。先只保证随机权重下 shape 全部正确。
- **闭卷回忆：** 画出 GPT-2 推理数据流：embedding → blocks → ln_f → lm_head → logits。
- **LeetCode（1h）— DP 进阶 Day 2：**
  - 416. Partition Equal Subset Sum（Medium）
  - 494. Target Sum（Medium）
- **产出：** `journal/W5/minimal_gpt2.py` 初版、shape test

### 5/3 周日 — 加载 HuggingFace GPT-2 权重

- **工程（5h）：** 写权重转换逻辑，把 HuggingFace GPT-2 small 权重加载到 minimal GPT-2。处理 Conv1D 权重转置，验证同一 prompt 下 logits `allclose`。
- **论文（1h）：** 精读 GPT-2 paper 的 model / training / sampling 部分，写 10 条压缩笔记。
- **LeetCode（1h）— DP 进阶 Day 3：**
  - 1143. Longest Common Subsequence（Medium）
  - 583. Delete Operation for Two Strings（Medium）
- **产出：** `journal/W5/load_hf_gpt2.md`、logits 对齐记录

### 5/4 周一 — Greedy / Top-k Sampling + W5 mini memo

- **工程（4h）：** 实现 greedy decoding 与 top-k sampling。和 HuggingFace `generate()` 做输出 sanity check，记录差异来源：采样随机性、temperature、top_k。
- **MLOps（1h）：** 把 minimal GPT-2 接入一个本地 CLI 或简单 FastAPI endpoint，用于后续 benchmark。
- **LeetCode（1h）— DP 进阶 Day 4：**
  - 72. Edit Distance（Medium）
  - 115. Distinct Subsequences（Hard，可只做思路 + 伪代码）
- **产出：** `journal/W5/generation_strategies.md`、W5 mini memo

---

## W6（5/5 - 5/11）：手动 KV Cache + Flash Attention 原理

> **本周假设：** KV Cache 在较长 prompt 上显著降低逐 token decode 的重复计算；Flash Attention 的主要收益来自减少 HBM 读写，而不是改变数学结果。
>
> **LeetCode 专题：** DP 收尾 → 二分查找 → 链表启动。

### 5/5 周二 — 手动 KV Cache decode

- **工程（5h）：** 给 minimal GPT-2 的 attention 加 `past_key_values` 输入输出。实现逐 token decode loop，验证无 cache 与有 cache 的生成文本一致。
- **闭卷回忆：** 画出 KV Cache 中 K/V tensor 的 shape 演化。
- **LeetCode（1h）— DP 进阶 Day 5：**
  - 5. Longest Palindromic Substring（Medium）
  - 647. Palindromic Substrings（Medium）
- **产出：** `journal/W5/kv_cache_decode.py`

### 5/6 周三 — KV Cache speedup benchmark

- **工程（4h）：** benchmark prompt length `{32, 64, 128, 256, 512, 1024}`，对比 no-cache vs cache 的 ms/token、tokens/sec、peak memory。
- **MLOps（1h）：** 给 benchmark 结果保存 JSON/CSV，统一字段：model、dtype、batch、seq_len、new_tokens、latency、memory。
- **LeetCode（1h）— DP 进阶 Day 6：**
  - 10. Regular Expression Matching（Hard，可只做 DP 状态定义）
  - 97. Interleaving String（Medium）
- **产出：** `journal/W5/kv_cache_benchmark.py`、KV Cache speedup 曲线

### 5/7 周四 — KV Cache 显存公式与实测对齐

- **工程（4h）：** 写 KV Cache 显存计算器，公式：`2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes`。用实测 peak memory 做误差分析。
- **论文（1h）：** 写一页“为什么 BERT 双向注意力不能直接用于自回归生成”的推理笔记。
- **LeetCode（1h）— 二分 Day 1：**
  - 704. Binary Search（Easy）
  - 35. Search Insert Position（Easy）
- **产出：** `journal/W5/kv_cache_memory_formula.md`

### 5/8 周五 — Flash Attention 论文精读

- **工程/论文（4h）：** 精读 FlashAttention: IO-Awareness、tiling、online softmax。写出 naive attention 的 HBM 读写瓶颈。
- **闭卷回忆：** 推导 online softmax 的 `m_i`、`l_i` 更新公式。
- **LeetCode（1h）— 二分 Day 2：**
  - 34. Find First and Last Position of Element in Sorted Array（Medium）
  - 33. Search in Rotated Sorted Array（Medium）
- **产出：** `journal/W6/flash_attention_notes.md`

### 5/9 周六 — Naive Attention baseline

- **工程（4h）：** 实现 naive attention baseline，记录不同 seq_len 下的显存和速度。确认 attention score 矩阵 `B x H x N x N` 是显存核心压力。
- **MLOps（1h）：** 给 benchmark 加命令行参数与固定随机种子。
- **LeetCode（1h）— 二分 Day 3：**
  - 153. Find Minimum in Rotated Sorted Array（Medium）
  - 74. Search a 2D Matrix（Medium）
  - 875. Koko Eating Bananas（Medium）
- **产出：** `journal/W6/naive_attention.py`

### 5/10 周日 — Tiling Attention 初版

- **工程（5h）：** 实现 Python 层面的 block tiling attention。先不追求速度，只验证输出与 naive attention 接近。
- **LeetCode（1h）— 链表 Day 1：**
  - 206. Reverse Linked List（Easy）
  - 21. Merge Two Sorted Lists（Easy）
- **产出：** `journal/W6/flash_attention_tiling.py` 初版

### 5/11 周一 — W6 收尾：Flash vs SDPA 对比

- **工程（4h）：** 对比 naive attention、tiling attention、`torch.nn.functional.scaled_dot_product_attention` 的显存/速度。画 seq_len vs memory 曲线。
- **复盘（1h）：** 写 W6 mini memo：Flash Attention 优化的是 IO，不是近似注意力。
- **LeetCode（1h）— 链表 Day 2：**
  - 141. Linked List Cycle（Easy）
  - 142. Linked List Cycle II（Medium）
- **产出：** `journal/W6/attention_memory_benchmark.py`、W6 memo

---

## W7（5/12 - 5/18）：MQA/GQA + KV Cache 显存边界

> **本周假设：** 在 decode 阶段，减少 KV heads 能线性降低 KV Cache 显存，并改善长上下文/大 batch 的可服务边界。
>
> **LeetCode 专题：** 链表收尾 → 树 / DFS / BFS 启动。

### 5/12 周二 — MHA / MQA / GQA 公式日

- **工程（3h）：** 写 MHA、MQA、GQA 的参数量与 KV Cache 大小公式。固定 hidden size，推导 `num_kv_heads` 对显存的影响。
- **论文（1.5h）：** 精读 GQA paper 的 introduction、method、checkpoint conversion 部分。
- **LeetCode（1h）— 链表 Day 3：**
  - 19. Remove Nth Node From End of List（Medium）
  - 24. Swap Nodes in Pairs（Medium）
- **产出：** `journal/W7/mqa_gqa_formula.md`

### 5/13 周三 — 实现 MQA

- **工程（5h）：** 在 attention 模块里实现 MQA：多个 query heads 共享 1 个 KV head。验证输出 shape、mask、cache append 逻辑。
- **闭卷回忆：** 口述 MHA 与 MQA 的 Q/K/V shape 差异。
- **LeetCode（1h）— 链表 Day 4：**
  - 2. Add Two Numbers（Medium）
  - 138. Copy List with Random Pointer（Medium）
  - 143. Reorder List（Medium）
- **产出：** `journal/W7/mqa_gqa_attention.py` MQA 部分

### 5/14 周四 — 实现 GQA

- **工程（5h）：** 实现 GQA：`num_kv_heads = num_heads // 4`，完成 KV repeat / expand 逻辑。和 MHA/MQA 统一接口。
- **LeetCode（1h）— 树 Day 1：**
  - 104. Maximum Depth of Binary Tree（Easy）
  - 226. Invert Binary Tree（Easy）
- **产出：** `journal/W7/mqa_gqa_attention.py` GQA 部分

### 5/15 周五 — MHA/MQA/GQA 显存 benchmark

- **工程（4h）：** 固定模型配置，对比 MHA/MQA/GQA 在不同 seq_len 和 batch 下的 KV Cache memory。
- **MLOps（1h）：** 把显存 benchmark 结果保存成统一 CSV，方便 W8 report 复用。
- **LeetCode（1h）— 树 Day 2：**
  - 100. Same Tree（Easy）
  - 101. Symmetric Tree（Easy）
- **产出：** `journal/W7/kv_cache_boundary.py` 初版

### 5/16 周六 — OOM 边界热力图

- **工程（5h）：** 扫描 batch x seq_len 网格，记录 MHA/MQA/GQA 的可运行边界。画显存边界热力图。
- **LeetCode（1h）— 树 Day 3：**
  - 102. Binary Tree Level Order Traversal（Medium）
  - 199. Binary Tree Right Side View（Medium）
- **产出：** `journal/W7/kv_cache_boundary_heatmap.png`

### 5/17 周日 — 服务侧含义：batching 与 cache

- **工程/MLOps（4h）：** 写一个简化 dynamic batching 模拟器：不同请求 prompt/new_tokens 组合下，观察 batch 对吞吐和 KV Cache 显存的影响。
- **LeetCode（1h）— 树 Day 4：**
  - 98. Validate Binary Search Tree（Medium）
  - 230. Kth Smallest Element in a BST（Medium）
- **产出：** `journal/W7/batching_kv_cache_note.md`

### 5/18 周一 — W7 memo

- **工程（3h）：** 写 W7 memo：MHA/MQA/GQA 的公式、实测显存、服务边界。
- **复盘（1h）：** 输出一张 MHA → MQA → GQA 演进图。
- **LeetCode（1h）— 树 Day 5：**
  - 105. Construct Binary Tree from Preorder and Inorder Traversal（Medium）
  - 236. Lowest Common Ancestor of a Binary Tree（Medium）
- **产出：** `journal/W7/w7_memo.md`

---

## W8（5/19 - 5/25）：Mini Project 1 收口

> **项目主题：** `Attention Mechanism and KV Cache: A Component-Level Inference Efficiency Analysis`
>
> **本周假设：** 一个好的推理组件报告必须同时给出速度、显存、质量/数值一致性和适用边界，而不是只报告 speedup。

### 5/19 周二 — 项目目录与实验协议

- **工程（4h）：** 创建 `journal/W8/attention_kv_project/`。统一实验配置：model size、dtype、seq_len、batch、device、warmup、重复次数。
- **LeetCode（1h）— 树 Day 6：**
  - 124. Binary Tree Maximum Path Sum（Hard，可只做递归状态）
  - 543. Diameter of Binary Tree（Easy）
- **产出：** `journal/W8/attention_kv_project/README.md`

### 5/20 周三 — 复现实验 1：prefill vs decode

- **工程（5h）：** 整理 W2/W5 数据，重新跑一组可复现 prefill vs decode benchmark。确认 decode latency 和 KV Cache speedup。
- **LeetCode（1h）— 树 Day 7：**
  - 200. Number of Islands（Medium）
  - 994. Rotting Oranges（Medium）
- **产出：** `results/prefill_decode.csv`、图表草稿

### 5/21 周四 — 复现实验 2：attention memory

- **工程（5h）：** 重新跑 naive / tiling / SDPA 在 seq_len `{512, 1024, 2048, 4096, 8192}` 下的显存和速度对比。
- **LeetCode（1h）— 堆 Day 1：**
  - 215. Kth Largest Element in an Array（Medium）
  - 347. Top K Frequent Elements（Medium，复做加强）
- **产出：** `results/attention_memory.csv`、attention memory 图

### 5/22 周五 — 复现实验 3：MHA/MQA/GQA

- **工程（5h）：** 重新跑 MHA/MQA/GQA 的 KV Cache memory 和 decode latency。记录 speed-memory tradeoff。
- **LeetCode（1h）— 堆 Day 2：**
  - 23. Merge k Sorted Lists（Hard）
  - 295. Find Median from Data Stream（Hard，可只做双堆思路）
- **产出：** `results/mha_mqa_gqa.csv`、KV heads 对比图

### 5/23 周六 — Report 初稿

- **写作（5h）：** 写 `attention_kv_report.md` 初稿：研究问题、实验设置、结果表、三张图、关键结论。
- **LeetCode（1h）— 区间 Day 1：**
  - 56. Merge Intervals（Medium）
  - 57. Insert Interval（Medium）
- **产出：** `journal/W8/attention_kv_report.md` 初稿

### 5/24 周日 — Report 打磨 + 负结果记录

- **工程/写作（4h）：** 增加 failure cases：tiling Python 版可能更慢、显存测量受 PyTorch allocator 影响、MQA/GQA 需要训练或权重转换才有真实质量意义。
- **LeetCode（1h）— 区间 Day 2：**
  - 435. Non-overlapping Intervals（Medium）
  - 452. Minimum Number of Arrows to Burst Balloons（Medium）
- **产出：** 负结果记录、实验限制说明

### 5/25 周一 — W8 项目收口

- **工程（3h）：** 清理脚本入口与 README，保证别人能复现实验。
- **写作（2h）：** 写 portfolio-facing 摘要：3 个 findings + 3 张图 + 1 段系统含义。
- **LeetCode（1h）— 贪心 Day 1：**
  - 55. Jump Game（Medium）
  - 45. Jump Game II（Medium）
- **产出：** `journal/W8/attention_kv_report.md` 完成版、W8 memo

---

## W9 预热（5/26 - 5/31）：项目发布 + 6 月量化准备

> **目标：** 不急着进入量化代码，先把 5 月的项目整理成能展示、能复现、能讲清楚的成果。

### 5/26 周二 — README 与 docs 更新

- **工程/写作（4h）：** 更新项目 README：加入 5 月 attention/KV Cache 项目入口、核心图、复现命令。
- **MLOps（1h）：** 给 benchmark 脚本补 `--help`、默认参数与输出目录。
- **LeetCode（1h）— 贪心 Day 2：**
  - 134. Gas Station（Medium）
  - 763. Partition Labels（Medium）
- **产出：** README 更新、docs 图表链接

### 5/27 周三 — 面试讲述版压缩

- **写作（4h）：** 把 5 月项目压缩成 5 分钟面试讲述：背景、瓶颈、方法、数据、结论、下一步。
- **闭卷回忆：** 白纸画出 KV Cache、Flash Attention、GQA 三张图。
- **LeetCode（1h）— 图论 Day 1：**
  - 133. Clone Graph（Medium）
  - 207. Course Schedule（Medium）
- **产出：** `docs/attention_kv_interview_pitch.md`

### 5/28 周四 — 测试与数值一致性

- **工程（4h）：** 给 minimal GPT-2、attention、KV Cache 写最小 pytest：shape、logits 对齐、cache/no-cache 文本一致、attention allclose。
- **LeetCode（1h）— 图论 Day 2：**
  - 210. Course Schedule II（Medium）
  - 417. Pacific Atlantic Water Flow（Medium）
- **产出：** `tests/test_attention_kv.py`

### 5/29 周五 — 6 月量化预研

- **论文/工程（4h）：** 预读 LLM.int8()，建立 6 月实验问题：FP16/INT8/INT4 的速度、显存、perplexity 如何联合评估。
- **MLOps（1h）：** 确认 perplexity evaluation dataset 与脚本入口。
- **LeetCode（1h）— 图论 Day 3：**
  - 684. Redundant Connection（Medium）
  - 261. Graph Valid Tree（Medium，可用并查集）
- **产出：** `journal/W9/quantization_prep.md`

### 5/30 周六 — 5 月 mini report

- **写作（5h）：** 编写 5 月 mini report：完成项、核心数据、未完成项、6 月计划调整。
- **LeetCode（1h）— 综合复盘 Day 1：**
  - 208. Implement Trie（Medium）
  - 211. Design Add and Search Words Data Structure（Medium）
- **产出：** `roadmap/2026_05_mini_report.md`

### 5/31 周日 — 月末复盘与归档

- **复盘（3h）：** 对照本月 checklist 打勾。把未完成项分类：阻塞 6 月 / 可丢弃 / 可周末补。
- **GitHub（1h）：** 整理提交、tag 或 release note 草稿。
- **LeetCode（1h）— 综合复盘 Day 2：**
  - 128. Longest Consecutive Sequence（Medium，复做）
  - 76. Minimum Window Substring（Hard，可只做滑窗模板复盘）
- **产出：** 5 月复盘、6 月第一周任务草案

---

## 5 月技能 checklist（5/31 前打勾）

### LLM Inference

- [ ] minimal GPT-2 forward shape 正确
- [ ] 可加载 HuggingFace GPT-2 权重并完成 logits 对齐
- [ ] 实现 greedy decoding 与 top-k sampling
- [ ] 手动实现 KV Cache decode
- [ ] 跑出 no-cache vs cache 的 speedup 曲线
- [ ] 能手算并实测 KV Cache 显存
- [ ] 精读并能讲清 Flash Attention 的 tiling + online softmax
- [ ] 实现 naive attention / tiling attention / SDPA 对比
- [ ] 实现 MHA / MQA / GQA 统一接口
- [ ] 跑出 batch x seq_len 显存边界图

### MLOps / 工程化

- [ ] FastAPI `/generate` 可调用 minimal GPT-2 或 HuggingFace GPT-2
- [ ] benchmark 输出统一 JSON/CSV
- [ ] benchmark 脚本可通过 CLI 参数复现
- [ ] 关键实验有 pytest 或 sanity check
- [ ] README 有 5 月项目入口、图表和复现命令

### LeetCode

- [ ] 动态规划进阶：12 题
- [ ] 二分查找：7 题
- [ ] 链表：9 题
- [ ] 树 / DFS / BFS：15 题
- [ ] 堆 / 区间 / 贪心：13 题
- [ ] 图论入门 + 综合复盘：9 题

### 写作与表达

- [ ] W5-W8 每周 mini memo
- [ ] `attention_kv_report.md` 完成
- [ ] 5 分钟面试讲述稿完成
- [ ] 5 月 mini report 完成
- [ ] 6 月量化实验预研完成

---

## 月末判断标准

到 5/31，不用追求所有代码都漂亮，但必须能回答下面 5 个问题：

1. KV Cache 为什么能加速 decode？它牺牲了什么？
2. Flash Attention 为什么说是 IO-aware？它和近似注意力有什么区别？
3. MQA/GQA 为什么能降低 KV Cache 显存？代价是什么？
4. 在 RTX 4090 Laptop 16GB 上，batch 和 seq_len 的可服务边界在哪里？
5. 如果要把这套实验变成真实 serving 系统，下一步应该优化 batching、PagedAttention 还是量化？

---

*最后更新：2026-05-06*
