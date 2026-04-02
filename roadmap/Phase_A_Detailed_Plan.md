# 阶段 A 详细执行计划：2026.04 - 2026.08（LLM Agent 架构版）

> **新的 Phase A 主线：** 不再走纯 LLM 应用开发，也不做泛泛的 prompt 工程，而是明确切到 **LLM Agent 架构设计的系统性研究**。
>
> **一句话定位：** 你要做的是 **systematic, component-level, evaluation-grounded LLM Agent Architecture research**。也就是不仅问"Agent 能不能完成任务"，还要问"它的哪些架构组件起关键作用，成本如何，失败在哪里"。
>
> **你的差异化故事：** 你不是只会接入 API 的应用型选手。你有计算机系统基础，所以你天然会从 **规划策略、记忆设计、工具集成、多智能体协调、失败模式** 这些维度去设计 Agent 架构研究问题。

---

## 这版 Phase A 到底在做什么

这一版计划统一围绕下面这个问题展开：

**在不同任务场景中，LLM Agent 的哪些架构设计决定了它的成功率、效率和鲁棒性？**

把它拆开，就是四个子问题：

1. **架构问题**
   - Agent 的规划模块、记忆系统、工具使用、多智能体协调各自贡献了什么？
   - 哪种架构组合在什么任务上更有效？

2. **规划问题**
   - 线性推理（CoT）、树状搜索（ToT）、反思校正（Reflexion）分别适合什么场景？
   - 规划深度提升效果，但成本如何？

3. **记忆问题**
   - in-context 记忆 vs 外部长期记忆 vs 反思式记忆，有什么本质区别？
   - 记忆设计如何影响 Agent 的跨轮次一致性？

4. **评测问题**
   - 不只看任务成功率，还要看：
     - planning steps count
     - tool call accuracy
     - self-correction rate
     - token cost per task
     - failure mode distribution

---

## 你这条线的科研故事

这版 Phase A 的目标，不是把你包装成"会用 AutoGen 搭应用的工程师"，而是把你包装成：

**一个做 LLM Agent 架构研究的研究者，能对规划、记忆、工具、多智能体协调做系统性的组件消融，并把 Agent 研究做成效果-成本-失败模式三位一体。**

更适合申请时讲成下面这种话：

- 我对 LLM Agent 的架构设计机制感兴趣
- 但我不满足于只做一个 Agent 的应用演示
- 我尤其关注规划策略消融、记忆系统设计、工具使用边界，以及 Agent 在现实任务中的失败模式分析

这会比单纯说"我做 Agent 应用"更强，也比单纯说"我用过 LangChain"更成体系。

---

## 这版 Phase A 的双主线

### 主线 A：LLM Agent 核心架构组件

你这一阶段优先研究三类架构问题：

1. **规划与推理模块（Planning & Reasoning）**
2. **记忆系统（Memory Architecture）**
3. **工具使用与多智能体协调（Tool Use & Multi-Agent）**

这些方向的共同特点是：

- 有清晰的组件边界，适合消融实验
- 有现成 benchmark 支撑（AgentBench, WebArena 等）
- 更容易切出干净的 research question

### 主线 B：架构分析视角下的评测与效率

你不会把架构研究当成纯理论工作，而是把它变成有实验支撑的组件分析：

1. **做 planning strategy ablation**
2. **做 memory design comparison**
3. **做 tool call accuracy 与 API 调用成本分析**
4. **做 failure mode taxonomy（失败模式分类）**

也就是说：

- 别人只问 "这个 Agent 对不对"
- 你会继续问 "是哪个组件让它对了，哪个组件让它错了，以及换一种设计会怎样"

---

## 适合你发第一篇的贡献形态

这一版 Phase A 默认优先产出下面四类东西：

1. **架构组件消融实验**
   - 例如 no-memory vs in-context memory vs external memory
   - 例如 CoT vs ReAct vs Reflexion on the same task set

2. **失败模式分类与分析**
   - 什么类型的任务最容易让 Agent 失败
   - 规划错误 vs 工具调用错误 vs 记忆遗漏 各占多少

3. **任务指标 + 效率指标联合分析**
   - 不只看成功率，还看 step count、token cost、self-correction rate

4. **跨架构对比框架**
   - 在同一任务集上公平比较不同 Agent 框架（AutoGen, MetaGPT, Reflexion 等）

---

## 整体节奏

| 时间段 | 主题 | 周数 |
|--------|------|------|
| 4月（W1-W4） | 基础设施 + Agent 架构认知 + ReAct/Reflexion 实现 + 最小评测框架 | 4周 |
| 5月（W5-W8） | 规划模块深入 + 策略消融 + CoT/ToT/Reflexion 对比实验 | 4周 |
| 6月（W9-W12） | 记忆系统 + 工具使用 + 跨组件消融实验 | 4周 |
| 7月（W13-W16） | 多智能体协调 + 综合评测 + 综合项目 | 4周 |
| 8月（W17-W20） | 消融 + 技术报告 / pre-paper + Phase B 准备 | 4周 |

**每周固定节奏：**

- 周一至周三：代码实验
- 周四：论文精读 + 高密度压缩
- 周五：整理图表、负结果、mini memo
- 周末半天：闭卷回忆 + 写知乎/公众号 + 补 README / report

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
- 1 份 failure taxonomy 或 error note

---

## 这版 Phase A 的关键指标

### 任务指标

- `task success rate`
- `planning step count`
- `tool call accuracy`
- `self-correction rate`
- `task completion trajectory`
- `cross-turn consistency`
- `failure mode distribution`

### 架构与效率指标

- token cost per task
- average planning steps
- tool call count
- memory hit rate
- context tokens consumed
- cost per successful task
- failure analysis breakdown

> 这一组指标就是你把 Agent 架构分析真正落实为可测量科研的地方。

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
     - 白纸画流程图
     - 口头讲 3 分钟
     - 不看资料写伪代码 / 指标定义 / failure taxonomy

3. **小规模复现**
   - 每个主题必须有一个最小可跑实验
   - 优先小任务集、小样本、单变量对照
   - 单次实验尽量控制在 `2 小时内`

### 统一输出格式

每个主题尽量留下这四类材料：

- `压缩卡片`
- `回忆记录`
- `最小复现脚本`
- `mini memo`

### 每日学习模板

1. **输入 30-60 分钟**
2. **压缩 15-20 分钟**
3. **复现 45-120 分钟**
4. **回忆 10-20 分钟**
5. **收口 5-10 分钟**

### 每周检验标准

- 能不能 5 分钟讲清这周任务？
- 能不能画出 Agent 的规划 / 记忆 / 工具调用流程图？
- 能不能说出至少 3 个组件失败原因？
- 能不能跑出最小对比 baseline？
- 能不能同时解释效果和成本？

---

## 4月：基础设施 + Agent 架构认知 + ReAct/Reflexion 实现

### 本月目标

- 搭建本地 LLM Agent 研究实验环境
- 读懂 Agent 架构全图
- 实现最小 ReAct 和 Reflexion Agent
- 建立第一个 Agent 评测框架
- 形成第一套 planning strategy comparison 基线

### W1（4/1-4/6）：环境搭建 + 架构认知 + 最小 Agent Loop

**本周三核执行法：**

- **高密度压缩：** 写 1 页 Agent 架构图（感知-规划-记忆-行动）+ 1 页评测指标卡
- **主动回忆：** 闭卷写出 ReAct、Reflexion、MemGPT 的架构区别
- **小规模复现：** 实现一个能在 5-10 个任务上跑的最小 Agent loop

**实验任务：**

- 配置本地环境：
  - `conda/mamba`
  - Python 3.10+
  - `langchain` / `langgraph` 或手写 Agent loop
  - 一个可调用的 LLM 后端
  - 基本任务评测工具
- 实现最小 ReAct 风格 Agent（无工具，只做 Reason+Act 框架）
- 选 5-10 个简单 QA 或推理任务做第一轮评测
- 同时记录：
  - 任务成功率
  - 规划步数
  - token 消耗

**产出：**

- `README.md`
- `00_basic_agent_loop.py`
- 1 张表：任务成功率 + 规划步数 + token 成本

### W2（4/7-4/13）：ReAct 精读 + 规划策略 baseline

**本周三核执行法：**

- **高密度压缩：** 做 1 张 ReAct vs 纯 CoT 的对比卡
- **主动回忆：** 闭卷讲清 ReAct 为什么比纯 CoT 更适合 Agent 任务
- **小规模复现：** 在相同任务上对比 CoT-only vs ReAct

**实验任务：**

- 比较：
  - 纯 LLM（直接回答）
  - CoT（链式推理）
  - ReAct（推理+行动交替）
- 记录：
  - 任务成功率
  - 规划步数
  - 错误类型
  - token 成本

**产出：**

- `01_planning_strategy_baseline.py`
- 1 份 mini memo：CoT vs ReAct 的效果与成本

### W3（4/14-4/20）：Reflexion 实现 + 自我校正机制实验

**本周三核执行法：**

- **高密度压缩：** 做 1 张 Reflexion 机制总图（失败->反思->再尝试）
- **主动回忆：** 白纸画出 Reflexion 的完整控制流
- **小规模复现：** 在失败案例上跑 Reflexion，对比有无反思的差异

**实验任务：**

- 实现 Reflexion 风格的反思循环：
  - 第一次失败后生成反思文本
  - 把反思注入下一轮上下文
  - 最多重试 N 轮
- 比较：
  - 无反思 vs 有反思
  - 成功率提升量
  - 额外 token 成本

**产出：**

- `02_reflexion_agent.py`
- 1 张 `success rate vs retry count vs token cost` 图

### W4（4/21-4/27）：4 月 mini project 收口

**本周三核执行法：**

- **高密度压缩：** 做 1 张 4 月总图：规划策略、指标、成本
- **主动回忆：** 脱稿讲清你现在的 Agent 规划研究框架
- **小规模复现：** 重跑 4 月最关键对比结果

**产出：**

- `03_april_mini_project/`
- `03_april_report.md`
- 1 张 planning strategy 对比总表（CoT / ReAct / Reflexion）

**月末检查点：**

- [ ] 你有稳定可复用的最小 Agent 评测框架
- [ ] 你能解释 CoT、ReAct、Reflexion 的架构区别与适用场景
- [ ] 你有 1 条规划策略与效果/成本的对比结果

---

## 5月：规划模块深入 + 策略消融 + CoT/ToT/Reflexion 对比实验

### 本月目标

- 把规划模块研究做深
- 系统比较线性推理 vs 树状搜索 vs 反思校正
- 建立第一个像论文原型的规划策略消融项目

### W5（4/28-5/4）：Tree of Thoughts 精读 + 树状规划实现

**实验任务：**

- 实现最简版 BFS/DFS 风格树状推理
- 在相同任务集上对比：
  - CoT（线性）
  - ToT-BFS（广度优先树）
  - ToT-DFS（深度优先树）
- 记录：
  - 成功率
  - 搜索节点数
  - token 成本
  - 最终答案质量

**产出：**

- `04_tree_of_thoughts_baseline.py`
- `04_planning_depth_cost_report.md`

### W6（5/5-5/11）：规划策略消融 + 任务类型分析

**实验任务：**

- 比较：
  - 推理任务（math, logic）上哪种规划最好
  - 工具使用任务上哪种规划最好
  - 长链路任务上哪种规划最好
- 统计：
  - 各任务类型的最优规划策略
  - 各策略的成本与成功率 tradeoff

**产出：**

- `05_planning_task_type_ablation.py`
- 1 张 `task type vs planning strategy vs success rate` 图

### W7（5/12-5/18）：规划失败模式分类

**这一周是失败分析视角的关键接入口。**

你不是简单统计成功率，而是：

- 把规划失败系统分类
- 让失败分析成为论文里的核心贡献之一

**实验任务：**

- 收集所有规划失败案例
- 分类：
  - 目标理解错误
  - 规划步骤循环
  - 工具调用错误
  - 上下文遗漏
  - 过早终止
- 比较不同规划策略的失败分布差异

**产出：**

- `06_planning_failure_taxonomy.py`
- 1 份 failure taxonomy memo

### W8（5/19-5/25）：Mini Project 1 收口

**项目主题建议：**

`Planning Strategy Comparison in LLM Agents: Effect, Cost, and Failure Modes`

**实验要求：**

- 1 个清楚任务定义
- 3 个规划策略对比（CoT / ReAct / ToT or Reflexion）
- 1 个成本维度分析
- 1 组 failure taxonomy

**产出：**

- `07_planning_project/`
- `07_planning_project_report.md`

**月末检查点：**

- [ ] 你能讲清三种规划策略的架构差异
- [ ] 你有任务类型 × 规划策略的对比结果
- [ ] 你有第一个像样的规划消融小项目

---

## 6月：记忆系统 + 工具使用 + 跨组件消融实验

### 本月目标

- 从规划走向记忆和工具使用
- 做 in-context memory vs external memory 的实验对比
- 引入 tool call accuracy 和 API 调用成本分析

### W9（5/26-6/1）：MemGPT 精读 + 记忆基线实验

**实验任务：**

- 构造 10-20 个跨轮次任务
- 比较：
  - 无记忆（每次重置上下文）
  - 全 in-context 记忆
  - 摘要式外部记忆
- 记录：
  - 跨轮次任务成功率
  - 记忆命中率
  - token 成本

**产出：**

- `08_memory_baseline.py`
- 1 份记忆类型对比统计

### W10（6/2-6/8）：Generative Agents 精读 + 反思式记忆实验

**实验任务：**

- 实现简版反思记忆机制：
  - 存储重要事件
  - 定期触发反思
  - 用反思内容更新记忆摘要
- 比较：
  - 普通外部记忆 vs 反思式记忆
  - 任务成功率
  - 记忆一致性

**产出：**

- `09_reflective_memory.py`
- `09_memory_comparison_report.md`

### W11（6/9-6/15）：工具使用 + API 调用准确性

**实验任务：**

- 给 Agent 加 3-5 个简单工具（搜索、计算、代码执行等）
- 观察：
  - 工具选择准确率
  - 工具参数生成准确率
  - 工具调用次数与任务成功率的关系
  - 工具使用带来的额外 token 成本

**产出：**

- `10_tool_use_baseline.py`
- 1 张 `tool accuracy vs task success vs cost` 图

### W12（6/16-6/22）：Mini Project 2 收口

**项目主题建议：**

`Memory Architecture and Tool Use in LLM Agents: A Component-Level Analysis`

**实验要求：**

- 至少 2 种记忆设计
- 至少 1 种工具集成
- 至少 1 组跨轮次失败分析

**产出：**

- `11_memory_tool_project/`
- `11_memory_tool_project_report.md`

**月末检查点：**

- [ ] 你能解释三种记忆设计的本质区别
- [ ] 你能分析工具调用错误的主要类型
- [ ] 你已经把记忆和工具两个组件放进了对比实验

---

## 7月：多智能体协调 + 综合评测 + 综合项目

### 本月目标

- 扩到多智能体协调研究
- 把规划、记忆、工具、协调四个维度串起来
- 做一个综合项目，把架构消融和效率分析一起纳入

### W13（6/23-6/29）：AutoGen 精读 + 多智能体基线

**实验任务：**

- 搭建一个最小 2-Agent 系统（User Proxy + Assistant）
- 在 5-10 个任务上对比：
  - 单 Agent
  - 2-Agent 协作
- 记录：
  - 任务成功率
  - 对话轮次
  - token 成本

**产出：**

- `12_multi_agent_baseline.py`
- 1 份 mini memo：单 Agent vs 多 Agent 的效果与成本

### W14（6/30-7/6）：MetaGPT 精读 + 角色分工实验

**实验任务：**

- 实现一个简单角色分工框架（规划者 + 执行者 + 检验者）
- 对比：
  - 无角色分工（一个 Agent 全做）
  - 有角色分工
- 重点看：
  - 任务成功率变化
  - 角色间通信成本
  - 分工是否带来实质提升

**产出：**

- `13_role_division_agent.py`
- 1 张角色分工效果结果表

### W15（7/7-7/13）：协调模式对比 + 通信效率分析

**实验任务：**

- 对比不同协调模式：
  - 顺序执行（Agent A → Agent B → Agent C）
  - 并行执行后汇总
  - 辩论式协调（Agent 互相质疑和修正）
- 分析：
  - 哪种模式最高效
  - 哪种模式最贵
  - 协调失败的主要原因

**产出：**

- `14_coordination_pattern_ablation.py`
- 1 张 `coordination mode vs success vs cost` 图

### W16（7/14-7/20）：综合项目

**项目主题建议：**

`LLM Agent Architecture Ablation: Planning, Memory, Tools, and Multi-Agent Coordination`

**实验要求：**

- 至少 3 个架构组件的消融
- 至少 2 类协调模式
- 同时比较任务指标与效率指标

**产出：**

- `15_agent_architecture_project/`
- `15_agent_architecture_report.md`

**月末检查点：**

- [ ] 你不再只是用现成框架，而是在做架构组件的可控对比研究
- [ ] 你有至少 2 个可展示项目
- [ ] 你能清楚说出自己研究里的架构消融价值

---

## 8月：消融 + 技术报告 / pre-paper + Phase B 准备

### 本月目标

- 把架构效果、成本、失败模式整理成论文式叙事
- 形成 1 份综合技术报告或 pre-paper 草稿
- 让你的 GitHub 仓库达到可展示水平

### W17（7/21-7/27）：补关键 ablation 和边界条件

**任务：**

- 精选少量关键变量：
  - planning strategy
  - memory type
  - tool count
  - agent count
  - coordination pattern
- 做边界条件总结：
  - 哪些任务最耗 token
  - 哪些任务最需要记忆
  - 哪些任务单 Agent 反而更好

### W18（7/28-8/3）：仓库整理 + dataset card + README

**任务：**

- 给核心项目补：
  - README
  - 任务集说明
  - 运行方式
  - 主要图表
- 从零 clone 自测复现

### W19（8/4-8/10）：写综合技术报告 / pre-paper

**任务：**

- 选 1 条最强主线写成完整文档：
  - 问题
  - 相关工作
  - 方法
  - 系统设置
  - 实验结果
  - 效率分析
  - failure mode analysis
  - 局限

**产出：**

- `tech_report.md` 或 `pre_paper_draft.md`

### W20（8/11-8/17）：Phase B 衔接准备

**任务：**

- 复盘哪一条线最适合继续深挖：
  - planning strategy research
  - memory architecture design
  - tool use & API integration
  - multi-agent coordination
  - cross-component ablation

**产出：**

- Phase B 阅读清单
- 下一阶段问题定义草案

---

## 完整论文阅读进度表

| 周次 | 论文 | 精读/粗读 | 状态 |
|------|------|-----------|------|
| W1 | A Survey on Large Language Model-based Autonomous Agents | 精读 | |
| W2 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 精读 | |
| W3 | ReAct: Synergizing Reasoning and Acting in Language Models | 精读 | |
| W4 | AgentBench: Evaluating LLMs as Agents | 精读 | |
| W5 | Reflexion: Language Agents with Verbal Reinforcement Learning | 精读 | |
| W6 | Tree of Thoughts | 精读 | |
| W7 | MemGPT: Towards LLMs as Operating Systems | 精读 | |
| W8 | Generative Agents: Interactive Simulacra of Human Behavior | 精读 | |
| W9 | Toolformer: Language Models Can Teach Themselves to Use Tools | 精读 | |
| W10 | Gorilla: Large Language Model Connected with Massive APIs | 粗读 | |
| W11 | AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation | 精读 | |
| W12 | MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework | 精读 | |
| W13 | CAMEL: Communicative Agents for 'Mind' Exploration | 粗读 | |
| W14 | WebArena: A Realistic Web Environment for Building Autonomous Agents | 精读 | |
| W15 | Voyager: An Open-Ended Embodied Agent with Large Language Models | 粗读 | |
| W19 | The Rise and Potential of Large Language Model Based Agents: A Survey | 粗读复盘 | |

---

## 阶段 A 交付物 Checklist

- [ ] 一个稳定可复用的 `llm_agent_eval_harness`
- [ ] 1 个规划策略消融项目（CoT / ReAct / ToT / Reflexion）
- [ ] 1 个记忆与工具使用组件分析项目
- [ ] 至少 1 个多智能体协调实验切片
- [ ] 4-6 份 mini research memo
- [ ] 至少 10 张高密度压缩卡片
- [ ] 至少 1 轮 7 天闭卷回忆记录
- [ ] 1 份综合技术报告或 pre-paper 草稿
- [ ] 1 个可公开展示的 GitHub 仓库
- [ ] 至少 1 套 failure taxonomy / failure mode 文档
- [ ] 至少 1 套 `planning-cost` 联合图表
- [ ] 至少 1 个 `success-rate-per-token` 风格指标分析

---

## 费用预估（本地机器优先）

| 项目 | 费用 |
|------|------|
| 本地 LLM 推理（Ollama / vLLM + 本地模型） | 默认 0 额外 API 成本 |
| 可选 API 基线（GPT-4.x / Claude） | 约 $30-100 |
| 额外数据存储 / 少量云机应急 | 约 $0-50 |
| **合计** | **约 $30-150** |

> 这版计划默认你优先用本地机器做小规模、可重复实验，把 Agent 架构分析的优势用在"组件消融"和"效果-成本-失败模式"三位一体分析上，而不是急着去做大规模 Agent 演示。
