# LLM Agent 架构精读论文路线图

> 这版路线图的明确定位是：
>
> **系统性研究 LLM Agent 的架构设计原理——规划、记忆、工具使用、多智能体协调——并用组件消融的研究视角，分析哪些架构设计真正决定了 Agent 的能力上限。**
>
> 你后面最想讲出的科研故事应该是：
>
> - 我做的不是部署一个 Agent 应用
> - 我做的是 Agent 架构的系统性研究
> - 我不只看任务成功率，也看规划步数、工具调用准确性、记忆命中率、token 成本和失败模式

---

## 这条路线的论文阅读原则

这版路线图分成两条并行线：

### 任务主线

围绕：

- Agent 规划模块（planning & reasoning）
- Agent 记忆系统（memory architecture）
- 工具使用与 API 集成（tool use & function calling）
- 多智能体协作（multi-agent coordination）

### 系统辅助线

围绕：

- Agent 评测基准与框架
- 长链路任务中的架构效率
- 失败分析与自我校正机制
- token 成本与规划步数的 tradeoff

也就是说，你读系统论文，不是为了单纯优化推理速度，而是为了让你对 Agent 架构的研究更有实验依据。

---

## 核心论文清单

### 一、评测与全景图

1. **A Survey on Large Language Model-based Autonomous Agents** (Wang et al., 2024)
   这是路线图的总地图。你要用它建立 LLM Agent 的架构版图——感知、规划、记忆、行动——并判断哪些组件研究已经饱和、哪些架构问题还适合单人深挖。

2. **The Rise and Potential of Large Language Model Based Agents: A Survey** (Xi et al., 2023)
   这篇对研究全景的描绘更偏问题导向。它帮你从"能做什么"切到"为什么能做、为什么不能做"，是建立研究问题意识的好入口。

3. **AgentBench: Evaluating LLMs as Agents** (Liu et al., 2023)
   这是 Agent 评测领域的基础基准论文。你需要它建立"评测意识"：Agent 研究为什么不能只靠主观感受，而需要覆盖多任务、多环境的系统化测试。

4. **WebArena: A Realistic Web Environment for Building Autonomous Agents** (Zhou et al., 2023)
   这篇帮你进入真实环境评测。Web 任务的完成需要工具使用、规划、错误恢复的完整协作，非常适合用来检验 Agent 架构的综合能力。

---

### 二、规划与推理模块

5. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022)
   这是 Agent 规划模块的基础论文。你需要先把 CoT 的原理吃透，才能真正理解后续 ReAct、ToT、Reflexion 的"在 CoT 之上加了什么"。

6. **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2023)
   这是你的核心精读论文之一。ReAct 把推理和行动交替融入一个统一框架，直接奠定了现代 LLM Agent 的基础形态。理解它，就理解了 Agent 为什么不是单纯的 chain-of-thought。

7. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** (Yao et al., 2023)
   这篇帮你理解"从单路线推理到树状搜索规划"的升级思路。它很适合你在研究中讨论"规划深度 vs 规划成本"的 tradeoff。

8. **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023)
   这是你最需要精读的自我校正类论文。它把失败经验转化为文本反馈、再用于下次决策，是一种非常干净的"无需梯度的 Agent 自我改进"机制。

---

### 三、记忆系统

9. **MemGPT: Towards LLMs as Operating Systems** (Packer et al., 2023)
   这篇帮你理解 Agent 如何突破有限上下文窗口，实现更长期的记忆管理。它类比操作系统的分页机制，是研究记忆架构设计的关键参照。

10. **Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., 2023)
    这篇展示了把记忆、规划、反思三者合一的完整 Agent 架构。虽然应用场景是社交模拟，但它对记忆检索、重要性评分、反思触发机制的设计，是你研究记忆模块时最值得参考的工程实现。

---

### 四、工具使用与 API 集成

11. **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023)
    这是工具使用方向的经典论文。它让模型在自监督框架下学会何时调用工具，非常适合你研究"工具使用能力的来源与边界"。

12. **Gorilla: Large Language Model Connected with Massive APIs** (Patil et al., 2023)
    这篇帮你理解大规模 API 集成场景下的工具选择问题。它的核心贡献是把"使用哪个 API"变成一个可评测的检索和生成任务。

---

### 五、多智能体系统

13. **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** (Wu et al., 2023)
    这是多智能体方向最重要的框架论文之一。你要从中理解 Agent 之间如何通过对话协作、如何分配角色、如何处理分歧。

14. **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework** (Hong et al., 2023)
    这篇把软件工程的角色分工引入多智能体系统，每个 Agent 扮演特定职能角色。它是研究"多智能体分工设计"的极好案例。

15. **CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society** (Li et al., 2023)
    这篇从角色扮演切入，研究两个 Agent 之间的自主协作。它的双角色对话框架很适合用来分析 Agent 协作中的信息传递效率和任务漂移问题。

---

### 六、具身与长链路任务

16. **Voyager: An Open-Ended Embodied Agent with Large Language Models** (Wang et al., 2023)
    这篇展示了 Agent 在开放式长链路任务中如何积累技能、更新记忆、不断进化。它对研究"Agent 如何在无明确终止条件的场景中持续学习"非常有参考价值。

---

## 个人优先级

### 第一优先级（必须精读并能讲出来）

- 1 `A Survey on Large Language Model-based Autonomous Agents`
- 3 `AgentBench`
- 5 `Chain-of-Thought Prompting`
- 6 `ReAct`
- 8 `Reflexion`
- 13 `AutoGen`

这几篇决定你能不能真正站稳三件事：

- Agent 架构的完整视图
- 规划与推理机制的核心原理
- 多智能体协作的基本框架

### 第二优先级（最好精读）

- 2 `The Rise and Potential of LLM-Based Agents Survey`
- 7 `Tree of Thoughts`
- 9 `MemGPT`
- 10 `Generative Agents`
- 11 `Toolformer`
- 14 `MetaGPT`

这几篇会帮你把问题从"会搭一个 Agent"升级成"知道下一步往哪里做 paper"。

### 第三优先级（粗读够用，但非常有叙事价值）

- 4 `WebArena`
- 12 `Gorilla`
- 15 `CAMEL`
- 16 `Voyager`

这四篇属于你的"任务场景与工具生态支撑论文"。它们不是你的主战场，但会让你的科研故事从普通 Agent 应用变成 `architecture-aware LLM Agent research`。

---

## 推荐阅读顺序

### 阶段 1：先把 Agent 架构脑子装上

1. `A Survey on Large Language Model-based Autonomous Agents`
2. `The Rise and Potential of Large Language Model Based Agents`
3. `AgentBench`
4. `Chain-of-Thought Prompting`

这四篇读完，你会先获得：

- LLM Agent 的架构全图
- 主流评测基准的理解
- 推理模块的基础认知

### 阶段 2：切进规划与自我校正主线

5. `ReAct`
6. `Tree of Thoughts`
7. `Reflexion`

这三篇读完，你会知道：

- Agent 规划的核心范式
- 为什么单路推理不够，树搜索何时有价值
- Agent 如何从失败中自我改进

### 阶段 3：深入记忆与工具使用

8. `MemGPT`
9. `Generative Agents`
10. `Toolformer`
11. `Gorilla`

这四篇读完，你会更清楚：

- 记忆系统的设计空间
- 工具使用如何扩展 Agent 能力边界
- 长期记忆与上下文窗口的 tradeoff

### 阶段 4：走向多智能体协作

12. `AutoGen`
13. `MetaGPT`
14. `CAMEL`

这三篇会帮你把研究叙事从"单 Agent 架构"扩成"多智能体协作系统设计"。

### 阶段 5：扩到真实任务场景

15. `WebArena`
16. `Voyager`

这两篇帮你：

- 理解 Agent 在真实环境中的长链路评测方法
- 把 Agent 架构研究落地到具体可测试的任务场景

---

## 论文和 Phase A 的周次对齐

| 周次 | 论文 | 作用 |
|------|------|------|
| W1 | A Survey on LLM-based Autonomous Agents | 建立 Agent 架构全图 |
| W2 | Chain-of-Thought Prompting | 建立规划推理基础 |
| W3 | ReAct | 进入 reason+act 核心框架 |
| W4 | AgentBench | 建立 Agent 评测基础 |
| W5 | Reflexion | 深入自我校正机制 |
| W6 | Tree of Thoughts | 研究树状规划策略 |
| W7 | MemGPT | 进入记忆系统主线 |
| W8 | Generative Agents | 记忆+规划+反思完整架构 |
| W9 | Toolformer | 建立工具使用原理认知 |
| W10 | Gorilla | 大规模 API 集成与工具选择 |
| W11 | AutoGen | 切入多智能体协作 |
| W12 | MetaGPT | 多智能体角色分工设计 |
| W13 | CAMEL | 双 Agent 协作与通信分析 |
| W14 | WebArena | 真实环境评测框架 |
| W15 | Voyager | 开放式长链路任务 |
| W19 | The Rise and Potential Survey | 做阶段复盘 |

---

## 最适合你写成 paper 的三种模板

### 模板 A：规划策略消融研究

**题型：**
给定同一任务集，系统比较不同规划策略的任务成功率、规划步数、token 成本和失败模式。

**为什么适合你：**

- 任务干净，变量单一
- baseline 明确（CoT vs ReAct vs ToT vs Reflexion）
- 很适合把"效果-成本"框架放进 Agent 研究

**贡献长相：**

- 规划策略对照实验
- 失败模式分类
- 成本-效果联合分析
- 任务类型与最优规划策略的映射

### 模板 B：记忆架构效率分析

**题型：**
比较：

- 无记忆 Agent
- 短期记忆（in-context）
- 外部长期记忆（MemGPT 风格）
- 反思式记忆（Generative Agents 风格）

同时报告：

- 任务成功率
- 记忆命中率
- 平均 token 成本
- 跨轮次任务中的一致性

**为什么适合你：**

- 既有架构深度，又有实验可操作性
- 不需要大量算力，适合单人推进

### 模板 C：多智能体协调模式对比

**题型：**
系统比较不同多智能体协调范式（顺序、并行、角色分工、辩论式）的：

- 任务完成质量
- 通信轮次
- 一致性与漂移问题
- 角色分工的有效性

这非常适合做成"协调成本 vs 任务收益"的联合分析。

---

## 暂时不要优先读的东西

对你现在这条线来说，以下内容暂时不该抢主线：

- 纯 RLHF / 对齐训练论文
- 纯长上下文扩展论文
- 纯 RAG 系统论文（除非直接服务于 Agent 记忆）
- 无 Agent 架构视角的纯应用论文
- 没有组件消融的整体 Agent 演示论文

原因很简单：

你现在最需要的是 **架构分析能力、评测设计能力、组件消融能力、失败模式分类能力**，而不是继续扩散注意力。

---

## 一句话总结

这份路线图的核心不是"读最火的 Agent 论文"，而是：

**围绕规划、记忆、工具使用、多智能体协调这些核心 Agent 架构组件，建立一套同时关注任务效果、token 成本、失败分析和架构消融的系统性 LLM Agent 研究能力。**
