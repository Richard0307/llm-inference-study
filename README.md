# llm-inference-study
# ### 4 月 1 日（周三）

- 主任务：搭起最小 Agent 研究实验地基
- 要做：
  - 整理本地 Python / LLM 后端环境信息
  - 选定 1 个本地或 API LLM 作为 4 月 baseline
  - 确定 Agent 任务样本来源（简单推理或工具使用任务）
- 当天输出：
  - 1 份环境记录
  - 1 份 baseline 候选模型清单
- 完成标准：
  - 知道自己 4 月准备用哪个模型和任务集

### 4 月 2 日（周四）

- 主任务：读 Agent 架构总览
- 要做：
  - 精读 `A Survey on Large Language Model-based Autonomous Agents`
  - 重点记住 Agent 的四个核心组件：感知、规划、记忆、行动
  - 写 1 页 Agent 架构总图
- 当天输出：
  - 结构化论文笔记
  - 1 张 Agent 架构组件卡
- 完成标准：
  - 能口头解释 LLM Agent 的四大组件及其关系

### 4 月 3 日（周五）

- 主任务：实现最小 Agent Loop
- 要做：
  - 写一个最简单的 Thought → Action → Observation 循环
  - 跑 5-10 个最小推理任务
  - 记录：任务成功率、步骤数、token 消耗
- 当天输出：
  - `00_basic_agent_loop.py`
  - 1 张最小结果表
- 完成标准：
  - Agent loop 跑通，能记录基础指标