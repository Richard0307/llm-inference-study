# 2026 年 4 月逐日执行版（LLM Agent 架构）

> 适用前提：
> - 本地有一台 4090 笔记本
> - 4 月主线是 `基础设施 + Agent 架构认知 + ReAct/Reflexion 实现 + 最小评测框架`
> - 这份文档服务于新的 `LLM Agent 架构` 版 Phase A

---

## 4 月总目标

- 搭建稳定可用的本地 LLM Agent 研究实验环境
- 跑通最小 ReAct 和 Reflexion Agent
- 学会记录任务成功率和规划成本
- 完成一个最小的 Agent 规划策略对比项目
- 形成第一套可复用的 `llm_agent_eval_harness`
- 设计并初步实现 plan-action deviation detector（论文核心机制）
- 建立系统性的 Agent failure taxonomy（论文分析支撑）

---

## 本月固定规则

- 周一到周三：代码实验
- 周四：论文精读 + 高密度压缩
- 周五：整理图表、负结果、mini memo
- 周末：闭卷回忆 + 知乎/公众号输出 + 补 README

**每天最低产出：**

- 1 次 GitHub 提交
- 1 份实验记录或论文笔记
- 1 次闭卷回忆
- 1 个可执行小结果

---

## W1：4/1 - 4/6

### 4 月 1 日（周三）

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

### 4 月 4 日（周六）

- 主任务：轻量复盘，不做重编码
- 要做：
  - 整理前 3 天结果
  - 写出最常见的 3 类 Agent 失败原因
  - 补 1 页"为什么 Agent 研究需要架构视角"
- 当天输出：
  - 1 页总结笔记
  - 1 份 Agent 失败类型清单（草稿）
- 完成标准：
  - 能说出至少 3 类 Agent 失败模式

### 4 月 5 日（周日）

- 主任务：输出一篇轻量内容
- 要做：
  - 写知乎或笔记
  - 主题：LLM Agent 到底是什么，它的架构为什么重要
- 当天输出：
  - 1 篇对外内容草稿
- 完成标准：
  - 有 1 份外部可读内容

### 4 月 6 日（周一）

- 主任务：加入规划步数与 token 成本追踪
- 要做：
  - 给 Agent loop 加上 step count 和 token cost logging
  - 统一结果表头
  - 做 1 次 GitHub 提交
- 当天输出：
  - agent loop v2
  - 统一结果记录模板
- 完成标准：
  - 结果里同时有任务成功率和规划成本字段

---

## W2：4/7 - 4/13

### 4 月 7 日（周二）

- 主任务：CoT 与纯回答对比实验
- 要做：
  - 比较"直接回答"vs "链式推理（CoT）"
  - 固定样本、固定模型
  - **新增**：在每步记录 `planned_action` vs `actual_action`，为后续 deviation 分析积累数据
- 当天输出：
  - `01_planning_strategy_baseline.py`
  - 1 张对比表
- 完成标准：
  - 至少完成 2 组规划策略对照
  - 结果表中包含 plan-action alignment 字段

### 4 月 8 日（周三）

- 主任务：加入 ReAct 风格规划
- 要做：
  - 实现 Reason + Act 交替的最简版 ReAct
  - 三路对比：直接 / CoT / ReAct
  - 记录：步骤数、成功率、token 成本
- 当天输出：
  - 更新后的三路对比表
  - 1 段观察结论
- 完成标准：
  - 能回答"ReAct 比 CoT 多了什么，贵了多少"

### 4 月 9 日（周四）

- 主任务：精读 `ReAct`
- 要做：
  - 精读 `ReAct: Synergizing Reasoning and Acting in Language Models`
  - 重点看 Reason 和 Act 如何交替工作
  - 写 1 张高密度压缩卡
- 当天输出：
  - 结构化论文笔记
  - 1 张 ReAct 架构卡
- 完成标准：
  - 能解释 ReAct 为什么比纯 CoT 在工具使用任务上更好

### 4 月 10 日（周五）

- 主任务：整理 W2 mini memo
- 要做：
  - 汇总规划策略的成功率-成本结果
  - 写 1 份 mini memo
  - 记录 1 组负结果（哪种情况 ReAct 没有提升）
- 当天输出：
  - `01_planning_strategy_memo.md`
- 完成标准：
  - memo 里有问题、结果、解释、局限

### 4 月 11 日（周六）

- 主任务：闭卷回忆 + 补漏
- 要做：
  - 闭卷画出 CoT / ReAct 的控制流
  - 如果有脚本没整理完，今天补齐
- 当天输出：
  - 1 份回忆记录
- 完成标准：
  - 不看资料能讲 3 分钟

### 4 月 12 日（周日）

- 主任务：知乎 / 公众号输出
- 要做：
  - 写一篇关于"CoT 和 ReAct 的本质区别"的文章
- 当天输出：
  - 1 篇外部内容
- 完成标准：
  - 内容能让非研究者理解为什么规划策略对 Agent 很重要

### 4 月 13 日（周一）

- 主任务：准备进入 Reflexion 实验 + 偏差分析铺垫
- 要做：
  - 整理失败案例样本（ReAct 答错的）
  - 设计最小反思触发机制
  - 写下 2 个 Reflexion 实验的待验证假设
  - **新增**：对 W2 所有失败案例做 plan-action deviation 标注，形成初始偏差数据集
  - **新增**：写下 1 个关于"能否在失败发生前检测偏差"的假设
- 当天输出：
  - 失败案例清单（含 deviation 标注）
  - 假设文档（含 deviation detector 假设）
- 完成标准：
  - 你知道 W3 的实验从哪里开始
  - 你有一份带偏差标注的失败案例集

---

## W3：4/14 - 4/20

### 4 月 14 日（周二）

- 主任务：Reflexion 实现 v1
- 要做：
  - 实现最简单的"失败 → 生成反思 → 重试"循环
  - 在失败样本上测试
- 当天输出：
  - `02_reflexion_agent.py`
  - 第一轮反思效果表
- 完成标准：
  - 至少有 1 组"有反思 vs 无反思"结果

### 4 月 15 日（周三）

- 主任务：Reflexion 深化实验
- 要做：
  - 测试反思轮次（1次 / 2次 / 3次）
  - 记录：成功率、额外 token 成本、步骤数
- 当天输出：
  - 反思轮次对比表
- 完成标准：
  - 能对比"反思 N 次是否值得"

### 4 月 16 日（周四）

- 主任务：精读 `Reflexion`
- 要做：
  - 重点理解 verbal reinforcement 的机制
  - 写 1 张"失败信息如何转化为改进提示"的卡片
- 当天输出：
  - 结构化论文笔记
  - 1 张 Reflexion 机制卡
- 完成标准：
  - 能说出 Reflexion 和普通 retry 的本质区别

### 4 月 17 日（周五）

- 主任务：整理 W3 结果
- 要做：
  - 画第一张 `success rate vs retry count vs token cost` 图
  - 写 1 份 mini memo
- 当天输出：
  - 图表
  - `02_reflexion_memo.md`
- 完成标准：
  - 图里至少有 3 个 retry 条件的对比

### 4 月 18 日（周六）

- 主任务：闭卷回忆 + failure taxonomy（论文方向2核心产出）
- 要做：
  - 画出 Reflexion 的完整控制流
  - 总结 Agent 规划失败的主要类型
  - **新增**：对每类失败标注"是否可被在线检测"（detectable before failure / only post-hoc）
  - **新增**：区分 plan-level failure vs execution-level failure
- 当天输出：
  - 1 张控制流图
  - 1 份 Agent failure taxonomy 草稿（含可检测性标注）
- 完成标准：
  - 能说出至少 4 类 Agent 规划失败模式
  - 其中至少 2 类标注为"可在线检测"——这就是 deviation detector 的目标失败类型

### 4 月 19 日（周日）

- 主任务：写外部输出
- 要做：
  - 写一篇"Reflexion 是如何让 Agent 从失败中学习的"
- 当天输出：
  - 1 篇对外内容
- 完成标准：
  - 能让非本领域读者看懂自我校正机制的意义

### 4 月 20 日（周一）

- 主任务：三路规划策略综合对比（论文方向4核心产出）
- 要做：
  - 把 CoT / ReAct / Reflexion 放在同一任务集上重跑
  - 统一结果格式，做完整三路对比表
  - **新增**：统一记录每个 agent 的 plan-action deviation rate（偏差率）
  - **新增**：按 failure taxonomy 分类统计三种架构的失败分布差异
- 当天输出：
  - 三路对比结果表（含 deviation rate 列）
  - 1 张"架构 × 失败类型"热力图草稿
- 完成标准：
  - 有清楚的三路规划策略对比数据
  - 能回答"哪种架构在哪类失败上最脆弱"

---

## W4：4/21 - 4/27（论文核心机制周：Plan-Action Deviation Detector）

### 4 月 21 日（周二）

- 主任务：设计 deviation detector 架构
- 要做：
  - 定义 deviation 的形式化表示：`(planned_action, actual_action, context) → deviation_score`
  - 设计 detector 的两种候选方案：
    - 方案 A：rule-based（基于 action type mismatch + step count heuristic）
    - 方案 B：LLM-as-judge（用同一个或更小的 LLM 做在线判断）
  - 确定 detector 的输入输出接口
- 当天输出：
  - `03_deviation_detector_design.md`（含架构图）
  - detector 接口定义代码
- 完成标准：
  - 能画出 detector 嵌入 agent loop 的位置图

### 4 月 22 日（周三）

- 主任务：实现 deviation detector v1
- 要做：
  - 实现 rule-based detector（方案 A）
  - 实现 LLM-as-judge detector（方案 B）
  - 在 W3 的三路对比数据上回测：detector 能否在失败发生前识别偏差
- 当天输出：
  - `03_deviation_detector.py`
  - detector 回测准确率表（precision / recall）
- 完成标准：
  - detector 在已有数据上能检出 >50% 的失败前偏差信号

### 4 月 23 日（周四）

- 主任务：精读 process supervision 相关论文
- 要做：
  - 精读 `Let's Verify Step by Step`（OpenAI process reward model）
  - 对比 outcome supervision vs process supervision 的设计哲学
  - 思考你的 detector 和 process reward model 的关系与区别
- 当天输出：
  - 结构化论文笔记
  - 1 张"你的 detector vs PRM vs Reflexion"对比卡
- 完成标准：
  - 能回答"你的方法和 process reward model 有什么不同"（审稿人必问）

### 4 月 24 日（周五）

- 主任务：detector 集成实验 + 论文骨架
- 要做：
  - 把 detector 嵌入 ReAct 和 Reflexion 的 agent loop
  - 测试"检测到偏差后自动触发纠正"的效果
  - 写论文 1-page outline（Title / Abstract sketch / 5 sections）
- 当天输出：
  - detector-integrated agent loop 实验结果
  - `paper_outline_v1.md`
- 完成标准：
  - 有"无 detector vs 有 detector"的成功率对比数据

### 4 月 25 日（周六）

- 主任务：闭卷回忆 + 结果整理
- 要做：
  - 闭卷讲一遍论文故事线：问题 → 现有方法局限 → 你的 detector → 实验证据
  - 整理所有实验数据到统一格式
  - 画 1 张核心结果图（架构 × detector × 成功率/token成本）
- 当天输出：
  - 1 份回忆记录
  - 1 张论文核心结果图草稿
- 完成标准：
  - 能 5 分钟讲清论文的完整故事

### 4 月 26 日（周日）

- 主任务：写阶段性复盘 + 外部输出
- 要做：
  - 写一篇阶段复盘，主题：从 agent loop 到 plan-action monitor 的研究路径
  - 可同步写知乎文章："为什么 LLM Agent 需要一个'计划监督员'"
- 当天输出：
  - 1 篇阶段复盘
- 完成标准：
  - 文中能同时写出 detector 的效果和成本视角

### 4 月 27 日（周一）

- 主任务：4 月 mini project 收口
- 要做：
  - 重跑关键实验（四路对比：CoT / ReAct / Reflexion / Reflexion+Detector）
  - 补 1 份 mini report，按论文结构组织
  - 做本月核心提交
- 当天输出：
  - `03_april_report.md`（含论文 framing 视角）
- 完成标准：
  - 4 月成果可以作为论文实验部分的初稿素材

---

## W5（4/28 - 4/30，提前进入 5 月主题）

### 4 月 28 日（周二）

- 主任务：预读 `Tree of Thoughts` + detector 在 ToT 上的适配思考
- 要做：
  - 重点理解树状规划与线性规划的设计差异
  - 思考：deviation detector 在分支搜索场景下该如何工作
  - 写 1 张方法对比卡
- 当天输出：
  - 结构化论文笔记
  - 1 张规划策略扩展卡（含 detector 适配笔记）
- 完成标准：
  - 能讲清"为什么有时候树状规划比线性更好"
  - 能说出 detector 在 ToT 上的一个设计挑战

### 4 月 29 日（周三）

- 主任务：ToT 风格 baseline 预实现 + detector 扩展
- 要做：
  - 做一个最简版 BFS 思路链实验
  - 尝试把 detector 嵌入 ToT loop，形成第 5 路对比
- 当天输出：
  - `04_tree_planning_preview.py`
  - ToT + detector 初步结果（即使不完整也记录）
- 完成标准：
  - 跑通最简版树状规划

### 4 月 30 日（周四）

- 主任务：4 月总复盘 + 论文路线图 + 5 月开题
- 要做：
  - 写 4 月总结
  - **新增**：写论文 related work 草稿大纲（列出需要对比的 5-8 篇核心论文）
  - **新增**：明确 5 月目标——完善 detector 实验 + 写初稿
  - 写下 2 个 5 月待验证假设（关于 detector 的泛化性和成本效率）
- 当天输出：
  - `2026_04_monthly_review.md`
  - `2026_05_hypotheses.md`（聚焦 detector 论文）
  - `paper_related_work_outline.md`
- 完成标准：
  - 你知道 5 月要验证什么
  - 论文从"想法"变成了"有骨架的草稿"

---

## 4 月结束时你应该拿到什么

- 一个稳定可复用的最小 `llm_agent_eval_harness`
- 一个能跑的 Agent 规划策略对比框架（CoT / ReAct / Reflexion / ToT）
- 一套任务成功率 + 规划步数 + token 成本 + deviation rate 联合指标
- **一个 plan-action deviation detector 原型**（rule-based + LLM-as-judge 两版）
- **一组"无 detector vs 有 detector"的对比实验数据**
- 至少 4 份像样的实验记录或 mini memo
- 至少 1 套 Agent failure taxonomy 草稿（含可检测性标注）
- **1 份论文 outline + related work 大纲**

---

## 4 月执行原则

- 每天只追一个最小问题
- 每周至少留下一张图，而不是只留结论
- 每次做对照时，只改一个变量
- 每次结果整理时，都同时写：
  - 这个方法是不是更准
  - 它是不是更贵
  - 它是不是更值得
- **论文导向原则**：每个实验设计时问自己"这个结果能放进论文的哪个 section"
