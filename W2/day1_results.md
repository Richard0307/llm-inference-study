# Day 1 Minimal Agent Loop Results / Day 1 最小 Agent Loop 结果

## 中文版

### Summary
- 后端：`Local Qwen via vLLM OpenAI API`
- 模型：`Qwen/Qwen3-8B-AWQ`
- 任务总数：`10`
- 成功数：`8`
- 失败数：`2`
- 成功率：`80.00%`
- 平均步骤数：`2.00`
- 平均 token 消耗：`637.60`
- 平均每步 token 消耗：`318.80`
- 当前步数预算：`2`
- 设计目标失败率：`20.00%`

### Success Table
| Task ID | Type | Question | Expected | Final Answer | Steps | Tool Calls | Tokens |
|---|---|---|---|---|---|---|---|
| `T1` | `calculator` | Use the calculator tool to compute 17 * 9 - 25. | `128` | `128` | `2` | `1` | `623` |
| `T2` | `calculator` | Use the calculator tool to compute (144 / 12) + 7. | `19` | `19` | `2` | `1` | `633` |
| `T3` | `search` | From the notes tool, which city hosts the 2026 Research Summit? | `hangzhou` | `Hangzhou` | `2` | `1` | `660` |
| `T4` | `search` | From the notes tool, what is the codename of Project Aurora? | `northstar` | `Northstar` | `2` | `1` | `613` |
| `T6` | `search` | Use the notes tool to answer: on what day is demo day? | `thursday` | `Demo day is on Thursday.` | `2` | `1` | `615` |
| `T7` | `calculator` | Use the calculator tool to compute 81 / 9 + 6. | `15` | `15` | `2` | `1` | `620` |
| `T8` | `search` | From the notes tool, what is Team Blue budget? | `3100` | `3100` | `2` | `1` | `602` |
| `T9` | `search` | From the notes tool, in which month is Project Aurora reviewed internally? | `september` | `September` | `2` | `1` | `619` |

### Failure Table
| Task ID | Type | Question | Expected | Final Answer | Failure Reason | Steps | Tool Calls | Tokens |
|---|---|---|---|---|---|---|---|---|
| `T5` | `search+calculate` | Use the notes tool to find Team Red and Team Blue budgets, then compute their total with the calculator tool. You must use both tools. | `7300` | `N/A` | `step_budget_exceeded_before_finish` | `2` | `2` | `691` |
| `T10` | `search+calculate` | Use the notes tool to find Team Red and Team Blue budgets, then compute the budget difference with the calculator tool. You must use both tools. | `1100` | `N/A` | `step_budget_exceeded_before_finish` | `2` | `2` | `700` |

### Analysis
- 这份结果记录的是真实任务、真实工具调用和真实模型 token 消耗，不再是随机模拟器。
- 当前的 20% 失败率是有意设计出来的：我把 `max_steps` 设成较小的预算，同时放入了两道必须经历 `search_notes -> calculate -> finish` 三段流程的任务。
- `step_budget_exceeded_before_finish` 会导致失败，是因为模型虽然已经完成了检索和计算，但在当前预算内没有机会再发出第三步 `finish`，因此无法正式提交最终答案。
- 从这个角度看，这类失败不是“模型完全不会做”，而是“当前 loop 预算和控制流设计不足以完成多工具任务”。
- 本次失败原因分布：`step_budget_exceeded_before_finish` x 2
- 除了 `step_budget_exceeded_before_finish`，后续实验里还可能出现这些失败原因：
- `missing_required_actions`：模型提前结束，没有按要求调用必要工具。
- `incorrect_final_answer`：工具调用流程看起来对了，但最终答案仍然算错或抽取错。
- `tool_execution_error`：动作格式不合法，或者工具输入表达式错误，导致执行失败。
- JSON 解析失败或输出格式漂移：模型没有按约定返回结构化字段，loop 会被中断。
- 检索噪声与歧义：如果 notes 更长、更乱或有冲突信息，search 工具可能把模型带偏。

---

## English Version

### Summary
- Backend: `Local Qwen via vLLM OpenAI API`
- Model: `Qwen/Qwen3-8B-AWQ`
- Task count: `10`
- Success count: `8`
- Failure count: `2`
- Success rate: `80.00%`
- Average steps: `2.00`
- Average tokens: `637.60`
- Average tokens per step: `318.80`
- Step budget: `2`
- Designed failure rate target: `20.00%`

### Success Table
| Task ID | Type | Question | Expected | Final Answer | Steps | Tool Calls | Tokens |
|---|---|---|---|---|---|---|---|
| `T1` | `calculator` | Use the calculator tool to compute 17 * 9 - 25. | `128` | `128` | `2` | `1` | `623` |
| `T2` | `calculator` | Use the calculator tool to compute (144 / 12) + 7. | `19` | `19` | `2` | `1` | `633` |
| `T3` | `search` | From the notes tool, which city hosts the 2026 Research Summit? | `hangzhou` | `Hangzhou` | `2` | `1` | `660` |
| `T4` | `search` | From the notes tool, what is the codename of Project Aurora? | `northstar` | `Northstar` | `2` | `1` | `613` |
| `T6` | `search` | Use the notes tool to answer: on what day is demo day? | `thursday` | `Demo day is on Thursday.` | `2` | `1` | `615` |
| `T7` | `calculator` | Use the calculator tool to compute 81 / 9 + 6. | `15` | `15` | `2` | `1` | `620` |
| `T8` | `search` | From the notes tool, what is Team Blue budget? | `3100` | `3100` | `2` | `1` | `602` |
| `T9` | `search` | From the notes tool, in which month is Project Aurora reviewed internally? | `september` | `September` | `2` | `1` | `619` |

### Failure Table
| Task ID | Type | Question | Expected | Final Answer | Failure Reason | Steps | Tool Calls | Tokens |
|---|---|---|---|---|---|---|---|---|
| `T5` | `search+calculate` | Use the notes tool to find Team Red and Team Blue budgets, then compute their total with the calculator tool. You must use both tools. | `7300` | `N/A` | `step_budget_exceeded_before_finish` | `2` | `2` | `691` |
| `T10` | `search+calculate` | Use the notes tool to find Team Red and Team Blue budgets, then compute the budget difference with the calculator tool. You must use both tools. | `1100` | `N/A` | `step_budget_exceeded_before_finish` | `2` | `2` | `700` |

### Analysis
- These results come from real tasks, real tool calls, and real model token usage rather than a simulator.
- The current 20% failure rate is intentional: `max_steps` is kept small while two tasks require the full `search_notes -> calculate -> finish` chain.
- `step_budget_exceeded_before_finish` causes failure because the agent may already have searched and calculated, but it still needs one more turn to submit the final answer with `finish`.
- In other words, this is not necessarily a “model cannot solve the task” failure. It is a control-budget failure caused by the current loop design.
- Failure reason breakdown: `step_budget_exceeded_before_finish` x 2
- Other plausible failure modes in later experiments include:
- `missing_required_actions`: the model stops early without using a required tool.
- `incorrect_final_answer`: the tool flow looks reasonable, but the final answer is still wrong.
- `tool_execution_error`: the action format or tool input is invalid, so the tool call fails.
- JSON parsing or formatting drift: the model does not return the required structured output.
- Retrieval noise or ambiguity: longer or conflicting notes may push the search step off track.
