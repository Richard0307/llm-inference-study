# Day 2 CoT vs Direct Strategy Comparison / Day 2 规划策略对比实验

## 中文版

### Summary
- 后端：`Local Qwen via vLLM OpenAI API`
- 模型：`Qwen/Qwen3-8B-AWQ`
- 总记录数：`20` （2 种策略 x 10 道题）
- 步数预算：`4`
- Direct：`8/10` 成功（`80.00%`），平均 token `659.7`
- CoT：`9/10` 成功（`90.00%`），平均 token `774.6`
- CoT 比 Direct 多花 token：`114.9`

### 成功任务
| Strategy | Task ID | Type | Steps | Tokens | Plan-Action Aligned |
|---|---|---|---|---|---|
| `direct` | `T1` | `calculator` | `2` | `476` | `False` |
| `direct` | `T2` | `calculator` | `2` | `446` | `False` |
| `direct` | `T3` | `search` | `2` | `472` | `False` |
| `direct` | `T4` | `search` | `2` | `437` | `False` |
| `direct` | `T6` | `search` | `2` | `425` | `False` |
| `direct` | `T7` | `calculator` | `2` | `411` | `False` |
| `direct` | `T8` | `search` | `2` | `448` | `False` |
| `direct` | `T9` | `search` | `2` | `430` | `False` |
| `cot` | `T1` | `calculator` | `2` | `608` | `False` |
| `cot` | `T2` | `calculator` | `2` | `592` | `False` |
| `cot` | `T3` | `search` | `2` | `645` | `False` |
| `cot` | `T4` | `search` | `2` | `578` | `False` |
| `cot` | `T5` | `search+calculate` | `3` | `1202` | `False` |
| `cot` | `T6` | `search` | `2` | `557` | `False` |
| `cot` | `T7` | `calculator` | `2` | `549` | `False` |
| `cot` | `T8` | `search` | `2` | `695` | `False` |
| `cot` | `T9` | `search` | `2` | `540` | `False` |

### 失败任务
| Strategy | Task ID | Type | Failure Reason | Steps | Tokens | Plan-Action Aligned |
|---|---|---|---|---|---|---|
| `direct` | `T5` | `search+calculate` | `tool_execution_error: Unsupported calculator expression: {'numbers': [4200, 3100], 'operation': 'sum'}` | `4` | `1533` | `False` |
| `direct` | `T10` | `search+calculate` | `tool_execution_error: Unsupported calculator expression: {'a': 4200, 'b': 3100}` | `4` | `1519` | `False` |
| `cot` | `T10` | `search+calculate` | `tool_execution_error: Unsupported calculator expression: {'a': 4200, 'b': 3100}` | `4` | `1780` | `False` |

### 分析
- Direct：8/10 成功，平均 `659.7` token
- CoT：9/10 成功，平均 `774.6` token
- CoT 多花了 `114.9` token，成功率的提升是否值得？
- plan-action 对齐：`0/20` 条（`0.00%`），低对齐率可能预示推理漂移
- 失败原因分布：`tool_execution_error: Unsupported calculator expression: {'numbers': [4200, 3100], 'operation': 'sum'}` x 1，`tool_execution_error: Unsupported calculator expression: {'a': 4200, 'b': 3100}` x 2

---

## English Version

### Summary
- Backend: `Local Qwen via vLLM OpenAI API`
- Model: `Qwen/Qwen3-8B-AWQ`
- Total records: `20` (2 strategies x 10 tasks)
- Step budget: `4`
- Direct: `8/10` success (`80.00%`), avg `659.7` tokens
- CoT: `9/10` success (`90.00%`), avg `774.6` tokens
- CoT extra token cost vs Direct: `114.9`

### Success Table
| Strategy | Task ID | Type | Steps | Tokens | Plan-Action Aligned |
|---|---|---|---|---|---|
| `direct` | `T1` | `calculator` | `2` | `476` | `False` |
| `direct` | `T2` | `calculator` | `2` | `446` | `False` |
| `direct` | `T3` | `search` | `2` | `472` | `False` |
| `direct` | `T4` | `search` | `2` | `437` | `False` |
| `direct` | `T6` | `search` | `2` | `425` | `False` |
| `direct` | `T7` | `calculator` | `2` | `411` | `False` |
| `direct` | `T8` | `search` | `2` | `448` | `False` |
| `direct` | `T9` | `search` | `2` | `430` | `False` |
| `cot` | `T1` | `calculator` | `2` | `608` | `False` |
| `cot` | `T2` | `calculator` | `2` | `592` | `False` |
| `cot` | `T3` | `search` | `2` | `645` | `False` |
| `cot` | `T4` | `search` | `2` | `578` | `False` |
| `cot` | `T5` | `search+calculate` | `3` | `1202` | `False` |
| `cot` | `T6` | `search` | `2` | `557` | `False` |
| `cot` | `T7` | `calculator` | `2` | `549` | `False` |
| `cot` | `T8` | `search` | `2` | `695` | `False` |
| `cot` | `T9` | `search` | `2` | `540` | `False` |

### Failure Table
| Strategy | Task ID | Type | Failure Reason | Steps | Tokens | Plan-Action Aligned |
|---|---|---|---|---|---|---|
| `direct` | `T5` | `search+calculate` | `tool_execution_error: Unsupported calculator expression: {'numbers': [4200, 3100], 'operation': 'sum'}` | `4` | `1533` | `False` |
| `direct` | `T10` | `search+calculate` | `tool_execution_error: Unsupported calculator expression: {'a': 4200, 'b': 3100}` | `4` | `1519` | `False` |
| `cot` | `T10` | `search+calculate` | `tool_execution_error: Unsupported calculator expression: {'a': 4200, 'b': 3100}` | `4` | `1780` | `False` |

### Analysis
- Direct: 8/10 success, avg `659.7` tokens.
- CoT: 9/10 success, avg `774.6` tokens.
- CoT costs `114.9` more tokens on average — is the accuracy gain worth it?
- Plan-action aligned in `0/20` records (`0.00%`). Low alignment may signal reasoning drift.
- Failure breakdown: `tool_execution_error: Unsupported calculator expression: {'numbers': [4200, 3100], 'operation': 'sum'}` x 1, `tool_execution_error: Unsupported calculator expression: {'a': 4200, 'b': 3100}` x 2

---
## 技术分析 / Technical Deep-dive

### 1. Plan-Action Aligned 为什么全是 False？

#### 什么是 Plan-Action Aligned

Plan-Action Aligned 检测的是：**模型在 thought 字段里"说要做的事"，和它实际执行的 action，是否一致。**

检测逻辑位于 `day2.py:505`：

```python
"aligned": action in thought.lower()
```

它检查 action 字符串（如 `"calculate"`、`"search_notes"`、`"finish"`）是否作为**精确子串**出现在 thought 的小写文本中。

#### 为什么 0/20 全是 False —— 举例说明

以 T1（计算 17×9−25）为例，模型返回的 JSON 可能是：

```json
{
  "thought": "I need to compute 17 * 9 - 25",
  "action": "calculate",
  "action_input": "17 * 9 - 25",
  "final_answer": ""
}
```

对齐检测：`"calculate" in "i need to compute 17 * 9 - 25"` → **False**

模型用的是 "compute"，不是 "calculate"。代码要求 thought 里出现**工具名的精确拼写**，但模型写的是自然语言同义词。

如果模型恰好写成：

```json
{ "thought": "I should use the calculate tool for this" }
```

那 `"calculate" in "i should use the calculate tool for this"` → **True**

但 Qwen3 在 `/no_think` 模式下 thought 字段通常很简短，不会刻意写出工具名。

对于 `search_notes` 更不可能匹配——模型会写 "search the notes" 或 "look up"，不会写出带下划线的 `"search_notes"`。

#### 结论

**0/20 全 False 是检测方法的 bug，不是模型推理漂移。** 模型实际上在正确地规划和执行，只是 thought 里用的是自然语言同义词而非精确工具名。要修复这个指标，应该使用语义关键词匹配替代精确子串匹配。

---

### 2. 这个实验实际体现了什么

从结果看，这个实验有效验证了两件事：

**a) CoT 比 Direct 推理成本更高**

CoT 平均消耗 `774.6` token，Direct 平均消耗 `659.7` token，CoT 多花 17.4%。CoT 的 system prompt 要求模型"写出完整推理链"，这直接导致 thought 字段更长、输出 token 更多。在成功率仅从 80% → 90%（多对了 1 个 task）的情况下，这个 token 开销是否值得取决于任务的容错要求。

**b) Agent Loop 实现完整**

实验完整实现了 决策→工具调用→观察→再决策 的 agent loop，包括步数预算、强制工具使用、错误记录和结果评估。

#### Agent Loop 的意义

Agent Loop 解决的核心问题是：**LLM 单次调用做不了的事。**

单次 LLM 调用 = 输入→输出，结束。它没法查外部数据、执行可靠计算、或根据中间结果决定下一步。Agent Loop 让 LLM 分步操作：

```
LLM 决策 → 调工具 → 拿到结果 → 再决策 → 再调工具 → ... → 最终回答
```

在本实验中，T5 和 T10 就是最简单的例子：先 `search_notes` 拿到预算数字（LLM 自身不知道这个数据），再 `calculate` 算总和（LLM 算数不可靠）。没有 loop，模型只能猜答案。

但本实验的任务太简单（最多 2 步工具调用），没有充分体现 loop 的价值。Agent Loop 真正的威力在于更复杂的场景：多轮工具调用中间结果决定走向、工具报错后的错误恢复、复杂任务的逐步分解等。

---

### 3. 失败任务分析：为什么 search+calculate 组合任务会失败

三个失败 task（Direct T5、Direct T10、CoT T10）的失败原因完全相同——`tool_execution_error`：

```
Unsupported calculator expression: {'numbers': [4200, 3100], 'operation': 'sum'}
Unsupported calculator expression: {'a': 4200, 'b': 3100}
```

**根因：** 模型把 `action_input` 写成了 JSON 对象（如 `{'a': 4200, 'b': 3100}`），但 `safe_calculate()` 期望的是纯数学表达式字符串（如 `"4200 + 3100"`）。这是模型对工具接口格式的理解错误，不是推理能力问题——模型正确找到了两个预算数字，只是用错了输入格式。

**为什么只在 search+calculate 组合任务失败？** 单步 calculate 任务（T1、T2、T7）的 question 本身就包含 `"17 * 9 - 25"` 这样的数学表达式，模型直接照抄即可。但组合任务中，模型需要自己构造 calculator 输入，它倾向于把搜索结果封装成结构化 JSON，而非写一个简单的算术表达式。

**改进方向：** 在 prompt 中明确声明工具输入格式（如 `"calculate accepts a math expression string like '4200 + 3100'"`），或加入 few-shot 示例展示正确用法。

