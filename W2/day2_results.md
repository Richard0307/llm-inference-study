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
