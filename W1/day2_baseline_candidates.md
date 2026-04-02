# Day 2 baseline 候选清单 / Day 2 Baseline Candidates

## 中文版

### 1. 推荐的 4 月 baseline
- 模型：`claude-sonnet-4-20250514`
- 显示名称：`Claude Sonnet API`
- 后端：`Anthropic API`
- 推荐理由：你已经在 llm-infer 中配置了 Anthropic key，而且 anthropic SDK 已安装。对当前 4 月 baseline 来说，这是最快能稳定开跑的 API 路径。

### 2. baseline 候选模型清单
| 模型 ID | 类型 | 后端 | 当前是否可用 | 优势 | 约束 | 备注 |
|---|---|---|---|---|---|---|
| `claude-sonnet-4-20250514` | api | `Anthropic API` | 是 | 对规划、推理和代码任务表现稳定；你已经在 llm-infer 环境里配置好了 Anthropic key，接入路径最短 | 存在 API 成本和网络依赖；为了可复现，需要固定模型 ID、temperature 和 max tokens | Anthropic SDK 已安装，且 ANTHROPIC_API_KEY 已配置 |
| `Qwen/Qwen2.5-7B-Instruct` | local | `vLLM or transformers` | 是 | 适合作为本地开源 baseline；中文和英文指令跟随能力都比较平衡 | 首次运行前仍需下载模型权重；16 GB 显存下要控制上下文长度和 batch 设置 | 本地 CUDA 栈可用，vLLM 已安装，但首次运行仍需下载权重 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | local | `vLLM or transformers` | 是 | 常见的英文开源 instruct baseline；便于和公开 agent 示例做横向比较 | 可能需要额外的模型访问授权；中文体验通常不如 Qwen 舒服 | 本地 CUDA 栈可用，vLLM 已安装，但首次运行仍需下载权重 |
| `gpt-4.1` | api | `OpenAI API` | 否 | 常见的强 API baseline；后续做 agent 原型时集成路径清晰 | 当前 OPENAI_API_KEY 还没有配置；存在 API 成本和网络依赖 | 需要同时满足 OpenAI SDK 已安装且 OPENAI_API_KEY 已配置 |

### 3. Agent 任务样本来源选项
#### 本地推理微型任务集
- 来源：仓库内自定义任务
- 规模：5 个短推理任务
- 保留理由：不依赖外部数据，最容易先拿到可重复 baseline
- 例子：
- 两步算术文字题
- 日历与日期推理
- 给定上下文的简短多跳推理

#### 本地工具使用微型任务集
- 来源：仓库内自定义任务
- 规模：5 个工具使用任务
- 保留理由：和 day 3 的 Thought -> Action -> Observation 循环直接对齐
- 例子：
- 调用计算器
- 读取或检索本地文件
- 字符串抽取或简单表格查找

#### 公共 benchmark 小切片
- 来源：GSM8K、HotpotQA 或 AgentBench 风格任务的小样本切片
- 规模：等本地微型任务集稳定后再接入
- 保留理由：后续做对外对比有价值，但第一周不必一开始就上重 benchmark
- 例子：
- 5 个 GSM8K 风格题目
- 5 个轻量 AgentBench 风格工具任务

### 4. 最终确定的 4 月任务集
- 任务集名称：`april_agent_micro_set_v1`
- 任务来源：`仓库内自定义本地任务`
- 组成：5 个简单推理任务
- 组成：5 个带计算器和本地文本检索的工具使用任务
- 选择理由：这个任务集最容易先跑通最小 agent loop，也方便记录 success rate、step count 和 token cost。

---

## English Version

### 1. Recommended April Baseline
- Model: `claude-sonnet-4-20250514`
- Display name: `Claude Sonnet API`
- Backend: `Anthropic API`
- Reason: Your Anthropic key is already configured in llm-infer and the anthropic SDK is installed. For the April baseline, this is the fastest stable API path to get running.

### 2. Candidate Model List
| Model ID | Type | Backend | Ready now | Strengths | Constraints | Note |
|---|---|---|---|---|---|---|
| `claude-sonnet-4-20250514` | api | `Anthropic API` | yes | Strong and stable on planning, reasoning, and coding tasks; Your Anthropic key is already configured in llm-infer, so setup friction is low | Has API cost and network dependency; For reproducibility, keep model ID, temperature, and max tokens fixed | Anthropic SDK is installed and ANTHROPIC_API_KEY is configured |
| `Qwen/Qwen2.5-7B-Instruct` | local | `vLLM or transformers` | yes | A solid open local baseline; Balanced bilingual instruction-following for Chinese and English | Model weights still need to be downloaded before the first run; On 16 GB VRAM, context length and batch settings should stay conservative | Local CUDA stack is ready and vLLM is installed, but model weights still need downloading |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | local | `vLLM or transformers` | yes | A common open English-heavy instruct baseline; Easy to compare against public agent examples | May require model access approval; Usually less comfortable than Qwen on Chinese prompts | Local CUDA stack is ready and vLLM is installed, but model weights still need downloading |
| `gpt-4.1` | api | `OpenAI API` | not yet | A common strong API baseline; Clear integration path for later agent prototyping | OPENAI_API_KEY is not configured right now; Has API cost and network dependency | Requires both the OpenAI SDK and OPENAI_API_KEY |

### 3. Task Sample Source Options
#### Local reasoning micro-set
- Source: Self-authored tasks stored in the repo
- Scale: 5 short reasoning tasks
- Why keep it: Fastest way to get a repeatable baseline without extra external dependencies
- Examples:
- Two-step arithmetic word problems
- Calendar and date reasoning
- Short multi-hop reasoning with provided context

#### Local tool-use micro-set
- Source: Self-authored tasks stored in the repo
- Scale: 5 tool-use tasks
- Why keep it: Directly aligned with the day-3 Thought -> Action -> Observation loop
- Examples:
- Calculator calls
- Local file read or search
- String extraction or simple table lookup

#### Public benchmark slices
- Source: Small curated slices from GSM8K, HotpotQA, or AgentBench-style tasks
- Scale: Add after the local micro-set becomes stable
- Why keep it: Useful later for external comparisons, but too heavy for the first week baseline
- Examples:
- 5 GSM8K-style questions
- 5 lightweight AgentBench-style tool tasks

### 4. Final April Task Set
- Task set name: `april_agent_micro_set_v1`
- Task source: `Self-authored local tasks stored in the repo`
- Composition: 5 simple reasoning tasks
- Composition: 5 tool-use tasks with calculator and local text search
- Why this set: This task set is the easiest starting point for a minimal agent loop and makes it easy to track success rate, step count, and token cost.
