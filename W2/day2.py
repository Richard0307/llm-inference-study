#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pathlib
import re
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - local-qwen mode should still work
    Anthropic = None  # type: ignore[assignment]


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-8B-AWQ"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_BACKEND = os.getenv("DAY2_BACKEND", "local-qwen")
DEFAULT_MAX_STEPS = 4
DEFAULT_MAX_OUTPUT_TOKENS = 220
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "day2_results.md"


# 策略一：直接回答（不要多余推理）
DIRECT_PROMPT = """You are a helpful assistant.
Answer the question using the required tool, then finish immediately.
Reply with strict JSON only.
Use exactly these keys: thought, action, action_input, final_answer.
action must be one of: calculate, search_notes, finish.
Do not include markdown fences or explanation outside JSON.
When action is finish, put the answer in final_answer.
Keep thought to one short sentence."""

# 策略二：链式推理（先想清楚再行动）
COT_PROMPT = """You are a helpful assistant that reasons step by step.
Before acting, think carefully: what do you need to do, and why?
Reply with strict JSON only.
Use exactly these keys: thought, action, action_input, final_answer.
action must be one of: calculate, search_notes, finish.
Do not include markdown fences or explanation outside JSON.
When action is finish, put the answer in final_answer.
In the thought field, write out your full reasoning chain."""


REFERENCE_NOTES = [
    {
        "title": "summit_note",
        "content": (
            "The 2026 Research Summit will be hosted in Hangzhou. "
            "Team Red budget is 4200 dollars. Team Blue budget is 3100 dollars."
        ),
    },
    {
        "title": "aurora_note",
        "content": (
            "Project Aurora codename is Northstar. "
            "Its internal review month is September."
        ),
    },
    {
        "title": "lab_schedule",
        "content": (
            "Orientation starts on Monday. "
            "The robotics workshop lasts 3 days. "
            "Demo day is Thursday."
        ),
    },
]


@dataclass
class Task:
    task_id: str
    task_type: str
    question: str
    expected_answer: str
    required_actions: tuple[str, ...]


TASKS = [
    Task("T1", "calculator", "Use the calculator tool to compute 17 * 9 - 25.", "128", ("calculate",)),
    Task("T2", "calculator", "Use the calculator tool to compute (144 / 12) + 7.", "19", ("calculate",)),
    Task("T3", "search", "From the notes tool, which city hosts the 2026 Research Summit?", "hangzhou", ("search_notes",)),
    Task("T4", "search", "From the notes tool, what is the codename of Project Aurora?", "northstar", ("search_notes",)),
    Task(
        "T5",
        "search+calculate",
        "Use the notes tool to find Team Red and Team Blue budgets, then compute their total with the calculator tool. You must use both tools.",
        "7300",
        ("search_notes", "calculate"),
    ),
    Task("T6", "search", "Use the notes tool to answer: on what day is demo day?", "thursday", ("search_notes",)),
    Task("T7", "calculator", "Use the calculator tool to compute 81 / 9 + 6.", "15", ("calculate",)),
    Task("T8", "search", "From the notes tool, what is Team Blue budget?", "3100", ("search_notes",)),
    Task("T9", "search", "From the notes tool, in which month is Project Aurora reviewed internally?", "september", ("search_notes",)),
    Task(
        "T10",
        "search+calculate",
        "Use the notes tool to find Team Red and Team Blue budgets, then compute the budget difference with the calculator tool. You must use both tools.",
        "1100",
        ("search_notes", "calculate"),
    ),
]


class ToolError(RuntimeError):
    pass


class BackendError(RuntimeError):
    pass


class AgentBackend(Protocol):
    backend_name: str
    model_name: str

    # 接口声明也加上 system_prompt，保持和实现类一致
    def decide(
        self,
        task: Task,
        history: list[dict[str, str]],
        tool_used: bool,
        max_output_tokens: int,
        system_prompt: str = DIRECT_PROMPT,
    ) -> tuple[dict[str, Any], int]:
        ...


class AnthropicBackend:
    backend_name = "Anthropic API"

    def __init__(self, model_name: str, temperature: float) -> None:
        if Anthropic is None:
            raise BackendError("The anthropic package is not available in the current environment.")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise BackendError(
                "ANTHROPIC_API_KEY is not available in the current shell. "
                "Run inside `conda activate llm-infer` or use `conda run -n llm-infer ...`."
            )
        self.client = Anthropic()
        self.model_name = model_name
        self.temperature = temperature

    # system_prompt 从外面传进来，这样 Direct 和 CoT 可以用不同的"答题说明"调用同一个函数
    def decide(
        self,
        task: Task,
        history: list[dict[str, str]],
        tool_used: bool,
        max_output_tokens: int,
        system_prompt: str = DIRECT_PROMPT,
    ) -> tuple[dict[str, Any], int]:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_output_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(task=task, history=history, tool_used=tool_used),
                }
            ],
        )
        text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        raw_text = "\n".join(text_blocks)
        parsed = extract_json_block(raw_text)
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return parsed, tokens


class LocalQwenBackend:
    backend_name = "Local Qwen via vLLM OpenAI API"

    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float,
        timeout_seconds: int,
        api_key: str,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    # system_prompt 从外面传进来，这样 Direct 和 CoT 可以用不同的"答题说明"调用同一个函数
    def decide(
        self,
        task: Task,
        history: list[dict[str, str]],
        tool_used: bool,
        max_output_tokens: int,
        system_prompt: str = DIRECT_PROMPT,
    ) -> tuple[dict[str, Any], int]:
        user_prompt = build_user_prompt(task=task, history=history, tool_used=tool_used)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n/no_think"},
            ],
            "temperature": self.temperature,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            url=self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BackendError(f"Local Qwen request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                "Could not reach the local vLLM server. Start it first, for example:\n"
                "`bash W1/start_qwen3_8b_vllm.sh`"
            ) from exc

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"Unexpected local Qwen response payload: {data}") from exc

        raw_text = flatten_message_content(message)
        parsed = extract_json_block(raw_text)
        usage = data.get("usage", {})
        tokens = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        return parsed, tokens


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def safe_calculate(expression: str) -> str:
    node = ast.parse(expression, mode="eval")
    allowed_bin_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
    }
    allowed_unary_ops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def evaluate(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return evaluate(current.body)
        if isinstance(current, ast.Constant) and isinstance(current.value, (int, float)):
            return float(current.value)
        if isinstance(current, ast.BinOp) and type(current.op) in allowed_bin_ops:
            return allowed_bin_ops[type(current.op)](evaluate(current.left), evaluate(current.right))
        if isinstance(current, ast.UnaryOp) and type(current.op) in allowed_unary_ops:
            return allowed_unary_ops[type(current.op)](evaluate(current.operand))
        raise ToolError(f"Unsupported calculator expression: {expression}")

    result = evaluate(node)
    if not math.isfinite(result):
        raise ToolError(f"Non-finite calculator result for expression: {expression}")
    if float(result).is_integer():
        return str(int(result))
    return f"{result:.6f}".rstrip("0").rstrip(".")


def search_notes(query: str) -> str:
    query_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", query)}
    scored_matches: list[tuple[int, dict[str, str]]] = []
    for note in REFERENCE_NOTES:
        haystack = f"{note['title']} {note['content']}".lower()
        score = sum(1 for term in query_terms if term in haystack)
        if score > 0:
            scored_matches.append((score, note))

    if not scored_matches:
        return "No relevant note found."

    scored_matches.sort(key=lambda item: item[0], reverse=True)
    top_notes = scored_matches[:2]
    return " | ".join(f"{note['title']}: {note['content']}" for _, note in top_notes)


def run_tool(action: str, action_input: str) -> str:
    if action == "calculate":
        return safe_calculate(action_input)
    if action == "search_notes":
        return search_notes(action_input)
    raise ToolError(f"Unknown action: {action}")


def extract_json_block(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    if start == -1:
        raise ValueError(f"Model output did not contain JSON: {text}")
    # raw_decode 从第一个 { 开始解析，解析完第一个完整 JSON 对象就停止
    # 这样即使模型在 JSON 后面多输出了文字或第二个 JSON，也不会报错
    try:
        obj, _ = json.JSONDecoder().raw_decode(stripped, start)
        return obj
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output did not contain valid JSON: {text}") from exc


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace(",", "")
    lowered = re.sub(r"[^a-z0-9\.\-\s]", " ", lowered)
    return " ".join(lowered.split())


def extract_numeric_tokens(text: str) -> list[str]:
    return [match.replace(",", "") for match in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)]


def build_user_prompt(task: Task, history: list[dict[str, str]], tool_used: bool) -> str:
    history_lines = []
    for index, item in enumerate(history, start=1):
        history_lines.append(f"Step {index} thought: {item['thought']}")
        history_lines.append(f"Step {index} action: {item['action']}")
        history_lines.append(f"Step {index} action_input: {item['action_input']}")
        history_lines.append(f"Step {index} observation: {item['observation']}")
    if not history_lines:
        history_lines.append("No previous steps.")

    return "\n".join(
        [
            f"Task ID: {task.task_id}",
            f"Question: {task.question}",
            f"Tool already used: {'yes' if tool_used else 'no'}",
            "Previous loop history:",
            *history_lines,
            "Return the next JSON action now.",
        ]
    )


def evaluate_result(task: Task, final_answer: str) -> bool:
    normalized_expected = normalize_answer(task.expected_answer)
    normalized_final = normalize_answer(final_answer)

    if normalized_final == normalized_expected:
        return True

    expected_numbers = extract_numeric_tokens(task.expected_answer)
    final_numbers = extract_numeric_tokens(final_answer)
    if expected_numbers and any(number in final_numbers for number in expected_numbers):
        return True

    if normalized_expected and normalized_expected in normalized_final:
        return True

    return False


def build_failure_reason(
    task: Task,
    final_answer: str,
    stop_reason: str,
    action_history: list[str],
    tool_errors: list[str],
) -> tuple[bool, str]:
    answer_correct = bool(final_answer) and evaluate_result(task, final_answer)
    missing_actions = [action for action in task.required_actions if action not in action_history]

    if tool_errors:
        return False, f"tool_execution_error: {tool_errors[0]}"

    if stop_reason != "finished":
        if missing_actions:
            return (
                False,
                "step_budget_exceeded_before_required_tools: missing actions "
                + ", ".join(missing_actions),
            )
        return False, "step_budget_exceeded_before_finish"

    if missing_actions:
        return False, "missing_required_actions: " + ", ".join(missing_actions)

    if not answer_correct:
        return False, "incorrect_final_answer"

    return True, "success"


def _build_result_dict(
    task: Task,
    final_answer: str,
    success: bool,
    failure_reason: str,
    steps: int,
    tool_calls: int,
    total_tokens: int,
    every_step_tokens: list[int],
    stop_reason: str,
    history: list[dict[str, str]],
    action_history: list[str],
    strategy_name: str,
    deviation_log: list[dict],
) -> dict[str, Any]:
    # 统一封装结果字典，消除 run_task 里两处 return 重复同一个大字典的问题
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "question": task.question,
        "expected_answer": task.expected_answer,
        "final_answer": final_answer,
        "success": success,
        "failure_reason": failure_reason,
        "steps": steps,
        "tool_calls": tool_calls,
        "tokens": total_tokens,
        "every_step_tokens": every_step_tokens,
        "stop_reason": stop_reason,
        "history": history,
        "action_history": action_history,
        "strategy": strategy_name,
        "deviation_log": deviation_log,
        # 只要有一步 aligned=False，整条记录就标记为未对齐
        "plan_action_aligned": all(d["aligned"] for d in deviation_log),
    }


def run_task(
    backend: AgentBackend,
    task: Task,
    max_steps: int,
    max_output_tokens: int,
    strategy_name: str = "direct",
    system_prompt: str = DIRECT_PROMPT,
) -> dict[str, Any]:
    # 整个 agent loop 的核心：决策 -> 工具调用 -> 记录历史，直到 finish 或超出步数预算
    history: list[dict[str, str]] = []
    total_tokens = 0
    tool_calls = 0
    every_step_tokens: list[int] = []
    final_answer = ""
    stop_reason = "max_steps_reached"
    action_history: list[str] = []
    tool_errors: list[str] = []
    deviation_log: list[dict] = []

    # 步数预算给 4 步，足够两种策略在单工具和双工具任务上都有机会完成
    for step in range(1, max_steps + 1):
        tool_used = tool_calls > 0
        decision, tokens = backend.decide(
            task=task,
            history=history,
            tool_used=tool_used,
            max_output_tokens=max_output_tokens,
            system_prompt=system_prompt,
        )
        total_tokens += tokens
        every_step_tokens.append(tokens)

        thought = str(decision.get("thought", "")).strip()
        action = str(decision.get("action", "")).strip()
        action_input = str(decision.get("action_input", "")).strip()
        proposed_final = str(decision.get("final_answer", "")).strip()
        action_history.append(action)

        # 对齐检测：检查 thought 里是否提到了实际要执行的 action
        # 这是最简单的启发式判断，后续可以替换成更精确的语义匹配
        deviation_log.append({
            "step": step,
            "planned_action": thought,
            "actual_action": action,
            "aligned": action in thought.lower(),
        })

        if action == "finish":
            if not tool_used:
                # 强制至少用一次工具，防止模型直接回答绕过工具调用
                history.append({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": "Finish rejected because at least one tool must be used first.",
                })
                continue
            final_answer = proposed_final
            stop_reason = "finished"
            history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": f"Final answer submitted: {final_answer}",
            })
            success, failure_reason = build_failure_reason(
                task=task,
                final_answer=final_answer,
                stop_reason=stop_reason,
                action_history=action_history,
                tool_errors=tool_errors,
            )
            return _build_result_dict(
                task, final_answer, success, failure_reason, step,
                tool_calls, total_tokens, every_step_tokens, stop_reason,
                history, action_history, strategy_name, deviation_log,
            )

        try:
            observation = run_tool(action, action_input)
            tool_calls += 1
        except Exception as exc:
            observation = f"Tool error: {exc}"
            tool_errors.append(str(exc))

        history.append({
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
        })

    # 超出步数预算，强制结束并评估结果
    success, failure_reason = build_failure_reason(
        task=task,
        final_answer=final_answer,
        stop_reason=stop_reason,
        action_history=action_history,
        tool_errors=tool_errors,
    )
    return _build_result_dict(
        task, final_answer, success, failure_reason, max_steps,
        tool_calls, total_tokens, every_step_tokens, stop_reason,
        history, action_history, strategy_name, deviation_log,
    )


def _compute_strategy_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    # 按策略分组并预计算所有对比指标，避免在 render 函数里重复过滤同一份数据
    direct = [r for r in results if r["strategy"] == "direct"]
    cot = [r for r in results if r["strategy"] == "cot"]
    direct_avg = statistics.mean(r["tokens"] for r in direct)
    cot_avg = statistics.mean(r["tokens"] for r in cot)
    return {
        "direct_results": direct,
        "cot_results": cot,
        "direct_success": sum(1 for r in direct if r["success"]),
        "cot_success": sum(1 for r in cot if r["success"]),
        "direct_avg_tokens": direct_avg,
        "cot_avg_tokens": cot_avg,
        "token_delta": cot_avg - direct_avg,
        "aligned_count": sum(1 for r in results if r["plan_action_aligned"]),
    }


def _render_results_table(rows: list[dict[str, Any]], include_failure_reason: bool) -> list[str]:
    # 通用表格渲染：include_failure_reason=True 渲染失败表，False 渲染成功表
    if include_failure_reason:
        lines = [
            "| Strategy | Task ID | Type | Failure Reason | Steps | Tokens | Plan-Action Aligned |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| `{r['strategy']}` | `{r['task_id']}` | `{r['task_type']}` | "
                f"`{r['failure_reason']}` | `{r['steps']}` | `{r['tokens']}` | "
                f"`{r['plan_action_aligned']}` |"
            )
    else:
        lines = [
            "| Strategy | Task ID | Type | Steps | Tokens | Plan-Action Aligned |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| `{r['strategy']}` | `{r['task_id']}` | `{r['task_type']}` | "
                f"`{r['steps']}` | `{r['tokens']}` | `{r['plan_action_aligned']}` |"
            )
    return lines


def _render_cn_section(
    results: list[dict[str, Any]],
    backend_name: str,
    model_name: str,
    max_steps: int,
    stats: dict[str, Any],
    success_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    failure_reason_counts: dict[str, int],
) -> list[str]:
    # 中文版报告：Summary + 成功/失败表 + 分析
    s = stats
    lines = [
        "## 中文版",
        "",
        "### Summary",
        f"- 后端：`{backend_name}`",
        f"- 模型：`{model_name}`",
        f"- 总记录数：`{len(results)}` （2 种策略 x {len(TASKS)} 道题）",
        f"- 步数预算：`{max_steps}`",
        f"- Direct：`{s['direct_success']}/{len(s['direct_results'])}` 成功"
        f"（`{s['direct_success']/len(s['direct_results']):.2%}`），平均 token `{s['direct_avg_tokens']:.1f}`",
        f"- CoT：`{s['cot_success']}/{len(s['cot_results'])}` 成功"
        f"（`{s['cot_success']/len(s['cot_results']):.2%}`），平均 token `{s['cot_avg_tokens']:.1f}`",
        f"- CoT 比 Direct 多花 token：`{s['token_delta']:.1f}`",
        "",
        "### 成功任务",
    ]
    lines.extend(_render_results_table(success_rows, include_failure_reason=False))
    lines.extend(["", "### 失败任务"])
    lines.extend(_render_results_table(failure_rows, include_failure_reason=True))
    lines.extend(["", "### 分析"])
    lines.append(
        f"- Direct：{s['direct_success']}/{len(s['direct_results'])} 成功，"
        f"平均 `{s['direct_avg_tokens']:.1f}` token"
    )
    lines.append(
        f"- CoT：{s['cot_success']}/{len(s['cot_results'])} 成功，"
        f"平均 `{s['cot_avg_tokens']:.1f}` token"
    )
    lines.append(f"- CoT 多花了 `{s['token_delta']:.1f}` token，成功率的提升是否值得？")
    lines.append(
        f"- plan-action 对齐：`{s['aligned_count']}/{len(results)}` 条"
        f"（`{s['aligned_count']/len(results):.2%}`），低对齐率可能预示推理漂移"
    )
    if failure_reason_counts:
        lines.append(
            "- 失败原因分布："
            + "，".join(f"`{reason}` x {count}" for reason, count in failure_reason_counts.items())
        )
    return lines


def _render_en_section(
    results: list[dict[str, Any]],
    backend_name: str,
    model_name: str,
    max_steps: int,
    stats: dict[str, Any],
    success_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    failure_reason_counts: dict[str, int],
) -> list[str]:
    # English version: same structure as Chinese section
    s = stats
    lines = [
        "## English Version",
        "",
        "### Summary",
        f"- Backend: `{backend_name}`",
        f"- Model: `{model_name}`",
        f"- Total records: `{len(results)}` (2 strategies x {len(TASKS)} tasks)",
        f"- Step budget: `{max_steps}`",
        f"- Direct: `{s['direct_success']}/{len(s['direct_results'])}` success"
        f" (`{s['direct_success']/len(s['direct_results']):.2%}`), avg `{s['direct_avg_tokens']:.1f}` tokens",
        f"- CoT: `{s['cot_success']}/{len(s['cot_results'])}` success"
        f" (`{s['cot_success']/len(s['cot_results']):.2%}`), avg `{s['cot_avg_tokens']:.1f}` tokens",
        f"- CoT extra token cost vs Direct: `{s['token_delta']:.1f}`",
        "",
        "### Success Table",
    ]
    lines.extend(_render_results_table(success_rows, include_failure_reason=False))
    lines.extend(["", "### Failure Table"])
    lines.extend(_render_results_table(failure_rows, include_failure_reason=True))
    lines.extend(["", "### Analysis"])
    lines.append(
        f"- Direct: {s['direct_success']}/{len(s['direct_results'])} success,"
        f" avg `{s['direct_avg_tokens']:.1f}` tokens."
    )
    lines.append(
        f"- CoT: {s['cot_success']}/{len(s['cot_results'])} success,"
        f" avg `{s['cot_avg_tokens']:.1f}` tokens."
    )
    lines.append(
        f"- CoT costs `{s['token_delta']:.1f}` more tokens on average"
        " — is the accuracy gain worth it?"
    )
    lines.append(
        f"- Plan-action aligned in `{s['aligned_count']}/{len(results)}` records"
        f" (`{s['aligned_count']/len(results):.2%}`). Low alignment may signal reasoning drift."
    )
    if failure_reason_counts:
        lines.append(
            "- Failure breakdown: "
            + ", ".join(f"`{reason}` x {count}" for reason, count in failure_reason_counts.items())
        )
    return lines


def render_markdown(
    results: list[dict[str, Any]],
    backend_name: str,
    model_name: str,
    max_steps: int,
) -> str:
    # 预计算所有指标一次，传给各 section 渲染函数复用，不重复过滤
    stats = _compute_strategy_stats(results)
    success_rows = [r for r in results if r["success"]]
    failure_rows = [r for r in results if not r["success"]]
    failure_reason_counts: dict[str, int] = {}
    for r in failure_rows:
        failure_reason_counts[r["failure_reason"]] = failure_reason_counts.get(r["failure_reason"], 0) + 1

    lines = ["# Day 2 CoT vs Direct Strategy Comparison / Day 2 规划策略对比实验", ""]
    lines.extend(_render_cn_section(
        results, backend_name, model_name, max_steps,
        stats, success_rows, failure_rows, failure_reason_counts,
    ))
    lines.extend(["", "---", ""])
    lines.extend(_render_en_section(
        results, backend_name, model_name, max_steps,
        stats, success_rows, failure_rows, failure_reason_counts,
    ))
    return "\n".join(lines) + "\n"


def write_results(
    results: list[dict[str, Any]],
    backend_name: str,
    model_name: str,
    max_steps: int,
    output_path: pathlib.Path,
) -> pathlib.Path:
    markdown = render_markdown(
        results=results,
        backend_name=backend_name,
        model_name=model_name,
        max_steps=max_steps,
    )
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def create_backend(args: argparse.Namespace) -> AgentBackend:
    if args.backend == "anthropic":
        return AnthropicBackend(model_name=args.anthropic_model, temperature=args.temperature)
    if args.backend == "local-qwen":
        return LocalQwenBackend(
            model_name=args.local_model,
            base_url=args.local_base_url,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            api_key=args.local_api_key,
        )
    raise BackendError(f"Unsupported backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 2: Compare Direct vs CoT planning strategies on a fixed task set."
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "local-qwen"],
        default=DEFAULT_BACKEND,
        help="Model backend to use. Defaults to DAY2_BACKEND env var or local-qwen.",
    )
    parser.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
    parser.add_argument("--local-model", default=os.getenv("LOCAL_QWEN_MODEL", DEFAULT_LOCAL_MODEL))
    parser.add_argument("--local-base-url", default=os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL))
    parser.add_argument("--local-api-key", default=os.getenv("LOCAL_LLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=str(DEFAULT_RESULTS_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = create_backend(args)
    all_results = []
    # 用同一批任务跑两遍，每次换一个 system_prompt，其余条件完全相同
    for strategy_name, system_prompt in [("direct", DIRECT_PROMPT), ("cot", COT_PROMPT)]:
        for task in TASKS:
            result = run_task(
                backend=backend,
                task=task,
                max_steps=args.max_steps,
                max_output_tokens=args.max_output_tokens,
                strategy_name=strategy_name,
                system_prompt=system_prompt,
            )
            all_results.append(result)

    results_path = write_results(
        results=all_results,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        max_steps=args.max_steps,
        output_path=pathlib.Path(args.output).resolve(),
    )
    print(f"Saved results to: {results_path}")
    print(f"Backend\t{backend.backend_name}")
    print(f"Model\t{backend.model_name}")
    print("Strategy\tTaskID\tSuccess\tSteps\tTokens\tFailureReason\tFinalAnswer")
    for result in all_results:
        print(
            f"{result['strategy']}\t{result['task_id']}\t{result['success']}\t{result['steps']}\t"
            f"{result['tokens']}\t{result['failure_reason']}\t{result['final_answer'] or 'N/A'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
