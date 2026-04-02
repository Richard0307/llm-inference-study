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
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic


MODEL_NAME = "claude-sonnet-4-20250514"
DEFAULT_MAX_STEPS = 2
DEFAULT_MAX_TOKENS = 220
RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "day3_results.md"


SYSTEM_PROMPT = """You are a minimal research agent.

You operate in a Thought -> Action -> Observation loop.
Available tools:
1. calculate(expression): evaluate an arithmetic expression.
2. search_notes(query): search a tiny local note collection.

Rules:
- You must use at least one tool before finishing.
- If the task asks for both search and calculation, you must actually call both tools.
- For arithmetic over note values, do not do mental math; use the calculator tool.
- Reply with strict JSON only.
- Use exactly these keys: thought, action, action_input, final_answer.
- action must be one of: calculate, search_notes, finish.
- Keep thought short and concrete.
- When action is finish, put the answer in final_answer.
"""


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
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model output did not contain JSON: {text}")
    return json.loads(match.group(0))


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


def ask_model(client: Anthropic, task: Task, history: list[dict[str, str]], tool_used: bool) -> tuple[dict[str, Any], int]:
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(task, history, tool_used)}],
    )
    text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    raw_text = "\n".join(text_blocks)
    parsed = extract_json_block(raw_text)
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return parsed, tokens


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


def run_task(client: Anthropic, task: Task, max_steps: int) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    total_tokens = 0
    tool_calls = 0
    final_answer = ""
    stop_reason = "max_steps_reached"
    action_history: list[str] = []
    tool_errors: list[str] = []

    for step in range(1, max_steps + 1):
        tool_used = tool_calls > 0
        decision, tokens = ask_model(client, task, history, tool_used)
        total_tokens += tokens

        thought = str(decision.get("thought", "")).strip()
        action = str(decision.get("action", "")).strip()
        action_input = str(decision.get("action_input", "")).strip()
        proposed_final = str(decision.get("final_answer", "")).strip()
        action_history.append(action)

        if action == "finish":
            if not tool_used:
                observation = "Finish rejected because at least one tool must be used first."
                history.append(
                    {
                        "thought": thought,
                        "action": action,
                        "action_input": action_input,
                        "observation": observation,
                    }
                )
                continue
            final_answer = proposed_final
            stop_reason = "finished"
            history.append(
                {
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": f"Final answer submitted: {final_answer}",
                }
            )
            success, failure_reason = build_failure_reason(
                task=task,
                final_answer=final_answer,
                stop_reason=stop_reason,
                action_history=action_history,
                tool_errors=tool_errors,
            )
            return {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question": task.question,
                "expected_answer": task.expected_answer,
                "final_answer": final_answer,
                "success": success,
                "failure_reason": failure_reason,
                "steps": step,
                "tool_calls": tool_calls,
                "tokens": total_tokens,
                "stop_reason": stop_reason,
                "history": history,
                "action_history": action_history,
            }

        try:
            observation = run_tool(action, action_input)
            tool_calls += 1
        except Exception as exc:
            observation = f"Tool error: {exc}"
            tool_errors.append(str(exc))

        history.append(
            {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
        )

    success, failure_reason = build_failure_reason(
        task=task,
        final_answer=final_answer,
        stop_reason=stop_reason,
        action_history=action_history,
        tool_errors=tool_errors,
    )
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "question": task.question,
        "expected_answer": task.expected_answer,
        "final_answer": final_answer,
        "success": success,
        "failure_reason": failure_reason,
        "steps": max_steps,
        "tool_calls": tool_calls,
        "tokens": total_tokens,
        "stop_reason": stop_reason,
        "history": history,
        "action_history": action_history,
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    success_count = sum(1 for result in results if result["success"])
    success_rate = success_count / len(results)
    average_steps = statistics.mean(result["steps"] for result in results)
    average_tokens = statistics.mean(result["tokens"] for result in results)
    success_rows = [result for result in results if result["success"]]
    failure_rows = [result for result in results if not result["success"]]
    failure_reason_counts: dict[str, int] = {}
    for result in failure_rows:
        failure_reason_counts[result["failure_reason"]] = (
            failure_reason_counts.get(result["failure_reason"], 0) + 1
        )

    lines = [
        "# Day 3 Minimal Agent Loop Results / Day 3 最小 Agent Loop 结果",
        "",
        "## 中文版",
        "",
        "### Summary",
        f"- 模型：`{MODEL_NAME}`",
        f"- 任务总数：`{len(results)}`",
        f"- 成功数：`{success_count}`",
        f"- 失败数：`{len(failure_rows)}`",
        f"- 成功率：`{success_rate:.2%}`",
        f"- 平均步骤数：`{average_steps:.2f}`",
        f"- 平均 token 消耗：`{average_tokens:.2f}`",
        f"- 设计目标失败率：`20.00%`",
        "",
        "### Success Table",
        "| Task ID | Type | Question | Expected | Final Answer | Steps | Tool Calls | Tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for result in success_rows:
        lines.append(
            f"| `{result['task_id']}` | `{result['task_type']}` | {result['question']} | "
            f"`{result['expected_answer']}` | `{result['final_answer'] or 'N/A'}` | "
            f"`{result['steps']}` | `{result['tool_calls']}` | `{result['tokens']}` |"
        )

    lines.extend(
        [
            "",
            "### Failure Table",
            "| Task ID | Type | Question | Expected | Final Answer | Failure Reason | Steps | Tool Calls | Tokens |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    for result in failure_rows:
        lines.append(
            f"| `{result['task_id']}` | `{result['task_type']}` | {result['question']} | "
            f"`{result['expected_answer']}` | `{result['final_answer'] or 'N/A'}` | "
            f"`{result['failure_reason']}` | `{result['steps']}` | `{result['tool_calls']}` | `{result['tokens']}` |"
        )

    lines.extend(["", "### Analysis"])
    lines.append("- 这份结果记录的是真实任务、真实工具调用和真实 API token 消耗，不再是随机模拟器。")
    lines.append(
        "- 当前的 20% 失败率是有意设计出来的：我把 `max_steps` 设成 `2`，同时放入了两道必须经历 "
        "`search_notes -> calculate -> finish` 三段流程的任务。"
    )
    lines.append(
        "- `step_budget_exceeded_before_finish` 会导致失败，是因为模型虽然已经完成了检索和计算，"
        "但在两步预算内没有机会再发出第三步 `finish`，因此无法正式提交最终答案。"
    )
    lines.append(
        "- 从这个角度看，这类失败不是“模型完全不会做”，而是“当前 loop 预算和控制流设计不足以完成多工具任务”。"
    )
    if failure_reason_counts:
        lines.append(
            "- 本次失败原因分布："
            + "，".join(f"`{reason}` x {count}" for reason, count in failure_reason_counts.items())
        )
    lines.append("- 除了 `step_budget_exceeded_before_finish`，后续实验里还可能出现这些失败原因：")
    lines.append("- `missing_required_actions`：模型提前结束，没有按要求调用必要工具。")
    lines.append("- `incorrect_final_answer`：工具调用流程看起来对了，但最终答案仍然算错或抽取错。")
    lines.append("- `tool_execution_error`：动作格式不合法，或者工具输入表达式错误，导致执行失败。")
    lines.append("- JSON 解析失败或输出格式漂移：模型没有按约定返回结构化字段，loop 会被中断。")
    lines.append("- 检索噪声与歧义：如果 notes 更长、更乱或有冲突信息，search 工具可能把模型带偏。")

    lines.extend(["", "---", "", "## English Version", "", "### Summary"])
    lines.append(f"- Model: `{MODEL_NAME}`")
    lines.append(f"- Task count: `{len(results)}`")
    lines.append(f"- Success count: `{success_count}`")
    lines.append(f"- Failure count: `{len(failure_rows)}`")
    lines.append(f"- Success rate: `{success_rate:.2%}`")
    lines.append(f"- Average steps: `{average_steps:.2f}`")
    lines.append(f"- Average tokens: `{average_tokens:.2f}`")
    lines.append("- Designed failure rate target: `20.00%`")
    lines.extend(["", "### Success Table"])
    lines.append("| Task ID | Type | Question | Expected | Final Answer | Steps | Tool Calls | Tokens |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for result in success_rows:
        lines.append(
            f"| `{result['task_id']}` | `{result['task_type']}` | {result['question']} | "
            f"`{result['expected_answer']}` | `{result['final_answer'] or 'N/A'}` | "
            f"`{result['steps']}` | `{result['tool_calls']}` | `{result['tokens']}` |"
        )

    lines.extend(["", "### Failure Table"])
    lines.append("| Task ID | Type | Question | Expected | Final Answer | Failure Reason | Steps | Tool Calls | Tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for result in failure_rows:
        lines.append(
            f"| `{result['task_id']}` | `{result['task_type']}` | {result['question']} | "
            f"`{result['expected_answer']}` | `{result['final_answer'] or 'N/A'}` | "
            f"`{result['failure_reason']}` | `{result['steps']}` | `{result['tool_calls']}` | `{result['tokens']}` |"
        )

    lines.extend(["", "### Analysis"])
    lines.append("- These results come from real tasks, real tool calls, and real API token usage rather than a simulator.")
    lines.append(
        "- The current 20% failure rate is intentional: `max_steps` is set to `2` while two tasks require the full "
        "`search_notes -> calculate -> finish` chain."
    )
    lines.append(
        "- `step_budget_exceeded_before_finish` causes failure because the agent may already have searched and calculated, "
        "but it still needs one more turn to submit the final answer with `finish`."
    )
    lines.append(
        "- In other words, this is not necessarily a “model cannot solve the task” failure. It is a control-budget failure "
        "caused by the current loop design."
    )
    if failure_reason_counts:
        lines.append(
            "- Failure reason breakdown: "
            + ", ".join(f"`{reason}` x {count}" for reason, count in failure_reason_counts.items())
        )
    lines.append("- Other plausible failure modes in later experiments include:")
    lines.append("- `missing_required_actions`: the model stops early without using a required tool.")
    lines.append("- `incorrect_final_answer`: the tool flow looks reasonable, but the final answer is still wrong.")
    lines.append("- `tool_execution_error`: the action format or tool input is invalid, so the tool call fails.")
    lines.append("- JSON parsing or formatting drift: the model does not return the required structured output.")
    lines.append("- Retrieval noise or ambiguity: longer or conflicting notes may push the search step off track.")
    return "\n".join(lines) + "\n"


def write_results(results: list[dict[str, Any]]) -> pathlib.Path:
    markdown = render_markdown(results)
    RESULTS_PATH.write_text(markdown, encoding="utf-8")
    return RESULTS_PATH


def ensure_api_key() -> None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not available in the current shell. "
        "Run this script inside the llm-infer environment, for example: "
        "`conda run -n llm-infer python W1/day3.py`."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal task-driven agent loop with Claude.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    ensure_api_key()
    client = Anthropic()
    results = [run_task(client, task, max_steps=args.max_steps) for task in TASKS]
    results_path = write_results(results)

    print(f"Saved markdown results to: {results_path}")
    print("TaskID\tSuccess\tSteps\tToolCalls\tTokens\tFailureReason\tFinalAnswer")
    for result in results:
        print(
            f"{result['task_id']}\t{result['success']}\t{result['steps']}\t"
            f"{result['tool_calls']}\t{result['tokens']}\t{result['failure_reason']}\t"
            f"{result['final_answer'] or 'N/A'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
