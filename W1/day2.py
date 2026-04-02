#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
from typing import Any


PROBE_CODE = r"""
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys

MODULES = [
    "torch",
    "transformers",
    "openai",
    "anthropic",
    "langchain",
    "langgraph",
    "vllm",
    "ollama",
]

packages = {}
for name in MODULES:
    try:
        module = importlib.import_module(name)
        packages[name] = {
            "installed": True,
            "version": getattr(module, "__version__", "unknown"),
        }
    except Exception as exc:
        packages[name] = {
            "installed": False,
            "version": None,
            "error": type(exc).__name__,
        }

api_keys = {}
for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY"]:
    api_keys[key] = bool(os.getenv(key))

gpu = {
    "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
    "name": None,
    "driver_version": None,
    "memory_total_mib": None,
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_name": None,
    "cuda_version": None,
}

if gpu["nvidia_smi_available"]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first_gpu = result.stdout.strip().splitlines()[0]
        name, driver_version, memory_total = [part.strip() for part in first_gpu.split(",")]
        gpu["name"] = name
        gpu["driver_version"] = driver_version
        gpu["memory_total_mib"] = memory_total
    except Exception:
        pass

if packages["torch"]["installed"]:
    import torch

    gpu["cuda_available"] = torch.cuda.is_available()
    gpu["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu["cuda_device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    gpu["cuda_version"] = torch.version.cuda

report = {
    "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    "conda_default_env": os.getenv("CONDA_DEFAULT_ENV"),
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "system": platform.system(),
    "release": platform.release(),
    "machine": platform.machine(),
    "packages": packages,
    "api_keys": api_keys,
    "gpu": gpu,
}

print(json.dumps(report, ensure_ascii=True))
"""


MODEL_CANDIDATES = [
    {
        "name_cn": "Claude Sonnet API",
        "name_en": "Claude Sonnet API",
        "model_id": "claude-sonnet-4-20250514",
        "kind": "api",
        "backend": "Anthropic API",
        "strengths_cn": [
            "对规划、推理和代码任务表现稳定",
            "你已经在 llm-infer 环境里配置好了 Anthropic key，接入路径最短",
        ],
        "strengths_en": [
            "Strong and stable on planning, reasoning, and coding tasks",
            "Your Anthropic key is already configured in llm-infer, so setup friction is low",
        ],
        "constraints_cn": [
            "存在 API 成本和网络依赖",
            "为了可复现，需要固定模型 ID、temperature 和 max tokens",
        ],
        "constraints_en": [
            "Has API cost and network dependency",
            "For reproducibility, keep model ID, temperature, and max tokens fixed",
        ],
    },
    {
        "name_cn": "Qwen2.5-7B-Instruct",
        "name_en": "Qwen2.5-7B-Instruct",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "kind": "local",
        "backend": "vLLM or transformers",
        "strengths_cn": [
            "适合作为本地开源 baseline",
            "中文和英文指令跟随能力都比较平衡",
        ],
        "strengths_en": [
            "A solid open local baseline",
            "Balanced bilingual instruction-following for Chinese and English",
        ],
        "constraints_cn": [
            "首次运行前仍需下载模型权重",
            "16 GB 显存下要控制上下文长度和 batch 设置",
        ],
        "constraints_en": [
            "Model weights still need to be downloaded before the first run",
            "On 16 GB VRAM, context length and batch settings should stay conservative",
        ],
    },
    {
        "name_cn": "Llama-3.1-8B-Instruct",
        "name_en": "Llama-3.1-8B-Instruct",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "kind": "local",
        "backend": "vLLM or transformers",
        "strengths_cn": [
            "常见的英文开源 instruct baseline",
            "便于和公开 agent 示例做横向比较",
        ],
        "strengths_en": [
            "A common open English-heavy instruct baseline",
            "Easy to compare against public agent examples",
        ],
        "constraints_cn": [
            "可能需要额外的模型访问授权",
            "中文体验通常不如 Qwen 舒服",
        ],
        "constraints_en": [
            "May require model access approval",
            "Usually less comfortable than Qwen on Chinese prompts",
        ],
    },
    {
        "name_cn": "OpenAI GPT-4.1 API",
        "name_en": "OpenAI GPT-4.1 API",
        "model_id": "gpt-4.1",
        "kind": "api",
        "backend": "OpenAI API",
        "strengths_cn": [
            "常见的强 API baseline",
            "后续做 agent 原型时集成路径清晰",
        ],
        "strengths_en": [
            "A common strong API baseline",
            "Clear integration path for later agent prototyping",
        ],
        "constraints_cn": [
            "当前 OPENAI_API_KEY 还没有配置",
            "存在 API 成本和网络依赖",
        ],
        "constraints_en": [
            "OPENAI_API_KEY is not configured right now",
            "Has API cost and network dependency",
        ],
    },
]


TASK_SOURCE_OPTIONS = [
    {
        "name_cn": "本地推理微型任务集",
        "name_en": "Local reasoning micro-set",
        "source_cn": "仓库内自定义任务",
        "source_en": "Self-authored tasks stored in the repo",
        "shape_cn": "5 个短推理任务",
        "shape_en": "5 short reasoning tasks",
        "examples_cn": [
            "两步算术文字题",
            "日历与日期推理",
            "给定上下文的简短多跳推理",
        ],
        "examples_en": [
            "Two-step arithmetic word problems",
            "Calendar and date reasoning",
            "Short multi-hop reasoning with provided context",
        ],
        "why_cn": "不依赖外部数据，最容易先拿到可重复 baseline",
        "why_en": "Fastest way to get a repeatable baseline without extra external dependencies",
    },
    {
        "name_cn": "本地工具使用微型任务集",
        "name_en": "Local tool-use micro-set",
        "source_cn": "仓库内自定义任务",
        "source_en": "Self-authored tasks stored in the repo",
        "shape_cn": "5 个工具使用任务",
        "shape_en": "5 tool-use tasks",
        "examples_cn": [
            "调用计算器",
            "读取或检索本地文件",
            "字符串抽取或简单表格查找",
        ],
        "examples_en": [
            "Calculator calls",
            "Local file read or search",
            "String extraction or simple table lookup",
        ],
        "why_cn": "和 day 3 的 Thought -> Action -> Observation 循环直接对齐",
        "why_en": "Directly aligned with the day-3 Thought -> Action -> Observation loop",
    },
    {
        "name_cn": "公共 benchmark 小切片",
        "name_en": "Public benchmark slices",
        "source_cn": "GSM8K、HotpotQA 或 AgentBench 风格任务的小样本切片",
        "source_en": "Small curated slices from GSM8K, HotpotQA, or AgentBench-style tasks",
        "shape_cn": "等本地微型任务集稳定后再接入",
        "shape_en": "Add after the local micro-set becomes stable",
        "examples_cn": [
            "5 个 GSM8K 风格题目",
            "5 个轻量 AgentBench 风格工具任务",
        ],
        "examples_en": [
            "5 GSM8K-style questions",
            "5 lightweight AgentBench-style tool tasks",
        ],
        "why_cn": "后续做对外对比有价值，但第一周不必一开始就上重 benchmark",
        "why_en": "Useful later for external comparisons, but too heavy for the first week baseline",
    },
]


PACKAGE_ORDER = [
    "torch",
    "transformers",
    "vllm",
    "openai",
    "anthropic",
    "langchain",
    "langgraph",
    "ollama",
]

API_KEY_ORDER = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY"]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe_conda_environment(conda_env: str) -> dict[str, Any]:
    result = run_command(["conda", "run", "-n", conda_env, "python", "-c", PROBE_CODE])
    return json.loads(result.stdout.strip().splitlines()[-1])


def collect_environment_report(conda_env: str) -> dict[str, Any]:
    report = probe_conda_environment(conda_env)
    report["probe_target_env"] = conda_env
    report["probe_runner_env"] = os.getenv("CONDA_DEFAULT_ENV") or "unknown"
    return report


def bool_status_cn(value: bool) -> str:
    return "已配置（值已隐藏）" if value else "未配置"


def bool_status_en(value: bool) -> str:
    return "configured (redacted)" if value else "not configured"


def package_status_cn(info: dict[str, Any]) -> str:
    if info["installed"]:
        return f"已安装（`{info['version']}`）"
    return "未安装"


def package_status_en(info: dict[str, Any]) -> str:
    if info["installed"]:
        return f"installed (`{info['version']}`)"
    return "not installed"


def model_ready_now(candidate: dict[str, Any], report: dict[str, Any]) -> tuple[bool, str, str]:
    packages = report["packages"]
    api_keys = report["api_keys"]
    gpu = report["gpu"]

    if candidate["backend"] == "Anthropic API":
        ready = api_keys["ANTHROPIC_API_KEY"] and packages["anthropic"]["installed"]
        return (
            ready,
            "Anthropic SDK 已安装，且 ANTHROPIC_API_KEY 已配置",
            "Anthropic SDK is installed and ANTHROPIC_API_KEY is configured",
        )

    if candidate["backend"] == "OpenAI API":
        ready = api_keys["OPENAI_API_KEY"] and packages["openai"]["installed"]
        return (
            ready,
            "需要同时满足 OpenAI SDK 已安装且 OPENAI_API_KEY 已配置",
            "Requires both the OpenAI SDK and OPENAI_API_KEY",
        )

    ready = gpu["cuda_available"] and packages["torch"]["installed"] and packages["transformers"]["installed"]
    if packages["vllm"]["installed"]:
        return (
            ready,
            "本地 CUDA 栈可用，vLLM 已安装，但首次运行仍需下载权重",
            "Local CUDA stack is ready and vLLM is installed, but model weights still need downloading",
        )
    return (
        ready,
        "本地 transformers 栈可用，vLLM 为可选项",
        "Local transformers stack is ready and vLLM is optional",
    )


def choose_baseline(report: dict[str, Any]) -> dict[str, str]:
    api_keys = report["api_keys"]
    packages = report["packages"]
    gpu = report["gpu"]

    if api_keys["ANTHROPIC_API_KEY"] and packages["anthropic"]["installed"]:
        return {
            "model_id": "claude-sonnet-4-20250514",
            "display_name": "Claude Sonnet API",
            "backend": "Anthropic API",
            "reason_cn": "你已经在 llm-infer 中配置了 Anthropic key，而且 anthropic SDK 已安装。对当前 4 月 baseline 来说，这是最快能稳定开跑的 API 路径。",
            "reason_en": "Your Anthropic key is already configured in llm-infer and the anthropic SDK is installed. For the April baseline, this is the fastest stable API path to get running.",
        }

    if api_keys["OPENAI_API_KEY"] and packages["openai"]["installed"]:
        return {
            "model_id": "gpt-4.1",
            "display_name": "OpenAI GPT-4.1 API",
            "backend": "OpenAI API",
            "reason_cn": "OpenAI key 和 SDK 都已准备好，先用 API baseline 可以减少本地模型服务的前期工作量。",
            "reason_en": "The OpenAI key and SDK are ready, so an API baseline can reduce local model-serving work in week 1.",
        }

    if gpu["cuda_available"] and packages["vllm"]["installed"] and packages["transformers"]["installed"]:
        return {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "display_name": "Qwen2.5-7B-Instruct",
            "backend": "Local vLLM",
            "reason_cn": "当前没有可用的 API key 时，本地 4090 + vLLM + transformers 是最干净的 baseline 组合。",
            "reason_en": "Without a ready API key, the local 4090 + vLLM + transformers stack is the cleanest baseline setup.",
        }

    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "display_name": "Qwen2.5-7B-Instruct",
        "backend": "Local transformers",
        "reason_cn": "当前环境最接近本地 baseline；如果 vLLM 临时不可用，可以先走 transformers 版本。",
        "reason_en": "The current environment is closest to a local baseline; if vLLM is temporarily unavailable, start with a transformers-based path.",
    }


def choose_task_set() -> dict[str, Any]:
    return {
        "name": "april_agent_micro_set_v1",
        "source_cn": "仓库内自定义本地任务",
        "source_en": "Self-authored local tasks stored in the repo",
        "composition_cn": [
            "5 个简单推理任务",
            "5 个带计算器和本地文本检索的工具使用任务",
        ],
        "composition_en": [
            "5 simple reasoning tasks",
            "5 tool-use tasks with calculator and local text search",
        ],
        "why_cn": "这个任务集最容易先跑通最小 agent loop，也方便记录 success rate、step count 和 token cost。",
        "why_en": "This task set is the easiest starting point for a minimal agent loop and makes it easy to track success rate, step count, and token cost.",
    }


def build_cn_environment_section(report: dict[str, Any], baseline: dict[str, str], task_set: dict[str, Any]) -> list[str]:
    packages = report["packages"]
    gpu = report["gpu"]
    api_keys = report["api_keys"]
    gpu_name = gpu["cuda_device_name"] or gpu["name"] or "未检测到"

    lines = [
        "## 中文版",
        "",
        "### 1. 环境概览",
        f"- 探测日期：`{dt.date.today().isoformat()}`",
        f"- 目标 conda 环境：`{report['probe_target_env']}`",
        f"- Python 可执行文件：`{report['python_executable']}`",
        f"- Python 版本：`{report['python_version']}`",
        f"- 平台：`{report['platform']}`",
        f"- 当前运行环境：`{report['probe_runner_env']}`",
        "",
        "### 2. GPU 与 CUDA",
        f"- `nvidia-smi` 是否可用：`{gpu['nvidia_smi_available']}`",
        f"- GPU：`{gpu_name}`",
        f"- 驱动版本：`{gpu['driver_version'] or 'unknown'}`",
        f"- 显存：`{gpu['memory_total_mib'] or 'unknown'}`",
        f"- Torch CUDA 可用：`{gpu['cuda_available']}`",
        f"- CUDA 设备数：`{gpu['cuda_device_count']}`",
        f"- Torch CUDA 版本：`{gpu['cuda_version'] or 'unknown'}`",
        "",
        "### 3. Python / LLM 后端包",
    ]
    for package_name in PACKAGE_ORDER:
        lines.append(f"- `{package_name}`：{package_status_cn(packages[package_name])}")

    lines.extend(
        [
            "",
            "### 4. API Key 状态",
            "- 注意：这里只显示状态，不会写入任何真实 key 值。",
        ]
    )
    for key_name in API_KEY_ORDER:
        lines.append(f"- `{key_name}`：{bool_status_cn(api_keys[key_name])}")

    lines.extend(
        [
            "",
            "### 5. 今日判断",
            "- Anthropic API 已可用，因此 API baseline 已经具备启动条件。",
            "- 本地 4090 + torch + transformers + vLLM 也可作为后备本地 baseline。",
            "- `langchain` 和 `langgraph` 还没装，第一版 agent loop 先手写最合适。",
            "",
            "### 6. 4 月 baseline 决策快照",
            f"- 选定模型：`{baseline['model_id']}`",
            f"- 后端方式：`{baseline['backend']}`",
            f"- 任务集：`{task_set['name']}`",
            f"- 任务来源：`{task_set['source_cn']}`",
            f"- 选择原因：{baseline['reason_cn']}",
        ]
    )
    return lines


def build_en_environment_section(report: dict[str, Any], baseline: dict[str, str], task_set: dict[str, Any]) -> list[str]:
    packages = report["packages"]
    gpu = report["gpu"]
    api_keys = report["api_keys"]
    gpu_name = gpu["cuda_device_name"] or gpu["name"] or "not detected"

    lines = [
        "## English Version",
        "",
        "### 1. Environment Summary",
        f"- Probe date: `{dt.date.today().isoformat()}`",
        f"- Target conda environment: `{report['probe_target_env']}`",
        f"- Python executable: `{report['python_executable']}`",
        f"- Python version: `{report['python_version']}`",
        f"- Platform: `{report['platform']}`",
        f"- Runner environment: `{report['probe_runner_env']}`",
        "",
        "### 2. GPU and CUDA",
        f"- `nvidia-smi` available: `{gpu['nvidia_smi_available']}`",
        f"- GPU: `{gpu_name}`",
        f"- Driver version: `{gpu['driver_version'] or 'unknown'}`",
        f"- GPU memory: `{gpu['memory_total_mib'] or 'unknown'}`",
        f"- Torch CUDA available: `{gpu['cuda_available']}`",
        f"- CUDA device count: `{gpu['cuda_device_count']}`",
        f"- Torch CUDA version: `{gpu['cuda_version'] or 'unknown'}`",
        "",
        "### 3. Python / LLM Backend Packages",
    ]
    for package_name in PACKAGE_ORDER:
        lines.append(f"- `{package_name}`: {package_status_en(packages[package_name])}")

    lines.extend(
        [
            "",
            "### 4. API Key Status",
            "- Note: only status is shown here; no real secret value is written into the markdown.",
        ]
    )
    for key_name in API_KEY_ORDER:
        lines.append(f"- `{key_name}`: {bool_status_en(api_keys[key_name])}")

    lines.extend(
        [
            "",
            "### 5. Assessment for Today",
            "- Anthropic API is ready now, so an API baseline can be started immediately.",
            "- The local 4090 + torch + transformers + vLLM stack is also available as a local fallback baseline.",
            "- `langchain` and `langgraph` are still missing, so the first agent loop should stay hand-written and minimal.",
            "",
            "### 6. April Baseline Decision Snapshot",
            f"- Selected model: `{baseline['model_id']}`",
            f"- Backend: `{baseline['backend']}`",
            f"- Task set: `{task_set['name']}`",
            f"- Task source: `{task_set['source_en']}`",
            f"- Why this choice: {baseline['reason_en']}",
        ]
    )
    return lines


def render_environment_record(report: dict[str, Any], baseline: dict[str, str], task_set: dict[str, Any]) -> str:
    lines = ["# Day 2 环境记录 / Day 2 Environment Record", ""]
    lines.extend(build_cn_environment_section(report, baseline, task_set))
    lines.extend(["", "---", ""])
    lines.extend(build_en_environment_section(report, baseline, task_set))
    return "\n".join(lines)


def build_cn_candidate_section(report: dict[str, Any], baseline: dict[str, str], task_set: dict[str, Any]) -> list[str]:
    lines = [
        "## 中文版",
        "",
        "### 1. 推荐的 4 月 baseline",
        f"- 模型：`{baseline['model_id']}`",
        f"- 显示名称：`{baseline['display_name']}`",
        f"- 后端：`{baseline['backend']}`",
        f"- 推荐理由：{baseline['reason_cn']}",
        "",
        "### 2. baseline 候选模型清单",
        "| 模型 ID | 类型 | 后端 | 当前是否可用 | 优势 | 约束 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidate in MODEL_CANDIDATES:
        ready, note_cn, _ = model_ready_now(candidate, report)
        lines.append(
            f"| `{candidate['model_id']}` | {candidate['kind']} | `{candidate['backend']}` | "
            f"{'是' if ready else '否'} | {'；'.join(candidate['strengths_cn'])} | "
            f"{'；'.join(candidate['constraints_cn'])} | {note_cn} |"
        )

    lines.extend(["", "### 3. Agent 任务样本来源选项"])
    for option in TASK_SOURCE_OPTIONS:
        lines.extend(
            [
                f"#### {option['name_cn']}",
                f"- 来源：{option['source_cn']}",
                f"- 规模：{option['shape_cn']}",
                f"- 保留理由：{option['why_cn']}",
                "- 例子：",
            ]
        )
        for example in option["examples_cn"]:
            lines.append(f"- {example}")
        lines.append("")

    lines.extend(
        [
            "### 4. 最终确定的 4 月任务集",
            f"- 任务集名称：`{task_set['name']}`",
            f"- 任务来源：`{task_set['source_cn']}`",
        ]
    )
    for item in task_set["composition_cn"]:
        lines.append(f"- 组成：{item}")
    lines.append(f"- 选择理由：{task_set['why_cn']}")
    return lines


def build_en_candidate_section(report: dict[str, Any], baseline: dict[str, str], task_set: dict[str, Any]) -> list[str]:
    lines = [
        "## English Version",
        "",
        "### 1. Recommended April Baseline",
        f"- Model: `{baseline['model_id']}`",
        f"- Display name: `{baseline['display_name']}`",
        f"- Backend: `{baseline['backend']}`",
        f"- Reason: {baseline['reason_en']}",
        "",
        "### 2. Candidate Model List",
        "| Model ID | Type | Backend | Ready now | Strengths | Constraints | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidate in MODEL_CANDIDATES:
        ready, _, note_en = model_ready_now(candidate, report)
        lines.append(
            f"| `{candidate['model_id']}` | {candidate['kind']} | `{candidate['backend']}` | "
            f"{'yes' if ready else 'not yet'} | {'; '.join(candidate['strengths_en'])} | "
            f"{'; '.join(candidate['constraints_en'])} | {note_en} |"
        )

    lines.extend(["", "### 3. Task Sample Source Options"])
    for option in TASK_SOURCE_OPTIONS:
        lines.extend(
            [
                f"#### {option['name_en']}",
                f"- Source: {option['source_en']}",
                f"- Scale: {option['shape_en']}",
                f"- Why keep it: {option['why_en']}",
                "- Examples:",
            ]
        )
        for example in option["examples_en"]:
            lines.append(f"- {example}")
        lines.append("")

    lines.extend(
        [
            "### 4. Final April Task Set",
            f"- Task set name: `{task_set['name']}`",
            f"- Task source: `{task_set['source_en']}`",
        ]
    )
    for item in task_set["composition_en"]:
        lines.append(f"- Composition: {item}")
    lines.append(f"- Why this set: {task_set['why_en']}")
    return lines


def render_baseline_candidates(report: dict[str, Any], baseline: dict[str, str], task_set: dict[str, Any]) -> str:
    lines = ["# Day 2 baseline 候选清单 / Day 2 Baseline Candidates", ""]
    lines.extend(build_cn_candidate_section(report, baseline, task_set))
    lines.extend(["", "---", ""])
    lines.extend(build_en_candidate_section(report, baseline, task_set))
    return "\n".join(lines)


def write_text(path: pathlib.Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate bilingual day-2 markdown outputs for the agent-study baseline setup."
    )
    parser.add_argument("--conda-env", default="llm-infer", help="Target conda environment name.")
    parser.add_argument(
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parent),
        help="Directory where markdown outputs will be written.",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = collect_environment_report(args.conda_env)
    baseline = choose_baseline(report)
    task_set = choose_task_set()

    environment_record = render_environment_record(report, baseline, task_set)
    baseline_candidates = render_baseline_candidates(report, baseline, task_set)

    environment_path = output_dir / "day2_environment_record.md"
    candidates_path = output_dir / "day2_baseline_candidates.md"

    write_text(environment_path, environment_record)
    write_text(candidates_path, baseline_candidates)

    summary = {
        "environment_record": str(environment_path),
        "baseline_candidates": str(candidates_path),
        "selected_baseline_model": baseline["model_id"],
        "selected_backend": baseline["backend"],
        "selected_task_set": task_set["name"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
