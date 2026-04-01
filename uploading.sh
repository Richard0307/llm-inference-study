#!/usr/bin/env bash

set -euo pipefail

# 1. 环境检查
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "错误：当前目录不是 Git 仓库。"
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"

# 2. 获取 Commit 信息
commit_msg="${1:-}"
if [[ -z "$commit_msg" ]]; then
  read -r -p "请输入 commit 信息: " commit_msg
fi
[[ -z "$commit_msg" ]] && { echo "错误：信息不能为空"; exit 1; }

# --- 核心修复逻辑开始 ---

echo "正在清理并准备上传列表..."

# 第一步：先把所有东西都加入暂存区（包括 README.md 的修改）
git add .

# 第二步：强制从暂存区中“拿掉”.gitignore
# 这样 Git 就会认为：除了 .gitignore，其他东西都要提交
if git ls-files --error-unmatch .gitignore >/dev/null 2>&1; then
    git rm --cached .gitignore >/dev/null 2>&1
fi

# --- 核心修复逻辑结束 ---

# 3. 提交检查
if git diff --cached --quiet; then
  echo "⚠️ 警告：Git 没有检测到任何需要提交的改动（README等可能未变动）。"
else
  git commit -m "$commit_msg"
  echo "✅ 已成功 Commit 改动。"
fi

# 4. 推送
echo "正在推送到远程..."
git push origin "$branch"
echo "🚀 上传完成：origin/$branch"