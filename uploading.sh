#!/usr/bin/env bash

set -euo pipefail

# 1. 环境检查
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "错误：当前目录不是 Git 仓库。"
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
ssh_over_443_cmd='ssh -o Hostname=ssh.github.com -o Port=443 -o StrictHostKeyChecking=accept-new'

# 2. 获取 Commit 信息
commit_msg="${1:-}"
if [[ -z "$commit_msg" ]]; then
  read -r -p "请输入 commit 信息: " commit_msg
fi
[[ -z "$commit_msg" ]] && { echo "错误：信息不能为空"; exit 1; }

# --- 核心修复逻辑开始 ---

echo "正在清理并准备上传列表..."

# 第一步：先把所有东西都加入暂存区
git add .

# 第二步：强制从暂存区中“拿掉”不想上传的文件
# 无论它们是已跟踪文件还是新加入的未跟踪文件，都不要把它们带进 commit
git reset -q HEAD -- .gitignore uploading.sh >/dev/null 2>&1 || true

# --- 核心修复逻辑结束 ---

# 3. 提交检查
if git diff --cached --quiet; then
  echo "⚠️ 警告：Git 没有检测到任何需要提交的改动（README等可能未变动）。"
else
  git commit -m "$commit_msg"
  echo "✅ 已成功 Commit 改动。"
fi

# 4. 推送
echo "正在通过 GitHub SSH 443 端口推送到远程..."
GIT_SSH_COMMAND="$ssh_over_443_cmd" git push origin "$branch"
echo "🚀 上传完成：origin/$branch"
