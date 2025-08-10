#!/bin/bash

# 🚀 使用Personal Access Token推送到GitHub

echo "🔑 使用Personal Access Token推送..."

# 设置凭据 (临时使用)
git config credential.helper store

# 推送windows分支 (保存原始main分支)
echo "📤 推送windows分支 (原始main分支备份)..."
echo "https://TianTian-O1:YOUR_GITHUB_TOKEN@github.com/TianTian-O1/boas.git" | git credential approve 2>/dev/null
git push origin windows

# 推送main分支 (NPU优化版本)
echo "🚀 推送main分支 (NPU优化版本)..."
git push origin main --force-with-lease

echo "✅ 推送完成！"
echo ""
echo "📊 远程分支状态:"
git fetch origin
git branch -r
