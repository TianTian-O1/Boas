#!/bin/bash

echo "🔑 配置Git认证..."

# 方法1: 尝试使用credential helper
git config --global credential.helper store
echo "✅ 已配置credential helper"

# 方法2: 尝试SSH (如果可用)
if [ -f ~/.ssh/id_rsa ] || [ -f ~/.ssh/id_ed25519 ]; then
    echo "🔑 检测到SSH密钥，尝试SSH认证..."
    git remote set-url origin git@github.com:TianTian-O1/boas.git
    echo "✅ 已设置SSH远程URL"
else
    echo "⚠️ 未找到SSH密钥"
    echo "💡 请确保你有GitHub访问权限，或者:"
    echo "   1. 设置Personal Access Token"
    echo "   2. 配置SSH密钥"
    echo "   3. 或者手动推送"
fi

echo ""
echo "🚀 准备推送..."
echo "推送命令:"
echo "  git push origin windows  # 保存原始main分支"
echo "  git push origin main --force-with-lease  # 推送优化版本"
