#!/bin/bash
# 直接配置GitHub Token推送

echo "========================================"
echo "GitHub Token 快速配置"
echo "========================================"
echo ""
echo "获取Personal Access Token步骤："
echo "1. 打开: https://github.com/settings/tokens"
echo "2. Generate new token (classic)"
echo "3. 勾选 'repo' 权限"
echo "4. 生成并复制token"
echo ""
echo "或使用已有token"
echo ""
read -p "请粘贴你的GitHub Personal Access Token: " TOKEN

if [ -z "$TOKEN" ]; then
    echo "❌ Token不能为空！"
    echo ""
    echo "请先获取token："
    echo "https://github.com/settings/tokens"
    exit 1
fi

# 配置git
echo ""
echo "配置Git..."
git config --global user.name "TianTian-O1"
git config --global user.email "your-email@example.com"

# 设置带token的远程URL
echo "设置远程仓库..."
git remote set-url origin https://${TOKEN}@github.com/TianTian-O1/boas.git

# 尝试推送
echo ""
echo "开始推送到GitHub..."
echo "目标: https://github.com/TianTian-O1/boas"
echo ""

# 先尝试获取远程信息
git fetch origin 2>/dev/null

# 检查是否有远程main分支
if git ls-remote --heads origin main | grep -q main; then
    echo "⚠️ 远程已有main分支，将覆盖..."
    
    # 备份原main到windows分支
    echo "备份原main分支到windows..."
    git fetch origin main:windows-backup 2>/dev/null
    git push origin windows-backup:windows 2>/dev/null
    
    # 强制推送新main
    echo "推送新的main分支..."
    git push -f origin main
else
    echo "推送到新的main分支..."
    git push -u origin main
fi

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
    echo ""
    echo "📌 仓库地址: https://github.com/TianTian-O1/boas"
    echo ""
    echo "下一步："
    echo "1. 查看仓库: https://github.com/TianTian-O1/boas"
    echo "2. 创建Release: https://github.com/TianTian-O1/boas/releases/new"
    echo "   - Tag: v1.0.0"
    echo "   - Title: BOAS v1.0.0"
    echo "3. 设置Topics: python, compiler, npu, ai, mlir"
    echo ""
    echo "🎉 恭喜！BOAS v1.0.0 已发布！"
else
    echo ""
    echo "❌ 推送失败"
    echo ""
    echo "可能的原因："
    echo "1. Token权限不足 (需要repo权限)"
    echo "2. Token已过期"
    echo "3. 网络问题"
    echo ""
    echo "请检查token并重试"
fi