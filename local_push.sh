#!/bin/bash
# 本地推送脚本 - 在你的本地机器上运行

echo "========================================"
echo "BOAS v1.0.0 本地推送脚本"
echo "========================================"

# 检查是否在正确的目录
if [ ! -f "README.md" ] || [ ! -d ".git" ]; then
    echo "❌ 请在BOAS项目目录中运行此脚本"
    exit 1
fi

# GitHub Token (已提供)
TOKEN="YOUR_GITHUB_TOKEN_HERE"

echo "配置Git..."
git config user.name "TianTian-O1"
git config user.email "boas-dev@example.com"

echo "设置远程仓库..."
git remote remove origin 2>/dev/null
git remote add origin https://${TOKEN}@github.com/TianTian-O1/boas.git

echo "获取远程分支信息..."
git fetch origin 2>/dev/null

# 备份原main到windows
if git ls-remote --heads origin main | grep -q main; then
    echo "备份原main分支到windows..."
    git branch windows origin/main 2>/dev/null
    git push origin windows 2>/dev/null
fi

echo "推送到main分支..."
git push -f origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
    echo ""
    echo "📌 查看: https://github.com/TianTian-O1/boas"
    echo ""
    echo "后续操作："
    echo "1. 创建Release: https://github.com/TianTian-O1/boas/releases/new"
    echo "   Tag: v1.0.0"
    echo "2. 添加描述和Topics"
else
    echo "❌ 推送失败"
fi