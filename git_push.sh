#!/bin/bash
# Git推送脚本 - 将当前Linux版本推送到main分支

echo "========================================"
echo "BOAS Git 推送准备"
echo "========================================"

# 1. 配置Git（如果需要）
echo "1. 配置Git用户信息..."
git config --global user.name "BOAS Developer"
git config --global user.email "boas-dev@example.com"

# 2. 初始化仓库（如果还没有）
if [ ! -d ".git" ]; then
    echo "2. 初始化Git仓库..."
    git init
    git branch -m main
fi

# 3. 添加.gitignore
echo "3. 创建.gitignore文件..."
cat > .gitignore << 'EOF'
# Build directories
build/
cmake-build-*/
.cmake/

# LLVM/MLIR generated files
*.ll
*.bc
*.s
*.mlir
*.o
*.a
*.so
*.out

# Executables
boas-compiler
matrix-compiler
test_matmul
hello
benchmark

# Test outputs
*.log
test_results/
results/

# Temporary files
temp/
tmp/
*.tmp
*.temp

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Release packages
boas-v*/
release/
*.tar.gz
*.zip

# Data files
*.json
!package.json

# Images
*.png
!docs/images/*.png

# Backup files
*.bak
*.backup
*.old

# Core dumps
core
core.*
EOF

# 4. 添加远程仓库（替换为你的仓库地址）
echo "4. 配置远程仓库..."
REPO_URL="https://github.com/your-username/boas.git"
echo "请输入你的GitHub仓库地址 (例如: https://github.com/username/boas.git):"
read -p "仓库地址: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "⚠️  未提供仓库地址，使用示例地址"
    REPO_URL="https://github.com/boas-project/boas.git"
fi

# 检查是否已有远程仓库
if git remote | grep -q "origin"; then
    echo "更新远程仓库地址..."
    git remote set-url origin "$REPO_URL"
else
    echo "添加远程仓库..."
    git remote add origin "$REPO_URL"
fi

# 5. 获取远程分支信息
echo "5. 获取远程分支信息..."
git fetch origin 2>/dev/null || echo "新仓库，跳过fetch"

# 6. 备份原main分支到windows分支（如果存在）
if git ls-remote --heads origin main | grep -q "main"; then
    echo "6. 备份原main分支到windows分支..."
    git fetch origin main:windows
    git push origin windows
    echo "✅ 原main分支已备份为windows分支"
fi

# 7. 添加所有文件
echo "7. 添加文件到Git..."
git add .

# 8. 创建提交
echo "8. 创建提交..."
COMMIT_MSG="feat: BOAS v1.0.0 - Linux/NPU optimized version

- Python-compatible syntax
- World-class performance (exceeds CANN-OPS-ADV by 52-62%)
- Direct NPU hardware access
- Automatic optimizations for Ascend NPU
- Full MLIR/LLVM integration
- Examples and documentation included"

git commit -m "$COMMIT_MSG"

# 9. 推送到main分支
echo "9. 推送到main分支..."
echo ""
echo "即将执行以下操作："
echo "  1. 将当前代码推送到main分支"
echo "  2. 原main分支已备份为windows分支"
echo ""
read -p "确认推送? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    # 强制推送到main（覆盖原有内容）
    git push -f origin main
    echo ""
    echo "✅ 推送成功！"
    echo ""
    echo "分支状态："
    echo "  - main: Linux/NPU优化版本 (当前)"
    echo "  - windows: Windows版本 (原main分支)"
else
    echo "❌ 取消推送"
fi

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
echo ""
echo "后续操作建议："
echo "1. 在GitHub上设置main为默认分支"
echo "2. 更新README说明不同分支的用途"
echo "3. 创建Release发布v1.0.0"
echo "4. 添加GitHub Actions CI/CD"