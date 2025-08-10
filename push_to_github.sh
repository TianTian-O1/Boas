#!/bin/bash
# 推送BOAS到GitHub - TianTian-O1/boas

echo "========================================"
echo "推送 BOAS v1.0.0 到 GitHub"
echo "仓库: https://github.com/TianTian-O1/boas"
echo "========================================"

# 检查是否已经有远程仓库
echo "当前远程仓库配置："
git remote -v

echo ""
echo "准备推送到 TianTian-O1/boas ..."
echo ""

# 尝试获取远程信息（检查仓库是否存在）
echo "检查远程仓库状态..."
git ls-remote origin 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ 远程仓库存在"
    
    # 检查是否有main分支
    if git ls-remote --heads origin main | grep -q main; then
        echo "⚠️  远程仓库已有main分支"
        echo ""
        echo "选项："
        echo "1. 强制覆盖main分支（原内容将丢失）"
        echo "2. 先备份原main到windows分支，再推送"
        echo "3. 取消推送"
        echo ""
        read -p "请选择 (1/2/3): " choice
        
        case $choice in
            1)
                echo "强制推送到main分支..."
                git push -f origin main
                ;;
            2)
                echo "备份原main分支到windows..."
                git fetch origin main:windows-backup
                git push origin windows-backup:windows
                echo "推送新的main分支..."
                git push -f origin main
                ;;
            3)
                echo "取消推送"
                exit 0
                ;;
            *)
                echo "无效选项"
                exit 1
                ;;
        esac
    else
        echo "推送到新的main分支..."
        git push -u origin main
    fi
else
    echo "⚠️  远程仓库可能不存在或无法访问"
    echo ""
    echo "请先在GitHub上创建仓库："
    echo "1. 访问 https://github.com/new"
    echo "2. 仓库名称: boas"
    echo "3. 不要初始化README、.gitignore或LICENSE"
    echo "4. 创建后再运行此脚本"
    echo ""
    read -p "仓库已创建？继续推送？(y/n): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo "推送到main分支..."
        git push -u origin main
    else
        echo "取消推送"
        exit 0
    fi
fi

# 检查推送结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
    echo ""
    echo "📌 仓库地址: https://github.com/TianTian-O1/boas"
    echo "📌 分支: main"
    echo ""
    echo "下一步操作："
    echo "1. 访问 https://github.com/TianTian-O1/boas/settings"
    echo "2. 设置main为默认分支"
    echo "3. 添加项目描述: 'High-performance AI compiler with Python syntax'"
    echo "4. 添加Topics: python, compiler, npu, ai, ascend, mlir"
    echo "5. 创建Release: https://github.com/TianTian-O1/boas/releases/new"
    echo "   - Tag: v1.0.0"
    echo "   - Title: BOAS v1.0.0 - Python Syntax, C++ Performance"
else
    echo ""
    echo "❌ 推送失败"
    echo ""
    echo "可能的原因："
    echo "1. 网络连接问题"
    echo "2. 需要GitHub认证（用户名/密码或token）"
    echo "3. 仓库不存在"
    echo ""
    echo "解决方法："
    echo "1. 检查网络连接"
    echo "2. 配置GitHub认证："
    echo "   git config --global credential.helper store"
    echo "3. 确保在 https://github.com/TianTian-O1 创建了 'boas' 仓库"
fi