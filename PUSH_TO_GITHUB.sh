#!/bin/bash
# Boas-NPU GitHub推送脚本

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         Boas-NPU 推送到GitHub                             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

echo "✅ 已完成的准备工作:"
echo "  - Git仓库已初始化"
echo "  - 远程仓库已添加: https://github.com/TianTian-O1/Boas.git"
echo "  - 分支已设置为main"
echo "  - Commit已创建 (53个文件, 6435行)"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  认证方式选择"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "请选择推送方式:"
echo ""
echo "方式1: 使用Personal Access Token (推荐)"
echo "方式2: 使用SSH Key"
echo "方式3: 查看手动推送命令"
echo ""

read -p "请输入选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  方式1: Personal Access Token"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "步骤:"
        echo "1. 访问: https://github.com/settings/tokens"
        echo "2. 点击 'Generate new token' → 'Generate new token (classic)'"
        echo "3. 设置:"
        echo "   - Note: Boas-NPU"
        echo "   - Expiration: 选择过期时间"
        echo "   - Scopes: 勾选 'repo'"
        echo "4. 生成并复制Token"
        echo ""
        read -p "请输入你的GitHub用户名: " username
        read -p "请输入你的Personal Access Token: " token
        echo ""
        echo "正在推送..."
        git push https://${username}:${token}@github.com/TianTian-O1/Boas.git main
        ;;
    2)
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  方式2: SSH Key"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "步骤:"
        echo "1. 生成SSH密钥（如果还没有）:"
        echo "   ssh-keygen -t ed25519 -C \"your_email@example.com\""
        echo ""
        echo "2. 复制公钥:"
        echo "   cat ~/.ssh/id_ed25519.pub"
        echo ""
        echo "3. 添加到GitHub:"
        echo "   访问: https://github.com/settings/keys"
        echo "   点击 'New SSH key'"
        echo "   粘贴公钥内容"
        echo ""
        read -p "SSH密钥已配置？按Enter继续..." 
        echo ""
        echo "更改远程URL为SSH格式..."
        git remote set-url origin git@github.com:TianTian-O1/Boas.git
        echo "正在推送..."
        git push -u origin main
        ;;
    3)
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  手动推送命令"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "使用Personal Access Token:"
        echo "─────────────────────────────────────────────────────────"
        echo "git push https://你的用户名:你的Token@github.com/TianTian-O1/Boas.git main"
        echo ""
        echo "或使用SSH:"
        echo "─────────────────────────────────────────────────────────"
        echo "git remote set-url origin git@github.com:TianTian-O1/Boas.git"
        echo "git push -u origin main"
        echo ""
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  推送完成！"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "访问你的仓库:"
echo "https://github.com/TianTian-O1/Boas"
echo ""
