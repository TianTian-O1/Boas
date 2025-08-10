#!/bin/bash
# GitHub 认证配置脚本

echo "========================================"
echo "GitHub 认证配置"
echo "========================================"
echo ""
echo "请选择认证方式："
echo "1. Personal Access Token (推荐)"
echo "2. SSH密钥"
echo "3. 临时用户名密码"
echo ""
read -p "选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "=== 配置 Personal Access Token ==="
        echo ""
        echo "步骤："
        echo "1. 访问 https://github.com/settings/tokens"
        echo "2. 点击 'Generate new token' → 'Generate new token (classic)'"
        echo "3. 设置："
        echo "   - Note: BOAS Push"
        echo "   - Expiration: 90 days (或自定义)"
        echo "   - Scopes: 勾选 'repo' (全部权限)"
        echo "4. 点击 'Generate token'"
        echo "5. 复制生成的token (只显示一次!)"
        echo ""
        read -p "请输入你的GitHub Personal Access Token: " token
        
        if [ -z "$token" ]; then
            echo "❌ Token不能为空"
            exit 1
        fi
        
        # 配置远程URL包含token
        echo "配置远程仓库..."
        git remote set-url origin https://${token}@github.com/TianTian-O1/boas.git
        
        echo "✅ Token配置成功！"
        echo ""
        echo "现在推送代码..."
        git push -f origin main
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 推送成功！"
            echo "📌 查看: https://github.com/TianTian-O1/boas"
        else
            echo "❌ 推送失败，请检查token权限"
        fi
        ;;
        
    2)
        echo ""
        echo "=== 配置 SSH 密钥 ==="
        echo ""
        
        # 检查是否已有SSH密钥
        if [ -f ~/.ssh/id_ed25519 ]; then
            echo "发现已有SSH密钥"
        else
            echo "生成新的SSH密钥..."
            read -p "请输入你的邮箱: " email
            ssh-keygen -t ed25519 -C "$email" -f ~/.ssh/id_ed25519 -N ""
        fi
        
        echo ""
        echo "你的SSH公钥："
        echo "-------------------"
        cat ~/.ssh/id_ed25519.pub
        echo "-------------------"
        echo ""
        echo "请复制上面的公钥，然后："
        echo "1. 访问 https://github.com/settings/keys"
        echo "2. 点击 'New SSH key'"
        echo "3. Title: BOAS Server"
        echo "4. 粘贴公钥"
        echo "5. 点击 'Add SSH key'"
        echo ""
        read -p "已添加SSH密钥？(y/n): " added
        
        if [ "$added" = "y" ] || [ "$added" = "Y" ]; then
            # 改用SSH URL
            git remote set-url origin git@github.com:TianTian-O1/boas.git
            
            # 添加GitHub到known_hosts
            ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
            
            echo "推送代码..."
            git push -f origin main
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ 推送成功！"
                echo "📌 查看: https://github.com/TianTian-O1/boas"
            else
                echo "❌ 推送失败，请检查SSH密钥配置"
            fi
        fi
        ;;
        
    3)
        echo ""
        echo "=== 临时用户名密码 ==="
        echo ""
        echo "注意：GitHub已不支持密码认证，必须使用Personal Access Token"
        echo "请选择选项1创建token"
        ;;
        
    *)
        echo "无效选项"
        exit 1
        ;;
esac