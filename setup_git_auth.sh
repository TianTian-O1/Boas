#!/bin/bash

# 🔑 设置Git认证

echo "🔑 设置Git认证..."

# 提示用户输入Personal Access Token
echo "请提供GitHub Personal Access Token (如果有的话):"
echo "或者我们将使用SSH密钥认证"
echo ""

# 检查是否有SSH密钥
if [ -f ~/.ssh/id_rsa ] || [ -f ~/.ssh/id_ed25519 ]; then
    echo "✅ 发现SSH密钥，尝试使用SSH认证..."
    
    # 更改远程URL为SSH
    git remote set-url origin git@github.com:TianTian-O1/boas.git
    echo "🔄 已更改远程URL为SSH: git@github.com:TianTian-O1/boas.git"
    
else
    echo "⚠️ 未发现SSH密钥"
    echo "📝 建议设置GitHub Personal Access Token"
    echo ""
    echo "获取Token的步骤:"
    echo "1. 前往 https://github.com/settings/tokens"
    echo "2. 生成新的Personal Access Token"
    echo "3. 选择 'repo' 权限"
    echo "4. 复制token"
    echo ""
    
    # 临时使用用户名密码方式
    git config --global credential.helper store
    echo "🔧 已配置凭据存储"
fi

echo "✅ Git认证配置完成"
