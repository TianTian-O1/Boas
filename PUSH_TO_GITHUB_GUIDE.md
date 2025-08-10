# 🚀 Boas项目Git分支重组与推送指南

## 📋 当前状态

✅ **本地分支结构已完成**:
- `main`: 包含最新NPU优化版本 (commit: 5a941bf)
- `windows`: 保存原始main分支内容 
- `windows-backup`: 备份分支

✅ **本地提交已完成**:
- 所有NPU优化代码已提交到本地main分支
- 包含完整的CANN集成、性能测试、可视化等功能

## 🔑 推送到GitHub的步骤

### 方案1: 使用Personal Access Token (推荐)

1. **获取GitHub Token**:
   ```bash
   # 前往: https://github.com/settings/tokens
   # 点击 "Generate new token (classic)"
   # 选择权限: repo (完整仓库访问)
   # 复制生成的token
   ```

2. **配置认证**:
   ```bash
   # 设置凭据存储
   git config --global credential.helper store
   
   # 首次推送时会提示输入用户名和密码
   # 用户名: TianTian-O1
   # 密码: [粘贴你的Personal Access Token]
   ```

3. **推送main分支** (强制推送，覆盖远程main):
   ```bash
   git push origin main --force-with-lease
   ```

4. **推送windows分支** (保存原始版本):
   ```bash
   git push origin windows
   ```

### 方案2: 修改认证问题后推送

如果遇到403错误，可以:

1. **检查远程URL**:
   ```bash
   git remote -v
   # 应该显示: https://github.com/TianTian-O1/boas.git
   ```

2. **重新设置远程URL** (如果需要):
   ```bash
   git remote set-url origin https://github.com/TianTian-O1/boas.git
   ```

3. **使用缓存凭据**:
   ```bash
   git config --global credential.helper cache
   git config --global credential.helper 'cache --timeout=3600'
   ```

### 方案3: SSH认证 (如果有SSH密钥)

1. **检查SSH密钥**:
   ```bash
   ls -la ~/.ssh/
   # 查找 id_rsa, id_rsa.pub 或 id_ed25519
   ```

2. **更改为SSH URL**:
   ```bash
   git remote set-url origin git@github.com:TianTian-O1/boas.git
   ```

3. **推送**:
   ```bash
   git push origin main --force-with-lease
   git push origin windows
   ```

## 🎯 推送命令汇总

```bash
# 第一步: 推送优化后的main分支 (覆盖远程)
git push origin main --force-with-lease

# 第二步: 推送windows分支 (保存原始版本)  
git push origin windows

# 第三步: 验证推送结果
git fetch origin
git branch -a
```

## ⚠️ 重要说明

- **`--force-with-lease`**: 安全的强制推送，避免覆盖他人更改
- **windows分支**: 保存了原始main分支内容，确保不丢失
- **备份建议**: 推送前可以创建本地备份

## 🏆 推送后的分支结构

```
GitHub仓库:
├── main (新) ← NPU优化完整版本 🚀
├── windows (新) ← 原始main分支内容 💾
└── 其他分支 (保持不变)
```

## 📊 新main分支包含的内容

✨ **NPU优化特性**:
- CANN Runtime完整集成
- NPU矩阵乘法优化
- 性能基准测试框架  
- 可视化图表生成
- MLIR优化管道

🎯 **性能目标**: 2028.6 GFLOPS (90% vs PyTorch)

📁 **项目结构**:
- 重新组织的目录结构
- 完整的文档和测试
- 优化脚本和工具

## 🚀 执行推送

运行以下命令开始推送:

```bash
# 快速推送脚本
./quick_commit.sh

# 或手动推送
git push origin main --force-with-lease
git push origin windows
```

**成功推送后，Boas项目将拥有最新的NPU优化版本作为主分支！** 🎉
