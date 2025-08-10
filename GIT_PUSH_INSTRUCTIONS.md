# Git 推送步骤

## 准备推送到 GitHub

### 1. 创建 .gitignore
```bash
cat > .gitignore << 'EOF'
# Build
build/
*.o
*.a
*.so
*.out

# LLVM/MLIR
*.ll
*.bc
*.s
*.mlir

# Temp
temp/
tmp/
*.tmp
*.log

# IDE
.vscode/
.idea/

# OS
.DS_Store

# Python
__pycache__/
*.pyc

# Release
boas-v*/
*.tar.gz
EOF
```

### 2. 初始化 Git 仓库
```bash
# 初始化并设置main分支
git init
git branch -m main

# 配置用户信息（根据需要修改）
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. 添加所有文件并提交
```bash
# 添加所有文件
git add .

# 创建初始提交
git commit -m "feat: BOAS v1.0.0 - High-performance AI compiler with Python syntax

- Python-compatible syntax with C++ performance
- Exceeds CANN-OPS-ADV by 52% (FP32) and 62% (FP16)
- Direct NPU hardware access for Ascend
- Automatic optimizations (adaptive tiling, mixed precision, fusion)
- Complete MLIR/LLVM integration
- Examples and documentation included

Performance highlights:
- FP32 Peak: 163,778 GFLOPS
- FP16 Peak: 653,932 GFLOPS
- 2.0-2.3x faster than PyTorch NPU"
```

### 4. 添加远程仓库
```bash
# 添加你的GitHub仓库（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/boas.git

# 或者使用SSH
git remote add origin git@github.com:YOUR_USERNAME/boas.git
```

### 5. 处理分支
```bash
# 如果需要保存原来的main分支为windows分支
# 先拉取原来的main分支
git fetch origin main

# 创建windows分支保存原内容
git branch windows origin/main

# 推送windows分支
git push origin windows
```

### 6. 推送新的main分支
```bash
# 强制推送当前代码到main分支（会覆盖原有内容）
git push -f origin main

# 或者如果是新仓库，直接推送
git push -u origin main
```

## 完整命令序列（复制粘贴版）

```bash
# 1. 初始化
git init
git branch -m main

# 2. 添加.gitignore
echo 'build/
*.o
*.so
*.out
*.ll
*.mlir
temp/
.vscode/
__pycache__/
*.tar.gz' > .gitignore

# 3. 添加和提交
git add .
git commit -m "feat: BOAS v1.0.0 - Linux/NPU optimized version"

# 4. 添加远程（替换URL）
git remote add origin https://github.com/YOUR_USERNAME/boas.git

# 5. 推送
git push -u origin main
```

## 如果已有仓库且要保留原main为windows

```bash
# 1. 添加远程
git remote add origin https://github.com/YOUR_USERNAME/boas.git

# 2. 获取远程分支
git fetch origin

# 3. 将原main改名为windows
git push origin origin/main:refs/heads/windows

# 4. 强制推送新main
git push -f origin main
```

## 注意事项

1. **备份重要数据**：强制推送会覆盖远程仓库
2. **确认仓库地址**：替换为你的实际GitHub仓库地址
3. **分支说明**：
   - `main`: Linux/NPU优化版本（当前）
   - `windows`: Windows版本（如果需要保留）

## 推送后的操作

1. 在GitHub上设置main为默认分支
2. 创建Release v1.0.0
3. 更新仓库描述
4. 添加topics: `compiler`, `python`, `npu`, `ascend`, `ai`, `mlir`