#!/bin/bash

# 初始化git仓库
git init

# 添加.gitignore文件
cat > .gitignore << 'EOF'
build/
.vscode/
*.o
*.a
EOF


# 创建def_print分支
git checkout -b matmul

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Add matmul operations"


# 添加远程仓库
# git remote add origin https://github.com/TianZhenGG/Boas.git

# 推送到远程
git push -u origin matmul