#!/bin/bash
# LLVM 20 实时编译状态监控脚本

BUILD_DIR="/tmp/llvm-20/build"
LOG_FILE="/root/Boas/Boas-linux/llvm_monitor.log"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 清屏函数
clear_screen() {
    clear
    echo -e "${BLUE}==================== LLVM 20 实时编译监控 ====================${NC}"
    echo "开始时间: $(date)"
    echo "监控目录: $BUILD_DIR"
    echo
}

# 获取编译统计
get_stats() {
    # 进程统计
    MAKE_PROCESSES=$(ps aux | grep "make.*llvm" | grep -v grep | wc -l)
    CPP_PROCESSES=$(ps aux | grep "c++.*llvm" | grep -v grep | wc -l)
    LINK_PROCESSES=$(ps aux | grep "cmake_link_script" | grep -v grep | wc -l)
    
    # 文件统计
    if [ -d "$BUILD_DIR" ]; then
        BUILD_SIZE=$(du -sh $BUILD_DIR 2>/dev/null | cut -f1)
        SO_FILES=$(find $BUILD_DIR -name "*.so" 2>/dev/null | wc -l)
        BIN_FILES=$(find $BUILD_DIR/bin -type f -executable 2>/dev/null | wc -l)
        MLIR_TOOLS=$(find $BUILD_DIR/bin -name "*mlir*" -executable 2>/dev/null | wc -l)
        CLANG_TOOLS=$(find $BUILD_DIR/bin -name "*clang*" 2>/dev/null | wc -l)
    else
        BUILD_SIZE="0"
        SO_FILES=0
        BIN_FILES=0
        MLIR_TOOLS=0
        CLANG_TOOLS=0
    fi
    
    # 内存使用
    MEMORY_USAGE=$(ps aux | grep -E "(make|c\+\+).*llvm" | grep -v grep | awk '{sum += $6} END {print sum/1024}' | cut -d. -f1)
    if [ -z "$MEMORY_USAGE" ]; then
        MEMORY_USAGE=0
    fi
}

# 显示状态
show_status() {
    get_stats
    
    echo -e "${YELLOW}📊 编译进程状态:${NC}"
    echo "  Make进程: $MAKE_PROCESSES"
    echo "  C++编译进程: $CPP_PROCESSES" 
    echo "  链接进程: $LINK_PROCESSES"
    echo "  内存使用: ${MEMORY_USAGE}MB"
    echo
    
    echo -e "${YELLOW}📁 构建统计:${NC}"
    echo "  构建大小: $BUILD_SIZE"
    echo "  共享库: $SO_FILES 个"
    echo "  可执行文件: $BIN_FILES 个"
    echo "  MLIR工具: $MLIR_TOOLS 个"
    echo "  Clang工具: $CLANG_TOOLS 个"
    echo
    
    # 编译阶段判断
    if [ $MAKE_PROCESSES -eq 0 ] && [ $CPP_PROCESSES -eq 0 ] && [ $LINK_PROCESSES -eq 0 ]; then
        echo -e "${GREEN}✅ 编译状态: 已完成！${NC}"
        return 0
    elif [ $LINK_PROCESSES -gt 0 ]; then
        echo -e "${BLUE}🔗 编译状态: 链接阶段${NC}"
    elif [ $CPP_PROCESSES -gt 0 ]; then
        echo -e "${YELLOW}🔧 编译状态: 源码编译中${NC}"
    else
        echo -e "${YELLOW}⏳ 编译状态: 其他阶段${NC}"
    fi
    
    # 最新生成的文件
    echo -e "${YELLOW}📄 最新生成文件:${NC}"
    if [ -d "$BUILD_DIR/bin" ]; then
        ls -lt $BUILD_DIR/bin/ 2>/dev/null | head -3 | tail -2 | while read line; do
            echo "  $line"
        done
    fi
    
    echo
    echo -e "${BLUE}更新时间: $(date)${NC}"
    echo "按 Ctrl+C 退出监控"
    echo "=================================================="
}

# 实时监控模式
real_time_monitor() {
    while true; do
        clear_screen
        show_status
        
        # 检查是否完成
        if [ $MAKE_PROCESSES -eq 0 ] && [ $CPP_PROCESSES -eq 0 ] && [ $LINK_PROCESSES -eq 0 ]; then
            echo -e "${GREEN}🎉 LLVM 20 编译完成！${NC}"
            echo "可以开始安装和测试了"
            break
        fi
        
        sleep 5
    done
}

# 简单状态检查
simple_check() {
    get_stats
    echo "编译进程: Make($MAKE_PROCESSES) + C++($CPP_PROCESSES) + 链接($LINK_PROCESSES)"
    echo "构建大小: $BUILD_SIZE | 工具: MLIR($MLIR_TOOLS) + Clang($CLANG_TOOLS)"
    
    if [ $MAKE_PROCESSES -eq 0 ] && [ $CPP_PROCESSES -eq 0 ] && [ $LINK_PROCESSES -eq 0 ]; then
        echo "状态: ✅ 编译完成"
    else
        echo "状态: ⏳ 编译中"
    fi
}

# 详细进程信息
detailed_processes() {
    echo "当前LLVM相关进程:"
    ps aux | grep -E "(make|c\+\+).*llvm" | grep -v grep | head -10
    echo
    echo "最占CPU的编译进程:"
    ps aux | grep -E "(make|c\+\+)" | grep -v grep | sort -k3 -nr | head -5
}

# 主菜单
case "$1" in
    "")
        # 默认实时监控
        real_time_monitor
        ;;
    "-s"|"--simple")
        simple_check
        ;;
    "-p"|"--processes")
        detailed_processes
        ;;
    "-h"|"--help")
        echo "LLVM编译监控脚本使用方法:"
        echo "  $0          - 实时监控模式"
        echo "  $0 -s       - 简单状态检查"
        echo "  $0 -p       - 详细进程信息"
        echo "  $0 -h       - 显示帮助"
        ;;
    *)
        echo "未知选项: $1"
        echo "使用 $0 -h 查看帮助"
        ;;
esac

# LLVM 20 实时编译状态监控脚本

BUILD_DIR="/tmp/llvm-20/build"
LOG_FILE="/root/Boas/Boas-linux/llvm_monitor.log"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 清屏函数
clear_screen() {
    clear
    echo -e "${BLUE}==================== LLVM 20 实时编译监控 ====================${NC}"
    echo "开始时间: $(date)"
    echo "监控目录: $BUILD_DIR"
    echo
}

# 获取编译统计
get_stats() {
    # 进程统计
    MAKE_PROCESSES=$(ps aux | grep "make.*llvm" | grep -v grep | wc -l)
    CPP_PROCESSES=$(ps aux | grep "c++.*llvm" | grep -v grep | wc -l)
    LINK_PROCESSES=$(ps aux | grep "cmake_link_script" | grep -v grep | wc -l)
    
    # 文件统计
    if [ -d "$BUILD_DIR" ]; then
        BUILD_SIZE=$(du -sh $BUILD_DIR 2>/dev/null | cut -f1)
        SO_FILES=$(find $BUILD_DIR -name "*.so" 2>/dev/null | wc -l)
        BIN_FILES=$(find $BUILD_DIR/bin -type f -executable 2>/dev/null | wc -l)
        MLIR_TOOLS=$(find $BUILD_DIR/bin -name "*mlir*" -executable 2>/dev/null | wc -l)
        CLANG_TOOLS=$(find $BUILD_DIR/bin -name "*clang*" 2>/dev/null | wc -l)
    else
        BUILD_SIZE="0"
        SO_FILES=0
        BIN_FILES=0
        MLIR_TOOLS=0
        CLANG_TOOLS=0
    fi
    
    # 内存使用
    MEMORY_USAGE=$(ps aux | grep -E "(make|c\+\+).*llvm" | grep -v grep | awk '{sum += $6} END {print sum/1024}' | cut -d. -f1)
    if [ -z "$MEMORY_USAGE" ]; then
        MEMORY_USAGE=0
    fi
}

# 显示状态
show_status() {
    get_stats
    
    echo -e "${YELLOW}📊 编译进程状态:${NC}"
    echo "  Make进程: $MAKE_PROCESSES"
    echo "  C++编译进程: $CPP_PROCESSES" 
    echo "  链接进程: $LINK_PROCESSES"
    echo "  内存使用: ${MEMORY_USAGE}MB"
    echo
    
    echo -e "${YELLOW}📁 构建统计:${NC}"
    echo "  构建大小: $BUILD_SIZE"
    echo "  共享库: $SO_FILES 个"
    echo "  可执行文件: $BIN_FILES 个"
    echo "  MLIR工具: $MLIR_TOOLS 个"
    echo "  Clang工具: $CLANG_TOOLS 个"
    echo
    
    # 编译阶段判断
    if [ $MAKE_PROCESSES -eq 0 ] && [ $CPP_PROCESSES -eq 0 ] && [ $LINK_PROCESSES -eq 0 ]; then
        echo -e "${GREEN}✅ 编译状态: 已完成！${NC}"
        return 0
    elif [ $LINK_PROCESSES -gt 0 ]; then
        echo -e "${BLUE}🔗 编译状态: 链接阶段${NC}"
    elif [ $CPP_PROCESSES -gt 0 ]; then
        echo -e "${YELLOW}🔧 编译状态: 源码编译中${NC}"
    else
        echo -e "${YELLOW}⏳ 编译状态: 其他阶段${NC}"
    fi
    
    # 最新生成的文件
    echo -e "${YELLOW}📄 最新生成文件:${NC}"
    if [ -d "$BUILD_DIR/bin" ]; then
        ls -lt $BUILD_DIR/bin/ 2>/dev/null | head -3 | tail -2 | while read line; do
            echo "  $line"
        done
    fi
    
    echo
    echo -e "${BLUE}更新时间: $(date)${NC}"
    echo "按 Ctrl+C 退出监控"
    echo "=================================================="
}

# 实时监控模式
real_time_monitor() {
    while true; do
        clear_screen
        show_status
        
        # 检查是否完成
        if [ $MAKE_PROCESSES -eq 0 ] && [ $CPP_PROCESSES -eq 0 ] && [ $LINK_PROCESSES -eq 0 ]; then
            echo -e "${GREEN}🎉 LLVM 20 编译完成！${NC}"
            echo "可以开始安装和测试了"
            break
        fi
        
        sleep 5
    done
}

# 简单状态检查
simple_check() {
    get_stats
    echo "编译进程: Make($MAKE_PROCESSES) + C++($CPP_PROCESSES) + 链接($LINK_PROCESSES)"
    echo "构建大小: $BUILD_SIZE | 工具: MLIR($MLIR_TOOLS) + Clang($CLANG_TOOLS)"
    
    if [ $MAKE_PROCESSES -eq 0 ] && [ $CPP_PROCESSES -eq 0 ] && [ $LINK_PROCESSES -eq 0 ]; then
        echo "状态: ✅ 编译完成"
    else
        echo "状态: ⏳ 编译中"
    fi
}

# 详细进程信息
detailed_processes() {
    echo "当前LLVM相关进程:"
    ps aux | grep -E "(make|c\+\+).*llvm" | grep -v grep | head -10
    echo
    echo "最占CPU的编译进程:"
    ps aux | grep -E "(make|c\+\+)" | grep -v grep | sort -k3 -nr | head -5
}

# 主菜单
case "$1" in
    "")
        # 默认实时监控
        real_time_monitor
        ;;
    "-s"|"--simple")
        simple_check
        ;;
    "-p"|"--processes")
        detailed_processes
        ;;
    "-h"|"--help")
        echo "LLVM编译监控脚本使用方法:"
        echo "  $0          - 实时监控模式"
        echo "  $0 -s       - 简单状态检查"
        echo "  $0 -p       - 详细进程信息"
        echo "  $0 -h       - 显示帮助"
        ;;
    *)
        echo "未知选项: $1"
        echo "使用 $0 -h 查看帮助"
        ;;
esac
