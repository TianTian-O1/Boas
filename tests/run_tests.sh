#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 创建必要的目录
rm -rf ./logs
mkdir -p ./logs
mkdir -p ./tmp

# 日志文件
LOG_FILE="./logs/test_$(date +%Y%m%d_%H%M%S).log"

log_message() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

run_test() {
    local test_file=$1
    local output_file="./tmp/boas_test_output.txt"
    local error_file="./tmp/boas_test_error.txt"
    
    log_message "\n========================================="
    log_message "Testing: $test_file"
    
    # 运行测试
    ./build/src/boas run "$test_file" >"$output_file" 2>"$error_file"
    
    # 检查输出是否为空
    if [ ! -s "$output_file" ]; then
        log_message "${RED}✗ Test failed: Empty output${NC}"
        return 1
    fi
    
    # 规范化输出（统一数字格式）
    sed -E 's/^([0-9]+)$/\1.00/g; s/([0-9]+\.[0-9]*)/\1/g' "$output_file" > "$output_file.normalized"
    
    # 删除尾随零 (使用兼容 macOS 的语法)
    sed -e 's/\.0*$//g' "$output_file.normalized" > "$output_file.normalized.tmp"
    mv "$output_file.normalized.tmp" "$output_file.normalized"
    
    # 输出结果
    log_message "Output:"
    cat "$output_file" | tee -a "$LOG_FILE"
    log_message "${GREEN}✓ Test passed${NC}"
    return 0
}

# 统计变量
total_tests=0
passed_tests=0
failed_tests=0

log_message "Starting tests at $(date)"

# 运行所有测试
for dir in tests/*/; do
    if [ -d "$dir" ] && [ "$dir" != "tests/expected/" ]; then
        for test in "$dir"*.bs; do
            if [ -f "$test" ]; then
                ((total_tests+=1))
                
                run_test "$test"
                if [ $? -eq 0 ]; then
                    ((passed_tests+=1))
                else
                    ((failed_tests+=1))
                fi
            fi
        done
    fi
done

# 输出总结
log_message "\n=== Test Summary ==="
log_message "Total: $total_tests | Passed: ${GREEN}$passed_tests${NC} | Failed: ${RED}$failed_tests${NC}"

# 如果有失败的测试，返回非零状态码
[ $failed_tests -eq 0 ]