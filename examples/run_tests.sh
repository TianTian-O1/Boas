#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 在颜色定义后添加
declare -A run_results
declare -A build_results

rm -rf ./logs
# 创建日志目录
mkdir -p ./logs
mkdir -p ./tmp

# 日志文件
LOG_FILE="./logs/test_$(date +%Y%m%d_%H%M%S).log"

log_message() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

run_test() {
    local test_file=$1
    local mode=$2
    local expected_file="examples/expected/$(basename ${test_file%.*})_${mode}.txt"
    local output_file="./tmp/boas_test_output.txt"
    local error_file="./tmp/boas_test_error.txt"
    
    log_message "\n========================================="
    log_message "Testing: $test_file"
    log_message "Mode: $mode"
    
    if [ "$mode" = "build" ]; then
        ./build/src/boas build "$test_file" -o ./tmp/test_out 2>"$error_file"
        if [ $? -ne 0 ]; then
            log_message "${RED}Build failed:${NC}"
            cat "$error_file" | tee -a "$LOG_FILE"
            return 1
        fi
        
        # 过滤掉编译成功的消息
        ./tmp/test_out >"$output_file" 2>"$error_file"
        grep -v "已生��可执行文件" "$output_file" > "$output_file.tmp"
        mv "$output_file.tmp" "$output_file"
    else
        ./build/src/boas run "$test_file" >"$output_file" 2>"$error_file"
    fi
    
   # 比较输出
if [ -f "$expected_file" ]; then
    # 规范化输出（统一数字格式）
    # 处理整数和浮点数
    sed -E 's/^([0-9]+)$/\1.00/g; s/([0-9]+\.[0-9]*)/\1/g' "$output_file" > "$output_file.normalized"
    sed -E 's/^([0-9]+)$/\1.00/g; s/([0-9]+\.[0-9]*)/\1/g' "$expected_file" > "$expected_file.normalized"
    
    # 删除尾随零
    sed -i 's/\.0*$//g' "$output_file.normalized"
    sed -i 's/\.0*$//g' "$expected_file.normalized"
    
    if diff "$output_file.normalized" "$expected_file.normalized" >/dev/null; then
        log_message "${GREEN}✓ Test passed${NC}"
        return 0
    else
        log_message "${RED}✗ Test failed${NC}"
        log_message "Expected output (normalized):"
        cat "$expected_file.normalized" | tee -a "$LOG_FILE"
        log_message "Actual output (normalized):"
        cat "$output_file.normalized" | tee -a "$LOG_FILE"
        log_message "Raw output:"
        cat "$output_file" | tee -a "$LOG_FILE"
        return 1
    fi
else
    log_message "${YELLOW}Creating new test output:${NC}"
    mkdir -p "$(dirname "$expected_file")"
    cat "$output_file" | tee -a "$LOG_FILE"
    cp "$output_file" "$expected_file"
    return 0
fi
}
# 统计变量
total_tests=0
passed_tests=0
failed_tests=0

log_message "Starting tests at $(date)"

# 运行所有测试
for dir in examples/*/; do
    if [ -d "$dir" ] && [ "$dir" != "examples/expected/" ]; then
        for test in "$dir"*.bs; do
            if [ -f "$test" ]; then
                ((total_tests+=2))  # 每个文件测试两种模式
                
               # 运行run模式测试
run_test "$test" "run"
run_result=$?
if [ $run_result -eq 0 ]; then
    ((passed_tests+=1))
    run_results[$test]="${GREEN}✓ Passed${NC}"
else
    ((failed_tests+=1))
    run_results[$test]="${RED}✗ Failed${NC}"
fi

# 运行build模式测试
run_test "$test" "build"
build_result=$?
if [ $build_result -eq 0 ]; then
    ((passed_tests+=1))
    build_results[$test]="${GREEN}✓ Passed${NC}"
else
    ((failed_tests+=1))
    build_results[$test]="${RED}✗ Failed${NC}"
fi
            fi
        done
    fi
done

# 输出总结
print_summary() {
    log_message "\n=== Test Summary ==="
    
    # 遍历所有测试文件
    for dir in examples/*/; do
        if [ -d "$dir" ] && [ "$dir" != "examples/expected/" ]; then
            for test in "$dir"*.bs; do
                if [ -f "$test" ]; then
                    local test_name=$(basename "$test")
                    log_message "$test_name: ${run_results[$test]} | ${build_results[$test]}"
                fi
            done
        fi
    done
    
    log_message "\nTotal: $total_tests | Passed: ${GREEN}$passed_tests${NC} | Failed: ${RED}$failed_tests${NC}"
}

print_summary $total_tests $passed_tests $failed_tests

# 如果有失败的测试，返回非零状态码
[ $failed_tests -eq 0 ]