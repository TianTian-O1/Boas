# PowerShell脚本用于Windows环境下运行基准测试

# 检查必要的命令是否存在
function Check-Command {
    param (
        [string]$command,
        [string]$description
    )
    if (!(Get-Command $command -ErrorAction SilentlyContinue)) {
        Write-Warning "$command is not installed, skipping $description benchmarks"
        return $false
    }
    return $true
}

# 检查是否在正确的目录
if (!(Test-Path "../results")) {
    Write-Error "Please run this script from the benchmark/scripts directory"
    exit 1
}

# 清理旧结果
if (Test-Path "../results/results.csv") {
    Remove-Item "../results/results.csv"
}

# 创建结果文件头
"name,language,size,time_ms" | Out-File -FilePath "../results/results.csv" -Encoding utf8

# 编译并运行C++基准测试
Write-Host "Building and running C++ benchmarks..."
New-Item -ItemType Directory -Force -Path "../build" | Out-Null
Set-Location "../build"

# 使用Visual Studio生成器
cmake -G "Visual Studio 17 2022" ..
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed"
    exit 1
}

# 构建Release版本
cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit 1
}

# 运行基准测试程序
if (Test-Path "Release/cpp_bench.exe") {
    $output = ./Release/cpp_bench.exe
    $output | Out-File -FilePath "../results/results.csv" -Encoding utf8 -Append
}
else {
    Write-Error "cpp_bench.exe not found"
    exit 1
}

# 运行Python基准测试
Write-Host "Running Python benchmarks..."
Set-Location "../src"
if (Test-Path "python_bench.py") {
    python python_bench.py
}
else {
    Write-Warning "python_bench.py not found"
}

# 运行Boas基准测试
Write-Host "Running Boas benchmarks..."
Set-Location "../../"  # 返回到项目根目录
if (Test-Path "benchmark/src/boas_bench.bs") {
    if (Test-Path "build/Release/matrix-compiler.exe") {
        Write-Host "Executing program..."
        ./build/Release/matrix-compiler.exe --run benchmark/src/boas_bench.bs
    }
    else {
        Write-Warning "matrix-compiler.exe not found"
    }
}

# 生成图表
Write-Host "Generating plots..."
Set-Location "benchmark/scripts"
python plot.py

Write-Host "Benchmark complete! Results are in ../results/comparison.png" 