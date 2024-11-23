#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <regex>
#include <map>
#include <array>
#include <memory>
#include <sstream>

struct BoasPerformanceMetrics {
    double compile_time_ms = 0.0;
    double mlir_gen_time_ms = 0.0;
    double llvm_ir_time_ms = 0.0;
    double execution_time_ms = 0.0;
    double memory_usage_mb = 0.0;
    double total_time_ms = 0.0;
    double peak_memory_kb = 0.0;
    double ipc = 0.0;
    std::map<std::string, double> phase_times;
};

class BoasPerformanceAnalyzer {
private:
    std::string boas_compiler_path;
    std::string output_dir;
    std::vector<int> matrix_sizes = {64, 128, 256, 512, 1024};

    std::string executeCommand(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        
        return result;
    }

    BoasPerformanceMetrics parseCompilerOutput(const std::string& output) {
        BoasPerformanceMetrics metrics;
        std::istringstream iss(output);
        std::string line;
        
        std::regex time_pattern(R"(Time taken: (\d+\.?\d*) ms)");
        std::regex phase_pattern(R"((\w+) phase took: (\d+\.?\d*) ms)");
        std::regex memory_pattern(R"(Memory usage: (\d+\.?\d*) MB)");
        
        while (std::getline(iss, line)) {
            std::smatch matches;
            if (std::regex_search(line, matches, time_pattern)) {
                metrics.execution_time_ms = std::stod(matches[1]);
            }
            else if (std::regex_search(line, matches, phase_pattern)) {
                std::string phase = matches[1];
                double time = std::stod(matches[2]);
                metrics.phase_times[phase] = time;
                
                if (phase == "MLIR") {
                    metrics.mlir_gen_time_ms = time;
                }
                else if (phase == "LLVM") {
                    metrics.llvm_ir_time_ms = time;
                }
            }
            else if (std::regex_search(line, matches, memory_pattern)) {
                metrics.memory_usage_mb = std::stod(matches[1]);
            }
        }
        
        metrics.compile_time_ms = metrics.mlir_gen_time_ms + metrics.llvm_ir_time_ms;
        return metrics;
    }

    std::string generateTestProgram(int size) {
        std::ostringstream program;
        program << "def main():\n"
                << "    A = tensor.random(" << size << ", " << size << ")\n"
                << "    B = tensor.random(" << size << ", " << size << ")\n"
                << "    print(\"Start\")\n"
                << "    C = tensor.matmul(A, B)\n"
                << "    print(\"End\")\n";
        return program.str();
    }

    void analyzeBottlenecks(int size, const BoasPerformanceMetrics& metrics, 
                           std::ofstream& report) {
        // 定义预期的峰值性能（根据硬件规格调整）
        const double expected_peak_gflops = 100.0; // 示例值，需要根据实际CPU调整
        
        report << "Performance Analysis:\n";
        
        // 编译时间分析
        double compile_ratio = metrics.compile_time_ms / metrics.total_time_ms;
        report << "Compilation overhead: " << (compile_ratio * 100) << "%\n";
        
        // 内存使用分析
        report << "Memory efficiency:\n";
        report << "- Peak memory: " << metrics.peak_memory_kb << " KB\n";
        report << "- Memory per matrix element: " 
               << (double)metrics.peak_memory_kb / (size * size) << " KB\n";
        
        // 计算效率分析
        double achieved_gflops = (2.0 * size * size * size) / 
                                (metrics.execution_time_ms * 1e6);
        report << "Computational efficiency:\n";
        report << "- GFLOPS: " << achieved_gflops << "\n";
        report << "- Instructions per cycle: " << metrics.ipc << "\n";
        
        // 提供优化建议
        if (compile_ratio > 0.3 || metrics.ipc < 1.0 || 
            achieved_gflops < expected_peak_gflops * 0.1) {
            report << "\nOptimization suggestions:\n";
            // ... 根据具体指标提供优化建议
        }
    }

public:
    BoasPerformanceAnalyzer(const std::string& compiler_path, const std::string& out_dir) 
        : boas_compiler_path(compiler_path), output_dir(out_dir) {}

    void runAnalysis() {
        std::map<int, BoasPerformanceMetrics> all_metrics;
        std::ofstream report(output_dir + "/boas_performance_report.txt");
        
        for (int size : matrix_sizes) {
            std::string program = generateTestProgram(size);
            std::string temp_file = output_dir + "/temp_" + std::to_string(size) + ".bs";
            
            std::ofstream prog_file(temp_file);
            prog_file << program;
            prog_file.close();
            
            std::string cmd = boas_compiler_path + " --dump-timing --dump-mlir --dump-llvm " 
                             + temp_file + " 2>&1";
            
            try {
                std::string output = executeCommand(cmd);
                all_metrics[size] = parseCompilerOutput(output);
                analyzeBottlenecks(size, all_metrics[size], report);
            } catch (const std::exception& e) {
                std::cerr << "Error analyzing size " << size << ": " << e.what() << std::endl;
            }
            
            std::remove(temp_file.c_str());
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <boas_compiler_path> <output_dir>\n";
        return 1;
    }

    BoasPerformanceAnalyzer analyzer(argv[1], argv[2]);
    analyzer.runAnalysis();
    return 0;
} 