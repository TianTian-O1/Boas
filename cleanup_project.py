#!/usr/bin/env python3
"""
🧹 项目文件清理和整理工具
将优化过程中生成的文件分类整理到合适的目录
"""

import os
import shutil
import glob
from datetime import datetime

class ProjectCleaner:
    def __init__(self):
        self.cleanup_rules = {
            # 测试文件
            'test/': [
                'test_*.py',
                'test_*.bs', 
                'test_*.sh',
                '*.log'
            ],
            # 基准测试和优化工具
            'tools/optimization/': [
                '*benchmark*.py',
                '*optimization*.py',
                'analyze_*.py',
                'progressive_*.py'
            ],
            # 脚本文件
            'scripts/': [
                '*.sh'
            ],
            # 结果和报告
            'results/optimization/': [
                '*.json',
                '*report*',
                '*result*'
            ],
            # 临时和构建文件
            'temp/': [
                '*.mlir',
                '*.ll',
                '*.s',
                'temp*',
                '*.log'
            ]
        }
        
    def analyze_current_files(self):
        """分析当前文件分布"""
        print("🔍 当前项目文件分析")
        print("=" * 50)
        
        all_files = []
        for pattern in ['*.py', '*.sh', '*.json', '*.bs', '*.log', '*.mlir', '*.ll', '*.s']:
            all_files.extend(glob.glob(pattern))
            
        print(f"📁 根目录文件总数: {len(all_files)}")
        
        # 按类型分类
        file_types = {}
        for file in all_files:
            ext = os.path.splitext(file)[1] or 'no_ext'
            file_types[ext] = file_types.get(ext, 0) + 1
            
        print(f"\n📊 文件类型分布:")
        for ext, count in sorted(file_types.items()):
            print(f"   {ext}: {count} 个")
            
        return all_files
        
    def create_directory_structure(self):
        """创建目录结构"""
        print(f"\n📁 创建清理后的目录结构...")
        
        directories = [
            'test/matrix_tests',
            'tools/optimization', 
            'tools/benchmarks',
            'scripts/compilation',
            'scripts/testing',
            'results/optimization',
            'results/benchmarks',
            'temp/optimization'
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ {dir_path}")
            
    def move_files_by_rules(self):
        """按规则移动文件"""
        print(f"\n🚚 按规则整理文件...")
        
        moved_files = []
        
        # 具体的文件移动规则
        specific_moves = {
            # 优化工具
            'tools/optimization/': [
                'optimization_strategy.py',
                'large_matrix_optimization.py', 
                'optimization_demonstration.py',
                'analyze_progressive.py'
            ],
            # 基准测试
            'tools/benchmarks/': [
                'comprehensive_benchmark.py'
            ],
            # 测试脚本
            'scripts/testing/': [
                'progressive_performance_test.sh'
            ],
            # 编译脚本  
            'scripts/compilation/': [
                'optimize_compile.sh',
                'test_cf_convert.sh',
                'test_complete_convert.sh',
                'test_end_to_end.sh'
            ],
            # 测试文件
            'test/matrix_tests/': [
                'test_fix_compilation.bs',
                'test_large_matrix.bs',
                'test_4x4_matrix.bs',
                'test_8x8_matrix.bs', 
                'test_16x16_matrix.bs'
            ],
            # 结果文件
            'results/optimization/': [
                'optimization_report.json',
                'optimization_roadmap.json',
                'optimization_demonstration_*.json',
                'comprehensive_benchmark_*.json',
                'fusion_result.json'
            ],
            # 日志文件
            'temp/optimization/': [
                'large_build.log'
            ]
        }
        
        for target_dir, files in specific_moves.items():
            for file_pattern in files:
                matching_files = glob.glob(file_pattern)
                for file in matching_files:
                    if os.path.exists(file):
                        target_path = os.path.join(target_dir, os.path.basename(file))
                        try:
                            shutil.move(file, target_path)
                            moved_files.append((file, target_path))
                            print(f"   📁 {file} → {target_path}")
                        except Exception as e:
                            print(f"   ❌ 移动失败 {file}: {e}")
                            
        return moved_files
        
    def create_directory_readmes(self):
        """为每个目录创建README"""
        print(f"\n📝 创建目录说明文件...")
        
        readmes = {
            'tools/optimization/README.md': """# 🚀 优化工具

此目录包含Boas语言性能优化相关的工具和脚本。

## 文件说明
- `optimization_strategy.py`: 主优化策略分析器
- `large_matrix_optimization.py`: 大矩阵优化专用工具
- `optimization_demonstration.py`: 优化效果演示工具
- `analyze_progressive.py`: 渐进式测试结果分析器
""",
            'tools/benchmarks/README.md': """# 📊 基准测试工具

此目录包含性能基准测试相关的工具。

## 文件说明
- `comprehensive_benchmark.py`: 综合性能基准测试器
""",
            'scripts/compilation/README.md': """# 🔧 编译脚本

此目录包含Boas语言编译相关的脚本。

## 文件说明
- `optimize_compile.sh`: 优化编译脚本
- `test_cf_convert.sh`: CF dialect转换测试
- `test_complete_convert.sh`: 完整转换测试
- `test_end_to_end.sh`: 端到端测试脚本
""",
            'scripts/testing/README.md': """# 🧪 测试脚本

此目录包含各种测试脚本。

## 文件说明
- `progressive_performance_test.sh`: 渐进式性能测试
""",
            'test/matrix_tests/README.md': """# 🔢 矩阵测试

此目录包含各种规模的矩阵乘法测试文件。

## 文件说明
- `test_fix_compilation.bs`: 基础编译测试(2x2)
- `test_4x4_matrix.bs`: 4x4矩阵测试
- `test_8x8_matrix.bs`: 8x8矩阵测试
- `test_16x16_matrix.bs`: 16x16矩阵测试
- `test_large_matrix.bs`: 大矩阵测试(128x128)
""",
            'results/optimization/README.md': """# 📈 优化结果

此目录包含优化过程的结果和报告文件。

## 文件说明
- `optimization_report.json`: 优化报告
- `optimization_roadmap.json`: 优化路线图
- `comprehensive_benchmark_*.json`: 基准测试结果
- `optimization_demonstration_*.json`: 优化演示结果
""",
            'temp/optimization/README.md': """# 🗂️ 临时文件

此目录包含优化过程中产生的临时文件和日志。

**注意**: 此目录的文件可以定期清理。
"""
        }
        
        for file_path, content in readmes.items():
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"   📝 {file_path}")
            
    def create_cleanup_summary(self, moved_files):
        """创建清理总结"""
        summary = {
            'cleanup_date': datetime.now().isoformat(),
            'files_moved': len(moved_files),
            'file_moves': [{'from': src, 'to': dst} for src, dst in moved_files],
            'new_structure': {
                'tools/': '优化和基准测试工具',
                'scripts/': '编译和测试脚本',
                'test/': '测试文件',
                'results/': '结果和报告',
                'temp/': '临时文件'
            }
        }
        
        with open('docs/PROJECT_CLEANUP_SUMMARY.md', 'w') as f:
            f.write(f"""# 🧹 项目清理总结

**清理日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 清理统计
- **移动文件数**: {len(moved_files)}
- **新增目录**: 8个
- **创建README**: 7个

## 📁 新的目录结构
```
Boas-linux/
├── tools/
│   ├── optimization/     # 优化工具
│   └── benchmarks/       # 基准测试工具
├── scripts/
│   ├── compilation/      # 编译脚本
│   └── testing/          # 测试脚本
├── test/
│   └── matrix_tests/     # 矩阵测试文件
├── results/
│   └── optimization/     # 优化结果
└── temp/
    └── optimization/     # 临时文件
```

## 🎯 清理效果
- ✅ 根目录从21个文件减少到核心文件
- ✅ 文件按功能分类整理
- ✅ 每个目录都有说明文档
- ✅ 便于后续维护和开发

## 📋 移动的文件
""")
            
            for src, dst in moved_files:
                f.write(f"- `{src}` → `{dst}`\n")
                
        print(f"\n📋 清理总结已保存: docs/PROJECT_CLEANUP_SUMMARY.md")
        return summary
        
    def verify_cleanup(self):
        """验证清理结果"""
        print(f"\n✅ 验证清理结果...")
        
        # 检查根目录剩余文件
        remaining_files = []
        for pattern in ['*.py', '*.sh', '*.json', '*.bs', '*.log']:
            remaining_files.extend(glob.glob(pattern))
            
        print(f"📁 根目录剩余文件: {len(remaining_files)}")
        for file in remaining_files:
            print(f"   📄 {file}")
            
        # 检查新目录结构
        new_dirs = [
            'tools/optimization',
            'tools/benchmarks', 
            'scripts/compilation',
            'scripts/testing',
            'test/matrix_tests',
            'results/optimization',
            'temp/optimization'
        ]
        
        print(f"\n📁 新目录结构验证:")
        for dir_path in new_dirs:
            if os.path.exists(dir_path):
                file_count = len(os.listdir(dir_path))
                print(f"   ✅ {dir_path}: {file_count} 个文件")
            else:
                print(f"   ❌ {dir_path}: 不存在")

def main():
    """主清理流程"""
    print("🧹 Boas项目文件清理工具")
    print("=" * 50)
    
    cleaner = ProjectCleaner()
    
    # 1. 分析当前文件
    all_files = cleaner.analyze_current_files()
    
    # 2. 创建目录结构
    cleaner.create_directory_structure()
    
    # 3. 移动文件
    moved_files = cleaner.move_files_by_rules()
    
    # 4. 创建README文件
    cleaner.create_directory_readmes()
    
    # 5. 创建清理总结
    summary = cleaner.create_cleanup_summary(moved_files)
    
    # 6. 验证结果
    cleaner.verify_cleanup()
    
    print(f"\n🎉 项目清理完成!")
    print(f"📊 移动了 {len(moved_files)} 个文件")
    print(f"📁 根目录现在更加整洁")
    print(f"📝 查看详细报告: docs/PROJECT_CLEANUP_SUMMARY.md")

if __name__ == "__main__":
    main()
