import unittest
import subprocess
import os

class ListTest(unittest.TestCase):
        # 编译目录
        
    def test_list_indexing(self):
        
        # 运行编译后的测试程序
        result = subprocess.run(
            [f"../build/tests/list_test", "basic.bs"],
            capture_output=True,
            text=True
        )
        
        # 打印输出以便调试
        print("\nTest output:")
        print(result.stdout)

if __name__ == '__main__':
    unittest.main()