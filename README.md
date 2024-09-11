为了能够通过一个命令直接将编写的源代码编译为二进制文件，完整的编译器流程可以集成到一个Python脚本中。我们将通过以下步骤实现这一目标：

1. **整合编译器前端**：词法分析、语法解析和AST生成。
2. **整合MLIR生成和优化**：通过MLIR生成中间表示并进行优化。
3. **整合LLVM代码生成和链接**：使用MLIR和LLVM工具链生成二进制文件。

### 最终目标：
你编写的程序会经过解析、MLIR生成、优化、LLVM后端处理，最终生成二进制文件，可以通过命令行直接运行，如下：
```bash
python my_compiler.py source_code.txt -o output_binary
```

### 1. 编写完整的Python编译器脚本

```python
import os
import subprocess
import ply.lex as lex
import ply.yacc as yacc
import sys

# ========== 词法分析 ==========
tokens = ['NUMBER', 'PLUS', 'MINUS', 'MULT', 'DIV', 'LPAREN', 'RPAREN', 'ID', 'EQUAL']

t_PLUS = r'\+'
t_MINUS = r'-'
t_MULT = r'\*'
t_DIV = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_EQUAL = r'='

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    return t

t_ignore = ' \t'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# ========== 语法分析 ==========
precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'MULT', 'DIV'),
)

class BinOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class Number:
    def __init__(self, value):
        self.value = value

class Assign:
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr

def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression MULT expression
                  | expression DIV expression'''
    p[0] = BinOp(p[1], p[2], p[3])

def p_expression_number(p):
    'expression : NUMBER'
    p[0] = Number(p[1])

def p_assignment(p):
    'statement : ID EQUAL expression'
    p[0] = Assign(p[1], p[3])

def p_error(p):
    print(f"Syntax error at {p.value}")

parser = yacc.yacc()

# ========== MLIR生成器 ==========
class MLIRGenerator:
    def __init__(self):
        self.code = []

    def generate(self, node):
        if isinstance(node, Number):
            return f"{node.value} : i32"
        elif isinstance(node, BinOp):
            left = self.generate(node.left)
            right = self.generate(node.right)
            if node.op == '+':
                return f"addi {left}, {right} : i32"
            elif node.op == '-':
                return f"subi {left}, {right} : i32"
            elif node.op == '*':
                return f"muli {left}, {right} : i32"
            elif node.op == '/':
                return f"divi {left}, {right} : i32"
        elif isinstance(node, Assign):
            expr = self.generate(node.expr)
            self.code.append(f"%{node.var} = {expr}")

    def get_mlir(self):
        return '\n'.join(self.code)

# ========== 生成二进制文件 ==========
def generate_binary(mlir_code, output_file):
    with open('temp.mlir', 'w') as f:
        f.write(mlir_code)

    # 使用MLIR工具链生成LLVM IR
    subprocess.run(["mlir-opt", "temp.mlir", "-convert-llvm", "-o", "temp_llvm.mlir"], check=True)

    # 使用mlir-translate将MLIR转换为LLVM IR
    subprocess.run(["mlir-translate", "--mlir-to-llvmir", "temp_llvm.mlir", "-o", "temp.ll"], check=True)

    # 使用llc将LLVM IR转换为目标文件
    subprocess.run(["llc", "-filetype=obj", "temp.ll", "-o", "temp.o"], check=True)

    # 链接生成可执行文件
    subprocess.run(["gcc", "temp.o", "-o", output_file], check=True)

    # 清理临时文件
    os.remove('temp.mlir')
    os.remove('temp_llvm.mlir')
    os.remove('temp.ll')
    os.remove('temp.o')

# ========== 主编译器入口 ==========
def compile_source_code(source_code, output_file):
    # 解析输入
    ast = parser.parse(source_code)
    
    # 生成MLIR代码
    generator = MLIRGenerator()
    generator.generate(ast)
    mlir_code = generator.get_mlir()
    print(f"Generated MLIR code:\n{mlir_code}")
    
    # 调用工具链生成二进制
    generate_binary(mlir_code, output_file)
    print(f"Compilation complete! Binary saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python my_compiler.py <source_file> -o <output_binary>")
        sys.exit(1)

    source_file = sys.argv[1]
    output_file = sys.argv[3]

    with open(source_file, 'r') as f:
        source_code = f.read()

    compile_source_code(source_code, output_file)
```

### 2. 测试编译器

#### 编写测试源代码文件：
例如编写一个简单的源代码文件 `source_code.txt`：
```txt
a = 5 + 3 * 2
```

#### 执行编译命令：
```bash
python my_compiler.py source_code.txt -o output_binary
```

#### 检查输出：
```bash
./output_binary
```

此时你应该看到正确的输出，并能够执行编译后的二进制文件。

### 3. 流程回顾

1. **词法分析和语法解析**：通过Ply解析输入的源代码，生成AST。
2. **MLIR生成**：通过遍历AST生成MLIR中间表示。
3. **MLIR优化与LLVM后端**：使用MLIR工具链生成优化后的LLVM IR，并通过LLVM工具链生成二进制文件。
4. **执行生成的二进制文件**：可以通过`gcc`链接，并最终生成可执行的二进制文件。

这样，你的编译器可以集成在一个命令中完成整个编译流程，将源代码编译为二进制文件并直接执行。