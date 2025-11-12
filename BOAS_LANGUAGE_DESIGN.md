# Boas Programming Language Design

**Version**: 0.2.0-alpha
**Status**: Language Design Phase
**Last Updated**: 2025-11-13

---

## 🎯 Language Vision

**Boas** is a modern, high-performance programming language designed for scientific computing, machine learning, and systems programming with seamless hardware acceleration.

### Design Philosophy

```
Python Simplicity + C++ Performance + Rust Safety + Go Concurrency
```

**Core Principles**:
- **Readable**: Python-style syntax for clarity
- **Fast**: C++ level performance through MLIR optimization
- **Safe**: Rust-inspired memory management
- **Concurrent**: Go-style goroutines and channels
- **Accelerated**: First-class GPU/NPU support

---

## 🌟 Language Features

### 1. Python-Style Syntax
```python
# Function definition
def matrix_multiply(a: Matrix[f32], b: Matrix[f32]) -> Matrix[f32]:
    # Type inference
    result = zeros_like(a @ b)

    # Simple loop syntax
    for i in range(a.rows):
        for j in range(b.cols):
            for k in range(a.cols):
                result[i, j] += a[i, k] * b[k, j]

    return result

# Main function
def main():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[5.0, 6.0], [7.0, 8.0]])
    c = matrix_multiply(a, b)
    print(c)
```

### 2. Strong Type System with Inference
```python
# Explicit types
def add(x: i32, y: i32) -> i32:
    return x + y

# Type inference
def multiply(x, y):
    return x * y  # Types inferred from usage

# Generic functions
def dot[T](a: Vector[T], b: Vector[T]) -> T:
    return sum(a[i] * b[i] for i in range(len(a)))

# Tensor types with shape information
def transform(x: Tensor[f32, [128, 256]]) -> Tensor[f32, [128, 512]]:
    w = Tensor[f32, [256, 512]].random()
    return x @ w
```

### 3. Rust-Inspired Memory Management

```python
# Ownership and borrowing
def process_data(data: owned Vector[f32]) -> Vector[f32]:
    # 'data' is moved, caller loses access
    result = data.transform()
    return result  # Ownership transferred to caller

def read_data(data: ref Vector[f32]) -> f32:
    # 'data' is borrowed immutably
    return data.sum()

def modify_data(data: mut Vector[f32]):
    # 'data' is borrowed mutably
    data[0] = 42.0

# Lifetime annotations
def get_slice[lifetime 'a](data: ref 'a Vector[f32], start: i32, end: i32) -> ref 'a [f32]:
    return data[start:end]

# Smart pointers
def create_shared() -> Rc[Matrix[f32]]:
    return Rc.new(Matrix.identity(100))
```

### 4. Go-Style Concurrency

```python
# Goroutines (lightweight threads)
async def compute_heavy(x: i32) -> i32:
    # Expensive computation
    return x * x

def main():
    # Launch concurrent tasks
    task1 = spawn compute_heavy(10)
    task2 = spawn compute_heavy(20)
    task3 = spawn compute_heavy(30)

    # Wait for results
    r1 = await task1
    r2 = await task2
    r3 = await task3

    print(r1 + r2 + r3)

# Channels for communication
def producer(ch: Channel[i32]):
    for i in range(100):
        ch.send(i)
    ch.close()

def consumer(ch: Channel[i32]):
    while value := ch.receive():
        print(value)

def main():
    ch = Channel[i32](buffer_size=10)
    spawn producer(ch)
    spawn consumer(ch)
```

### 5. Hardware Acceleration (GPU/NPU)

```python
# Device annotations
@device(npu)
def matmul_npu(a: Tensor[f32], b: Tensor[f32]) -> Tensor[f32]:
    return a @ b

@device(gpu)
def process_batch(data: Tensor[f32, [batch, features]]) -> Tensor[f32]:
    # Automatically runs on GPU
    return relu(data @ weights + bias)

# Explicit device management
def train_model():
    with device.NPU(0) as npu:
        model = Model().to(npu)
        for batch in dataloader:
            output = model(batch)  # Runs on NPU
            loss = compute_loss(output)
            loss.backward()

# Multi-device execution
@parallel(devices=[NPU(0), NPU(1), GPU(0)])
def distributed_matmul(a: Tensor[f32], b: Tensor[f32]) -> Tensor[f32]:
    # Automatically distributed across devices
    return a @ b
```

### 6. Advanced Features

#### a. Pattern Matching
```python
def describe(value):
    match value:
        case 0:
            print("Zero")
        case x if x > 0:
            print("Positive")
        case x if x < 0:
            print("Negative")
        case Vector(x, y, z):
            print(f"3D vector: {x}, {y}, {z}")
        case Matrix(rows, cols) if rows == cols:
            print(f"Square matrix: {rows}x{cols}")
        case _:
            print("Unknown")
```

#### b. Compile-Time Computation
```python
# Compile-time constants
const N: i32 = 1024
const BLOCK_SIZE: i32 = N / 16

# Compile-time functions
@compile_time
def fibonacci(n: i32) -> i32:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Use in type system
def process(data: Tensor[f32, [fibonacci(10), fibonacci(10)]]):
    pass  # Tensor shape computed at compile time
```

#### c. Operator Overloading
```python
class Matrix[T]:
    def __init__(self, data: List[List[T]]):
        self.data = data

    def __add__(self, other: Matrix[T]) -> Matrix[T]:
        # Element-wise addition
        return Matrix([[a + b for a, b in zip(row1, row2)]
                       for row1, row2 in zip(self.data, other.data)])

    def __matmul__(self, other: Matrix[T]) -> Matrix[T]:
        # Matrix multiplication
        return matrix_multiply(self, other)

    def __getitem__(self, index: (i32, i32)) -> T:
        i, j = index
        return self.data[i][j]
```

#### d. Macros and Metaprogramming
```python
# Simple macro
@macro
def debug_print(expr):
    return quote:
        print(f"{unquote(expr)} = {unquote(expr)}")

# Usage
x = 42
debug_print(x)  # Expands to: print(f"x = {x}")

# Compile-time code generation
@generate
def create_matrix_types():
    for size in [2, 3, 4]:
        yield quote:
            class Matrix{size}D:
                shape = ({size}, {size})
```

---

## 🏗️ Technical Architecture

### MLIR Integration

```
Boas Source Code
    ↓ [Parser]
Boas AST
    ↓ [Semantic Analysis]
Typed Boas AST
    ↓ [Lowering]
Boas MLIR Dialect
    ↓ [Optimization Passes]
    ├─→ [CPU Path]     Linalg → Loops → LLVM IR → Machine Code
    ├─→ [GPU Path]     Linalg → GPU → CUDA/ROCm
    └─→ [NPU Path]     Linalg → HFusion → HIVM → NPU Binary
```

### Extended Boas Dialect Operations

```mlir
// Current: Matrix multiplication
%result = boas.matmul %lhs, %rhs : tensor<MxK> × tensor<KxN> → tensor<MxN>

// Planned extensions:
%result = boas.conv2d %input, %kernel : tensor<NxCxHxW> × tensor<OxCxKxK> → tensor<NxOxH'xW'>
%result = boas.relu %input : tensor<*xf32> → tensor<*xf32>
%result = boas.pool2d %input : tensor<NxCxHxW> → tensor<NxCxH'xW'>

// Control flow
boas.if %condition {
    boas.yield %true_value
} else {
    boas.yield %false_value
}

boas.for %i = %start to %end step %step {
    %val = boas.load %array[%i]
    boas.yield
}

// Memory operations
%ptr = boas.alloc : !boas.ptr<f32>
boas.store %value, %ptr : f32, !boas.ptr<f32>
%loaded = boas.load %ptr : !boas.ptr<f32>
boas.dealloc %ptr : !boas.ptr<f32>

// Device operations
%npu = boas.get_device "npu" : !boas.device
%tensor_npu = boas.to_device %tensor, %npu : tensor<*xf32> → tensor<*xf32>
%result = boas.execute_on %npu {
    %r = boas.matmul %a, %b
    boas.yield %r
}

// Async operations
%future = boas.async {
    %r = boas.heavy_compute %input
    boas.yield %r
}
%result = boas.await %future
```

---

## 📚 Standard Library

### Core Modules

```python
# math module
from boas.math import sin, cos, sqrt, pow, abs

# linalg module
from boas.linalg import (
    matmul, dot, cross, norm,
    inv, det, eig, svd,
    qr, cholesky
)

# tensor module
from boas.tensor import (
    Tensor, zeros, ones, randn,
    reshape, transpose, concatenate
)

# nn module (neural networks)
from boas.nn import (
    Linear, Conv2d, BatchNorm,
    ReLU, Softmax, Dropout,
    Module, Sequential
)

# device module
from boas.device import (
    Device, CPU, GPU, NPU,
    get_device_count, synchronize
)

# concurrent module
from boas.concurrent import (
    spawn, await, Channel,
    Mutex, RwLock, Atomic
)

# io module
from boas.io import (
    read_file, write_file,
    print, input, format
)
```

### Example: Neural Network

```python
from boas.nn import Module, Linear, ReLU
from boas.tensor import Tensor

class MLP(Module):
    def __init__(self, input_dim: i32, hidden_dim: i32, output_dim: i32):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor[f32]) -> Tensor[f32]:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@device(npu)
def train_step(model: MLP, batch: Tensor[f32], labels: Tensor[f32]) -> f32:
    # Forward pass
    output = model.forward(batch)

    # Compute loss
    loss = cross_entropy(output, labels)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.item()

def main():
    model = MLP(784, 256, 10).to(device.NPU(0))
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for batch, labels in dataloader:
            loss = train_step(model, batch, labels)
            print(f"Epoch {epoch}, Loss: {loss}")
```

---

## 🗺️ Implementation Roadmap

### Phase 1: Language Core (Months 1-3)
- [x] Basic matrix operations (Complete)
- [ ] Lexer and Parser
- [ ] Type system and inference
- [ ] Basic MLIR dialect extensions
- [ ] Control flow (if, for, while)
- [ ] Functions and closures

### Phase 2: Memory Management (Months 4-6)
- [ ] Ownership system
- [ ] Borrow checker
- [ ] Lifetime analysis
- [ ] Smart pointers (Rc, Arc, Box)
- [ ] RAII and destructors

### Phase 3: Concurrency (Months 7-9)
- [ ] Async/await syntax
- [ ] Lightweight threads (goroutines)
- [ ] Channels and message passing
- [ ] Synchronization primitives
- [ ] Work stealing scheduler

### Phase 4: Hardware Acceleration (Months 10-12)
- [x] NPU IR generation (Complete)
- [ ] NPU runtime execution (85% complete)
- [ ] GPU support (CUDA/ROCm)
- [ ] Multi-device orchestration
- [ ] Automatic kernel fusion

### Phase 5: Advanced Features (Months 13-18)
- [ ] Pattern matching
- [ ] Macros and metaprogramming
- [ ] Compile-time computation
- [ ] Generic programming
- [ ] Traits and interfaces

### Phase 6: Standard Library (Months 19-24)
- [ ] Math and linalg modules
- [ ] Neural network module
- [ ] I/O and file handling
- [ ] Data structures (Vector, HashMap, etc.)
- [ ] Package manager

---

## 📊 Comparison with Other Languages

| Feature | Boas | Python | C++ | Rust | Julia |
|---------|------|--------|-----|------|-------|
| **Syntax Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Memory Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Concurrency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hardware Accel** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Learning Curve** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🎓 Example Programs

### 1. Hello World
```python
def main():
    print("Hello, Boas!")
```

### 2. Fibonacci
```python
def fibonacci(n: i32) -> i32:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"fib({i}) = {fibonacci(i)}")
```

### 3. Concurrent Web Scraper
```python
from boas.net import http
from boas.concurrent import spawn, Channel

async def fetch_url(url: str) -> str:
    response = await http.get(url)
    return response.text

def main():
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]

    results = Channel[str](buffer_size=10)

    for url in urls:
        spawn {
            content = await fetch_url(url)
            results.send(content)
        }

    for _ in range(len(urls)):
        content = results.receive()
        print(f"Fetched {len(content)} bytes")
```

### 4. NPU-Accelerated ML Training
```python
from boas.nn import Module, Linear, ReLU, CrossEntropyLoss
from boas.optim import SGD
from boas.data import DataLoader
from boas.device import NPU

class Classifier(Module):
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

@device(npu)
def train_epoch(model, dataloader, optimizer, loss_fn):
    total_loss = 0.0

    for batch, labels in dataloader:
        # Forward
        output = model(batch)
        loss = loss_fn(output, labels)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # Setup
    device = NPU(0)
    model = Classifier().to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()

    train_data = DataLoader("mnist_train.csv", batch_size=64)

    # Training loop
    for epoch in range(10):
        loss = train_epoch(model, train_data, optimizer, loss_fn)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## 🔧 Development Tools

### 1. Compiler
```bash
# Compile to executable
boasc program.boas -o program

# Compile with optimization
boasc program.boas -O3 -o program

# Compile for specific target
boasc program.boas --target=npu -o program

# Show MLIR IR
boasc program.boas --emit-mlir

# Show LLVM IR
boasc program.boas --emit-llvm
```

### 2. REPL (Interactive)
```bash
$ boas
Boas 0.2.0 REPL
>>> x = [1, 2, 3, 4, 5]
>>> y = [i * i for i in x]
>>> print(y)
[1, 4, 9, 16, 25]
```

### 3. Package Manager
```bash
# Install package
boas install numpy

# Create new project
boas new my_project

# Run tests
boas test

# Build release
boas build --release
```

### 4. Documentation
```bash
# Generate docs
boas doc

# Run examples
boas run examples/matmul.boas
```

---

## 📖 Language Specification

### File Extension
- Source files: `.boas`
- Header files: `.boas.h`

### Encoding
- UTF-8 by default

### Keywords
```
def, class, return, if, else, elif, for, while, break, continue,
match, case, import, from, as, try, except, finally, raise,
async, await, spawn, yield, const, mut, ref, owned,
true, false, None, and, or, not, in, is
```

### Primitive Types
```
i8, i16, i32, i64, i128      # Signed integers
u8, u16, u32, u64, u128      # Unsigned integers
f16, f32, f64                # Floating point
bool                          # Boolean
str                           # String
char                          # Character
```

### Composite Types
```
List[T]                       # Dynamic array
Tuple[T1, T2, ...]           # Fixed-size tuple
Dict[K, V]                    # Hash map
Set[T]                        # Hash set
Option[T]                     # Optional value
Result[T, E]                  # Result with error
Tensor[T, Shape]             # Multi-dimensional array
Matrix[T]                     # 2D tensor
Vector[T]                     # 1D tensor
```

---

## 🚀 Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/TianTian-O1/Boas.git
cd Boas

# Build compiler
./build.sh

# Install
sudo make install
```

### First Program
```python
# hello.boas
def main():
    print("Hello, Boas!")
```

```bash
# Compile and run
boasc hello.boas -o hello
./hello
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas needing help**:
- Parser and frontend development
- MLIR pass optimization
- Standard library implementation
- Documentation and tutorials
- Testing and benchmarks

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 📞 Contact

- **GitHub**: https://github.com/TianTian-O1/Boas
- **Email**: 410771376@qq.com
- **Documentation**: https://boas-lang.org (coming soon)

---

**Status**: 🚧 Under Active Development

**Current Version**: 0.2.0-alpha
**Core Compiler**: 95% complete
**Language Features**: 10% complete
**Standard Library**: 5% complete

**Next Milestone**: Complete parser and basic type system (Q1 2026)
