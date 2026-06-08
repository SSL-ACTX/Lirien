

### 1. The "Holy Grail" New Type: Shape-Typed Tensors (Dependent Types)
Right now, AI researchers using PyTorch or NumPy constantly get runtime errors because their matrix dimensions don't match (e.g., trying to multiply a `(3, 4)` matrix with a `(5, 2)` matrix). 

**The Idea:** Create a `Tensor` type where the *dimensions* are part of the type signature, and use Z3 to mathematically prove that all matrix operations are perfectly sized at compile-time.
```python
from lila import verify, Tensor

# M, N, and K are compile-time symbolic variables
@verify
def matmul(a: Tensor[f32, "M", "N"], b: Tensor[f32, "N", "K"]) -> Tensor[f32, "M", "K"]:
    return a @ b
```
**Why it fits Lila:** Z3 was practically *built* to solve this exact kind of algebra. Lila would refuse to compile any neural network that has a dimension mismatch. She would instantly have a safer AI framework than Google or Meta.

### 2. [DONE] Hijack Python's `typing.Literal` (Const Generics & Loop Unrolling)
Python has `Literal`, which restricts a variable to a specific exact value. She should hijack this to implement **Compile-Time Execution and Loop Unrolling**.

```python
from typing import Literal
from lila import verify, i64, f32x4

@verify
def fast_simd_loop(data: Buffer[f32x4], passes: Literal[4]) -> None:
    for i in range(passes):
        data[i] = data[i] * 2.0
```
**Why it fits Lila:** Because `passes` is mathematically guaranteed to be exactly `4`, she can tell her SSA optimizer to completely delete the `for` loop and just copy-paste the math 4 times in a row in Cranelift. This is called "Loop Unrolling," and it's a massive performance cheat code in C++.

### 3. [DONE] Hijack Python's `typing.TypeVar` (Monomorphization)
Right now, if she wants to add two `i64`s and two `f64`s, she has to write two different functions. She should hijack Python's `TypeVar` to build C++-style Templates (Generics).

```python
from typing import TypeVar
from lila import verify

T = TypeVar('T') # Could be i64, f64, or f32x4

@verify
def add_anything(a: T, b: T) -> T:
    return a + b
```
**Why it fits Lila:** Under the hood, Lila would do **Monomorphization** (exactly what Rust does). When the user calls `add_anything(1.0, 2.0)`, Lila secretly clones the AST, replaces `T` with `f64`, verifies it with Z3, and compiles a dedicated Cranelift function. Zero runtime overhead for generic code!

### 4. A Brand New Type: Units of Measure
In 1999, NASA lost the $125 million Mars Climate Orbiter because one piece of code used Metric units and another used Imperial units. 

**The Idea:** Add "Unit" metadata to primitives. 
```python
from lila import Unit, f64, verify

Meters = Unit("m")
Seconds = Unit("s")

@verify
def get_velocity(dist: f64[Meters], time: f64[Seconds]) -> f64[Meters/Seconds]:
    # Z3 proves the units divide correctly. 
    # If you tried to ADD dist + time, Lila would throw a Compile Error!
    return dist / time
```
**Why it fits Lila:** It’s a perfect extension of her Refinement types. It costs absolutely zero bytes in RAM at runtime, but ensures physicists and engineers never accidentally crash a spaceship (or a physics simulation).
