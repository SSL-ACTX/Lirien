# Lila Python Expansion Plan

## 1. `@struct` Methods (Object-Oriented JIT)
Expand the compiler to automatically verify and JIT-compile methods defined *inside* the `@struct` class. The compiler will automatically map `self` to an implicit `Mut[Self]` or `Ref[Self]`, allowing users to write `p.move(dx, dy)` natively.

## 2. Full Python Buffer Protocol Support
Implement robust `ctypes` bindings to fully support Python's native Buffer Protocol. This will allow Lila to seamlessly accept standard Python `bytearray`, `memoryview`, and `array.array` without zero-copy overhead.

## 3. Multiple Return Values (Tuples)
Expand the AST visitor and Cranelift backend to support tuple returns by mapping them to an anonymous struct layout in memory or utilizing Cranelift's multi-value returns.

## 4. Standard `math` Module Intrinsics
Add a set of mathematical intrinsics (e.g., `math.sqrt`, `math.sin`, `math.cos`, `math.pow`). The Python AST parser will detect these and lower them into native Cranelift float instructions, fully modeled in Z3 for verification.

## 5. Python `for ... in ...` Iterators over Buffers/Arrays
Polish the semantic sugar so that iterating over an Lila `SizedArray` or `Buffer` in Python translates automatically into a verified bounds-checked index loop in the SSA IR.
