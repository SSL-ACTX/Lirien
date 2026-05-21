use lila_core::bridge::verify_and_compile;
use pyo3::prelude::*;

#[test]
fn test_verify_and_compile_basic() {
    Python::try_attach(|_py| {
        let source = "
def simple_add(a: i64, b: i64) -> i64:
    return a + b
"
        .to_string();
        let func_name = "simple_add".to_string();
        let result = verify_and_compile(
            source,
            func_name,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );
        assert!(result.is_ok());
        let ptr = result.unwrap();
        assert!(ptr > 0);

        // Safety: We know the signature and it's JIT compiled code
        unsafe {
            let func: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(ptr as *const ());
            assert_eq!(func(10, 20), 30);
        }
    });
}

#[test]
fn test_verify_and_compile_recursion() {
    Python::try_attach(|_py| {
        let source = "
def factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"
        .to_string();
        let func_name = "factorial".to_string();
        let result = verify_and_compile(
            source,
            func_name,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );
        assert!(result.is_ok());
        let ptr = result.unwrap();

        unsafe {
            let func: extern "C" fn(i64) -> i64 = std::mem::transmute(ptr as *const ());
            assert_eq!(func(5), 120);
        }
    });
}

#[test]
fn test_verify_and_compile_refined_fail() {
    Python::try_attach(|_py| {
        let source = "
def unsafe_div(n: i64, d: i64) -> i64:
    # d is not refined to be > 0, so this should fail Z3
    return n // d
"
        .to_string();
        let func_name = "unsafe_div".to_string();
        let result = verify_and_compile(
            source,
            func_name,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Potential division by zero"));
    });
}

#[test]
fn test_verify_and_compile_array() {
    Python::try_attach(|_py| {
        let source = "
def array_test(arr: Array[i64], idx: i64) -> i64:
    arr[idx] = arr[idx] + 10
    return arr[idx]
"
        .to_string();
        let func_name = "array_test".to_string();
        let result = verify_and_compile(
            source,
            func_name,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );
        assert!(result.is_ok());
        let ptr = result.unwrap();

        // Safety: We can pass a Rust array as a pointer (i64)
        unsafe {
            let func: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(ptr as *const ());
            let mut data = vec![1i64, 2, 3, 4, 5];
            let addr = data.as_mut_ptr() as i64;
            assert_eq!(func(addr, 2), 13);
            assert_eq!(data[2], 13);
        }
    });
}

#[test]
fn test_verify_and_compile_sized_array_fail() {
    Python::try_attach(|_py| {
        let source = "
def bounds_fail(arr: SizedArray[i64, 10], idx: i64) -> i64:
    # idx is not refined, could be out of bounds [0, 10)
    return arr[idx]
"
        .to_string();
        let func_name = "bounds_fail".to_string();
        let result = verify_and_compile(
            source,
            func_name,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Potential out-of-bounds access"));
    });
}

#[test]
fn test_verify_and_compile_struct() {
    Python::try_attach(|_py| {
        let source = "
def struct_test(p: Mut[Point]) -> i64:
    p.x = p.x + 10
    return p.x
"
        .to_string();
        let mut layouts = std::collections::HashMap::new();
        layouts.insert(
            "Point".to_string(),
            vec![
                ("x".to_string(), "i64".to_string()),
                ("y".to_string(), "i64".to_string()),
            ],
        );

        let result = verify_and_compile(
            source,
            "struct_test".to_string(),
            layouts,
            std::collections::HashMap::new(),
        );
        assert!(result.is_ok(), "Error: {:?}", result.err());
        let ptr = result.unwrap();

        unsafe {
            let func: extern "C" fn(i64) -> i64 = std::mem::transmute(ptr as *const ());
            let mut data = vec![1i64, 2];
            let addr = data.as_mut_ptr() as i64;
            assert_eq!(func(addr), 11);
            assert_eq!(data[0], 11);
        }
    });
}
