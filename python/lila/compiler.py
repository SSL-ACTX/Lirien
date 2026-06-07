import inspect
import ast
import textwrap
import os
import ctypes
from typing import Callable, TypeVar, Any, Dict, List, Tuple
from . import lila_bridge
from .types import TYPE_MAP, Buffer, SizedArray, Closure, FnPointer

T = TypeVar("T", bound=Callable)


def configure_tracing(config: Dict[str, str]):
    """
    Configure granular tracing for Lila components.

    Example:
        configure_tracing({"liveness": "debug", "verify": "info"})
    """
    lila_bridge.configure_tracing(config)


def get_cpu_info() -> Dict[str, str]:
    """
    Get information about the host CPU architecture and enabled SIMD features.
    """
    return lila_bridge.get_cpu_info()


def parallel_for(range_obj: range, body_fn: Callable[[int], None]):
    """
    Statically verified parallel loop.
    """
    for i in range_obj:
        body_fn(i)


# Component names for tracing
LIVENESS = "liveness"
VERIFY = "verify"
Z3 = "verify::z3"
SSA = "ssa"
BACKEND = "backend"
BRIDGE = "bridge"
ALL = "all"


class VerificationError(Exception):
    """Raised when Lila formal verification or JIT compilation fails in strict mode."""

    pass


def format_verification_error(func_name: str, source: str, error: str) -> str:
    import re

    # Try to find offset in the error message
    match = re.search(r"at offset (\d+)", error)
    if match:
        offset = int(match.group(1))
        # Remove the offset info from the error message for cleaner display
        clean_error = error.replace(match.group(0), "").strip()

        # Find line and column from offset
        lines = source.splitlines()
        curr_offset = 0
        target_line_idx = 0
        target_col = 0
        for i, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            if curr_offset <= offset < curr_offset + line_len:
                target_line_idx = i
                target_col = offset - curr_offset
                break
            curr_offset += line_len

        # Format pretty error
        res = [f"Lila Verification Failed for '{func_name}': {clean_error}"]
        res.append(f"  at line {target_line_idx + 1}, col {target_col + 1}:")
        res.append("")

        # Context lines
        start_idx = max(0, target_line_idx - 1)
        end_idx = min(len(lines), target_line_idx + 2)
        for i in range(start_idx, end_idx):
            prefix = "> " if i == target_line_idx else "  "
            res.append(f"{prefix}{i + 1:4} | {lines[i]}")
            if i == target_line_idx:
                res.append("       | " + " " * target_col + "^")

        return "\n".join(res)

    return f"Lila Verification Failed for '{func_name}': {error}"


def _get_type_name(ty: Any) -> str:
    """Consistently convert a Python-side type to its Lila IR string representation."""
    if ty is None or ty is type(None):
        return "None"

    if isinstance(ty, (list, tuple)):
        return "(" + ", ".join(_get_type_name(t) for t in ty) + ")"

    # Handle Refined types
    if hasattr(ty, "base_type") and hasattr(ty, "predicate"):
        base_name = _get_type_name(ty.base_type)
        try:
            pred_src = inspect.getsource(ty.predicate).strip()
            if "lambda" in pred_src:
                start = pred_src.find("lambda")
                pred_src = pred_src[start:]
                if pred_src.endswith(","):
                    pred_src = pred_src[:-1]
                if pred_src.endswith("]"):
                    pred_src = pred_src[:-1]
            return f"Refined[{base_name}, {pred_src}]"
        except:
            return base_name

    # Handle Annotated types (Buffer, Box, etc.)
    if hasattr(ty, "__metadata__"):
        origin = getattr(ty, "__origin__", ty)
        origin_str = str(origin).lower()
        inner = ty.__metadata__[0]

        # Check for Tuple in Annotated
        if "tuple" in origin_str:
            if isinstance(inner, (list, tuple)):
                return "(" + ", ".join(_get_type_name(t) for t in inner) + ")"
            return f"({_get_type_name(inner)})"

        inner_name = _get_type_name(inner)
        if "buffer" in origin_str:
            return f"Buffer[{inner_name}]"
        if "box" in origin_str:
            return f"Box[{inner_name}]"
        if "sizedarray" in origin_str:
            # SizedArray metadata usually has (base_type, size)
            if isinstance(inner, tuple) and len(inner) == 2:
                return f"SizedArray[{_get_type_name(inner[0])}, {inner[1]}]"
        return inner_name

    # Handle standard Tuples
    origin = getattr(ty, "__origin__", None)
    if origin is tuple or origin is Tuple or "tuple" in str(ty).lower():
        args = getattr(ty, "__args__", [])
        if args:
            return "(" + ", ".join(_get_type_name(t) for t in args) + ")"
        return "(i64, i64)"  # Default

    # Handle Higher-Order types
    if (
        "fnpointer" in str(ty).lower()
        or "closure" in str(ty).lower()
        or "callable" in str(ty).lower()
    ):
        args = getattr(ty, "__args__", [])
        if len(args) == 2:
            arg_tys, ret_ty = args
            arg_str = "[" + ", ".join(_get_type_name(t) for t in arg_tys) + "]"
            return f"{'Closure' if 'closure' in str(ty).lower() else 'FnPointer'}[{arg_str}, {_get_type_name(ret_ty)}]"

    if hasattr(ty, "__name__"):
        return ty.__name__

    return str(ty).split(".")[-1].replace("'>", "").lower()


def _discover_types(
    func: Callable, initial_struct_layouts: Dict
) -> Tuple[Dict, Dict, Dict]:
    """Scan for struct layouts, enum layouts, and type aliases referenced in the function's scope."""
    struct_layouts = initial_struct_layouts.copy() if initial_struct_layouts else {}
    enum_layouts = {}
    type_aliases = {}

    # Combine globals and closure variables
    scope = func.__globals__.copy()
    try:
        # Also try to get caller's locals to find types defined in the same function
        f_tmp = inspect.currentframe()
        while f_tmp:
            if "lila/compiler.py" not in f_tmp.f_code.co_filename:
                scope.update(f_tmp.f_locals)
                break
            f_tmp = f_tmp.f_back

        closure_vars = inspect.getclosurevars(func)
        scope.update(closure_vars.nonlocals)
        scope.update(closure_vars.globals)
    except:
        pass

    for name, obj in scope.items():
        if getattr(obj, "__lila_struct__", False) and name not in struct_layouts:
            struct_layouts[name] = [
                (f_name, _get_type_name(f_ty)) for f_name, f_ty in obj.__lila_fields__
            ]
        elif getattr(obj, "__lila_enum__", False) and name not in enum_layouts:
            layout = []
            for v_name in getattr(obj, "__lila_variants__", []):
                v_ty = obj.__lila_variant_types__[v_name]
                v_ty_name = _get_type_name(v_ty)
                layout.append((v_name, v_ty_name))
                if (
                    v_ty is not None
                    and hasattr(v_ty, "__lila_fields__")
                    and v_ty_name not in struct_layouts
                ):
                    struct_layouts[v_ty_name] = [
                        (f_name, _get_type_name(f_ty))
                        for f_name, f_ty in v_ty.__lila_fields__
                    ]
            enum_layouts[name] = layout
        elif hasattr(obj, "base_type") and hasattr(obj, "predicate"):
            # It's likely a Refined type instance
            try:
                pred_src = inspect.getsource(obj.predicate).strip()
                if "lambda" in pred_src:
                    start = pred_src.find("lambda")
                    pred_src = pred_src[start:]
                    if pred_src.endswith(","):
                        pred_src = pred_src[:-1]
                    if pred_src.endswith("]"):
                        pred_src = pred_src[:-1]

                base_ty = obj.base_type

                if hasattr(base_ty, "__origin__") and hasattr(base_ty, "__metadata__"):
                    # Handle Annotated (e.g. Buffer[i64])
                    origin_name = getattr(
                        base_ty.__origin__, "__name__", str(base_ty.__origin__)
                    )
                    # Extract the first metadata (the item type)
                    item_ty = base_ty.__metadata__[0]
                    base_ty_name = f"{origin_name}[{_get_type_name(item_ty)}]"
                else:
                    base_ty_name = getattr(base_ty, "__name__", str(base_ty))

                type_aliases[name] = f"Refined[{base_ty_name}, {pred_src}]"
            except (TypeError, AttributeError, SyntaxError, OSError):
                pass
    return struct_layouts, enum_layouts, type_aliases


def _get_ctypes_type(ann_str: str) -> Any:
    """Map a type name string to a ctypes type."""
    # Sort keys by length descending to match 'f32x4' before 'f32'
    for name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
        if name in ann_str:
            return TYPE_MAP[name]
    return ctypes.c_int64


def _map_ctypes_arguments(
    sig: inspect.Signature, class_name: str = None
) -> Tuple[List[Any], List[Tuple]]:
    """Determine ctypes argument types and mapping info from function signature."""
    c_args = []
    arg_map = []  # Track which Python arg maps to which C args

    for param in sig.parameters.values():
        ann = param.annotation

        if param.name == "self" and ann == inspect.Parameter.empty and class_name:
            c_args.append(ctypes.c_void_p)
            arg_map.append(("pointer", len(c_args) - 1))
            continue

        # Unwrap Refined type if necessary
        actual_ann = getattr(ann, "base_type", ann)
        ann_str = str(actual_ann).lower()

        from typing import get_origin, Annotated

        origin = get_origin(actual_ann) or actual_ann
        is_buffer = (
            (
                origin is Annotated
                and getattr(actual_ann, "__metadata__", (None,))[0] == "buffer"
            )
            or (isinstance(origin, type) and issubclass(origin, Buffer))
            or "buffer" in ann_str
        )

        is_ptr_wrapper = False
        if isinstance(origin, type) and issubclass(
            origin, (SizedArray, Closure, FnPointer, Callable)
        ):
            is_ptr_wrapper = True
        if not is_ptr_wrapper and any(
            x in ann_str
            for x in [
                "sizedarray",
                "closure",
                "fnpointer",
                "callable",
                "f32x4",
                "i32x4",
                "f64x2",
                "i64x2",
            ]
        ):
            is_ptr_wrapper = True

        if is_buffer:
            # Buffer is a Fat Pointer (ptr, len)
            c_args.append(ctypes.c_void_p)
            c_args.append(ctypes.c_int64)

            # Determine item size for length calculation
            item_size = 8
            if hasattr(actual_ann, "__metadata__"):
                item_ty = actual_ann.__metadata__[0]
                if getattr(item_ty, "__lila_struct__", False):
                    item_size = ctypes.sizeof(item_ty.__lila_ctypes__)
                else:
                    item_ty_str = str(item_ty).lower()
                    for name, cty in TYPE_MAP.items():
                        if name in item_ty_str:
                            item_size = ctypes.sizeof(cty)
                            break
            else:
                for name, cty in TYPE_MAP.items():
                    if name in ann_str:
                        item_size = ctypes.sizeof(cty)
                        break
            arg_map.append(("buffer", len(c_args) - 2, item_size))
        elif (
            is_ptr_wrapper
            or getattr(actual_ann, "__lila_struct__", False)
            or getattr(actual_ann, "__lila_enum__", False)
        ):
            c_args.append(ctypes.c_void_p)
            arg_map.append(("pointer", len(c_args) - 1))
        else:
            c_args.append(_get_ctypes_type(ann_str))
            arg_map.append(("value", len(c_args) - 1))
    return c_args, arg_map


def _handle_pointer_return(
    ret_ann: Any, c_args: List[Any], arg_map: List[Any]
) -> Tuple[Any, Any, List[Any], List[Any]]:
    """Handle return-by-pointer for tuples and SIMD types, adjusting argument mapping."""
    ret_ann_str = str(ret_ann).lower()

    # Detect if we need return-by-pointer (SRet style)
    is_tuple = "tuple" in ret_ann_str
    is_simd = any(x in ret_ann_str for x in ["f32x4", "i32x4", "f64x2", "i64x2"])

    if not (is_tuple or is_simd):
        return False, None, c_args, arg_map

    try:
        from .types import i64

        ResultStruct = None
        if is_tuple:
            # Try to extract inner types for Tuple
            if hasattr(ret_ann, "__args__"):
                tuple_types = ret_ann.__args__
            else:
                tuple_types = [i64, i64]  # Default

            tuple_fields = []
            for i, t in enumerate(tuple_types):
                t_str = str(t).lower()
                c_ty = ctypes.c_int64
                for name, cty in TYPE_MAP.items():
                    if name in t_str:
                        c_ty = cty
                        break
                tuple_fields.append((f"f{i}", c_ty))

            class TupleReturn(ctypes.Structure):
                _fields_ = tuple_fields

            ResultStruct = TupleReturn
        else:
            # SIMD Return - Match the specific vector type exactly if possible
            # Sort keys by length descending to match 'f32x4' before 'f32'
            for name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
                if name in ret_ann_str:
                    ResultStruct = TYPE_MAP[name]
                    break

        if ResultStruct is None:
            return False, None, c_args, arg_map

        new_c_args = [ctypes.POINTER(ResultStruct)] + c_args
        new_arg_map = []
        for info in arg_map:
            # Adjust arg_map indices because we inserted a pointer at index 0
            new_arg_map.append((info[0], info[1] + 1) + info[2:])

        return True, ResultStruct, new_c_args, new_arg_map
    except Exception as e:
        print(f"[Lila Warning] Failed to handle pointer return for {ret_ann}: {e}")
        return False, None, c_args, arg_map


def _create_jit_wrapper(
    code_ptr: Any, arg_types: List[Any], ret_type: Any, is_closure: bool = False
) -> Callable:
    """Create a ctypes wrapper for a JIT-compiled function pointer, supporting recursion for higher-order functions."""
    c_args = []
    arg_map = []
    if is_closure:
        c_args.append(ctypes.c_void_p)  # ctx_ptr

    for i, arg_ty in enumerate(arg_types):
        arg_ty_str = str(arg_ty).lower()
        if "buffer" in arg_ty_str:
            c_args.append(ctypes.c_void_p)
            c_args.append(ctypes.c_int64)
            arg_map.append(("buffer", len(c_args) - 2, 8))  # Simplified
        elif any(
            x in arg_ty_str
            for x in [
                "sizedarray",
                "fnpointer",
                "callable",
                "closure",
                "f32x4",
                "i32x4",
                "f64x2",
                "i64x2",
            ]
        ):
            c_args.append(ctypes.c_void_p)
            arg_map.append(("pointer", len(c_args) - 1))
        else:
            c_args.append(_get_ctypes_type(arg_ty_str))
            arg_map.append(("value", len(c_args) - 1))

    is_ptr_return, TupleReturn, c_args, arg_map = _handle_pointer_return(
        ret_type, c_args, arg_map
    )

    if is_ptr_return:
        c_ret = None
        if "tuple" in str(ret_type).lower():
            if hasattr(ret_type, "__args__"):
                tuple_types = ret_type.__args__
            else:
                from .types import i64

                tuple_types = [i64, i64]
    else:
        ret_ty_str = str(ret_type).lower()
        if "none" in ret_ty_str or ret_type is None:
            c_ret = None
        else:
            c_ret = _get_ctypes_type(ret_ty_str)

    # If it's a closure, the code_ptr passed here is the closure_ptr.
    # We need to load the actual function address from closure_ptr[0].
    actual_fn_ptr = code_ptr
    if is_closure:
        actual_fn_ptr = ctypes.cast(code_ptr, ctypes.POINTER(ctypes.c_void_p))[0]

    c_func = ctypes.CFUNCTYPE(c_ret, *c_args)(actual_fn_ptr)

    from .types import FnPointer, Closure, i64

    def jit_call(*args):
        processed_args, ret_struct = _prepare_runtime_args(
            args, arg_map, c_args, is_ptr_return, TupleReturn
        )
        if is_closure:
            # If it's a pointer return, TupleReturn* is at index 0, so ctx_ptr is at index 1.
            # Otherwise, ctx_ptr is at index 0.
            insert_idx = 1 if is_ptr_return else 0
            processed_args.insert(insert_idx, code_ptr)

        res = c_func(*processed_args)

        if is_ptr_return:
            if "tuple" in str(ret_type).lower():
                # Convert ctypes structure back to Python tuple
                return tuple(
                    getattr(ret_struct, f"f{i}") for i in range(len(tuple_types))
                )
            else:
                # SIMD Return
                return ret_struct

        # Recursively wrap if the return type is another function
        ret_ty_str = str(ret_type).lower()
        if (
            "fnpointer" in ret_ty_str
            or "closure" in ret_ty_str
            or "callable" in ret_ty_str
            or isinstance(ret_type, FnPointer)
        ):
            is_cls = "closure" in ret_ty_str or isinstance(ret_type, Closure)

            # Extract arg_types and ret_type
            inner_arg_types = [i64, i64]
            inner_ret_type = i64
            if isinstance(ret_type, FnPointer):
                inner_arg_types = ret_type.arg_types
                inner_ret_type = ret_type.ret_type
            elif hasattr(ret_type, "__metadata__"):
                params = ret_type.__metadata__[0]
                if isinstance(params, tuple) and len(params) == 2:
                    inner_arg_types, inner_ret_type = params
            elif hasattr(ret_type, "__args__"):
                params = ret_type.__args__
                if len(params) == 2:
                    inner_arg_types, inner_ret_type = params

            return _create_jit_wrapper(
                res,
                inner_arg_types,
                inner_ret_type,
                is_closure=is_cls,
            )

        return res

    jit_call.__lila_ptr__ = code_ptr
    return jit_call


def _prepare_runtime_args(
    args: Tuple,
    arg_map: List[Any],
    c_args: List[Any],
    is_tuple_return: bool,
    TupleReturn: Any,
) -> Tuple[List[Any], Any]:
    """Map Python arguments to ctypes arguments based on the provided mapping and return type."""
    processed_args = []
    ret_struct = None
    if is_tuple_return:
        ret_struct = TupleReturn()
        processed_args.append(ctypes.byref(ret_struct))

    for i, arg_info in enumerate(arg_map):
        arg_type = arg_info[0]
        c_idx = arg_info[1]
        arg = args[i]
        if arg_type == "buffer":
            item_size = arg_info[2]
            # Support NumPy and Python Buffer Protocol
            if hasattr(arg, "ctypes"):
                processed_args.append(ctypes.c_void_p(arg.ctypes.data))
                processed_args.append(ctypes.c_int64(arg.size))
            elif hasattr(arg, "__array_interface__"):
                processed_args.append(
                    ctypes.c_void_p(arg.__array_interface__["data"][0])
                )
                processed_args.append(ctypes.c_int64(arg.size))
            else:
                # Standard Python Buffer Protocol (bytearray, array.array, memoryview)
                try:
                    mv = memoryview(arg)
                    if not mv.contiguous:
                        raise ValueError("Buffer must be contiguous")

                    # Use ctypes to get a pointer to the buffer
                    ArrayType = ctypes.c_char * mv.nbytes
                    # from_buffer requires a writable buffer if it's not a read-only type
                    # but ctypes handles this.
                    c_buf = ArrayType.from_buffer(arg)
                    processed_args.append(ctypes.addressof(c_buf))
                    # Use the expected item_size to calculate length
                    processed_args.append(ctypes.c_int64(mv.nbytes // item_size))
                except Exception as e:
                    raise TypeError(
                        f"Argument {i} does not support the Buffer Protocol or is not contiguous: {e}"
                    )
        elif arg_type == "pointer":
            if hasattr(arg, "_ctypes_obj"):
                processed_args.append(ctypes.addressof(arg._ctypes_obj))
            elif isinstance(arg, ctypes.Structure):
                processed_args.append(ctypes.addressof(arg))
            elif hasattr(arg, "__lila_ptr__"):
                processed_args.append(ctypes.c_void_p(arg.__lila_ptr__))
            else:
                processed_args.append(arg)
        else:
            target_cty = c_args[c_idx]
            # Check if it's already a ctypes object (Structure, Array, or Pointer)
            if (
                isinstance(arg, target_cty)
                or hasattr(arg, "_type_")
                or hasattr(arg, "_fields_")
                or hasattr(arg, "_obj_")
            ):
                processed_args.append(arg)
            else:
                processed_args.append(target_cty(arg))
    return processed_args, ret_struct


def _setup_logging(log_level: str) -> Tuple[str, str]:
    """Override LILA_LOG level and return (log_level, old_log_level) for restoration."""
    if log_level:
        old_log = os.environ.get("LILA_LOG", "info")
        lila_bridge.set_log_level(log_level)
        os.environ["LILA_LOG"] = log_level
        return log_level, old_log
    return None, None


def _restore_logging(log_level: str, old_log: str):
    """Restore the original LILA_LOG level."""
    if log_level:
        lila_bridge.set_log_level(old_log)
        os.environ["LILA_LOG"] = old_log


def _prepare_source_and_name(
    func: Callable, class_name: str = None, method_name: str = None
) -> Tuple[str, str]:
    """Extract and dedent source code, handling method name overrides and AST adjustments."""
    source = textwrap.dedent(inspect.getsource(func))
    target_func_name = method_name if method_name else func.__name__

    if class_name:
        tree = ast.parse(source)
        func_def = tree.body[0]
        func_def.name = target_func_name

        if func_def.args.args and func_def.args.args[0].arg == "self":
            if not func_def.args.args[0].annotation:
                func_def.args.args[0].annotation = ast.Name(
                    id=class_name, ctx=ast.Load()
                )

        source = ast.unparse(tree)

    return source, target_func_name


def _check_runtime_refinements(sig: inspect.Signature, args: Tuple):
    """Validate runtime refinements for arguments."""
    for i, param in enumerate(sig.parameters.values()):
        if i < len(args):
            ann = param.annotation
            if hasattr(ann, "predicate") and ann.predicate:
                if not ann.predicate(args[i]):
                    raise ValueError(
                        f"Runtime Refinement Violation for argument '{param.name}': "
                        f"Value {args[i]} does not satisfy the predicate."
                    )


def _wrap_return_value(res: Any, ret_ann: Any) -> Any:
    """Wrap the JIT return value if it represents a higher-order function."""
    from .types import FnPointer, Closure, i64

    ret_ann_str = str(ret_ann).lower()
    if (
        "fnpointer" in ret_ann_str
        or "callable" in ret_ann_str
        or "closure" in ret_ann_str
        or isinstance(ret_ann, FnPointer)
    ):
        is_cls = "closure" in ret_ann_str or isinstance(ret_ann, Closure)

        # Extract arg_types and ret_type
        arg_types = [i64, i64]
        ret_type = i64

        if isinstance(ret_ann, FnPointer):
            arg_types = ret_ann.arg_types
            ret_type = ret_ann.ret_type
        elif hasattr(ret_ann, "__metadata__"):
            params = ret_ann.__metadata__[0]
            if isinstance(params, tuple) and len(params) == 2:
                arg_types, ret_type = params
        elif hasattr(ret_ann, "__args__"):
            # Fallback for some typing constructs
            params = ret_ann.__args__
            if len(params) == 2:
                arg_types, ret_type = params

        return _create_jit_wrapper(res, arg_types, ret_type, is_closure=is_cls)
    return res


def _get_ctypes_return_type(ret_ann: Any) -> Any:
    """Determine the ctypes return type from the annotation."""
    # Unwrap Refined type if necessary
    actual_ann = getattr(ret_ann, "base_type", ret_ann)
    ret_ann_str = str(actual_ann).lower()

    if (
        actual_ann is None
        or actual_ann is inspect.Signature.empty
        or "none" in ret_ann_str
    ):
        return None
    return _get_ctypes_type(ret_ann_str)


def _create_wrapper(
    func: Callable,
    code_ptr: int,
    c_args: List[Any],
    arg_map: List[Any],
    sig: inspect.Signature,
    is_ptr_return: bool,
    TupleReturn: Any,
    tuple_types: List[Any],
):
    """Generate the final Python wrapper that handles runtime checks and interop."""
    c_ret = None if is_ptr_return else _get_ctypes_return_type(sig.return_annotation)
    c_func = ctypes.CFUNCTYPE(c_ret, *c_args)(code_ptr)

    def wrapper(*args):
        _check_runtime_refinements(sig, args)

        processed_args, ret_struct = _prepare_runtime_args(
            args, arg_map, c_args, is_ptr_return, TupleReturn
        )

        res = c_func(*processed_args)

        if is_ptr_return:
            if "tuple" in str(sig.return_annotation).lower():
                return tuple(
                    getattr(ret_struct, f"f{i}") for i in range(len(tuple_types))
                )
            else:
                # SIMD Return
                return ret_struct

        return _wrap_return_value(res, sig.return_annotation)

    print(f"[Lila] JIT compiled '{func.__name__}' successfully.")
    wrapper.__lila_jit__ = True
    wrapper.__lila_ptr__ = code_ptr
    return wrapper


def verify(
    strict: bool = True,
    log_level: str = None,
    _struct_layouts: dict = None,
    _class_name: str = None,
    _method_name: str = None,
) -> Callable:
    """
    Decorator to trigger formal verification and JIT compilation.

    :param strict: If True, raises VerificationError on failure. If False, falls back to Python.
    :param log_level: Override LILA_LOG level (e.g., 'info', 'debug', 'warn').
    """

    # Handle the case where the decorator is used without parentheses: @verify
    if callable(strict) and log_level is None and _struct_layouts is None:
        func = strict
        # Re-call verify with defaults
        return verify(strict=True)(func)

    def decorator(func: T) -> T:
        log_lvl, old_log = _setup_logging(log_level)
        source, target_func_name = _prepare_source_and_name(
            func, _class_name, _method_name
        )

        try:
            struct_layouts, enum_layouts, type_aliases = _discover_types(
                func, _struct_layouts
            )

            try:
                code_ptr = lila_bridge.verify_and_compile(
                    source, target_func_name, struct_layouts, enum_layouts, type_aliases
                )
            finally:
                _restore_logging(log_lvl, old_log)

            sig = inspect.signature(func)
            c_args, arg_map = _map_ctypes_arguments(sig, _class_name)
            is_ptr_return, TupleReturn, c_args, arg_map = _handle_pointer_return(
                sig.return_annotation, c_args, arg_map
            )

            tuple_types = []
            if is_ptr_return:
                if "tuple" in str(sig.return_annotation).lower():
                    if hasattr(sig.return_annotation, "__args__"):
                        tuple_types = sig.return_annotation.__args__
                    else:
                        from .types import i64

                        tuple_types = [i64, i64]

            return _create_wrapper(
                func,
                code_ptr,
                c_args,
                arg_map,
                sig,
                is_ptr_return,
                TupleReturn,
                tuple_types,
            )
        except Exception as e:
            error_msg = format_verification_error(func.__name__, source, str(e))
            if strict:
                raise VerificationError(error_msg) from e
            else:
                print(f"[Lila Warning] {error_msg}. Falling back to Python.")
                func.__lila_jit__ = False
                return func

    # Support both @verify and @verify(strict=False)
    if callable(strict):
        f = strict
        strict = True
        return decorator(f)
    return decorator
