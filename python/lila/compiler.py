import inspect
import ast
import textwrap
from typing import Callable, TypeVar
from . import lila_core

T = TypeVar("T", bound=Callable)

import ctypes


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
    import os

    # Handle the case where the decorator is used without parentheses: @verify
    if callable(strict) and log_level is None and _struct_layouts is None:
        func = strict
        # Re-call verify with defaults
        return verify(strict=True)(func)

    def decorator(func: T) -> T:
        # Phase 0: Setup Logging Override
        if log_level:
            old_log = os.environ.get("LILA_LOG", "info")
            lila_core.set_log_level(log_level)
            os.environ["LILA_LOG"] = log_level

        # Phase 1: AST Extraction
        try:
            source = textwrap.dedent(inspect.getsource(func))

            # Use the provided method name if available, otherwise fallback to function name
            target_func_name = _method_name if _method_name else func.__name__

            if _class_name:
                tree = ast.parse(source)
                func_def = tree.body[0]
                # Rename the function in the AST to match the registry name
                func_def.name = target_func_name

                if func_def.args.args and func_def.args.args[0].arg == "self":
                    if not func_def.args.args[0].annotation:
                        func_def.args.args[0].annotation = ast.Subscript(
                            value=ast.Name(id="Hand", ctx=ast.Load()),
                            slice=ast.Name(id=_class_name, ctx=ast.Load()),
                            ctx=ast.Load(),
                        )
                source = ast.unparse(tree)

            # Scan for struct layouts, enum layouts, and type aliases referenced in the function's scope
            struct_layouts = _struct_layouts.copy() if _struct_layouts else {}
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
                if (
                    getattr(obj, "__lila_struct__", False)
                    and name not in struct_layouts
                ):
                    struct_layouts[name] = obj.__lila_fields__
                elif getattr(obj, "__lila_enum__", False) and name not in enum_layouts:
                    layout = []
                    for v_name in getattr(obj, "__lila_variants__", []):
                        v_ty = obj.__lila_variant_types__[v_name]
                        v_ty_name = v_ty.__name__
                        layout.append((v_name, v_ty_name))
                        if v_ty_name not in struct_layouts:
                            struct_layouts[v_ty_name] = getattr(
                                v_ty, "__lila_fields__", []
                            )
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
                        if hasattr(base_ty, "__origin__") and hasattr(
                            base_ty, "__metadata__"
                        ):
                            # Handle Annotated (e.g. Buffer[i64])
                            origin_name = getattr(
                                base_ty.__origin__, "__name__", str(base_ty.__origin__)
                            )
                            # Extract the first metadata (the item type)
                            item_ty = base_ty.__metadata__[0]
                            base_ty_name = f"{origin_name}[{item_ty}]"
                        else:
                            base_ty_name = getattr(base_ty, "__name__", str(base_ty))

                        type_aliases[name] = f"Refined[{base_ty_name}, {pred_src}]"
                    except:
                        pass

            try:
                code_ptr = lila_core.verify_and_compile(
                    source, target_func_name, struct_layouts, enum_layouts, type_aliases
                )
            finally:
                if log_level:
                    lila_core.set_log_level(old_log)
                    os.environ["LILA_LOG"] = old_log

            # Create a ctypes function pointer
            from .types import TYPE_MAP

            sig = inspect.signature(func)

            c_args = []
            arg_map = []  # Track which Python arg maps to which C args

            for param in sig.parameters.values():
                ann = param.annotation

                if (
                    param.name == "self"
                    and ann == inspect.Parameter.empty
                    and _class_name
                ):
                    c_args.append(ctypes.c_void_p)
                    arg_map.append(("pointer", len(c_args) - 1))
                    continue

                # Unwrap Refined type if necessary
                if hasattr(ann, "base_type"):
                    actual_ann = ann.base_type
                else:
                    actual_ann = ann

                ann_str = str(actual_ann).lower()

                # Default to i64 if unknown
                c_ty = ctypes.c_int64

                if "buffer" in ann_str:
                    # Buffer is a Fat Pointer (ptr, len)
                    c_args.append(ctypes.c_void_p)
                    c_args.append(ctypes.c_int64)

                    # Determine item size for length calculation
                    item_size = 8
                    for name, cty in TYPE_MAP.items():
                        if name in ann_str:
                            item_size = ctypes.sizeof(cty)
                            break
                    arg_map.append(("buffer", len(c_args) - 2, item_size))
                elif (
                    any(x in ann_str for x in ["hand", "peek", "sizedarray", "closure"])
                    or getattr(actual_ann, "__lila_struct__", False)
                    or getattr(actual_ann, "__lila_enum__", False)
                    or "fnpointer" in ann_str
                    or "callable" in ann_str
                ):
                    c_args.append(ctypes.c_void_p)
                    arg_map.append(("pointer", len(c_args) - 1))
                else:
                    for name, cty in TYPE_MAP.items():
                        if name in ann_str:
                            c_ty = cty
                            break
                    c_args.append(c_ty)
                    arg_map.append(("value", len(c_args) - 1))

            ret_ann = sig.return_annotation
            ret_ann_str = str(ret_ann).lower()

            is_tuple_return = False
            if "tuple" in ret_ann_str:
                is_tuple_return = True
                # Generate a dynamic ctypes Structure for the tuple
                try:
                    # Try to extract inner types
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

                    c_ret = None  # Void return
                    # Pointer is the FIRST argument
                    c_args.insert(0, ctypes.POINTER(TupleReturn))
                    # Adjust arg_map indices
                    for i in range(len(arg_map)):
                        info = arg_map[i]
                        arg_map[i] = (info[0], info[1] + 1) + info[2:]
                except Exception as e:
                    print(f"[Lila Warning] Failed to parse Tuple return: {e}")
                    is_tuple_return = False

            if not is_tuple_return:
                if (
                    ret_ann is None
                    or ret_ann is inspect.Signature.empty
                    or ret_ann_str == "none"
                ):
                    c_ret = None
                else:
                    c_ret = ctypes.c_int64
                    for name, cty in TYPE_MAP.items():
                        if name in ret_ann_str:
                            c_ret = cty
                            break

            def get_ctypes_type(ann_str):
                c_ty = ctypes.c_int64
                for name, cty in TYPE_MAP.items():
                    if name in ann_str:
                        return cty
                return c_ty

            def create_jit_wrapper(code_ptr, arg_types, ret_type, is_closure=False):
                c_args = []
                if is_closure:
                    c_args.append(ctypes.c_void_p)  # ctx_ptr

                for arg_ty in arg_types:
                    arg_ty_str = str(arg_ty).lower()
                    if "buffer" in arg_ty_str:
                        c_args.append(ctypes.c_void_p)
                        c_args.append(ctypes.c_int64)
                    elif any(
                        x in arg_ty_str
                        for x in [
                            "hand",
                            "peek",
                            "sizedarray",
                            "fnpointer",
                            "callable",
                            "closure",
                        ]
                    ):
                        c_args.append(ctypes.c_void_p)
                    else:
                        c_args.append(get_ctypes_type(arg_ty_str))

                ret_ty_str = str(ret_type).lower()
                if "none" in ret_ty_str or ret_type is None:
                    c_ret = None
                elif "tuple" in ret_ty_str:
                    # TODO: Handle tuple return in nested wrappers
                    c_ret = ctypes.c_void_p
                else:
                    c_ret = get_ctypes_type(ret_ty_str)

                # If it's a closure, the code_ptr passed here is the closure_ptr.
                # We need to load the actual function address from closure_ptr[0].
                actual_fn_ptr = code_ptr
                if is_closure:
                    actual_fn_ptr = ctypes.cast(
                        code_ptr, ctypes.POINTER(ctypes.c_void_p)
                    )[0]

                c_func = ctypes.CFUNCTYPE(c_ret, *c_args)(actual_fn_ptr)

                def jit_call(*args):
                    processed_args = []
                    if is_closure:
                        processed_args.append(code_ptr)

                    arg_idx = 0
                    for arg_ty in arg_types:
                        arg = args[arg_idx]
                        arg_ty_str = str(arg_ty).lower()
                        if "buffer" in arg_ty_str:
                            if hasattr(arg, "ctypes"):
                                processed_args.append(ctypes.c_void_p(arg.ctypes.data))
                                processed_args.append(ctypes.c_int64(arg.size))
                            else:
                                mv = memoryview(arg)
                                item_size = ctypes.sizeof(get_ctypes_type(arg_ty_str))
                                processed_args.append(
                                    ctypes.addressof(
                                        (ctypes.c_char * mv.nbytes).from_buffer(arg)
                                    )
                                )
                                processed_args.append(
                                    ctypes.c_int64(mv.nbytes // item_size)
                                )
                        elif any(
                            x in arg_ty_str
                            for x in [
                                "hand",
                                "peek",
                                "sizedarray",
                                "fnpointer",
                                "callable",
                                "closure",
                            ]
                        ):
                            if hasattr(arg, "__lila_ptr__"):
                                processed_args.append(ctypes.c_void_p(arg.__lila_ptr__))
                            else:
                                processed_args.append(arg)
                        else:
                            processed_args.append(get_ctypes_type(arg_ty_str)(arg))
                        arg_idx += 1

                    res = c_func(*processed_args)

                    # Recursively wrap if the return type is another function
                    from .types import FnPointer, Closure

                    if (
                        "fnpointer" in ret_ty_str
                        or "callable" in ret_ty_str
                        or "closure" in ret_ty_str
                        or isinstance(ret_type, FnPointer)
                    ):
                        is_cls = "closure" in ret_ty_str or isinstance(
                            ret_type, Closure
                        )
                        return create_jit_wrapper(
                            res,
                            ret_type.arg_types,
                            ret_type.ret_type,
                            is_closure=is_cls,
                        )

                    return res

                jit_call.__lila_ptr__ = code_ptr
                return jit_call

            c_func = ctypes.CFUNCTYPE(c_ret, *c_args)(code_ptr)

            def wrapper(*args):
                # Runtime Refinement Checks
                for i, param in enumerate(sig.parameters.values()):
                    if i < len(args):
                        ann = param.annotation
                        if hasattr(ann, "predicate") and ann.predicate:
                            if not ann.predicate(args[i]):
                                raise ValueError(
                                    f"Runtime Refinement Violation for argument '{param.name}': "
                                    f"Value {args[i]} does not satisfy the predicate."
                                )

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
                                processed_args.append(
                                    ctypes.c_int64(mv.nbytes // item_size)
                                )
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
                        processed_args.append(target_cty(arg))

                res = c_func(*processed_args)

                if is_tuple_return:
                    # Convert ctypes structure back to Python tuple
                    return tuple(
                        getattr(ret_struct, f"f{i}") for i in range(len(tuple_types))
                    )

                # Wrap returned function pointer/closure
                from .types import FnPointer, Closure

                if (
                    "fnpointer" in ret_ann_str
                    or "callable" in ret_ann_str
                    or "closure" in ret_ann_str
                    or isinstance(ret_ann, FnPointer)
                ):
                    is_cls = "closure" in ret_ann_str or isinstance(ret_ann, Closure)
                    return create_jit_wrapper(
                        res, ret_ann.arg_types, ret_ann.ret_type, is_closure=is_cls
                    )

                return res

            print(f"[Lila] JIT compiled '{func.__name__}' successfully.")
            wrapper.__lila_jit__ = True
            wrapper.__lila_ptr__ = code_ptr
            return wrapper
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
