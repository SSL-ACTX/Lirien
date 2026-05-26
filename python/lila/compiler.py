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


def verify(
    strict: bool = True,
    _struct_layouts: dict = None,
    _class_name: str = None,
    _method_name: str = None,
) -> Callable[[T], T]:
    """
    Decorator to trigger formal verification and JIT compilation.

    :param strict: If True, raises VerificationError on failure. If False, falls back to Python.
    """

    def decorator(func: T) -> T:
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
                            value=ast.Name(id="Mut", ctx=ast.Load()),
                            slice=ast.Name(id=_class_name, ctx=ast.Load()),
                            ctx=ast.Load(),
                        )
                source = ast.unparse(tree)

            # Scan for struct layouts and type aliases referenced in the function's scope
            struct_layouts = _struct_layouts.copy() if _struct_layouts else {}
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

            code_ptr = lila_core.verify_and_compile(
                source, target_func_name, struct_layouts, type_aliases
            )

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
                elif any(x in ann_str for x in ["mut", "ref", "sizedarray"]) or getattr(
                    actual_ann, "__lila_struct__", False
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

            c_func = ctypes.CFUNCTYPE(c_ret, *c_args)(code_ptr)

            def wrapper(*args):
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
                return res

            print(f"[Lila] JIT compiled '{func.__name__}' successfully.")
            wrapper.__lila_jit__ = True
            return wrapper
        except Exception as e:
            error_msg = f"Lila Verification Failed for '{func.__name__}': {e}"
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
