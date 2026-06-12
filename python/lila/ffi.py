import inspect
import ctypes
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Callable,
)
from .types import TYPE_MAP, Buffer, SizedArray, Closure, FnPointer, Box, Tensor
from .signatures import _get_type_name, _value_to_lila_type


def _get_ctypes_type(ann_str: str) -> Any:
    """Map a type name string to a ctypes type."""
    # Sort keys by length descending to match 'f32x4' before 'f32'
    for name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
        if name in ann_str:
            return TYPE_MAP[name]
    return ctypes.c_int64


def _map_ctypes_arguments(
    sig: inspect.Signature, class_name: str = None, type_mapping: Dict[str, str] = None
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
        ann_str = _get_type_name(actual_ann, type_mapping).lower()

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

        is_tensor = (
            isinstance(origin, type) and issubclass(origin, Tensor)
        ) or "tensor" in ann_str

        is_ptr_wrapper = False
        if isinstance(origin, type) and issubclass(
            origin, (SizedArray, Closure, FnPointer, Callable, Box, Tensor)
        ):
            is_ptr_wrapper = True
        if not is_ptr_wrapper and any(
            x in ann_str
            for x in [
                "sizedarray",
                "closure",
                "fnpointer",
                "callable",
                "box",
                "tensor",
                "f32x4",
                "i32x4",
                "f64x2",
                "i64x2",
                "i8x16",
                "u8x16",
                "i16x8",
                "u16x8",
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
        elif is_tensor:
            c_args.append(ctypes.c_void_p)
            dim_count = 2  # default to 2D
            if origin is Annotated and hasattr(actual_ann, "__metadata__"):
                metadata = actual_ann.__metadata__[0]
                if isinstance(metadata, tuple) and len(metadata) > 1:
                    dim_count = len(metadata[1])  # length of the shape tuple
            for _ in range(dim_count):
                c_args.append(ctypes.c_int64)
            arg_map.append(("tensor", len(c_args) - 1 - dim_count, dim_count))
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
    ret_ann: Any,
    c_args: List[Any],
    arg_map: List[Any],
    type_mapping: Dict[str, str] = None,
) -> Tuple[Any, Any, List[Any], List[Any]]:
    """Handle return-by-pointer for tuples and SIMD types, adjusting argument mapping."""
    ret_ann_str = _get_type_name(ret_ann, type_mapping).lower()
    raw_ann_str = str(ret_ann).lower()

    # Detect if we need return-by-pointer (SRet style)
    is_tuple = "tuple" in raw_ann_str or (
        ret_ann_str.startswith("(") and ret_ann_str.endswith(")")
    )
    is_simd = any(
        x in ret_ann_str
        for x in [
            "f32x4",
            "i32x4",
            "f64x2",
            "i64x2",
            "i8x16",
            "u8x16",
            "i16x8",
            "u16x8",
        ]
    )

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
                t_str = _get_type_name(t, type_mapping).lower()
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
                "i8x16",
                "u8x16",
                "i16x8",
                "u16x8",
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
        processed_args, ret_struct, anchors = _prepare_runtime_args(
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
    is_ptr_return: bool,
    TupleReturn: Any,
) -> Tuple[List[Any], Any, List[Any]]:
    """Map Python arguments to ctypes arguments, tracking anchors for lifetime."""
    processed_args = []
    anchors = []
    ret_struct = None

    if is_ptr_return:
        ret_struct = TupleReturn()
        processed_args.append(ctypes.byref(ret_struct))

    for i, arg_info in enumerate(arg_map):
        arg_type = arg_info[0]
        c_idx = arg_info[1]
        arg = args[i]

        if arg_type == "buffer":
            item_size = arg_info[2]
            if hasattr(arg, "ctypes"):
                processed_args.append(ctypes.c_void_p(arg.ctypes.data))
                processed_args.append(ctypes.c_int64(arg.size))
            elif hasattr(arg, "__array_interface__"):
                processed_args.append(
                    ctypes.c_void_p(arg.__array_interface__["data"][0])
                )
                processed_args.append(ctypes.c_int64(arg.size))
            else:
                try:
                    mv = memoryview(arg)
                    if not mv.contiguous:
                        arg = mv.tobytes()
                        mv = memoryview(arg)
                        anchors.append(arg)

                    ArrayType = ctypes.c_char * mv.nbytes
                    c_buf = ArrayType.from_buffer(arg)
                    processed_args.append(ctypes.addressof(c_buf))
                    processed_args.append(ctypes.c_int64(mv.nbytes // item_size))
                    anchors.append(c_buf)
                except Exception as e:
                    raise TypeError(f"Argument {i} failed buffer conversion: {e}")
        elif arg_type == "tensor":
            dim_count = arg_info[2]
            processed_args.append(ctypes.c_void_p(arg.ptr))
            for j in range(dim_count):
                processed_args.append(ctypes.c_int64(arg.shape[j]))
        elif arg_type == "pointer":
            if hasattr(arg, "_ctypes_obj"):
                processed_args.append(ctypes.addressof(arg._ctypes_obj))
            elif isinstance(arg, Box):
                if hasattr(arg.value, "_ctypes_obj"):
                    processed_args.append(ctypes.addressof(arg.value._ctypes_obj))
                else:
                    c_ty = _get_ctypes_type(_value_to_lila_type(arg.value))
                    c_val = c_ty(arg.value)
                    processed_args.append(ctypes.byref(c_val))
                    anchors.append(c_val)
            elif isinstance(arg, ctypes.Structure):
                processed_args.append(ctypes.addressof(arg))
            elif hasattr(arg, "__lila_ptr__"):
                processed_args.append(ctypes.c_void_p(arg.__lila_ptr__))
            else:
                processed_args.append(arg)
        else:
            target_cty = c_args[c_idx]
            if hasattr(arg, "_ctypes_obj"):
                processed_args.append(arg._ctypes_obj)
            elif (
                isinstance(arg, target_cty)
                or hasattr(arg, "_type_")
                or hasattr(arg, "_fields_")
                or hasattr(arg, "_obj_")
            ):
                processed_args.append(arg)
            else:
                processed_args.append(target_cty(arg))

    return processed_args, ret_struct, anchors


def _check_runtime_refinements(sig: inspect.Signature, args: Tuple):
    """Validate runtime refinements for arguments."""
    for i, param in enumerate(sig.parameters.values()):
        if i < len(args):
            ann = param.annotation
            if hasattr(ann, "predicate") and ann.predicate:
                if callable(ann.predicate):
                    if not ann.predicate(args[i]):
                        raise ValueError(
                            f"Runtime Refinement Violation for argument '{param.name}': "
                            f"Value {args[i]} does not satisfy the predicate."
                        )
                # If predicate is a string, it's a symbolic refinement for the verifier,
                # we skip it at runtime.


def _wrap_return_value(
    res: Any,
    ret_ann: Any,
    type_mapping: Dict[str, str] = None,
    sig: inspect.Signature = None,
    args: Tuple = None,
) -> Any:
    """Wrap the JIT return value if it represents a higher-order function or Tensor."""
    from .types import FnPointer, Closure, i64, Tensor
    from typing import get_origin

    actual_ann = getattr(ret_ann, "base_type", ret_ann)
    origin = get_origin(actual_ann) or actual_ann
    is_tensor = (
        isinstance(origin, type) and issubclass(origin, Tensor)
    ) or "tensor" in str(ret_ann).lower()

    if is_tensor and sig and args:
        # Build symbol table for dimensions
        sym_table = {}
        for i, param in enumerate(sig.parameters.values()):
            if i < len(args):
                p_ann = param.annotation
                p_orig = get_origin(p_ann) or p_ann
                is_p_tensor = (
                    isinstance(p_orig, type) and issubclass(p_orig, Tensor)
                ) or "tensor" in str(p_ann).lower()
                if is_p_tensor:
                    if hasattr(p_ann, "__metadata__"):
                        p_meta = p_ann.__metadata__[0]
                        if isinstance(p_meta, tuple) and len(p_meta) > 1:
                            p_shape = p_meta[1]
                            arg_val = args[i]
                            if hasattr(arg_val, "shape"):
                                for dim_name, actual_size in zip(
                                    p_shape, arg_val.shape
                                ):
                                    sym_table[dim_name] = actual_size

        # Determine return shape
        ret_shape = []
        if hasattr(actual_ann, "__metadata__"):
            ret_meta = actual_ann.__metadata__[0]
            if isinstance(ret_meta, tuple) and len(ret_meta) > 1:
                for dim_name in ret_meta[1]:
                    ret_shape.append(
                        sym_table.get(dim_name, 1)
                    )  # Default 1 if not found

        import ctypes

        # We assume the returned pointer is a ctypes.c_void_p or int
        ptr_val = res
        if not isinstance(ptr_val, int):
            ptr_val = ctypes.cast(ptr_val, ctypes.c_void_p).value

        # Reconstruct the Tensor
        # Since we just have a raw pointer, we wrap it in a ctypes array of the correct size
        item_ty_str = (
            str(ret_meta[0]).lower() if hasattr(actual_ann, "__metadata__") else "f32"
        )
        item_cty = ctypes.c_float
        for name, cty in TYPE_MAP.items():
            if name in item_ty_str:
                item_cty = cty
                break

        total_size = 1
        for d in ret_shape:
            total_size *= d

        ArrayType = item_cty * total_size
        c_buf = ArrayType.from_address(ptr_val)
        return Tensor(c_buf, tuple(ret_shape))

    ret_ann_str = _get_type_name(ret_ann, type_mapping).lower()
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


def _get_ctypes_return_type(ret_ann: Any, type_mapping: Dict[str, str] = None) -> Any:
    """Determine the ctypes return type from the annotation."""
    # Unwrap Refined type if necessary
    actual_ann = getattr(ret_ann, "base_type", ret_ann)
    ret_ann_str = _get_type_name(actual_ann, type_mapping).lower()

    if (
        actual_ann is None
        or actual_ann is inspect.Signature.empty
        or "none" in ret_ann_str
    ):
        return None

    from typing import get_origin
    from .types import Tensor, Buffer

    origin = get_origin(actual_ann) or actual_ann
    if (
        (isinstance(origin, type) and issubclass(origin, (Tensor, Buffer)))
        or "tensor" in ret_ann_str
        or "buffer" in ret_ann_str
    ):
        return ctypes.c_int64

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
    type_mapping: Dict[str, str] = None,
):
    """Generate the final Python wrapper that handles runtime checks and interop."""
    c_ret = (
        None
        if is_ptr_return
        else _get_ctypes_return_type(sig.return_annotation, type_mapping)
    )
    c_func = ctypes.CFUNCTYPE(c_ret, *c_args)(code_ptr)

    def wrapper(*args):
        _check_runtime_refinements(sig, args)

        processed_args, ret_struct, anchors = _prepare_runtime_args(
            args, arg_map, c_args, is_ptr_return, TupleReturn
        )

        res = c_func(*processed_args)

        if is_ptr_return:
            ret_ann_str = _get_type_name(sig.return_annotation, type_mapping).lower()
            raw_ann_str = str(sig.return_annotation).lower()
            if "tuple" in raw_ann_str or (
                ret_ann_str.startswith("(") and ret_ann_str.endswith(")")
            ):
                return tuple(
                    getattr(ret_struct, f"f{i}") for i in range(len(tuple_types))
                )
            else:
                # SIMD Return
                return ret_struct

        return _wrap_return_value(res, sig.return_annotation, type_mapping, sig, args)

    print(f"[Lila] JIT compiled '{func.__name__}' successfully.")
    wrapper.__lila_jit__ = True
    wrapper.__lila_ptr__ = code_ptr
    return wrapper
