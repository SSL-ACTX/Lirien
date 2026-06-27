import inspect
import ctypes
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Callable,
    get_origin,
    get_args,
    Tuple as typing_Tuple,
)
from .types.base import TYPE_MAP
from .types.memory import Buffer, SizedArray, Box, Tensor, List
from .types.functions import Closure, FnPointer
from .compiler import (
    _get_type_name,
    _value_to_lirien_type,
    is_named_tuple,
    is_typed_dict,
    _get_refinement_parts,
)


def _get_ctypes_type(ann_str: str) -> Any:
    """Map a type name string to a ctypes type."""
    # Prioritize specific Lirien types to avoid 'float' matching 'float32'
    priority_types = [
        "f32x4",
        "i32x4",
        "f64x2",
        "i64x2",
        "i8x16",
        "u8x16",
        "i16x8",
        "u16x8",
        "f64",
        "f32",
        "i64",
        "u64",
        "i32",
        "u32",
        "i16",
        "u16",
        "i8",
        "u8",
        "bool",
    ]
    for name in priority_types:
        if name in ann_str:
            return TYPE_MAP[name]

    # Fallback for other types
    for name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
        if name in ann_str:
            return TYPE_MAP[name]
    return ctypes.c_int64


_optional_ctypes_cache = {}


def get_optional_ctypes(inner_cty):
    if inner_cty in _optional_ctypes_cache:
        return _optional_ctypes_cache[inner_cty]

    class OptionalCtypes(ctypes.Structure):
        _fields_ = [("has_value", ctypes.c_bool), ("value", inner_cty)]

    _optional_ctypes_cache[inner_cty] = OptionalCtypes
    return OptionalCtypes


def _is_value_optional(ann):
    from typing import get_origin, get_args, Union
    import types
    import sys

    actual_ann = getattr(ann, "base_type", ann)
    origin = get_origin(actual_ann) or actual_ann
    if origin is Union or (sys.version_info >= (3, 10) and origin is types.UnionType):
        args = get_args(actual_ann)
        has_none = any(arg is type(None) or arg is None for arg in args)
        if has_none:
            non_none_args = [
                arg for arg in args if arg is not type(None) and arg is not None
            ]
            if non_none_args:
                from .compiler.signature_helpers import _is_box_type

                return not _is_box_type(non_none_args[0]), non_none_args[0]
    return False, None


def _get_flattened_ctypes_types(
    ty: Any, type_mapping: Dict[str, str] = None
) -> List[Any]:
    """Recursively discover all basic ctypes types for a given Lirien type."""
    from .types import i64
    from typing import Tuple as typing_Tuple, get_origin

    if is_named_tuple(ty):
        res = []
        for f_name in ty._fields:
            f_ann = ty.__annotations__.get(f_name, i64)
            res.extend(_get_flattened_ctypes_types(f_ann, type_mapping))
        return res

    origin = get_origin(ty)
    if origin is tuple or origin is typing_Tuple:
        args = get_args(ty)
        res = []
        for arg in args:
            res.extend(_get_flattened_ctypes_types(arg, type_mapping))
        return res

    ty_str = _get_type_name(ty, type_mapping).lower()
    return [_get_ctypes_type(ty_str)]


def _flatten_values(obj: Any) -> List[Any]:
    """Recursively flatten all values in a Tuple or NamedTuple tree."""
    res = []
    if is_named_tuple(type(obj)) or isinstance(obj, (list, tuple)):
        for val in obj:
            if is_named_tuple(type(val)) or isinstance(val, (list, tuple)):
                res.extend(_flatten_values(val))
            else:
                res.append(val)
    else:
        res.append(obj)
    return res


def _unflatten_values(ty: Any, flattened_values: List[Any]) -> Any:
    """Recursively reconstruct a Tuple or NamedTuple from flattened values."""
    from .types import i64
    from typing import Tuple as typing_Tuple, get_origin, get_args

    if is_named_tuple(ty):
        fields_vals = []
        idx = 0
        for f_name in ty._fields:
            f_ann = ty.__annotations__.get(f_name, i64)
            count = len(_get_flattened_ctypes_types(f_ann))
            fields_vals.append(
                _unflatten_values(f_ann, flattened_values[idx : idx + count])
            )
            idx += count
        return ty(*fields_vals)

    origin = get_origin(ty)
    if origin is tuple or origin is typing_Tuple:
        args = get_args(ty)
        res = []
        idx = 0
        for arg in args:
            count = len(_get_flattened_ctypes_types(arg))
            res.append(_unflatten_values(arg, flattened_values[idx : idx + count]))
            idx += count
        return tuple(res)

    return flattened_values[0]


def _map_ctypes_arguments(
    sig: inspect.Signature, class_name: str = None, type_mapping: Dict[str, str] = None
) -> Tuple[List[Any], List[Any]]:
    """Map Python function parameters to ctypes types and tracking info."""
    c_args = []
    arg_map = []  # List of (type, c_idx, [metadata])

    for i, param in enumerate(sig.parameters.values()):
        ann = param.annotation

        if param.name == "self" and class_name:
            c_args.append(ctypes.c_void_p)
            arg_map.append(("pointer", len(c_args) - 1))
            continue

        actual_ann = getattr(ann, "base_type", ann)

        # Resolve actual_ann from type_mapping if it was substituted
        if (
            isinstance(actual_ann, type)
            and type_mapping
            and actual_ann.__name__ in type_mapping
        ):
            actual_ann = type_mapping[actual_ann.__name__]
        elif type_mapping and getattr(actual_ann, "__name__", None) in type_mapping:
            actual_ann = type_mapping[actual_ann.__name__]
        elif type_mapping and str(actual_ann) in type_mapping:
            actual_ann = type_mapping[str(actual_ann)]

        ann_str = _get_type_name(actual_ann, type_mapping).lower()

        from typing import get_origin, Annotated

        origin = get_origin(actual_ann) or actual_ann

        if is_named_tuple(actual_ann):
            # Unpack NamedTuple into multiple arguments recursively
            flattened_ctypes = _get_flattened_ctypes_types(actual_ann, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("named_tuple", start_idx, len(flattened_ctypes)))
            continue

        from typing import Tuple as typing_Tuple

        if origin is tuple or origin is typing_Tuple:
            # Unpack standard Tuple into multiple arguments recursively
            flattened_ctypes = _get_flattened_ctypes_types(actual_ann, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("tuple", start_idx, len(flattened_ctypes)))
            continue

        is_buffer = (
            isinstance(origin, type) and issubclass(origin, Buffer)
        ) or "buffer" in ann_str

        is_tensor = (
            isinstance(origin, type) and issubclass(origin, Tensor)
        ) or "tensor" in ann_str

        is_ptr_wrapper = False
        if isinstance(origin, type) and issubclass(
            origin, (SizedArray, Closure, FnPointer, Callable, Box, Tensor, List)
        ):
            is_ptr_wrapper = True

        # Check for Protocol (duck typing)
        if (
            not is_ptr_wrapper
            and hasattr(actual_ann, "_is_protocol")
            and actual_ann._is_protocol
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
                "list",
                "nullable",
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
            c_args.append(ctypes.c_void_p)  # Ptr
            c_args.append(ctypes.c_int64)  # Len
            item_size = 8

            # Check metadata for item type
            item_ty = None
            if origin is Annotated and hasattr(actual_ann, "__metadata__"):
                item_ty = actual_ann.__metadata__[0]

            if item_ty is Ellipsis:
                # Inferred type from ellipsis
                ellipsis_key = f"__ellipsis_{param.name}"
                if type_mapping and ellipsis_key in type_mapping:
                    # mapping[key] is a list [type_name] or [dim1, dim2, ...]
                    m_val = type_mapping[ellipsis_key]
                    if isinstance(m_val, list) and len(m_val) > 0:
                        item_ty_str = str(m_val[0]).lower()
                        priority_types = [
                            "f32x4",
                            "i32x4",
                            "f64x2",
                            "i64x2",
                            "i8x16",
                            "u8x16",
                            "i16x8",
                            "u16x8",
                            "f64",
                            "f32",
                            "i64",
                            "u64",
                            "i32",
                            "u32",
                            "i16",
                            "u16",
                            "i8",
                            "u8",
                            "bool",
                        ]
                        for name in priority_types:
                            if name in item_ty_str:
                                item_size = ctypes.sizeof(TYPE_MAP[name])
                                break
            elif item_ty is not None:
                if isinstance(item_ty, type) and issubclass(
                    item_ty, ctypes._SimpleCData
                ):
                    item_size = ctypes.sizeof(item_ty)
                elif _get_refinement_parts(item_ty) != (None, None):
                    item_base_ty, _ = _get_refinement_parts(item_ty)
                    item_ty_str = _get_type_name(item_base_ty, type_mapping).lower()
                    for name in [
                        "f32x4",
                        "i32x4",
                        "f64x2",
                        "i64x2",
                        "i8x16",
                        "u8x16",
                        "i16x8",
                        "u16x8",
                        "f64",
                        "f32",
                        "i64",
                        "u64",
                        "i32",
                        "u32",
                        "i16",
                        "u16",
                        "i8",
                        "u8",
                        "bool",
                    ]:
                        if name in item_ty_str:
                            item_size = ctypes.sizeof(TYPE_MAP[name])
                            break
                elif getattr(item_ty, "__lirien_struct__", False):
                    item_size = ctypes.sizeof(item_ty.__lirien_ctypes__)
                else:
                    item_ty_str = str(item_ty).lower()
                    for name in [
                        "f32x4",
                        "i32x4",
                        "f64x2",
                        "i64x2",
                        "i8x16",
                        "u8x16",
                        "i16x8",
                        "u16x8",
                        "f64",
                        "f32",
                        "i64",
                        "u64",
                        "i32",
                        "u32",
                        "i16",
                        "u16",
                        "i8",
                        "u8",
                        "bool",
                    ]:
                        if name in item_ty_str:
                            item_size = ctypes.sizeof(TYPE_MAP[name])
                            break
            else:
                # Fallback to ann_str
                for name in [
                    "f32x4",
                    "i32x4",
                    "f64x2",
                    "i64x2",
                    "i8x16",
                    "u8x16",
                    "i16x8",
                    "u16x8",
                    "f64",
                    "f32",
                    "i64",
                    "u64",
                    "i32",
                    "u32",
                    "i16",
                    "u16",
                    "i8",
                    "u8",
                    "bool",
                ]:
                    if name in ann_str:
                        item_size = ctypes.sizeof(TYPE_MAP[name])
                        break
            arg_map.append(("buffer", len(c_args) - 2, item_size))
        elif is_tensor:
            c_args.append(ctypes.c_void_p)
            dim_count = 2  # default to 2D
            if origin is Annotated and hasattr(actual_ann, "__metadata__"):
                metadata = actual_ann.__metadata__[0]
                if isinstance(metadata, tuple) and len(metadata) > 1:
                    shape = metadata[1]
                    # Handle Unpack in shape
                    resolved_dim_count = 0
                    for s in shape:
                        s_origin = get_origin(s)
                        if s_origin is not None and "Unpack" in str(s_origin):
                            s_args = get_args(s)
                            if (
                                s_args
                                and type_mapping
                                and s_args[0].__name__ in type_mapping
                            ):
                                unpack_val = type_mapping[s_args[0].__name__]
                                if isinstance(unpack_val, (list, tuple)):
                                    resolved_dim_count += len(unpack_val)
                                else:
                                    resolved_dim_count += 1
                        else:
                            resolved_dim_count += 1
                    dim_count = resolved_dim_count
            for _ in range(dim_count):
                c_args.append(ctypes.c_int64)
            arg_map.append(("tensor", len(c_args) - 1 - dim_count, dim_count))
        elif (
            is_ptr_wrapper
            or getattr(actual_ann, "__lirien_struct__", False)
            or getattr(actual_ann, "__lirien_enum__", False)
            or is_typed_dict(actual_ann)
        ):
            c_args.append(ctypes.c_void_p)
            arg_map.append(("pointer", len(c_args) - 1, actual_ann))
        elif is_named_tuple(actual_ann):
            # Unpack NamedTuple into multiple arguments recursively
            flattened_ctypes = _get_flattened_ctypes_types(actual_ann, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("named_tuple", start_idx, len(flattened_ctypes)))
        else:
            c_args.append(_get_ctypes_type(ann_str))
            arg_map.append(("value", len(c_args) - 1))
    return c_args, arg_map


def _handle_pointer_return(
    ret_ann: Any,
    c_args: List[Any],
    arg_map: List[Any],
    type_mapping: Dict[str, str] = None,
) -> Tuple[bool, Any, List[Any], List[Any], List[Any]]:
    ret_ann_str = _get_type_name(ret_ann, type_mapping).lower()
    raw_ann_str = str(ret_ann).lower()
    is_val_opt, inner_type = _is_value_optional(ret_ann)
    is_struct = getattr(ret_ann, "__lirien_struct__", False)

    if is_val_opt or is_struct:
        if is_val_opt:
            if getattr(inner_type, "__lirien_struct__", False):
                inner_cty = inner_type.__lirien_ctypes__
            else:
                inner_cty = _get_ctypes_type(_get_type_name(inner_type, type_mapping))
            ResultStruct = get_optional_ctypes(inner_cty)
        else:
            ResultStruct = ret_ann.__lirien_ctypes__

        new_c_args = [ctypes.c_void_p] + c_args
        new_arg_map = []
        for info in arg_map:
            # Adjust index for SRet
            new_arg_map.append((info[0], info[1] + 1) + info[2:])
        return True, ResultStruct, new_c_args, new_arg_map, []

    # Detect if we need return-by-pointer (SRet style)
    is_tuple = (
        "tuple" in raw_ann_str
        or (ret_ann_str.startswith("(") and ret_ann_str.endswith(")"))
        or is_named_tuple(ret_ann)
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
        return False, None, c_args, arg_map, []

    from .types import i64

    if is_named_tuple(ret_ann) or get_origin(ret_ann) in [tuple, typing_Tuple]:
        flattened_ctypes = _get_flattened_ctypes_types(ret_ann, type_mapping)

        class TupleResult(ctypes.Structure):
            _fields_ = [(f"f{i}", cty) for i, cty in enumerate(flattened_ctypes)]

        if len(flattened_ctypes) <= 2:
            # Return by registers
            return False, TupleResult, c_args, arg_map, flattened_ctypes
        else:
            # Return by pointer (SRet)
            c_args.insert(0, ctypes.POINTER(TupleResult))
            # Shift existing arg_map indices
            new_arg_map = []
            for item in arg_map:
                if item[0] == "named_tuple" or item[0] == "tuple":
                    new_arg_map.append((item[0], item[1] + 1, item[2]))
                else:
                    new_arg_map.append((item[0], item[1] + 1))
            return True, TupleResult, c_args, new_arg_map, flattened_ctypes

    try:
        ResultStruct = None
        tuple_types = []

        if is_tuple:
            # Try to extract inner types for Tuple
            if hasattr(ret_ann, "__args__"):
                tuple_types = list(ret_ann.__args__)
            else:
                tuple_types = [i64, i64]  # Default

            tuple_fields = []
            for i, t in enumerate(tuple_types):
                f_ty_str = _get_type_name(t, type_mapping).lower()
                tuple_fields.append((f"f{i}", _get_ctypes_type(f_ty_str)))

            class TupleReturn(ctypes.Structure):
                _fields_ = tuple_fields

            ResultStruct = TupleReturn
        elif is_simd:
            # SIMD Return - Match the specific vector type exactly if possible
            # Sort keys by length descending to match 'f32x4' before 'f32'
            for name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
                if name in ret_ann_str:
                    ResultStruct = TYPE_MAP[name]
                    break

        if ResultStruct is None:
            return False, None, c_args, arg_map, []

        new_c_args = [ctypes.c_void_p] + c_args
        new_arg_map = []
        for info in arg_map:
            # Adjust arg_map indices because we inserted a pointer at index 0
            new_arg_map.append((info[0], info[1] + 1) + info[2:])

        return True, ResultStruct, new_c_args, new_arg_map, tuple_types
    except Exception as e:
        print(f"[Lirien Warning] Failed to setup result structure: {e}")
        return False, None, c_args, arg_map, []


def _create_jit_wrapper(
    code_ptr: int,
    arg_types: List[Any],
    ret_type: Any,
    is_closure: bool = False,
    type_mapping: Dict[str, str] = None,
    name: str = None,
):
    """Create a high-performance wrapper for a JIT-compiled function or closure."""
    import ctypes
    from .types import FnPointer, Closure, i64, TYPE_MAP

    c_args = []
    arg_map = []

    if is_closure:
        c_args.append(ctypes.c_void_p)  # ctx_ptr

    for i, ty in enumerate(arg_types):
        ty_str = _get_type_name(ty, type_mapping).lower()
        if "buffer" in ty_str:
            c_args.append(ctypes.c_void_p)
            c_args.append(ctypes.c_int64)
            item_size = 8
            for name, cty in TYPE_MAP.items():
                if name in ty_str:
                    item_size = ctypes.sizeof(cty)
                    break
            arg_map.append(("buffer", len(c_args) - 2, item_size))
        elif any(
            x in ty_str
            for x in [
                "sizedarray",
                "fnpointer",
                "callable",
                "closure",
                "box",
                "tensor",
                "list",
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
        elif is_named_tuple(ty):
            # Unpack NamedTuple recursively
            flattened_ctypes = _get_flattened_ctypes_types(ty, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("named_tuple", start_idx, len(flattened_ctypes)))
        elif get_origin(ty) is tuple or get_origin(ty) is typing_Tuple:
            # Unpack standard Tuple recursively
            flattened_ctypes = _get_flattened_ctypes_types(ty, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("tuple", start_idx, len(flattened_ctypes)))
        else:
            c_args.append(_get_ctypes_type(ty_str))
            arg_map.append(("value", len(c_args) - 1))

    # Create temporary signature for _handle_pointer_return
    params = [
        inspect.Parameter(
            f"p{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ty
        )
        for i, ty in enumerate(arg_types)
    ]
    dummy_sig = inspect.Signature(params, return_annotation=ret_type)

    is_ptr_return, TupleReturn, c_args, arg_map, tuple_types = _handle_pointer_return(
        ret_type, c_args, arg_map, type_mapping
    )

    if is_ptr_return:
        c_ret = None
    elif is_named_tuple(ret_type):
        c_ret = TupleReturn  # register-based return as structure
    else:
        ret_ty_str = _get_type_name(ret_type, type_mapping).lower()
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

    def jit_call(*args):
        processed_args, ret_struct, anchors, sync_backs = _prepare_runtime_args(
            args, arg_map, c_args, is_ptr_return, TupleReturn
        )
        if is_closure:
            # If it's a pointer return, TupleReturn* is at index 0, so ctx_ptr is at index 1.
            # Otherwise, ctx_ptr is at index 0.
            insert_idx = 1 if is_ptr_return else 0
            processed_args.insert(insert_idx, code_ptr)

        res = c_func(*processed_args)

        # Sync back changes for TypedDict
        for target_dict, struct_obj in sync_backs:
            for field_name, _ in struct_obj._fields_:
                target_dict[field_name] = getattr(struct_obj, field_name)

        if (
            is_named_tuple(ret_type) or get_origin(ret_type) in [tuple, typing_Tuple]
        ) and not is_ptr_return:
            # Construct Tuple/NamedTuple from registers
            flattened_res = [getattr(res, f"f{i}") for i in range(len(tuple_types))]
            return _unflatten_values(ret_type, flattened_res)

        if is_ptr_return:
            if is_named_tuple(ret_type) or get_origin(ret_type) in [
                tuple,
                typing_Tuple,
            ]:
                flattened_res = [
                    getattr(ret_struct, f"f{i}") for i in range(len(tuple_types))
                ]
                return _unflatten_values(ret_type, flattened_res)
            elif "tuple" in str(ret_type).lower():
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
                type_mapping=type_mapping,
            )

        return res

    jit_call.__lirien_ptr__ = code_ptr
    jit_call.__lirien_jit__ = True
    jit_call.__lirien_closure__ = is_closure
    if name:
        jit_call.__name__ = name
    return jit_call


def _prepare_runtime_args(
    args: Tuple,
    arg_map: List[Any],
    c_args: List[Any],
    is_ptr_return: bool,
    TupleReturn: Any,
) -> Tuple[List[Any], Any, List[Any], List[Any]]:
    """Map Python arguments to ctypes arguments, tracking anchors for lifetime and sync-backs."""
    processed_args = []
    anchors = []
    sync_backs = []
    ret_struct = None

    if is_ptr_return:
        ret_struct = TupleReturn()
        processed_args.append(ctypes.byref(ret_struct))
    else:
        ret_struct = None

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
            is_val_opt = False
            inner_type = None
            if len(arg_info) > 2:
                is_val_opt, inner_type = _is_value_optional(arg_info[2])

            if is_val_opt:
                if getattr(inner_type, "__lirien_struct__", False):
                    inner_cty = inner_type.__lirien_ctypes__
                else:
                    inner_cty = _get_ctypes_type(_get_type_name(inner_type))
                OptStruct = get_optional_ctypes(inner_cty)
                if arg is None:
                    c_val = OptStruct(has_value=False)
                else:
                    val_obj = arg._ctypes_obj if hasattr(arg, "_ctypes_obj") else arg
                    c_val = OptStruct(has_value=True, value=val_obj)
                processed_args.append(ctypes.c_void_p(ctypes.addressof(c_val)))
                anchors.append(c_val)
            elif hasattr(arg, "__lirien_ptr__"):
                processed_args.append(ctypes.c_void_p(arg.__lirien_ptr__))
            elif hasattr(arg, "_ctypes_obj"):
                # If it's already a pointer or c_void_p, pass it directly.
                # Otherwise, take the address of the struct.
                if isinstance(arg._ctypes_obj, (ctypes.c_void_p, ctypes._Pointer)):
                    processed_args.append(arg._ctypes_obj)
                else:
                    processed_args.append(
                        ctypes.c_void_p(ctypes.addressof(arg._ctypes_obj))
                    )
            elif isinstance(arg, Box):
                if hasattr(arg.value, "_ctypes_obj"):
                    processed_args.append(
                        ctypes.c_void_p(ctypes.addressof(arg.value._ctypes_obj))
                    )
                else:
                    c_ty = _get_ctypes_type(_value_to_lirien_type(arg.value))
                    c_val = c_ty(arg.value)
                    processed_args.append(ctypes.byref(c_val))
                    anchors.append(c_val)
            elif (
                isinstance(arg, dict)
                and len(arg_info) > 2
                and is_typed_dict(arg_info[2])
            ):
                # Convert dict to TypedDict structure
                TD = arg_info[2]
                if hasattr(TD, "__lirien_ctypes__"):
                    struct_obj = TD.__lirien_ctypes__(**arg)
                    processed_args.append(ctypes.c_void_p(ctypes.addressof(struct_obj)))
                    anchors.append(struct_obj)
                    sync_backs.append((arg, struct_obj))
                else:
                    processed_args.append(arg)
            elif isinstance(arg, ctypes.Structure):
                processed_args.append(ctypes.c_void_p(ctypes.addressof(arg)))
            else:
                processed_args.append(arg)
        elif arg_type in ["named_tuple", "tuple"]:
            start_c_idx = arg_info[1]
            total_count = arg_info[2]
            flattened = _flatten_values(arg)
            for j in range(total_count):
                field_val = flattened[j]
                target_cty = c_args[start_c_idx + j]
                processed_args.append(target_cty(field_val))
        else:
            target_cty = c_args[c_idx]
            if hasattr(arg, "_ctypes_obj"):
                # If the target ctypes type is a pointer, pass the address of the struct.
                # This handles cases where a Protocol was not resolved to a specific struct
                # during _map_ctypes_arguments but the object has a struct layout.
                if isinstance(target_cty, type) and issubclass(
                    target_cty, (ctypes.c_void_p, ctypes._Pointer)
                ):
                    processed_args.append(
                        ctypes.c_void_p(ctypes.addressof(arg._ctypes_obj))
                    )
                else:
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

    return processed_args, ret_struct, anchors, sync_backs


def _check_runtime_refinements(
    sig: inspect.Signature, args: Tuple, mapping: Dict[str, Any] = None
):
    """Validate runtime refinements for arguments."""
    for i, param in enumerate(sig.parameters.values()):
        if i < len(args):
            ann = param.annotation
            _, predicate = _get_refinement_parts(ann)
            if predicate is not None:
                if callable(predicate):
                    try:
                        res = predicate(args[i])
                        # If it returned a symbolic TypeExpr, evaluate it using the current mapping
                        if hasattr(res, "evaluate") and mapping:
                            res = res.evaluate(mapping)

                        if not res:
                            raise ValueError(
                                f"Runtime Refinement Violation for argument '{param.name}': "
                                f"Value {args[i]} does not satisfy the predicate."
                            )
                    except NameError:
                        # Cross-parameter refinement predicates might fail at runtime
                        # because other parameters are not in the lambda's closure.
                        # We skip these at runtime and rely on Z3's static proof.
                        pass


def _wrap_return_value(
    res: Any,
    ret_ann: Any,
    type_mapping: Dict[str, str] = None,
    sig: inspect.Signature = None,
    args: Tuple = None,
) -> Any:
    """Wrap the JIT return value if it represents a higher-order function or Tensor."""
    from .types import FnPointer, Closure, i64, Tensor
    from .types.memory import List as LirienList
    from typing import get_origin

    actual_ann = getattr(ret_ann, "base_type", ret_ann)
    origin = get_origin(actual_ann) or actual_ann

    is_list = (
        isinstance(origin, type) and issubclass(origin, LirienList)
    ) or "list" in str(ret_ann).lower()

    if is_list:
        import ctypes
        from typing import get_args, Annotated

        ptr_val = res
        if not isinstance(ptr_val, int):
            ptr_val = ctypes.cast(ptr_val, ctypes.c_void_p).value

        if get_origin(actual_ann) is Annotated:
            ann_args = get_args(actual_ann)
            elem_type = ann_args[1] if len(ann_args) > 1 else None
        else:
            ann_args = get_args(actual_ann)
            elem_type = ann_args[0] if ann_args else None
        return LirienList(c_ptr=ctypes.c_void_p(ptr_val), elem_type=elem_type)

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

        # Extract arg_types, ret_type, and target_name
        arg_types = [i64, i64]
        ret_type = i64
        target_name = None

        if isinstance(ret_ann, FnPointer):
            arg_types = ret_ann.arg_types
            ret_type = ret_ann.ret_type
        elif hasattr(ret_ann, "__metadata__"):
            params = ret_ann.__metadata__[0]
            if isinstance(params, tuple) and len(params) >= 2:
                arg_types, ret_type = params[0], params[1]
                if len(params) > 2:
                    target_name = params[2]
        elif hasattr(ret_ann, "__args__"):
            params = ret_ann.__args__
            if len(params) >= 2:
                arg_types, ret_type = params[0], params[1]
                if len(params) > 2:
                    target_name = params[2]

        return _create_jit_wrapper(
            res,
            arg_types,
            ret_type,
            is_closure=is_cls,
            type_mapping=type_mapping,
            name=target_name,
        )
    return res


def _get_ctypes_return_type(ret_ann: Any, type_mapping: Dict[str, str] = None) -> Any:
    """Determine the ctypes return type from the annotation."""
    # Unwrap Refined / Annotated refinement type if necessary
    base_ty, _ = _get_refinement_parts(ret_ann)
    actual_ann = base_ty if base_ty is not None else ret_ann
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


def _extract_runtime_asserts(func: Callable, sig: inspect.Signature):
    import ast
    import inspect
    import textwrap

    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_def = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )

        preconds = []
        postconds = []
        param_names = list(sig.parameters.keys())

        for stmt in func_def.body:
            if isinstance(stmt, ast.Assert):
                lambda_node = ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg=name) for name in param_names],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        vararg=None,
                        kwarg=None,
                    ),
                    body=stmt.test,
                )
                ast.fix_missing_locations(lambda_node)
                expr_code = compile(
                    ast.Expression(body=lambda_node), "<assert>", "eval"
                )
                lambda_fn = eval(expr_code, func.__globals__)
                preconds.append(lambda_fn)
            else:
                break

        def scan_blocks(body):
            for i in range(len(body)):
                if i + 1 < len(body):
                    if isinstance(body[i], ast.Assert) and isinstance(
                        body[i + 1], ast.Return
                    ):
                        ret_name = None
                        if isinstance(body[i + 1].value, ast.Name):
                            ret_name = body[i + 1].value.id

                        post_params = list(param_names)
                        for opt in [ret_name, "res", "return_val"]:
                            if opt and opt not in post_params:
                                post_params.append(opt)

                        lambda_node = ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[ast.arg(arg=name) for name in post_params],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[],
                                vararg=None,
                                kwarg=None,
                            ),
                            body=body[i].test,
                        )
                        ast.fix_missing_locations(lambda_node)
                        expr_code = compile(
                            ast.Expression(body=lambda_node), "<assert>", "eval"
                        )
                        lambda_fn = eval(expr_code, func.__globals__)
                        postconds.append((lambda_fn, post_params, ret_name))
                if hasattr(body[i], "body"):
                    scan_blocks(body[i].body)
                if hasattr(body[i], "orelse"):
                    scan_blocks(body[i].orelse)

        scan_blocks(func_def.body)
        return preconds, postconds
    except Exception:
        return [], []


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
    assert_preconds, assert_postconds = _extract_runtime_asserts(func, sig)

    if (
        is_named_tuple(sig.return_annotation)
        or get_origin(sig.return_annotation) in [tuple, typing_Tuple]
    ) and not is_ptr_return:
        c_ret = TupleReturn
    else:
        c_ret = (
            None
            if is_ptr_return
            else _get_ctypes_return_type(sig.return_annotation, type_mapping)
        )
    c_func = ctypes.CFUNCTYPE(c_ret, *c_args)(code_ptr)

    def wrapper(*args):
        def _call_impl(*args):
            _check_runtime_refinements(sig, args, type_mapping)

            processed_args, ret_struct, anchors, sync_backs = _prepare_runtime_args(
                args, arg_map, c_args, is_ptr_return, TupleReturn
            )

            res = c_func(*processed_args)

            # Sync back changes for TypedDict
            for target_dict, struct_obj in sync_backs:
                for field_name, _ in struct_obj._fields_:
                    target_dict[field_name] = getattr(struct_obj, field_name)

            if (
                is_named_tuple(sig.return_annotation)
                or get_origin(sig.return_annotation) in [tuple, typing_Tuple]
            ) and not is_ptr_return:
                # Construct Tuple/NamedTuple from registers
                flattened_res = [getattr(res, f"f{i}") for i in range(len(tuple_types))]
                return _unflatten_values(sig.return_annotation, flattened_res)

            if is_ptr_return:
                ret_ann_str = _get_type_name(
                    sig.return_annotation, type_mapping
                ).lower()
                raw_ann_str = str(sig.return_annotation).lower()
                if is_named_tuple(sig.return_annotation) or get_origin(
                    sig.return_annotation
                ) in [tuple, typing_Tuple]:
                    # Construct Tuple/NamedTuple
                    flattened_res = [
                        getattr(ret_struct, f"f{i}") for i in range(len(tuple_types))
                    ]
                    return _unflatten_values(sig.return_annotation, flattened_res)
                elif "tuple" in raw_ann_str or (
                    ret_ann_str.startswith("(") and ret_ann_str.endswith(")")
                ):
                    return tuple(
                        getattr(ret_struct, f"f{i}") for i in range(len(tuple_types))
                    )
                else:
                    is_val_opt, inner_type = _is_value_optional(sig.return_annotation)
                    if is_val_opt:
                        if ret_struct.has_value:
                            val_obj = ret_struct.value
                            if getattr(inner_type, "__lirien_struct__", False):
                                struct_inst = inner_type.__new__(inner_type)
                                struct_inst._ctypes_obj = val_obj
                                return struct_inst
                            return val_obj
                        else:
                            return None
                    elif getattr(sig.return_annotation, "__lirien_struct__", False):
                        struct_inst = sig.return_annotation.__new__(
                            sig.return_annotation
                        )
                        struct_inst._ctypes_obj = ret_struct
                        return struct_inst
                    else:
                        # SIMD Return
                        return ret_struct

            return _wrap_return_value(
                res, sig.return_annotation, type_mapping, sig, args
            )

        # Check preconditions
        preconds = getattr(wrapper, "__lirien_preconditions__", []) + getattr(
            func, "__lirien_preconditions__", []
        )
        if preconds or assert_preconds:
            bound = sig.bind(*args)
            bound.apply_defaults()
            for prec in preconds:
                prec_sig = inspect.signature(prec)
                prec_args = {
                    name: bound.arguments[name]
                    for name in prec_sig.parameters
                    if name in bound.arguments
                }
                if not prec(**prec_args):
                    raise ValueError(
                        f"Runtime Precondition Violation for '{func.__name__}': "
                        f"Arguments do not satisfy precondition."
                    )
            eval_locals = bound.arguments.copy()
            for lambda_fn in assert_preconds:
                if not lambda_fn(**eval_locals):
                    raise AssertionError("Precondition violation: assert failed.")

        res_val = _call_impl(*args)

        # Check postconditions
        postconds = getattr(wrapper, "__lirien_postconditions__", []) + getattr(
            func, "__lirien_postconditions__", []
        )
        if postconds or assert_postconds:
            bound = sig.bind(*args)
            bound.apply_defaults()
            for post in postconds:
                post_sig = inspect.signature(post)
                params = list(post_sig.parameters.keys())
                post_args = {}
                if params:
                    ret_name = params[0]
                    post_args[ret_name] = res_val
                    for name in params[1:]:
                        if name in bound.arguments:
                            post_args[name] = bound.arguments[name]
                if not post(**post_args):
                    raise ValueError(
                        f"Runtime Postcondition Violation for '{func.__name__}': "
                        f"Return value does not satisfy postcondition."
                    )
            for lambda_fn, post_params, ret_name in assert_postconds:
                eval_locals = bound.arguments.copy()
                if ret_name:
                    eval_locals[ret_name] = res_val
                eval_locals["res"] = res_val
                eval_locals["return_val"] = res_val
                call_args = {k: eval_locals[k] for k in post_params if k in eval_locals}
                if not lambda_fn(**call_args):
                    raise AssertionError("Postcondition violation: assert failed.")

        return res_val

    print(f"[Lirien] JIT compiled '{func.__name__}' successfully.")
    wrapper.__lirien_jit__ = True
    wrapper.__lirien_ptr__ = code_ptr
    wrapper.__name__ = func.__name__
    wrapper.__lirien_closure__ = False
    return wrapper
