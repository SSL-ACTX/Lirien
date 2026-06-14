import inspect
import ast
import sys
import types
import ctypes
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    TypeVar,
    Union,
    get_origin,
    get_args,
    Annotated,
)
from .types import Box, TYPE_MAP


def is_named_tuple(cls):
    """Check if a class is a subclass of typing.NamedTuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and hasattr(cls, "_fields")
        and hasattr(cls, "__annotations__")
    )


def _get_type_name(ty: Any, type_mapping: Dict[str, str] = None) -> str:
    """Consistently convert a Python-side type to its Lila IR string representation."""
    if type_mapping:
        if isinstance(ty, str) and ty in type_mapping:
            return _get_type_name(type_mapping[ty], type_mapping)
        if hasattr(ty, "__name__") and ty.__name__ in type_mapping:
            return _get_type_name(type_mapping[ty.__name__], type_mapping)
        if str(ty) in type_mapping:
            return _get_type_name(type_mapping[str(ty)], type_mapping)

    if ty is None or ty is type(None):
        return "None"

    if is_named_tuple(ty):
        return ty.__name__

    if isinstance(ty, (list, tuple)):
        return "(" + ", ".join(_get_type_name(t, type_mapping) for t in ty) + ")"

    # Handle TypeVar
    if isinstance(ty, TypeVar):
        if type_mapping and ty.__name__ in type_mapping:
            return type_mapping[ty.__name__]
        return ty.__name__

    # Handle Refined types
    if hasattr(ty, "base_type") and hasattr(ty, "predicate"):
        base_name = _get_type_name(ty.base_type, type_mapping)
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

        # Handle Literal
        if "literal" in origin_str:
            args = getattr(ty, "__args__", [])
            if args:
                return f"Literal[{args[0]}]"

        inner = ty.__metadata__[0]

        # Handle Tensor
        if "tensor" in origin_str:
            base_ty, shape = inner
            shape_str = ", ".join(
                f'"{s}"' if isinstance(s, str) else ("..." if s is Ellipsis else str(s))
                for s in shape
            )
            return f"Tensor[{_get_type_name(base_ty, type_mapping)}, {shape_str}]"

        # Check for Tuple in Annotated
        if "tuple" in origin_str:
            if isinstance(inner, (list, tuple)):
                return (
                    "("
                    + ", ".join(_get_type_name(t, type_mapping) for t in inner)
                    + ")"
                )
            return f"({_get_type_name(inner, type_mapping)})"

        # Handle Higher-Order types in Annotated
        if any(h in origin_str for h in ["callable", "fnpointer", "closure"]):
            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                arg_tys, ret_ty = inner
                if isinstance(arg_tys, (list, tuple)):
                    arg_str = (
                        "["
                        + ", ".join(_get_type_name(t, type_mapping) for t in arg_tys)
                        + "]"
                    )
                else:
                    arg_str = f"[{_get_type_name(arg_tys, type_mapping)}]"

                kind = "FnPointer"
                if "closure" in origin_str or "callable" in origin_str:
                    kind = "Closure"
                return f"{kind}[{arg_str}, {_get_type_name(ret_ty, type_mapping)}]"

        if "buffer" in origin_str:
            inner_name = _get_type_name(inner, type_mapping)
            return f"Buffer[{inner_name}]"
        if "box" in origin_str:
            inner_name = _get_type_name(inner, type_mapping)
            return f"Box[{inner_name}]"
        if "sizedarray" in origin_str:
            # SizedArray metadata usually has (base_type, size)
            if isinstance(inner, tuple) and len(inner) == 2:
                return (
                    f"SizedArray[{_get_type_name(inner[0], type_mapping)}, {inner[1]}]"
                )

        # Fallback for standard Annotated[T, metadata]
        return _get_type_name(origin, type_mapping)

    # Handle standard Tuples
    origin = getattr(ty, "__origin__", None)
    # Handle Union types (including Optional)
    origin = get_origin(ty)
    if (
        origin is Union
        or (
            hasattr(sys.modules.get("typing"), "_UnionGenericAlias")
            and isinstance(ty, sys.modules.get("typing")._UnionGenericAlias)
        )
        or (sys.version_info >= (3, 10) and origin is types.UnionType)
    ):
        args = get_args(ty)
        # Check for Box[T] | None (Optional[Box[T]])
        if len(args) == 2:
            box_ty = None
            has_none = False
            for arg in args:
                arg_origin = get_origin(arg)
                if arg_origin is Annotated:
                    inner_args = get_args(arg)
                    if inner_args and inner_args[0] is Box:
                        box_ty = arg
                elif arg is type(None) or arg is None:
                    has_none = True

            if box_ty and has_none:
                inner_name = _get_type_name(box_ty, type_mapping)
                return f"Nullable[{inner_name}]"

    if origin:
        origin_str = str(origin)
        if "Literal" in origin_str:
            args = getattr(ty, "__args__", [])
            if args:
                return f"Literal[{args[0]}]"

    if origin is tuple or origin is Tuple or "tuple" in str(ty).lower():
        args = getattr(ty, "__args__", [])
        if args:
            return "(" + ", ".join(_get_type_name(t, type_mapping) for t in args) + ")"
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
            arg_str = (
                "[" + ", ".join(_get_type_name(t, type_mapping) for t in arg_tys) + "]"
            )
            return f"{'Closure' if 'closure' in str(ty).lower() else 'FnPointer'}[{arg_str}, {_get_type_name(ret_ty, type_mapping)}]"

    if hasattr(ty, "__name__"):
        return ty.__name__

    return str(ty).split(".")[-1].replace("'>", "").lower()


def _discover_types(
    func: Callable, initial_struct_layouts: Dict, type_mapping: Dict[str, str] = None
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
            if (
                "lila/compiler.py" not in f_tmp.f_code.co_filename
                and "lila/decorators.py" not in f_tmp.f_code.co_filename
                and "lila/signatures.py" not in f_tmp.f_code.co_filename
                and "lila/ffi.py" not in f_tmp.f_code.co_filename
            ):
                scope.update(f_tmp.f_locals)
                break
            f_tmp = f_tmp.f_back

        closure_vars = inspect.getclosurevars(func)
        scope.update(closure_vars.nonlocals)
        scope.update(closure_vars.globals)
    except:
        pass

    if type_mapping:
        for val in type_mapping.values():
            if hasattr(val, "__name__"):
                scope[val.__name__] = val

    for name, obj in scope.items():
        if is_named_tuple(obj):
            if name not in struct_layouts:
                struct_layouts[name] = [
                    (
                        f_name,
                        _get_type_name(
                            obj.__annotations__.get(f_name, "i64"), type_mapping
                        ),
                    )
                    for f_name in obj._fields
                ]
            # Tag it so we can separate it later
            obj.__lila_named_tuple__ = True
        elif getattr(obj, "__lila_struct__", False) and name not in struct_layouts:
            struct_layouts[name] = [
                (f_name, _get_type_name(f_ty, type_mapping))
                for f_name, f_ty in obj.__lila_fields__
            ]
        elif getattr(obj, "__lila_enum__", False) and name not in enum_layouts:
            layout = []
            for v_name in getattr(obj, "__lila_variants__", []):
                v_ty = obj.__lila_variant_types__[v_name]
                v_ty_name = _get_type_name(v_ty, type_mapping)
                layout.append((v_name, v_ty_name))
                if (
                    v_ty is not None
                    and hasattr(v_ty, "__lila_fields__")
                    and v_ty_name not in struct_layouts
                ):
                    struct_layouts[v_ty_name] = [
                        (f_name, _get_type_name(f_ty, type_mapping))
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
                    base_ty_name = (
                        f"{origin_name}[{_get_type_name(item_ty, type_mapping)}]"
                    )
                else:
                    base_ty_name = getattr(base_ty, "__name__", str(base_ty))

                type_aliases[name] = f"Refined[{base_ty_name}, {pred_src}]"
            except (TypeError, AttributeError, SyntaxError, OSError):
                pass
    return struct_layouts, enum_layouts, type_aliases


def _value_to_lila_type(val: Any) -> str:
    """Map a runtime Python value to its corresponding Lila type name."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "i64"
    if isinstance(val, float):
        return "f64"

    # Handle Lila-wrapped objects (Structs, Enums, SizedArrays)
    if (
        hasattr(val, "__lila_struct__")
        or hasattr(val, "__lila_enum__")
        or is_named_tuple(type(val))
    ):
        return val.__class__.__name__

    # SIMD and common wrappers
    name = val.__class__.__name__
    if name in ["f32x4", "i32x4", "f64x2", "i64x2", "i8x16", "u8x16", "i16x8", "u16x8"]:
        return name

    # Handle NumPy-like arrays or memoryviews (Buffer Protocol)
    if hasattr(val, "dtype"):
        dt = str(val.dtype).lower()
        if "float32" in dt:
            return "f32"
        if "float64" in dt:
            return "f64"
        if "int64" in dt:
            return "i64"
        if "uint64" in dt:
            return "u64"
        if "int32" in dt:
            return "i32"
        if "uint32" in dt:
            return "u32"
        if "int16" in dt:
            return "i16"
        if "uint16" in dt:
            return "u16"
        if "int8" in dt:
            return "i8"
        if "uint8" in dt:
            return "u8"
        if "bool" in dt:
            return "bool"
    elif isinstance(val, memoryview):
        fmt = val.format
        if fmt == "f":
            return "f32"
        if fmt == "d":
            return "f64"
        if fmt in ("b", "c"):
            return "i8"
        if fmt == "B":
            return "u8"
        if fmt == "h":
            return "i16"
        if fmt == "H":
            return "u16"
        if fmt in ("i", "l"):
            return "i32"
        if fmt in ("I", "L"):
            return "u32"
        if fmt == "q":
            return "i64"
        if fmt == "Q":
            return "u64"
        if fmt == "?":
            return "bool"

    # Fallback for Boxed values
    if isinstance(val, Box):
        return _value_to_lila_type(val.value)

    if isinstance(val, ctypes.Array):
        # Map ctypes array to Buffer[T]
        elt_cty = val._type_
        elt_lila = "i64"
        for lila_name, cty in TYPE_MAP.items():
            if cty == elt_cty:
                elt_lila = lila_name
                break
        return f"Buffer[{elt_lila}]"

    if isinstance(val, ctypes._Pointer):
        return "pointer"

    if isinstance(val, ctypes.Structure) and not hasattr(val, "__lila_struct__"):
        return val.__class__.__name__

    return "i64"


def _find_typevars(ann, found):
    """Recursively find all TypeVars in a type annotation."""
    if isinstance(ann, TypeVar):
        found.add(ann)
    elif hasattr(ann, "__args__"):
        for arg in ann.__args__:
            _find_typevars(arg, found)


def _get_all_typevars(sig: inspect.Signature):
    """Extract all TypeVars from a function signature."""
    tvars = set()
    for param in sig.parameters.values():
        _find_typevars(param.annotation, tvars)
    _find_typevars(sig.return_annotation, tvars)
    return tvars


class TypeSubstitutor(ast.NodeTransformer):
    """AST visitor to replace TypeVar names with concrete type names."""

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            val = self.mapping[node.id]
            name = getattr(val, "__name__", str(val))
            return ast.Name(id=name, ctx=node.ctx)
        return self.generic_visit(node)
