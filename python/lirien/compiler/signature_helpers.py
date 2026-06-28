import inspect
import sys
import types
import ctypes
from typing import (
    Any,
    Callable,
    Tuple,
    TypeVar,
    TypeVarTuple,
    Union,
    get_origin,
    get_args,
    Annotated,
    Final,
)
from ..types.base import TYPE_MAP
from ..types.memory import Box


def is_named_tuple(cls):
    """Check if a class is a subclass of typing.NamedTuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and hasattr(cls, "_fields")
        and hasattr(cls, "__annotations__")
    )


def is_typed_dict(cls):
    """Check if a class is a typing.TypedDict."""
    return hasattr(cls, "__annotations__") and hasattr(cls, "__total__")


def _get_refinement_parts(ann: Any) -> tuple[Any, Any]:
    """
    Extract (base_type, predicate) from a refinement type annotation.
    Supports both Refined[T, pred] and PEP 593 Annotated[T, pred] refinement format.
    """
    if hasattr(ann, "base_type") and hasattr(ann, "predicate"):
        return ann.base_type, ann.predicate

    if hasattr(ann, "__metadata__"):
        origin = getattr(ann, "__origin__", ann)
        inner = ann.__metadata__[0]
        if hasattr(inner, "__lirien_symbolic__") or (
            callable(inner) and not isinstance(inner, type)
        ):
            return origin, inner

    return None, None


def _clean_lambda_source(predicate: Any) -> str:
    """Extract a clean lambda source expression from a predicate using AST when possible."""
    if hasattr(predicate, "__lirien_symbolic__"):
        return str(predicate)
    if not callable(predicate):
        return str(predicate)
    try:
        import ast
        import inspect

        src = inspect.getsource(predicate).strip()
        # Try parsing the source block to find lambda nodes
        try:
            tree = ast.parse(src)
            lambdas = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Lambda):
                    lambdas.append(node)
            if lambdas:
                return ast.unparse(lambdas[0]).strip()
        except SyntaxError:
            pass

        # String-based fallback
        if "lambda" in src:
            start = src.find("lambda")
            src = src[start:]
            # Try parsing prefixes of src until it becomes a valid lambda expression.
            for i in range(len(src), 6, -1):
                try:
                    ast.parse(src[:i])
                    return src[:i].strip()
                except SyntaxError:
                    continue
        return src
    except Exception:
        return "None"


def _is_box_type(ann: Any) -> bool:
    """Helper to check if an annotation represents a Box pointer type (including forward refs)."""
    if isinstance(ann, str):
        s = ann.strip()
        return (
            s.startswith("Box[")
            or s.startswith("Box ")
            or s.startswith("Optional[Box[")
            or s.endswith("| None")
        )

    origin = get_origin(ann) or ann
    if origin is Box:
        return True
    if hasattr(origin, "__name__") and origin.__name__ == "Box":
        return True
    if origin is Annotated:
        args = get_args(ann)
        if args and _is_box_type(args[0]):
            return True
    if "box" in str(ann).lower():
        return True
    return False


def _get_type_name(ty: Any, type_mapping: dict[str, Any] = None) -> str:
    """Consistently convert a Python-side type to its Lirien IR string representation."""
    if getattr(ty, "__lirien_specialized__", False):
        return ty.__name__

    # Handle typing.NewType
    if hasattr(ty, "__supertype__"):
        return _get_type_name(ty.__supertype__, type_mapping)

    # Detect Optionals in strings and rewrite to Nullable name
    if isinstance(ty, str):
        s = ty.strip()
        is_optional = "optional" in s.lower() or ("|" in s and "none" in s.lower())
        if is_optional:
            if s.startswith("Optional[") and s.endswith("]"):
                inner = s[len("Optional[") : -1]
                return f"Nullable[{inner}]"
            elif "|" in s:
                parts = [p.strip() for p in s.split("|")]
                non_none_parts = [p for p in parts if p.lower() != "none"]
                if non_none_parts:
                    return f"Nullable[{non_none_parts[0]}]"

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

    # Handle Refined / Annotated refinement types
    base_ty, predicate = _get_refinement_parts(ty)
    if base_ty is not None and predicate is not None:
        base_name = _get_type_name(base_ty, type_mapping)
        pred_src = _clean_lambda_source(predicate)
        return f"Refined[{base_name}, {pred_src}]"

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

        # Handle Box
        if (
            origin is Box
            or (hasattr(origin, "__name__") and origin.__name__ == "Box")
            or "box" in origin_str
        ):
            return f"Box[{_get_type_name(inner, type_mapping)}]"

        # Handle Tensor
        if "tensor" in origin_str:
            if isinstance(inner, tuple) and len(inner) == 2:
                base_ty, shape = inner
                shape_str = ", ".join(
                    f'"{s}"'
                    if isinstance(s, str)
                    else ("..." if s is Ellipsis else str(s))
                    for s in shape
                )
                return f"Tensor[{_get_type_name(base_ty, type_mapping)}, {shape_str}]"
            else:
                base_ty = inner
                return f"Tensor[{_get_type_name(base_ty, type_mapping)}]"

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
            if isinstance(inner, tuple) and len(inner) == 2:
                return (
                    f"SizedArray[{_get_type_name(inner[0], type_mapping)}, {inner[1]}]"
                )

        return _get_type_name(origin, type_mapping)

    # Handle subscripted generics (e.g. List[T])
    origin = get_origin(ty)

    # Handle typing.Final
    if (
        origin is Final
        or (hasattr(origin, "__name__") and origin.__name__ == "Final")
        or "final" in str(origin).lower()
    ):
        args = get_args(ty)
        if args:
            return _get_type_name(args[0], type_mapping)

    # Handle Union types (including Optional)
    if (
        origin is Union
        or (
            hasattr(sys.modules.get("typing"), "_UnionGenericAlias")
            and isinstance(ty, sys.modules.get("typing")._UnionGenericAlias)
        )
        or (sys.version_info >= (3, 10) and origin is types.UnionType)
    ):
        args = get_args(ty)
        has_none = any(arg is type(None) or arg is None for arg in args)
        if has_none:
            non_none_args = [
                arg for arg in args if arg is not type(None) and arg is not None
            ]
            if non_none_args:
                inner_name = _get_type_name(non_none_args[0], type_mapping)
                return f"Nullable[{inner_name}]"

    if origin is not None:
        args = get_args(ty)
        origin_name = getattr(origin, "__name__", str(origin))
        if type_mapping and origin_name in type_mapping:
            specialized = type_mapping[origin_name]
            if getattr(specialized, "__lirien_specialized__", False):
                return specialized.__name__

        if origin is tuple or origin is Tuple or "tuple" in str(ty).lower():
            if args:
                return (
                    "(" + ", ".join(_get_type_name(t, type_mapping) for t in args) + ")"
                )
            return "(i64, i64)"  # Default

        if (
            "fnpointer" in str(ty).lower()
            or "closure" in str(ty).lower()
            or "callable" in str(ty).lower()
        ):
            if len(args) >= 2:
                arg_tys, ret_ty = args[0], args[1]
                target_name = args[2] if len(args) > 2 else None

                if not isinstance(arg_tys, (list, tuple)):
                    arg_tys = [arg_tys]

                arg_str = (
                    "["
                    + ", ".join(_get_type_name(t, type_mapping) for t in arg_tys)
                    + "]"
                )
                base = "Closure" if "closure" in str(ty).lower() else "FnPointer"
                if target_name:
                    return f'{base}[{arg_str}, {_get_type_name(ret_ty, type_mapping)}, "{target_name}"]'
                return f"{base}[{arg_str}, {_get_type_name(ret_ty, type_mapping)}]"

        if args:
            return (
                f"{origin_name}["
                + ", ".join(_get_type_name(t, type_mapping) for t in args)
                + "]"
            )
        return origin_name

    if hasattr(ty, "__name__"):
        return ty.__name__

    return str(ty).split(".")[-1].replace("'>", "").lower()


def _discover_types(
    func: Callable,
    initial_struct_layouts: dict,
    type_mapping: dict[str, Any] = None,
    initial_enum_layouts: dict = None,
    initial_typed_dict_layouts: dict = None,
) -> tuple[dict, dict, dict, dict]:
    """Scan for struct layouts, enum layouts, and type aliases referenced in the function's scope."""
    struct_layouts = initial_struct_layouts.copy() if initial_struct_layouts else {}
    enum_layouts = initial_enum_layouts.copy() if initial_enum_layouts else {}
    typed_dict_layouts = (
        initial_typed_dict_layouts.copy() if initial_typed_dict_layouts else {}
    )
    type_aliases = {}

    scope = func.__globals__.copy()
    try:
        f_tmp = inspect.currentframe()
        while f_tmp:
            if (
                "lirien/compiler/" not in f_tmp.f_code.co_filename
                and "lirien/compiler.py" not in f_tmp.f_code.co_filename
                and "lirien/decorators.py" not in f_tmp.f_code.co_filename
                and "lirien/signatures.py" not in f_tmp.f_code.co_filename
                and "lirien/ffi.py" not in f_tmp.f_code.co_filename
            ):
                scope.update(f_tmp.f_locals)
                break
            f_tmp = f_tmp.f_back

        closure_vars = inspect.getclosurevars(func)
        scope.update(closure_vars.nonlocals)
        scope.update(closure_vars.globals)
    except Exception:
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
            obj.__lirien_named_tuple__ = True
        elif is_typed_dict(obj) and name not in typed_dict_layouts:
            typed_dict_layouts[name] = [
                (
                    f_name,
                    _get_type_name(f_ty, type_mapping),
                )
                for f_name, f_ty in obj.__annotations__.items()
            ]

            fields = []
            for f_name, f_ty in obj.__annotations__.items():
                ty_name = _get_type_name(f_ty, type_mapping).lower()
                cty = TYPE_MAP.get("i64")
                for match_name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
                    if match_name in ty_name:
                        cty = TYPE_MAP[match_name]
                        break
                fields.append((f_name, cty))

            class TypedDictStruct(ctypes.Structure):
                _fields_ = fields

            obj.__lirien_ctypes__ = TypedDictStruct
            obj.__lirien_typed_dict__ = True
        elif getattr(obj, "__lirien_struct__", False) and name not in struct_layouts:
            struct_layouts[name] = [
                (f_name, _get_type_name(f_ty, type_mapping))
                for f_name, f_ty in obj.__lirien_fields__
            ]
        elif getattr(obj, "__lirien_enum__", False) and name not in enum_layouts:
            layout = []
            variants = getattr(obj, "__lirien_variant_types__", {})
            for v_name, v_ty in variants.items():
                v_ty_name = _get_type_name(v_ty, type_mapping)
                layout.append((v_name, v_ty_name))
                if (
                    v_ty is not None
                    and hasattr(v_ty, "__lirien_fields__")
                    and v_ty_name not in struct_layouts
                ):
                    struct_layouts[v_ty_name] = [
                        (f_name, _get_type_name(f_ty, type_mapping))
                        for f_name, f_ty in v_ty.__lirien_fields__
                    ]
            enum_layouts[name] = layout
        else:
            base_ty, predicate = _get_refinement_parts(obj)
            if base_ty is not None and predicate is not None:
                try:
                    pred_src = _clean_lambda_source(predicate)
                    if hasattr(base_ty, "__origin__") and hasattr(
                        base_ty, "__metadata__"
                    ):
                        origin_name = getattr(
                            base_ty.__origin__, "__name__", str(base_ty.__origin__)
                        )
                        item_ty = base_ty.__metadata__[0]
                        base_ty_name = (
                            f"{origin_name}[{_get_type_name(item_ty, type_mapping)}]"
                        )
                    else:
                        base_ty_name = getattr(base_ty, "__name__", str(base_ty))

                    type_aliases[name] = f"Refined[{base_ty_name}, {pred_src}]"
                except (TypeError, AttributeError, SyntaxError, OSError):
                    pass

    return struct_layouts, enum_layouts, type_aliases, typed_dict_layouts


def _value_to_lirien_type(val: Any) -> str:
    """Map a runtime Python value to its corresponding Lirien type name."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "i64"
    if isinstance(val, float):
        return "f64"

    if (
        hasattr(val, "__lirien_struct__")
        or hasattr(val, "__lirien_enum__")
        or is_named_tuple(type(val))
        or hasattr(val, "__lirien_specialized__")
    ):
        if hasattr(val, "__lirien_specialized__"):
            return val.__name__
        return val.__class__.__name__

    name = val.__class__.__name__
    if name in ["f32x4", "i32x4", "f64x2", "i64x2", "i8x16", "u8x16", "i16x8", "u16x8"]:
        return name

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

    if isinstance(val, Box):
        return _value_to_lirien_type(val.value)

    if isinstance(val, ctypes.Array):
        elt_cty = val._type_
        elt_lirien = "i64"
        for lirien_name, cty in TYPE_MAP.items():
            if cty == elt_cty:
                elt_lirien = lirien_name
                break
        return f"Buffer[{elt_lirien}]"

    if isinstance(val, ctypes._Pointer):
        return "pointer"

    if isinstance(val, ctypes.Structure) and not hasattr(val, "__lirien_struct__"):
        return val.__class__.__name__

    return "unknown"


def _get_all_typevars(sig: inspect.Signature) -> set:
    """Extract all TypeVars from a function signature."""
    tvars = set()
    for param in sig.parameters.values():
        _find_typevars(param.annotation, tvars)
    _find_typevars(sig.return_annotation, tvars)
    return tvars


def _find_typevars(ann: Any, found: set = None) -> set:
    """Recursively find all TypeVars, LirienTypeVars, and TypeVarTuples in a type annotation."""
    from typing import TypeVar
    from ..types.arithmetic import TypeExpr

    if found is None:
        found = set()

    if isinstance(ann, (TypeVar, TypeVarTuple)) or hasattr(ann, "__lirien_typevar__"):
        found.add(ann)
        return found

    if isinstance(ann, TypeExpr):
        for arg in ann.args:
            _find_typevars(arg, found)
        return found

    origin = get_origin(ann)
    if origin is not None and "Unpack" in str(origin):
        args = get_args(ann)
        if args:
            _find_typevars(args[0], found)
        return found

    if hasattr(ann, "__args__"):
        for arg in ann.__args__:
            _find_typevars(arg, found)

    if hasattr(ann, "__metadata__"):
        for arg in ann.__metadata__:
            _find_typevars(arg, found)

    if isinstance(ann, (list, tuple)):
        for arg in ann:
            _find_typevars(arg, found)

    return found
