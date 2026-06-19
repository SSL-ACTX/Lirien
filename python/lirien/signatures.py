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
    TypeVarTuple,
    Union,
    get_origin,
    get_args,
    Annotated,
)
from .types.base import TYPE_MAP
from .types.memory import Box


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


def _get_type_name(ty: Any, type_mapping: Dict[str, Any] = None) -> str:
    """Consistently convert a Python-side type to its Lirien IR string representation."""
    if getattr(ty, "__lirien_specialized__", False):
        return ty.__name__

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
                # Fallback for older tensor format
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

        # Fallback for standard Annotated[T, metadata]
        return _get_type_name(origin, type_mapping)

    # Handle subscripted generics (e.g. List[T])
    origin = get_origin(ty)

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

        # Handle Higher-Order types
        if (
            "fnpointer" in str(ty).lower()
            or "closure" in str(ty).lower()
            or "callable" in str(ty).lower()
        ):
            if len(args) >= 2:
                arg_tys, ret_ty = args[0], args[1]
                target_name = args[2] if len(args) > 2 else None

                # Ensure arg_tys is a list-like
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

        # Standard generic formatting
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
    initial_struct_layouts: Dict,
    type_mapping: Dict[str, Any] = None,
    initial_enum_layouts: Dict = None,
    initial_typed_dict_layouts: Dict = None,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Scan for struct layouts, enum layouts, and type aliases referenced in the function's scope."""
    struct_layouts = initial_struct_layouts.copy() if initial_struct_layouts else {}
    enum_layouts = initial_enum_layouts.copy() if initial_enum_layouts else {}
    typed_dict_layouts = (
        initial_typed_dict_layouts.copy() if initial_typed_dict_layouts else {}
    )
    type_aliases = {}

    # Combine globals and closure variables
    scope = func.__globals__.copy()
    try:
        # Also try to get caller's locals to find types defined in the same function
        f_tmp = inspect.currentframe()
        while f_tmp:
            if (
                "lirien/compiler.py" not in f_tmp.f_code.co_filename
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
            obj.__lirien_named_tuple__ = True
        elif is_typed_dict(obj) and name not in typed_dict_layouts:
            # It's a TypedDict
            typed_dict_layouts[name] = [
                (
                    f_name,
                    _get_type_name(f_ty, type_mapping),
                )
                for f_name, f_ty in obj.__annotations__.items()
            ]

            # Generate a ctypes structure for interop
            fields = []
            for f_name, f_ty in obj.__annotations__.items():
                ty_name = _get_type_name(f_ty, type_mapping).lower()
                # Find the best match in TYPE_MAP
                cty = TYPE_MAP.get("i64")  # default
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

    # Handle Lirien-wrapped objects (Structs, Enums, SizedArrays)
    if (
        hasattr(val, "__lirien_struct__")
        or hasattr(val, "__lirien_enum__")
        or is_named_tuple(type(val))
        or hasattr(val, "__lirien_specialized__")
    ):
        if hasattr(val, "__lirien_specialized__"):
            return val.__name__
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
        return _value_to_lirien_type(val.value)

    if isinstance(val, ctypes.Array):
        # Map ctypes array to Buffer[T]
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
    from .types.arithmetic import TypeExpr

    if found is None:
        found = set()

    if isinstance(ann, (TypeVar, TypeVarTuple)) or hasattr(ann, "__lirien_typevar__"):
        found.add(ann)
        return found

    if isinstance(ann, TypeExpr):
        for arg in ann.args:
            _find_typevars(arg, found)
        return found

    # Handle Unpack (for TypeVarTuple)
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


class TypeSubstitutor(ast.NodeTransformer):
    """AST visitor to replace TypeVar names with concrete type names or literals."""

    def __init__(self, mapping: Dict[str, Any]):
        self.mapping = mapping

    def visit_Call(self, node):
        # We MUST NOT call generic_visit first if we want to replace the whole Call node
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                name = node.args[0].id
                if name in self.mapping:
                    val = self.mapping[name]
                    if isinstance(val, (list, tuple)):
                        return ast.Constant(value=len(val))
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        # Constant fold arithmetic ops if both sides are now constants
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            l, r = node.left.value, node.right.value
            if isinstance(l, (int, float)) and isinstance(r, (int, float)):
                res = None
                if isinstance(node.op, ast.Add):
                    res = l + r
                elif isinstance(node.op, ast.Sub):
                    res = l - r
                elif isinstance(node.op, ast.Mult):
                    res = l * r
                elif isinstance(node.op, ast.FloorDiv):
                    res = l // r
                elif isinstance(node.op, ast.Div):
                    res = l / r
                elif isinstance(node.op, ast.Mod):
                    res = l % r
                elif isinstance(node.op, ast.Pow):
                    res = l**r

                if res is not None:
                    return ast.Constant(value=res)
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.operand, ast.Constant):
            val = node.operand.value
            if isinstance(val, (int, float)):
                if isinstance(node.op, ast.USub):
                    return ast.Constant(value=-val)
                elif isinstance(node.op, ast.UAdd):
                    return ast.Constant(value=val)
        return node

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.mapping:
            val = self.mapping[node.value.id]
            if getattr(val, "__lirien_specialized__", False):
                return ast.Name(id=val.__name__, ctx=node.ctx)

        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in self.mapping:
            val = self.mapping[node.id]
            if isinstance(val, int):
                return ast.Constant(value=val)
            name = getattr(val, "__name__", str(val))
            return ast.Name(id=name, ctx=node.ctx)
        return self.generic_visit(node)
