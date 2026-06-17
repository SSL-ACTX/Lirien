import inspect
import ast
import textwrap
import os
from typing import (
    Callable,
    TypeVar,
    Any,
    Dict,
    Tuple,
    get_origin,
    get_args,
    get_overloads,
    Annotated,
)
from . import lila_bridge
from .types.memory import Buffer, Box, Tensor, SizedArray
from .signatures import (
    _get_type_name,
    _discover_types,
    _get_all_typevars,
    TypeSubstitutor,
    _value_to_lila_type,
    is_named_tuple,
)
from .ffi import (
    _map_ctypes_arguments,
    _handle_pointer_return,
    _create_wrapper,
)

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

    if target_func_name == "<lambda>":
        target_func_name = "lila_lambda"

    if class_name or func.__name__ == "<lambda>":
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


def _has_ellipsis(ann):
    """Check if a type annotation contains an Ellipsis (...)."""
    if ann is Ellipsis:
        return True
    # Handle Annotated types
    if hasattr(ann, "__metadata__"):
        metadata = ann.__metadata__
        if metadata:
            # metadata[0] can be Ellipsis (Buffer[...]) or a tuple (Tensor[T, ...])
            if any(_has_ellipsis(m) for m in metadata):
                return True
            if isinstance(metadata[0], (list, tuple)):
                if any(_has_ellipsis(m) for m in metadata[0]):
                    return True
                # Tensor/SizedArray nested ellipsis check
                if len(metadata[0]) > 1 and isinstance(metadata[0][1], (list, tuple)):
                    if any(m is Ellipsis for m in metadata[0][1]):
                        return True
    if hasattr(ann, "__args__"):
        return any(_has_ellipsis(arg) for arg in ann.__args__)
    if isinstance(ann, (list, tuple)):
        return any(_has_ellipsis(arg) for arg in ann)
    return False


def _has_protocol(ann):
    """Check if a type annotation is a typing.Protocol."""
    return getattr(ann, "_is_protocol", False)


class MonomorphizedFunction:
    """Handles lazy monomorphization of generic functions using TypeVars."""

    def __init__(
        self,
        func,
        typevars,
        strict,
        log_level,
        struct_layouts,
        class_name=None,
        method_name=None,
        timeout=5000,
    ):
        self.func = func
        self.__code__ = func.__code__
        self.typevars = typevars  # Set of TypeVar objects
        self.strict = strict
        self.log_level = log_level
        self.struct_layouts = struct_layouts
        self.class_name = class_name
        self.method_name = method_name
        self.timeout = timeout
        self.cache = {}
        self.sig = inspect.signature(func)
        self.__lila_jit__ = True

    def _match_typevars(
        self, annotation: Any, val: Any, mapping: Dict[str, Any], param_name: str = None
    ):
        """Recursively match TypeVars and Protocols in the annotation against the runtime value."""
        # 1. Base case: annotation is a TypeVar
        from typing import TypeVar, TypeVarTuple

        if isinstance(annotation, (TypeVar, TypeVarTuple)):
            name = annotation.__name__
            if name not in mapping:
                # Support Const Generics: match integers or tuples of integers to TypeVars
                if isinstance(val, (int, tuple, list)):
                    mapping[name] = val
                    return

                # Store the actual class if it's a Lila object or NamedTuple
                cls = val.__class__
                if (
                    hasattr(cls, "__lila_struct__")
                    or hasattr(cls, "__lila_enum__")
                    or is_named_tuple(cls)
                ):
                    mapping[name] = cls
                else:
                    mapping[name] = _value_to_lila_type(val)
            return

        # Handle Protocol
        if _has_protocol(annotation):
            name = annotation.__name__
            if name not in mapping:
                mapping[name] = (
                    val.__class__
                    if hasattr(val, "__lila_struct__")
                    else _value_to_lila_type(val)
                )
            return

        # Handle Higher-Order types (Callable, Closure, FnPointer)
        if _has_callable(annotation):
            if hasattr(val, "__lila_jit__") and param_name:
                mapping[f"__callable_{param_name}"] = val
            return

        # 2. Handle Annotated types (Buffer, Tensor, Box, etc.)
        origin = get_origin(annotation)
        if origin is Annotated:
            args = get_args(annotation)
            actual_origin = args[0]
            metadata = args[1:]

            # Buffer[T] -> Annotated[Buffer, T]
            if actual_origin is Buffer:
                if metadata:
                    if metadata[0] is Ellipsis:
                        if param_name:
                            mapping[f"__ellipsis_{param_name}"] = [
                                _value_to_lila_type(val)
                            ]
                    else:
                        self._match_typevars(metadata[0], val, mapping, param_name)
                return

            # Tensor[T, shape] -> Annotated[Tensor, (T, shape)]
            if actual_origin is Tensor:
                if metadata and isinstance(metadata[0], tuple) and len(metadata[0]) > 0:
                    base_ty, shape = metadata[0]
                    self._match_typevars(base_ty, val, mapping, param_name)

                    actual_shape = getattr(val, "shape", ())
                    if any(s is Ellipsis for s in shape):
                        if param_name:
                            # Match ellipsis
                            try:
                                ellipsis_idx = shape.index(Ellipsis)
                                num_before = ellipsis_idx
                                num_after = len(shape) - ellipsis_idx - 1
                                if len(actual_shape) >= num_before + num_after:
                                    ellipsis_part = actual_shape[
                                        num_before : len(actual_shape) - num_after
                                    ]
                                    mapping[f"__ellipsis_{param_name}"] = list(
                                        ellipsis_part
                                    )
                            except ValueError:
                                pass
                    else:
                        # Match individual shape elements (Const Generics / Variadic)
                        actual_idx = 0
                        for s in shape:
                            s_origin = get_origin(s)
                            if s_origin is not None and "Unpack" in str(s_origin):
                                # It's a TypeVarTuple
                                s_args = get_args(s)
                                if s_args:
                                    # Calculate how many elements are left to match
                                    # (This assumes only one Unpack per shape for now, which is standard)
                                    num_others = len(shape) - 1
                                    num_unpack = len(actual_shape) - num_others
                                    unpack_val = actual_shape[
                                        actual_idx : actual_idx + num_unpack
                                    ]
                                    self._match_typevars(
                                        s_args[0], unpack_val, mapping, param_name
                                    )
                                    actual_idx += num_unpack
                            else:
                                if actual_idx < len(actual_shape):
                                    self._match_typevars(
                                        s, actual_shape[actual_idx], mapping, param_name
                                    )
                                    actual_idx += 1
                    return

            # Box[T] -> Annotated[Box, T]
            if actual_origin is Box:
                if metadata:
                    # Look inside the Box if val is a Box instance
                    inner_val = val.value if isinstance(val, Box) else val
                    self._match_typevars(metadata[0], inner_val, mapping, param_name)
                return

            # SizedArray[T, size] -> Annotated[SizedLilaArray, (T, size)]
            if actual_origin is SizedArray or (
                hasattr(actual_origin, "__name__")
                and "SizedLilaArray" in actual_origin.__name__
            ):
                if metadata and isinstance(metadata[0], tuple) and len(metadata[0]) > 0:
                    self._match_typevars(metadata[0][0], val, mapping, param_name)
                    if len(metadata[0]) > 1:
                        if metadata[0][1] is Ellipsis:
                            if hasattr(val, "__len__") and param_name:
                                mapping[f"__ellipsis_{param_name}"] = [len(val)]
                        else:
                            # Const generic size
                            if hasattr(val, "__len__"):
                                self._match_typevars(
                                    metadata[0][1], len(val), mapping, param_name
                                )
                return

        # 3. Handle standard generics (Tuples)
        if origin is tuple or origin is Tuple:
            args = get_args(annotation)
            if isinstance(val, (list, tuple)) and len(val) == len(args):
                for arg_ann, arg_val in zip(args, val):
                    self._match_typevars(arg_ann, arg_val, mapping, param_name)
            return

        # 4. Recurse for other generic types if needed (e.g. List[T])
        args = get_args(annotation)
        if args:
            for arg in args:
                self._match_typevars(arg, val, mapping, param_name)

    def __call__(self, *args, **kwargs):
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()

        mapping = {}
        for param_name, val in bound.arguments.items():
            param = self.sig.parameters[param_name]
            self._match_typevars(param.annotation, val, mapping, param_name)

        # Match return annotation if it has TypeVars or Ellipsis
        # (Though matching against 'None' won't do much unless we have return-type inference)
        self._match_typevars(self.sig.return_annotation, None, mapping, "return")

        # Create a stable cache key
        cache_key = tuple(
            sorted(
                (k, tuple(v) if isinstance(v, list) else v) for k, v in mapping.items()
            )
        )

        if cache_key not in self.cache:
            self.cache[cache_key] = self._specialize(mapping)

        return self.cache[cache_key](*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        import types

        return types.MethodType(self.__call__, instance)

    def _specialize(self, mapping):
        source, target_name = _prepare_source_and_name(
            self.func, self.class_name, self.method_name
        )

        # Specialized function name to avoid collisions
        suffix_parts = []
        for k, v in sorted(mapping.items()):
            if k.startswith("__ellipsis_"):
                if all(isinstance(x, int) for x in v):
                    suffix_parts.append("rank" + str(len(v)))
                else:
                    # For Buffer element types or other non-rank ellipsis
                    suffix_parts.extend([str(x) for x in v])
            elif k.startswith("__callable_"):
                # Use the target function's name
                name = getattr(v, "__name__", str(v))
                suffix_parts.append(name)
            else:
                if isinstance(v, (list, tuple)):
                    suffix_parts.append("rank" + str(len(v)))
                else:
                    name = getattr(v, "__name__", str(v))
                    suffix_parts.append(name)

        # Sanitize specialized name for Rust/Cranelift compatibility
        specialized_name = target_name + "_" + "_".join(suffix_parts)
        specialized_name = (
            specialized_name.replace("[", "_")
            .replace("]", "_")
            .replace(",", "_")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
        )

        tree = ast.parse(source)
        tree.body[0].name = specialized_name

        # Expand Ellipsis in AST
        class EllipsisExpander(ast.NodeTransformer):
            def __init__(self, mapping, scope):
                self.mapping = mapping
                self.scope = scope
                self.current_param = None

            def visit_arg(self, node):
                old_param = self.current_param
                self.current_param = node.arg
                node.annotation = (
                    self.visit(node.annotation) if node.annotation else None
                )
                self.current_param = old_param
                return node

            def visit_FunctionDef(self, node):
                # Visit args manually to set current_param
                node.args = self.visit(node.args)

                # Visit return annotation
                old_param = self.current_param
                self.current_param = "return"
                if node.returns:
                    node.returns = self.visit(node.returns)
                self.current_param = old_param

                # Visit body
                node.body = [self.visit(stmt) for stmt in node.body]
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    func_obj = self.scope.get(func_name)

                    if isinstance(func_obj, MonomorphizedFunction):
                        call_mapping = {}
                        sig = inspect.signature(func_obj.func)
                        params = list(sig.parameters.items())

                        for i, arg in enumerate(node.args):
                            if i < len(params):
                                p_name, _ = params[i]
                                if isinstance(arg, ast.Name):
                                    mapping_key = f"__callable_{arg.id}"
                                    if mapping_key in self.mapping:
                                        call_mapping[f"__callable_{p_name}"] = (
                                            self.mapping[mapping_key]
                                        )

                        if call_mapping:
                            callee_target_name = func_obj.func.__name__
                            callee_suffix_parts = []
                            for k, v in sorted(call_mapping.items()):
                                name = getattr(v, "__name__", str(v))
                                if name in ("jit_call", "wrapper", "lila_lambda"):
                                    ptr = getattr(v, "__lila_ptr__", 0)
                                    callee_suffix_parts.append(f"{name}_{hex(ptr)[2:]}")
                                else:
                                    callee_suffix_parts.append(name)

                            specialized_callee = (
                                f"{callee_target_name}_{'_'.join(callee_suffix_parts)}"
                            )
                            node.func.id = specialized_callee

                            func_obj._specialize(call_mapping)

                return node

            def visit_Subscript(self, node):
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id in (
                    "Tensor",
                    "SizedArray",
                    "Buffer",
                ):
                    # Handle single Ellipsis (Buffer[...])
                    if (
                        isinstance(node.slice, ast.Constant)
                        and node.slice.value is Ellipsis
                    ):
                        ellipsis_key = f"__ellipsis_{self.current_param}"
                        if ellipsis_key in self.mapping:
                            type_name = self.mapping[ellipsis_key][0]
                            node.slice = ast.Name(id=type_name, ctx=ast.Load())
                        return node

                    # Handle Tuple with Ellipsis or TypeVarTuple
                    if isinstance(node.slice, ast.Tuple):
                        new_elts = []
                        for elt in node.slice.elts:
                            if isinstance(elt, ast.Constant) and elt.value is Ellipsis:
                                ellipsis_key = f"__ellipsis_{self.current_param}"
                                if ellipsis_key in self.mapping:
                                    for dim in self.mapping[ellipsis_key]:
                                        new_elts.append(ast.Constant(value=dim))
                                else:
                                    # Fallback to first ellipsis found if mapping by name fails
                                    for k, v in self.mapping.items():
                                        if k.startswith("__ellipsis_"):
                                            for dim in v:
                                                new_elts.append(ast.Constant(value=dim))
                                            break
                            elif (
                                isinstance(elt, ast.Subscript)
                                and isinstance(elt.value, ast.Name)
                                and elt.value.id == "Unpack"
                            ):
                                # It's Unpack[Shape]
                                if isinstance(elt.slice, ast.Name):
                                    shape_name = elt.slice.id
                                    if shape_name in self.mapping:
                                        dims = self.mapping[shape_name]
                                        if isinstance(dims, (list, tuple)):
                                            for dim in dims:
                                                new_elts.append(ast.Constant(value=dim))
                            else:
                                new_elts.append(elt)
                        node.slice.elts = new_elts

                # Handle Callable/Closure/FnPointer specialization
                if isinstance(node.value, ast.Name) and node.value.id in (
                    "Callable",
                    "Closure",
                    "FnPointer",
                ):
                    callable_key = f"__callable_{self.current_param}"
                    if callable_key in self.mapping:
                        target_val = self.mapping[callable_key]
                        target_name = getattr(target_val, "__name__", str(target_val))
                        is_closure = getattr(target_val, "__lila_closure__", False)

                        # Specialize Callable to concrete type
                        if node.value.id == "Callable":
                            node.value.id = "Closure" if is_closure else "FnPointer"

                        # Only inject target_name if it's a specific function name
                        is_generic = (
                            target_name in ("jit_call", "wrapper")
                            or "at 0x" in target_name
                        )

                        if not is_generic:
                            if isinstance(node.slice, ast.Tuple):
                                if len(node.slice.elts) == 2:
                                    node.slice.elts.append(
                                        ast.Constant(value=target_name)
                                    )
                                elif len(node.slice.elts) > 2:
                                    node.slice.elts[2] = ast.Constant(value=target_name)
                            elif isinstance(
                                node.slice, (ast.List, ast.Constant, ast.Name)
                            ):
                                node.slice = ast.Tuple(
                                    elts=[node.slice, ast.Constant(value=target_name)],
                                    ctx=ast.Load(),
                                )

                return node

        tree = EllipsisExpander(mapping, self.func.__globals__).visit(tree)

        transformer = TypeSubstitutor(
            {k: v for k, v in mapping.items() if not k.startswith("__ellipsis_")}
        )
        tree = transformer.visit(tree)
        specialized_source = ast.unparse(tree)

        log_lvl, old_log = _setup_logging(self.log_level)
        try:
            struct_layouts, enum_layouts, type_aliases, typed_dict_layouts = (
                _discover_types(self.func, self.struct_layouts, mapping)
            )

            # Separate NamedTuple layouts
            named_tuple_layouts = {}
            scope = self.func.__globals__.copy()
            # Also check closure vars
            try:
                closure_vars = inspect.getclosurevars(self.func)
                scope.update(closure_vars.nonlocals)
                scope.update(closure_vars.globals)
            except:
                pass

            # Also add types from mapping
            for val in mapping.values():
                if hasattr(val, "__name__"):
                    scope[val.__name__] = val

            for name in list(struct_layouts.keys()):
                obj = scope.get(name)
                if obj and getattr(obj, "__lila_named_tuple__", False):
                    named_tuple_layouts[name] = struct_layouts.pop(name)

            code_ptr = lila_bridge.verify_and_compile(
                specialized_source,
                specialized_name,
                struct_layouts,
                enum_layouts,
                type_aliases,
                named_tuple_layouts,
                typed_dict_layouts,
                self.timeout,
            )
        except Exception as e:
            error_msg = format_verification_error(
                self.func.__name__, specialized_source, str(e)
            )
            if self.strict:
                raise VerificationError(error_msg) from e
            else:
                print(f"[Lila Warning] {error_msg}. Falling back to Python.")
                return self.func
        finally:
            _restore_logging(log_lvl, old_log)

        # Create specialized wrapper
        c_args, arg_map = _map_ctypes_arguments(self.sig, self.class_name, mapping)

        ret_ann = self.sig.return_annotation
        if hasattr(ret_ann, "__name__") and ret_ann.__name__ in mapping:
            ret_ann = mapping[ret_ann.__name__]
        elif str(ret_ann) in mapping:
            ret_ann = mapping[str(ret_ann)]

        is_ptr_return, TupleReturn, c_args, arg_map, tuple_types = (
            _handle_pointer_return(ret_ann, c_args, arg_map, mapping)
        )

        return _create_wrapper(
            self.func,
            code_ptr,
            c_args,
            arg_map,
            self.sig.replace(return_annotation=ret_ann),
            is_ptr_return,
            TupleReturn,
            tuple_types,
            mapping,
        )


class OverloadedFunction:
    """Handles runtime dispatch and lazy compilation for overloaded functions."""

    def __init__(
        self,
        func,
        overloads,
        strict,
        log_level,
        struct_layouts,
        class_name=None,
        method_name=None,
        timeout=5000,
    ):
        self.func = func
        self.__code__ = func.__code__
        self.overloads = overloads
        self.strict = strict
        self.log_level = log_level
        self.struct_layouts = struct_layouts
        self.class_name = class_name
        self.method_name = method_name
        self.timeout = timeout
        self.cache = {}
        self.base_sig = inspect.signature(func)
        self.__lila_jit__ = True

    def _match_overload(self, *args, **kwargs):
        """Find the first overload that matches the runtime argument types."""
        for overload_func in self.overloads:
            sig = inspect.signature(overload_func)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                match = True
                mapping = {}
                for param_name, val in bound.arguments.items():
                    param = sig.parameters[param_name]
                    if param.annotation is inspect.Parameter.empty:
                        continue

                    # Support 'self' in methods - we trust it matches if it's a method
                    if param_name == "self" and self.class_name:
                        continue

                    val_lila_type = _value_to_lila_type(val)
                    ann_name = _get_type_name(param.annotation)
                    val_name = _get_type_name(val_lila_type)

                    if ann_name.lower() != val_name.lower():
                        # Allow narrow matching: if sig says f32 and we have a float (f64), allow it.
                        # The JIT specialized build will handle the downcast.
                        is_numeric_match = False
                        a_low = ann_name.lower()
                        v_low = val_name.lower()
                        if a_low in ("f32", "f64") and v_low in ("f32", "f64"):
                            is_numeric_match = True
                        elif (
                            a_low
                            in ("i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64")
                            and v_low == "i64"
                        ):
                            is_numeric_match = True

                        if not is_numeric_match:
                            match = False
                            break
                    mapping[param_name] = param.annotation

                if match:
                    return overload_func, sig, mapping
            except TypeError:
                continue
        return None, None, None

    def __call__(self, *args, **kwargs):
        overload_func, sig, mapping = self._match_overload(*args, **kwargs)
        if not overload_func:
            arg_types = [_get_type_name(_value_to_lila_type(arg)) for arg in args]
            raise TypeError(
                f"No matching Lila overload found for '{self.func.__name__}' with argument types {arg_types}"
            )

        # Create a stable cache key based on the matched signature
        cache_key = tuple(sorted((k, _get_type_name(v)) for k, v in mapping.items()))
        # Also include return type in cache key
        ret_type_name = _get_type_name(sig.return_annotation)
        full_cache_key = cache_key + (("__return__", ret_type_name),)

        if full_cache_key not in self.cache:
            self.cache[full_cache_key] = self._specialize(sig, full_cache_key)

        return self.cache[full_cache_key](*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        import types

        return types.MethodType(self.__call__, instance)

    def _specialize(self, sig, cache_key):
        source, target_name = _prepare_source_and_name(
            self.func, self.class_name, self.method_name
        )

        suffix = "_".join(v for k, v in cache_key if not k.startswith("__"))
        specialized_name = f"{target_name}_{suffix}"
        # Sanitize specialized name for Rust/Cranelift compatibility
        specialized_name = (
            specialized_name.replace("[", "_")
            .replace("]", "_")
            .replace(",", "_")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
        )

        tree = ast.parse(source)
        func_def = tree.body[0]
        func_def.name = specialized_name

        # Inject annotations from the matched overload signature into the implementation AST
        class AnnotationInjector(ast.NodeTransformer):
            def __init__(self, sig):
                self.sig = sig

            def visit_FunctionDef(self, node):
                # Update parameters
                for arg in node.args.args:
                    if arg.arg in self.sig.parameters:
                        ann = self.sig.parameters[arg.arg].annotation
                        if ann is not inspect.Parameter.empty:
                            ann_name = _get_type_name(ann)
                            # Parse type name into AST node
                            try:
                                ann_node = ast.parse(ann_name).body[0].value
                                arg.annotation = ann_node
                            except:
                                # Fallback to Name if parsing fails for some reason
                                arg.annotation = ast.Name(id=ann_name, ctx=ast.Load())

                # Update return annotation
                if self.sig.return_annotation is not inspect.Signature.empty:
                    ret_name = _get_type_name(self.sig.return_annotation)
                    try:
                        ret_node = ast.parse(ret_name).body[0].value
                        node.returns = ret_node
                    except:
                        node.returns = ast.Name(id=ret_name, ctx=ast.Load())

                return node

        tree = AnnotationInjector(sig).visit(tree)
        specialized_source = ast.unparse(tree)

        log_lvl, old_log = _setup_logging(self.log_level)
        try:
            struct_layouts, enum_layouts, type_aliases, typed_dict_layouts = (
                _discover_types(self.func, self.struct_layouts)
            )

            # Separate NamedTuple layouts
            named_tuple_layouts = {}
            scope = self.func.__globals__.copy()
            # Also check closure vars
            try:
                closure_vars = inspect.getclosurevars(self.func)
                scope.update(closure_vars.nonlocals)
                scope.update(closure_vars.globals)
            except:
                pass

            for name in list(struct_layouts.keys()):
                obj = scope.get(name)
                if obj and getattr(obj, "__lila_named_tuple__", False):
                    named_tuple_layouts[name] = struct_layouts.pop(name)

            code_ptr = lila_bridge.verify_and_compile(
                specialized_source,
                specialized_name,
                struct_layouts,
                enum_layouts,
                type_aliases,
                named_tuple_layouts,
                typed_dict_layouts,
                self.timeout,
            )
        except Exception as e:
            error_msg = format_verification_error(
                self.func.__name__, specialized_source, str(e)
            )
            if self.strict:
                raise VerificationError(error_msg) from e
            else:
                print(f"[Lila Warning] {error_msg}. Falling back to Python.")
                return self.func
        finally:
            _restore_logging(log_lvl, old_log)

        # Map to ctypes and create wrapper
        c_args, arg_map = _map_ctypes_arguments(sig, self.class_name)
        is_ptr_return, TupleReturn, c_args, arg_map, tuple_types = (
            _handle_pointer_return(sig.return_annotation, c_args, arg_map)
        )

        return _create_wrapper(
            self.func,
            code_ptr,
            c_args,
            arg_map,
            sig,
            is_ptr_return,
            TupleReturn,
            tuple_types,
        )


def _has_callable(ann):
    """Check if a type annotation contains Callable, Closure, or FnPointer."""
    if ann is None:
        return False
    from .types import FnPointer, Closure

    if (
        isinstance(ann, (FnPointer, Closure))
        or "callable" in str(ann).lower()
        or "closure" in str(ann).lower()
        or "fnpointer" in str(ann).lower()
    ):
        return True
    if hasattr(ann, "__args__"):
        return any(_has_callable(arg) for arg in ann.__args__)
    return False


def verify(
    strict: bool = True,
    log_level: str = None,
    timeout: int = 5000,
    _struct_layouts: dict = None,
    _class_name: str = None,
    _method_name: str = None,
) -> Callable:
    """
    Decorator to trigger formal verification and JIT compilation.

    :param strict: If True, raises VerificationError on failure. If False, falls back to Python.
    :param log_level: Override LILA_LOG level (e.g., 'info', 'debug', 'warn').
    :param timeout: Verification timeout in milliseconds (default 5000).
    """

    # Handle the case where the decorator is used without parentheses: @verify
    if callable(strict) and log_level is None and _struct_layouts is None:
        func = strict
        # Re-call verify with defaults
        return verify(strict=True)(func)

    def decorator(func: T) -> T:
        overloads = get_overloads(func)
        if overloads:
            return OverloadedFunction(
                func,
                overloads,
                strict,
                log_level,
                _struct_layouts,
                _class_name,
                _method_name,
                timeout,
            )

        sig = inspect.signature(func)
        typevars = _get_all_typevars(sig)
        has_ellipsis = any(
            _has_ellipsis(p.annotation) for p in sig.parameters.values()
        ) or _has_ellipsis(sig.return_annotation)
        has_protocol = any(_has_protocol(p.annotation) for p in sig.parameters.values())
        has_callable = any(_has_callable(p.annotation) for p in sig.parameters.values())

        if typevars or has_ellipsis or has_protocol or has_callable:
            return MonomorphizedFunction(
                func,
                typevars,
                strict,
                log_level,
                _struct_layouts,
                _class_name,
                _method_name,
                timeout,
            )

        log_lvl, old_log = _setup_logging(log_level)
        source, target_func_name = _prepare_source_and_name(
            func, _class_name, _method_name
        )

        try:
            struct_layouts, enum_layouts, type_aliases, typed_dict_layouts = (
                _discover_types(func, _struct_layouts)
            )

            # Separate NamedTuple layouts from struct_layouts
            named_tuple_layouts = {}
            scope = func.__globals__.copy()
            # Also check closure vars
            try:
                closure_vars = inspect.getclosurevars(func)
                scope.update(closure_vars.nonlocals)
                scope.update(closure_vars.globals)
            except:
                pass

            for name in list(struct_layouts.keys()):
                obj = scope.get(name)
                if obj and getattr(obj, "__lila_named_tuple__", False):
                    named_tuple_layouts[name] = struct_layouts.pop(name)

            try:
                code_ptr = lila_bridge.verify_and_compile(
                    source,
                    target_func_name,
                    struct_layouts,
                    enum_layouts,
                    type_aliases,
                    named_tuple_layouts,
                    typed_dict_layouts,
                    timeout,
                )
            finally:
                _restore_logging(log_lvl, old_log)

            sig = inspect.signature(func)
            c_args, arg_map = _map_ctypes_arguments(sig, _class_name)
            is_ptr_return, TupleReturn, c_args, arg_map, tuple_types = (
                _handle_pointer_return(
                    sig.return_annotation, c_args, arg_map, type_aliases
                )
            )

            return _create_wrapper(
                func,
                code_ptr,
                c_args,
                arg_map,
                sig,
                is_ptr_return,
                TupleReturn,
                tuple_types,
                type_aliases,
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
