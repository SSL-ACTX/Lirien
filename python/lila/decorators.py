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
    Annotated,
)
from . import lila_bridge
from .types import Buffer, Box, Tensor
from .signatures import (
    _get_type_name,
    _discover_types,
    _get_all_typevars,
    TypeSubstitutor,
    _value_to_lila_type,
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
    ):
        self.func = func
        self.typevars = typevars  # Set of TypeVar objects
        self.strict = strict
        self.log_level = log_level
        self.struct_layouts = struct_layouts
        self.class_name = class_name
        self.method_name = method_name
        self.cache = {}
        self.sig = inspect.signature(func)

    def _match_typevars(self, annotation: Any, val: Any, mapping: Dict[str, str]):
        """Recursively match TypeVars in the annotation against the runtime value."""
        # 1. Base case: annotation is a TypeVar
        if isinstance(annotation, TypeVar):
            name = annotation.__name__
            if name not in mapping:
                mapping[name] = _value_to_lila_type(val)
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
                    self._match_typevars(metadata[0], val, mapping)
                return

            # Tensor[T, shape] -> Annotated[Tensor, (T, shape)]
            if actual_origin is Tensor:
                if metadata and isinstance(metadata[0], tuple) and len(metadata[0]) > 0:
                    self._match_typevars(metadata[0][0], val, mapping)
                return

            # Box[T] -> Annotated[Box, T]
            if actual_origin is Box:
                if metadata:
                    # Look inside the Box if val is a Box instance
                    inner_val = val.value if isinstance(val, Box) else val
                    self._match_typevars(metadata[0], inner_val, mapping)
                return

            # SizedArray[T, size] -> Annotated[SizedLilaArray, (T, size)]
            if (
                hasattr(actual_origin, "__name__")
                and "SizedLilaArray" in actual_origin.__name__
            ):
                if metadata and isinstance(metadata[0], tuple) and len(metadata[0]) > 0:
                    self._match_typevars(metadata[0][0], val, mapping)
                return

        # 3. Handle standard generics (Tuples)
        if origin is tuple or origin is Tuple:
            args = get_args(annotation)
            if isinstance(val, (list, tuple)) and len(val) == len(args):
                for arg_ann, arg_val in zip(args, val):
                    self._match_typevars(arg_ann, arg_val, mapping)
            return

        # 4. Recurse for other generic types if needed (e.g. List[T])
        args = get_args(annotation)
        if args:
            for arg in args:
                # We don't know how to destructure 'val' for unknown generics,
                # but we can at least try to find TypeVars in them.
                # If they are linked to other arguments, they might be resolved elsewhere.
                self._match_typevars(arg, val, mapping)

    def __call__(self, *args, **kwargs):
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()

        mapping = {}
        for param_name, val in bound.arguments.items():
            param = self.sig.parameters[param_name]
            self._match_typevars(param.annotation, val, mapping)

        # Create a stable cache key
        cache_key = tuple(sorted(mapping.items()))

        if cache_key not in self.cache:
            self.cache[cache_key] = self._specialize(mapping)

        return self.cache[cache_key](*args, **kwargs)

    def _specialize(self, mapping):
        source, target_name = _prepare_source_and_name(
            self.func, self.class_name, self.method_name
        )

        # Specialized function name to avoid collisions
        specialized_name = (
            target_name + "_" + "_".join(v for k, v in sorted(mapping.items()))
        )

        tree = ast.parse(source)
        tree.body[0].name = specialized_name
        transformer = TypeSubstitutor(mapping)
        tree = transformer.visit(tree)
        specialized_source = ast.unparse(tree)

        log_lvl, old_log = _setup_logging(self.log_level)
        try:
            struct_layouts, enum_layouts, type_aliases = _discover_types(
                self.func, self.struct_layouts, mapping
            )
            code_ptr = lila_bridge.verify_and_compile(
                specialized_source,
                specialized_name,
                struct_layouts,
                enum_layouts,
                type_aliases,
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
        is_ptr_return, TupleReturn, c_args, arg_map = _handle_pointer_return(
            self.sig.return_annotation, c_args, arg_map, mapping
        )

        tuple_types = []
        if is_ptr_return:
            if "tuple" in _get_type_name(self.sig.return_annotation, mapping).lower():
                if hasattr(self.sig.return_annotation, "__args__"):
                    tuple_types = self.sig.return_annotation.__args__
                else:
                    from .types import i64

                    tuple_types = [i64, i64]

        return _create_wrapper(
            self.func,
            code_ptr,
            c_args,
            arg_map,
            self.sig,
            is_ptr_return,
            TupleReturn,
            tuple_types,
            mapping,
        )


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
        sig = inspect.signature(func)
        typevars = _get_all_typevars(sig)

        if typevars:
            return MonomorphizedFunction(
                func,
                typevars,
                strict,
                log_level,
                _struct_layouts,
                _class_name,
                _method_name,
            )

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
