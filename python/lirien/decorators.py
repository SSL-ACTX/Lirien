import inspect
import ast
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
from . import lirien_bridge
from .types.memory import Buffer, Box, Tensor, SizedArray
from .compiler import (
    TypeSubstitutor,
    EllipsisExpander,
    RefinementSanitizer,
    _get_type_name,
    _discover_types,
    _get_all_typevars,
    _value_to_lirien_type,
    _get_refinement_parts,
    _prepare_source_and_name,
    _has_protocol,
    _has_callable,
    _needs_monomorphization,
    is_named_tuple,
)
from .diagnostics import (
    VerificationError,
    format_verification_error,
    _setup_logging,
    _restore_logging,
    _is_verification_disabled,
)
from .ffi import (
    _map_ctypes_arguments,
    _handle_pointer_return,
    _create_wrapper,
)

T = TypeVar("T", bound=Callable)


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
        enum_layouts=None,
        named_tuple_layouts=None,
        typed_dict_layouts=None,
        verify=True,
    ):
        self.func = func
        self.__code__ = func.__code__
        self.typevars = typevars  # Set of TypeVar objects
        self.strict = strict
        self.log_level = log_level
        self.struct_layouts = struct_layouts
        self.enum_layouts = enum_layouts
        self.named_tuple_layouts = named_tuple_layouts
        self.typed_dict_layouts = typed_dict_layouts
        self.class_name = class_name
        self.method_name = method_name
        self.timeout = timeout
        self.verify = verify
        self.cache = {}
        self.sig = inspect.signature(func)
        self.__lirien_jit__ = True

    def _match_typevars(
        self, annotation: Any, val: Any, mapping: Dict[str, Any], param_name: str = None
    ):
        """Recursively match TypeVars and Protocols in the annotation against the runtime value."""
        # 1. Base case: annotation is a TypeVar
        from typing import TypeVar, TypeVarTuple
        from .types.arithmetic import TypeExpr

        if (
            isinstance(annotation, (TypeVar, TypeVarTuple))
            or hasattr(annotation, "__lirien_typevar__")
            or isinstance(annotation, TypeExpr)
        ):
            if isinstance(annotation, TypeExpr):
                # Try to evaluate it if mapping is already somewhat populated
                # but we usually match against inputs which are base vars.
                # If it's in the return type, we evaluate it after inputs are matched.
                return

            name = annotation.__name__
            if name not in mapping:
                # Support Const Generics: match integers or tuples of integers to TypeVars
                # Prevent NamedTuple instances from being mistaken for shape tuples
                if isinstance(val, (int, list)) or (
                    isinstance(val, tuple) and not is_named_tuple(val.__class__)
                ):
                    mapping[name] = val
                    return

                # Store the actual class if it's a Lirien object or NamedTuple
                cls = val.__class__
                if (
                    hasattr(cls, "__lirien_struct__")
                    or hasattr(cls, "__lirien_enum__")
                    or is_named_tuple(cls)
                ):
                    mapping[name] = cls
                else:
                    mapping[name] = _value_to_lirien_type(val)
            return

        # Handle Protocol
        if _has_protocol(annotation):
            name = annotation.__name__
            if name not in mapping:
                cls = val.__class__
                if (
                    hasattr(cls, "__lirien_struct__")
                    or hasattr(cls, "__lirien_enum__")
                    or is_named_tuple(cls)
                ):
                    mapping[name] = cls
                else:
                    mapping[name] = _value_to_lirien_type(val)
            return

        # Handle Higher-Order types (Callable, Closure, FnPointer)
        if _has_callable(annotation):
            if hasattr(val, "__lirien_jit__") and param_name:
                mapping[f"__callable_{param_name}"] = val
            return

        # Handle specialized Lirien types
        if getattr(annotation, "__lirien_specialized__", False):
            origin = getattr(annotation, "__lirien_origin__", None)
            if origin:
                name = origin.__name__
                if name not in mapping:
                    mapping[name] = annotation
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
                                _value_to_lirien_type(val)
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

            # SizedArray[T, size] -> Annotated[SizedLirienArray, (T, size)]
            if actual_origin is SizedArray or (
                hasattr(actual_origin, "__name__")
                and "SizedLirienArray" in actual_origin.__name__
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

    def _get_specialized_name(self, mapping):
        """Generate a stable, unique name for a specialized version of this function."""
        from .types.arithmetic import TypeExpr

        new_mapping = mapping.copy()
        for k, v in list(new_mapping.items()):
            if isinstance(v, TypeExpr):
                new_mapping[k] = v.evaluate(new_mapping)

        target_name = self.method_name if self.method_name else self.func.__name__
        if target_name == "<lambda>":
            target_name = "lirien_lambda"

        # Specialized function name to avoid collisions
        suffix_parts = []
        for k, v in sorted(new_mapping.items()):
            if k.startswith("__ellipsis_"):
                if all(isinstance(x, int) for x in v):
                    suffix_parts.append("rank" + str(len(v)))
                else:
                    # For Buffer element types or other non-rank ellipsis
                    suffix_parts.extend([str(x) for x in v])
            elif k.startswith("__callable_"):
                # Use the target function's name
                name = getattr(v, "__name__", str(v))
                if name in ("jit_call", "wrapper", "lirien_lambda"):
                    ptr = getattr(v, "__lirien_ptr__", 0)
                    suffix_parts.append(f"{name}_{hex(ptr)[2:]}")
                else:
                    suffix_parts.append(name)
            else:
                if isinstance(v, (list, tuple)):
                    suffix_parts.append("rank" + str(len(v)))
                else:
                    name = getattr(v, "__name__", str(v))
                    suffix_parts.append(name)

        # Sanitize specialized name for Rust/Cranelift compatibility
        specialized_name = target_name + "_" + "_".join(suffix_parts)
        import re

        specialized_name = re.sub(r"[^a-zA-Z0-9_]", "_", specialized_name)
        return re.sub(r"_+", "_", specialized_name).strip("_")

    def _specialize(self, mapping):
        # 0. Evaluate symbolic expressions (TypeExpr) in the mapping
        from .types.arithmetic import TypeExpr

        # Replace TypeExprs in mapping with their evaluated values
        new_mapping = mapping.copy()
        for k, v in list(new_mapping.items()):
            if isinstance(v, TypeExpr):
                new_mapping[k] = v.evaluate(new_mapping)

        source, _ = _prepare_source_and_name(
            self.func, self.class_name, self.method_name
        )
        specialized_name = self._get_specialized_name(new_mapping)

        tree = ast.parse(source)
        tree.body[0].name = specialized_name

        # Expand Ellipsis and Handle Nested Specialization in AST
        # Get full scope (globals + nonlocals)
        scope = self.func.__globals__.copy()
        try:
            cv = inspect.getclosurevars(self.func)
            scope.update(cv.nonlocals)
        except:
            pass

        tree = EllipsisExpander(new_mapping, scope, self).visit(tree)

        transformer_mapping = {
            k: v for k, v in new_mapping.items() if not k.startswith("__")
        }
        # Redirect recursive calls to the specialized version
        target_name = self.method_name if self.method_name else self.func.__name__
        transformer_mapping[target_name] = specialized_name

        # Also map the origin name to the specialized name for generic types
        for k, v in list(transformer_mapping.items()):
            if hasattr(v, "__name__") and "_" in v.__name__ and k in v.__name__:
                # It's likely a specialized class Opt_i64 for origin Opt
                # We want to replace 'Opt' with 'Opt_i64' in the AST
                pass  # Already handled if key is 'Opt'

        # Inject annotations and sanitize refinement types to prevent Rust-side syntax errors
        tree = RefinementSanitizer(inspect.signature(self.func)).visit(tree)

        transformer = TypeSubstitutor(transformer_mapping)
        tree = transformer.visit(tree)
        specialized_source = ast.unparse(tree)

        log_lvl, old_log = _setup_logging(self.log_level)
        try:
            struct_layouts, enum_layouts, type_aliases, typed_dict_layouts = (
                _discover_types(
                    self.func,
                    self.struct_layouts,
                    new_mapping,
                    initial_enum_layouts=self.enum_layouts,
                    initial_typed_dict_layouts=self.typed_dict_layouts,
                )
            )
            # Separate NamedTuple layouts
            named_tuple_layouts = (
                self.named_tuple_layouts.copy() if self.named_tuple_layouts else {}
            )
            scope = self.func.__globals__.copy()
            # Also check closure vars
            try:
                closure_vars = inspect.getclosurevars(self.func)
                scope.update(closure_vars.nonlocals)
                scope.update(closure_vars.globals)
            except:
                pass

            # Also add types from mapping
            for val in new_mapping.values():
                if hasattr(val, "__name__"):
                    scope[val.__name__] = val

            for name in list(struct_layouts.keys()):
                obj = scope.get(name)
                if obj and getattr(obj, "__lirien_named_tuple__", False):
                    named_tuple_layouts[name] = struct_layouts.pop(name)

            code_ptr = lirien_bridge.verify_and_compile(
                specialized_source,
                specialized_name,
                struct_layouts,
                enum_layouts,
                type_aliases,
                named_tuple_layouts,
                typed_dict_layouts,
                self.timeout,
                self.verify,
            )
        except Exception as e:
            error_msg = format_verification_error(
                self.func.__name__, specialized_source, str(e)
            )
            if self.strict:
                raise VerificationError(error_msg) from e
            else:
                print(f"[Lirien Warning] {error_msg}. Falling back to Python.")
                return self.func
        finally:
            _restore_logging(log_lvl, old_log)

        # Create specialized wrapper
        c_args, arg_map = _map_ctypes_arguments(self.sig, self.class_name, new_mapping)

        ret_ann = self.sig.return_annotation
        if hasattr(ret_ann, "__name__") and ret_ann.__name__ in new_mapping:
            ret_ann = new_mapping[ret_ann.__name__]
        elif str(ret_ann) in new_mapping:
            ret_ann = new_mapping[str(ret_ann)]

        is_ptr_return, TupleReturn, c_args, arg_map, tuple_types = (
            _handle_pointer_return(ret_ann, c_args, arg_map, new_mapping)
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
            new_mapping,
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
        verify=True,
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
        self.verify = verify
        self.cache = {}
        self.base_sig = inspect.signature(func)
        self.__lirien_jit__ = True

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

                    val_lirien_type = _value_to_lirien_type(val)
                    # Unwrap refinement annotations to base types for overload resolution matching
                    base_ty, predicate = _get_refinement_parts(param.annotation)
                    ann_to_check = base_ty if base_ty is not None else param.annotation
                    ann_name = _get_type_name(ann_to_check)
                    val_name = _get_type_name(val_lirien_type)

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

                    if predicate is not None and callable(predicate):
                        try:
                            res = predicate(val)
                            if not res:
                                match = False
                                break
                        except NameError:
                            pass
                        except Exception:
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
            arg_types = [_get_type_name(_value_to_lirien_type(arg)) for arg in args]
            raise TypeError(
                f"No matching Lirien overload found for '{self.func.__name__}' with argument types {arg_types}"
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
        import re

        specialized_name = re.sub(r"[^a-zA-Z0-9_]", "_", specialized_name)
        specialized_name = re.sub(r"_+", "_", specialized_name).strip("_")

        tree = ast.parse(source)
        func_def = tree.body[0]
        func_def.name = specialized_name

        # Inject annotations and sanitize refinement types to prevent Rust-side syntax errors
        tree = RefinementSanitizer(sig, sanitize_all_types=True).visit(tree)
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
                if obj and getattr(obj, "__lirien_named_tuple__", False):
                    named_tuple_layouts[name] = struct_layouts.pop(name)

            code_ptr = lirien_bridge.verify_and_compile(
                specialized_source,
                specialized_name,
                struct_layouts,
                enum_layouts,
                type_aliases,
                named_tuple_layouts,
                typed_dict_layouts,
                self.timeout,
                self.verify,
            )
        except Exception as e:
            error_msg = format_verification_error(
                self.func.__name__, specialized_source, str(e)
            )
            if self.strict:
                raise VerificationError(error_msg) from e
            else:
                print(f"[Lirien Warning] {error_msg}. Falling back to Python.")
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
    verify: bool = True,
    _struct_layouts: dict = None,
    _enum_layouts: dict = None,
    _named_tuple_layouts: dict = None,
    _typed_dict_layouts: dict = None,
    _class_name: str = None,
    _method_name: str = None,
) -> Callable:
    """
    Decorator to trigger formal verification and JIT compilation.

    :param strict: If True, raises VerificationError on failure. If False, falls back to Python.
    :param log_level: Override LILA_LOG level (e.g., 'info', 'debug', 'warn').
    :param timeout: Verification timeout in milliseconds (default 5000).
    :param verify: If True, performs Z3 verification. If False, JITs directly without verification.
    """
    if _is_verification_disabled():
        verify = False

    # Handle the case where the decorator is used without parentheses: @verify
    if (
        callable(strict)
        and log_level is None
        and _struct_layouts is None
        and _enum_layouts is None
    ):
        func = strict
        # Re-call verify with defaults
        return globals()["verify"](strict=True, verify=verify)(func)

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
                verify=verify,
            )

        sig = inspect.signature(func)
        typevars = _get_all_typevars(sig)

        should_monomorphize = (
            typevars
            or any(
                _needs_monomorphization(p.annotation) for p in sig.parameters.values()
            )
            or _needs_monomorphization(sig.return_annotation)
        )

        if should_monomorphize:
            return MonomorphizedFunction(
                func,
                typevars,
                strict,
                log_level,
                _struct_layouts,
                _class_name,
                _method_name,
                timeout,
                enum_layouts=_enum_layouts,
                named_tuple_layouts=_named_tuple_layouts,
                typed_dict_layouts=_typed_dict_layouts,
                verify=verify,
            )

        log_lvl, old_log = _setup_logging(log_level)
        source, target_func_name = _prepare_source_and_name(
            func, _class_name, _method_name
        )

        try:
            struct_layouts, enum_layouts, type_aliases, typed_dict_layouts = (
                _discover_types(
                    func,
                    _struct_layouts,
                    initial_enum_layouts=_enum_layouts,
                    initial_typed_dict_layouts=_typed_dict_layouts,
                )
            )

            # Separate NamedTuple layouts from struct_layouts
            named_tuple_layouts = (
                _named_tuple_layouts.copy() if _named_tuple_layouts else {}
            )
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
                if obj and getattr(obj, "__lirien_named_tuple__", False):
                    named_tuple_layouts[name] = struct_layouts.pop(name)

            try:
                code_ptr = lirien_bridge.verify_and_compile(
                    source,
                    target_func_name,
                    struct_layouts,
                    enum_layouts,
                    type_aliases,
                    named_tuple_layouts,
                    typed_dict_layouts,
                    timeout,
                    verify,
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
                print(f"[Lirien Warning] {error_msg}. Falling back to Python.")
                func.__lirien_jit__ = False
                return func

    # Support both @verify and @verify(strict=False)
    if callable(strict):
        f = strict
        strict = True
        return decorator(f)
    return decorator


def jit(
    strict: bool = True,
    log_level: str = None,
    timeout: int = 5000,
) -> Callable:
    """
    Decorator to compile a function directly to native machine code via Cranelift,
    bypassing Z3 formal verification.

    :param strict: If True, raises CompilationError/VerificationError on compilation failure.
    :param log_level: Override LILA_LOG level.
    :param timeout: Timeout in milliseconds.
    """
    if callable(strict):
        func = strict
        return verify(strict=True, verify=False)(func)
    return verify(strict=strict, log_level=log_level, timeout=timeout, verify=False)


def parallel_for(range_obj: range, body_fn: Callable[[int], None]):
    """
    Statically verified parallel loop.
    """
    for i in range_obj:
        body_fn(i)


def requires(predicate: Callable) -> Callable:
    """
    Specifies a precondition for the JIT function using a lambda.
    E.g., @requires(lambda x: x > 0)
    """

    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "__lirien_preconditions__"):
            func.__lirien_preconditions__ = []
        func.__lirien_preconditions__.append(predicate)
        return func

    return decorator


def ensures(predicate: Callable) -> Callable:
    """
    Specifies a postcondition for the JIT function using a lambda.
    E.g., @ensures(lambda res, x: res > x)
    """

    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "__lirien_postconditions__"):
            func.__lirien_postconditions__ = []
        func.__lirien_postconditions__.append(predicate)
        return func

    return decorator


def invariant(predicate: Callable) -> None:
    """
    Specifies a loop invariant using a lambda.
    E.g., invariant(lambda: i >= 0)
    """
    pass


verify.requires = requires
verify.ensures = ensures
verify.invariant = invariant
