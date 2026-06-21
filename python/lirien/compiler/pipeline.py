import inspect
import ast
import textwrap
from typing import Callable, Tuple, TypeVar, TypeVarTuple, get_origin, get_args


def _prepare_source_and_name(
    func: Callable, class_name: str = None, method_name: str = None
) -> Tuple[str, str]:
    """Extract and dedent source code, handling method name overrides and AST adjustments."""
    source = textwrap.dedent(inspect.getsource(func))
    target_func_name = method_name if method_name else func.__name__

    if target_func_name == "<lambda>":
        target_func_name = "lirien_lambda"

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
    if hasattr(ann, "__metadata__"):
        metadata = ann.__metadata__
        if metadata:
            if any(_has_ellipsis(m) for m in metadata):
                return True
            if isinstance(metadata[0], (list, tuple)):
                if any(_has_ellipsis(m) for m in metadata[0]):
                    return True
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


def _has_callable(ann):
    """Check if a type annotation contains Callable, Closure, or FnPointer."""
    if ann is None:
        return False
    from ..types import FnPointer, Closure

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


def _needs_monomorphization(ann):
    """Check if a type annotation needs monomorphization/specialization."""
    if not ann or ann is inspect.Parameter.empty:
        return False

    if isinstance(ann, (TypeVar, TypeVarTuple)) or hasattr(ann, "__lirien_typevar__"):
        return True

    origin = get_origin(ann)
    if origin is not None:
        if hasattr(origin, "__parameters__") and origin.__parameters__:
            return True
        for arg in get_args(ann):
            if _needs_monomorphization(arg):
                return True

    if isinstance(ann, (list, tuple)):
        for arg in ann:
            if _needs_monomorphization(arg):
                return True

    if getattr(ann, "__lirien_specialized__", False):
        return True

    if _has_ellipsis(ann) or _has_protocol(ann) or _has_callable(ann):
        return True

    return False
