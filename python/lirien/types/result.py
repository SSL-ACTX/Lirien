from typing import Generic, TypeVar
from .definitions import enum

T = TypeVar("T")
E = TypeVar("E")


class Ok:
    def __init__(self, value):
        self.value = value


class Err:
    def __init__(self, value):
        self.value = value


class Result(Generic[T, E]):
    """
    Zero-overhead Result type for safe error handling.
    Usage: res: Result[i64, i64]
    """

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Result requires [T, E]")

        T_ty, E_ty = params

        # We create a specialized class for these types
        class ResultInstance:
            __annotations__ = {"Ok": T_ty, "Err": E_ty}

        T_name = getattr(T_ty, "__name__", str(T_ty))
        E_name = getattr(E_ty, "__name__", str(E_ty))
        ResultInstance.__name__ = f"Result_{T_name}_{E_name}"

        # Apply the enum decorator to make it a tagged union
        specialized = enum(ResultInstance)
        return specialized


__all__ = ["Result", "Ok", "Err"]
