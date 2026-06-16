from typing import Annotated


class FnPointer:
    """
    Represents a raw function pointer.
    Usage: f: FnPointer[[i64, i64], i64]
    """

    def __init__(self, arg_types, ret_type):
        self.arg_types = arg_types
        self.ret_type = ret_type

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("FnPointer requires [[arg_types], ret_type]")
        return Annotated[cls, params]


class Callable:
    """
    Represents a generic callable (function or closure).
    """

    def __class_getitem__(cls, params):
        return Annotated[cls, params]


class Closure:
    """
    Represents a function with captured state.
    """

    def __class_getitem__(cls, params):
        return Annotated[cls, params]
