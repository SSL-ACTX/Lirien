import ctypes
from typing import TypeVar, Generic, Annotated
from .base import TYPE_MAP, f32

T = TypeVar("T")


class Refined:
    """
    Liquid Type: A base type refined by a logical predicate.
    Example: Refined[i32, lambda x: x > 0]
    """

    def __init__(self, base_type, predicate):
        self.base_type = base_type
        self.predicate = predicate

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            return cls(params, None)
        return cls(params[0], params[1])


class SizedArray(Generic[T]):
    """
    Statically sized array.
    Usage: arr: SizedArray[i32, 10]
    """

    def __init__(self, base_type, size):
        self.base_type = base_type
        self.size = size

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("SizedArray requires [type, size]")

        base_type, size = params

        if size is Ellipsis or isinstance(size, TypeVar):
            return Annotated[cls, (base_type, size)]

        base_ty_str = getattr(base_type, "__name__", str(base_type)).lower()

        # Default to i64
        cty = ctypes.c_int64
        if hasattr(base_type, "__lila_struct__"):
            cty = base_type.__lila_ctypes__
        elif isinstance(base_type, type) and issubclass(base_type, ctypes.Structure):
            cty = base_type
        else:
            base_ty_str = getattr(base_type, "__name__", str(base_type)).lower()
            for name, ct in TYPE_MAP.items():
                if name in base_ty_str:
                    cty = ct
                    break

        class SizedCtypesArray(ctypes.Structure):
            _fields_ = [("data", cty * size)]

        class SizedLilaArray:
            def __init__(self, *args):
                self._ctypes_obj = SizedCtypesArray()
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    vals = args[0]
                else:
                    vals = args

                for i, val in enumerate(vals):
                    if i >= size:
                        break
                    if hasattr(val, "_ctypes_obj"):
                        self._ctypes_obj.data[i] = val._ctypes_obj
                    else:
                        self._ctypes_obj.data[i] = val

            def __getitem__(self, idx):
                return self._ctypes_obj.data[idx]

            def __setitem__(self, idx, val):
                if hasattr(val, "_ctypes_obj"):
                    self._ctypes_obj.data[idx] = val._ctypes_obj
                else:
                    self._ctypes_obj.data[idx] = val

            def __len__(self):
                return size

        SizedLilaArray.__name__ = f"SizedArray_{base_ty_str}_{size}"
        return Annotated[SizedLilaArray, (base_type, size)]


class Buffer:
    """
    Represents an external memory buffer (e.g. NumPy array).
    Usage: arr: Buffer[i32]
    """

    @classmethod
    def alloc(cls, size: int) -> "memoryview":
        raise NotImplementedError("Use the specialized type hint's alloc method")

    def __class_getitem__(cls, base_type):
        if base_type is Ellipsis:
            return Annotated[cls, Ellipsis]

        class BufferAnnotated:
            @classmethod
            def alloc(cls_ann, count: int):
                if getattr(base_type, "__lila_struct__", False):
                    item_cty = base_type.__lila_ctypes__
                else:
                    item_ty_str = str(base_type).lower()
                    item_cty = ctypes.c_int64
                    for name, cty in TYPE_MAP.items():
                        if name in item_ty_str:
                            item_cty = cty
                            break

                ArrayType = item_cty * count
                return ArrayType()

        anno = Annotated[cls, base_type]
        anno.alloc = BufferAnnotated.alloc
        return anno


class Box:
    """
    Represents a heap-allocated (boxed) value.
    Usage: val: Box[i64]
    """

    def __init__(self, value):
        self.value = value
        if value is not None:
            if hasattr(value, "_ctypes_obj"):
                ptr = ctypes.pointer(value._ctypes_obj)
                self._ctypes_obj = ctypes.cast(ptr, ctypes.c_void_p)
            else:
                from ..signatures import _value_to_lila_type

                c_ty = TYPE_MAP.get(_value_to_lila_type(value), ctypes.c_int64)
                ptr = ctypes.pointer(c_ty(value))
                self._ctypes_obj = ctypes.cast(ptr, ctypes.c_void_p)
        else:
            self._ctypes_obj = None

    def __class_getitem__(cls, base_type):
        return Annotated[cls, base_type]


class Tensor(Generic[T]):
    """
    Statically shape-typed Tensor.
    Usage: a: Tensor[f32, "M", "N"]
    """

    def __init__(self, data, shape: tuple, base_cty=ctypes.c_float):
        self.shape = shape
        self.__data = data
        self.__base_cty = base_cty
        self.size = 1
        for dim in shape:
            self.size *= dim

        if isinstance(data, ctypes.Array):
            self.ptr = ctypes.addressof(data)
        else:
            arr = (base_cty * self.size)()

            def flatten(l):
                for item in l:
                    if isinstance(item, (list, tuple)):
                        yield from flatten(item)
                    else:
                        yield item

            for i, val in enumerate(flatten(data)):
                arr[i] = val
            self.__data = arr
            self.ptr = ctypes.addressof(arr)

    @classmethod
    def alloc(cls, shape: tuple, base_type=f32):
        item_cty = ctypes.c_float
        if getattr(base_type, "__lila_struct__", False):
            item_cty = base_type.__lila_ctypes__
        else:
            item_ty_str = str(base_type).lower()
            for name, cty in TYPE_MAP.items():
                if name in item_ty_str:
                    item_cty = cty
                    break
        size = 1
        for dim in shape:
            size *= dim
        arr = (item_cty * size)()
        return cls(arr, shape, item_cty)

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        flat_idx = 0
        stride = 1
        for i in reversed(range(len(self.shape))):
            flat_idx += idxs[i] * stride
            stride *= self.shape[i]
        return self.__data[flat_idx]

    def __setitem__(self, idxs, val):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        flat_idx = 0
        stride = 1
        for i in reversed(range(len(self.shape))):
            flat_idx += idxs[i] * stride
            stride *= self.shape[i]
        self.__data[flat_idx] = val

    @property
    def __lila_ptr__(self):
        return self.ptr

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) < 2:
            raise TypeError("Tensor requires [type, *shape]")
        base_type = params[0]
        shape = params[1:]
        return Annotated[cls, (base_type, shape)]


class Array(Generic[T]):
    def __init__(self, size: int, initial_val: T = 0):
        self.data = [initial_val] * size

    def __getitem__(self, idx: int) -> T:
        return self.data[idx]

    def __setitem__(self, idx: int, val: T):
        self.data[idx] = val
