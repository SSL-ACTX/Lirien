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

        from .arithmetic import TypeExpr

        if (
            size is Ellipsis
            or isinstance(size, (TypeVar, TypeExpr))
            or hasattr(size, "__lirien_typevar__")
        ):
            return Annotated[cls, (base_type, size)]

        base_ty_str = getattr(base_type, "__name__", str(base_type)).lower()

        # Default to i64
        cty = ctypes.c_int64
        if hasattr(base_type, "__lirien_struct__"):
            cty = base_type.__lirien_ctypes__
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

        class SizedLirienArray:
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

        SizedLirienArray.__name__ = f"SizedArray_{base_ty_str}_{size}"
        return Annotated[SizedLirienArray, (base_type, size)]


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
                if getattr(base_type, "__lirien_struct__", False):
                    item_cty = base_type.__lirien_ctypes__
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
                from ..compiler import _value_to_lirien_type

                c_ty = TYPE_MAP.get(_value_to_lirien_type(value), ctypes.c_int64)
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
        if getattr(base_type, "__lirien_struct__", False):
            item_cty = base_type.__lirien_ctypes__
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
    def __lirien_ptr__(self):
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


class ListHeaderStructure(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
    ]


class List(Generic[T]):
    """
    Verified Growable List.
    Usage: l = List[i64]()
    """

    def __init__(self, c_ptr=None, elem_type=None):
        self._elem_type = elem_type
        if c_ptr is not None:
            if isinstance(c_ptr, int):
                self._ctypes_obj = ctypes.c_void_p(c_ptr)
            else:
                self._ctypes_obj = c_ptr
        else:
            header = ListHeaderStructure(data=None, len=0, cap=0)
            self._header_ref = header
            self._ctypes_obj = ctypes.pointer(header)

    @property
    def elem_type(self):
        if self._elem_type is None:
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class:
                from typing import get_args

                args = get_args(orig_class)
                if args:
                    self._elem_type = args[0]
        return self._elem_type

    @property
    def _header(self):
        if isinstance(self._ctypes_obj, ctypes.c_void_p):
            return ctypes.cast(
                self._ctypes_obj, ctypes.POINTER(ListHeaderStructure)
            ).contents
        else:
            return self._ctypes_obj.contents

    def append(self, val):
        header = self._header
        elem_cty = ctypes.c_int64
        is_struct = False
        elem_t = self.elem_type
        if elem_t is not None:
            if getattr(elem_t, "__lirien_struct__", False):
                elem_cty = elem_t.__lirien_ctypes__
                is_struct = True
            else:
                elem_ty_str = str(elem_t).lower()
                for name, cty in TYPE_MAP.items():
                    if name in elem_ty_str:
                        elem_cty = cty
                        break

        elem_size = ctypes.sizeof(elem_cty)

        if header.len == header.cap:
            new_cap = 4 if header.cap == 0 else header.cap * 2
            new_data_cty = ctypes.c_ubyte * (new_cap * elem_size)
            new_data_obj = new_data_cty()
            if header.data:
                ctypes.memmove(new_data_obj, header.data, header.len * elem_size)
            if not hasattr(self, "_data_buffers"):
                self._data_buffers = []
            self._data_buffers.append(new_data_obj)
            header.data = ctypes.cast(
                ctypes.pointer(new_data_obj), ctypes.c_void_p
            ).value
            header.cap = new_cap

        elem_addr = header.data + header.len * elem_size
        if is_struct:
            val_obj = val._ctypes_obj if hasattr(val, "_ctypes_obj") else val
            ctypes.memmove(elem_addr, ctypes.addressof(val_obj), elem_size)
        else:
            elem_cty.from_address(elem_addr).value = val
        header.len += 1

    def __getitem__(self, idx):
        header = self._header
        len_val = header.len
        if idx < 0:
            idx = int(len_val) + idx
        if idx < 0 or idx >= len_val:
            raise IndexError("List index out of range")

        elem_cty = ctypes.c_int64
        is_struct = False
        elem_t = self.elem_type
        if elem_t is not None:
            if getattr(elem_t, "__lirien_struct__", False):
                elem_cty = elem_t.__lirien_ctypes__
                is_struct = True
            else:
                elem_ty_str = str(elem_t).lower()
                for name, cty in TYPE_MAP.items():
                    if name in elem_ty_str:
                        elem_cty = cty
                        break

        elem_size = ctypes.sizeof(elem_cty)
        elem_addr = header.data + idx * elem_size
        if is_struct:
            val = elem_cty.from_address(elem_addr)
            wrapper = elem_t.__new__(elem_t)
            wrapper._ctypes_obj = val
            return wrapper
        else:
            return elem_cty.from_address(elem_addr).value

    def __setitem__(self, idx, val):
        header = self._header
        len_val = header.len
        if idx < 0:
            idx = int(len_val) + idx
        if idx < 0 or idx >= len_val:
            raise IndexError("List index out of range")

        elem_cty = ctypes.c_int64
        is_struct = False
        elem_t = self.elem_type
        if elem_t is not None:
            if getattr(elem_t, "__lirien_struct__", False):
                elem_cty = elem_t.__lirien_ctypes__
                is_struct = True
            else:
                elem_ty_str = str(elem_t).lower()
                for name, cty in TYPE_MAP.items():
                    if name in elem_ty_str:
                        elem_cty = cty
                        break

        elem_size = ctypes.sizeof(elem_cty)
        elem_addr = header.data + idx * elem_size
        if is_struct:
            val_obj = val._ctypes_obj if hasattr(val, "_ctypes_obj") else val
            ctypes.memmove(elem_addr, ctypes.addressof(val_obj), elem_size)
        else:
            elem_cty.from_address(elem_addr).value = val

    def __len__(self):
        return self._header.len

    def __class_getitem__(cls, base_type):
        class ListAnnotated:
            def __new__(cls_ann, *args, **kwargs):
                return List(*args, elem_type=base_type, **kwargs)

        anno = Annotated[cls, base_type]
        anno.new_instance = lambda c_ptr: List(c_ptr, elem_type=base_type)
        return anno


class SymbolicExpr:
    __lirien_symbolic__ = True

    def __init__(self, expr_str, fn):
        self.expr_str = expr_str
        self.fn = fn

    def __str__(self):
        return f"lambda x: {self.expr_str}"

    def __call__(self, val):
        return self.fn(val)

    # Comparisons
    def __gt__(self, other):
        return SymbolicExpr(f"x > {other}", lambda x: self(x) > other)

    def __lt__(self, other):
        return SymbolicExpr(f"x < {other}", lambda x: self(x) < other)

    def __ge__(self, other):
        return SymbolicExpr(f"x >= {other}", lambda x: self(x) >= other)

    def __le__(self, other):
        return SymbolicExpr(f"x <= {other}", lambda x: self(x) <= other)

    def __eq__(self, other):
        return SymbolicExpr(f"x == {other}", lambda x: self(x) == other)

    def __ne__(self, other):
        return SymbolicExpr(f"x != {other}", lambda x: self(x) != other)

    # Logical operations (bitwise in Python, logical context in refinements)
    def __and__(self, other):
        if hasattr(other, "expr_str"):
            return SymbolicExpr(
                f"({self.expr_str}) & ({other.expr_str})",
                lambda x: bool(self(x)) and bool(other(x)),
            )
        return SymbolicExpr(f"({self.expr_str}) & {other}", lambda x: self(x) & other)

    def __or__(self, other):
        if hasattr(other, "expr_str"):
            return SymbolicExpr(
                f"({self.expr_str}) | ({other.expr_str})",
                lambda x: bool(self(x)) or bool(other(x)),
            )
        return SymbolicExpr(f"({self.expr_str}) | {other}", lambda x: self(x) | other)

    def __invert__(self):
        return SymbolicExpr(f"not ({self.expr_str})", lambda x: not self(x))

    # Basic arithmetic
    def __mod__(self, other):
        return SymbolicExpr(f"({self.expr_str}) % {other}", lambda x: self(x) % other)

    def __add__(self, other):
        return SymbolicExpr(f"({self.expr_str}) + {other}", lambda x: self(x) + other)

    def __sub__(self, other):
        return SymbolicExpr(f"({self.expr_str}) - {other}", lambda x: self(x) - other)

    def __mul__(self, other):
        return SymbolicExpr(f"({self.expr_str}) * {other}", lambda x: self(x) * other)


V = SymbolicExpr("x", lambda x: x)
