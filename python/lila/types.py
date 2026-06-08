import ctypes
import sys
from typing import TypeVar, Generic, Annotated

T = TypeVar("T")


# Lila Native Types
class LilaType:
    pass


class i8(int, LilaType):
    pass


class u8(int, LilaType):
    pass


class i16(int, LilaType):
    pass


class u16(int, LilaType):
    pass


class i32(int, LilaType):
    pass


class u32(int, LilaType):
    pass


class i64(int, LilaType):
    pass


class u64(int, LilaType):
    pass


class f32(float, LilaType):
    pass


class f64(float, LilaType):
    pass


class f32x4(ctypes.Structure):
    _align_ = 16
    _fields_ = [
        ("f0", ctypes.c_float),
        ("f1", ctypes.c_float),
        ("f2", ctypes.c_float),
        ("f3", ctypes.c_float),
    ]

    def __init__(self, *args):
        if len(args) == 4:
            self.f0, self.f1, self.f2, self.f3 = args
        elif len(args) == 1:
            self.f0 = self.f1 = self.f2 = self.f3 = args[0]

    def __getitem__(self, idx):
        return [self.f0, self.f1, self.f2, self.f3][idx]

    def __add__(self, other):
        return f32x4(*(self[i] + other[i] for i in range(4)))

    def __sub__(self, other):
        return f32x4(*(self[i] - other[i] for i in range(4)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return f32x4(*(self[i] * other for i in range(4)))
        return f32x4(*(self[i] * other[i] for i in range(4)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return f32x4(*(self[i] / other for i in range(4)))
        return f32x4(*(self[i] / other[i] for i in range(4)))


class i32x4(ctypes.Structure):
    _align_ = 16
    _fields_ = [
        ("f0", ctypes.c_int32),
        ("f1", ctypes.c_int32),
        ("f2", ctypes.c_int32),
        ("f3", ctypes.c_int32),
    ]

    def __init__(self, *args):
        if len(args) == 4:
            self.f0, self.f1, self.f2, self.f3 = args
        elif len(args) == 1:
            self.f0 = self.f1 = self.f2 = self.f3 = args[0]

    def __getitem__(self, idx):
        return [self.f0, self.f1, self.f2, self.f3][idx]

    def __add__(self, other):
        return i32x4(*(self[i] + other[i] for i in range(4)))

    def __sub__(self, other):
        return i32x4(*(self[i] - other[i] for i in range(4)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return i32x4(*(self[i] * other for i in range(4)))
        return i32x4(*(self[i] * other[i] for i in range(4)))


class f64x2(ctypes.Structure):
    _align_ = 16
    _fields_ = [("f0", ctypes.c_double), ("f1", ctypes.c_double)]

    def __init__(self, *args):
        if len(args) == 2:
            self.f0, self.f1 = args
        elif len(args) == 1:
            self.f0 = self.f1 = args[0]

    def __getitem__(self, idx):
        return [self.f0, self.f1][idx]

    def __add__(self, other):
        return f64x2(*(self[i] + other[i] for i in range(2)))

    def __sub__(self, other):
        return f64x2(*(self[i] - other[i] for i in range(2)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return f64x2(*(self[i] * other for i in range(2)))
        return f64x2(*(self[i] * other[i] for i in range(2)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return f64x2(*(self[i] / other for i in range(2)))
        return f64x2(*(self[i] / other[i] for i in range(2)))


class i64x2(ctypes.Structure):
    _align_ = 16
    _fields_ = [("f0", ctypes.c_int64), ("f1", ctypes.c_int64)]

    def __init__(self, *args):
        if len(args) == 2:
            self.f0, self.f1 = args
        elif len(args) == 1:
            self.f0 = self.f1 = args[0]

    def __getitem__(self, idx):
        return [self.f0, self.f1][idx]

    def __add__(self, other):
        return i64x2(*(self[i] + other[i] for i in range(2)))

    def __sub__(self, other):
        return i64x2(*(self[i] - other[i] for i in range(2)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return i64x2(*(self[i] * other for i in range(2)))
        return i64x2(*(self[i] * other[i] for i in range(2)))


# Type Mapping to ctypes
TYPE_MAP = {
    "i8": ctypes.c_int8,
    "u8": ctypes.c_uint8,
    "i16": ctypes.c_int16,
    "u16": ctypes.c_uint16,
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "int": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "float": ctypes.c_double,
    "bool": ctypes.c_bool,
    "f32x4": f32x4,
    "i32x4": i32x4,
    "f64x2": f64x2,
    "i64x2": i64x2,
}


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
                val = self._ctypes_obj.data[idx]
                # If the element is a ctypes object (e.g. SIMD vector), wrap it back if needed
                # However, for SIMD we usually return the raw ctypes object if it's not wrapped.
                # Let's ensure consistency.
                return val

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
        """
        Allocates a contiguous bytearray for a given number of elements.
        Must be called on a specialized type hint (e.g., Buffer[Point3D].alloc(10)).
        """
        raise NotImplementedError("Use the specialized type hint's alloc method")

    def __class_getitem__(cls, base_type):
        from typing import Annotated
        import ctypes

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

                # Return a ctypes array directly. It implements the buffer protocol
                # and allows direct indexing/field access without exposing ctypes to the user.
                ArrayType = item_cty * count
                return ArrayType()

        # We merge the base Buffer functionality into the Annotated object
        # by making it a subclass or just injecting alloc.
        anno = Annotated[cls, base_type]
        # Attach alloc to the annotated type object directly
        anno.alloc = BufferAnnotated.alloc
        return anno


class Box:
    """
    Represents a heap-allocated (boxed) value.
    Usage: val: Box[i64]
    """

    def __init__(self, value):
        self.value = value

    def __class_getitem__(cls, base_type):
        return Annotated[cls, base_type]


class Array(Generic[T]):
    def __init__(self, size: int, initial_val: T = 0):
        self.data = [initial_val] * size

    def __getitem__(self, idx: int) -> T:
        return self.data[idx]

    def __setitem__(self, idx: int, val: T):
        self.data[idx] = val


def struct(cls):
    """
    Decorator to mark a class as a flat Lila Struct.
    Generates a ctypes Structure behind the scenes.
    """
    fields = getattr(cls, "__annotations__", {})
    field_list = []
    ctypes_fields = []

    for name, ty in fields.items():
        actual_ty = getattr(ty, "base_type", ty)
        ty_str = str(actual_ty).lower()

        # Handle nested structs
        if hasattr(actual_ty, "__lila_struct__"):
            field_list.append((name, actual_ty))
            ctypes_fields.append((name, actual_ty.__lila_ctypes__))
            continue

        # Map to ctypes
        c_ty = ctypes.c_int64
        found = False
        for type_name, ct in TYPE_MAP.items():
            if type_name in ty_str:
                c_ty = ct
                found = True
                break

        if not found:
            # Fallback for complex types or strings
            c_ty = ctypes.c_void_p

        field_list.append((name, actual_ty))
        ctypes_fields.append((name, c_ty))

    class LilaCtypesStruct(ctypes.Structure):
        _fields_ = ctypes_fields

    cls.__lila_struct__ = True
    cls.__lila_fields__ = field_list
    cls.__lila_ctypes__ = LilaCtypesStruct
    cls.__match_args__ = tuple(name for name, _ in field_list)

    # Add a constructor that accepts values for fields
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        processed_args = []
        processed_kwargs = {}

        # Handle positional args
        for i, arg in enumerate(args):
            if hasattr(arg, "_ctypes_obj"):
                processed_args.append(arg._ctypes_obj)
            else:
                processed_args.append(arg)

        # Handle keyword args
        for key, value in kwargs.items():
            if hasattr(value, "_ctypes_obj"):
                processed_kwargs[key] = value._ctypes_obj
            else:
                processed_kwargs[key] = value

        self._ctypes_obj = LilaCtypesStruct(*processed_args, **processed_kwargs)
        if original_init is not object.__init__:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    # Add property accessors that wrap nested structs
    for name, ftype in field_list:

        def make_getter(fname, fty):
            def getter(self_ref):
                val = getattr(self_ref._ctypes_obj, fname)
                # If the field is a struct, we need to wrap the raw ctypes buffer in its Lila class
                if hasattr(fty, "__lila_struct__"):
                    wrapper = fty.__new__(fty)
                    wrapper._ctypes_obj = val
                    return wrapper
                return val

            return getter

        def make_setter(fname):
            def setter(self_ref, val):
                if hasattr(val, "_ctypes_obj"):
                    setattr(self_ref._ctypes_obj, fname, val._ctypes_obj)
                else:
                    setattr(self_ref._ctypes_obj, fname, val)

            return setter

        setattr(cls, name, property(make_getter(name, ftype), make_setter(name)))

    return cls


value = struct


def enum(cls):
    """
    Decorator to mark a class as a Tagged Union (Enum) for Lila.
    Generates a ctypes Structure with a tag and a Union payload.
    """
    # Grab the annotations, which define the variants.
    fields = getattr(cls, "__annotations__", {})

    variant_names = []
    variant_types = {}
    union_fields = []

    # Helper to generate a dummy struct for empty or primitive variants
    def make_payload_struct(name, fields):
        class PayloadStruct(ctypes.Structure):
            _fields_ = fields

        PayloadStruct.__name__ = name
        return PayloadStruct

    for idx, (name, ty) in enumerate(fields.items()):
        variant_names.append(name)
        variant_types[name] = ty

        actual_ty = getattr(ty, "base_type", ty)
        origin = getattr(actual_ty, "__origin__", None)
        typing_tuple = getattr(sys.modules.get("typing"), "Tuple", None)

        if ty is None:
            # Empty variant
            union_fields.append(
                (name, make_payload_struct(f"{cls.__name__}_{name}_payload", []))
            )
        elif hasattr(ty, "__metadata__") and "Box" in str(ty.__origin__):
            # Boxed variant
            union_fields.append((name, ctypes.c_void_p))
        elif hasattr(ty, "__lila_ctypes__"):
            union_fields.append((name, ty.__lila_ctypes__))
        elif (
            origin is tuple
            or (typing_tuple and origin is typing_tuple)
            or isinstance(actual_ty, tuple)
        ):
            # Tuple payload
            tuple_elts = getattr(
                actual_ty, "__args__", actual_ty if isinstance(actual_ty, tuple) else []
            )
            tuple_fields = []
            for i, t in enumerate(tuple_elts):
                if hasattr(t, "__metadata__") and "Box" in str(
                    getattr(t, "__origin__", None)
                ):
                    tuple_fields.append((f"f{i}", ctypes.c_void_p))
                else:
                    t_str = str(t).lower()
                    c_ty = ctypes.c_int64
                    for n, cty in TYPE_MAP.items():
                        if n in t_str:
                            c_ty = cty
                            break
                    tuple_fields.append((f"f{i}", c_ty))
            union_fields.append(
                (
                    name,
                    make_payload_struct(f"{cls.__name__}_{name}_payload", tuple_fields),
                )
            )
        else:
            # Primitive type
            ty_str = str(ty).lower()
            c_ty = ctypes.c_int64
            for n, cty in TYPE_MAP.items():
                if n in ty_str:
                    c_ty = cty
                    break
            union_fields.append(
                (
                    name,
                    make_payload_struct(
                        f"{cls.__name__}_{name}_payload", [("val", c_ty)]
                    ),
                )
            )

    class LilaCtypesUnion(ctypes.Union):
        _fields_ = union_fields

    class LilaCtypesEnum(ctypes.Structure):
        _fields_ = [("tag", ctypes.c_uint8), ("payload", LilaCtypesUnion)]

    cls.__lila_enum__ = True
    cls.__lila_variants__ = variant_names
    cls.__lila_variant_types__ = variant_types
    cls.__lila_ctypes__ = LilaCtypesEnum

    # We don't override __init__ in the same way; instead, we provide classmethods for variants
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        self._ctypes_obj = LilaCtypesEnum()
        if original_init is not object.__init__:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    # Generate constructor and accessor methods for each variant
    for idx, (name, ty) in enumerate(variant_types.items()):

        def make_variant_methods(variant_name, variant_type, tag_idx):
            payload_cty = dict(union_fields)[variant_name]

            @classmethod
            def constructor(cls_ref, *args, **kwargs):
                instance = cls_ref.__new__(cls_ref)
                instance._ctypes_obj = LilaCtypesEnum()
                instance._ctypes_obj.tag = tag_idx

                if variant_type is None:
                    pass
                elif hasattr(variant_type, "__metadata__") and "Box" in str(
                    variant_type.__origin__
                ):
                    # Boxed variant
                    payload_instance = args[0]
                    if hasattr(payload_instance, "_ctypes_obj"):
                        # Get a pointer to the object
                        ptr = ctypes.cast(
                            ctypes.pointer(payload_instance._ctypes_obj),
                            ctypes.c_void_p,
                        )
                        setattr(instance._ctypes_obj.payload, variant_name, ptr)
                    else:
                        # Fallback for primitives?
                        pass
                elif hasattr(variant_type, "__lila_ctypes__"):
                    # Check if we are passing an already constructed instance of the variant type
                    if len(args) == 1 and isinstance(args[0], variant_type):
                        payload_instance = args[0]
                    else:
                        payload_instance = variant_type(*args, **kwargs)

                    setattr(
                        instance._ctypes_obj.payload,
                        variant_name,
                        payload_instance._ctypes_obj,
                    )
                elif isinstance(variant_type, tuple):
                    # args should match tuple elements
                    payload_obj = payload_cty()
                    for i, arg in enumerate(args):
                        if i < len(variant_type):
                            v_ty = variant_type[i]
                            if hasattr(v_ty, "__metadata__") and "Box" in str(
                                v_ty.__origin__
                            ):
                                # Boxed element
                                if hasattr(arg, "_ctypes_obj"):
                                    ptr = ctypes.cast(
                                        ctypes.pointer(arg._ctypes_obj), ctypes.c_void_p
                                    )
                                    setattr(payload_obj, f"f{i}", ptr)
                                else:
                                    setattr(payload_obj, f"f{i}", arg)
                            else:
                                setattr(payload_obj, f"f{i}", arg)
                    setattr(instance._ctypes_obj.payload, variant_name, payload_obj)
                else:
                    # Primitive
                    payload_obj = payload_cty()
                    payload_obj.val = args[0] if args else 0
                    setattr(instance._ctypes_obj.payload, variant_name, payload_obj)

                return instance

            def is_variant(self):
                return self._ctypes_obj.tag == tag_idx

            def as_variant(self):
                if self._ctypes_obj.tag != tag_idx:
                    raise ValueError(
                        f"Tried to access Enum as {variant_name} but tag is {self._ctypes_obj.tag}"
                    )

                raw_payload = getattr(self._ctypes_obj.payload, variant_name)
                if variant_type is None:
                    return None
                elif hasattr(variant_type, "__metadata__") and "Box" in str(
                    variant_type.__origin__
                ):
                    # Boxed variant: raw_payload is a c_void_p
                    inner_ty = variant_type.__metadata__[0]
                    # This might be a string (lazy type)
                    if isinstance(inner_ty, str):
                        # Try to resolve from the class's module
                        module = sys.modules.get(cls.__module__)
                        if module and hasattr(module, inner_ty):
                            inner_ty = getattr(module, inner_ty)
                        elif inner_ty == cls.__name__:
                            inner_ty = cls

                    if hasattr(inner_ty, "__lila_ctypes__"):
                        wrapper = inner_ty.__new__(inner_ty)
                        # We need to cast void_p back to the struct type
                        wrapper._ctypes_obj = ctypes.cast(
                            raw_payload, ctypes.POINTER(inner_ty.__lila_ctypes__)
                        ).contents
                        return wrapper
                    return raw_payload
                elif hasattr(variant_type, "__lila_ctypes__"):
                    wrapper = variant_type.__new__(variant_type)
                    wrapper._ctypes_obj = raw_payload
                    return wrapper
                elif isinstance(variant_type, tuple):
                    res = []
                    for i, v_ty in enumerate(variant_type):
                        raw_val = getattr(raw_payload, f"f{i}")
                        if hasattr(v_ty, "__metadata__") and "Box" in str(
                            v_ty.__origin__
                        ):
                            inner_ty = v_ty.__metadata__[0]
                            if isinstance(inner_ty, str):
                                if inner_ty == cls.__name__:
                                    inner_ty = cls

                            if hasattr(inner_ty, "__lila_ctypes__"):
                                wrapper = inner_ty.__new__(inner_ty)
                                wrapper._ctypes_obj = ctypes.cast(
                                    raw_val, ctypes.POINTER(inner_ty.__lila_ctypes__)
                                ).contents
                                res.append(wrapper)
                            else:
                                res.append(raw_val)
                        else:
                            res.append(raw_val)
                    return tuple(res)
                else:
                    # Primitive
                    return raw_payload.val

            return constructor, is_variant, as_variant

        ctor, is_var, as_var = make_variant_methods(name, ty, idx)
        setattr(cls, name, ctor)
        setattr(cls, f"is_{name}", is_var)
        setattr(cls, f"as_{name}", as_var)

    return cls


adt = enum


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


__all__ = [
    "struct",
    "enum",
    "adt",
    "i8",
    "u8",
    "i16",
    "u16",
    "i32",
    "u32",
    "i64",
    "u64",
    "f32",
    "f64",
    "f32x4",
    "i32x4",
    "f64x2",
    "i64x2",
    "Refined",
    "SizedArray",
    "Buffer",
    "Box",
    "FnPointer",
    "Callable",
    "Closure",
    "TYPE_MAP",
]
