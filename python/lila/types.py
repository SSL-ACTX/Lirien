import ctypes
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


class SizedArrayMeta(type):
    def __getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("SizedArray requires [type, size]")

        base_type, size = params
        base_ty_str = str(base_type).lower()

        # Default to i64
        cty = ctypes.c_int64
        for name, ct in TYPE_MAP.items():
            if name in base_ty_str:
                cty = ct
                break

        # Create a specific class for this array type
        class SizedArrayInstance:
            _ctypes_type = cty * size

            def __init__(self, initial_data=None):
                self._ctypes_obj = self._ctypes_type()
                if initial_data:
                    for i, val in enumerate(initial_data):
                        if i < size:
                            self._ctypes_obj[i] = val

            def __getitem__(self, idx):
                return self._ctypes_obj[idx]

            def __setitem__(self, idx, val):
                self._ctypes_obj[idx] = val

            def __len__(self):
                return size

        SizedArrayInstance.__name__ = f"SizedArray_{base_ty_str}_{size}"
        return SizedArrayInstance


class SizedArray(metaclass=SizedArrayMeta):
    """
    Seamless memory-backed array.
    Usage: arr = SizedArray[i64, 5]([1, 2, 3, 4, 5])
    """

    pass


class BufferMeta(type):
    def __getitem__(cls, base_type):
        base_ty_str = getattr(base_type, "__name__", str(base_type)).lower()
        return Annotated[cls, base_ty_str]


class Buffer(metaclass=BufferMeta):
    """
    Represents an external memory buffer (e.g. NumPy array).
    Usage: arr: Buffer[i32]
    """

    pass


class HandMeta(type):
    def __getitem__(cls, base_type):
        base_ty_str = str(base_type).lower()
        cty = ctypes.c_int64
        for name, ct in TYPE_MAP.items():
            if name in base_ty_str:
                cty = ct
                break

        class HandInstance:
            _ctypes_type = cty

            def __init__(self, val=0):
                self._ctypes_obj = self._ctypes_type(val)

            @property
            def val(self):
                return self._ctypes_obj.value

            @val.setter
            def val(self, new_val):
                self._ctypes_obj.value = new_val

            def __repr__(self):
                return f"Hand[{base_ty_str}]({self.val})"

        HandInstance.__name__ = f"Hand_{base_ty_str}"
        return HandInstance

    def __call__(cls, val):
        # Handle Hand(10) -> defaults to i64
        return cls[i64](val)


class Hand(metaclass=HandMeta):
    """
    Seamless mutable reference to a single value.
    Usage: m = Hand(10) or m = Hand[i64](10)
    """

    pass


class PeekMeta(type):
    def __getitem__(cls, base_type):
        base_ty_str = str(base_type).lower()
        cty = ctypes.c_int64
        for name, ct in TYPE_MAP.items():
            if name in base_ty_str:
                cty = ct
                break

        class PeekInstance:
            _ctypes_type = cty

            def __init__(self, val=0):
                self._ctypes_obj = self._ctypes_type(val)

            @property
            def val(self):
                return self._ctypes_obj.value

            def __repr__(self):
                return f"Peek[{base_ty_str}]({self.val})"

        PeekInstance.__name__ = f"Peek_{base_ty_str}"
        return PeekInstance

    def __call__(cls, val):
        return cls[i64](val)


class Peek(metaclass=PeekMeta):
    """
    Seamless immutable reference to a single value.
    Usage: r = Peek(10)
    """

    pass


class HeldMeta(type):
    def __getitem__(cls, base_type):
        return cls


class Held(metaclass=HeldMeta):
    pass


class Array(Generic[T]):
    def __init__(self, size: int, initial_val: T = 0):
        self.data = [initial_val] * size

    def __getitem__(self, idx: int) -> T:
        return self.data[idx]

    def __setitem__(self, idx: int, val: T):
        self.data[idx] = val


def struct(cls):
    """
    Decorator to mark a class as a Memory Struct for Lila.
    Automatically generates a ctypes structure for interop.
    """
    fields = getattr(cls, "__annotations__", {})

    field_list = []
    ctypes_fields = []

    # Store field types to handle wrap/unwrap
    field_types = {}

    for name, ty in fields.items():
        ty_str = str(ty).lower()

        # Check if it's a known Lila struct class
        is_lila_struct = False
        if hasattr(ty, "__lila_struct__"):
            is_lila_struct = True
            lila_ty = ty.__name__
        elif "tuple" in ty_str:
            lila_ty = ty_str
        elif "i8" in ty_str:
            lila_ty = "i8"
        elif "u8" in ty_str:
            lila_ty = "u8"
        elif "i16" in ty_str:
            lila_ty = "i16"
        elif "u16" in ty_str:
            lila_ty = "u16"
        elif "i32" in ty_str:
            lila_ty = "i32"
        elif "u32" in ty_str:
            lila_ty = "u32"
        elif "i64" in ty_str or "int" in ty_str:
            lila_ty = "i64"
        elif "u64" in ty_str:
            lila_ty = "u64"
        elif "f32" in ty_str:
            lila_ty = "f32"
        elif "f64" in ty_str or "float" in ty_str:
            lila_ty = "f64"
        elif "bool" in ty_str:
            lila_ty = "bool"
        elif "tuple" in ty_str:
            lila_ty = ty_str
        else:
            lila_ty = "unknown"

        field_list.append((name, lila_ty))
        field_types[name] = ty

        # Build ctypes field
        if lila_ty in TYPE_MAP:
            ctypes_fields.append((name, TYPE_MAP[lila_ty]))
        elif is_lila_struct:
            # Inline struct!
            ctypes_fields.append((name, ty.__lila_ctypes__))
        else:
            ctypes_fields.append((name, ctypes.c_void_p))

    # Generate the ctypes Structure class dynamically
    class LilaCtypesStruct(ctypes.Structure):
        _fields_ = ctypes_fields

    cls.__lila_struct__ = True
    cls.__lila_fields__ = field_list
    cls.__lila_ctypes__ = LilaCtypesStruct

    # Add a constructor that accepts values for fields
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # Resolve all arguments to their ctypes representation
        processed_args = []
        for arg in args:
            if hasattr(arg, "_ctypes_obj"):
                processed_args.append(arg._ctypes_obj)
            else:
                processed_args.append(arg)

        processed_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "_ctypes_obj"):
                processed_kwargs[k] = v._ctypes_obj
            else:
                processed_kwargs[k] = v

        self._ctypes_obj = LilaCtypesStruct(*processed_args, **processed_kwargs)
        if original_init is not object.__init__:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    # Wrap properties to access ctypes fields
    for field_name, _ in ctypes_fields:
        field_type = field_types[field_name]

        def make_accessors(fname, ftype):
            def getter(self):
                val = getattr(self._ctypes_obj, fname)
                # If it's a nested struct, wrap the raw ctypes buffer in its Lila class
                if hasattr(ftype, "__lila_struct__"):
                    wrapper = ftype.__new__(ftype)
                    wrapper._ctypes_obj = val
                    return wrapper
                return val

            def setter(self, val):
                # If setting an Lila object, extract its underlying ctypes object
                if hasattr(val, "_ctypes_obj"):
                    setattr(self._ctypes_obj, fname, val._ctypes_obj)
                else:
                    setattr(self._ctypes_obj, fname, val)

            return getter, setter

        get, set = make_accessors(field_name, field_type)
        setattr(cls, field_name, property(get, set))

    # Decorate all custom methods with @verify automatically
    from .compiler import verify

    for attr_name, attr_value in list(vars(cls).items()):
        if callable(attr_value) and not attr_name.startswith("__"):
            # It's a regular method, let's verify and JIT compile it!
            # Use ClassName_MethodName to avoid collisions in the global registry
            prefixed_name = f"{cls.__name__}_{attr_name}"
            jitted = verify(
                _struct_layouts={cls.__name__: cls.__lila_fields__},
                _class_name=cls.__name__,
                _method_name=prefixed_name,
            )(attr_value)
            setattr(cls, attr_name, jitted)

    return cls


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

    for idx, (name, ty) in enumerate(fields.items()):
        variant_names.append(name)
        variant_types[name] = ty
        if hasattr(ty, "__lila_ctypes__"):
            union_fields.append((name, ty.__lila_ctypes__))
        else:
            raise TypeError(f"Enum variant {name} must be a @struct")

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
            @classmethod
            def constructor(cls_ref, *args, **kwargs):
                instance = cls_ref.__new__(cls_ref)
                instance._ctypes_obj = LilaCtypesEnum()
                instance._ctypes_obj.tag = tag_idx
                # Construct the payload struct
                payload_instance = variant_type(*args, **kwargs)
                setattr(
                    instance._ctypes_obj.payload,
                    variant_name,
                    payload_instance._ctypes_obj,
                )
                return instance

            def is_variant(self):
                return self._ctypes_obj.tag == tag_idx

            def as_variant(self):
                if self._ctypes_obj.tag != tag_idx:
                    raise ValueError(
                        f"Tried to access Enum as {variant_name} but tag is {self._ctypes_obj.tag}"
                    )
                wrapper = variant_type.__new__(variant_type)
                wrapper._ctypes_obj = getattr(self._ctypes_obj.payload, variant_name)
                return wrapper

            return constructor, is_variant, as_variant

        ctor, is_var, as_var = make_variant_methods(name, ty, idx)
        setattr(cls, name, ctor)
        setattr(cls, f"is_{name}", is_var)
        setattr(cls, f"as_{name}", as_var)

    return cls


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
            return cls(params, None)
        return cls(params[0], params[1])


class Callable:
    """Alias for FnPointer for better Python compatibility."""

    def __class_getitem__(cls, params):
        return FnPointer.__class_getitem__(params)


class Closure(FnPointer):
    """
    Represents a closure (function pointer + environment).
    Usage: f: Closure[[i64], i64]
    """

    pass


__all__ = [
    "Hand",
    "Peek",
    "Held",
    "struct",
    "enum",
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
    "Refined",
    "SizedArray",
    "Buffer",
    "FnPointer",
    "Callable",
    "Closure",
    "TYPE_MAP",
]
