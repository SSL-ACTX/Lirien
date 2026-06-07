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


class SizedArray:
    """
    Seamless memory-backed array.
    Usage: arr = SizedArray[i64, 5]([1, 2, 3, 4, 5])
    """

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("SizedArray requires [type, size]")

        base_type, size = params
        base_ty_str = getattr(base_type, "__name__", str(base_type)).lower()

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


class Buffer:
    """
    Represents an external memory buffer (e.g. NumPy array).
    Usage: arr: Buffer[i32]
    """

    def __class_getitem__(cls, base_type):
        base_ty_str = getattr(base_type, "__name__", str(base_type)).lower()
        return Annotated[cls, base_ty_str]


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
    cls.__match_args__ = tuple(name for name, _ in field_list)

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

    # Helper to generate a dummy struct for empty or primitive variants
    def make_payload_struct(name, fields):
        class PayloadStruct(ctypes.Structure):
            _fields_ = fields

        PayloadStruct.__name__ = name
        return PayloadStruct

    for idx, (name, ty) in enumerate(fields.items()):
        variant_names.append(name)
        variant_types[name] = ty

        if ty is None:
            # Empty variant
            union_fields.append(
                (name, make_payload_struct(f"{cls.__name__}_{name}_payload", []))
            )
        elif hasattr(ty, "__metadata__") and "Box" in str(ty.__origin__):
            # Boxed variant
            inner_ty = ty.__metadata__[0]
            # Since this might be recursive, we might need a lazy pointer or a late-bound type.
            # ctypes.POINTER(None) can work as a raw pointer.
            union_fields.append((name, ctypes.c_void_p))
        elif hasattr(ty, "__lila_ctypes__"):
            union_fields.append((name, ty.__lila_ctypes__))
        elif isinstance(ty, tuple):
            # Tuple payload
            tuple_fields = []
            for i, t in enumerate(ty):
                if hasattr(t, "__metadata__") and "Box" in str(t.__origin__):
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
                                # Try to resolve from the class's module
                                module = sys.modules.get(cls.__module__)
                                if module and hasattr(module, inner_ty):
                                    inner_ty = getattr(module, inner_ty)
                                elif inner_ty == cls.__name__:
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
