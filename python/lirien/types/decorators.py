import ctypes
import inspect
import sys
from typing import get_origin, get_args
from .base import TYPE_MAP
from .memory import Box


def struct(cls):
    """
    Decorator to mark a class as a flat Lirien Struct.
    Generates a ctypes Structure behind the scenes.
    """
    from ..decorators import verify
    from ..signatures import _get_type_name

    # Check if this is a Generic struct
    typevars = set()
    if hasattr(cls, "__parameters__"):
        typevars = set(cls.__parameters__)

    if typevars:

        class GenericStruct:
            @classmethod
            def __class_getitem__(cls_ref, params):
                if not isinstance(params, tuple):
                    params = (params,)

                if len(params) != len(typevars):
                    raise TypeError(
                        f"Generic struct {cls.__name__} expects {len(typevars)} type parameters, got {len(params)}"
                    )

                mapping = {
                    tvar.__name__: param for tvar, param in zip(typevars, params)
                }
                param_names = []
                for p in params:
                    if hasattr(p, "__name__"):
                        param_names.append(p.__name__)
                    else:
                        param_names.append(str(p))

                specialized_name = f"{cls.__name__}_{'_'.join(param_names)}"

                # Create specialized class
                specialized_cls = type(
                    specialized_name, (cls,), {"__module__": cls.__module__}
                )
                specialized_cls.__lirien_specialized__ = True
                specialized_cls.__lirien_origin__ = cls
                specialized_cls.__lirien_params__ = params

                # Update annotations with substituted types
                orig_annotations = getattr(cls, "__annotations__", {})
                new_annotations = {}
                for name, ty in orig_annotations.items():
                    if hasattr(ty, "__name__") and ty.__name__ in mapping:
                        new_annotations[name] = mapping[ty.__name__]
                    elif str(ty) in mapping:
                        new_annotations[name] = mapping[str(ty)]
                    else:
                        new_annotations[name] = ty

                specialized_cls.__annotations__ = new_annotations
                # Re-apply @struct to the specialized class
                return struct(specialized_cls)

        return GenericStruct

    fields = getattr(cls, "__annotations__", {})
    field_list = []
    ctypes_fields = []

    for name, ty in fields.items():
        actual_ty = getattr(ty, "base_type", ty)
        ty_str = str(actual_ty).lower()

        if hasattr(actual_ty, "__lirien_struct__"):
            field_list.append((name, actual_ty))
            ctypes_fields.append((name, actual_ty.__lirien_ctypes__))
            continue

        c_ty = ctypes.c_int64
        found = False
        for type_name, ct in TYPE_MAP.items():
            if type_name in ty_str:
                c_ty = ct
                found = True
                break

        if not found:
            c_ty = ctypes.c_void_p

        field_list.append((name, actual_ty))
        ctypes_fields.append((name, c_ty))

    class LirienCtypesStruct(ctypes.Structure):
        _fields_ = ctypes_fields

    cls.__lirien_struct__ = True
    cls.__lirien_fields__ = field_list
    cls.__lirien_ctypes__ = LirienCtypesStruct
    cls.__match_args__ = tuple(name for name, _ in field_list)

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        processed_args = []
        processed_kwargs = {}

        for i, arg in enumerate(args):
            if hasattr(arg, "_ctypes_obj"):
                processed_args.append(arg._ctypes_obj)
            else:
                processed_args.append(arg)

        for key, value in kwargs.items():
            if hasattr(value, "_ctypes_obj"):
                processed_kwargs[key] = value._ctypes_obj
            else:
                processed_kwargs[key] = value

        self._ctypes_obj = LirienCtypesStruct(*processed_args, **processed_kwargs)
        if original_init is not object.__init__:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    struct_layout = {
        cls.__name__: [(f_name, _get_type_name(f_ty)) for f_name, f_ty in field_list]
    }
    methods = []
    for name, method in inspect.getmembers(cls):
        if name.startswith("__") and name.endswith("__"):
            continue
        if name == "new_init":
            continue
        if inspect.isfunction(method) or hasattr(method, "__lirien_jit__"):
            methods.append((name, method))

    methods.sort(key=lambda x: x[1].__code__.co_firstlineno)

    for name, method in methods:
        if hasattr(method, "__lirien_jit__"):
            if hasattr(method, "class_name") and method.class_name is None:
                method.class_name = cls.__name__
                method.method_name = f"{cls.__name__}_{name}"
                method.struct_layouts = struct_layout
            continue

        verified_method = verify(
            _class_name=cls.__name__,
            _method_name=f"{cls.__name__}_{name}",
            _struct_layouts=struct_layout,
        )(method)
        setattr(cls, name, verified_method)

    for name, ftype in field_list:

        def make_getter(fname, fty):
            def getter(self_ref):
                val = getattr(self_ref._ctypes_obj, fname)
                if hasattr(fty, "__lirien_struct__"):
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

    def struct_repr(self):
        field_strs = []
        for fname, _ in field_list:
            val = getattr(self, fname)
            field_strs.append(f"{fname}={repr(val)}")
        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    def struct_eq(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return all(
            getattr(self, fname) == getattr(other, fname) for fname, _ in field_list
        )

    cls.__repr__ = struct_repr
    cls.__eq__ = struct_eq

    return cls


value = struct


def enum(cls):
    """
    Decorator to mark a class as a Tagged Union (Enum) for Lirien.
    Generates a ctypes Structure with a tag and a Union payload.
    """
    from ..signatures import _get_type_name

    # Check if this is a Generic ADT
    typevars = set()
    if hasattr(cls, "__parameters__"):
        typevars = set(cls.__parameters__)

    if typevars:

        class GenericADT:
            @classmethod
            def __class_getitem__(cls_ref, params):
                if not isinstance(params, tuple):
                    params = (params,)

                if len(params) != len(typevars):
                    raise TypeError(
                        f"Generic ADT {cls.__name__} expects {len(typevars)} type parameters, got {len(params)}"
                    )

                mapping = {
                    tvar.__name__: param for tvar, param in zip(typevars, params)
                }
                param_names = []
                for p in params:
                    if hasattr(p, "__name__"):
                        param_names.append(p.__name__)
                    else:
                        param_names.append(str(p))

                specialized_name = f"{cls.__name__}_{'_'.join(param_names)}"

                # Create specialized class
                specialized_cls = type(
                    specialized_name, (cls,), {"__module__": cls.__module__}
                )
                specialized_cls.__lirien_specialized__ = True
                specialized_cls.__lirien_origin__ = cls
                specialized_cls.__lirien_params__ = params

                # Update annotations with substituted types
                orig_annotations = getattr(cls, "__annotations__", {})
                new_annotations = {}

                def _substitute_types(ty):
                    if hasattr(ty, "__name__") and ty.__name__ in mapping:
                        return mapping[ty.__name__]
                    if str(ty) in mapping:
                        return mapping[str(ty)]

                    # Handle Box (represented as Annotated[Box, "Name"])
                    if hasattr(ty, "__metadata__"):
                        origin = getattr(ty, "__origin__", None)
                        if origin is Box or (
                            hasattr(origin, "__name__") and origin.__name__ == "Box"
                        ):
                            inner = ty.__metadata__[0]
                            if isinstance(inner, str):
                                # If it's a recursive reference to the current generic class
                                if "[" in inner:
                                    # e.g. "List[T]"
                                    origin_name = inner.split("[")[0]
                                    if origin_name == cls.__name__:
                                        return Box[specialized_cls]
                                elif inner == cls.__name__:
                                    return Box[specialized_cls]

                                # Check mapping for other generic origins
                                for k, v in mapping.items():
                                    if k in inner:
                                        if getattr(v, "__lirien_specialized__", False):
                                            return Box[v]

                                return Box[inner]

                            return Box[_substitute_types(inner)]

                    if isinstance(ty, tuple):
                        return tuple(_substitute_types(elt) for elt in ty)

                    origin = get_origin(ty)
                    if origin:
                        args = get_args(ty)
                        if args:
                            new_args = tuple(_substitute_types(arg) for arg in args)
                            return origin[new_args]

                    return ty

                for name, ty in orig_annotations.items():
                    new_annotations[name] = _substitute_types(ty)

                specialized_cls.__annotations__ = new_annotations
                specialized_cls.__lirien_variant_types__ = new_annotations
                return enum(specialized_cls)

        return GenericADT

    fields = getattr(cls, "__annotations__", {})
    variant_names = []
    variant_types = {}
    union_fields = []

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
            union_fields.append(
                (name, make_payload_struct(f"{cls.__name__}_{name}_payload", []))
            )
        elif hasattr(ty, "__metadata__") and "Box" in str(ty.__origin__):
            union_fields.append((name, ctypes.c_void_p))
        elif hasattr(ty, "__lirien_ctypes__"):
            union_fields.append((name, ty.__lirien_ctypes__))
        elif (
            origin is tuple
            or (typing_tuple and origin is typing_tuple)
            or isinstance(actual_ty, tuple)
        ):
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

    class LirienCtypesUnion(ctypes.Union):
        _fields_ = union_fields

    class LirienCtypesEnum(ctypes.Structure):
        _fields_ = [("tag", ctypes.c_uint8), ("payload", LirienCtypesUnion)]

    cls.__lirien_enum__ = True
    cls.__lirien_variants__ = variant_names
    cls.__lirien_variant_types__ = variant_types
    cls.__lirien_ctypes__ = LirienCtypesEnum

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        self._ctypes_obj = LirienCtypesEnum()
        if original_init is not object.__init__:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    enum_layout = {
        cls.__name__: [
            (v_name, _get_type_name(v_ty)) for v_name, v_ty in variant_types.items()
        ]
    }
    methods = []
    for name, method in inspect.getmembers(cls):
        if name.startswith("__") and name.endswith("__"):
            continue
        if name == "new_init":
            continue
        if inspect.isfunction(method) or hasattr(method, "__lirien_jit__"):
            methods.append((name, method))

    methods.sort(key=lambda x: x[1].__code__.co_firstlineno)

    from ..decorators import verify

    for name, method in methods:
        if hasattr(method, "__lirien_jit__"):
            if hasattr(method, "class_name") and method.class_name is None:
                method.class_name = cls.__name__
                method.method_name = f"{cls.__name__}_{name}"
                method.enum_layouts = enum_layout
            continue

        verified_method = verify(
            _class_name=cls.__name__,
            _method_name=f"{cls.__name__}_{name}",
            _enum_layouts=enum_layout,
        )(method)
        setattr(cls, name, verified_method)

    for idx, (name, ty) in enumerate(variant_types.items()):

        def make_variant_methods(variant_name, variant_type, tag_idx):
            payload_cty = dict(union_fields)[variant_name]

            @classmethod
            def constructor(cls_ref, *args, **kwargs):
                instance = cls_ref.__new__(cls_ref)
                instance._ctypes_obj = LirienCtypesEnum()
                instance._ctypes_obj.tag = tag_idx

                if variant_type is None:
                    pass
                elif hasattr(variant_type, "__metadata__") and "Box" in str(
                    variant_type.__origin__
                ):
                    payload_instance = args[0]
                    if hasattr(payload_instance, "_ctypes_obj"):
                        ptr = ctypes.cast(
                            ctypes.pointer(payload_instance._ctypes_obj),
                            ctypes.c_void_p,
                        )
                        setattr(instance._ctypes_obj.payload, variant_name, ptr)
                elif hasattr(variant_type, "__lirien_ctypes__"):
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
                    payload_obj = payload_cty()
                    for i, arg in enumerate(args):
                        if i < len(variant_type):
                            v_ty = variant_type[i]
                            if hasattr(v_ty, "__metadata__") and "Box" in str(
                                v_ty.__origin__
                            ):
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
                    inner_ty = variant_type.__metadata__[0]
                    if isinstance(inner_ty, str):
                        module = sys.modules.get(cls.__module__)
                        if module and hasattr(module, inner_ty):
                            inner_ty = getattr(module, inner_ty)
                        elif inner_ty == cls.__name__:
                            inner_ty = cls
                    if hasattr(inner_ty, "__lirien_ctypes__"):
                        wrapper = inner_ty.__new__(inner_ty)
                        wrapper._ctypes_obj = ctypes.cast(
                            raw_payload, ctypes.POINTER(inner_ty.__lirien_ctypes__)
                        ).contents
                        return wrapper
                    return raw_payload
                elif hasattr(variant_type, "__lirien_ctypes__"):
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
                            if isinstance(inner_ty, str) and inner_ty == cls.__name__:
                                inner_ty = cls
                            if hasattr(inner_ty, "__lirien_ctypes__"):
                                wrapper = inner_ty.__new__(inner_ty)
                                wrapper._ctypes_obj = ctypes.cast(
                                    raw_val, ctypes.POINTER(inner_ty.__lirien_ctypes__)
                                ).contents
                                res.append(wrapper)
                            else:
                                res.append(raw_val)
                        else:
                            res.append(raw_val)
                    return tuple(res)
                else:
                    return raw_payload.val

            return constructor, is_variant, as_variant

        ctor, is_var, as_var = make_variant_methods(name, ty, idx)
        setattr(cls, name, ctor)
        setattr(cls, f"is_{name}", is_var)
        setattr(cls, f"as_{name}", as_var)

    def enum_repr(self):
        tag = self._ctypes_obj.tag
        variants = getattr(self.__class__, "__lirien_variants__", [])
        if tag < 0 or tag >= len(variants):
            return f"{self.__class__.__name__}(<invalid tag {tag}>)"
        v_name = variants[tag]
        variant_types = getattr(self.__class__, "__lirien_variant_types__", {})
        v_ty = variant_types.get(v_name)
        if v_ty is None:
            return f"{self.__class__.__name__}.{v_name}()"
        as_var_fn = getattr(self, f"as_{v_name}")
        payload = as_var_fn()
        if isinstance(v_ty, tuple):
            return f"{self.__class__.__name__}.{v_name}{repr(payload)}"
        else:
            return f"{self.__class__.__name__}.{v_name}({repr(payload)})"

    def enum_eq(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self._ctypes_obj.tag != other._ctypes_obj.tag:
            return False
        tag = self._ctypes_obj.tag
        variants = getattr(self.__class__, "__lirien_variants__", [])
        v_name = variants[tag]
        variant_types = getattr(self.__class__, "__lirien_variant_types__", {})
        v_ty = variant_types.get(v_name)
        if v_ty is None:
            return True
        as_var_fn_self = getattr(self, f"as_{v_name}")
        as_var_fn_other = getattr(other, f"as_{v_name}")
        return as_var_fn_self() == as_var_fn_other()

    cls.__repr__ = enum_repr
    cls.__eq__ = enum_eq

    return cls


adt = enum
