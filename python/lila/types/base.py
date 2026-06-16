import ctypes


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
    "box": ctypes.c_void_p,
    "pointer": ctypes.c_void_p,
    "nullable": ctypes.c_void_p,
    "optional": ctypes.c_void_p,
}
