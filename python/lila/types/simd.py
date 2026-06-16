import ctypes
from .base import TYPE_MAP


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
        return f32x4(*(self[i] * other for i in range(4)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return f32x4(*(self[i] / other for i in range(4)))
        return f32x4(*(self[i] / other for i in range(4)))


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
        return i32x4(*(self[i] * other for i in range(4)))


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
        return f64x2(*(self[i] * other for i in range(2)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return f64x2(*(self[i] / other for i in range(2)))
        return f64x2(*(self[i] / other for i in range(2)))


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
        return i64x2(*(self[i] * other for i in range(2)))


class i8x16(ctypes.Structure):
    _align_ = 16
    _fields_ = [(f"f{i}", ctypes.c_int8) for i in range(16)]

    def __init__(self, *args):
        if len(args) == 16:
            for i, val in enumerate(args):
                setattr(self, f"f{i}", val)
        elif len(args) == 1:
            for i in range(16):
                setattr(self, f"f{i}", args[0])

    def __getitem__(self, idx):
        return [getattr(self, f"f{i}") for i in range(16)][idx]

    def __add__(self, other):
        return i8x16(*(self[i] + other[i] for i in range(16)))

    def __sub__(self, other):
        return i8x16(*(self[i] - other[i] for i in range(16)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return i8x16(*(self[i] * other for i in range(16)))
        return i8x16(*(self[i] * other for i in range(16)))


class u8x16(ctypes.Structure):
    _align_ = 16
    _fields_ = [(f"f{i}", ctypes.c_uint8) for i in range(16)]

    def __init__(self, *args):
        if len(args) == 16:
            for i, val in enumerate(args):
                setattr(self, f"f{i}", val)
        elif len(args) == 1:
            for i in range(16):
                setattr(self, f"f{i}", args[0])

    def __getitem__(self, idx):
        return [getattr(self, f"f{i}") for i in range(16)][idx]

    def __add__(self, other):
        return u8x16(*(self[i] + other[i] for i in range(16)))

    def __sub__(self, other):
        return u8x16(*(self[i] - other[i] for i in range(16)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return u8x16(*(self[i] * other for i in range(16)))
        return u8x16(*(self[i] * other for i in range(16)))


class i16x8(ctypes.Structure):
    _align_ = 16
    _fields_ = [(f"f{i}", ctypes.c_int16) for i in range(8)]

    def __init__(self, *args):
        if len(args) == 8:
            for i, val in enumerate(args):
                setattr(self, f"f{i}", val)
        elif len(args) == 1:
            for i in range(8):
                setattr(self, f"f{i}", args[0])

    def __getitem__(self, idx):
        return [getattr(self, f"f{i}") for i in range(8)][idx]

    def __add__(self, other):
        return i16x8(*(self[i] + other[i] for i in range(8)))

    def __sub__(self, other):
        return i16x8(*(self[i] - other[i] for i in range(8)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return i16x8(*(self[i] * other for i in range(8)))
        return i16x8(*(self[i] * other for i in range(8)))


class u16x8(ctypes.Structure):
    _align_ = 16
    _fields_ = [(f"f{i}", ctypes.c_uint16) for i in range(8)]

    def __init__(self, *args):
        if len(args) == 8:
            for i, val in enumerate(args):
                setattr(self, f"f{i}", val)
        elif len(args) == 1:
            for i in range(8):
                setattr(self, f"f{i}", args[0])

    def __getitem__(self, idx):
        return [getattr(self, f"f{i}") for i in range(8)][idx]

    def __add__(self, other):
        return u16x8(*(self[i] + other[i] for i in range(8)))

    def __sub__(self, other):
        return u16x8(*(self[i] - other[i] for i in range(8)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return u16x8(*(self[i] * other for i in range(8)))
        return u16x8(*(self[i] * other for i in range(8)))


TYPE_MAP.update(
    {
        "f32x4": f32x4,
        "i32x4": i32x4,
        "f64x2": f64x2,
        "i64x2": i64x2,
        "i8x16": i8x16,
        "u8x16": u8x16,
        "i16x8": i16x8,
        "u16x8": u16x8,
    }
)
