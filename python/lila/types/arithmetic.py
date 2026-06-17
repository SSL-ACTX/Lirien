import typing


class TypeExpr:
    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __add__(self, other):
        return TypeExpr("+", (self, other))

    def __sub__(self, other):
        return TypeExpr("-", (self, other))

    def __mul__(self, other):
        return TypeExpr("*", (self, other))

    def __floordiv__(self, other):
        return TypeExpr("//", (self, other))

    def __radd__(self, other):
        return TypeExpr("+", (other, self))

    def __rsub__(self, other):
        return TypeExpr("-", (other, self))

    def __rmul__(self, other):
        return TypeExpr("*", (other, self))

    def __rfloordiv__(self, other):
        return TypeExpr("//", (other, self))

    def evaluate(self, mapping):
        eval_args = []
        for arg in self.args:
            if isinstance(arg, TypeExpr):
                eval_args.append(arg.evaluate(mapping))
            elif hasattr(arg, "__lila_typevar__"):
                # It's a LilaTypeVar
                name = arg.__name__
                val = mapping.get(name, mapping.get(arg, arg))
                eval_args.append(val)
            elif isinstance(arg, typing.TypeVar):
                val = mapping.get(arg.__name__, mapping.get(arg, arg))
                eval_args.append(val)
            else:
                eval_args.append(arg)

        a, b = eval_args
        if (
            isinstance(a, (typing.TypeVar, TypeExpr))
            or isinstance(b, (typing.TypeVar, TypeExpr))
            or hasattr(a, "__lila_typevar__")
            or hasattr(b, "__lila_typevar__")
        ):
            return self

        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "//":
            return a // b
        return self


class LilaTypeVar:
    def __init__(self, name, *args, **kwargs):
        self._tvar = typing.TypeVar(name, *args, **kwargs)
        self.__lila_typevar__ = True
        self.__name__ = name

    def __add__(self, other):
        return TypeExpr("+", (self, other))

    def __sub__(self, other):
        return TypeExpr("-", (self, other))

    def __mul__(self, other):
        return TypeExpr("*", (self, other))

    def __floordiv__(self, other):
        return TypeExpr("//", (self, other))

    def __radd__(self, other):
        return TypeExpr("+", (other, self))

    def __rsub__(self, other):
        return TypeExpr("-", (other, self))

    def __rmul__(self, other):
        return TypeExpr("*", (other, self))

    def __rfloordiv__(self, other):
        return TypeExpr("//", (other, self))

    def __getattr__(self, name):
        return getattr(self._tvar, name)

    def __repr__(self):
        return f"LilaTypeVar({self.__name__})"


if typing.TYPE_CHECKING:
    # IDEs and Mypy will see the standard TypeVar
    TypeVar = typing.TypeVar
else:
    # At runtime, Python uses our arithmetic-capable wrapper
    def TypeVar(name, *args, **kwargs):
        return LilaTypeVar(name, *args, **kwargs)
