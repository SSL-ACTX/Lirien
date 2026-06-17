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

    def __lt__(self, other):
        return TypeExpr("<", (self, other))

    def __le__(self, other):
        return TypeExpr("<=", (self, other))

    def __gt__(self, other):
        return TypeExpr(">", (self, other))

    def __ge__(self, other):
        return TypeExpr(">=", (self, other))

    def __and__(self, other):
        return TypeExpr("&", (self, other))

    def __or__(self, other):
        return TypeExpr("|", (self, other))

    def __radd__(self, other):
        return TypeExpr("+", (other, self))

    def __rsub__(self, other):
        return TypeExpr("-", (other, self))

    def __rmul__(self, other):
        return TypeExpr("*", (other, self))

    def __rfloordiv__(self, other):
        return TypeExpr("//", (other, self))

    def __rlt__(self, other):
        return TypeExpr("<", (other, self))

    def __rle__(self, other):
        return TypeExpr("<=", (other, self))

    def __rgt__(self, other):
        return TypeExpr(">", (other, self))

    def __rge__(self, other):
        return TypeExpr(">=", (other, self))

    def __rand__(self, other):
        return TypeExpr("&", (other, self))

    def __ror__(self, other):
        return TypeExpr("|", (other, self))

    def evaluate(self, mapping):
        eval_args = []
        for arg in self.args:
            if isinstance(arg, TypeExpr):
                eval_args.append(arg.evaluate(mapping))
            elif hasattr(arg, "__lila_typevar__"):
                # It's a LilaTypeVar
                name = arg.__name__
                # Try name first, then object itself
                if name in mapping:
                    eval_args.append(mapping[name])
                elif arg in mapping:
                    eval_args.append(mapping[arg])
                else:
                    eval_args.append(arg)
            elif isinstance(arg, typing.TypeVar):
                name = arg.__name__
                if name in mapping:
                    eval_args.append(mapping[name])
                elif arg in mapping:
                    eval_args.append(mapping[arg])
                else:
                    eval_args.append(arg)
            else:
                eval_args.append(arg)

        a, b = eval_args
        if (
            isinstance(a, (typing.TypeVar, TypeExpr))
            or isinstance(b, (typing.TypeVar, TypeExpr))
            or hasattr(a, "__lila_typevar__")
            or hasattr(b, "__lila_typevar__")
        ):
            return TypeExpr(self.op, (a, b))

        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "//":
            return a // b
        if self.op == "<":
            return a < b
        if self.op == "<=":
            return a <= b
        if self.op == ">":
            return a > b
        if self.op == ">=":
            return a >= b
        if self.op == "&":
            return a & b
        if self.op == "|":
            return a | b
        return self

    def __bool__(self):
        # Strict evaluation: a TypeExpr cannot be used in a boolean context
        # unless it has been fully evaluated to a concrete literal.
        res = self.evaluate({})
        if isinstance(res, TypeExpr):
            raise RuntimeError(
                f"Symbolic TypeExpr '{self.op}' with args {self.args} cannot be "
                "evaluated to a boolean because it contains unresolved TypeVars. "
                "This usually indicates a logic error in runtime refinement checks."
            )
        return bool(res)

    def __eq__(self, other):
        if not isinstance(other, TypeExpr):
            return False
        return self.op == other.op and self.args == other.args

    def __hash__(self):
        return hash((self.op, self.args))


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

    def __lt__(self, other):
        return TypeExpr("<", (self, other))

    def __le__(self, other):
        return TypeExpr("<=", (self, other))

    def __gt__(self, other):
        return TypeExpr(">", (self, other))

    def __ge__(self, other):
        return TypeExpr(">=", (self, other))

    def __and__(self, other):
        return TypeExpr("&", (self, other))

    def __or__(self, other):
        return TypeExpr("|", (self, other))

    def __radd__(self, other):
        return TypeExpr("+", (other, self))

    def __rsub__(self, other):
        return TypeExpr("-", (other, self))

    def __rmul__(self, other):
        return TypeExpr("*", (other, self))

    def __rfloordiv__(self, other):
        return TypeExpr("//", (other, self))

    def __rlt__(self, other):
        return TypeExpr("<", (other, self))

    def __rle__(self, other):
        return TypeExpr("<=", (other, self))

    def __rgt__(self, other):
        return TypeExpr(">", (other, self))

    def __rge__(self, other):
        return TypeExpr(">=", (other, self))

    def __rand__(self, other):
        return TypeExpr("&", (other, self))

    def __ror__(self, other):
        return TypeExpr("|", (other, self))

    def __getattr__(self, name):
        return getattr(self._tvar, name)

    def __repr__(self):
        return f"LilaTypeVar({self.__name__})"

    def __eq__(self, other):
        if hasattr(other, "__lila_typevar__"):
            return self.__name__ == other.__name__
        if isinstance(other, typing.TypeVar):
            return self.__name__ == other.__name__
        return False

    def __hash__(self):
        return hash(self.__name__)


if typing.TYPE_CHECKING:
    # IDEs and Mypy will see the standard TypeVar
    TypeVar = typing.TypeVar
else:
    # At runtime, Python uses our arithmetic-capable wrapper
    def TypeVar(name, *args, **kwargs):
        return LilaTypeVar(name, *args, **kwargs)
