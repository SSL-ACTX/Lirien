import ast
import inspect
from typing import Any, Dict


class TypeSubstitutor(ast.NodeTransformer):
    """AST visitor to replace TypeVar names with concrete type names or literals."""

    def __init__(self, mapping: Dict[str, Any]):
        self.mapping = mapping

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                name = node.args[0].id
                if name in self.mapping:
                    val = self.mapping[name]
                    if isinstance(val, (list, tuple)):
                        return ast.Constant(value=len(val))
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            l, r = node.left.value, node.right.value
            if isinstance(l, (int, float)) and isinstance(r, (int, float)):
                res = None
                if isinstance(node.op, ast.Add):
                    res = l + r
                elif isinstance(node.op, ast.Sub):
                    res = l - r
                elif isinstance(node.op, ast.Mult):
                    res = l * r
                elif isinstance(node.op, ast.FloorDiv):
                    res = l // r
                elif isinstance(node.op, ast.Div):
                    res = l / r
                elif isinstance(node.op, ast.Mod):
                    res = l % r
                elif isinstance(node.op, ast.Pow):
                    res = l**r

                if res is not None:
                    return ast.Constant(value=res)
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.operand, ast.Constant):
            val = node.operand.value
            if isinstance(val, (int, float)):
                if isinstance(node.op, ast.USub):
                    return ast.Constant(value=-val)
                elif isinstance(node.op, ast.UAdd):
                    return ast.Constant(value=val)
        return node

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.mapping:
            val = self.mapping[node.value.id]
            if getattr(val, "__lirien_specialized__", False):
                return ast.Name(id=val.__name__, ctx=node.ctx)

        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in self.mapping:
            val = self.mapping[node.id]
            if isinstance(val, int):
                return ast.Constant(value=val)
            name = getattr(val, "__name__", str(val))
            return ast.Name(id=name, ctx=node.ctx)
        return self.generic_visit(node)


class EllipsisExpander(ast.NodeTransformer):
    """AST visitor to expand Ellipsis annotations and specialize calls recursively."""

    def __init__(self, mapping, scope, parent_mf):
        self.mapping = mapping
        self.scope = scope
        self.parent_mf = parent_mf
        self.current_param = None

    def visit_arg(self, node):
        old_param = self.current_param
        self.current_param = node.arg
        node.annotation = self.visit(node.annotation) if node.annotation else None
        self.current_param = old_param
        return node

    def visit_FunctionDef(self, node):
        node.args = self.visit(node.args)
        old_param = self.current_param
        self.current_param = "return"
        if node.returns:
            node.returns = self.visit(node.returns)
        self.current_param = old_param
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_Call(self, node):
        from ..decorators import MonomorphizedFunction
        from .signature_helpers import _get_all_typevars

        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            func_obj = self.scope.get(func_name)
            if not func_obj and func_name == self.parent_mf.func.__name__:
                func_obj = self.parent_mf

            if isinstance(func_obj, MonomorphizedFunction):
                call_mapping = {}
                callee_sig = inspect.signature(func_obj.func)
                callee_typevars = _get_all_typevars(callee_sig)
                callee_tvar_names = {t.__name__ for t in callee_typevars}

                for k, v in self.mapping.items():
                    if k in callee_tvar_names:
                        call_mapping[k] = v

                params = list(callee_sig.parameters.items())
                for i, arg in enumerate(node.args):
                    if i < len(params):
                        p_name, _ = params[i]
                        if isinstance(arg, ast.Name):
                            mapping_key = f"__callable_{arg.id}"
                            if mapping_key in self.mapping:
                                call_mapping[f"__callable_{p_name}"] = self.mapping[
                                    mapping_key
                                ]

                if call_mapping:
                    specialized_callee = func_obj._get_specialized_name(call_mapping)
                    node.func.id = specialized_callee

                    call_key = tuple(
                        sorted(
                            (k, tuple(v) if isinstance(v, list) else v)
                            for k, v in call_mapping.items()
                        )
                    )

                    if call_key not in func_obj.cache:
                        func_obj.cache[call_key] = None
                        func_obj.cache[call_key] = func_obj._specialize(call_mapping)

        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in (
            "Tensor",
            "SizedArray",
            "Buffer",
        ):
            if isinstance(node.slice, ast.Constant) and node.slice.value is Ellipsis:
                ellipsis_key = f"__ellipsis_{self.current_param}"
                if ellipsis_key in self.mapping:
                    type_name = self.mapping[ellipsis_key][0]
                    node.slice = ast.Name(id=type_name, ctx=ast.Load())
                return node

            if isinstance(node.slice, ast.Tuple):
                new_elts = []
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Constant) and elt.value is Ellipsis:
                        ellipsis_key = f"__ellipsis_{self.current_param}"
                        if ellipsis_key in self.mapping:
                            for dim in self.mapping[ellipsis_key]:
                                new_elts.append(ast.Constant(value=dim))
                        else:
                            for k, v in self.mapping.items():
                                if k.startswith("__ellipsis_"):
                                    for dim in v:
                                        new_elts.append(ast.Constant(value=dim))
                                    break
                    elif (
                        isinstance(elt, ast.Subscript)
                        and isinstance(elt.value, ast.Name)
                        and elt.value.id == "Unpack"
                    ):
                        if isinstance(elt.slice, ast.Name):
                            shape_name = elt.slice.id
                            if shape_name in self.mapping:
                                dims = self.mapping[shape_name]
                                if isinstance(dims, (list, tuple)):
                                    for dim in dims:
                                        new_elts.append(ast.Constant(value=dim))
                    else:
                        new_elts.append(elt)
                node.slice.elts = new_elts

        if isinstance(node.value, ast.Name) and node.value.id in (
            "Callable",
            "Closure",
            "FnPointer",
        ):
            callable_key = f"__callable_{self.current_param}"
            if callable_key in self.mapping:
                target_val = self.mapping[callable_key]
                target_name = getattr(target_val, "__name__", str(target_val))
                is_closure = getattr(target_val, "__lirien_closure__", False)

                if node.value.id == "Callable":
                    node.value.id = "Closure" if is_closure else "FnPointer"

                is_generic = (
                    target_name in ("jit_call", "wrapper") or "at 0x" in target_name
                )

                if not is_generic:
                    if isinstance(node.slice, ast.Tuple):
                        if len(node.slice.elts) == 2:
                            node.slice.elts.append(ast.Constant(value=target_name))
                        elif len(node.slice.elts) > 2:
                            node.slice.elts[2] = ast.Constant(value=target_name)
                    elif isinstance(node.slice, (ast.List, ast.Constant, ast.Name)):
                        node.slice = ast.Tuple(
                            elts=[node.slice, ast.Constant(value=target_name)],
                            ctx=ast.Load(),
                        )

        return node


class RefinementSanitizer(ast.NodeTransformer):
    """AST visitor to rewrite Annotated and Refinement type annotations inline."""

    def __init__(self, sig, sanitize_all_types: bool = False):
        self.sig = sig
        self.sanitize_all_types = sanitize_all_types

    def process_ann(self, ann, name_hint):
        from .signature_helpers import (
            _get_refinement_parts,
            _get_type_name,
            _clean_lambda_source,
        )

        if ann is inspect.Parameter.empty:
            return None
        base_ty, predicate = _get_refinement_parts(ann)
        if base_ty is not None and predicate is not None:
            base_name = _get_type_name(base_ty)
            pred_src = _clean_lambda_source(predicate)
            ref_str = f"Refined[{base_name}, {pred_src}]"
            try:
                return ast.parse(ref_str).body[0].value
            except Exception:
                try:
                    return ast.parse(base_name).body[0].value
                except Exception:
                    return ast.Name(id=base_name, ctx=ast.Load())

        if self.sanitize_all_types:
            ann_name = _get_type_name(ann)
            try:
                return ast.parse(ann_name).body[0].value
            except Exception:
                return ast.Name(id=ann_name, ctx=ast.Load())

        return None

    def visit_FunctionDef(self, node):
        # Update parameters
        for arg in node.args.args:
            if arg.arg in self.sig.parameters:
                ann = self.sig.parameters[arg.arg].annotation
                new_ann = self.process_ann(ann, arg.arg)
                if new_ann is not None:
                    arg.annotation = new_ann

        # Update return annotation
        if self.sig.return_annotation is not inspect.Signature.empty:
            new_ret = self.process_ann(self.sig.return_annotation, "return")
            if new_ret is not None:
                node.returns = new_ret

        return node
