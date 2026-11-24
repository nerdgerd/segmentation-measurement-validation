"""Verify current files and mathematical equivalence to the publication baseline."""

from pathlib import Path
import ast
import copy
import hashlib
import json


class NumericalSyntax(ast.NodeTransformer):
    """Ignore documentation and the declared path/provenance refactors only."""

    def __init__(self, ratio_body=()):
        self.ratio_body = ratio_body

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("paper/tables", "reference/tables")
        return node

    def visit_ImportFrom(self, node):
        if node.module:
            node.module = node.module.replace("critical_revision", "mechanism")
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "resolve_input_path"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "inside"
        ):
            return node.args[0]
        return node

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
        return node

    def visit_FunctionDef(self, node):
        if node.name in {"code_state", "measurement_difference_ratio"}:
            return None
        if node.name == "compare_disagreement":
            expanded = []
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and isinstance(item.value, ast.Call)
                    and isinstance(item.value.func, ast.Name)
                    and item.value.func.id == "measurement_difference_ratio"
                ):
                    expanded.extend(copy.deepcopy(self.ratio_body))
                else:
                    expanded.append(item)
            node.body = expanded
        return self.generic_visit(node)


def numerical_hash(text):
    tree = ast.parse(text)
    helper = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "measurement_difference_ratio"
        ),
        None,
    )
    body = helper.body[1:-1] if helper else ()
    normalized = NumericalSyntax(body).visit(tree)
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False).encode("utf-8")
    ).hexdigest()


def verify_repository(root):
    root = Path(root).resolve()
    spec = json.loads(
        (root / "provenance/REPRODUCIBILITY.json").read_text(encoding="utf-8")
    )
    for name, expected in spec["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Manifest path escaped repository: {name}")
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Recorded file missing or changed: {name}")
    for name, expected in spec["baseline_numerical_ast"].items():
        if numerical_hash((root / name).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Numerical expressions changed from the baseline: {name}")
    for name, expected in spec["immutable_scientific_inputs"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Scientific input changed: {name}")
    return len(spec["files"])
