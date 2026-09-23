import ast
from pathlib import Path


def test_verifier_cannot_import_benchmark_dependencies():
    root = Path(__file__).resolve().parents[2] / "src/etv"
    forbidden = {"etv_bench", "torch", "ntops", "ninetoothed", "numpy"}
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not {name.name.split(".")[0] for name in node.names} & forbidden, path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, path
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                if node.args and isinstance(node.args[0], ast.Constant):
                    assert node.args[0].value.split(".")[0] not in forbidden, path
