"""Structural DAG hashing remains semantic and safe across process boundaries."""

from dataclasses import fields
from fractions import Fraction
import os
import pickle
from pathlib import Path
import subprocess
import sys

from etv.ir import FLOAT, Expr, Type, const, var
from etv.report import serialize


def shared_dag(depth):
    expression = var("shared", FLOAT)
    for _ in range(depth):
        expression = Expr("fadd", FLOAT, (expression, expression))
    return expression


def test_shared_dag_hash_visits_each_node_once(monkeypatch):
    original = Type.__hash__
    calls = 0

    def counted(typ):
        nonlocal calls
        calls += 1
        return original(typ)

    monkeypatch.setattr(Type, "__hash__", counted)
    expression = shared_dag(30)
    expected = hash(expression)
    assert calls == 31
    assert hash(expression) == expected
    assert calls == 31
    assert len({expression, expression}) == 1


def test_cache_is_not_a_report_or_equality_field():
    expression = Expr("fadd", FLOAT, (var("x", FLOAT), const(Fraction(1, 3), FLOAT)))
    before = serialize(expression)
    hash(expression)
    assert serialize(expression) == before
    assert [field.name for field in fields(Expr)] == ["op", "type", "args", "attrs"]
    reconstructed = Expr(expression.op, expression.type, expression.args, expression.attrs)
    assert expression == reconstructed
    assert hash(expression) == hash(reconstructed)
    assert len({expression, reconstructed}) == 1


def test_pickle_drops_hash_memo_and_preserves_shared_subexpressions():
    expression = shared_dag(20)
    hash(expression)
    restored = pickle.loads(pickle.dumps(expression))
    assert "_structural_hash" not in restored.__dict__
    assert restored.args[0] is restored.args[1]
    assert restored == expression
    assert hash(restored) == hash(expression)


def test_hash_recomputed_under_new_python_hash_seed(tmp_path):
    expression = shared_dag(12)
    hash(expression)
    path = tmp_path / "expression.pickle"
    path.write_bytes(pickle.dumps(expression))
    code = """
import pickle,sys
from etv.ir import FLOAT,Expr,var
expression=pickle.load(open(sys.argv[1],'rb'))
assert '_structural_hash' not in expression.__dict__
fresh=var('shared',FLOAT)
for _ in range(12): fresh=Expr('fadd',FLOAT,(fresh,fresh))
assert expression==fresh
assert hash(expression)==hash(fresh)
assert {expression: 'value'}[fresh]=='value'
"""
    source = str(Path(__file__).resolve().parents[2] / "src")
    for seed in ("11", "97"):
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=source)
        subprocess.run([sys.executable, "-c", code, str(path)], env=environment, check=True)


def test_structural_equality_visits_shared_pairs_once(monkeypatch):
    left, right = shared_dag(32), shared_dag(32)
    original = Type.__eq__
    calls = 0

    def counted(left_type, right_type):
        nonlocal calls
        calls += 1
        return original(left_type, right_type)

    monkeypatch.setattr(Type, "__eq__", counted)
    assert left == right
    assert calls <= 33


def test_hash_collision_does_not_imply_equality(monkeypatch):
    monkeypatch.setattr(Expr, "__hash__", lambda self: 7)
    x, y = var("x", FLOAT), var("y", FLOAT)
    assert x != y
    left = Expr("fadd", FLOAT, (x, x))
    right = Expr("fadd", FLOAT, (x, y))
    assert left != right
    assert len({left, right}) == 2
    assert left != object()


def test_substitute_prunes_unaffected_dag_without_copying():
    from etv.ir import substitute

    expression = shared_dag(30)
    assert substitute(expression, ()) is expression
    assert substitute(expression, (("unrelated", const(3, FLOAT)),)) is expression
    changed = substitute(expression, (("shared", const(2, FLOAT)),))
    assert changed is not expression and changed.args[0] is changed.args[1]
    assert "_free_variables" in expression.__dict__
    restored = pickle.loads(pickle.dumps(expression))
    assert "_free_variables" not in restored.__dict__
    assert restored == expression


def test_simplify_preserves_unmodified_shared_nodes():
    from etv.semantics import simplify

    expression = shared_dag(30)
    assert simplify(expression) is expression
