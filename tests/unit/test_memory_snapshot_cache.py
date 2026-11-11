"""Fixed-memory DAG caching must remain local to one state/binding snapshot."""

from etv.ir import BOOL, FLOAT, Expr, ProgramPair, ProgramSequence, StorageRef, const, var
from etv.memory import _Builder
from etv.pairspec import parse_pair_spec
from test_contracts import minimal, write


def builder(tmp_path):
    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    return _Builder(ProgramPair(ProgramSequence(()), ProgramSequence(())), spec)


def load(offset, mask=None):
    return Expr(
        "load",
        FLOAT,
        (offset, mask or const(True, BOOL), const("-inf", FLOAT)),
        (("storage", StorageRef("lhs", "lhs.0", "arg0", "Internal")),),
    )


def test_shared_load_respects_new_state_and_lane_bindings(tmp_path):
    memory = builder(tmp_path)
    loaded = load(var("__lane"))
    value = Expr("fadd", FLOAT, (loaded, loaded))
    state = {("Internal", 0): const(2, FLOAT), ("Internal", 1): const(3, FLOAT)}
    assert memory.resolve_fixed(value, state, {"Internal"}, (("__lane", const(0)),)) == const(
        4, FLOAT
    )
    assert memory.resolve_fixed(value, state, {"Internal"}, (("__lane", const(1)),)) == const(
        6, FLOAT
    )
    state[("Internal", 0)] = const(5, FLOAT)
    assert memory.resolve_fixed(value, state, {"Internal"}, (("__lane", const(0)),)) == const(
        10, FLOAT
    )


def test_mask_default_and_uninitialized_status_do_not_leak_between_calls(tmp_path):
    memory = builder(tmp_path)
    expression = load(const(0), var("enabled", BOOL))
    assert memory.resolve_fixed(
        expression, {}, {"Internal"}, (("enabled", const(False, BOOL)),)
    ) == const("-inf", FLOAT)
    assert (
        memory.resolve_fixed(expression, {}, {"Internal"}, (("enabled", const(True, BOOL)),)).op
        == "undefined"
    )
    state = {("Internal", 0): const(7, FLOAT)}
    assert memory.resolve_fixed(
        expression, state, {"Internal"}, (("enabled", const(True, BOOL)),)
    ) == const(7, FLOAT)


def test_large_symbolic_solver_unknown_keeps_report_instead_of_enumeration(tmp_path, monkeypatch):
    from etv.memory import _Stop, build_obligations
    from test_observation_encoding import encoded_data, program

    path = write(tmp_path, encoded_data(tmp_path, numel=300))
    spec = parse_pair_spec(path)
    pair = program(spec, const(1, FLOAT), const(True, BOOL), numel=300)

    def exhausted(self):
        self.fact("symbolic.started")
        raise _Stop("UNKNOWN", "SMT_UNKNOWN")

    def should_not_enumerate(self):
        raise AssertionError("solver timeout must preserve formal UNKNOWN")

    monkeypatch.setattr(_Builder, "symbolic", exhausted)
    monkeypatch.setattr(_Builder, "fixed", should_not_enumerate)
    result = build_obligations(pair, spec)
    assert (result.status, result.reason) == ("UNKNOWN", "SMT_UNKNOWN")
    assert result.obligations[-1].id == "symbolic.started"
    assert not result.roots


def test_large_unsupported_symbolic_layout_can_still_enumerate(tmp_path, monkeypatch):
    from etv.memory import _Stop, build_obligations
    from test_observation_encoding import encoded_data, program

    path = write(tmp_path, encoded_data(tmp_path, numel=300))
    spec = parse_pair_spec(path)
    pair = program(spec, const(1, FLOAT), const(True, BOOL), numel=300)

    def unsupported(self):
        raise _Stop("UNKNOWN", "SYMBOLIC_OBSERVATION_LAYOUT_UNSUPPORTED")

    monkeypatch.setattr(_Builder, "symbolic", unsupported)
    result = build_obligations(pair, spec)
    assert result.status == "closed"
    assert len(result.roots) == 300
