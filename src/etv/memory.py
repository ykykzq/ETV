"""Step composition and observable writer obligations."""

from dataclasses import dataclass, replace
from math import prod

import z3

from .errors import ETVError, ResourceLimit, UnsupportedSemantics
from .ir import (
    BOOL,
    INDEX,
    Expr,
    ProgramPair,
    ProgramSequence,
    RootPair,
    StorageRef,
    const,
    substitute,
    var,
    walk,
)
from .pairspec import PairSpec
from .result import Counterexample, Obligation
from .semantics import evaluate, simplify
from .smt import Encoder, QueryResult, query


@dataclass(frozen=True)
class ObligationSet:
    roots: tuple[RootPair, ...]
    obligations: tuple[Obligation, ...]
    status: str = "closed"
    reason: str = ""
    counterexample: Counterexample | None = None


class _Stop(Exception):
    def __init__(
        self, status: str, reason: str, counterexample: Counterexample | None = None
    ) -> None:
        self.status, self.reason, self.counterexample = status, reason, counterexample


class _Builder:
    def __init__(self, pair: ProgramPair, spec: PairSpec) -> None:
        self.pair, self.spec = pair, spec
        self.encoder = Encoder(spec.index_bits)
        self.premises = tuple(
            self.encoder.encode(p.expr)
            for p in spec.predicates
            if p.expr is not None and p.kind in ("parameter", "constraints", "custom")
        )
        self.obligations: list[Obligation] = []
        self.dependencies = tuple(p.id for p in spec.predicates)

    def check(
        self,
        name: str,
        bad: z3.BoolRef,
        premises: tuple[z3.BoolRef, ...] = (),
        reason: str = "SMT_UNKNOWN",
    ) -> QueryResult:
        answer = query(bad, self.premises + premises, self.spec.limits.smt_timeout_ms)
        self.obligations.append(
            Obligation(name, "smt", answer.status, self.dependencies, answer.detail, answer.model)
        )
        if answer.status != "unsat":
            raise _Stop("UNKNOWN", "SMT_UNKNOWN" if answer.status == "unknown" else reason)
        return answer

    def fact(self, name: str, detail: str = "") -> None:
        self.obligations.append(Obligation(name, "structural", "closed", self.dependencies, detail))

    def premises_valid(self) -> None:
        nonempty = query(z3.BoolVal(True), self.premises, self.spec.limits.smt_timeout_ms)
        self.obligations.append(
            Obligation(
                "predicates.nonempty",
                "smt",
                nonempty.status,
                self.dependencies,
                nonempty.detail,
                nonempty.model,
            )
        )
        if nonempty.status != "sat":
            raise _Stop(
                "UNKNOWN", "SMT_UNKNOWN" if nonempty.status == "unknown" else "PREDICATES_UNSAT"
            )
        for predicate in self.spec.predicates:
            if predicate.expr is not None:
                self.check(
                    "defined." + predicate.id,
                    z3.Not(self.encoder.defined(predicate.expr)),
                    reason="INTEGER_DEFINEDNESS_NOT_PROVED",
                )
        self.check(
            "observation.domain",
            z3.Or(
                z3.Not(self.encoder.defined(self.spec.observation.numel)),
                self.encoder.encode(self.spec.observation.numel) <= 0,
            ),
            reason="INVALID_OBSERVATION_DOMAIN",
        )
        for side in (self.pair.lhs, self.pair.rhs):
            for step in side.steps:
                for kernel in step:
                    self.check(
                        "grid." + kernel.launch_id,
                        z3.Or(
                            z3.Not(self.encoder.defined(kernel.programs)),
                            self.encoder.encode(kernel.programs) <= 0,
                            self.encoder.encode(kernel.programs) > 2**31,
                        ),
                        reason="INVALID_PROGRAM_DOMAIN",
                    )
                    for i, parameter in enumerate(kernel.parameters):
                        self.check(
                            f"binding.{kernel.launch_id}.{i}",
                            z3.Not(self.encoder.defined(parameter)),
                            reason="INTEGER_DEFINEDNESS_NOT_PROVED",
                        )

    def no_alias(self) -> None:
        declared = {
            frozenset((a, b))
            for group in self.spec.disjoint
            for a in group
            for b in group
            if a != b
        }
        required = set(self.spec.observation.require_disjoint)
        needed = {frozenset((a, b)) for a in required for b in required if a != b}
        for launches in (self.spec.lhs, self.spec.rhs):
            memory = {ep.role for launch in launches for ep in launch.abi if ep.kind != "scalar"}
            written = {store.role for launch in launches for store in launch.stores}
            needed.update(frozenset((a, b)) for a in written for b in memory if a != b)
        if needed - declared:
            raise _Stop("UNKNOWN", "NO_ALIAS_NOT_DECLARED")
        self.fact("memory.no_alias", "declared separation covers all read/write role pairs")

    def resolve_fixed(
        self,
        expr: Expr,
        state: dict[tuple[str, int], Expr],
        written: set[str],
        replacements: tuple[tuple[str, Expr], ...],
    ) -> Expr:
        expr = simplify(substitute(expr, replacements), self.spec.index_bits)
        if expr.op == "load":
            storage = expr.attr("storage")
            assert isinstance(storage, StorageRef)
            offset, mask, default = expr.args
            try:
                enabled = evaluate(mask, index_bits=self.spec.index_bits)
            except (KeyError, UnsupportedSemantics):
                raise _Stop("UNKNOWN", "DYNAMIC_MEMORY_ACCESS") from None
            if not enabled:
                return self.resolve_fixed(default, state, written, ())
            try:
                address = int(evaluate(offset, index_bits=self.spec.index_bits))
            except (KeyError, UnsupportedSemantics):
                raise _Stop("UNKNOWN", "DYNAMIC_MEMORY_ACCESS") from None
            if storage.scalar_index is not None:
                if address != storage.base_offset + storage.scalar_index:
                    raise _Stop("UNKNOWN", "ABI_SCALAR_INDEX_MISMATCH")
                return var("scalar." + storage.logical_role, expr.type)
            if address < 0:
                raise _Stop("UNKNOWN", "UNDEFINED_LOAD_REACHABLE")
            key = (storage.logical_role, address)
            if key in state:
                return state[key]
            if storage.logical_role in written:
                return Expr("undefined", expr.type)
            return Expr("read", expr.type, (const(address),), (("role", storage.logical_role),))
        return simplify(
            Expr(
                expr.op,
                expr.type,
                tuple(self.resolve_fixed(a, state, written, ()) for a in expr.args),
                expr.attrs,
            ),
            self.spec.index_bits,
        )

    def fixed_side(self, sequence: ProgramSequence, side: str, numel: int) -> dict[int, Expr]:
        state: dict[tuple[str, int], Expr] = {}
        written = {
            s.storage.logical_role for step in sequence.steps for k in step for s in k.stores
        }
        instances = 0
        for step_index, step in enumerate(sequence.steps):
            pending: dict[tuple[str, int], Expr] = {}
            for kernel in step:
                programs = int(evaluate(kernel.programs, index_bits=self.spec.index_bits))
                for store in kernel.stores:
                    count = programs * prod(store.lane_shape)
                    instances += count
                    if instances > self.spec.limits.max_instances:
                        raise ResourceLimit("complete program/lane enumeration")
                    for pid in range(programs):
                        for lane in range(prod(store.lane_shape)):
                            replace = (("__pid", const(pid)), ("__lane", const(lane)))
                            mask = simplify(substitute(store.mask, replace), self.spec.index_bits)
                            try:
                                active = bool(evaluate(mask, index_bits=self.spec.index_bits))
                            except (KeyError, UnsupportedSemantics):
                                raise _Stop("UNKNOWN", "DYNAMIC_MEMORY_ACCESS") from None
                            if not active:
                                continue
                            offset = simplify(
                                substitute(store.offset, replace), self.spec.index_bits
                            )
                            address = int(evaluate(offset, index_bits=self.spec.index_bits))
                            role = store.storage.logical_role
                            index = str(address)
                            if role == self.spec.observation.role and not 0 <= address < numel:
                                witness = Counterexample(
                                    index,
                                    (("program", str(pid)), ("lane", str(lane))),
                                    index if side == "lhs" else None,
                                    index if side == "rhs" else None,
                                    active if side == "lhs" else None,
                                    active if side == "rhs" else None,
                                    None,
                                    None,
                                    "Replay the declared launch at program/lane; the active address is outside the observation domain.",
                                )
                                raise _Stop("DISPROVED", "ADDRESS_MISMATCH", witness)
                            if (role, address) in pending:
                                witness = Counterexample(
                                    index,
                                    (("step", str(step_index)),),
                                    index,
                                    index,
                                    True,
                                    True,
                                    None,
                                    None,
                                    "Enumerate the declared step: two active writers target this role and offset.",
                                )
                                raise _Stop("DISPROVED", "WRITE_RACE", witness)
                            value = self.resolve_fixed(store.value, state, written, replace)
                            self.check(
                                f"defined.{side}.{step_index}.{role}.{address}",
                                z3.Not(self.encoder.defined(value)),
                                reason=self.undefined_reason(value),
                            )
                            pending[(role, address)] = value
            state.update(pending)
            self.fact(f"memory.{side}.step{step_index}", f"{len(pending)} simultaneous writes")
        self.fact(
            f"memory.{side}.enumeration", f"{instances} complete program/lane/store instances"
        )
        return {
            offset: value
            for (role, offset), value in state.items()
            if role == self.spec.observation.role
        }

    @staticmethod
    def undefined_reason(expr: Expr) -> str:
        if any(e.op == "undefined" for e in walk(expr)):
            return "UNDEFINED_LOAD_REACHABLE"
        if any(e.op in ("fdiv", "sqrt", "rsqrt", "log", "acosh", "pow") for e in walk(expr)):
            return "FLOAT_DEFINEDNESS_NOT_PROVED"
        return "INTEGER_DEFINEDNESS_NOT_PROVED"

    def fixed(self) -> tuple[RootPair, ...]:
        numel = int(evaluate(self.spec.observation.numel, index_bits=self.spec.index_bits))
        if numel > self.spec.limits.max_instances:
            raise ResourceLimit("observation element enumeration")
        lhs = self.fixed_side(self.pair.lhs, "lhs", numel)
        rhs = self.fixed_side(self.pair.rhs, "rhs", numel)
        expected = (
            set(range(numel))
            if self.spec.observation.require_full_coverage
            else lhs.keys() | rhs.keys()
        )
        for index in sorted(expected):
            if index not in lhs or index not in rhs:
                reason = (
                    "MASK_MISMATCH" if (index in lhs) != (index in rhs) else "COVERAGE_MISMATCH"
                )
                witness = Counterexample(
                    str(index),
                    (),
                    str(index) if index in lhs else None,
                    str(index) if index in rhs else None,
                    index in lhs,
                    index in rhs,
                    None,
                    None,
                    "Enumerate all declared program/lane instances and compare final active writers at this index.",
                )
                raise _Stop("DISPROVED", reason, witness)
        self.fact("memory.coverage", f"{len(expected)} observed elements")
        self.fact("memory.address_mask", "equal final active address domains; no duplicate writes")
        return tuple(RootPair(lhs[i], rhs[i], const(i)) for i in sorted(expected))

    def symbolic_value(
        self,
        expr: Expr,
        assumptions: tuple[z3.BoolRef, ...],
        key: Expr,
        label: str,
        written: set[str],
    ) -> Expr:
        expr = simplify(expr, self.spec.index_bits)
        if expr.op == "load":
            storage = expr.attr("storage")
            assert isinstance(storage, StorageRef)
            offset, mask, default = expr.args
            mask_z3 = self.encoder.encode(mask)
            self.check(
                label + ".load_defined",
                z3.Not(
                    z3.And(
                        self.encoder.defined(mask),
                        z3.Implies(mask_z3, self.encoder.defined(offset)),
                    )
                ),
                assumptions,
                "INTEGER_DEFINEDNESS_NOT_PROVED",
            )
            always = query(
                z3.Not(mask_z3), self.premises + assumptions, self.spec.limits.smt_timeout_ms
            )
            if always.status == "unknown":
                raise _Stop("UNKNOWN", "SMT_UNKNOWN")
            if storage.logical_role in written:
                loaded = Expr("undefined", expr.type)
            elif storage.scalar_index is not None:
                self.check(
                    label + ".scalar_index",
                    self.encoder.encode(offset) != storage.base_offset + storage.scalar_index,
                    assumptions,
                    "ABI_SCALAR_INDEX_MISMATCH",
                )
                loaded = var("scalar." + storage.logical_role, expr.type)
            else:
                self.check(
                    label + ".load_address",
                    self.encoder.encode(offset) < 0,
                    assumptions + (mask_z3,),
                    "UNDEFINED_LOAD_REACHABLE",
                )
                equal = query(
                    self.encoder.encode(offset) != self.encoder.encode(key),
                    self.premises + assumptions + (mask_z3,),
                    self.spec.limits.smt_timeout_ms,
                )
                if equal.status == "unknown":
                    raise _Stop("UNKNOWN", "SMT_UNKNOWN")
                if equal.status == "unsat":
                    self.fact(label + ".load_alignment", storage.logical_role)
                    offset = key
                loaded = Expr("read", expr.type, (offset,), (("role", storage.logical_role),))
            if always.status == "unsat":
                self.fact(label + ".load_mask")
                return loaded
            default = self.symbolic_value(
                default, assumptions + (z3.Not(mask_z3),), key, label + ".default", written
            )
            return Expr("select", expr.type, (mask, loaded, default))
        return Expr(
            expr.op,
            expr.type,
            tuple(
                self.symbolic_value(arg, assumptions, key, label + f".{i}", written)
                for i, arg in enumerate(expr.args)
            ),
            expr.attrs,
        )

    def symbolic(self) -> tuple[RootPair, ...]:
        if len(self.spec.lhs) != 1 or len(self.spec.rhs) != 1:
            raise _Stop("UNKNOWN", "PARAMETERIZED_LAUNCH_DAG_UNSUPPORTED")
        # This encoding requires proofs for every intermediate integer range.
        self.encoder = Encoder(self.spec.index_bits, mathematical_int=True)
        key = var("k")
        k = self.encoder.encode(key)
        n = self.encoder.encode(self.spec.observation.numel)
        observed_domain = (k >= 0, k < n)
        values: list[Expr] = []
        for side, sequence in (("lhs", self.pair.lhs), ("rhs", self.pair.rhs)):
            kernel = sequence.steps[0][0]
            if len(kernel.stores) != 1:
                raise _Stop("UNKNOWN", "PARAMETERIZED_MULTI_STORE_UNSUPPORTED")
            store = kernel.stores[0]
            if store.storage.logical_role != self.spec.observation.role:
                raise _Stop("UNKNOWN", "OBSERVATION_WRITER_MISSING")
            size = prod(store.lane_shape)
            candidate = (
                ("__pid", Expr("div", INDEX, (key, const(size)))),
                ("__lane", Expr("rem", INDEX, (key, const(size)))),
            )
            address = substitute(store.offset, candidate)
            mask = substitute(store.mask, candidate)
            programs = self.encoder.encode(kernel.programs)
            pid, lane = z3.Ints(f"{side}.pid {side}.lane")
            encoder = Encoder(
                self.spec.index_bits, {"__pid": pid, "__lane": lane}, mathematical_int=True
            )
            domain = (pid >= 0, pid < programs, lane >= 0, lane < size)
            active = encoder.encode(store.mask)
            actual = encoder.encode(store.offset)
            self.check(
                f"memory.{side}.defined_address",
                z3.Not(z3.And(encoder.defined(store.mask), encoder.defined(store.offset))),
                domain,
                "INTEGER_DEFINEDNESS_NOT_PROVED",
            )
            self.check(
                f"memory.{side}.coverage",
                z3.Or(
                    k / size >= programs,
                    self.encoder.encode(address) != k,
                    z3.Not(self.encoder.encode(mask)),
                ),
                observed_domain,
                "COVERAGE_NOT_PROVED",
            )
            self.check(
                f"memory.{side}.address",
                z3.Or(actual < 0, actual >= n),
                domain + (active,),
                "ADDRESS_NOT_PROVED",
            )
            # An injective canonical linear index gives a constructive writer and
            # discharges uniqueness without trusting an inferred inverse.
            self.check(
                f"memory.{side}.uniqueness",
                actual != pid * size + lane,
                domain + (active,),
                "UNIQUE_WRITER_NOT_PROVED",
            )
            expr = substitute(store.value, candidate)
            expr = self.symbolic_value(
                expr, observed_domain, key, f"memory.{side}", {store.storage.logical_role}
            )
            self.check(
                f"defined.{side}.value",
                z3.Not(self.encoder.defined(expr)),
                observed_domain,
                self.undefined_reason(expr),
            )
            values.append(expr)
        domain_expr = Expr(
            "and",
            BOOL,
            (
                Expr("ge", BOOL, (key, const(0))),
                Expr("lt", BOOL, (key, self.spec.observation.numel)),
            ),
        )
        premises = tuple(
            p.expr
            for p in self.spec.predicates
            if p.expr is not None and p.kind in ("parameter", "constraints", "custom")
        )
        return (RootPair(values[0], values[1], key, premises + (domain_expr,)),)


def build_obligations(pair: ProgramPair, spec: PairSpec) -> ObligationSet:
    builder = _Builder(pair, spec)
    try:
        if len(spec.lhs) > 1 and len(spec.rhs) > 1:
            raise _Stop("UNKNOWN", "BOTH_SIDES_MULTI_LAUNCH_UNSUPPORTED")
        builder.premises_valid()
        builder.no_alias()
        roots = builder.symbolic() if spec.parameters else builder.fixed()
        return ObligationSet(roots, tuple(builder.obligations))
    except _Stop as stop:
        if spec.parameters and stop.reason in {
            "COVERAGE_NOT_PROVED",
            "ADDRESS_NOT_PROVED",
            "UNIQUE_WRITER_NOT_PROVED",
        }:
            witness = _replay_memory_model(pair, spec, builder.obligations[-1])
            if witness is not None:
                replay_fact = Obligation(
                    "memory.counterexample_replay",
                    "enumeration",
                    "closed",
                    (builder.obligations[-1].id,),
                    "exact specialized replay",
                )
                return replace(witness, obligations=tuple(builder.obligations) + (replay_fact,))
        return ObligationSet(
            (), tuple(builder.obligations), stop.status, stop.reason, stop.counterexample
        )


def _replay_memory_model(
    pair: ProgramPair, spec: PairSpec, obligation: Obligation
) -> ObligationSet | None:
    if obligation.status != "sat":
        return None
    try:
        model = dict(obligation.model)
        values = {p.name: int(model.get(p.name, str(p.minimum))) for p in spec.parameters}
        replacements = tuple((name, const(value)) for name, value in sorted(values.items()))
        for predicate in spec.predicates:
            if predicate.expr is not None and predicate.kind in (
                "parameter",
                "constraints",
                "custom",
            ):
                if not evaluate(predicate.expr, values, spec.index_bits):
                    return None

        def concrete(expr: Expr) -> Expr:
            return substitute(expr, replacements)

        def sequence(program: ProgramSequence) -> ProgramSequence:
            return ProgramSequence(
                tuple(
                    tuple(
                        replace(
                            kernel,
                            programs=concrete(kernel.programs),
                            parameters=tuple(concrete(p) for p in kernel.parameters),
                            effects=tuple(concrete(e) for e in kernel.effects),
                            stores=tuple(
                                replace(
                                    store,
                                    logical_index=concrete(store.logical_index),
                                    offset=concrete(store.offset),
                                    mask=concrete(store.mask),
                                    value=concrete(store.value),
                                )
                                for store in kernel.stores
                            ),
                        )
                        for kernel in step
                    )
                    for step in program.steps
                )
            )

        specialized = replace(
            spec,
            parameters=(),
            observation=replace(spec.observation, numel=concrete(spec.observation.numel)),
            predicates=tuple(
                replace(p, expr=concrete(p.expr) if p.expr is not None else None)
                for p in spec.predicates
            ),
        )
        replay = build_obligations(ProgramPair(sequence(pair.lhs), sequence(pair.rhs)), specialized)
        if replay.status != "DISPROVED" or replay.counterexample is None:
            return None
        counterexample = replace(
            replay.counterexample,
            parameters=tuple((name, str(value)) for name, value in sorted(values.items()))
            + replay.counterexample.parameters,
            replay="Specialize the declared parameters to this model, then exhaustively enumerate all program/lane instances. "
            + replay.counterexample.replay,
        )
        return replace(replay, counterexample=counterexample)
    except (ETVError, ValueError, KeyError, ZeroDivisionError):
        return None
