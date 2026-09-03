"""Sequential launch semantics for an explicitly pre-partitioned program side."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple

from .evaluator import evaluate_program
from .ir import Expr, Program
from .model import Evaluation, InputError, LaunchSpec, PairSpec, RolePair


@dataclass(frozen=True)
class ProducedValue:
    """One active store value and the memory visible before its launch."""

    value_id: str
    launch_id: str
    logical_index: int
    output_role: str
    offset: int
    value: Expr
    prior_memory: Mapping[tuple[str, int], "ProducedValue"]


@dataclass(frozen=True)
class LaunchDependency:
    expression_path: str
    child: "LaunchNode"


@dataclass(frozen=True)
class LaunchNode:
    """A launch output instance with explicit prior-launch dependencies."""

    key: str
    launch_id: str
    semantic: str
    value: Expr
    dependencies: Tuple[LaunchDependency, ...]


@dataclass(frozen=True)
class SequenceEvaluation:
    side: str
    launches: Tuple[LaunchSpec, ...]
    programs: Tuple[Program, ...]
    evaluations: Tuple[Evaluation, ...]
    final: Mapping[int, ProducedValue]
    produced_launch_ids: Tuple[str, ...]


def _launch_spec(spec: PairSpec, side: str, launch: LaunchSpec) -> PairSpec:
    # The same logical storage may use a different physical argument in every
    # kernel.  Isolate each launch from the pair-wide endpoint names.
    roles = tuple(
        RolePair(logical=logical, lhs=endpoint, rhs=endpoint)
        for logical, endpoint in sorted(launch.roles.items())
    )
    side_bindings = {
        name: dict(values) for name, values in spec.predicates.side_bindings.items()
    }
    selected = dict(side_bindings.get(side, {}))
    selected.update(launch.bindings)
    side_bindings[side] = selected
    predicates = replace(
        spec.predicates,
        abi=roles,
        side_bindings=side_bindings,
    )
    return replace(spec, predicates=predicates, launch_sequences={})


def _check_launch_writes(evaluation: Evaluation, launch_id: str) -> None:
    logical: dict[int, Any] = {}
    addresses: dict[tuple[str, int], Any] = {}
    for record in evaluation.active:
        if record.logical_index in logical:
            raise InputError(
                f"launch {launch_id!r} assigns logical index "
                f"{record.logical_index} more than once",
                "DUPLICATE_LOGICAL_WRITE",
            )
        logical[record.logical_index] = record
        address = (record.output_role, int(record.offset))
        if address in addresses:
            raise InputError(
                f"launch {launch_id!r} has multiple active lanes writing "
                f"{address[0]}[{address[1]}]",
                "RACE_FREEDOM_NOT_PROVED",
            )
        addresses[address] = record


def evaluate_launch_sequence(
    spec: PairSpec,
    side: str,
    launches: Sequence[LaunchSpec],
    programs: Sequence[Program],
) -> SequenceEvaluation:
    """Enumerate ordered launches and build their exact finite memory flow."""

    if side not in {"lhs", "rhs"}:
        raise ValueError(f"invalid pair side {side!r}")
    if len(launches) != len(programs) or len(launches) < 2:
        raise InputError(
            "a launch sequence needs at least two matching specs and programs",
            "INVALID_LAUNCH_SEQUENCE",
        )
    memory: dict[tuple[str, int], ProducedValue] = {}
    evaluations: list[Evaluation] = []
    produced_ids: list[str] = []
    cursor = 0
    while cursor < len(launches):
        step = launches[cursor].step
        end = cursor + 1
        while end < len(launches) and launches[end].step == step:
            end += 1
        prior = dict(memory)
        pending: list[ProducedValue] = []
        step_addresses: set[tuple[str, int]] = set()
        for launch, program in zip(
            launches[cursor:end], programs[cursor:end], strict=True
        ):
            evaluation = evaluate_program(
                program, _launch_spec(spec, side, launch), side
            )
            _check_launch_writes(evaluation, launch.launch_id)
            for record in evaluation.active:
                if record.offset is None or record.value is None:
                    raise InputError(
                        f"launch {launch.launch_id!r} produced an incomplete active record",
                        "INVALID_LAUNCH_SEQUENCE",
                    )
                address = (record.output_role, int(record.offset))
                if address in step_addresses:
                    raise InputError(
                        f"launch step {step} has multiple store components writing "
                        f"{address[0]}[{address[1]}]",
                        "RACE_FREEDOM_NOT_PROVED",
                    )
                step_addresses.add(address)
                produced = ProducedValue(
                    value_id=(
                        f"{launch.launch_id}:p{record.pid}:l{record.lane}:"
                        f"{record.output_role}:{record.offset}"
                    ),
                    launch_id=launch.launch_id,
                    logical_index=record.logical_index,
                    output_role=record.output_role,
                    offset=int(record.offset),
                    value=record.value,
                    prior_memory=prior,
                )
                pending.append(produced)
            evaluations.append(evaluation)
        # Store components in one execution step share the same prior memory
        # and become visible together to the next step.
        for produced in pending:
            memory[(produced.output_role, produced.offset)] = produced
            produced_ids.append(produced.value_id)
        cursor = end

    final: dict[int, ProducedValue] = {}
    final_addresses: dict[int, int] = {}
    for (role, offset), produced in memory.items():
        if role != spec.contract.output_role:
            continue
        if produced.logical_index in final:
            raise InputError(
                "final launch sequence maps one logical output index to multiple "
                "physical addresses",
                "DUPLICATE_LOGICAL_WRITE",
            )
        if offset in final_addresses:
            raise InputError(
                "final launch sequence maps multiple logical output indices to one "
                "physical address",
                "RACE_FREEDOM_NOT_PROVED",
            )
        final[produced.logical_index] = produced
        final_addresses[offset] = produced.logical_index
    if not final:
        raise InputError(
            f"launch sequence never writes observed role {spec.contract.output_role!r}",
            "MISSING_OUTPUT_WRITE",
        )
    return SequenceEvaluation(
        side=side,
        launches=tuple(launches),
        programs=tuple(programs),
        evaluations=tuple(evaluations),
        final=final,
        produced_launch_ids=tuple(produced_ids),
    )


def _read_address(expression: Expr) -> tuple[str, int] | None:
    if expression.op != "read" or len(expression.args) != 1:
        return None
    offset = expression.args[0]
    if offset.op != "const_int":
        return None
    return str(expression.data), int(offset.data)


def build_launch_tree(
    produced: ProducedValue,
    semantics: Mapping[str, str],
    *,
    key: str = "root",
) -> LaunchNode:
    """Recover the prior-launch dependency tree of one final stored value."""

    dependencies: list[LaunchDependency] = []

    def visit(expression: Expr, path: str) -> None:
        address = _read_address(expression)
        dependency = produced.prior_memory.get(address) if address is not None else None
        if dependency is not None:
            child_key = f"{key}.dep[{len(dependencies)}]"
            child = build_launch_tree(dependency, semantics, key=child_key)
            dependencies.append(LaunchDependency(path, child))
            return
        for index, argument in enumerate(expression.args):
            visit(argument, f"{path}.args[{index}]")

    visit(produced.value, "root")
    return LaunchNode(
        key=key,
        launch_id=produced.launch_id,
        semantic=semantics.get(produced.launch_id, produced.launch_id),
        value=produced.value,
        dependencies=tuple(dependencies),
    )


def flatten_launch_tree(root: LaunchNode) -> Tuple[LaunchNode, ...]:
    result: list[LaunchNode] = []

    def visit(node: LaunchNode) -> None:
        result.append(node)
        for dependency in node.dependencies:
            visit(dependency.child)

    visit(root)
    return tuple(result)


def launch_tree_topology(root: LaunchNode) -> tuple:
    def expression_topology(expression: Expr, path: str, cuts: set[str]) -> tuple:
        if path in cuts:
            return ("launch_dependency", expression.sort.value)
        return (
            expression.op,
            expression.sort.value,
            tuple(
                expression_topology(argument, f"{path}.args[{index}]", cuts)
                for index, argument in enumerate(expression.args)
            ),
        )

    cuts = {dependency.expression_path for dependency in root.dependencies}
    return (
        root.launch_id,
        expression_topology(root.value, "root", cuts),
        tuple(launch_tree_topology(item.child) for item in root.dependencies),
    )


def compose_launch_value(produced: ProducedValue) -> Expr:
    """Inline prior launch stores according to sequential memory semantics."""

    def visit(expression: Expr) -> Expr:
        address = _read_address(expression)
        dependency = produced.prior_memory.get(address) if address is not None else None
        if dependency is not None:
            return compose_launch_value(dependency)
        arguments = tuple(visit(argument) for argument in expression.args)
        if arguments == expression.args:
            return expression
        return Expr(
            expression.op,
            args=arguments,
            data=expression.data,
            sort=expression.sort,
        )

    return visit(produced.value)


def replace_launch_dependencies(
    node: LaunchNode, replacements: Mapping[str, Expr]
) -> Expr:
    """Cut direct child launch reads out of a typed launch-local expression."""

    def visit(expression: Expr, path: str) -> Expr:
        if path in replacements:
            return replacements[path]
        arguments = tuple(
            visit(argument, f"{path}.args[{index}]")
            for index, argument in enumerate(expression.args)
        )
        if arguments == expression.args:
            return expression
        return Expr(
            expression.op,
            args=arguments,
            data=expression.data,
            sort=expression.sort,
        )

    return visit(node.value, "root")
