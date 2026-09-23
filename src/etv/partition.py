"""Paired PDG proposals. Every proposed boundary remains a proof goal."""

from dataclasses import dataclass
import hashlib
import json

from .errors import InputError
from .ir import Expr, RootPair, walk
from .jsonio import JSON, array, obj, string
from .pairspec import PairSpec


@dataclass(frozen=True)
class Node:
    id: str
    expr: Expr


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class PDG:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    roots: tuple[str, ...]


@dataclass(frozen=True)
class Part:
    id: str
    lhs: str
    rhs: str
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class PartitionPlan:
    parts: tuple[Part, ...]
    source: str


def build_pdg(expressions: tuple[Expr, ...], side: str) -> PDG:
    ordered = dict.fromkeys(node for expression in expressions for node in walk(expression))
    ids = {expr: f"{side}.{i}" for i, expr in enumerate(ordered)}
    nodes = tuple(Node(ids[expr], expr) for expr in ordered)
    edges = []
    for expr in ordered:
        for index, arg in enumerate(expr.args):
            kind = (
                "control"
                if (expr.op == "select" and index == 0) or (expr.op == "load" and index == 1)
                else "data"
            )
            edges.append(Edge(ids[arg], ids[expr], kind))
        if expr.op in ("load", "read"):
            for arg in expr.args[:1]:
                edges.append(Edge(ids[arg], ids[expr], "memory"))
    return PDG(nodes, tuple(edges), tuple(ids[expr] for expr in expressions))


def validate_plan(raw: JSON, lhs: PDG, rhs: PDG, spec: PairSpec) -> PartitionPlan:
    data = obj(raw, "parts")
    left, right = {n.id: n.expr for n in lhs.nodes}, {n.id: n.expr for n in rhs.nodes}
    parts = []
    previous: dict[str, Part] = {}
    for raw_part in array(data["parts"]):
        part = obj(raw_part, "id lhs rhs dependencies")
        item = Part(
            string(part["id"]),
            string(part["lhs"]),
            string(part["rhs"]),
            tuple(string(x) for x in array(part["dependencies"])),
        )
        if item.id in previous or item.lhs not in left or item.rhs not in right:
            raise InputError("INVALID_PARTITION", "unknown or duplicate node/part id")
        if (
            left[item.lhs].type != right[item.rhs].type
            or not set(item.dependencies) <= previous.keys()
        ):
            raise InputError("INVALID_PARTITION", "type or topological order mismatch")
        ldeps, rdeps = set(walk(left[item.lhs])), set(walk(right[item.rhs]))
        for dependency in item.dependencies:
            boundary = previous[dependency]
            if left[boundary.lhs] not in ldeps or right[boundary.rhs] not in rdeps:
                raise InputError("INVALID_PARTITION", "dependency is outside the selected subgraph")
        previous[item.id] = item
        parts.append(item)
    if not parts or len(parts) > spec.partition.max_parts:
        raise InputError("INVALID_PARTITION", "partition count outside bounds")
    return PartitionPlan(tuple(parts), "machine_checked")


def propose_plan(roots: tuple[RootPair, ...], spec: PairSpec) -> PartitionPlan:
    lhs = build_pdg(tuple(root.lhs for root in roots), "lhs")
    rhs = build_pdg(tuple(root.rhs for root in roots), "rhs")
    right = {node.expr: node.id for node in rhs.nodes if node.expr.args}
    parts: list[JSON] = []
    for node in lhs.nodes:
        if node.expr in right and node.expr.args:
            parts.append(
                {
                    "id": f"part.{len(parts)}",
                    "lhs": node.id,
                    "rhs": right[node.expr],
                    "dependencies": [],
                }
            )
        if len(parts) == spec.partition.max_parts:
            break
    if not parts:
        parts.append({"id": "part.0", "lhs": lhs.roots[0], "rhs": rhs.roots[0], "dependencies": []})
    return validate_plan({"parts": parts}, lhs, rhs, spec)


def fingerprint(plan: PartitionPlan) -> str:
    return hashlib.sha256(
        json.dumps(
            [(p.id, p.lhs, p.rhs, p.dependencies) for p in plan.parts], sort_keys=True
        ).encode()
    ).hexdigest()
