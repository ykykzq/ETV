"""LLM-proposed and machine-checked decomposition of paired compute graphs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from .llm import DeepSeekClient, LLMError, _nodes, _resolve
from .ir import Expr, Program, Sort
from .model import PairSpec

_PARTITION_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class PartitionError(LLMError):
    """Raised when an LLM partition proposal fails a structural check."""

    def __init__(self, message: str, audit: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.audit = dict(audit or {})


@dataclass(frozen=True)
class RootFamily:
    family_id: str
    members: Tuple[Tuple[Any, Expr, Expr], ...]


@dataclass(frozen=True)
class SubgraphBatch:
    """One paired subgraph obligation, instantiated for a root family."""

    partition_id: str
    family_id: str
    semantic: str
    lhs_path: str
    rhs_path: str
    dependencies: Tuple[str, ...]
    root_pairs: Tuple[Tuple[str, Expr, Expr], ...]


@dataclass(frozen=True)
class PartitionPlan:
    """A dependency-ordered collection of independently checkable obligations."""

    batches: Tuple[SubgraphBatch, ...]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class _Proposal:
    partition_id: str
    family_id: str
    semantic: str
    lhs_path: str
    rhs_path: str
    dependencies: Tuple[str, ...] = ()


def _topology(expression: Expr) -> tuple:
    return (
        expression.op,
        expression.sort.value,
        tuple(_topology(argument) for argument in expression.args),
    )


def _families(
    roots: Sequence[Tuple[Any, Expr, Expr]],
) -> Tuple[RootFamily, ...]:
    grouped: dict[tuple, list[Tuple[Any, Expr, Expr]]] = {}
    for item in roots:
        key = (_topology(item[1]), _topology(item[2]))
        grouped.setdefault(key, []).append(item)
    return tuple(
        RootFamily(f"family_{index}", tuple(members))
        for index, members in enumerate(grouped.values())
    )


def _program_json(program: Program) -> dict:
    return {
        "name": program.name,
        "frontend": program.frontend,
        "frontend_version": program.frontend_version,
        "launch": {
            "programs": program.programs.to_json(),
            "lanes": program.lanes,
        },
        "stores": [
            {
                "block": store.block,
                "logical_index": store.logical_index.to_json(),
                "offset": store.offset.to_json(),
                "mask": store.mask.to_json(),
                "value": store.value.to_json(),
            }
            for store in program.stores
        ],
    }


def _pair_context(spec: PairSpec) -> dict:
    def value_json(value: int | Expr) -> Any:
        return value if isinstance(value, int) else value.to_json()

    def frontend_json(side: str) -> dict:
        frontend = spec.lhs_frontend if side == "lhs" else spec.rhs_frontend
        return {
            "kind": frontend.kind,
            "function": frontend.function,
            "programs": (
                None if frontend.programs is None else frontend.programs.to_json()
            ),
        }

    return {
        "pair_id": spec.pair_id,
        "semantic_mode": spec.semantic_mode,
        "frontends": {
            "lhs": frontend_json("lhs"),
            "rhs": frontend_json("rhs"),
        },
        "roles": [
            {
                "logical": role.logical,
                "lhs": {
                    "kind": role.lhs.kind,
                    "name": role.lhs.name,
                    "index": role.lhs.index,
                },
                "rhs": {
                    "kind": role.rhs.kind,
                    "name": role.rhs.name,
                    "index": role.rhs.index,
                },
            }
            for role in spec.roles
        ],
        "assumptions_for_llm": list(spec.assumptions),
        "predicates": {
            "bindings": {
                name: value_json(value)
                for name, value in sorted(spec.facts.bindings.items())
            },
            "side_bindings": {
                side: {
                    name: value_json(value)
                    for name, value in sorted(
                        spec.facts.side_bindings.get(side, {}).items()
                    )
                }
                for side in ("lhs", "rhs")
            },
            "parameters": {
                name: {"min": parameter.minimum, "max": parameter.maximum}
                for name, parameter in sorted(spec.facts.parameters.items())
            },
            "ids": list(spec.predicates.predicate_ids),
            "custom": [item.to_json() for item in spec.predicates.custom],
            "constraints": [item.to_json() for item in spec.facts.constraints],
            "disjoint": [sorted(group) for group in spec.facts.disjoint_groups],
        },
        "contract": {
            "output_role": spec.contract.output_role,
            "output_numel": spec.contract.output_numel.to_json(),
        },
    }


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return descendant.startswith(ancestor + ".args[")


def _path_depth(path: str) -> int:
    return path.count(".args[")


def _nearest_parent(
    proposal: _Proposal, proposals: Sequence[_Proposal], side: str
) -> Optional[str]:
    path = getattr(proposal, f"{side}_path")
    ancestors = [
        item
        for item in proposals
        if item.partition_id != proposal.partition_id
        and _is_ancestor(getattr(item, f"{side}_path"), path)
    ]
    if not ancestors:
        return None
    return max(
        ancestors, key=lambda item: _path_depth(getattr(item, f"{side}_path"))
    ).partition_id


def _replace_dependencies(
    expression: Expr,
    path: str,
    replacements: Mapping[str, Expr],
) -> Expr:
    if path in replacements:
        return replacements[path]
    if not expression.args:
        return expression
    arguments = tuple(
        _replace_dependencies(argument, f"{path}.args[{index}]", replacements)
        for index, argument in enumerate(expression.args)
    )
    if arguments == expression.args:
        return expression
    return Expr(
        expression.op, args=arguments, data=expression.data, sort=expression.sort
    )


def _subgraph_expression(
    root: Expr,
    root_path: str,
    child_paths: Mapping[str, Expr],
) -> Expr:
    selected = _resolve(root, root_path)
    relative = {
        "root" + path[len(root_path) :]: replacement
        for path, replacement in child_paths.items()
    }
    return _replace_dependencies(selected, "root", relative)


def _count_nodes(expression: Expr) -> int:
    return 1 + sum(_count_nodes(argument) for argument in expression.args)


def _validate_proposals(
    response: Mapping[str, Any],
    families: Sequence[RootFamily],
    spec: PairSpec,
) -> Tuple[_Proposal, ...]:
    unknown_top_level = sorted(set(response) - {"partitions"})
    if unknown_top_level:
        raise PartitionError(
            "partition response has unknown fields: " + ", ".join(unknown_top_level)
        )
    raw = response.get("partitions")
    if not isinstance(raw, list):
        raise PartitionError("partition response must contain a partitions list")
    if not spec.partition.min_partitions <= len(raw) <= spec.partition.max_partitions:
        raise PartitionError(
            "partition count is outside PairSpec partition limits: "
            f"got {len(raw)}, expected {spec.partition.min_partitions}.."
            f"{spec.partition.max_partitions}"
        )
    family_by_id = {family.family_id: family for family in families}
    proposals: list[_Proposal] = []
    seen_ids: set[str] = set()
    seen_paths: dict[tuple[str, str], set[str]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PartitionError(f"partitions[{index}] must be an object")
        unknown = sorted(
            set(item) - {"id", "family", "semantic", "lhs_path", "rhs_path"}
        )
        if unknown:
            raise PartitionError(
                f"partitions[{index}] has unknown fields: {', '.join(unknown)}"
            )
        partition_id = item.get("id")
        family_id = item.get("family")
        semantic = item.get("semantic")
        lhs_path = item.get("lhs_path")
        rhs_path = item.get("rhs_path")
        if not isinstance(partition_id, str) or not _PARTITION_ID.fullmatch(
            partition_id
        ):
            raise PartitionError(f"partitions[{index}].id is not a stable identifier")
        if partition_id in seen_ids:
            raise PartitionError(f"duplicate partition id {partition_id!r}")
        if not isinstance(family_id, str) or family_id not in family_by_id:
            raise PartitionError(f"partition {partition_id!r} names an unknown family")
        if not isinstance(semantic, str) or not semantic.strip():
            raise PartitionError(f"partition {partition_id!r} needs a semantic label")
        if not isinstance(lhs_path, str) or not isinstance(rhs_path, str):
            raise PartitionError(f"partition {partition_id!r} paths must be strings")
        family = family_by_id[family_id]
        lhs = _resolve(family.members[0][1], lhs_path)
        rhs = _resolve(family.members[0][2], rhs_path)
        if lhs.sort != Sort.FLOAT or rhs.sort != Sort.FLOAT:
            raise PartitionError(
                f"partition {partition_id!r} must pair abstract-float compute roots"
            )
        if lhs.sort != rhs.sort:
            raise PartitionError(f"partition {partition_id!r} pairs different sorts")
        if lhs_path != "root" and not lhs.args:
            raise PartitionError(f"partition {partition_id!r} selects an lhs leaf")
        if rhs_path != "root" and not rhs.args:
            raise PartitionError(f"partition {partition_id!r} selects an rhs leaf")
        for side, path in (("lhs", lhs_path), ("rhs", rhs_path)):
            key = (family_id, side)
            if path in seen_paths.setdefault(key, set()):
                raise PartitionError(
                    f"multiple partitions select {family_id} {side} path {path!r}"
                )
            seen_paths[key].add(path)
        seen_ids.add(partition_id)
        proposals.append(
            _Proposal(
                partition_id=partition_id,
                family_id=family_id,
                semantic=semantic.strip(),
                lhs_path=lhs_path,
                rhs_path=rhs_path,
            )
        )

    for family in families:
        family_proposals = [
            item for item in proposals if item.family_id == family.family_id
        ]
        roots = [
            item
            for item in family_proposals
            if item.lhs_path == "root" and item.rhs_path == "root"
        ]
        if len(roots) != 1:
            raise PartitionError(
                f"{family.family_id} must have exactly one paired root partition"
            )
        for item in family_proposals:
            lhs_parent = _nearest_parent(item, family_proposals, "lhs")
            rhs_parent = _nearest_parent(item, family_proposals, "rhs")
            if lhs_parent != rhs_parent:
                raise PartitionError(
                    f"partition {item.partition_id!r} has inconsistent lhs/rhs dependency topology"
                )

    children: dict[str, list[str]] = {item.partition_id: [] for item in proposals}
    for family in families:
        family_proposals = [
            item for item in proposals if item.family_id == family.family_id
        ]
        for item in family_proposals:
            parent = _nearest_parent(item, family_proposals, "lhs")
            if parent is not None:
                children[parent].append(item.partition_id)
    return tuple(
        _Proposal(
            partition_id=item.partition_id,
            family_id=item.family_id,
            semantic=item.semantic,
            lhs_path=item.lhs_path,
            rhs_path=item.rhs_path,
            dependencies=tuple(sorted(children[item.partition_id])),
        )
        for item in proposals
    )


def _ordered(proposals: Sequence[_Proposal]) -> Tuple[_Proposal, ...]:
    by_id = {item.partition_id: item for item in proposals}
    result: list[_Proposal] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(partition_id: str) -> None:
        if partition_id in visited:
            return
        if partition_id in visiting:
            raise PartitionError("partition dependency graph contains a cycle")
        visiting.add(partition_id)
        for dependency in by_id[partition_id].dependencies:
            visit(dependency)
        visiting.remove(partition_id)
        visited.add(partition_id)
        result.append(by_id[partition_id])

    for item in proposals:
        visit(item.partition_id)
    return tuple(result)


def _build_batches(
    proposals: Sequence[_Proposal], families: Sequence[RootFamily]
) -> Tuple[SubgraphBatch, ...]:
    family_by_id = {family.family_id: family for family in families}
    proposal_by_id = {item.partition_id: item for item in proposals}
    batches: list[SubgraphBatch] = []
    for item in _ordered(proposals):
        family = family_by_id[item.family_id]
        root_pairs = []
        for member_index, (label, lhs_root, rhs_root) in enumerate(family.members):
            lhs_replacements = {}
            rhs_replacements = {}
            for dependency_id in item.dependencies:
                dependency = proposal_by_id[dependency_id]
                token_name = (
                    f"partition:{item.family_id}:{dependency_id}:{member_index}"
                )
                lhs_replacements[dependency.lhs_path] = Expr(
                    "input", data=token_name, sort=Sort.FLOAT
                )
                rhs_replacements[dependency.rhs_path] = Expr(
                    "input", data=token_name, sort=Sort.FLOAT
                )
            lhs = _subgraph_expression(lhs_root, item.lhs_path, lhs_replacements)
            rhs = _subgraph_expression(rhs_root, item.rhs_path, rhs_replacements)
            root_pairs.append(
                (f"{item.family_id}:{item.partition_id}:{label}", lhs, rhs)
            )
        batches.append(
            SubgraphBatch(
                partition_id=item.partition_id,
                family_id=item.family_id,
                semantic=item.semantic,
                lhs_path=item.lhs_path,
                rhs_path=item.rhs_path,
                dependencies=item.dependencies,
                root_pairs=tuple(root_pairs),
            )
        )
    return tuple(batches)


def propose_partition_plan(
    spec: PairSpec,
    lhs_program: Program,
    rhs_program: Program,
    roots: Sequence[Tuple[Any, Expr, Expr]],
    client: Optional[DeepSeekClient] = None,
) -> PartitionPlan:
    """Make one whole-pair LLM call, then validate and materialize its proposal."""

    if not spec.partition.enabled:
        raise PartitionError("partitioning is disabled")
    if not roots:
        raise PartitionError("there are no compute roots to partition")
    families = _families(roots)
    payload = {
        "task": (
            "Partition both complete programs into corresponding semantic compute "
            "subgraphs. Subgraphs on one side should be logically independent except "
            "for explicit parent-child data dependencies."
        ),
        "pair": _pair_context(spec),
        "programs": {
            "lhs": _program_json(lhs_program),
            "rhs": _program_json(rhs_program),
        },
        "root_families": [
            {
                "id": family.family_id,
                "member_count": len(family.members),
                "member_labels": [str(item[0]) for item in family.members],
                "lhs_nodes": _nodes(family.members[0][1]),
                "rhs_nodes": _nodes(family.members[0][2]),
            }
            for family in families
        ],
        "output_schema": {
            "partitions": [
                {
                    "id": "stable_unique_identifier",
                    "family": "family_0",
                    "semantic": "short semantic operation name",
                    "lhs_path": "root.args[0]",
                    "rhs_path": "root.args[1]",
                }
            ]
        },
        "requirements": [
            "Return JSON only and use only supplied family ids and node paths.",
            "Include exactly one root/root partition for every family.",
            "Pair semantic operations even when their positions differ between sides.",
            "Select only abstract-float operation nodes, never leaves.",
            "Use nested or disjoint paths so dependencies are explicit.",
            f"Return between {spec.partition.min_partitions} and "
            f"{spec.partition.max_partitions} partitions in total.",
            "Do not assert equivalence; ETV proves every proposed pair separately.",
        ],
    }
    client = client or DeepSeekClient(spec)
    response, call_audit = client.complete_json(
        "program_partitioning",
        (
            "Return JSON only. You propose paired semantic boundaries; ETV checks "
            "coverage, ownership, dependency topology, and equivalence."
        ),
        payload,
    )
    if not isinstance(response, dict):
        raise PartitionError(
            "partition response must be a JSON object",
            {"whole_program_calls": 1, "call": call_audit},
        )
    try:
        proposals = _validate_proposals(response, families, spec)
        batches = _build_batches(proposals, families)
    except LLMError as exc:
        raise PartitionError(
            str(exc),
            {
                "whole_program_calls": 1,
                "call": call_audit,
            },
        ) from exc
    definitions = [
        {
            "id": batch.partition_id,
            "family": batch.family_id,
            "semantic": batch.semantic,
            "lhs_path": batch.lhs_path,
            "rhs_path": batch.rhs_path,
            "dependencies": list(batch.dependencies),
            "instances": len(batch.root_pairs),
            "exclusive_lhs_nodes": sum(
                _count_nodes(item[1]) for item in batch.root_pairs
            ),
            "exclusive_rhs_nodes": sum(
                _count_nodes(item[2]) for item in batch.root_pairs
            ),
        }
        for batch in batches
    ]
    return PartitionPlan(
        batches=batches,
        audit={
            "enabled": True,
            "status": "validated",
            "provider": spec.llm.provider,
            "configured_model": spec.llm.model,
            "whole_program_calls": 1,
            "call": call_audit,
            "root_families": [
                {"id": family.family_id, "members": len(family.members)}
                for family in families
            ],
            "machine_checks": {
                "paired_root_coverage": True,
                "unique_node_ownership": True,
                "matching_dependency_topology": True,
                "acyclic_dependency_order": True,
                "float_compute_roots": True,
            },
            "partitions": definitions,
        },
    )
