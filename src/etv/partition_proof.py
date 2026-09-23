"""Dependency-ordered checks of proposed PDG boundary equalities."""

from dataclasses import dataclass

from .eqsat import saturate
from .ir import RootPair
from .pairspec import PairSpec
from .partition import PartitionPlan, build_pdg
from .result import PartitionFact, RuleFact
from .rewrites import Rule, RuleRegistry


@dataclass(frozen=True)
class PartitionCheck:
    registry: RuleRegistry
    facts: tuple[PartitionFact, ...]
    applications: tuple[tuple[str, int], ...]
    fallback: bool = False


def check_plan(
    plan: PartitionPlan, roots: tuple[RootPair, ...], registry: RuleRegistry, spec: PairSpec
) -> PartitionCheck:
    left = {n.id: n.expr for n in build_pdg(tuple(r.lhs for r in roots), "lhs").nodes}
    right = {n.id: n.expr for n in build_pdg(tuple(r.rhs for r in roots), "rhs").nodes}
    facts: list[PartitionFact] = []
    proven: dict[str, Rule] = {}
    counts: dict[str, int] = {}
    for part in plan.parts:
        local = RuleRegistry(
            registry.rules + tuple(proven[d] for d in part.dependencies), registry.facts
        )
        result = saturate(left[part.lhs], right[part.rhs], local, spec.limits)
        for name, count in result.applications:
            counts[name] = counts.get(name, 0) + count
        merged = bool(result.merged and result.merged[0] and not result.reason)
        facts.append(
            PartitionFact(
                part.id,
                (part.lhs,),
                (part.rhs,),
                part.dependencies,
                "proved" if merged else "fallback",
                result.reason,
            )
        )
        if not merged:
            return PartitionCheck(registry, tuple(facts), (), True)
        proven[part.id] = Rule(
            "partition." + part.id,
            "relation",
            left[part.lhs],
            right[part.rhs],
            (),
            "checked_partition",
            "",
        )
    additions = tuple(proven.values())
    rule_facts = tuple(
        RuleFact(
            rule.id,
            rule.source,
            "",
            "egglog",
            "formal",
            detail="depends on checked boundary equality",
        )
        for rule in additions
    )
    return PartitionCheck(
        RuleRegistry(registry.rules + additions, registry.facts + rule_facts),
        tuple(facts),
        tuple(sorted(counts.items())),
    )
