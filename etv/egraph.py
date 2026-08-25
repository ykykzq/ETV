"""A small e-graph with congruence closure, rebuilding, and an audit ledger."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Set, Tuple

from .model import Expr, Limits, Sort
from .rules import Pattern, Rule


@dataclass(frozen=True)
class ENode:
    op: str
    children: Tuple[int, ...]
    data: Any
    sort: Sort

    def render(self) -> str:
        suffix = "" if self.data is None else f":{self.data}"
        return f"{self.op}{suffix}({','.join(str(child) for child in self.children)})"


class EGraph:
    """Minimal equality-saturation engine used only on local compute slices."""

    def __init__(self) -> None:
        self.parent: List[int] = []
        self.size: List[int] = []
        self.classes: Dict[int, Set[ENode]] = {}
        self.memo: Dict[ENode, int] = {}
        self.merge_log: List[dict] = []
        self._expr_cache: Dict[Expr, int] = {}

    def find(self, eclass: int) -> int:
        root = eclass
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[eclass] != eclass:
            parent = self.parent[eclass]
            self.parent[eclass] = root
            eclass = parent
        return root

    def _new_class(self, enode: ENode) -> int:
        eclass = len(self.parent)
        self.parent.append(eclass)
        self.size.append(1)
        self.classes[eclass] = {enode}
        self.memo[enode] = eclass
        return eclass

    def _canonical(self, enode: ENode) -> ENode:
        return ENode(
            op=enode.op,
            children=tuple(self.find(child) for child in enode.children),
            data=enode.data,
            sort=enode.sort,
        )

    def add_enode(self, enode: ENode) -> int:
        canonical = self._canonical(enode)
        existing = self.memo.get(canonical)
        if existing is not None:
            return self.find(existing)
        return self._new_class(canonical)

    def add_expr(self, expr: Expr) -> int:
        cached = self._expr_cache.get(expr)
        if cached is not None:
            return self.find(cached)
        children = tuple(self.add_expr(arg) for arg in expr.args)
        eclass = self.add_enode(ENode(expr.op, children, expr.data, expr.sort))
        self._expr_cache[expr] = eclass
        return eclass

    def union(self, lhs: int, rhs: int, reason: Mapping[str, Any]) -> Tuple[int, bool]:
        lhs_root = self.find(lhs)
        rhs_root = self.find(rhs)
        if lhs_root == rhs_root:
            return lhs_root, False
        if self.size[lhs_root] < self.size[rhs_root]:
            lhs_root, rhs_root = rhs_root, lhs_root
        self.parent[rhs_root] = lhs_root
        self.size[lhs_root] += self.size[rhs_root]
        self.classes[lhs_root].update(self.classes.pop(rhs_root))
        entry = {
            "merge_id": len(self.merge_log),
            "lhs_eclass": lhs_root,
            "rhs_eclass": rhs_root,
        }
        entry.update(dict(reason))
        self.merge_log.append(entry)
        return lhs_root, True

    def rebuild(self) -> int:
        congruence_merges = 0
        while True:
            owners: Dict[ENode, int] = {}
            canonical_nodes: Dict[int, Set[ENode]] = {}
            duplicate: Optional[Tuple[int, int, ENode]] = None
            for root in sorted(self.classes):
                current = self.find(root)
                if current != root:
                    continue
                nodes = {self._canonical(node) for node in self.classes[root]}
                canonical_nodes[root] = nodes
                for node in nodes:
                    owner = owners.get(node)
                    if owner is not None and self.find(owner) != self.find(root):
                        duplicate = (owner, root, node)
                        break
                    owners[node] = root
                if duplicate is not None:
                    break
            if duplicate is None:
                for root, nodes in canonical_nodes.items():
                    if self.find(root) == root:
                        self.classes[root] = nodes
                self.memo = {node: self.find(root) for node, root in owners.items()}
                return congruence_merges
            lhs, rhs, node = duplicate
            _, changed = self.union(
                lhs,
                rhs,
                {"kind": "congruence", "detail": node.render(), "evidence": "CONGRUENCE"},
            )
            if changed:
                congruence_merges += 1

    def roots(self) -> Tuple[int, ...]:
        return tuple(sorted(root for root in self.classes if self.find(root) == root))

    def nodes(self, eclass: int) -> Tuple[ENode, ...]:
        root = self.find(eclass)
        return tuple(sorted((self._canonical(node) for node in self.classes[root]), key=lambda node: node.render()))

    @property
    def enode_count(self) -> int:
        return sum(len(nodes) for root, nodes in self.classes.items() if self.find(root) == root)

    def equivalent(self, lhs: int, rhs: int) -> bool:
        return self.find(lhs) == self.find(rhs)

    def _match(
        self,
        pattern: Pattern,
        eclass: int,
        substitution: Mapping[str, int],
    ) -> Iterator[Dict[str, int]]:
        root = self.find(eclass)
        if pattern.variable is not None:
            existing = substitution.get(pattern.variable)
            if existing is None:
                result = dict(substitution)
                result[pattern.variable] = root
                yield result
            elif self.find(existing) == root:
                yield dict(substitution)
            return

        for enode in self.nodes(root):
            if enode.op != pattern.op or len(enode.children) != len(pattern.args):
                continue
            if pattern.match_data and enode.data != pattern.data:
                continue
            states = [dict(substitution)]
            for child_pattern, child_class in zip(pattern.args, enode.children):
                next_states: List[Dict[str, int]] = []
                for state in states:
                    next_states.extend(self._match(child_pattern, child_class, state))
                states = next_states
                if not states:
                    break
            yield from states

    def _instantiate(self, pattern: Pattern, substitution: Mapping[str, int]) -> int:
        if pattern.variable is not None:
            return self.find(substitution[pattern.variable])
        children = tuple(self._instantiate(arg, substitution) for arg in pattern.args)
        return self.add_enode(ENode(pattern.op or "", children, pattern.data, pattern.sort))

    def saturate(self, rules: Iterable[Rule], limits: Limits) -> dict:
        started = time.monotonic()
        applications: Set[Tuple[str, int, Tuple[Tuple[str, int], ...]]] = set()
        stop_reason = "SATURATED"
        iterations = 0
        for iteration in range(limits.max_iterations):
            iterations = iteration + 1
            changed = False
            for rule in rules:
                for root in self.roots():
                    matches = list(self._match(rule.lhs, root, {}))
                    for substitution in matches:
                        normalized = tuple(sorted((key, self.find(value)) for key, value in substitution.items()))
                        key = (rule.rule_id, self.find(root), normalized)
                        if key in applications:
                            continue
                        applications.add(key)
                        rhs = self._instantiate(rule.rhs, substitution)
                        _, merged = self.union(
                            root,
                            rhs,
                            {
                                "kind": "rewrite",
                                "rule_id": rule.rule_id,
                                "evidence": rule.evidence.value,
                                "validator": rule.validator,
                            },
                        )
                        changed = changed or merged
                        if self.enode_count > limits.max_enodes:
                            stop_reason = "ENODE_LIMIT"
                            self.rebuild()
                            return self._stats(iterations, started, stop_reason)
                        if (time.monotonic() - started) * 1000 > limits.timeout_ms:
                            stop_reason = "TIME_LIMIT"
                            self.rebuild()
                            return self._stats(iterations, started, stop_reason)
            congruence = self.rebuild()
            changed = changed or congruence > 0
            if not changed:
                break
        else:
            stop_reason = "ITERATION_LIMIT"
        return self._stats(iterations, started, stop_reason)

    def _stats(self, iterations: int, started: float, stop_reason: str) -> dict:
        rule_counts = Counter(
            item.get("rule_id") for item in self.merge_log if item.get("kind") == "rewrite"
        )
        return {
            "iterations": iterations,
            "enodes": self.enode_count,
            "eclasses": len(self.roots()),
            "merges": len(self.merge_log),
            "rule_applications": {key: rule_counts[key] for key in sorted(rule_counts) if key is not None},
            "stop_reason": stop_reason,
        }
