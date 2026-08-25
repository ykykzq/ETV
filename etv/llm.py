"""Optional DeepSeek assistance for selecting expression pairs and proposing rules."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .ir import Expr
from .model import InputError, PairSpec
from .rules import Rule
from .schema import parse_rewrite_rule


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMAssistance:
    rules: tuple[Rule, ...]
    audit: Mapping[str, Any]


class DeepSeekClient:
    def __init__(self, spec: PairSpec) -> None:
        self.config = spec.llm

    def complete_json(
        self, purpose: str, system: str, payload: Mapping[str, Any]
    ) -> tuple[dict, dict]:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is required when PairSpec llm.enabled is true"
            )
        user = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        prompt_sha256 = hashlib.sha256(
            (system + "\n" + user).encode("utf-8")
        ).hexdigest()
        request_body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 4096,
            "stream": False,
        }
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_ms / 1000
            ) as response:
                raw = response.read()
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise LLMError(
                f"DeepSeek {purpose} request failed: {type(exc).__name__}"
            ) from exc
        try:
            envelope = json.loads(raw.decode("utf-8"))
            content = envelope["choices"][0]["message"]["content"]
            if not content:
                raise ValueError("empty model content")
            value = json.loads(content)
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise LLMError(
                f"DeepSeek {purpose} returned an invalid JSON response"
            ) from exc
        return value, {
            "purpose": purpose,
            "provider": "deepseek",
            "model": envelope.get("model", self.config.model),
            "prompt_sha256": prompt_sha256,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "usage": envelope.get("usage", {}),
        }


def _nodes(expression: Expr, path: str = "root") -> list[dict]:
    result = [
        {
            "path": path,
            "sort": expression.sort.value,
            "rendered": expression.render(),
            "expression": expression.to_json(),
        }
    ]
    for index, argument in enumerate(expression.args):
        result.extend(_nodes(argument, f"{path}.args[{index}]"))
    return result


def _resolve(expression: Expr, path: str) -> Expr:
    if path == "root":
        return expression
    current = expression
    remaining = path
    if not remaining.startswith("root"):
        raise LLMError(f"invalid selected node path {path!r}")
    remaining = remaining[4:]
    while remaining:
        if not remaining.startswith(".args["):
            raise LLMError(f"invalid selected node path {path!r}")
        end = remaining.find("]")
        if end < 6:
            raise LLMError(f"invalid selected node path {path!r}")
        try:
            index = int(remaining[6:end])
            current = current.args[index]
        except (ValueError, IndexError) as exc:
            raise LLMError(f"selected node path does not exist: {path!r}") from exc
        remaining = remaining[end + 1 :]
    return current


def _available_requirements(spec: PairSpec) -> list[dict]:
    values = [
        {"kind": "binding_equals", "name": name, "value": value}
        for name, value in sorted(spec.facts.bindings.items())
        if isinstance(value, int)
    ]
    values.extend(
        {"kind": "assumption", "text": text} for text in spec.facts.assumptions
    )
    values.extend(
        {"kind": "disjoint", "roles": sorted(group)}
        for group in spec.facts.disjoint_groups
    )
    values.extend(
        {"kind": "constraint", "expression": expression.to_json()}
        for expression in spec.facts.constraints
    )
    return values


def propose_rules(
    spec: PairSpec,
    candidates: Sequence[tuple[str, Expr, Expr]],
    client: Optional[DeepSeekClient] = None,
) -> LLMAssistance:
    """Select useful unmatched nodes and generate strictly parsed conditional rules."""

    if not spec.llm.enabled or not spec.llm.generate_rules:
        return LLMAssistance((), {"enabled": False, "calls": []})
    client = client or DeepSeekClient(spec)
    limited = tuple(candidates[: spec.llm.max_candidates])
    calls: list[dict] = []
    selected: list[tuple[str, Expr, Expr, str, str]] = []

    if spec.llm.select_nodes:
        selection_payload = {
            "task": "Select expression-node pairs whose equality would help connect unmatched output roots.",
            "candidates": [
                {
                    "label": label,
                    "lhs_nodes": _nodes(lhs),
                    "rhs_nodes": _nodes(rhs),
                }
                for label, lhs, rhs in limited
            ],
            "output_schema": {
                "pairs": [
                    {"label": "candidate label", "lhs_path": "root", "rhs_path": "root"}
                ]
            },
        }
        selection, audit = client.complete_json(
            "node_selection",
            "Return JSON only. Select existing paths exactly; do not claim that selected nodes are equal.",
            selection_payload,
        )
        calls.append(audit)
        by_label = {label: (lhs, rhs) for label, lhs, rhs in limited}
        raw_pairs = selection.get("pairs", [])
        if not isinstance(raw_pairs, list):
            raise LLMError("DeepSeek node selection must contain a pairs list")
        for item in raw_pairs[: spec.llm.max_candidates]:
            if not isinstance(item, dict):
                raise LLMError("DeepSeek selected node entries must be objects")
            label = item.get("label")
            lhs_path = item.get("lhs_path")
            rhs_path = item.get("rhs_path")
            if (
                label not in by_label
                or not isinstance(lhs_path, str)
                or not isinstance(rhs_path, str)
            ):
                raise LLMError("DeepSeek selected an unknown candidate or invalid path")
            lhs, rhs = by_label[label]
            selected.append(
                (
                    label,
                    _resolve(lhs, lhs_path),
                    _resolve(rhs, rhs_path),
                    lhs_path,
                    rhs_path,
                )
            )
    else:
        selected = [(label, lhs, rhs, "root", "root") for label, lhs, rhs in limited]

    if not selected:
        return LLMAssistance(
            (), {"enabled": True, "calls": calls, "selected_nodes": []}
        )
    requirements = _available_requirements(spec)
    if not requirements:
        raise LLMError(
            "conditional LLM rule generation requires at least one machine-readable fact gate"
        )

    generation_payload = {
        "task": "Propose conditional equality rewrite rules that may connect the selected nodes.",
        "selected_nodes": [
            {
                "label": label,
                "lhs_path": lhs_path,
                "rhs_path": rhs_path,
                "lhs": lhs.to_json(),
                "rhs": rhs.to_json(),
            }
            for label, lhs, rhs, lhs_path, rhs_path in selected
        ],
        "available_fact_requirements": requirements,
        "rule_schema": {
            "rules": [
                {
                    "id": "stable_identifier",
                    "kind": "algebraic or trusted_fact",
                    "statement": "human-readable equality",
                    "lhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
                    "rhs": {"op": "fadd", "args": [{"match": "b"}, {"match": "a"}]},
                    "requires": [requirements[0]],
                }
            ]
        },
        "requirements": [
            "Return JSON only.",
            "Every rule must contain at least one requires entry copied exactly from available_fact_requirements.",
            "Use match metavariables for reusable subexpressions.",
            "Do not invent facts or operations outside the supplied expression language.",
        ],
    }
    generated, audit = client.complete_json(
        "rule_generation",
        "Return JSON only. Proposed rules are untrusted candidates and must be conditional on supplied facts.",
        generation_payload,
    )
    calls.append(audit)
    declarations = generated.get("rules", [])
    if not isinstance(declarations, list):
        raise LLMError("DeepSeek rule generation must contain a rules list")
    rules: list[Rule] = []
    seen: set[str] = set()
    prompt_sha256 = audit["prompt_sha256"]
    for index, declaration in enumerate(declarations[: spec.llm.max_candidates]):
        if not isinstance(declaration, dict):
            raise LLMError("DeepSeek rule entries must be objects")
        item = dict(declaration)
        if not item.get("requires"):
            raise LLMError(
                "DeepSeek generated an unconditional rule; conditional rules are required"
            )
        item["provenance"] = {
            "generated_by": "llm",
            "generator": audit["model"],
            "prompt_sha256": prompt_sha256,
        }
        try:
            rule = parse_rewrite_rule(item, f"DeepSeek.rules[{index}]")
        except InputError as exc:
            raise LLMError(f"DeepSeek generated an invalid rule: {exc}") from exc
        if rule.rule_id in seen:
            raise LLMError(f"DeepSeek generated duplicate rule id {rule.rule_id!r}")
        seen.add(rule.rule_id)
        rules.append(rule)
    return LLMAssistance(
        tuple(rules),
        {
            "enabled": True,
            "provider": "deepseek",
            "configured_model": spec.llm.model,
            "selected_nodes": [
                {
                    "label": label,
                    "lhs_path": lhs_path,
                    "rhs_path": rhs_path,
                    "lhs": lhs.render(),
                    "rhs": rhs.render(),
                }
                for label, lhs, rhs, lhs_path, rhs_path in selected
            ],
            "generated_rule_ids": [rule.rule_id for rule in rules],
            "calls": calls,
        },
    )
