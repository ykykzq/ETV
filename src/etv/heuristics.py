"""Optional bounded LLM proposals; network output is never formal evidence."""

from dataclasses import dataclass
import hashlib
import json
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .errors import InputError
from .ir import RootPair
from .jsonio import JSON, array, obj, string
from .pairspec import PairSpec
from .partition import PDG, PartitionPlan, build_pdg, validate_plan
from .report import serialize
from .rewrites import Rule, parse_rule_data


@dataclass(frozen=True)
class Proposal:
    rules: tuple[Rule, ...] = ()
    plan: PartitionPlan | None = None
    diagnostic: str = ""
    sha256: str = ""


def _request(spec: PairSpec, phase: str, payload: dict[str, JSON]) -> JSON:
    config = spec.llm
    base = os.environ.get(config.base_url_env, "")
    key = os.environ.get(config.api_key_env, "")
    parsed = urlparse(base)
    if (
        not base
        or not key
        or parsed.scheme not in ("https", "http")
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise InputError("LLM_UNAVAILABLE", "missing or invalid configured endpoint/credentials")
    body = {
        "model": config.model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "Return only a JSON object conforming to the requested schema. Node IDs are opaque. Do not return paths, commands, or modify assumptions.",
            },
            {"role": "user", "content": json.dumps({"phase": phase, **payload}, sort_keys=True)},
        ],
    }
    request = Request(
        base.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urlopen(request, timeout=config.timeout_ms / 1000) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise InputError("INVALID_LLM_RESPONSE", "response size exceeded")
    envelope = json.loads(raw)
    content = envelope["choices"][0]["message"]["content"]
    return json.loads(content)


def _nodes(graph: PDG) -> list[JSON]:
    return [{"id": node.id, "expression": serialize(node.expr)} for node in graph.nodes]


def propose(spec: PairSpec, roots: tuple[RootPair, ...]) -> Proposal:
    if not spec.llm.enabled:
        return Proposal()
    left = build_pdg(tuple(r.lhs for r in roots), "lhs")
    right = build_pdg(tuple(r.rhs for r in roots), "rhs")
    try:
        selection = obj(
            _request(
                spec,
                "select_candidates",
                {
                    "schema": {"candidates": [{"lhs": "node id", "rhs": "node id"}]},
                    "lhs": _nodes(left),
                    "rhs": _nodes(right),
                    "assumptions": list(spec.assumptions_for_llm),
                    "max_candidates": spec.partition.max_parts,
                },
            ),
            "candidates",
        )
        candidates = array(selection["candidates"])
        if len(candidates) > spec.partition.max_parts:
            raise InputError("INVALID_LLM_RESPONSE", "too many candidate nodes")
        left_ids, right_ids = {n.id for n in left.nodes}, {n.id for n in right.nodes}
        for candidate in candidates:
            node = obj(candidate, "lhs rhs")
            if string(node["lhs"]) not in left_ids or string(node["rhs"]) not in right_ids:
                raise InputError("INVALID_LLM_RESPONSE", "unknown node id")
        response = obj(
            _request(
                spec,
                "generate_rules_and_partition",
                {
                    "candidates": candidates,
                    "lhs": _nodes(left),
                    "rhs": _nodes(right),
                    "predicate_ids": [p.id for p in spec.predicates],
                    "schema": {
                        "rewrites": {"format": "etv-rewrite-v1", "rules": []},
                        "partition": {
                            "parts": [
                                {
                                    "id": "part.0",
                                    "lhs": "node id",
                                    "rhs": "node id",
                                    "dependencies": [],
                                }
                            ]
                        },
                    },
                    "rewrite_pattern": "pvar/type, const/type, or op/type/args; rules have id, kind, lhs, rhs, requires, validation.method",
                    "semantics": "abstract_float",
                },
            ),
            "rewrites",
            "partition",
        )
        sha = hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
        rules = parse_rule_data(response["rewrites"], "llm:" + spec.llm.model, sha, generated=True)
        gates = {p.id for p in spec.predicates}
        if any(
            not set(rule.requires) <= gates or rule.id.startswith(("builtin.", "partition."))
            for rule in rules
        ):
            raise InputError("INVALID_LLM_RESPONSE", "unknown gate or reserved rule id")
        plan = (
            validate_plan(response["partition"], left, right, spec)
            if "partition" in response and spec.partition.enabled
            else None
        )
        return Proposal(rules, plan, sha256=sha)
    except Exception as exc:
        # Providers can include endpoints in errors. Only the exception class is logged.
        return Proposal(diagnostic="LLM fallback: " + type(exc).__name__)
