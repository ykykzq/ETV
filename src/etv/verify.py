"""The verification state machine. Heuristics cannot close memory obligations."""

from dataclasses import replace
import importlib.metadata
import logging
from pathlib import Path
import uuid

from .counterexample import find_counterexample
from .eqsat import saturate_roots
from .errors import ETVError, InputError
from .frontend import parse_ttir
from .heuristics import propose
from .ir import ProgramPair, ProgramSequence, walk
from .jsonio import digest
from .lift import lift_kernel, scalar_type
from .logging import event
from .memory import build_obligations
from .pairspec import PairSpec, parse_pair_spec
from .partition import propose_plan
from .partition_proof import check_plan
from .result import (
    EqualityFact,
    InputRecord,
    LaunchFact,
    Obligation,
    PartitionFact,
    VerificationReport,
)
from .rewrites import admit_rules, parse_rules


def _versions() -> tuple[tuple[str, str], ...]:
    versions = []
    for name in ("egglog", "z3-solver", "triton"):
        try:
            versions.append((name, importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            versions.append((name, "not installed"))
    return tuple(versions)


def _manifest(spec: PairSpec, spec_hash: str) -> tuple[InputRecord, ...]:
    root = spec.path.parent

    def name(path: Path) -> str:
        return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

    result = [InputRecord(spec.path.name, spec_hash, "spec")]
    result.extend(
        InputRecord(name(launch.file), digest(launch.file), "ttir", launch.id)
        for launch in spec.lhs + spec.rhs
    )
    result.extend(InputRecord(name(r.file), digest(r.file), "rewrite") for r in spec.rewrites)
    return tuple(result)


def _programs(spec: PairSpec, run_id: str) -> ProgramPair:
    modules = {}
    for launch in spec.lhs + spec.rhs:
        key = (launch.file, launch.function)
        if key not in modules:
            modules[key] = parse_ttir(*key)
            event(
                run_id,
                spec.pair_id,
                "frontend",
                "ttir_parsed",
                file=launch.file.name,
                function=launch.function,
                sha256=modules[key].source_hash,
            )
    sequences = []
    for launches in (spec.lhs, spec.rhs):
        kernels = []
        for launch in launches:
            kernel = lift_kernel(modules[(launch.file, launch.function)], launch)
            arguments = modules[(launch.file, launch.function)].function.regions[0][0].arguments
            role_types = {role.name: role.element for role in spec.roles}
            for endpoint in launch.abi:
                typ = arguments[int(endpoint.name.removeprefix("arg"))].type
                if scalar_type(typ.element or typ) != role_types[endpoint.role]:
                    raise InputError("ABI_ROLE_MISSING", "role type disagrees with TTIR argument")
            kernels.append((launch.step, kernel))
            event(
                run_id,
                spec.pair_id,
                "frontend",
                "kernel_lifted",
                launch=launch.id,
                stores=len(kernel.stores),
            )
        sequences.append(
            ProgramSequence(
                tuple(
                    tuple(k for step, k in kernels if step == index)
                    for index in sorted({step for step, _ in kernels})
                )
            )
        )
    return ProgramPair(sequences[0], sequences[1])


def _verify(spec: PairSpec, report: VerificationReport) -> VerificationReport:
    for source in spec.rewrites:
        parse_rules(source.file)
    pair = _programs(spec, report.run_id)
    frontend_facts = tuple(
        Obligation(
            "frontend." + launch.id,
            "parse_verify_lift",
            "closed",
            ("input." + launch.id,),
            launch.function,
        )
        for launch in spec.lhs + spec.rhs
    )
    memory = build_obligations(pair, spec)
    report = replace(
        report,
        obligations=frontend_facts
        + tuple(
            replace(
                obligation,
                dependencies=obligation.dependencies + tuple(f.id for f in frontend_facts),
            )
            for obligation in memory.obligations
        ),
    )
    for obligation in memory.obligations:
        event(
            report.run_id,
            spec.pair_id,
            "smt",
            "obligation_finished",
            id=obligation.id,
            status=obligation.status,
        )
    if memory.status != "closed":
        return replace(
            report,
            status="DISPROVED" if memory.status == "DISPROVED" else "UNKNOWN",
            reason=memory.reason,
            counterexample=memory.counterexample,
        )
    proposal = propose(spec, memory.roots)
    if proposal.diagnostic:
        event(
            report.run_id,
            spec.pair_id,
            "partition",
            "partition_fallback",
            detail=proposal.diagnostic,
        )
        report = replace(
            report, partitions=(PartitionFact("llm", (), (), (), "fallback", proposal.diagnostic),)
        )
    try:
        registry = admit_rules(spec, memory.roots, proposal.rules)
    except InputError:
        if not proposal.rules:
            raise
        event(
            report.run_id,
            spec.pair_id,
            "rewrite",
            "partition_fallback",
            detail="generated rule admission failed",
        )
        report = replace(
            report,
            partitions=report.partitions
            + (
                PartitionFact(
                    "llm.rules", (), (), (), "fallback", "generated rule admission failed"
                ),
            ),
        )
        registry = admit_rules(spec, memory.roots)
    for fact in registry.facts:
        event(
            report.run_id,
            spec.pair_id,
            "rewrite",
            "rewrite_rejected" if fact.admission == "rejected" else "rewrite_admitted",
            id=fact.id,
            admission=fact.admission,
        )
    report = replace(report, rewrites=registry.facts)
    partition_counts: tuple[tuple[str, int], ...] = ()
    if spec.partition.enabled:
        try:
            plan = proposal.plan or propose_plan(memory.roots, spec)
            checked = check_plan(plan, memory.roots, registry, spec)
            registry = checked.registry
            partition_counts = checked.applications
            report = replace(report, partitions=report.partitions + checked.facts)
            if checked.fallback:
                event(
                    report.run_id,
                    spec.pair_id,
                    "partition",
                    "partition_fallback",
                    detail="boundary equality not proved",
                )
        except Exception as exc:
            detail = "partition fallback: " + type(exc).__name__
            report = replace(
                report,
                partitions=report.partitions
                + (PartitionFact("partition", (), (), (), "fallback", detail),),
            )
            event(report.run_id, spec.pair_id, "partition", "partition_fallback", detail=detail)
    result = saturate_roots(memory.roots, registry, spec.limits)
    event(
        report.run_id,
        spec.pair_id,
        "egglog",
        "saturation_finished",
        merged=all(result.merged),
        iterations=result.iterations,
        enodes=result.enodes,
        reason=result.reason,
    )
    counts = dict(result.applications)
    for name, count in partition_counts:
        counts[name] = counts.get(name, 0) + count
    facts = tuple(
        replace(f, matches=counts.get(f.id, 0), used=bool(counts.get(f.id, 0)))
        for f in registry.facts
    )
    deps = (
        tuple(o.id for o in report.obligations)
        + tuple(f.id for f in facts if f.used)
        + tuple(p.id for p in report.partitions if p.status == "proved")
    )
    equalities = tuple(
        EqualityFact(
            f"equality.{i}",
            root.lhs,
            root.rhs,
            str(root.index),
            merged,
            deps,
            result.iterations if i == 0 else 0,
            result.enodes,
        )
        for i, (root, merged) in enumerate(zip(memory.roots, result.merged, strict=True))
    )
    trusted = tuple(f.id for f in facts if f.used and f.admission == "admitted_unverified")
    report = replace(
        report,
        rewrites=facts,
        equalities=equalities,
        trusted_axioms=report.trusted_axioms + trusted,
    )
    if result.reason:
        return replace(
            report,
            reason="RESOURCE_LIMIT" if result.reason == "RESOURCE_LIMIT" else "INTERNAL_ERROR",
            diagnostic=result.reason,
        )
    if all(result.merged):
        return replace(report, status="PROVED", reason="OBSERVABLE_MEMORY_EQUIVALENT")
    checks = []
    for i, (root, merged) in enumerate(zip(memory.roots, result.merged, strict=True)):
        if merged:
            continue
        check = find_counterexample(root, spec)
        checks.append(
            Obligation(f"value.counterexample.{i}", "smt", check.status, deps, check.detail)
        )
        if check.status == "sat":
            return replace(
                report,
                status="DISPROVED",
                reason="COMPUTE_MISMATCH",
                counterexample=check.counterexample,
                obligations=report.obligations + tuple(checks),
            )
        if check.status == "unknown":
            break
    reason = (
        "CAST_EQUIVALENCE_NOT_REWRITTEN"
        if any(
            e.op in ("trunci", "extsi", "extui", "index_cast", "index_castui", "sitofp", "uitofp")
            for root in memory.roots
            for expr in (root.lhs, root.rhs)
            for e in walk(expr)
        )
        else "NO_EQUIVALENCE_PROOF"
    )
    return replace(report, reason=reason, obligations=report.obligations + tuple(checks))


def verify(spec_path: Path, allow_external: bool = False) -> VerificationReport:
    run_id = uuid.uuid4().hex
    report = VerificationReport(
        run_id, spec_path.stem, "UNKNOWN", "INTERNAL_ERROR", versions=_versions()
    )
    event(run_id, report.pair_id, "schema", "verification_started")
    try:
        spec_hash = digest(spec_path)
        report = replace(report, inputs=(InputRecord(spec_path.name, spec_hash, "spec"),))
        spec = parse_pair_spec(spec_path, allow_external)
        if digest(spec_path) != spec_hash:
            raise InputError("FILE_HASH_MISMATCH", str(spec_path))
        inputs = _manifest(spec, spec_hash)
        launches = tuple(
            LaunchFact(
                launch.id,
                launch.side,
                launch.step,
                len(launch.stores),
                launch.file.name,
                launch.function,
                tuple(
                    sorted(
                        ep.role
                        for ep in launch.abi
                        if ep.role not in {s.role for s in launch.stores}
                    )
                ),
                tuple(sorted({s.role for s in launch.stores})),
            )
            for launch in spec.lhs + spec.rhs
        )
        report = replace(
            report,
            pair_id=spec.pair_id,
            inputs=inputs,
            index_bits=spec.index_bits,
            predicates=spec.predicates,
            launches=launches,
            trusted_axioms=tuple(p.id for p in spec.predicates if p.trusted),
        )
        event(run_id, report.pair_id, "schema", "spec_loaded", inputs=len(inputs))
        report = _verify(spec, report)
        if _manifest(spec, digest(spec_path)) != inputs:
            raise InputError("FILE_HASH_MISMATCH", "input changed during verification")
    except ETVError as exc:
        report = replace(
            report,
            status="UNKNOWN",
            reason=exc.reason,
            diagnostic=exc.detail,
            unsupported=(exc.detail,) if "UNSUPPORTED" in exc.reason else (),
        )
    except Exception as exc:
        logging.getLogger("etv").exception(
            "verification_error",
            extra={"run_id": run_id, "pair_id": report.pair_id, "phase": "verdict"},
        )
        report = replace(
            report,
            status="UNKNOWN",
            reason="INTERNAL_ERROR",
            diagnostic=type(exc).__name__ + ": " + str(exc),
        )
    event(
        run_id,
        report.pair_id,
        "verdict",
        "verification_finished",
        status=report.status,
        reason=report.reason,
    )
    return report
