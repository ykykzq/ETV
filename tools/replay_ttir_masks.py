"""Replay concrete TTIR writer-mask witnesses without invoking SMT or egglog.

This uses the verifier's checked TTIR frontend/IR interpreter, not an independent
proof implementation and not execution of the full Python operator on a GPU.
"""

import argparse
import hashlib
import json
from math import prod
from pathlib import Path

from etv.pairspec import parse_pair_spec
from etv.semantics import evaluate
from etv.verify import _programs


def physical_offset(spec, side, index):
    if not spec.observation.layouts:
        return index
    view = spec.observation.layouts[0 if side == "lhs" else 1]
    return view.offset + sum(
        ((index // prod(view.shape[axis + 1 :])) % size) * view.stride[axis]
        for axis, size in enumerate(view.shape)
    )


def written_offsets(sequence, spec, side):
    written = set()
    for step in sequence.steps:
        for kernel in step:
            for store in kernel.stores:
                if store.storage.logical_role != spec.observation.role_for_side(side):
                    continue
                programs = int(evaluate(kernel.programs, index_bits=spec.index_bits))
                for pid in range(programs):
                    for lane in range(prod(store.lane_shape)):
                        values = {"__pid": pid, "__lane": lane}
                        if evaluate(store.mask, values, index_bits=spec.index_bits):
                            written.add(int(evaluate(store.offset, values, index_bits=spec.index_bits)))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "analysis/summary.json").read_text())
    assert summary["execution_scope"] == "ttir-only"
    rows = []
    for row in summary["cases"]:
        if row["status"] != "DISPROVED" or row.get("reason") != "MASK_MISMATCH":
            continue
        report_path = args.run / "analysis" / row["case_id"] / "report.json"
        report = json.loads(report_path.read_text())
        pair_path = args.run / "work" / row["case_id"] / "pair.json"
        spec = parse_pair_spec(pair_path)
        assert spec.execution_scope == report["execution_scope"] == "ttir-only"
        assert not any(launch.host_operation for launch in spec.lhs + spec.rhs)
        assert hashlib.sha256(pair_path.read_bytes()).hexdigest() == report["inputs"]["spec"]["sha256"]
        witness = report["counterexample"]
        index = int(witness["index"])
        assert 0 <= index < int(evaluate(spec.observation.numel, index_bits=spec.index_bits))
        program = _programs(spec, "strict-mask-replay")
        observed = {
            side: physical_offset(spec, side, index)
            in written_offsets(getattr(program, side), spec, side)
            for side in ("lhs", "rhs")
        }
        assert observed["lhs"] != observed["rhs"]
        assert observed == {side: witness[side + "_mask"] for side in ("lhs", "rhs")}
        rows.append(
            {
                "case_id": row["case_id"],
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "replayed": True,
                "execution_scope": "ttir-only",
                "method": "concrete TTIR writer enumeration; shared frontend, no SMT/egglog",
                "independent_certificate": False,
                "index": index,
                "observed_masks": observed,
            }
        )
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"mask_witnesses_replayed": len(rows), "independent_certificate": False}))


if __name__ == "__main__":
    main()
